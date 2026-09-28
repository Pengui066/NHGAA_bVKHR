"""Evaluate a trained Psi network on the validation frames.

Produces the first quantitative report card of the reproduction:
coverage PSNR (hair pixels + whole image), coverage SSIM, tangent
angular error, for the raw 1 spp input (baseline) and the network
reconstruction, plus labeled comparison images per style.

Metrics follow the paper's convention: computed only on pixels that
correspond to hair in the (high-sample) reference. LPIPS on shaded
images comes later, once the analytic position reconstruction and the
deferred shading are in the loop.

Usage:
    python nn/evaluate.py --checkpoint runs/psi_v1/best.pt --data dumps/prepared_full
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

from dataset import ValDataset
from models import SpatialNet


def psnr(mse: float) -> float:
    return float("inf") if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def postprocess(reconstruction: torch.Tensor, mask_logit: torch.Tensor) -> tuple:
    """Inference-time behavior from the paper: clamp coverage, renormalize
    tangents, threshold the support mask and suppress predictions outside
    the reconstructed hair support."""
    coverage = reconstruction[:, 0:1].clamp(0.0, 1.0)
    tangent = F.normalize(reconstruction[:, 1:4], dim=1, eps=1e-6)
    support = (torch.sigmoid(mask_logit) > 0.5).float()
    return coverage * support, tangent * support


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/psi_v1/best.pt")
    parser.add_argument("--data", default="dumps/prepared_full")
    parser.add_argument("--out", default="runs/psi_v1/evaluation")
    options = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(options.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SpatialNet(base_channels=32).to(device)
    state = torch.load(options.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {options.checkpoint} (step {state.get('step', '?')})")

    dataset = ValDataset(options.data, split="train")
    print(f"evaluating {len(dataset)} validation frames...")

    rows = []
    seen_styles = set()
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            batch_input = sample["input"].unsqueeze(0).to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model(batch_input)
            pred_coverage, pred_tangent = postprocess(prediction["reconstruction"],
                                                      prediction["mask_logit"])

            gt_coverage = sample["gt_coverage"].unsqueeze(0).to(device)
            gt_tangent = sample["gt_tangent"].unsqueeze(0).to(device)
            gt_mask = (sample["gt_mask"].unsqueeze(0).to(device) > 0.5)

            in_coverage = batch_input[:, 0:1]
            in_tangent = F.normalize(batch_input[:, 1:4], dim=1, eps=1e-6)

            # ---- metrics on GT hair pixels ----
            hair = gt_mask[0, 0].cpu().numpy()
            hair_count = hair.sum()
            if hair_count < 1:
                continue

            # (1, 1, H, W) -> (H, W): drop batch AND channel axes.
            pred_cov_np = to_numpy(pred_coverage)[0, 0]
            in_cov_np = to_numpy(in_coverage)[0, 0]
            gt_cov_np = to_numpy(gt_coverage)[0, 0]

            pred_err = (pred_cov_np - gt_cov_np)[hair > 0]
            in_err = (in_cov_np - gt_cov_np)[hair > 0]
            row = {
                "style": sample["style"],
                "frame": Path(dataset.samples[index][1]).stem,
                "hair_pixels": int(hair_count),
                "pred_coverage_psnr": psnr(float((pred_err ** 2).mean())),
                "input_coverage_psnr": psnr(float((in_err ** 2).mean())),
                "pred_coverage_ssim": float(ssim(gt_cov_np, pred_cov_np, data_range=1.0)),
                "input_coverage_ssim": float(ssim(gt_cov_np, in_cov_np, data_range=1.0)),
            }

            # Tangent angular error in degrees (hair pixels).
            pred_tan_np = to_numpy(pred_tangent)[0].transpose(1, 2, 0)[hair > 0]
            gt_tan_np = to_numpy(gt_tangent)[0].transpose(1, 2, 0)[hair > 0]
            in_tan_np = to_numpy(in_tangent)[0].transpose(1, 2, 0)[hair > 0]

            def angular(a, b):
                dot = np.clip((a * b).sum(axis=1), -1.0, 1.0)
                return np.degrees(np.arccos(np.abs(dot)))

            row["pred_tangent_deg"] = float(np.mean(angular(pred_tan_np, gt_tan_np)))
            row["input_tangent_deg"] = float(np.mean(angular(in_tan_np, gt_tan_np)))

            rows.append(row)

            # Comparison image for the first validation frame of each style.
            if sample["style"] not in seen_styles:
                seen_styles.add(sample["style"])
                def gray(a):
                    return np.clip(a, 0, 1)

                input_band = gray(in_cov_np)
                pred_band = gray(pred_cov_np)
                gt_band = gray(gt_cov_np)
                error_band = gray(np.abs(pred_cov_np - gt_cov_np) * 4.0)

                strips = [input_band, pred_band, gt_band, error_band]
                labels = ["input coverage (1 spp)", "Psi reconstruction",
                          "GT coverage (16 spp)", "|Psi - GT| x4 error"]
                height, width = input_band.shape
                canvas = np.ones((len(strips) * (height + 26) + 4, width), dtype=np.float32)
                import PIL.Image
                import PIL.ImageDraw
                image = PIL.Image.fromarray((canvas * 255).astype(np.uint8))
                draw = PIL.ImageDraw.Draw(image)
                for i, (strip, label) in enumerate(zip(strips, labels)):
                    y = i * (height + 26)
                    image.paste(PIL.Image.fromarray((strip * 255).astype(np.uint8)), (0, y + 26))
                    draw.text((8, y + 6), label, fill="black")
                image.save(out_dir / f"{row['style']}_{row['frame']}_comparison.png")

    # ---- aggregate ----
    def agg(key: str, styles: bool = True) -> dict:
        values = [r[key] for r in rows]
        result = {"mean": float(np.mean(values)), "frames": len(values)}
        if styles:
            by_style = {}
            for r in rows:
                by_style.setdefault(r["style"], []).append(r[key])
            result["by_style"] = {s: float(np.mean(v)) for s, v in by_style.items()}
        return result

    summary = {
        "checkpoint": options.checkpoint,
        "frames": len(rows),
        "coverage_psnr": {"input": agg("input_coverage_psnr"),
                          "psi": agg("pred_coverage_psnr")},
        "coverage_ssim": {"input": agg("input_coverage_ssim"),
                          "psi": agg("pred_coverage_ssim")},
        "tangent_angular_deg": {"input": agg("input_tangent_deg"),
                                "psi": agg("pred_tangent_deg")},
        "per_frame": rows,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print("\n================ first report card (hair pixels only) ================")
    for key in ["coverage_psnr", "coverage_ssim", "tangent_angular_deg"]:
        for variant in ["input", "psi"]:
            entry = summary[key][variant]
            by_style = " | ".join(f"{s}: {v:.2f}" for s, v in entry["by_style"].items())
            print(f"{key:20s} {variant:5s} mean {entry['mean']:8.3f}   {by_style}")
    print(f"saved metrics to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
