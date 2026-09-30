"""Image-domain metrics over the shaded results (paper Table 1 style).

For every eval frame of a dump:
  Reference = deferred shading of the 16 spp GT G-buffer, downsampled
  Input     = deferred shading of the 1 spp input G-buffer (baseline)
  Ours      = deferred shading of the reconstructed G-buffer

All images are linear-light; they are converted to sRGB before the
metrics (standard practice for PSNR/SSIM/LPIPS in rendering papers).
PSNR is computed only on hair pixels of the reference (the paper's
convention); SSIM on the hair bounding box; LPIPS on the full frame.

Usage:
    python nn/metrics.py --dump dumps/dataset_full/ponytail_eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred, gt = pred[mask], gt[mask]
    mse = float(((pred - gt) ** 2).mean())
    return float("inf") if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def ssim_gray(pred: np.ndarray, gt: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    a = np.dot(pred[..., :3], [0.299, 0.587, 0.114])
    b = np.dot(gt[..., :3], [0.299, 0.587, 0.114])
    return float(structural_similarity(a, b, data_range=1.0))


def load_shaded(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f4")
    size = int(np.sqrt(raw.size // 4))
    return raw.reshape(size, size, 4) if size * size * 4 == raw.size else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--recon-source", default="recon",
                        help="subdir holding the shaded recon images")
    parser.add_argument("--lpips", action="store_true")
    options = parser.parse_args()

    dump = Path(options.dump)
    meta = json.loads((dump / "meta.json").read_text())
    width, height = meta["input_resolution"]
    gt_width, gt_height = meta["gt_resolution"]
    ssaa = meta["ssaa_factor"]

    import PIL.Image

    lpips_model = None
    if options.lpips:
        import lpips as lpips_pkg
        lpips_model = lpips_pkg.LPIPS(net="alex").to(
            "cuda" if torch.cuda.is_available() else "cpu")

    frames = sorted((dump / "input").glob("*_shaded.f32"))
    rows = []

    for input_shaded_path in frames:
        tag = input_shaded_path.stem.replace("_shaded", "")
        ours_path = dump / options.recon_source / f"{tag}_shaded_offline.f32"
        gt_path = dump / "gt" / f"{tag}_shaded.f32"
        if not ours_path.exists() or not gt_path.exists():
            print(f"skipping {tag}: missing shaded images")
            continue

        input_img = np.fromfile(input_shaded_path, dtype="<f4").reshape(height, width, 4)[..., :3]
        ours_img = np.fromfile(ours_path, dtype="<f4").reshape(height, width, 4)[..., :3]
        gt_img = np.fromfile(gt_path, dtype="<f4").reshape(gt_height, gt_width, 4)[..., :3]
        gt_cov = np.fromfile(dump / "gt" / f"{tag}_coverage.f16", dtype="<f2").astype(np.float32)
        gt_cov = gt_cov.reshape(gt_height, gt_width, 1)

        # Box downsample the GT reference and coverage to the input size.
        def box_downsample(img: np.ndarray) -> np.ndarray:
            channels = [np.array(PIL.Image.fromarray(
                (np.clip(img[..., c], 0, 1) * 255).astype(np.uint8)).resize(
                (width, height), PIL.Image.BOX)).astype(np.float32)[..., None] / 255.0
                for c in range(img.shape[2])]
            return np.concatenate(channels, axis=-1)

        gt_img_small = box_downsample(gt_img)
        gt_cov_small = box_downsample(gt_cov)

        hair = gt_cov_small[..., 0] > 0.02
        ys, xs = np.where(hair)
        box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)) if len(ys) else None

        srgb_input = linear_to_srgb(input_img)
        srgb_ours = linear_to_srgb(ours_img)
        srgb_gt = linear_to_srgb(gt_img_small)

        row = {
            "frame": tag,
            "hair_px": int(hair.sum()),
            "input_psnr": psnr(srgb_input, srgb_gt, hair),
            "ours_psnr": psnr(srgb_ours, srgb_gt, hair),
            "input_ssim": ssim_gray(srgb_input[box], srgb_gt[box]) if box else np.nan,
            "ours_ssim": ssim_gray(srgb_ours[box], srgb_gt[box]) if box else np.nan,
        }

        if lpips_model is not None:
            def to_lpips(img):
                t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None] * 2 - 1
                return t.to(next(lpips_model.parameters()).device)
            with torch.no_grad():
                row["input_lpips"] = float(lpips_model(to_lpips(srgb_input), to_lpips(srgb_gt)))
                row["ours_lpips"] = float(lpips_model(to_lpips(srgb_ours), to_lpips(srgb_gt)))

        rows.append(row)
        print(f"{tag}: PSNR input {row['input_psnr']:.2f} -> ours {row['ours_psnr']:.2f} dB | "
              f"SSIM {row['input_ssim']:.4f} -> {row['ours_ssim']:.4f}"
              + (f" | LPIPS {row['input_lpips']:.4f} -> {row['ours_lpips']:.4f}"
                 if lpips_model is not None else ""), flush=True)

    if not rows:
        print("no frames evaluated")
        return

    summary = {}
    for key in ["input_psnr", "ours_psnr", "input_ssim", "ours_ssim",
                "input_lpips", "ours_lpips"]:
        values = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        if values:
            summary[key] = float(np.mean(values))

    print("\n================ image-domain report card ================")
    print(f"frames: {len(rows)} | hair-pixel PSNR, hair-box SSIM, full-frame LPIPS (sRGB)")
    for key, value in summary.items():
        print(f"  {key:12s} {value:8.4f}")

    (dump / options.recon_source / "image_metrics.json").write_text(
        json.dumps({"per_frame": rows, "mean": summary}, indent=2))
    print(f"saved {dump / options.recon_source / 'image_metrics.json'}")


if __name__ == "__main__":
    main()
