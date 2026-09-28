"""Validate the analytic position reconstruction on prepared frames.

Runs the five-step pipeline in two configurations per frame:
  oracle - GT coverage and GT tangent as the "neural" outputs
           (validates the analytic method itself)
  psi    - the trained Psi network's predicted coverage/tangent
           (validates the integrated pipeline)

Errors are Euclidean world-space distances against the GT position on
GT hair pixels, split into pixels where the 1 spp input HAD a sample
(valid) and pixels where it had NONE (missing - the reconstruction
targets). Unfilled pixels (no geometric evidence within the propagation
range) carry no position and are reported separately. Baseline: naive
nearest-valid-position copy without tangent guidance.

Usage:
    python nn/test_position_reconstruction.py --frames 4 --stride 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from dataset import ValDataset
from models import SpatialNet
from position_reconstruction import reconstruct_positions


def predict_psi(model, sample, device):
    batch_input = sample["input"].unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
        prediction = model(batch_input)
    coverage = prediction["reconstruction"][:, 0:1].clamp(0, 1)
    tangent = torch.nn.functional.normalize(prediction["reconstruction"][:, 1:4],
                                            dim=1, eps=1e-6)
    support = (torch.sigmoid(prediction["mask_logit"]) > 0.5).float()
    coverage = (coverage * support)[0, 0].cpu().numpy()
    tangent = (tangent * support)[0].cpu().numpy().transpose(1, 2, 0)
    return coverage, tangent


def errors(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
           naive: np.ndarray | None = None) -> dict:
    if mask.sum() == 0:
        return {"mean": np.nan, "median": np.nan, "p90": np.nan,
                "wild_fraction": np.nan, "beats_naive": np.nan}
    err = np.linalg.norm(pred - gt, axis=-1)[mask]
    result = {"mean": float(err.mean()), "median": float(np.median(err)),
              "p90": float(np.percentile(err, 90)),
              "wild_fraction": float((err > 10.0).mean())}
    if naive is not None:
        naive_err = np.linalg.norm(naive - gt, axis=-1)[mask]
        result["beats_naive"] = float((err < naive_err).mean())
    return result


def naive_nearest_copy(coverage_input: np.ndarray, position_input: np.ndarray,
                       target_mask: np.ndarray) -> np.ndarray:
    valid = coverage_input > 0.0
    _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
    out = position_input[tuple(indices)]
    return np.where(target_mask[..., None], out, position_input)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/psi_v1/best.pt")
    parser.add_argument("--data", default="dumps/prepared_full")
    parser.add_argument("--frames", type=int, default=4, help="val frames per style")
    parser.add_argument("--stride", type=int, default=32, help="frame stride")
    options = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialNet(base_channels=32).to(device)
    state = torch.load(options.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    dataset = ValDataset(options.data, split="train")
    picks = list(range(0, len(dataset), options.stride))[: options.frames * 5]

    results = {"oracle": [], "psi": []}
    processed = 0
    for index in picks:
        sample = dataset[index]
        data = np.load(dataset.samples[index][1])

        # matrices were saved post-transpose in prepare_dataset.py - they
        # are already the true view/projection matrices.
        view = data["matrices"][0].astype(np.float64)
        projection = data["matrices"][1].astype(np.float64)

        gt_hair = data["gt/hair_mask"][..., 0] > 0.5
        gt_position = data["gt/position"]
        in_coverage = data["input/coverage"][..., 0]
        in_position = data["input/position"]
        in_depth = data["input/depth"][..., 0]

        missing = gt_hair & (in_coverage <= 0.0)
        valid_px = gt_hair & (in_coverage > 0.0)
        if missing.sum() == 0:
            continue

        psi_coverage, psi_tangent = predict_psi(model, sample, device)
        configs = {
            "oracle": (sample["gt_coverage"][0].numpy(),
                       sample["gt_tangent"].numpy().transpose(1, 2, 0)),
            "psi": (psi_coverage, psi_tangent),
        }

        for name, (cov, tan) in configs.items():
            result = reconstruct_positions(cov, tan, in_coverage, in_position,
                                           in_depth, view, projection)
            # Unfilled pixels carry no position (the origin) - exclude them
            # from the errors and report their count separately.
            filled = gt_hair & (valid_px | result["has_repair"] | result["has_vote"])
            unfilled = gt_hair & ~valid_px & ~result["has_repair"] & ~result["has_vote"]
            results[name].append({
                "frame": f"{sample['style']}_{Path(dataset.samples[index][1]).stem}",
                "missing_px": int(missing.sum()),
                "valid_px": int(valid_px.sum()),
                "unfilled_px": int(unfilled.sum()),
                "all": errors(result["position"], gt_position, filled),
                "missing": errors(result["position"], gt_position, filled & missing,
                                  naive=naive_nearest_copy(in_coverage, in_position,
                                                           filled & missing)),
                "valid": errors(result["position"], gt_position, valid_px),
                "naive_missing": errors(naive_nearest_copy(in_coverage, in_position, missing),
                                        gt_position, missing),
            })
        processed += 1
        print(f"[{processed}] {sample['style']} done (missing px: {int(missing.sum())})",
              flush=True)

    print("\n=============== position reconstruction report ===============")
    for name, entries in results.items():
        if not entries:
            continue

        def agg(key, field):
            values = [e[key][field] for e in entries if not np.isnan(e[key][field])]
            return float(np.mean(values)) if values else float("nan")

        print(f"\n[{name}] over {len(entries)} frames "
              f"(world units; strand radius ~0.3; wild = error > 10)")
        print(f"  valid px   : mean {agg('valid', 'mean'):8.4f}  median {agg('valid', 'median'):8.4f}  "
              f"p90 {agg('valid', 'p90'):8.4f}  wild {agg('valid', 'wild_fraction') * 100:5.2f}%")
        print(f"  missing px : mean {agg('missing', 'mean'):8.4f}  median {agg('missing', 'median'):8.4f}  "
              f"p90 {agg('missing', 'p90'):8.4f}  wild {agg('missing', 'wild_fraction') * 100:5.2f}%")
        print(f"  beats naive on {agg('missing', 'beats_naive') * 100:.1f}% of missing pixels")
        print(f"  naive copy : mean {agg('naive_missing', 'mean'):8.4f}  median {agg('naive_missing', 'median'):8.4f}  "
              f"p90 {agg('naive_missing', 'p90'):8.4f}  wild {agg('naive_missing', 'wild_fraction') * 100:5.2f}%")
        print(f"  all hair px: mean {agg('all', 'mean'):8.4f}  median {agg('all', 'median'):8.4f}  "
              f"unfilled {sum(e['unfilled_px'] for e in entries)} px")

    out = Path("runs/position_reconstruction_report.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
