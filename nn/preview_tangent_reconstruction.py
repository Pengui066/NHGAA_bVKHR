"""Tangent comparison strips for the Psi reconstruction (input / Psi / GT).

Companion to the coverage comparison strips written by nn/evaluate.py:
same validation frames (first frame per style by default), same tangent
display convention as utils/preview_dump.py (R/X, G/Y, B/Z; t -> 0.5*t+0.5;
hair pixels only, background black). Each band caption carries the mean
angular error against the GT on hair pixels, so the improvement is
readable straight off the image - same numbers, same protocol
(|dot|, hair-pixel mean) as the tangent_angular_deg entry in metrics.json.

Bands (top to bottom):
  1. input tangent (1 spp)    raw undersampled G-buffer
  2. Psi reconstruction       renormalized + support-suppressed
  3. GT tangent (16 spp)      oriented average of GT sub-samples
  4. angular error vs GT      |Psi - GT| angle, [0, 90] deg -> [0, 1]

Usage:
    python nn/preview_tangent_reconstruction.py \
        --checkpoint runs/psi_v1/best.pt --data dumps/prepared_full

Outputs <out>/{style}_{frame}_tangent_comparison.png. Defaults write
next to the coverage strips in runs/psi_v1/evaluation/, so both
comparison images sit side by side in one folder.

Split note: --split defaults to "train", matching nn/evaluate.py, so
the frames shown here are the SAME frames as the existing coverage
strips. For held-out evaluation frames pass --split eval --val-stride 1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import ValDataset
from models import SpatialNet


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def postprocess(reconstruction: torch.Tensor, mask_logit: torch.Tensor) -> tuple:
    """Inference-time behavior, identical to nn/evaluate.py: clamp coverage,
    renormalize tangents, threshold the support mask and suppress
    predictions outside the reconstructed hair support."""
    coverage = reconstruction[:, 0:1].clamp(0.0, 1.0)
    tangent = F.normalize(reconstruction[:, 1:4], dim=1, eps=1e-6)
    support = (torch.sigmoid(mask_logit) > 0.5).float()
    return coverage * support, tangent * support


def tangent_rgb(tangent: np.ndarray, coverage: np.ndarray,
                seam_free: bool = False) -> np.ndarray:
    """Paper-style tangent display: R/X, G/Y, B/Z, hair pixels only.
    Copied from utils/preview_dump.py (keep in sync). Returns float
    RGB in [0, 1] instead of a PIL image so bands compose as arrays."""
    t = np.nan_to_num(tangent[:, :, :3], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if seam_free:
        t = np.abs(t)
    rgb = t if seam_free else t * 0.5 + 0.5
    rgb *= (coverage[:, :, 0] > 0)[:, :, None]
    return np.clip(rgb, 0.0, 1.0)


def angular_error_map(pred: np.ndarray, gt: np.ndarray,
                      hair: np.ndarray) -> np.ndarray:
    """Per-pixel angle (degrees) between pred and GT tangents on hair
    pixels, 0 elsewhere. Sign-invariant via |dot|, matching both the
    canonical orientation T == -T and the metrics protocol."""
    dot = np.clip((pred * gt).sum(axis=-1), -1.0, 1.0)
    angle = np.degrees(np.arccos(np.abs(dot)))
    angle[~hair] = 0.0
    return angle


def build_strip(bands, labels):
    """Stacks (H, W, 3) float [0,1] bands into one labeled column image.
    Same layout as the coverage strips in nn/evaluate.py."""
    import PIL.Image
    import PIL.ImageDraw

    height, width = bands[0].shape[:2]
    canvas = np.ones((len(bands) * (height + 26) + 4, width, 3), dtype=np.float32)
    image = PIL.Image.fromarray((canvas * 255).astype(np.uint8))
    draw = PIL.ImageDraw.Draw(image)
    for i, (band, label) in enumerate(zip(bands, labels)):
        y = i * (height + 26)
        image.paste(PIL.Image.fromarray((band * 255).astype(np.uint8)), (0, y + 26))
        draw.text((8, y + 6), label, fill="black")
    return image


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="runs/psi_v1/best.pt")
    parser.add_argument("--data", default="dumps/prepared_full")
    parser.add_argument("--out", default="runs/psi_v1/evaluation")
    parser.add_argument("--split", default="train",
                        help="dataset split; default matches nn/evaluate.py "
                             "so frames align with the coverage strips")
    parser.add_argument("--val-stride", type=int, default=16,
                        help="keep every Nth frame (ValDataset default: 16)")
    parser.add_argument("--frames-per-style", type=int, default=1,
                        help="how many comparison strips to write per style")
    parser.add_argument("--seam-free", action="store_true",
                        help="sign-invariant |t| display, see utils/preview_dump.py")
    options = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(options.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SpatialNet(base_channels=32).to(device)
    state = torch.load(options.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {options.checkpoint} (step {state.get('step', '?')})")

    dataset = ValDataset(options.data, split=options.split,
                         val_stride=options.val_stride)
    print(f"{len(dataset)} frames ({options.split} split), "
          f"{options.frames_per_style} strip(s) per style")

    strips_per_style: dict[str, int] = {}
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            style = sample["style"]
            if strips_per_style.get(style, 0) >= options.frames_per_style:
                continue

            batch_input = sample["input"].unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                prediction = model(batch_input)
            pred_coverage, pred_tangent = postprocess(prediction["reconstruction"],
                                                      prediction["mask_logit"])

            gt_tangent = sample["gt_tangent"].unsqueeze(0).to(device)
            gt_mask = to_numpy((sample["gt_mask"] > 0.5).unsqueeze(0))[0, 0]

            in_tangent = F.normalize(batch_input[:, 1:4], dim=1, eps=1e-6)

            # (H, W, 3) display views; prepared npz stores GT already
            # box-downsampled to the input resolution, so all bands share H x W.
            in_tan_np = to_numpy(in_tangent)[0].transpose(1, 2, 0)
            pred_tan_np = to_numpy(pred_tangent)[0].transpose(1, 2, 0)
            gt_tan_np = to_numpy(gt_tangent)[0].transpose(1, 2, 0)
            in_cov_np = to_numpy(batch_input)[0, 0:1].transpose(1, 2, 0)
            pred_cov_np = to_numpy(pred_coverage)[0].transpose(1, 2, 0)
            gt_cov_np = sample["gt_coverage"].permute(1, 2, 0).numpy()

            hair = gt_mask > 0.5
            in_err = float(np.mean(angular_error_map(in_tan_np, gt_tan_np, hair)[hair]))
            pred_err = float(np.mean(angular_error_map(pred_tan_np, gt_tan_np, hair)[hair]))

            error_band = np.repeat(
                np.clip(angular_error_map(pred_tan_np, gt_tan_np, hair) / 90.0,
                        0.0, 1.0)[:, :, None], 3, axis=2)

            frame = Path(dataset.samples[index][1]).stem
            orientation = "sign-invariant |t|" if options.seam_free else "camera-aligned sign"
            bands = [
                tangent_rgb(in_tan_np, in_cov_np, options.seam_free),
                tangent_rgb(pred_tan_np, pred_cov_np, options.seam_free),
                tangent_rgb(gt_tan_np, gt_cov_np, options.seam_free),
                error_band,
            ]
            labels = [
                f"{style} {frame}: input tangent (1 spp) - mean err {in_err:.2f} deg",
                f"Psi reconstruction - mean err {pred_err:.2f} deg",
                f"GT tangent (16 spp), {orientation}",
                "angular error |Psi - GT|, white = 90+ deg",
            ]
            image = build_strip(bands, labels)
            image.save(out_dir / f"{style}_{frame}_tangent_comparison.png")
            strips_per_style[style] = strips_per_style.get(style, 0) + 1
            print(f"  {style} {frame}: input {in_err:.2f} deg -> psi {pred_err:.2f} deg")

    print(f"\n{sum(strips_per_style.values())} tangent strips written to {out_dir}")


if __name__ == "__main__":
    main()
