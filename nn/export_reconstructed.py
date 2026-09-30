"""Export the reconstructed G-buffer (Psi + hybrid positions) as vkhr
dump-compatible channel files, ready for the --shade mode.

Per eval frame:
  coverage*  Psi coverage (clamped, support-mask suppressed)
  tangent*   Psi tangent (renormalized, suppressed)
  position*  hybrid: real 1spp samples where valid; analytic
             reconstruction where it filled; naive nearest-valid copy
             elsewhere (completeness fallback)
  -> depth*  project(position*) through the frame VP (z/w, [0,1])

Files land in <dump_dir>/recon/: frame_XXXXX_coverage.f16 (R16F),
frame_XXXXX_tangent.f16 (RGBA16F, alpha=1), frame_XXXXX_depth.f32 (R32F)
at the INPUT resolution - exactly the layouts the input/ folder uses.

Usage:
    python nn/export_reconstructed.py --data dumps/prepared_full \
        --dump dumps/dataset_full/ponytail_eval
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from dataset import ValDataset
from models import SpatialNet
from position_reconstruction import reconstruct_positions


def naive_fill(coverage_input: np.ndarray, position_input: np.ndarray,
               target: np.ndarray) -> np.ndarray:
    valid = coverage_input > 0.0
    _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
    out = position_input[tuple(indices)]
    return np.where(target[..., None], out, position_input)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/psi_v1/best.pt")
    parser.add_argument("--data", default="dumps/prepared_full")
    parser.add_argument("--dump", required=True,
                        help="raw dump dir of the eval split (for input channels)")
    parser.add_argument("--out-name", default="recon")
    options = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialNet(base_channels=32).to(device)
    state = torch.load(options.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    dump_dir = Path(options.dump)
    out_dir = dump_dir / options.out_name
    out_dir.mkdir(exist_ok=True)

    # The eval npz runs mirror the raw dump runs one-to-one.
    run_name = dump_dir.name  # e.g. ponytail_eval
    npz_dir = Path(options.data) / run_name

    dataset = ValDataset(options.data, split="eval")  # all frames, no holdout
    frames = sorted(npz_dir.glob("frame_*.npz"))
    print(f"exporting {len(frames)} frames -> {out_dir}")

    for frame_path in frames:
        tag = frame_path.stem
        data = np.load(frame_path)

        view = data["matrices"][0].astype(np.float64)
        projection = data["matrices"][1].astype(np.float64)
        view_projection = projection @ view

        x = np.concatenate([data["input/coverage"], data["input/tangent"]], axis=-1)
        batch = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=device.type == "cuda"):
            prediction = model(batch)
        coverage = prediction["reconstruction"][:, 0:1].clamp(0, 1)
        tangent = F.normalize(prediction["reconstruction"][:, 1:4], dim=1, eps=1e-6)
        support = (torch.sigmoid(prediction["mask_logit"]) > 0.5).float()
        coverage = (coverage * support)[0, 0].cpu().numpy().astype(np.float16)
        tangent = (tangent * support)[0].cpu().numpy().transpose(1, 2, 0)

        in_coverage = data["input/coverage"][..., 0]
        in_position = data["input/position"]
        in_depth = data["input/depth"][..., 0]

        result = reconstruct_positions(coverage.astype(np.float32), tangent,
                                       in_coverage, in_position, in_depth,
                                       view, projection)

        # Hybrid position: real sample / analytic / naive, in that order.
        position = result["position"]
        naive_target = np.ones_like(in_coverage, dtype=bool)
        position = naive_fill(in_coverage, position, naive_target)

        # Project to hardware depth for the shading pass.
        hom = np.concatenate([position, np.ones(position.shape[:-1] + (1,))], axis=-1)
        clip = hom @ view_projection.T
        depth = np.clip(clip[..., 2] / clip[..., 3], 0.0, 1.0).astype(np.float32)

        tangent_rgba = np.concatenate(
            [tangent, np.ones(tangent.shape[:2] + (1,), dtype=np.float32)],
            axis=-1).astype(np.float16)

        coverage.astype("<f2").tofile(out_dir / f"{tag}_coverage.f16")
        tangent_rgba.astype("<f2").tofile(out_dir / f"{tag}_tangent.f16")
        depth.astype("<f4").tofile(out_dir / f"{tag}_depth.f32")
        print(f"  {tag} done", flush=True)

    print(f"reconstructed G-buffer written to {out_dir}")


if __name__ == "__main__":
    main()
