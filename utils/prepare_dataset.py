#!/usr/bin/env python3
"""Prepare a vkhr G-buffer dump dataset for neural network training.

Reads the raw channel dumps produced by `vkhr --dump` and writes one
compressed .npz per frame containing:

    input/coverage   (H,W,1) f32  undersampled GPAA coverage
    input/tangent    (H,W,3) f32  world-space, canonical orientation
    input/position   (H,W,3) f32  reconstructed from input depth + camera
    input/depth      (H,W,1) f32  hardware depth
    input/motion     (H,W,2) f32  backward motion in NDC
    gt/coverage      (H,W,1) f32  high-sample GT, box-downsampled
    gt/tangent       (H,W,3) f32  oriented average of GT sub-samples
    gt/position      (H,W,3) f32  frontmost GT position (reconstructed)
    gt/depth         (H,W,1) f32  frontmost GT depth
    gt/hair_mask     (H,W,1) f32  1 where GT coverage > 0 (support mask)
    matrices         view/projection/camera arrays for this frame

GT downsampling details: coverage is averaged over the hair sub-samples
of each ssaa x ssaa block; tangents are already sign-canonicalized at
render time, so they are averaged directly and renormalized; depth and
position take the frontmost (minimum depth) hair sub-sample.

Usage:
    python utils/prepare_dataset.py dumps/dataset_full --out dumps/prepared
    python utils/prepare_dataset.py dumps/pilot_ssaa2 --workers 8
"""
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def reconstruct_position(depth: np.ndarray, view: np.ndarray,
                         projection: np.ndarray) -> np.ndarray:
    """World position from a hardware depth buffer (H,W,1) and matrices."""
    height, width = depth.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width]
    ndc_x = 2.0 * (xs + 0.5) / width - 1.0
    ndc_y = 2.0 * (ys + 0.5) / height - 1.0

    inv_view_projection = np.linalg.inv(projection @ view)

    clip = np.stack([ndc_x, ndc_y, depth[..., 0], np.ones_like(ndc_x)], axis=-1)
    world = clip @ inv_view_projection.T
    return (world[..., :3] / world[..., 3:4]).astype(np.float32)


def downsample_gt(coverage: np.ndarray, tangent: np.ndarray, depth: np.ndarray,
                  ssaa: int) -> dict:
    """Box-downsample the GT channels from (H*ssaa, W*ssaa) to (H, W).

    All arrays are float32 with channel-last layout; tangent is already
    canonically oriented, so a straight average cannot cancel.
    """
    height, width = coverage.shape[0] // ssaa, coverage.shape[1] // ssaa

    cov = coverage.reshape(height, ssaa, width, ssaa)
    tan = tangent.reshape(height, ssaa, width, ssaa, 3)
    dep = depth.reshape(height, ssaa, width, ssaa)

    hair = cov > 0.0  # sub-sample hit hair?

    # Coverage: mean over the hair sub-samples of each block.
    cov_sum = np.where(hair, cov, 0.0).sum(axis=(1, 3))
    cov_count = np.maximum(hair.sum(axis=(1, 3)), 1)
    gt_coverage = (cov_sum / cov_count)[..., None]

    # Tangent: mean over hair sub-samples (pre-oriented), renormalized.
    tan_masked = tan * hair[..., None]
    tan_sum = tan_masked.sum(axis=(1, 3))
    tan_norm = np.linalg.norm(tan_sum, axis=-1, keepdims=True)
    gt_tangent = tan_sum / np.maximum(tan_norm, 1e-6)

    # Depth/position: frontmost hair sub-sample (minimum depth).
    dep_masked = np.where(hair, dep, np.float32(1.0))
    gt_depth = dep_masked.min(axis=(1, 3))[..., None]

    gt_mask = (gt_coverage[..., 0] > 0.0).astype(np.float32)[..., None]

    return {
        "coverage": gt_coverage.astype(np.float32),
        "tangent": gt_tangent.astype(np.float32),
        "depth": gt_depth.astype(np.float32),
        "hair_mask": gt_mask,
    }


def process_frame(args):
    dump_dir, tag, out_dir = args
    dump_dir = Path(dump_dir)
    meta = json.loads((dump_dir / "meta.json").read_text())
    frame_meta = next(f for f in meta["frames"] if f["frame"] == int(tag.split("_")[1]))

    width, height = meta["input_resolution"]
    gt_width, gt_height = meta["gt_resolution"]
    ssaa = meta["ssaa_factor"]

    # glm is column-major and the JSON stores columns sequentially, so a
    # row-major reshape yields the transpose - fix it before using.
    view = np.array(frame_meta["view"], dtype=np.float64).reshape(4, 4).T
    projection = np.array(frame_meta["projection"], dtype=np.float64).reshape(4, 4).T

    def load(sub: str, suffix: str, dtype: str, channels: int, w: int, h: int):
        path = dump_dir / sub / f"{tag}{suffix}"
        data = np.fromfile(path, dtype=dtype).astype(np.float32)
        # Some hair files carry zero-length tangents at strand tips: the
        # old dumps wrote NaN/Inf there. Zero them out defensively.
        data[np.isnan(data)] = 0.0
        data[np.isinf(data)] = 0.0
        return data.reshape(h, w, channels)

    # Undersampled input.
    in_coverage = load("input", "_coverage.f16", "<f2", 1, width, height)
    in_tangent = load("input", "_tangent.f16", "<f2", 4, width, height)[:, :, :3]
    in_motion = load("input", "_motion.f16", "<f2", 2, width, height)
    in_depth = load("input", "_depth.f32", "<f4", 1, width, height)

    in_tangent /= np.maximum(np.linalg.norm(in_tangent, axis=-1, keepdims=True), 1e-6)
    in_position = reconstruct_position(in_depth, view, projection)

    # High-sample ground truth, downsampled to the base resolution.
    gt_coverage = load("gt", "_coverage.f16", "<f2", 1, gt_width, gt_height)
    gt_tangent = load("gt", "_tangent.f16", "<f2", 4, gt_width, gt_height)[:, :, :3]
    gt_depth = load("gt", "_depth.f32", "<f4", 1, gt_width, gt_height)

    gt = downsample_gt(gt_coverage, gt_tangent, gt_depth, ssaa)
    gt["position"] = reconstruct_position(gt["depth"], view, projection)

    out_path = Path(out_dir) / f"{tag}.npz"
    np.savez_compressed(
        out_path,
        **{f"input/{k}": v for k, v in {
            "coverage": in_coverage.astype(np.float32),
            "tangent": in_tangent.astype(np.float32),
            "position": in_position,
            "depth": in_depth.astype(np.float32),
            "motion": in_motion.astype(np.float32),
        }.items()},
        **{f"gt/{k}": v for k, v in gt.items()},
        matrices=np.stack([view, projection]).astype(np.float32),
        camera_position=np.array(frame_meta["camera_position"], dtype=np.float32),
    )
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_root", help="e.g. dumps/dataset_full")
    parser.add_argument("--out", default=None, help="output directory (default: <dump_root>_prepared)")
    parser.add_argument("--workers", type=int, default=8)
    options = parser.parse_args()

    dump_root = Path(options.dump_root)
    out_root = Path(options.out) if options.out else Path(str(dump_root) + "_prepared")
    out_root.mkdir(parents=True, exist_ok=True)

    tasks, style_index = [], {}
    for dump_dir in sorted(p for p in dump_root.iterdir() if (p / "meta.json").exists()):
        meta = json.loads((dump_dir / "meta.json").read_text())
        if not meta.get("complete", False):
            print(f"skipping incomplete {dump_dir.name}")
            continue

        style = dump_dir.name.rsplit("_", 1)[0]
        split = dump_dir.name.rsplit("_", 1)[1]
        style_index.setdefault(style, {})[split] = str(dump_dir)

        width, height = meta["input_resolution"]
        tags = sorted(p.name[: -len("_coverage.f16")]
                      for p in (dump_dir / "input").glob("*_coverage.f16"))
        frame_out = out_root / dump_dir.name
        frame_out.mkdir(parents=True, exist_ok=True)

        for tag in tags:
            tasks.append((dump_dir, tag, frame_out))

    print(f"preparing {len(tasks)} frames with {options.workers} workers...")
    with ProcessPoolExecutor(max_workers=options.workers) as pool:
        for i, path in enumerate(pool.map(process_frame, tasks)):
            if (i + 1) % 50 == 0:
                print(f"  {i + 1} / {len(tasks)}")

    index = {
        "styles": {style: splits for style, splits in style_index.items()},
        "frames_per_run": {d.name: len(list(d.glob('*.npz'))) for d in sorted(out_root.iterdir()) if d.is_dir()},
        "layout": {
            "input": ["coverage", "tangent", "position", "depth", "motion"],
            "gt": ["coverage", "tangent", "position", "depth", "hair_mask"],
        },
    }
    (out_root / "index.json").write_text(json.dumps(index, indent=2))
    print(f"done: {out_root}")


if __name__ == "__main__":
    main()
