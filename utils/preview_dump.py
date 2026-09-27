#!/usr/bin/env python3
"""Preview a vkhr G-buffer dump: converts the raw channel files into
PNG contact sheets for quick visual inspection.

Usage: python utils/preview_dump.py dumps/test
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_raw(path: Path, dtype: str, channels: int, width: int, height: int) -> np.ndarray:
    data = np.fromfile(path, dtype=dtype)
    expected = width * height * channels
    if data.size != expected:
        raise ValueError(f"{path.name}: expected {expected} values, got {data.size}")
    return data.reshape(height, width, channels)


def tonemap(image: np.ndarray) -> np.ndarray:
    """Maps arbitrary float data to displayable uint8."""
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(image, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1e-6
    return (np.clip((image - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def to_u8(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "linear":  # assume light values in [0, ~1+]
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return tonemap(image)


def channel_grid(name: str, image: np.ndarray, mode: str) -> Image.Image:
    """Renders an HxWxC float image (C <= 4) as a labelled grayscale/RGB tile."""
    channels = [image[:, :, i] for i in range(image.shape[2])]
    tiles = [to_u8(c[None, :, :].transpose(1, 2, 0).repeat(3, 2), mode) for c in channels]
    tiles = [t.reshape(image.shape[0], image.shape[1], 3) for t in tiles]

    # pad to 4 tiles for a consistent 2x2 grid.
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))

    top = np.concatenate(tiles[0:2], axis=1)
    bottom = np.concatenate(tiles[2:4], axis=1)
    grid = np.concatenate([top, bottom], axis=0)

    return Image.fromarray(grid)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dump_dir = Path(sys.argv[1])
    meta = json.loads((dump_dir / "meta.json").read_text())

    width, height = meta["input_resolution"]
    gt_width, gt_height = meta["gt_resolution"]

    out_dir = dump_dir / "preview"
    out_dir.mkdir(exist_ok=True)

    frames = sorted((dump_dir / "input").glob("*_coverage.f16"))
    for input_coverage_path in frames:
        tag = input_coverage_path.stem.replace("_coverage", "")

        input_coverage = load_raw(dump_dir / "input" / f"{tag}_coverage.f16", "<f2", 1, width, height)
        input_tangent = load_raw(dump_dir / "input" / f"{tag}_tangent.f16", "<f2", 4, width, height)
        input_depth = load_raw(dump_dir / "input" / f"{tag}_depth.f32", "<f4", 1, width, height)
        input_shaded = load_raw(dump_dir / "input" / f"{tag}_shaded.f32", "<f4", 4, width, height)
        input_background = load_raw(dump_dir / "input" / f"{tag}_background.f32", "<f4", 4, width, height)

        gt_coverage = load_raw(dump_dir / "gt" / f"{tag}_coverage.f16", "<f2", 1, gt_width, gt_height)
        gt_shaded = load_raw(dump_dir / "gt" / f"{tag}_shaded.f32", "<f4", 4, gt_width, gt_height)

        # 1. The shaded input (the "Input" baseline) and shaded GT (Reference).
        Image.fromarray(to_u8(input_shaded[:, :, :3], "linear")).save(out_dir / f"{tag}_shaded_input.png")
        reference = Image.fromarray(to_u8(gt_shaded[:, :, :3], "linear"))
        reference.save(out_dir / f"{tag}_shaded_reference.png")

        # 2. Coverage: input on top, GT (downsampled for display) below.
        gt_small = np.array(reference.resize((width, height), Image.BOX))
        coverage_pair = np.concatenate([
            to_u8(input_coverage, "tonemap").reshape(height, width, 1).repeat(3, 2),
            to_u8(gt_small[..., :3], "linear"),
        ], axis=0)
        Image.fromarray(coverage_pair).save(out_dir / f"{tag}_coverage_pair.png")

        # 3. Tangent visualization (world-space, canonical orientation).
        channel_grid("tangent", input_tangent[:, :, :3], "linear").save(
            out_dir / f"{tag}_tangent_channels.png")

        # 4. Depth + background.
        channel_grid("depth", input_depth, "tonemap").save(out_dir / f"{tag}_depth.png")
        Image.fromarray(to_u8(input_background[:, :, :3], "linear")).save(
            out_dir / f"{tag}_background.png")

        # 5. Motion vectors (if present).
        motion_path = dump_dir / "input" / f"{tag}_motion.f16"
        if motion_path.exists():
            motion = load_raw(motion_path, "<f2", 2, width, height)
            motion_rgb = np.zeros((height, width, 3), dtype=np.float32)
            motion_rgb[:, :, 0] = motion[:, :, 0] * 0.5 + 0.5
            motion_rgb[:, :, 1] = motion[:, :, 1] * 0.5 + 0.5
            Image.fromarray(to_u8(motion_rgb, "linear")).save(out_dir / f"{tag}_motion.png")
            moved = np.count_nonzero(np.abs(motion).sum(axis=2) > 1e-4)
            print(f"{tag}: motion pixels {moved} / {width * height}")

        hair_pixels = np.count_nonzero(input_coverage > 0)
        gt_small_cov = np.array(
            Image.fromarray((np.clip(gt_coverage, 0, 1) * 255).astype(np.uint8)
                            .reshape(gt_height, gt_width)).resize((width, height), Image.BOX))
        print(f"{tag}: input hair pixels {hair_pixels} ({100.0 * hair_pixels / (width * height):.2f}%), "
              f"GT hair pixels {np.count_nonzero(gt_small_cov > 2)}")

    print(f"previews written to {out_dir}")


if __name__ == "__main__":
    main()
