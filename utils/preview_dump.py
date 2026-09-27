#!/usr/bin/env python3
"""Preview a vkhr G-buffer dump: converts the raw channel files into
PNG images for quick visual inspection.

Usage: python utils/preview_dump.py dumps/test
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_raw(path: Path, dtype: str, channels: int, width: int, height: int) -> np.ndarray:
    data = np.fromfile(path, dtype=dtype)
    expected = width * height * channels
    if data.size != expected:
        raise ValueError(
            f"{path.name}: expected {expected} values ({width}x{height}x{channels}), "
            f"got {data.size} ({data.size / channels:.0f} pixels).\n"
            "The dump in this directory is incomplete or was written by a run with "
            "different settings. Re-run the vkhr dump command into a fresh directory."
        )
    return data.reshape(height, width, channels)


def tonemap(image: np.ndarray) -> np.ndarray:
    """Maps arbitrary float data to displayable uint8 (1st-99.5th percentile)."""
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(image, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1e-6
    return (np.clip((image - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def to_u8(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "linear":  # assume light values in [0, ~1+]
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return tonemap(image)


def grayscale(image: np.ndarray, mode: str) -> Image.Image:
    return Image.fromarray(to_u8(image, "tonemap" if mode == "auto" else mode))


def labeled(band: Image.Image, text: str) -> Image.Image:
    """Returns the band with a white caption strip above it."""
    strip = 26
    out = Image.new("RGB", (band.width, band.height + strip), "white")
    out.paste(band, (0, strip))
    draw = ImageDraw.Draw(out)
    draw.text((8, 6), text, fill="black")
    return out


def channel_grid(image: np.ndarray, mode: str) -> Image.Image:
    """Renders an HxWxC float image (C <= 4) as a 2x2 grid of channels."""
    channels = [image[:, :, i] for i in range(image.shape[2])]
    tiles = [to_u8(c, mode) for c in channels]
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))

    top = np.concatenate(tiles[0:2], axis=1)
    bottom = np.concatenate(tiles[2:4], axis=1)
    return Image.fromarray(np.concatenate([top, bottom], axis=0))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dump_dir = Path(sys.argv[1])
    meta = json.loads((dump_dir / "meta.json").read_text())

    if not meta.get("complete", True):
        print(f"error: '{dump_dir}' holds an INCOMPLETE dump (meta.json says "
              "complete=false). The dump process was interrupted or is still "
              "running. Re-run the vkhr dump command.")
        sys.exit(1)

    width, height = meta["input_resolution"]
    gt_width, gt_height = meta["gt_resolution"]
    ssaa = meta.get("ssaa_factor", gt_width // width)

    out_dir = dump_dir / "preview"
    out_dir.mkdir(exist_ok=True)

    print(f"{dump_dir}: {width}x{height} input, {gt_width}x{gt_height} GT "
          f"(ssaa x{ssaa})")

    frames = sorted((dump_dir / "input").glob("*_coverage.f16"))
    for input_coverage_path in frames:
        tag = input_coverage_path.stem.replace("_coverage", "")

        input_coverage = load_raw(dump_dir / "input" / f"{tag}_coverage.f16", "<f2", 1, width, height)
        input_tangent = load_raw(dump_dir / "input" / f"{tag}_tangent.f16", "<f2", 4, width, height)
        input_depth = load_raw(dump_dir / "input" / f"{tag}_depth.f32", "<f4", 1, width, height)

        gt_coverage = load_raw(dump_dir / "gt" / f"{tag}_coverage.f16", "<f2", 1, gt_width, gt_height)

        # Training dumps (--dump-shaded no) hold only the G-buffer channels;
        # evaluation dumps also carry the shaded input/reference images.
        input_shaded_path = dump_dir / "input" / f"{tag}_shaded.f32"
        if input_shaded_path.exists():
            input_shaded = load_raw(input_shaded_path, "<f4", 4, width, height)
            input_background = load_raw(dump_dir / "input" / f"{tag}_background.f32", "<f4", 4, width, height)
            gt_shaded = load_raw(dump_dir / "gt" / f"{tag}_shaded.f32", "<f4", 4, gt_width, gt_height)

            Image.fromarray(to_u8(input_shaded[:, :, :3], "linear")).save(out_dir / f"{tag}_shaded_input.png")
            Image.fromarray(to_u8(gt_shaded[:, :, :3], "linear")).save(out_dir / f"{tag}_shaded_reference.png")

        # 2. The coverage comparison, band by band, with captions:
        #    top    - the undersampled input coverage (grayscale),
        #    middle - the high-sample GT coverage, downsampled (grayscale),
        #    bottom - the GT after deferred shading (the "Reference" look).
        gt_coverage_small = np.array(grayscale(gt_coverage[:, :, 0], "linear")
                                     .resize((width, height), Image.BOX))

        bands = Image.new("RGB", (width, 3 * height), "black")
        bands.paste(grayscale(input_coverage[:, :, 0], "linear"), (0, 0))
        bands.paste(Image.fromarray(gt_coverage_small), (0, height))

        caption = f"{tag}: input coverage (1 spp, broken) | GT coverage ({ssaa}x SSAA, converged)"

        if input_shaded_path.exists():
            gt_shaded_small = np.array(Image.fromarray(to_u8(gt_shaded[:, :, :3], "linear"))
                                       .resize((width, height), Image.BOX))
            bands.paste(Image.fromarray(gt_shaded_small), (0, 2 * height))
            caption += " | GT shaded reference"

        comparison = labeled(bands, caption)
        comparison.save(out_dir / f"{tag}_coverage_comparison.png")

        # 3. Tangent visualization (world-space, canonical orientation).
        channel_grid(input_tangent[:, :, :3], "linear").save(
            out_dir / f"{tag}_tangent_channels.png")

        # 4. Depth (+ background, when present).
        channel_grid(input_depth, "auto").save(out_dir / f"{tag}_depth.png")
        if input_shaded_path.exists():
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
        gt_hair_pixels = np.count_nonzero(gt_coverage > 0) / (ssaa * ssaa)
        print(f"{tag}: input hair pixels {hair_pixels} "
              f"({100.0 * hair_pixels / (width * height):.2f}%), "
              f"GT hair pixels ~{gt_hair_pixels:.0f} (equivalent)")

    print(f"previews written to {out_dir}")


if __name__ == "__main__":
    main()
