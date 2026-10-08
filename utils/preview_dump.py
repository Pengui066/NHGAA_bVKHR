#!/usr/bin/env python3
"""Preview a vkhr G-buffer dump: converts the raw channel files into
PNG images for quick visual inspection.

Usage: python utils/preview_dump.py dumps/test [--seam-free]
"""
import argparse
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


def tangent_rgb(tangent: np.ndarray, coverage: np.ndarray,
                seam_free: bool = False) -> Image.Image:
    """Paper-style tangent display: R/X, G/Y, B/Z, drawn on hair pixels
    only (background stays black).

    Default encoding is t -> 0.5*t + 0.5. The stored tangent sign is
    camera-aligned, which flips whole strand bundles where they run
    perpendicular to the view direction (e.g. a hanging ponytail) and
    shows up as a hard hue seam between t and -t.

    seam_free=True switches to the sign-invariant encoding rgb = |t|:
    a strand tangent satisfies t == -t, so the sign carries no
    displayable information and component magnitudes are the honest
    seam-free image. Per-pixel sign re-orientation does NOT work — any
    per-pixel sign convention is discontinuous somewhere on the sphere
    of directions and merely moves the seam (observed: it reappears on
    dominant-axis boundaries). Trade-off: |t| identifies directions that
    differ by mirroring one component, which are distinct lines; for a
    preview this is far less misleading than fake sign seams. Display
    only — the dump's stored convention is untouched.
    """
    t = np.nan_to_num(tangent[:, :, :3], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if seam_free:
        t = np.abs(t)
    rgb = t if seam_free else t * 0.5 + 0.5
    rgb *= (coverage[:, :, 0] > 0)[:, :, None]
    return Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))


def tonemap_pair(a: np.ndarray, b: np.ndarray):
    """Tonemaps two images against a SHARED percentile range so their
    brightness/contrast can be compared directly."""
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(np.concatenate([a.ravel(), b.ravel()]), [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1e-6
    stretch = lambda x: Image.fromarray(
        (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8))
    return stretch(a), stretch(b)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a vkhr G-buffer dump into preview PNGs.")
    parser.add_argument("dump_dir", type=Path, help="e.g. dumps/dataset_full/ponytail_eval")
    parser.add_argument("--seam-free", action="store_true",
                        help="re-orient tangent colors so the camera-alignment "
                             "sign flip does not show as a hue seam (display only)")
    args = parser.parse_args()
    seam_free = args.seam_free

    dump_dir = args.dump_dir
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

        # 3. Tangent: paper-style single RGB image, plus an input-vs-GT
        # comparison with the GT resampled to the input display scale.
        gt_tangent = load_raw(dump_dir / "gt" / f"{tag}_tangent.f16", "<f2", 4, gt_width, gt_height)

        input_tangent_rgb = tangent_rgb(input_tangent, input_coverage, seam_free)
        input_tangent_rgb.save(out_dir / f"{tag}_tangent_rgb.png")

        gt_tangent_small = tangent_rgb(gt_tangent, gt_coverage, seam_free).resize((width, height), Image.BOX)
        bands = Image.new("RGB", (width, 2 * height), "black")
        bands.paste(input_tangent_rgb, (0, 0))
        bands.paste(gt_tangent_small, (0, height))
        orientation = "sign-invariant |t| display" if seam_free else "camera-aligned sign"
        labeled(bands, f"{tag}: input tangent (1 spp) | GT tangent ({ssaa}x SSAA), {orientation}").save(
            out_dir / f"{tag}_tangent_comparison.png")

        # 4. Depth: single view, plus an input-vs-GT comparison stretched to
        # a shared range (so both bands are directly comparable).
        gt_depth = load_raw(dump_dir / "gt" / f"{tag}_depth.f32", "<f4", 1, gt_width, gt_height)
        channel_grid(input_depth, "auto").save(out_dir / f"{tag}_depth.png")

        input_depth_img, gt_depth_img = tonemap_pair(input_depth[:, :, 0], gt_depth[:, :, 0])
        bands = Image.new("RGB", (width, 2 * height), "black")
        bands.paste(input_depth_img, (0, 0))
        bands.paste(gt_depth_img.resize((width, height), Image.BOX), (0, height))
        labeled(bands, f"{tag}: input depth (1 spp) | GT depth ({ssaa}x SSAA)").save(
            out_dir / f"{tag}_depth_comparison.png")
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
