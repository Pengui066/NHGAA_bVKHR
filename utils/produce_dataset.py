#!/usr/bin/env python3
"""Batch-produce the hair G-buffer training dataset by driving vkhr's
dump mode over all hairstyles.

Every (style, split) pair becomes one dump run with a distinct seed, so
runs are individually resumable: a run whose meta.json says
"complete": true is skipped on re-invocation.

Pilot default: 5 styles x 32 training frames (G-buffer channels only,
~68 MB/frame) + 8 evaluation frames per style (full channels with the
shaded input/reference images, ~215 MB/frame).

Usage:
    python utils/produce_dataset.py --pilot            # ~20 GB, 200 frames
    python utils/produce_dataset.py --full             # ~90 GB, 1000 frames
    python utils/produce_dataset.py --styles wstraight # subset of styles
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VKHR = REPO / "bin" / "vkhr.exe"
OUT = REPO / "dumps" / "dataset"

# Per-style overrides: the wigs need closer cameras (their bounding
# volume includes the long hair hanging down) and thicker strands to
# show structure at 720p. Distances are in bounding radii.
STYLES = {
    "ponytail":  {"scene": "ponytail.vkhr",  "distance": (2.5, 5.0), "radius": (0.25, 0.60)},
    "bear":      {"scene": "bear.vkhr",      "distance": (2.0, 4.0), "radius": (0.30, 0.70)},
    "wstraight": {"scene": "wstraight.vkhr", "distance": (1.2, 2.5), "radius": (1.50, 3.00)},
    "wwavy":     {"scene": "wwavy.vkhr",     "distance": (1.2, 2.5), "radius": (1.50, 3.00)},
    "wcurly":    {"scene": "wcurly.vkhr",    "distance": (1.2, 2.5), "radius": (1.50, 3.00)},
}


def run_dump(style: str, split: str, frames: int, seed: int, shaded: bool,
             ssaa: int, out_root: Path) -> bool:
    info = STYLES[style]
    out_dir = out_root / f"{style}_{split}"

    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("complete"):
            print(f"[skip] {out_dir.name}: already complete")
            return True
        print(f"[redo] {out_dir.name}: incomplete dump found, re-running")

    distance = info["distance"]
    command = [
        str(VKHR),
        "--dump", "yes",
        "--dump-dir", str(out_dir),
        "--dump-frames", str(frames),
        "--dump-ssaa", str(ssaa),
        "--camera-script", "random",
        "--camera-seed", str(seed),
        "--distance-min", str(distance[0]),
        "--distance-max", str(distance[1]),
        "--elevation-min", "-20",
        "--elevation-max", "40",
        "--radius-min", str(info["radius"][0]),
        "--radius-max", str(info["radius"][1]),
        "--light-random", "yes",
        "--dump-shaded", "yes" if shaded else "no",
        str(REPO / "share" / "scenes" / info["scene"]),
    ]

    started = time.time()
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    elapsed = time.time() - started

    meta_path = out_dir / "meta.json"
    complete = meta_path.exists() and json.loads(meta_path.read_text()).get("complete", False)

    status = "ok" if complete and result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[done] {out_dir.name}: {frames} frames in {elapsed:.0f}s "
          f"({elapsed / max(frames, 1):.1f}s/frame) - {status}")

    if not complete:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="5 styles x 32 train + 8 eval frames")
    parser.add_argument("--full", action="store_true", help="5 styles x 160 train + 40 eval frames")
    parser.add_argument("--styles", nargs="*", default=list(STYLES), help="subset of styles")
    parser.add_argument("--train-frames", type=int, default=None)
    parser.add_argument("--eval-frames", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--ssaa", type=int, default=4, help="GT supersampling per axis")
    parser.add_argument("--out-root", default=str(OUT))
    options = parser.parse_args()

    if options.pilot:
        train_frames, eval_frames = 32, 8
    elif options.full:
        train_frames, eval_frames = 160, 40
    else:
        train_frames = options.train_frames or 32
        eval_frames = options.eval_frames or 8

    total_gb, total_frames, failures = 0.0, 0, []

    for style in options.styles:
        if style not in STYLES:
            print(f"unknown style: {style}, choose from {list(STYLES)}")
            sys.exit(1)

        out_root = Path(options.out_root)

        # Training split: G-buffer channels only (network supervision).
        if run_dump(style, "train", train_frames, options.seed_base, shaded=False,
                    ssaa=options.ssaa, out_root=out_root):
            total_frames += train_frames
        else:
            failures.append(f"{style}_train")

        # Evaluation split: full channels incl. shaded input/reference.
        if run_dump(style, "eval", eval_frames, options.seed_base + 500, shaded=True,
                    ssaa=options.ssaa, out_root=out_root):
            total_frames += eval_frames
        else:
            failures.append(f"{style}_eval")

    print(f"\nplanned frames this session: {total_frames}")
    if failures:
        print(f"FAILED runs: {failures}")
        sys.exit(1)
    print("all dumps complete")


if __name__ == "__main__":
    main()
