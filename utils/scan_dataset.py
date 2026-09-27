#!/usr/bin/env python3
"""Scan a produced dataset and verify its integrity: complete flags,
file presence, per-frame sizes, hair-pixel coverage statistics and the
input-vs-GT sampling gap (the quantity the network has to close).

Usage:
    python utils/scan_dataset.py dumps/dataset
    python utils/scan_dataset.py dumps/dataset --sample 8   # stats on every Nth frame
"""
import json
import sys
from pathlib import Path

import numpy as np

EXPECTED_TRAIN = {
    "input": ["_coverage.f16", "_tangent.f16", "_motion.f16", "_depth.f32"],
    "gt": ["_coverage.f16", "_tangent.f16", "_depth.f32"],
}
EXPECTED_EVAL = {
    "input": EXPECTED_TRAIN["input"] + ["_background.f32", "_shaded.f32"],
    "gt": EXPECTED_TRAIN["gt"] + ["_background.f32", "_shaded.f32"],
}

BYTES_PER_PIXEL = {
    "_coverage.f16": 2, "_tangent.f16": 8, "_motion.f16": 4,
    "_depth.f32": 4, "_background.f32": 16, "_shaded.f32": 16,
}


def check_frame(frame_dir: Path, tag: str, width: int, height: int, expected: dict) -> str:
    for sub, channels in expected.items():
        for suffix in channels:
            path = frame_dir / sub / (tag + suffix)
            if not path.exists():
                return f"missing {sub}/{tag}{suffix}"
            expected_size = width * height * BYTES_PER_PIXEL[suffix]
            actual_size = path.stat().st_size
            if suffix == "_tangent.f16" and sub == "input":
                expected_size = width * height * 8
            if actual_size != expected_size:
                return f"size mismatch {sub}/{tag}{suffix}: {actual_size} != {expected_size}"
    return "ok"


def coverage_stats(path: Path, width: int, height: int, stride: int = 7) -> dict:
    """Downsampled coverage statistics (fast: reads a strided sample)."""
    raw = np.fromfile(path, dtype="<f2")
    sample = raw[::stride]
    coverage = np.clip(sample, 0, 1)
    return {
        "hair_fraction": float(np.count_nonzero(coverage > 0)) / coverage.size,
        "mean_coverage": float(coverage.sum()) / coverage.size,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    root = Path(sys.argv[1])
    sample_stride = 7
    if "--sample" in sys.argv:
        sample_stride = int(sys.argv[sys.argv.index("--sample") + 1])

    dump_dirs = sorted(p for p in root.iterdir() if (p / "meta.json").exists())
    if not dump_dirs:
        print(f"no dumps found under {root}")
        sys.exit(1)

    problems = []
    print(f"{'run':24s} {'frames':>6s} {'res':>11s} {'hair%':>6s} {'GT hair%':>8s} "
          f"{'gap':>5s} {'cov mean':>8s} {'state':>8s}")
    print("-" * 86)

    for dump_dir in dump_dirs:
        meta = json.loads((dump_dir / "meta.json").read_text())
        if not meta.get("complete", False):
            print(f"{dump_dir.name:24s} {'-':>6s} {'-':>11s} {'-':>6s} {'-':>8s} "
                  f"{'-':>5s} {'-':>8s} {'INCOMPLETE':>8s}")
            problems.append(f"{dump_dir.name}: incomplete")
            continue

        width, height = meta["input_resolution"]
        gt_width, gt_height = meta["gt_resolution"]
        expected = EXPECTED_EVAL if meta.get("dump_shaded", True) else EXPECTED_TRAIN

        input_dir = dump_dir / "input"
        tags = sorted(p.name[: -len("_coverage.f16")]
                      for p in input_dir.glob("*_coverage.f16"))

        # Spot-check sizes on the first and last frames.
        for tag in (tags[0], tags[-1]):
            state = check_frame(dump_dir, tag, width, height, expected)
            if state != "ok":
                problems.append(f"{dump_dir.name}/{tag}: {state}")

        # Hair statistics on strided frames.
        hair_fractions, gt_fractions, means = [], [], []
        for tag in tags[::sample_stride]:
            in_cov = coverage_stats(input_dir / f"{tag}_coverage.f16", width, height)
            gt_cov = coverage_stats(dump_dir / "gt" / f"{tag}_coverage.f16", gt_width, gt_height)
            hair_fractions.append(in_cov["hair_fraction"])
            gt_fractions.append(gt_cov["hair_fraction"])
            means.append(in_cov["mean_coverage"])

        gap = (np.mean(gt_fractions) - np.mean(hair_fractions)) if hair_fractions else 0.0
        print(f"{dump_dir.name:24s} {len(tags):>6d} {f'{width}x{height}':>11s} "
              f"{100 * np.mean(hair_fractions):>5.1f}% {100 * np.mean(gt_fractions):>7.1f}% "
              f"{100 * gap:>4.1f}% {np.mean(means):>8.4f} {'ok' if not any(dump_dir.name in p for p in problems) else 'BAD':>8s}")

    print()
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("dataset integrity: OK")


if __name__ == "__main__":
    main()
