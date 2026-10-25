"""Gate 1 of the ψ-toggle feature: same-input ψ consistency.

Replays a debug dump's own decoded input channels (coverage_decoded.f32 +
tangent_decoded.f32 — the exact fp16->fp32 values the C++ packed) through
the SAME cached fp16 TensorRT engine the C++ path uses, and compares the
outputs bit-exactly (max_abs must be 0.0) against the dump's recon files.
Both the old (main) and the new (feature) build must satisfy this, which
makes their ψ stages bit-identical through the model — no cross-run hash
comparison (F-25).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["PATH"] = r"D:\pengui000\Denoise\vkhr\bin;" + os.environ["PATH"]

import numpy as np
import torch  # noqa: F401
import onnxruntime as ort

ROOT = Path(r"D:\pengui000\Denoise\vkhr")
MODEL = ROOT / "runs/psi_v1/psi.onnx"
CACHE = ROOT / "runs/trt_cache"


def make_session() -> ort.InferenceSession:
    options = {
        "device_id": "0",
        "trt_fp16_enable": "True",  # the dump-producing runs used --nn-fp16 yes
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": str(CACHE),
        "trt_builder_optimization_level": "2",
        "trt_timing_cache_enable": "True",
        "trt_timing_cache_path": str(CACHE),
    }
    return ort.InferenceSession(str(MODEL), ort.SessionOptions(),
                                providers=[("TensorrtExecutionProvider", options),
                                           ("CUDAExecutionProvider", {})])


def check_dump(dump_dir: Path) -> bool:
    meta = None
    cov = np.fromfile(dump_dir / "coverage_decoded.f32", dtype=np.float32)
    tan = np.fromfile(dump_dir / "tangent_decoded.f32", dtype=np.float32)
    # The decoded channels already carry the exact per-pixel values the
    # C++ packed into [C, T] order (H, W, 4) -> NCHW.
    h = w = int(round(cov.size ** 0.5))
    # Prefer the meta resolution if present.
    import json
    if (dump_dir / "meta.json").exists():
        meta = json.loads((dump_dir / "meta.json").read_text())
        w, h = meta["resolution"]  # stored as { width, height }
    cov = cov[: h * w]
    tan = tan[: 3 * h * w]
    x = np.empty((h, w, 4), dtype=np.float32)
    x[..., 0] = cov.reshape(h, w)
    x[..., 1:4] = tan.reshape(h, w, 3)
    x = np.ascontiguousarray(x).transpose(2, 0, 1)[None]

    out = sess.run(None, {"input": x})
    out_cov = out[0][0, 0]
    out_tan = out[1][0].transpose(1, 2, 0)

    ref_cov = np.fromfile(dump_dir / "recon_coverage.f32", dtype=np.float32).reshape(h, w)
    ref_tan = np.fromfile(dump_dir / "recon_tangent.f32", dtype=np.float32).reshape(h, w, 3)
    dc = float(np.abs(out_cov - ref_cov).max())
    dt = float(np.abs(out_tan - ref_tan).max())
    ok = dc == 0.0 and dt == 0.0
    print(f"{dump_dir.name}: coverage max_abs {dc:.3e} | tangent max_abs {dt:.3e} => "
          f"{'BIT-IDENTICAL' if ok else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    sess = make_session()
    dirs = [Path(d) for d in sys.argv[1:]]
    results = [check_dump(d) for d in dirs]
    print("VERDICT:", "PASS" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)
