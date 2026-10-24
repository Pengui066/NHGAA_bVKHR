"""S4-3 fp16 acceptance criteria v1.2 (declared 2026-10-24, BEFORE running).

Same-input PAIRED test: one readback frame feeds the fp32 and the fp16
TensorRT engines; the GATE is PSNR(fp16_out, GT) >= PSNR(fp32_out, GT)
- 0.1 dB on coverage (over the GT hair mask) for ponytail/bear/wcurly,
1 frame each, plus fp16_out free of NaN/inf. The tangent angular error
and the direct PSNR(fp16_out, fp32_out) are recorded as reference only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["PATH"] = r"D:\pengui000\Denoise\vkhr\bin;" + os.environ["PATH"]

import numpy as np
import torch  # noqa: F401  (loads the CUDA chain DLLs first, project convention)
import onnxruntime as ort

ROOT = Path(r"D:\pengui000\Denoise\vkhr")
MODEL = ROOT / "runs/psi_v1/psi.onnx"
PREPARED = ROOT / "dumps/prepared_full"
CACHE = ROOT / "runs/trt_cache"
STYLES = ["ponytail", "bear", "wcurly"]
GATE_DB = 0.1


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(((pred - gt) ** 2).mean())
    return float("inf") if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def angle_error_deg(pred_tan: np.ndarray, gt_tan: np.ndarray, eps: float = 1e-8) -> float:
    p = pred_tan / np.maximum(np.linalg.norm(pred_tan, axis=-1, keepdims=True), eps)
    g = gt_tan / np.maximum(np.linalg.norm(gt_tan, axis=-1, keepdims=True), eps)
    cos = np.clip(np.sum(p * g, axis=-1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos).mean()))


def make_session(fp16: bool) -> ort.InferenceSession:
    options = {
        "device_id": "0",
        "trt_fp16_enable": "True" if fp16 else "False",
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": str(CACHE),
        "trt_builder_optimization_level": "2",  # parity with the C++ --nn-provider trt
    }
    return ort.InferenceSession(
        str(MODEL), ort.SessionOptions(),
        providers=[("TensorrtExecutionProvider", options),
                   ("CUDAExecutionProvider", {})])


def load_frame(style: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = sorted((PREPARED / f"{style}_train").glob("frame_*.npz"))
    data = np.load(frames[0])
    x = np.concatenate([data["input/coverage"], data["input/tangent"]], axis=-1)
    x = np.ascontiguousarray(x).transpose(2, 0, 1)[None].astype(np.float32)
    return x, data["gt/coverage"][..., 0], data["gt/tangent"]


def main() -> int:
    print(f"gate: PSNR(fp16,GT) >= PSNR(fp32,GT) - {GATE_DB} dB "
          f"(coverage, GT-coverage>0.1 mask); styles {STYLES}; NaN/inf free",
          flush=True)
    sess32 = make_session(False)
    sess16 = make_session(True)
    print("sessions ready:", sess32.get_providers(), flush=True)

    verdicts = []
    for style in STYLES:
        x, gt_cov, gt_tan = load_frame(style)
        mask = gt_cov > 0.1
        print(f"[{style}] frame {PREPARED / f'{style}_train'} input {x.shape} "
              f"hair px {int(mask.sum())}", flush=True)

        out32 = sess32.run(None, {"input": x})
        out16 = sess16.run(None, {"input": x})
        cov32, tan32 = out32[0][0, 0], out32[1][0].transpose(1, 2, 0)
        cov16, tan16 = out16[0][0, 0], out16[1][0].transpose(1, 2, 0)

        nan16 = bool(np.isnan(cov16).any() or np.isnan(tan16).any()
                     or np.isinf(cov16).any() or np.isinf(tan16).any())
        psnr32 = psnr(cov32[mask], gt_cov[mask])
        psnr16 = psnr(cov16[mask], gt_cov[mask])
        direct = psnr(cov16[mask], cov32[mask])
        a32 = angle_error_deg(tan32[mask], gt_tan[mask])
        a16 = angle_error_deg(tan16[mask], gt_tan[mask])
        delta = psnr16 - psnr32
        passed = delta >= -GATE_DB and not nan16
        verdicts.append(passed)
        print(f"  coverage PSNR fp32 {psnr32:.3f} dB | fp16 {psnr16:.3f} dB | "
              f"delta {delta:+.3f} dB (gate -{GATE_DB}) | direct(fp16,fp32) "
              f"{direct:.2f} dB | tangent fp32 {a32:.4f} deg | fp16 {a16:.4f} deg | "
              f"NaN/inf {nan16} => {'PASS' if passed else 'FAIL'}", flush=True)

    print("VERDICT:", "PASS" if all(verdicts) else "FAIL", flush=True)
    return 0 if all(verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
