"""Chain audit for the real-time neural path (--nn-debug-dump).

Answers two questions with numbers:
  1. Chain fidelity: does the C++ pipeline's reconstructed output match
     running the exported ONNX graph in Python on the SAME readback data?
     (max abs diff should be ~1e-3, the fp16 upload quantizes nothing here
     since both sides work in fp32.)
  2. Distribution: how do the real-time 1spp coverage and the reconstructed
     coverage compare with the offline prepared dataset (same hair style)?
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
from pathlib import Path

import numpy as np
import torch  # adds torch/lib to the DLL path (CUDA chain for the ORT CUDA EP)
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nn"))


def load_real_time(dump_dir: Path):
    cov = np.fromfile(dump_dir / "coverage_readback.f16",
                      dtype=np.uint16).view(np.float16).astype(np.float32)
    tan = np.fromfile(dump_dir / "tangent_readback.f16",
                      dtype=np.uint16).view(np.float16).astype(np.float32)
    meta = json.load(open(dump_dir / "meta.json"))
    h, w = meta["resolution"]
    cov = cov.reshape(h, w)
    tan = tan.reshape(h, w, 4)[..., :3]
    # The same sanitization the C++ side applies (contract: nan_to_num).
    cov = np.where(np.isfinite(cov), cov, 0.0)
    tan = np.where(np.isfinite(tan), tan, 0.0)
    return cov, tan, meta


def main():
    dump_dir = Path("dumps/nn_debug")
    cov, tan, meta = load_real_time(dump_dir)
    h, w = cov.shape
    print(f"resolution {w}x{h} | camera {meta['camera_position']}")

    # ---- 1. chain fidelity: Python ONNX vs the C++ reconstruction ----
    x = np.stack([cov, tan[..., 0], tan[..., 1], tan[..., 2]], axis=0)[None]
    session = ort.InferenceSession("runs/psi_v1/psi.onnx",
                                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    outputs = session.run(None, {"input": np.ascontiguousarray(x)})
    py_cov = np.asarray(outputs[0])[0, 0]
    py_tan = np.asarray(outputs[1])[0].transpose(1, 2, 0)

    cpp_cov = np.fromfile(dump_dir / "recon_coverage.f32",
                          dtype=np.float32).reshape(h, w)
    cpp_tan = np.fromfile(dump_dir / "recon_tangent.f32",
                          dtype=np.float32).reshape(h, w, 3)

    print(f"\n=== chain fidelity (Python ONNX vs C++ recon) ===")
    print(f"coverage: max_abs {np.abs(py_cov - cpp_cov).max():.3e}  "
          f"mean_abs {np.abs(py_cov - cpp_cov).mean():.3e}")
    print(f"tangent:  max_abs {np.abs(py_tan - cpp_tan).max():.3e}  "
          f"mean_abs {np.abs(py_tan - cpp_tan).mean():.3e}")

    # ---- 2. distributions vs the offline prepared dataset ----
    ref = np.load("dumps/prepared_full/ponytail_train/frame_00000.npz")
    off_in = ref["input/coverage"][..., 0]
    off_gt = ref["gt/coverage"][..., 0]

    def hair_stats(a):
        hair = a > 0.001
        if hair.sum() == 0:
            return 0.0, 0.0
        return float(hair.mean()), float(a[hair].mean())

    print(f"\n=== coverage distributions (ponytail) ===")
    for name, arr in [("real-time readback (1spp)", cov),
                      ("real-time recon (psi)", cpp_cov),
                      ("offline input (1spp)", off_in),
                      ("offline GT (16spp)", off_gt)]:
        frac, mean = hair_stats(arr)
        print(f"{name:28s} hair-fraction {frac:6.1%}  mean-coverage-on-hair {mean:.3f}")

    # Offline psi output on the same offline frame, for a recon-vs-recon
    # distribution comparison.
    off_tan = ref["input/tangent"]
    x_off = np.stack([off_in, off_tan[..., 0], off_tan[..., 1], off_tan[..., 2]],
                     axis=0)[None]
    off_recon = np.asarray(session.run(None, {"input": np.ascontiguousarray(x_off)})[0])[0, 0]
    frac, mean = hair_stats(off_recon)
    print(f"{'offline recon (psi)':28s} hair-fraction {frac:6.1%}  mean-coverage-on-hair {mean:.3f}")

    # ---- 3. shaded image brightness (for the darkening decomposition) ----
    shaded = np.fromfile(dump_dir / "shaded.f32", dtype=np.float32).reshape(h, w, 4)
    hair = cpp_cov > 0.001
    print(f"\n=== shaded.f32 (linear RGBA32F) ===")
    print(f"hair-pixel mean RGB: {shaded[hair, :3].mean(axis=0)}")
    print(f"whole-image mean RGB: {shaded[:, :, :3].mean(axis=0)}")
    print(f"background (no-hair) mean RGB: {shaded[~hair, :3].mean(axis=0)}")


if __name__ == "__main__":
    main()
