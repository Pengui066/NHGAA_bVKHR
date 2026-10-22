"""Numerical consistency check: PyTorch Psi vs the exported ONNX graph.

Validates the two substitutions the ONNX export relies on:
  - scaled_dot_product_attention -> explicit matmul attention (math backend
    of SDPA is mathematically identical; ONNX cannot trace the fused op)
  - ONNX Runtime kernels vs PyTorch kernels (CPU and CUDA execution
    providers, fp32)

Runs the full inference-time pipeline (network + postprocess from
evaluate.py) on real held-out frames and on a synthetic stress batch,
then reports max/mean absolute differences per output and saves a JSON
report. Input sizes must be multiples of 4 (stride-2 downsampling
constraint of the network's decoder upsample); the window attention pads
internally to multiples of 8.

The support-gated outputs (coverage, tangent) are additionally scored on
the pixels where both implementations agree on the support mask, so that
threshold flips on near-zero logits (which only happen on adversarial
inputs such as pure noise) do not mask genuine graph differences.

Usage:
    python nn/test_onnx_consistency.py --checkpoint runs/psi_v1/best.pt \
        --onnx runs/psi_v1/psi.onnx --data dumps/prepared_full
"""
from __future__ import annotations

import os

# The conda MKL numpy and pip torch ship different libiomp5md.dll copies;
# without this guard the process aborts at OpenMP init (OMP Error #15).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import ValDataset
from models import SpatialNet
from export_onnx import PsiInference

OUTPUT_NAMES = ["coverage", "tangent", "reconstruction", "mask_logit"]
LOGIT_INDEX = OUTPUT_NAMES.index("mask_logit")
GATED = {"coverage", "tangent"}


def load_models(checkpoint: str, device: torch.device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    step = state.get("step", "?")
    models = {}
    for impl in ["sdpa", "matmul"]:
        net = SpatialNet(base_channels=32, attn_impl=impl).to(device)
        net.load_state_dict(state["model"])
        net.eval()
        models[impl] = PsiInference(net)
    return models, step


def torch_outputs(model: PsiInference, x: torch.Tensor):
    with torch.no_grad():
        return [t.cpu().numpy() for t in model(x)]


def ort_outputs(session, x: np.ndarray):
    start = time.perf_counter()
    outs = session.run(None, {"input": np.ascontiguousarray(x)})
    elapsed = time.perf_counter() - start
    return [np.asarray(o) for o in outs], elapsed


def compare(name: str, ref: np.ndarray, got: np.ndarray,
            agree: np.ndarray | None = None) -> dict:
    diff = np.abs(got - ref)
    result = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
              "shape": list(got.shape)}
    if agree is not None:
        mask = np.broadcast_to(agree, diff.shape)
        result["max_abs_agree"] = float(diff[mask].max())
        result["mean_abs_agree"] = float(diff[mask].mean())
    print(f"    {name:14s} max_abs {result['max_abs']:.3e}  "
          f"mean_abs {result['mean_abs']:.3e}"
          + (f"  | agree-only {result['max_abs_agree']:.3e}"
             if agree is not None else ""))
    return result


def run_group(group_name: str, refs: list, gots: list) -> dict:
    """Compare one full output set against another, flip-aware."""
    print(f"  [{group_name}]")
    # Support = sigmoid(logit) > 0.5  <=>  logit > 0; a flip means the two
    # logits straddle zero.
    agree = (refs[LOGIT_INDEX] > 0) == (gots[LOGIT_INDEX] > 0)
    flips = int((~agree).sum())
    print(f"    support-mask flips: {flips} / {agree.size} "
          f"({flips / agree.size:.2e})")
    group = {"support_flips": flips, "support_flip_fraction": flips / agree.size}
    for name, ref, got in zip(OUTPUT_NAMES, refs, gots):
        group[name] = compare(name, ref, got, agree if name in GATED else None)
    return group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/psi_v1/best.pt")
    parser.add_argument("--onnx", default="runs/psi_v1/psi.onnx")
    parser.add_argument("--data", default="dumps/prepared_full")
    parser.add_argument("--frames", type=int, default=3,
                        help="number of real held-out frames to check")
    parser.add_argument("--out", default="runs/psi_v1/evaluation/onnx_consistency.json")
    options = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch device: {device}")

    models, step = load_models(options.checkpoint, device)
    print(f"loaded checkpoint step {step}")

    import onnxruntime as ort
    available = ort.get_available_providers()
    print(f"ort providers available: {available}")
    sessions = {}
    if "CUDAExecutionProvider" in available:
        try:
            sessions["CUDAExecutionProvider"] = ort.InferenceSession(
                options.onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        except Exception as exc:
            print(f"CUDAExecutionProvider session failed ({exc}); CPU-only")
    sessions["CPUExecutionProvider"] = ort.InferenceSession(
        options.onnx, providers=["CPUExecutionProvider"])

    # ---- inputs ----
    inputs = []

    rng = np.random.default_rng(0)
    # Stress frame 1: 512x516 is a multiple of 4 (the network's stride-2
    # downsampling requires it) but NOT a multiple of 8, so the window
    # attention padding path is exercised. Contains exact 0/1 coverage and
    # an empty pixel (zero coverage + zero tangent).
    stress = np.zeros((2, 4, 512, 516), dtype=np.float32)
    stress[0, 0] = rng.uniform(0, 1, size=(1, 512, 516))
    stress[0, 1:] = rng.standard_normal(size=(3, 512, 516))
    stress[0, :, 200, 300] = 0.0                          # empty pixel
    inputs.append(("stress_padding_512x516", stress[:1], False))
    # Stress frame 2 (adversarial): near-zero tangents at the normalize-eps
    # boundary. The direction of a ~1e-8 vector is ill-conditioned under any
    # fp perturbation, so tangent diffs here are expected and excluded from
    # the verdict; real data has unit-length or exactly-zero tangents.
    stress[1, 0] = 1.0
    stress[1, 1:] = 1e-8 * rng.standard_normal(size=(3, 512, 516))
    inputs.append(("stress_zero_tangent (adversarial)", stress[1:2], True))

    dataset = ValDataset(options.data, split="train")
    for index in range(min(options.frames, len(dataset))):
        sample = dataset[index]
        x = sample["input"].numpy()[None]  # (1, 4, H, W)
        label = f"{sample['style']}_{Path(dataset.samples[index][1]).stem}"
        inputs.append((label, x, False))
    print(f"inputs: {[label for label, _, _ in inputs]}")

    report = {"checkpoint": options.checkpoint, "onnx": options.onnx,
              "step": step, "device": str(device), "per_input": {}}

    for label, x_np, adversarial in inputs:
        print(f"\n=== {label} {x_np.shape} ===")
        x_t = torch.from_numpy(x_np).to(device)

        # PyTorch references: original sdpa model (training/eval path) and the
        # matmul model actually exported.
        ref_sdpa = torch_outputs(models["sdpa"], x_t)
        ref_matmul = torch_outputs(models["matmul"], x_t)

        entry = {"shape": list(x_np.shape), "adversarial": adversarial}
        entry["torch_sdpa_vs_matmul"] = run_group("torch sdpa vs matmul",
                                                  ref_sdpa, ref_matmul)

        for provider, session in sessions.items():
            got, elapsed = ort_outputs(session, x_np)
            group_name = f"ort {provider} vs torch matmul ({elapsed * 1000:.0f} ms)"
            entry[f"ort_{provider}_vs_matmul"] = run_group(group_name, ref_matmul, got)

            if provider == "CUDAExecutionProvider":
                entry[f"ort_{provider}_vs_sdpa"] = run_group(
                    "ort CUDA vs torch sdpa (eval-path reference)", ref_sdpa, got)

        report["per_input"][label] = entry

    Path(options.out).parent.mkdir(parents=True, exist_ok=True)
    Path(options.out).write_text(json.dumps(report, indent=2))
    print(f"\nsaved report to {options.out}")

    # ---- verdict ----
    # Graph correctness is judged on the visually meaningful signals: the
    # support-gated coverage/tangent on pixels where both implementations
    # agree on the mask, and the raw reconstruction. The raw mask LOGIT is
    # not scored on absolute error (its downstream role is the sign only);
    # it is covered by the support-flip fraction. Adversarial near-zero-
    # tangent inputs are ill-conditioned by construction and excluded.
    worst = 0.0
    worst_flips = 0.0
    for label, entry in report["per_input"].items():
        if entry.get("adversarial"):
            continue
        for group in entry.values():
            if not isinstance(group, dict):
                continue
            worst_flips = max(worst_flips, group.get("support_flip_fraction", 0.0))
            for name, metrics in group.items():
                if isinstance(metrics, dict) and "max_abs" in metrics \
                        and name != "mask_logit":
                    value = metrics.get("max_abs_agree", metrics["max_abs"])
                    worst = max(worst, value)
    # fp32 kernels across implementations reassociate sums differently; the
    # observed noise on real data is ~1e-3 absolute on outputs in [-4, 4].
    tolerance = 5e-3
    status = "PASS" if worst < tolerance and worst_flips < 1e-4 else "FAIL"
    print(f"\nworst max_abs across all comparisons (mask-aware, "
          f"logits via flip count): {worst:.3e} (tolerance {tolerance:.0e}), "
          f"worst support flip fraction {worst_flips:.2e} (limit 1e-4) "
          f"-> {status}")


if __name__ == "__main__":
    main()
