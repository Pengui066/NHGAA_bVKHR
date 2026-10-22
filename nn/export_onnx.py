"""Export the trained Psi network to ONNX for real-time integration.

The exported graph is self-contained: it wraps the inference-time
postprocessing from evaluate.py (coverage clamp, tangent renormalization,
support-mask gating) so the runtime side only has to feed a 4-channel
G-buffer [C, T] and read back 4 channels [C, T]. The raw reconstruction
and the support-mask logit are exposed as extra outputs because the
temporal network Gamma consumes them (paper section 4.2).

The window attention is exported with the explicit matmul formulation
(attn_impl="matmul"): ONNX cannot trace the fused
scaled_dot_product_attention op, and the explicit form is mathematically
identical to its math backend.

Input height and width must be multiples of 4: the stride-2 conv
downsampling and the decoder's bilinear upsample only agree at those
sizes (the window attention pads internally to multiples of 8).

Usage:
    python nn/export_onnx.py --checkpoint runs/psi_v1/best.pt --out runs/psi_v1/psi.onnx
"""
from __future__ import annotations

import os

# The conda MKL numpy and pip torch ship different libiomp5md.dll copies;
# without this guard the process aborts at OpenMP init (OMP Error #15).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import SpatialNet


class PsiInference(nn.Module):
    """Self-contained inference wrapper around the trained Psi network.

    Input  x: (B, 4, H, W) = [coverage, world-space tangent]
    Output coverage:    (B, 1, H, W) clamped to [0, 1], gated by support
    Output tangent:     (B, 3, H, W) renormalized, gated by support
    Output reconstruction: raw network reconstruction Z (B, 4, H, W)
    Output mask_logit:  raw support-mask logit L (B, 1, H, W)
    """

    def __init__(self, net: SpatialNet):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor):
        prediction = self.net(x)
        reconstruction = prediction["reconstruction"]
        mask_logit = prediction["mask_logit"]

        coverage = reconstruction[:, 0:1].clamp(0.0, 1.0)
        tangent = F.normalize(reconstruction[:, 1:4], dim=1, eps=1e-6)
        support = (torch.sigmoid(mask_logit) > 0.5).float()
        return coverage * support, tangent * support, reconstruction, mask_logit


def export(options: argparse.Namespace) -> None:
    device = torch.device("cpu")
    net = SpatialNet(base_channels=32, attn_impl="matmul")
    state = torch.load(options.checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(state["model"])
    print(f"loaded {options.checkpoint} (step {state.get('step', '?')})")

    wrapper = PsiInference(net).to(device)
    # eval() must be applied to the WRAPPER, not just the inner net:
    # torch.onnx.export restores the wrapper's training flag after exporting,
    # and a wrapper that was never set to eval gets switched to train mode,
    # which silently runs its BatchNorms on batch statistics afterwards.
    wrapper.eval()
    example = torch.zeros((1, 4, options.height, options.width), device=device)

    dynamic_axes = {}
    if options.dynamic:
        shared = {0: "batch", 2: "height", 3: "width"}
        dynamic_axes = {"input": shared, "coverage": shared, "tangent": shared,
                        "reconstruction": shared, "mask_logit": shared}

    out_path = Path(options.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example,
            out_path,
            input_names=["input"],
            output_names=["coverage", "tangent", "reconstruction", "mask_logit"],
            dynamic_axes=dynamic_axes,
            opset_version=18,
            do_constant_folding=True,
        )
    print(f"exported {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Smoke check: run the graph through ONNX Runtime on CPU and compare it
    # against the PyTorch wrapper on a 256x256 crop of a real dumped frame.
    # (Random noise is a bad probe: the support-mask logits cluster near 0
    # and flip at the >0.5 threshold on fp noise, which reads as huge diffs
    # in the support-gated outputs even though the graph is fine.)
    import onnx
    onnx.checker.check_model(str(out_path))
    import onnxruntime as ort

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    print(f"ort providers: {session.get_providers()}")

    try:
        from dataset import ValDataset
        frame = ValDataset("dumps/prepared_full", split="train")[0]["input"]
        probe = frame.numpy()[:, 100:356, 400:656][None]  # 256x256 crop
        label = "real frame crop"
    except Exception as exc:
        print(f"no prepared data for the smoke check ({exc}); using random probe")
        rng = np.random.default_rng(0)
        probe = np.zeros((1, 4, 256, 256), dtype=np.float32)
        probe[:, 0] = rng.uniform(0, 1, size=(1, 1, 256, 256))
        probe[:, 1:] = rng.standard_normal(size=(1, 3, 256, 256))
        label = "random probe"

    with torch.no_grad():
        ref = wrapper(torch.from_numpy(probe))
    start = time.perf_counter()
    ort_out = session.run(None, {"input": np.ascontiguousarray(probe)})
    elapsed = time.perf_counter() - start

    names = ["coverage", "tangent", "reconstruction", "mask_logit"]
    for name, ref_t in zip(names, ref):
        diff = np.abs(np.asarray(ort_out[names.index(name)]) - ref_t.numpy())
        print(f"{name:14s} max|ort - torch| = {diff.max():.3e}  "
              f"({elapsed * 1000:.0f} ms for 256x256 {label})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/psi_v1/best.pt")
    parser.add_argument("--out", default="runs/psi_v1/psi.onnx")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--dynamic", action="store_true",
                        help="export with dynamic batch/height/width axes "
                             "(needed to run tiles of arbitrary size)")
    options = parser.parse_args()
    export(options)


if __name__ == "__main__":
    main()
