# ONNX Runtime (vendored)

Microsoft ONNX Runtime for the neural reconstruction path (S3 of the
NeuralHairGAA reproduction), version 1.20.x matching the Python side.

Committed pieces:

- `include/` - ONNX Runtime C/C++ headers, from the official
  `onnxruntime-win-x64-1.20.0.zip` release (the header set and import
  library are provider-independent; the CPU zip is a 65 MB download
  instead of the 338 MB GPU one).
- `lib/onnxruntime.lib` - import library from the same zip. The C++ API
  resolves every call through the single `OrtGetApiBase` export, so the
  import lib is tiny and works for any provider build.
- `LICENSE` / `ThirdPartyNotices.txt` - MIT license, same as the rest of
  the vendored dependencies.

Local-only pieces (`Gpu/`, gitignored - the premake post-build step copies
them next to vkhr.exe when present):

- `onnxruntime.dll`, `onnxruntime_providers_shared.dll`,
  `onnxruntime_providers_cuda.dll` - from the pip wheel
  `onnxruntime-gpu==1.20.2` (`site-packages/onnxruntime/capi/`).
- The CUDA execution provider's dependency chain (hard imports, verified
  with dumpbin): `cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`,
  `cufft64_11.dll`, `cudnn64_9.dll` and the split cuDNN family
  (`cudnn_*64_9.dll`) - from torch 2.7.0+cu126's `torch/lib/`, the exact
  chain the Python-side CUDA EP already runs on.
  If cuDNN fails to load in the field, a fresh copy of the official
  `onnxruntime-win-x64-gpu-1.20.0.zip` `Gpu/` folder can replace this set.

On a fresh clone, assemble `Gpu/` as above (or download the GPU zip and
extract its `Gpu/` folder here), then rebuild - the post-build copy step
picks the DLLs up automatically.
