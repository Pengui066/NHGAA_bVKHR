# NHGAA_bVKHR — Neural Hair G-Buffer Anti-Aliasing, reproduced on VKHR

An **unofficial, independent reproduction** of:

> Chenghao Wu et al., *Real-Time Neural Hair G-Buffer Anti-Aliasing*,
> SIGGRAPH Asia 2026 · [arXiv:2605.17557](https://arxiv.org/abs/2605.17557)

built on top of [VKHR](https://github.com/CaffeineViking/vkhr), the MIT-licensed
hybrid hair renderer (near-distance rasterizer + far-distance raymarch LOD)
from EGSR 2019.

> This is **not** the official implementation. The authors' code is expected to
> be released separately — watch the paper's arXiv / project page for it.

## What this reproduction does

A rasterized strand hair G-buffer rendered at 1 sample per pixel is severely
aliased: coverage has holes, tangents break at silhouettes. The paper
reconstructs it with a small CNN (its "Ψ"), recovers missing position data
analytically, and feeds the reconstructed G-buffer into physically-based
deferred hair shading — a hair-specific alternative to general post-upscalers
such as DLSS/FSR.

This repository follows the same pipeline on VKHR:

1. **Dataset generation** — each frame is rendered twice by the extended
   renderer: a 720p · 1 spp input G-buffer and a 2880p · SSAA 4x (16 spp)
   reference G-buffer (coverage / tangent / motion / depth), dumped to disk
   with per-frame metadata.
2. **Ψ reconstruction network** — a compact CNN trained on the dumped
   G-buffers reconstructs coverage and tangents from the 1 spp input.
3. **Position reconstruction** — the paper's Section 5 five-step analytic
   method recovers positions where coverage was reconstructed but depth is
   missing.
4. **Deferred shading + evaluation** — the reconstructed G-buffer is shaded
   offline (Kajiya-Kay + approximated deep shadow maps) and scored against
   the reference rendering, in the spirit of the paper's Table 1.

## Status & preliminary results

| Stage | Status |
|---|---|
| Offline dual-resolution G-buffer dump mode (renderer side) | done |
| Dataset preparation (npz training data, hair-gated GT downsampling) | done |
| Ψ training (50k steps, 5 styles, 900 frames) | done |
| Analytic position reconstruction (paper §5) + validation | done |
| Offline deferred shading + image-domain metrics | done |
| Side-by-side comparison against the official code | pending (not yet released) |

Preliminary numbers, held-out frames, five hair styles (ponytail, bear, three
wigs):

| Metric | 1 spp input | Ψ reconstruction |
|---|---:|---:|
| G-buffer coverage PSNR (hair pixels, dB) | 11.9 | 21.2 (**+9.3**) |
| Shaded image PSNR (hair pixels, dB) | 26.6 | 32.5 (**+5.9**) |
| Shaded image SSIM | 0.84 | 0.93 |

Unofficial reproduction numbers, not directly comparable to the paper's tables
(different renderer, scenes and masks). Conventions: per-frame averaged PSNR
over hair-pixel masks; see `nn/evaluate.py` / `nn/metrics.py` for the exact
implementations. Tangent angular error (8.8° → 6.7°) is reported as a
diagnostic metric and is not part of the paper's tables.

## Repository layout

```
nn/         PyTorch pipeline: dataset, models, losses, training, evaluation,
            analytic position reconstruction (paper §5), shaded-image metrics
utils/      dump -> npz dataset preparation (GT downsampling)
src/        the VKHR renderer, extended with the offline dump mode
            (strand G-buffer pass, dual-resolution capture, camera scripting)
include/    C++ headers (as upstream, plus dump-mode additions)
share/      shaders and assets (as upstream)
foreign/    third-party dependencies (as upstream)
UPSTREAM_README.md  the original VKHR readme — renderer docs (controls,
            scene format, shading models, screenshots)
license.md  MIT license, inherited from VKHR and kept intact
```

## Building the renderer

Get the sources first — both flags matter: `--recursive` fetches the submodule
dependencies (`foreign/glm`, `imgui`, `stb`, `json`, `tinyobjloader`), and
[git-lfs](https://git-lfs.com/) is required because the prebuilt import
libraries and the model/style assets are stored in LFS (a clone without LFS
gets 130-byte pointer files instead of the real data):

```bash
git clone --recursive https://github.com/Pengui066/NHGAA_bVKHR.git
git lfs install && git lfs pull   # only needed if git-lfs wasn't active during the clone
```

Requires the [Vulkan SDK](https://vulkan.lunarg.com/) and
[premake5](https://premake.github.io/). Same as upstream VKHR:

```bash
premake5 vs2022         # Windows: generates a Visual Studio 2022 solution
premake5 gmake && make  # Linux (embree3 / glfw3 via your package manager)
```

**Windows runtime note:** the built `bin/vkhr.exe` expects `embree3.dll`,
`glfw3.dll`, `tbb.dll` and `tbbmalloc.dll` next to it. These are not tracked
on this branch — take them from the
[upstream VKHR repository](https://github.com/CaffeineViking/vkhr)'s `bin/`
directory, or from the official Embree 3.2.4 / GLFW 3.2.1 / TBB distributions.

Python side: PyTorch, NumPy, scikit-image, Pillow
(LPIPS optional: `pip install lpips`).

## Pipeline quick start

```bash
# 1) Dump dual G-buffers (720p 1 spp input + 2880p SSAA 4x reference)
vkhr <scene> --dump --dump-dir dumps/dataset_full \
      --dump-frames 160 --dump-ssaa 4 --camera-script <script>

# 2) Pack dumps into training npz
python utils/prepare_dataset.py dumps/dataset_full --out dumps/prepared_full

# 3) Train Ψ (designed to survive a shared desktop GPU: micro-batching +
#    gradient accumulation, out-of-process supervision, automatic resume)
python nn/train.py --data dumps/prepared_full --out runs/psi_v1

# 4) Evaluate the G-buffer reconstruction
python nn/evaluate.py --checkpoint runs/psi_v1/best.pt --data dumps/prepared_full

# 5) Deferred shading + image-domain metrics (Table 1 style)
python nn/metrics.py --dump dumps/dataset_full/ponytail_eval   # add --lpips if installed
```

## Reproduction notes

Honest findings from the reproduction so far — a full report is planned once
the official code is available for side-by-side comparison:

- **Position reconstruction, honest boundary.** The paper's §5 analytic
  chain propagation is implemented and validated in isolation (direction
  correct → error halves), but on our data (720p + GPAA geometry
  anti-aliasing, 1–2 px gaps) it does **not** beat a plain nearest-neighbor
  baseline (median error 0.68 vs. 0.35, win rate 38% in the test harness).
  We record this as a genuine reproduction finding and the first item to
  re-check against the official implementation. See
  `nn/test_position_reconstruction.py`.
- **Metric conventions.** PSNR is averaged per frame (both domains); hair
  pixel masks and the SSIM evaluation domain differ slightly between the
  G-buffer and shaded-image reports (details in the two metric scripts).
  These are documented as-is rather than silently matched to the paper.

## Credits & license

- The entire rendering substrate is **[VKHR](https://github.com/CaffeineViking/vkhr)**
  by Erik S. V. Jansson (CaffeineViking), EGSR 2019 — thank you for the clean
  and hackable base. Its MIT license is kept intact (`license.md`), and its
  original documentation (controls, scene format, shading models, screenshots)
  is preserved as [`UPSTREAM_README.md`](UPSTREAM_README.md).
- All method credit for the anti-aliasing approach belongs to the paper's
  authors. Any deviation or error in this reproduction is this repository's
  own.

MIT — see `license.md`.
