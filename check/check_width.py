import numpy as np
from pathlib import Path

root = Path("dumps/dataset_full")
print(root)
for style_dir in sorted(root.glob("*_eval")):
    ins = sorted((style_dir / "input").glob("*_coverage.f16"))
    gts = sorted((style_dir / "gt").glob("*_coverage.f16"))
    if not ins or not gts:
        print(f"{style_dir.name:20s} [未找到 coverage 文件]"); continue
    ratios = []
    for f_in, f_gt in list(zip(ins, gts))[:3]:
        a = np.fromfile(f_in,  dtype="<f2").astype(np.float64)
        b = np.fromfile(f_gt, dtype="<f2").astype(np.float64)
        ssaa2 = b.size // a.size      # GT 像素数 = ssaa² × 输入像素数
        ratios.append(a.sum() / max(b.sum() / ssaa2, 1e-9))
    print(f"{style_dir.name:20s} ssaa²={ssaa2:<3d} 质量比 in/GT = {np.mean(ratios):.2f}  (帧数 {len(ins)})")
