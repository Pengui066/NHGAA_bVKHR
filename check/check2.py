import numpy as np
from pathlib import Path

SSAA, HIN, WIN = 4, 720, 1280
HGT, WGT = HIN*SSAA, WIN*SSAA

def box(b, transpose=False):                      # GT 4x4 块平均 → 720p
    h, w = (WIN, HIN) if transpose else (HIN, WIN)
    img = b.reshape(WGT, HGT) if transpose else b.reshape(HGT, WGT)
    return img.reshape(h, SSAA, w, SSAA).mean(axis=(1, 3))

styles = ["ponytail", "bear", "wcurly"]
NB = 8
for style in styles:
    d = Path(f"dumps/dataset_full/{style}_eval")
    ins = sorted((d/"input").glob("*_coverage.f16"))
    gts = sorted((d/"gt").glob("*_coverage.f16"))
    si = np.zeros(NB); sg = np.zeros(NB); cnt = np.zeros(NB); rat = []
    transpose = False
    for f_in, f_gt in list(zip(ins, gts))[:6]:
        a = np.fromfile(f_in, dtype="<f2").astype(np.float32).ravel()
        b = np.fromfile(f_gt, dtype="<f2").astype(np.float32).ravel()
        g = box(b, transpose)
        if rat == []:                     # 第一帧：自动判定行列方向并记住
            t = box(b, transpose=True)
            transpose = abs(np.corrcoef(a, t.ravel())[0,1]) > abs(np.corrcoef(a, g.ravel())[0,1])
            if transpose:
                g = t
        rat.append(float(a.sum() / max(g.sum(), 1e-9)))
        gf = g.ravel()
        mf = gf > 0.02                    # 排除纯背景像素
        idx = np.clip((gf[mf]*NB).astype(int), 0, NB-1)
        si += np.bincount(idx, weights=a[mf], minlength=NB)
        sg += np.bincount(idx, weights=gf[mf], minlength=NB)
        cnt += np.bincount(idx, minlength=NB)
    per_bin = (si/np.maximum(cnt,1)) / (sg/np.maximum(cnt,1))
    print(f"{style}: 总比={np.mean(rat):.2f}")
    print("  按答案浓度分箱(稀疏→拥挤):", " ".join(f"{x:.2f}" for x in per_bin))
