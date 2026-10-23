# Ψ 网络 ONNX 前后处理归属表

> 本文件是实时集成（S3，C++ 侧）关于"图内/图外各做什么"的唯一依据。
> 导出物：`runs/psi_v1/psi.onnx`（由 `nn/export_onnx.py` 从 `runs/psi_v1/best.pt`
> step 33000 导出，opset 18）。导出所用注意力为显式 matmul 形式
> （`attn_impl="matmul"`），与训练/评估路径的 `scaled_dot_product_attention`
> 数学等价（已验证，见 `runs/psi_v1/evaluation/onnx_consistency.json`，PASS）。

## 图签名

```
输入  input: float32 [B, 4, H, W]  NCHW
            ch0 = coverage 覆盖度（原始值，原样送入网络；clamp 不在输入侧，
                  只作用于图内输出侧的重构结果，见下方归属表"图内"行）
            ch1..3 = tangent 世界空间切向（原始值，图内会归一化）
输出  0 coverage:      float32 [B, 1, H, W]  最终门控后覆盖度
输出  1 tangent:       float32 [B, 3, H, W]  最终门控后切向
输出  2 reconstruction: float32 [B, 4, H, W]  原始重构 Z = X + sR + H（调试/下游）
输出  3 mask_logit:    float32 [B, 1, H, W]  原始 support 掩码 logit L（Γ 时序网用）

batch / H / W 均动态；**H、W 必须是 4 的倍数**（stride-2 下采样与双线性
上采样只在该约束下尺寸对齐；512²、720×1280 满足。窗口注意力在图内自行
padding 到 8 的倍数，调用方无需处理）。
```

## 归属表

### 输入侧（图外，调用方负责）

| 步骤 | 归属 | 具体操作 | 依据 |
|---|---|---|---|
| 读文件 | 图外 | 加载 dump 的 npz（`input/coverage`、`input/tangent`） | dataset.py `_load` |
| 数值清洗 | 图外 | `nan_to_num`（NaN/±inf → 0） | dataset.py `_load` |
| 通道拼接 | 图外 | [C] + [T] → (H, W, 4) | dataset.py `__getitem__` |
| 布局转换 | 图外 | HWC → NCHW (4, H, W)，float32，连续内存，自行加 batch 维 | dataset.py `to_tensor` |
| 数值缩放 | 图外 | **不做任何归一化/缩放**（coverage 已是物理值，tangent 世界系单位向量） | 训练与推理一致 |

### 图内（psi.onnx 已包含，调用方不再重复）

| 步骤 | 归属 | 具体操作 |
|---|---|---|
| Ψ 前向 | 图内 | SpatialNet 完整前向（双分支编码器 → 瓶颈+窗口注意力 → 解码器 → 滤波分支） |
| coverage clamp | **图内** | `clamp(reconstruction[:, 0:1], 0, 1)` |
| tangent 归一化 | **图内** | `normalize(reconstruction[:, 1:4], dim=1, eps=1e-6)` |
| support 判定 | **图内** | `(sigmoid(mask_logit) > 0.5).float()` 硬门控 |
| 门控乘法 | **图内** | `coverage *= support`、`tangent *= support` |

> 图内后处理与评估管线 `evaluate.py::postprocess` 逐行一致——导出即"评估语义"。

### 输出侧（图外，调用方负责）

| 步骤 | 归属 | 具体操作 |
|---|---|---|
| 直接消费 | 图外 | 输出 0/1 已是**最终值**：coverage ∈ [0,1]；tangent 为单位向量或零（被门控的像素）。**不得再 clamp/归一化** |
| support 复算 | 图外（可选） | 需要显式掩码时用输出 3：`support = logit > 0`（⇔ sigmoid > 0.5） |
| 位置/深度 | **不在图中** | position reconstruction 是独立解析阶段（`nn/position_reconstruction.py`），图只负责 C、T、L |
| Γ 时序网输入 | 图外（后续） | 输出 2/3 正是为了 Γ 需要 S(5) = [C, T, L] 而保留的原始通道 |

## C++ 侧备忘

- 输入须满足：NCHW、float32、连续、H/W 为 4 的倍数（分块尺寸若启用亦须满足）；
  **数值清洗（`nan_to_num`：NaN/±inf → 0）是调用方责任**——零长度切向会产出 NaN
  （bear 数据有前科），必须在上传前清洗，图内不做。
- 输出 0/1 直接写入 G-buffer 的 coverage/tangent 通道，交给现有延迟着色；
  不要再经过 evaluate/postprocess 一类的逻辑。
- 精度 fp32；跨实现数值一致性已验证（CPU EP / CUDA EP vs PyTorch，
  `onnx_consistency.json`：掩码一致像素最大绝对差 ≤ 1.1e-3，support 翻转 ≤ 3.3e-6）。
- 路径与导出方式：`python nn/export_onnx.py --dynamic`（换 checkpoint 后重导同图）。
