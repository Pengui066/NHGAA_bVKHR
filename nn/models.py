"""Neural hair G-buffer reconstruction networks.

Implements the spatial reconstruction network Psi from "Real-Time
Neural Hair G-Buffer Anti-Aliasing" (Wu et al., SIGGRAPH Asia 2026,
arXiv:2605.17557), section 4.1:

- Input  X = [C, T] in R^{H x W x 4}   (coverage, world-space tangent)
- Output S = [C^s, T^s, L^s] in R^{H x W x 5}
    * 4 residual channels R for coverage+tangent, combined as
      Z = X + s * R + H  (s: learned scalar scale)
    * 1 support-mask logit L
- Dual-branch encoder (coverage | tangent), N=32 base channels,
  N -> 2N -> 4N at H -> H/2 -> H/4, residual blocks + stride-2 convs.
- Bottleneck: 8N channels at H/4, residual blocks + window attention
  (8x8 windows, half-window cyclic shift, 4 heads, GroupNorm pre-norm,
  1x1 conv QKV, FFN with GELU and expansion 2).
- Decoder: two bilinear-upsample stages with 1x1 channel halving and
  concatenated encoder skips.
- Hierarchical filtering branch: 1x1 convs predict per-channel sigmoid
  filtering coefficients kappa and a blending coefficient beta at the
  4x and 2x scales of the average-pooled input (Vogels-style kernel
  prediction, simplified to channel-wise coefficients).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Two 3x3 convolutions with batch norm, ReLU and an identity skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class WindowAttention(nn.Module):
    """Swin-style window attention over 8x8 windows with a half-window
    cyclic shift, GroupNorm pre-normalization and 1x1 conv projections."""

    def __init__(self, channels: int, heads: int = 4, window_size: int = 8,
                 shifted: bool = False):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.window_size = window_size
        self.shifted = shifted

        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=True)
        self.proj = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, h, w = x.shape
        ws = self.window_size

        # Pad so both dimensions are multiples of the window size.
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, hp, wp = x.shape

        residual = x
        x = self.norm(x)

        # Half-window cyclic shift.
        if self.shifted:
            x = torch.roll(x, shifts=(-ws // 2, -ws // 2), dims=(2, 3))

        qkv = self.qkv(x).reshape(b, 3, self.heads, c // self.heads, hp * wp)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (B, heads, ch, HW)

        # Window partition: (B, heads, n_windows, ws*ws, ch)
        q = q.reshape(b, self.heads, c // self.heads, hp // ws, ws, wp // ws, ws)
        k = k.reshape(b, self.heads, c // self.heads, hp // ws, ws, wp // ws, ws)
        v = v.reshape(b, self.heads, c // self.heads, hp // ws, ws, wp // ws, ws)
        q = q.permute(0, 1, 3, 5, 4, 6, 2).reshape(b, self.heads, -1, ws * ws, c // self.heads)
        k = k.permute(0, 1, 3, 5, 4, 6, 2).reshape(b, self.heads, -1, ws * ws, c // self.heads)
        v = v.permute(0, 1, 3, 5, 4, 6, 2).reshape(b, self.heads, -1, ws * ws, c // self.heads)

        out = F.scaled_dot_product_attention(q, k, v)  # flash/mem-efficient
        out = out.reshape(b, self.heads, hp // ws, wp // ws, ws, ws, c // self.heads)
        out = out.permute(0, 1, 6, 2, 4, 3, 5).reshape(b, c, hp, wp)

        if self.shifted:
            out = torch.roll(out, shifts=(ws // 2, ws // 2), dims=(2, 3))

        out = self.proj(out)
        return identity + out[:, :, :h, :w]


class AttentionBlock(nn.Module):
    """A regular and a shifted window-attention sub-layer, each followed
    by a feed-forward network (two 1x1 convs, GELU, expansion 2)."""

    def __init__(self, channels: int, heads: int = 4, window_size: int = 8):
        super().__init__()
        self.attn = WindowAttention(channels, heads, window_size, shifted=False)
        self.attn_shifted = WindowAttention(channels, heads, window_size, shifted=True)
        self.norm1 = nn.GroupNorm(min(32, channels), channels)
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.attn_shifted(x)
        x = x + self.ffn(self.norm2(x))
        return x


class FilteringBranch(nn.Module):
    """Hierarchical filtering: predicts per-channel sigmoid coefficients
    kappa and a blending coefficient beta at the 4x and 2x scales, applied
    to the average-pooled input buffers (eqs. 1-4 of the paper)."""

    def __init__(self, bottleneck_channels: int, decoder_channels: int,
                 input_channels: int = 4):
        super().__init__()
        self.input_channels = input_channels
        self.kappa_4x = nn.Conv2d(bottleneck_channels, input_channels + 1, 1)
        self.kappa_2x = nn.Conv2d(decoder_channels, input_channels + 1, 1)

    def forward(self, x: torch.Tensor, bottleneck_feat: torch.Tensor,
                decoder_feat: torch.Tensor) -> torch.Tensor:
        x_4x = F.avg_pool2d(x, 4)
        x_2x = F.avg_pool2d(x, 2)

        k4 = torch.sigmoid(self.kappa_4x(bottleneck_feat))
        kappa_4x, beta_4x = k4[:, :self.input_channels], k4[:, self.input_channels:]

        k2 = torch.sigmoid(self.kappa_2x(decoder_feat))
        kappa_2x, beta_2x = k2[:, :self.input_channels], k2[:, self.input_channels:]

        h_4x = kappa_4x * x_4x
        h_2x = kappa_2x * x_2x

        beta_4x_up = F.interpolate(beta_4x, size=h_2x.shape[-2:], mode="bilinear",
                                   align_corners=False)
        h_4x_up = F.interpolate(h_4x, size=h_2x.shape[-2:], mode="bilinear",
                                align_corners=False)
        h_2x_fused = (1.0 - beta_4x_up) * h_2x + beta_4x_up * h_4x_up

        beta_2x_up = F.interpolate(beta_2x, size=x.shape[-2:], mode="bilinear",
                                   align_corners=False)
        h_up = F.interpolate(h_2x_fused, size=x.shape[-2:], mode="bilinear",
                             align_corners=False)
        return h_up * beta_2x_up


# ---------------------------------------------------------------------------
# The spatial reconstruction network Psi
# ---------------------------------------------------------------------------

class SpatialNet(nn.Module):
    def __init__(self, base_channels: int = 32, input_channels: int = 4,
                 output_channels: int = 5):
        super().__init__()
        n = base_channels
        self.n = n

        # Dual-branch stems.
        self.stem_c = nn.Conv2d(1, n, 3, padding=1)
        self.stem_t = nn.Conv2d(3, n, 3, padding=1)

        # Encoder stage 1: residual blocks at H, stride-2 down to H/2 (N -> 2N).
        self.enc1_c = ResBlock(n)
        self.enc1_t = ResBlock(n)
        self.down1_c = nn.Conv2d(n, 2 * n, 3, stride=2, padding=1)
        self.down1_t = nn.Conv2d(n, 2 * n, 3, stride=2, padding=1)

        # Encoder stage 2: H/2, down to H/4 (2N -> 4N).
        self.enc2_c = ResBlock(2 * n)
        self.enc2_t = ResBlock(2 * n)
        self.down2_c = nn.Conv2d(2 * n, 4 * n, 3, stride=2, padding=1)
        self.down2_t = nn.Conv2d(2 * n, 4 * n, 3, stride=2, padding=1)

        # Bottleneck: 8N at H/4 with residual blocks + window attention.
        self.bottleneck = nn.Sequential(ResBlock(8 * n), ResBlock(8 * n))
        self.attention = AttentionBlock(8 * n, heads=4, window_size=8)

        # Decoder stage 1 (H/4 -> H/2): bilinear up + 1x1 halving, fused
        # with the concatenated branch skips (2N + 2N).
        self.up1 = nn.Conv2d(8 * n, 4 * n, 1)
        self.fuse1 = nn.Conv2d(8 * n, 4 * n, 3, padding=1)

        # Decoder stage 2 (H/2 -> H): 4N -> 2N, fused with N + N skips.
        self.up2 = nn.Conv2d(4 * n, 2 * n, 1)
        self.fuse2 = nn.Conv2d(4 * n, 2 * n, 3, padding=1)

        # Output head: 4 residual channels + 1 mask logit.
        self.head = nn.Conv2d(2 * n, output_channels, 3, padding=1)

        # Hierarchical filtering branch.
        self.filtering = FilteringBranch(bottleneck_channels=8 * n,
                                         decoder_channels=4 * n,
                                         input_channels=input_channels)

        # Learned scalar residual scale.
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> dict:
        """x: (B, 4, H, W) = [coverage, tangent(3)]. Returns a dict with
        'reconstruction' Z = X + s R + H (4ch) and 'mask_logit' (1ch)."""
        c = x[:, 0:1]
        t = x[:, 1:4]

        # Encoder.
        e1_c = self.enc1_c(self.stem_c(c))   # N   @ H
        e1_t = self.enc1_t(self.stem_t(t))   # N   @ H
        e2_c = self.down1_c(e1_c)            # 2N  @ H/2
        e2_t = self.down1_t(e1_t)
        e2_c = self.enc2_c(e2_c)
        e2_t = self.enc2_t(e2_t)
        e3_c = self.down2_c(e2_c)            # 4N  @ H/4
        e3_t = self.down2_t(e2_t)

        # Bottleneck.
        b = torch.cat([e3_c, e3_t], dim=1)   # 8N  @ H/4
        b = self.bottleneck(b)
        b = self.attention(b)

        # Decoder with concatenated skips.
        d1 = self.up1(F.interpolate(b, scale_factor=2, mode="bilinear",
                                    align_corners=False))
        skip1 = torch.cat([e2_c, e2_t], dim=1)              # 4N @ H/2
        d1 = self.fuse1(torch.cat([d1, skip1], dim=1))      # 4N @ H/2

        d2 = self.up2(F.interpolate(d1, scale_factor=2, mode="bilinear",
                                    align_corners=False))
        skip2 = torch.cat([e1_c, e1_t], dim=1)              # 2N @ H
        d2 = self.fuse2(torch.cat([d2, skip2], dim=1))      # 2N @ H

        head = self.head(d2)                                 # 5  @ H
        residual, mask_logit = head[:, :4], head[:, 4:5]

        h = self.filtering(x, b, d1)
        reconstruction = x + self.residual_scale * residual + h

        return {"reconstruction": reconstruction, "mask_logit": mask_logit,
                "residual": residual, "filtered": h}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
