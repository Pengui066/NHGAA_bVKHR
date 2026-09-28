"""Losses for hair G-buffer reconstruction.

The paper's loss definitions live in unpublished supplementary material,
so these are our documented reproduction choices:

- coverage: L1 between predicted and GT coverage (everywhere).
- tangent: L1 plus an angular term 1 - |cos(t_pred, t_gt)|, evaluated
  only on GT hair pixels and averaged over them.
- support mask: binary cross-entropy with logits (everywhere).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HairGBufferLoss(nn.Module):
    def __init__(self, w_coverage: float = 1.0, w_tangent: float = 1.0,
                 w_mask: float = 0.5, w_angular: float = 1.0):
        super().__init__()
        self.w_coverage = w_coverage
        self.w_tangent = w_tangent
        self.w_mask = w_mask
        self.w_angular = w_angular

    def forward(self, prediction: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        pred = prediction["reconstruction"]
        pred_coverage = pred[:, 0:1]
        pred_tangent = pred[:, 1:4]
        pred_logit = prediction["mask_logit"]

        gt_coverage = batch["gt_coverage"]
        gt_tangent = batch["gt_tangent"]
        gt_mask = batch["gt_mask"]

        loss_coverage = F.l1_loss(pred_coverage, gt_coverage)

        hair = (gt_mask > 0.5).float()
        hair_count = hair.sum().clamp(min=1.0)

        loss_tangent_l1 = ((pred_tangent - gt_tangent).abs().sum(dim=1, keepdim=True) * hair).sum() / (hair_count * 3)

        pred_tangent_hat = F.normalize(pred_tangent, dim=1, eps=1e-6)
        cos = (pred_tangent_hat * gt_tangent).sum(dim=1, keepdim=True).abs()
        loss_tangent_angular = ((1.0 - cos) * hair).sum() / hair_count

        loss_mask = F.binary_cross_entropy_with_logits(pred_logit, gt_mask)

        total = (self.w_coverage * loss_coverage
                 + self.w_tangent * (loss_tangent_l1 + self.w_angular * loss_tangent_angular)
                 + self.w_mask * loss_mask)

        terms = {
            "loss": total.detach(),
            "coverage": loss_coverage.detach(),
            "tangent_l1": loss_tangent_l1.detach(),
            "tangent_angular": loss_tangent_angular.detach(),
            "mask": loss_mask.detach(),
        }
        return total, terms
