"""Listwise losses targeted at NDCG@5."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def lambda_ndcg_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    top_k: int = 5,
    sigma: float = 1.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """LambdaLoss/LambdaRank surrogate using delta NDCG@``top_k`` weights.

    All pair construction is done independently inside each packed search
    boundary, so no gradient can couple hotels from different searches.
    """

    losses: list[torch.Tensor] = []
    boundaries = cu_seqlens.detach().cpu().tolist()
    device = scores.device

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group_scores = scores[start:end]
        group_labels = labels[start:end]
        group_weights = None if sample_weights is None else sample_weights[start:end].to(dtype=group_scores.dtype)
        n = group_scores.numel()
        if n <= 1 or torch.max(group_labels) <= 0:
            continue

        gains = torch.pow(2.0, group_labels) - 1.0
        ideal_gains = torch.sort(gains, descending=True).values[:top_k]
        ideal_positions = torch.arange(ideal_gains.numel(), device=device, dtype=group_scores.dtype)
        ideal_discounts = 1.0 / torch.log2(ideal_positions + 2.0)
        idcg = torch.sum(ideal_gains * ideal_discounts)
        if idcg <= eps:
            continue

        predicted_order = torch.argsort(group_scores.detach(), descending=True)
        rank_positions = torch.empty(n, device=device, dtype=torch.long)
        rank_positions[predicted_order] = torch.arange(n, device=device)
        discounts = 1.0 / torch.log2(rank_positions.to(group_scores.dtype) + 2.0)
        discounts = torch.where(rank_positions < top_k, discounts, torch.zeros_like(discounts))

        label_diff = group_labels[:, None] - group_labels[None, :]
        positive_pairs = label_diff > 0
        if not positive_pairs.any():
            continue

        score_diff = group_scores[:, None] - group_scores[None, :]
        gain_delta = torch.abs(gains[:, None] - gains[None, :])
        discount_delta = torch.abs(discounts[:, None] - discounts[None, :])
        delta_ndcg = (gain_delta * discount_delta / idcg).detach()

        pair_loss = F.softplus(-sigma * score_diff) * delta_ndcg
        if group_weights is not None:
            pair_weights = group_weights[:, None].expand_as(pair_loss).clamp_min(0.1)
            weighted = pair_loss[positive_pairs] * pair_weights[positive_pairs]
            losses.append(weighted.sum() / pair_weights[positive_pairs].sum().clamp_min(1.0))
        else:
            losses.append(pair_loss[positive_pairs].sum() / positive_pairs.sum().clamp_min(1))

    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


class LambdaNDCGLoss(nn.Module):
    def __init__(self, top_k: int = 5, sigma: float = 1.0) -> None:
        super().__init__()
        self.top_k = top_k
        self.sigma = sigma

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        cu_seqlens: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return lambda_ndcg_loss(
            scores,
            labels,
            cu_seqlens,
            sample_weights=sample_weights,
            top_k=self.top_k,
            sigma=self.sigma,
        )


def pairwise_logistic_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    sigma: float = 1.0,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    boundaries = cu_seqlens.detach().cpu().tolist()

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group_scores = scores[start:end]
        group_labels = labels[start:end]
        n = group_scores.numel()
        if n <= 1 or torch.max(group_labels) <= 0:
            continue
        positive_pairs = (group_labels[:, None] - group_labels[None, :]) > 0
        if not positive_pairs.any():
            continue
        score_diff = group_scores[:, None] - group_scores[None, :]
        pair_loss = F.softplus(-sigma * score_diff)
        if sample_weights is not None:
            group_weights = sample_weights[start:end].to(dtype=group_scores.dtype).clamp_min(0.1)
            pair_weights = group_weights[:, None].expand_as(pair_loss)
            weighted = pair_loss[positive_pairs] * pair_weights[positive_pairs]
            losses.append(weighted.sum() / pair_weights[positive_pairs].sum().clamp_min(1.0))
        else:
            losses.append(pair_loss[positive_pairs].mean())

    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def listnet_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    temperature: float = 1.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    boundaries = cu_seqlens.detach().cpu().tolist()

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group_scores = scores[start:end]
        group_labels = labels[start:end]
        if group_scores.numel() <= 1 or torch.max(group_labels) <= 0:
            continue

        gains = torch.pow(2.0, group_labels.to(dtype=group_scores.dtype)) - 1.0
        if sample_weights is not None:
            gains = gains * sample_weights[start:end].to(dtype=group_scores.dtype).clamp_min(0.1)
        target = gains / gains.sum().clamp_min(eps)
        log_probs = F.log_softmax(group_scores / max(temperature, eps), dim=0)
        losses.append(-(target.detach() * log_probs).sum())

    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()
