"""Top-k KL distillation (Variant C) aligned with project tech doc §4.4."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def topk_kl_loss(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    *,
    temperature: float = 4.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    student_logits: [N, V] at supervised positions (flattened batch*seq)
    teacher_topk_ids: [N, K]
    teacher_topk_logits: [N, K] — raw logits (same T as student before softmax)
    Returns scalar mean KL over N positions.
    """
    T = temperature
    s_probs = F.softmax(student_logits / T, dim=-1)
    t_probs = F.softmax(teacher_topk_logits / T, dim=-1)

    s_topk = s_probs.gather(-1, teacher_topk_ids)
    s_topk = s_topk / (s_topk.sum(dim=-1, keepdim=True) + eps)

    log_s = torch.log(s_topk.clamp_min(eps))
    loss_kl = F.kl_div(log_s, t_probs, reduction="batchmean") * (T**2)
    return loss_kl
