"""CE / top-k KL for Qwen3-VL distill（Variant A/B/C/BC）。"""

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


def ce_shift_supervised_mean(
    shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


def ce_shift_trace_answer(
    shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    *,
    prompt_len: int,
    trace_tok_len: int,
    answer_tok_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (L_CE(trace), L_CE(answer))；trace 为空时第一项为与图同设备的 0。"""
    device = shift_logits.device
    S1 = shift_labels.size(1)
    j = torch.arange(S1, device=device)
    P = prompt_len
    valid = shift_labels[0] != -100
    mask_tr = (j + 1 >= P) & (j + 1 < P + trace_tok_len) & valid
    mask_ans = (
        (j + 1 >= P + trace_tok_len)
        & (j + 1 < P + trace_tok_len + answer_tok_len)
        & valid
    )

    ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)

    def masked_mean(mask: torch.Tensor) -> torch.Tensor:
        if not mask.any():
            return ce.sum() * 0.0
        return ce[0, mask].mean()

    return masked_mean(mask_tr), masked_mean(mask_ans)
