import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.losses import topk_kl_loss


def test_topk_kl_finite_random():
    torch.manual_seed(0)
    V, K, N = 1000, 20, 8
    student_logits = torch.randn(N, V)
    topk_ids = torch.topk(student_logits, K, dim=-1).indices
    teacher_logits = student_logits.gather(-1, topk_ids).detach()
    loss = topk_kl_loss(student_logits, topk_ids, teacher_logits, temperature=4.0)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_topk_kl_near_zero_when_teacher_logits_match_student_on_topk():
    torch.manual_seed(1)
    N, V, K = 2, 256, 32
    s = torch.randn(N, V)
    idx = torch.topk(s, K, dim=-1).indices
    tl = s.gather(-1, idx).detach()
    loss = topk_kl_loss(s, idx, tl, temperature=4.0)
    assert loss.item() < 1e-4
