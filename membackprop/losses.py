"""Row functions and the small-tensor losses built on their outputs."""

import torch
import torch.nn.functional as F
from torch import Tensor


def token_nll(logits: Tensor, labels: Tensor) -> Tensor:
    """Per-row negative log-likelihood, with label -100 contributing zero.

    Args:
        logits: Tensor of shape ``(n, vocab)``.
        labels: Integer tensor of shape ``(n,)``.

    Returns:
        Tensor of shape ``(n,)``.
    """
    return F.cross_entropy(logits.float(), labels, reduction="none")


def token_logp_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """Per-row selected log-probability and entropy.

    Args:
        logits: Tensor of shape ``(n, vocab)``.
        labels: Integer tensor of shape ``(n,)``.

    Returns:
        Tensor of shape ``(n, 2)`` holding ``(logp, entropy)``.
    """
    log_p = logits.float().log_softmax(-1)
    logp = log_p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return torch.stack((logp, -(log_p.exp() * log_p).sum(-1)), dim=-1)


def focal(logits: Tensor, labels: Tensor, gamma: float = 2.0) -> Tensor:
    """Per-row focal loss.

    Args:
        logits: Tensor of shape ``(n, vocab)``.
        labels: Integer tensor of shape ``(n,)``.
        gamma: Focusing exponent.

    Returns:
        Tensor of shape ``(n,)``.
    """
    nll = token_nll(logits, labels)
    return (1 - torch.exp(-nll)) ** gamma * nll


def distill_kl(logits: Tensor, teacher_logits: Tensor) -> Tensor:
    """Per-row forward KL from teacher to student.

    Args:
        logits: Student logits of shape ``(n, vocab)``.
        teacher_logits: Teacher logits of shape ``(n, vocab)``.

    Returns:
        Tensor of shape ``(n,)``.
    """
    student = logits.float().log_softmax(-1)
    teacher = teacher_logits.float().log_softmax(-1)
    return F.kl_div(student, teacher, reduction="none", log_target=True).sum(-1)


def mean_nll(nll: Tensor, labels: Tensor) -> Tensor:
    """Average per-token losses over labels that are not -100."""
    return nll.sum() / (labels != -100).sum()


def grpo_loss(
    logp: Tensor,
    old_logp: Tensor,
    ref_logp: Tensor,
    advantages: Tensor,
    mask: Tensor,
    beta: float = 0.1,
    epsilon: float = 0.2,
) -> Tensor:
    """Clipped GRPO objective with a per-token KL penalty.

    Args:
        logp: Policy log-probabilities of shape ``(batch, seq)``.
        old_logp: Sampling-policy log-probabilities, same shape.
        ref_logp: Reference-model log-probabilities, same shape.
        advantages: Group-normalized advantages of shape ``(batch,)``.
        mask: Completion mask of shape ``(batch, seq)``.
        beta: KL penalty coefficient.
        epsilon: Ratio clipping range.

    Returns:
        Scalar loss averaged per completion, then over the batch.
    """
    ratio = torch.exp(logp - old_logp)
    adv = advantages.unsqueeze(1)
    policy = -torch.min(ratio * adv, ratio.clamp(1 - epsilon, 1 + epsilon) * adv)
    log_ratio = ref_logp - logp
    per_token = policy + beta * (torch.exp(log_ratio) - log_ratio - 1)
    return ((per_token * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
