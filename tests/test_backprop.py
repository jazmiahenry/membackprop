"""Gradient-equivalence tests against ordinary autograd."""

import pytest
import torch
from torch import nn

from membackprop import chunked_head_apply
from membackprop.losses import (
    distill_kl,
    focal,
    grpo_loss,
    token_logp_entropy,
    token_nll,
)

ROWS, DIM, VOCAB = (3, 7), 16, 50


def compare(head, row_fn, *aligned, chunk_size=4, autocast=None, exact=False):
    """Assert outputs, hidden gradient and head gradients match plain autograd."""
    torch.manual_seed(0)
    base = torch.randn(*ROWS, DIM)
    results = []
    for chunked in (False, True):
        hidden = base.clone().requires_grad_()
        head.zero_grad()
        flat = [t.reshape(-1, *t.shape[2:]) for t in aligned]
        with torch.autocast("cpu", enabled=autocast is not None, dtype=autocast):
            if chunked:
                y = chunked_head_apply(
                    hidden, head, row_fn, *aligned, chunk_size=chunk_size
                )
            else:
                y = row_fn(head(hidden.reshape(-1, DIM)), *flat)
                y = y.reshape(*ROWS, *y.shape[1:])
        torch.manual_seed(1)
        # Non-uniform upstream gradient, so dropping dL/dy cannot go unnoticed.
        (y * torch.randn_like(y)).sum().backward()
        grads = [p.grad.clone() for p in head.parameters() if p.grad is not None]
        results.append([y.detach(), hidden.grad, *grads])
    assert len(results[0]) == len(results[1])
    tolerance = {"rtol": 0, "atol": 0} if exact else {"rtol": 1e-5, "atol": 1e-6}
    for expected, actual in zip(*results):
        torch.testing.assert_close(actual, expected, **tolerance)


@pytest.fixture
def labels():
    """Random labels including ignored positions."""
    torch.manual_seed(2)
    out = torch.randint(0, VOCAB, ROWS)
    out[0, :2] = -100
    return out


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 5, 21, 100])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("row_fn", [token_nll, focal])
def test_loss_functions(labels, row_fn, bias, chunk_size):
    """Cross entropy and focal loss match for every chunking."""
    compare(nn.Linear(DIM, VOCAB, bias=bias), row_fn, labels, chunk_size=chunk_size)


def test_logp_entropy(labels):
    """Multi-column row outputs match."""
    compare(nn.Linear(DIM, VOCAB), token_logp_entropy, labels.clamp(min=0))


def test_distillation():
    """A vocabulary-sized aligned tensor works."""
    compare(nn.Linear(DIM, VOCAB), distill_kl, torch.randn(*ROWS, VOCAB))


def test_arbitrary_function():
    """A function with no known closed-form shortcut works."""
    compare(nn.Linear(DIM, VOCAB), lambda z: torch.sin(z).logsumexp(-1) * z.std(-1))


def test_frozen_head(labels):
    """A frozen head still passes gradient to hidden states."""
    head = nn.Linear(DIM, VOCAB).requires_grad_(False)
    compare(head, token_nll, labels)


def test_lora_head(labels):
    """A frozen base with trainable low-rank factors works."""

    class LoraHead(nn.Module):
        """Minimal LoRA projection."""

        def __init__(self):
            """Create a frozen base and factors."""
            super().__init__()
            self.base = nn.Linear(DIM, VOCAB).requires_grad_(False)
            self.a = nn.Linear(DIM, 4, bias=False)
            self.b = nn.Linear(4, VOCAB, bias=False)

        def forward(self, x):
            """Apply base plus low-rank update."""
            return self.base(x) + self.b(self.a(x))

    compare(LoraHead(), token_nll, labels)


def test_frozen_hidden_trains_head(labels):
    """Head gradients are right when hidden states need none."""
    head, hidden = nn.Linear(DIM, VOCAB), torch.randn(*ROWS, DIM)
    chunked_head_apply(hidden, head, token_nll, labels, chunk_size=4).sum().backward()
    actual = head.weight.grad.clone()
    head.zero_grad()
    token_nll(head(hidden).reshape(-1, VOCAB), labels.reshape(-1)).sum().backward()
    torch.testing.assert_close(actual, head.weight.grad)


def test_autocast_bf16_single_chunk(labels):
    """Backward recomputation replays the forward autocast state."""
    compare(
        nn.Linear(DIM, VOCAB),
        token_nll,
        labels,
        chunk_size=100,
        autocast=torch.bfloat16,
        exact=True,
    )


def test_autocast_bf16_chunks(labels):
    """Outputs and hidden gradients stay bit-exact across chunks.

    Head gradients are excluded: each chunk's gradient rounds in bf16 before the
    float32 sum, so they legitimately differ from a single bf16 matmul.
    """
    compare(
        nn.Linear(DIM, VOCAB).requires_grad_(False),
        token_nll,
        labels,
        autocast=torch.bfloat16,
        exact=True,
    )


def test_autocast_cache_not_poisoned(labels):
    """A later projection in the same autocast region keeps its gradient."""
    head, hidden = nn.Linear(DIM, VOCAB), torch.randn(*ROWS, DIM)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        chunked_head_apply(hidden, head, token_nll, labels, chunk_size=4)
        head(hidden).float().sum().backward()
    assert head.weight.grad is not None


def test_dropout_replay():
    """Backward uses the dropout mask the forward pass drew."""
    head = nn.Sequential(nn.Dropout(0.5), nn.Linear(DIM, VOCAB, bias=False))
    hidden = torch.randn(40, DIM).requires_grad_()
    masks = []
    head[0].register_forward_hook(lambda m, i, o: masks.append(o.detach() == 0))
    y = chunked_head_apply(hidden, head, lambda z: z.sum(-1), chunk_size=8)
    torch.rand(5)  # Advance the generator between forward and backward.
    before = torch.get_rng_state()
    y.sum().backward()
    assert all(torch.equal(a, b) for a, b in zip(masks[:5], masks[5:]))
    assert torch.equal(before, torch.get_rng_state())


def test_tied_embedding(labels):
    """A weight shared by embedding and head sums both gradient paths."""
    tokens, grads = torch.randint(0, VOCAB, ROWS), []
    weight = torch.randn(VOCAB, DIM)
    for chunked in (False, True):
        embed, head = nn.Embedding(VOCAB, DIM), nn.Linear(DIM, VOCAB, bias=False)
        embed.weight = head.weight = nn.Parameter(weight.clone())
        hidden = embed(tokens).tanh()
        if chunked:
            y = chunked_head_apply(hidden, head, token_nll, labels, chunk_size=4)
        else:
            y = token_nll(head(hidden).reshape(-1, VOCAB), labels.reshape(-1))
        y.sum().backward()
        grads.append(embed.weight.grad)
    torch.testing.assert_close(*grads)


def test_grpo(labels):
    """The GRPO objective has identical loss and gradients."""
    labels, head = labels.clamp(min=0), nn.Linear(DIM, VOCAB)
    base, mask = torch.randn(*ROWS, DIM), torch.ones(ROWS)
    mask[1, 4:] = 0
    with torch.no_grad():
        old = token_logp_entropy(head(base).reshape(-1, VOCAB), labels.reshape(-1))
        old = old[:, 0].reshape(ROWS) + 0.3 * torch.randn(ROWS)
    ref, adv = old + 0.2 * torch.randn(ROWS), torch.tensor([1.0, -0.5, -0.5])
    results = []
    for chunk_size in (10_000, 4, None):
        hidden = base.clone().requires_grad_()
        head.zero_grad()
        if chunk_size:
            y = chunked_head_apply(
                hidden, head, token_logp_entropy, labels, chunk_size=chunk_size
            )
        else:
            y = token_logp_entropy(head(hidden).reshape(-1, VOCAB), labels.reshape(-1))
            y = y.reshape(*ROWS, 2)
        loss = grpo_loss(y[..., 0], old, ref, adv, mask) - 0.01 * y[..., 1].mean()
        loss.backward()
        results.append((loss.detach(), hidden.grad, head.weight.grad.clone()))
    for other in results[1:]:
        torch.testing.assert_close(other, results[0])


def test_gradcheck():
    """Numerical differentiation agrees in float64."""
    head = nn.Linear(4, 6).double()
    hidden = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)

    def fn(x, *_):
        """Map hidden states to per-row outputs."""
        return chunked_head_apply(
            x, head, lambda z: z.log_softmax(-1)[:, 0], chunk_size=2
        )

    assert torch.autograd.gradcheck(fn, (hidden, head.weight, head.bias))


def test_rejects_differentiable_aligned():
    """Aligned tensors whose gradient would be lost are refused."""
    teacher = torch.randn(*ROWS, VOCAB, requires_grad=True)
    with pytest.raises(ValueError):
        chunked_head_apply(
            torch.randn(*ROWS, DIM),
            nn.Linear(DIM, VOCAB),
            distill_kl,
            teacher,
            chunk_size=4,
        )
