"""Backprop through a large output head without materializing all logits.

Given ``y = f(head(X))`` where ``f`` maps each row of logits to a small output,
the forward pass evaluates chunks of rows without a graph and keeps only ``y``.
The backward pass rebuilds one chunk of logits at a time and lets autograd push
the upstream gradient ``dL/dy`` through ``f``. No derivative is hand-written.
"""

from collections.abc import Callable
from contextlib import contextmanager

import torch
from torch import Tensor, nn
from torch.autograd.function import once_differentiable
from torch.func import functional_call

RowFunction = Callable[..., Tensor]


def _capture_rng(device: torch.device) -> list[Tensor]:
    """Snapshot the CPU generator and, if applicable, the CUDA generator."""
    states = [torch.get_rng_state()]
    if device.type == "cuda":
        states.append(torch.cuda.get_rng_state(device))
    return states


@contextmanager
def _replayed_rng(device: torch.device, states: list[Tensor]):
    """Run a block under captured RNG states, then restore the current ones."""
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(states[0])
        if devices:
            torch.cuda.set_rng_state(states[1], device)
        yield


class MemoryEfficientLinear(torch.autograd.Function):
    """Apply ``row_fn(head(x), *aligned)`` in chunks, recomputing in backward.

    Use :func:`chunked_head_apply`, which extracts the parameters that must be
    passed as explicit inputs for autograd to route gradients to them.
    """

    @staticmethod
    def forward(ctx, hidden, head, row_fn, chunk_size, aligned, names, *params):
        """Evaluate every chunk without building a graph.

        Args:
            ctx: Autograd context.
            hidden: Tensor of shape ``(rows, dim)``.
            head: Module mapping ``(n, dim)`` to ``(n, vocab)``.
            row_fn: Function mapping ``(logits, *aligned)`` to ``(n, ...)``.
            chunk_size: Maximum rows of logits alive at once.
            aligned: Tuple of tensors with one entry per row.
            names: Names of the trainable head parameters.
            *params: Trainable head parameters, ordered like ``names``.

        Returns:
            Per-row outputs of shape ``(rows, ...)``.
        """
        device_type = hidden.device.type
        ctx.autocast = {
            "device_type": device_type,
            "enabled": torch.is_autocast_enabled(device_type),
            "dtype": torch.get_autocast_dtype(device_type),
            # A cast cached inside this no-grad region has no history. Reused by
            # the caller, it would silently drop that caller's weight gradient.
            "cache_enabled": False,
        }
        ctx.rng = _capture_rng(hidden.device)
        ctx.head, ctx.row_fn, ctx.names = head, row_fn, names
        ctx.chunk_size, ctx.n_aligned = chunk_size, len(aligned)
        ctx.save_for_backward(hidden, *aligned, *params)
        with torch.autocast(**ctx.autocast):
            outputs = [
                row_fn(head(chunk), *rest)
                for chunk, *rest in _split(chunk_size, hidden, *aligned)
            ]
        return torch.cat(outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        """Recompute each chunk with a graph and pull ``upstream`` through it."""
        hidden, *saved = ctx.saved_tensors
        aligned, params = saved[: ctx.n_aligned], saved[ctx.n_aligned :]
        needs_hidden = ctx.needs_input_grad[0]
        leaves = [p.detach().requires_grad_() for p in params]
        hidden_grad = torch.empty_like(hidden) if needs_hidden else None
        param_grads = [None] * len(leaves)

        start = 0
        with _replayed_rng(hidden.device, ctx.rng):
            for chunk, dy, *rest in _split(ctx.chunk_size, hidden, upstream, *aligned):
                chunk = chunk.detach().requires_grad_(needs_hidden)
                with torch.enable_grad(), torch.autocast(**ctx.autocast):
                    logits = functional_call(
                        ctx.head, dict(zip(ctx.names, leaves)), (chunk,)
                    )
                    output = ctx.row_fn(logits, *rest)
                del logits
                inputs = ([chunk] if needs_hidden else []) + leaves
                grads = list(torch.autograd.grad(output, inputs, dy))
                if needs_hidden:
                    hidden_grad[start : start + len(chunk)] = grads.pop(0)
                for i in reversed(range(len(leaves))):
                    # Popping drops the last reference, so a head-sized gradient
                    # is freed before the next chunk allocates another.
                    if param_grads[i] is None:
                        param_grads[i] = grads.pop()
                    else:
                        param_grads[i] += grads.pop()
                start += len(chunk)
        return (hidden_grad, *[None] * 5, *param_grads)


def _split(chunk_size: int, *tensors: Tensor):
    """Yield matching row chunks of every tensor."""
    return zip(*(t.split(chunk_size) for t in tensors))


def chunked_head_apply(
    hidden: Tensor,
    head: nn.Module,
    row_fn: RowFunction,
    *aligned: Tensor,
    chunk_size: int,
) -> Tensor:
    """Compute ``row_fn(head(hidden), *aligned)`` holding few logits in memory.

    ``row_fn`` must treat rows independently, because reductions such as a mean
    have to run on the returned per-row outputs. It must also be a pure function
    of its arguments: gradients do not reach tensors it captures by closure.

    Args:
        hidden: Hidden states of shape ``(..., dim)``.
        head: Output projection, for example ``nn.Linear(dim, vocab)``.
        row_fn: Maps ``(n, vocab)`` logits and aligned rows to ``(n, ...)``.
        *aligned: Non-differentiable tensors whose leading dimensions match
            ``hidden.shape[:-1]``, such as labels.
        chunk_size: Maximum rows of logits alive at once.

    Returns:
        Outputs of shape ``(*hidden.shape[:-1], ...)``.

    Raises:
        ValueError: If an aligned tensor requires grad, since it would be dropped.
    """
    if any(t.requires_grad for t in aligned):
        raise ValueError("Aligned tensors must not require grad; detach them first.")
    lead = hidden.shape[:-1]
    flat = [t.reshape(-1, *t.shape[len(lead) :]) for t in (hidden, *aligned)]
    trainable = {n: p for n, p in head.named_parameters() if p.requires_grad}
    out = MemoryEfficientLinear.apply(
        flat[0],
        head,
        row_fn,
        chunk_size,
        tuple(flat[1:]),
        tuple(trainable),
        *trainable.values(),
    )
    return out.reshape(*lead, *out.shape[1:])
