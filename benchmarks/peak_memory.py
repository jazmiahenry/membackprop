"""Peak memory of the output head under three strategies.

Uses the CUDA allocator statistics when available. On CPU it counts live tensor
storages instead, which shows tensor memory but not allocator behavior.
"""

import argparse
import time
import weakref

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from torch.utils.checkpoint import checkpoint

from membackprop import chunked_head_apply
from membackprop.losses import mean_nll, token_logp_entropy, token_nll


class StorageTracker(TorchDispatchMode):
    """Track peak bytes of CPU storages created while active."""

    def __init__(self):
        """Start with nothing allocated."""
        super().__init__()
        self.live, self.peak, self.seen = 0, 0, set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Run the op and record any storage it created."""
        out = func(*args, **(kwargs or {}))
        # Register inputs first: in-place ops and views return existing storages.
        inputs = {
            t.untyped_storage()._cdata
            for t in tree_leaves((args, kwargs))
            if isinstance(t, torch.Tensor)
        }
        for t in tree_leaves(out):
            if isinstance(t, torch.Tensor):
                self._record(t.untyped_storage(), inputs)
        return out

    def _record(self, storage, inputs):
        """Count a storage once and release it when it is freed."""
        key, size = storage._cdata, storage.nbytes()
        if key in self.seen or key in inputs:
            return
        self.seen.add(key)
        self.live += size
        self.peak = max(self.peak, self.live)
        weakref.finalize(storage, self._release, key, size)

    def _release(self, key, size):
        """Stop counting a freed storage."""
        self.seen.discard(key)
        self.live -= size


def run_full(hidden, head, row_fn, labels, chunk):
    """Materialize every logit."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    return row_fn(head(flat), labels.reshape(-1))


def run_checkpoint(hidden, head, row_fn, labels, chunk):
    """Loop over ``torch.utils.checkpoint``, the simplest alternative."""
    flat, lab = hidden.reshape(-1, hidden.shape[-1]), labels.reshape(-1)
    return torch.cat(
        [
            checkpoint(lambda x, y: row_fn(head(x), y), x, y, use_reentrant=False)
            for x, y in zip(flat.split(chunk), lab.split(chunk))
        ]
    )


def run_chunked(hidden, head, row_fn, labels, chunk):
    """Use the custom autograd function."""
    return chunked_head_apply(hidden, head, row_fn, labels, chunk_size=chunk).flatten(
        0, 1
    )


def measure(strategy, args, row_fn, reduce_fn, frozen):
    """Return peak GiB, seconds, loss and hidden gradient for one strategy."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    head = nn.Linear(args.dim, args.vocab, bias=False, device=device)
    head.requires_grad_(not frozen)
    hidden = torch.randn(args.batch, args.seq, args.dim, device=device)
    hidden.requires_grad_()
    labels = torch.randint(0, args.vocab, (args.batch, args.seq), device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    tracker, start = StorageTracker(), time.perf_counter()
    with tracker:
        out = strategy(hidden, head, row_fn, labels, args.chunk)
        loss = reduce_fn(out, labels.reshape(-1))
        del out
        loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
    else:
        peak = tracker.peak
    seconds = time.perf_counter() - start
    return peak / 2**30, seconds, loss.item(), hidden.grad


def main():
    """Print a table comparing the strategies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--vocab", type=int, default=128_256)
    parser.add_argument("--chunk", type=int, default=256)
    args = parser.parse_args()
    tasks = {
        "ce": (token_nll, mean_nll),
        "grpo-logp+entropy": (token_logp_entropy, lambda y, _: y.mean()),
    }
    print("task,head,strategy,peak_gib,seconds,loss,grad_max_abs_diff")
    for task, fns in tasks.items():
        for frozen in (True, False):
            reference = None
            for fn in (run_full, run_checkpoint, run_chunked):
                peak, seconds, loss, grad = measure(fn, args, *fns, frozen)
                reference = grad if reference is None else reference
                diff = (grad - reference).abs().max().item()
                head = "frozen" if frozen else "trained"
                print(
                    f"{task},{head},{fn.__name__},{peak:.3f},{seconds:.2f},"
                    f"{loss:.6f},{diff:.2e}"
                )


if __name__ == "__main__":
    main()
