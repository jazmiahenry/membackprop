# membackprop

Memory-efficient backprop through large LLM output heads, plus Llama 3.x from
scratch in PyTorch. This answers challenge E of the (now closed) Unsloth puzzles.

## Idea

For `y = f(head(X))`, where `f` maps each row of logits to a small output:

1. **Forward:** evaluate chunks of rows without a graph. Keep only `y`.
2. **Backward:** rebuild one chunk of logits, then let autograd push the upstream
   gradient `dL/dy` through `f` and the head. Free it and continue.

No derivative is hand-written, so any row-wise `f` works: cross entropy, focal,
distillation KL, GRPO log-probabilities with entropy.

```python
y = chunked_head_apply(hidden, lm_head, token_logp_entropy, labels, chunk_size=256)
loss = grpo_loss(y[..., 0], old_logp, ref_logp, advantages, mask)
```

## Results

Measured on an Apple M1 Max **CPU** in float32. Peak counts live tensor storages,
not CUDA allocator memory. No CUDA result is claimed here.

Peak GiB, 4 x 2048 tokens, dim 1024, vocab 128256, 256 rows per chunk:

| task | head | all logits | `checkpoint` loop | this repo |
|---|---|---|---|---|
| cross entropy | frozen | 11.742 | 0.397 | 0.398 |
| cross entropy | trained | 11.742 | 1.499 | **1.133** |
| GRPO logp + entropy | frozen | 19.570 | 0.642 | 0.643 |
| GRPO logp + entropy | trained | 19.570 | 1.499 | **1.133** |

Peak GiB, 2 x 1024 tokens, dim 2048:

| task | head | all logits | `checkpoint` loop | this repo |
|---|---|---|---|---|
| cross entropy | frozen | 2.936 | 0.381 | 0.383 |
| cross entropy | trained | 2.936 | 2.951 | **2.097** |
| GRPO logp + entropy | trained | 4.893 | 2.951 | **2.097** |

- A loop over `torch.utils.checkpoint` is just as good when the head is frozen.
  The gain here is with a trained head, where a head-sized gradient is freed
  between chunks.
- With a trained head the floor is two head-sized buffers, so the saving on a
  small batch is under 50% (2.936 to 2.097 for cross entropy).
- Hidden-state gradients were bit-identical across strategies in these runs.

**Llama 3.2 1B:** 20 AdamW steps (last 2 layers, norm and tied embedding trained, float32)
with full and chunked logits. Maximum loss difference: 1.4e-6
(`results/llama1b_losses.csv`). Logits match transformers within 7.5e-5.

## Limits

- `f` must treat rows independently and must not close over tensors that need grad.
- Under bf16 autocast, outputs and hidden gradients are bit-exact, but a trained
  head's gradient rounds per chunk, so it differs slightly from one bf16 matmul.
- First-order gradients only.

## Reproduce

```bash
pip install -e '.[dev]'
pytest
python benchmarks/peak_memory.py                # uses CUDA statistics if available
python benchmarks/peak_memory.py --batch 4 --seq 2048 --dim 1024
python benchmarks/train_llama.py --model <llama-3.2-1b dir> --text <file> --out out.csv
```

Built with AI assistance (Claude Code). Apache-2.0.
