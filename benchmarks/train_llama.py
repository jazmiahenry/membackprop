"""Train Llama 3.2 1B with full and chunked logits and compare the losses."""

import argparse
import csv

import torch

from membackprop import chunked_head_apply
from membackprop.llama import Llama
from membackprop.losses import mean_nll, token_nll


def make_batches(path, steps, batch, seq):
    """Tokenize a text file into ``(inputs, labels)`` batches."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path["model"])
    ids = torch.tensor(tokenizer(open(path["text"]).read())["input_ids"])
    need = steps * batch * (seq + 1)
    assert len(ids) >= need, f"need {need} tokens, text has {len(ids)}"
    blocks = ids[:need].view(steps, batch, seq + 1)
    return [(b[:, :-1], b[:, 1:]) for b in blocks]


def train(model_path, batches, chunk, trainable_layers, device):
    """Train from the same checkpoint and return per-step losses.

    Args:
        model_path: Local checkpoint directory.
        batches: List of ``(inputs, labels)``.
        chunk: Chunk size, or ``None`` to materialize all logits.
        trainable_layers: Number of final layers to train with the tied head.
        device: Torch device string.

    Returns:
        List of float losses.
    """
    model = Llama.from_pretrained(model_path).to(device)
    for layer in model.layers[:-trainable_layers]:
        layer.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5)
    losses = []
    for inputs, labels in batches:
        inputs, labels = inputs.to(device), labels.to(device)
        hidden = model(inputs)
        if chunk is None:
            logits = model.lm_head(hidden).flatten(0, 1)
            nll = token_nll(logits, labels.reshape(-1))
        else:
            nll = chunked_head_apply(
                hidden, model.lm_head, token_nll, labels, chunk_size=chunk
            )
        loss = mean_nll(nll, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"chunk={chunk} step={len(losses)} loss={losses[-1]:.6f}", flush=True)
    return losses


def main():
    """Run both variants and write a CSV of losses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    batches = make_batches(
        {"model": args.model, "text": args.text}, args.steps, args.batch, args.seq
    )
    runs = [
        train(args.model, batches, chunk, args.layers, args.device)
        for chunk in (None, args.chunk)
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "full", "chunked", "abs_diff"])
        for i, (a, b) in enumerate(zip(*runs), start=1):
            writer.writerow([i, a, b, abs(a - b)])
    print("max abs diff", max(abs(a - b) for a, b in zip(*runs)))


if __name__ == "__main__":
    main()
