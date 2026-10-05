"""Transformer baseline for the episodic tasks (experiments 11-22).

Trains a small decoder-only transformer (GPT-style) by next-word prediction on exactly
the stories the bit-vector network sees (exported with `DUMP_STORIES=dir` from
`examples/episodic.rs`), and scores it the same way: is the most likely word at the
answer position the answer, on test stories, split into seen and held-out pairs.

Each story is one sequence (the whole story is in the transformer's context window, its
native working memory). Loss is on every position, as the bit-vector predictor learns
every next word. Reports, per model size and number of passes over the training set:
held-out / seen accuracy, parameters, weight memory, training passes, and single-thread
time per word for training and inference.

Usage: python3 transformer_baseline.py STORIES_FILE [--sizes tiny,small,medium]
       [--epochs 1,3,10,30] [--seed 0]
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

SIZES = {
    # name: (layers, d_model, heads)
    "tiny": (2, 32, 2),
    "small": (2, 64, 4),
    "medium": (4, 128, 4),
}


def load(path):
    train, test = [], []
    for line in open(path):
        split, held, answer_at, words = line.rstrip("\n").split("\t")
        (train if split == "train" else test).append((words.split(), int(answer_at), held == "1"))
    return train, test


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask, pad):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, key_padding_mask=pad, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, layers, d, heads, ctx):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)

    def forward(self, ids, pad):
        n = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(n))
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
        for b in self.blocks:
            x = b(x, mask, pad)
        return self.head(self.ln(x))


def batches(stories, vocab, size, shuffle, gen):
    order = torch.randperm(len(stories), generator=gen).tolist() if shuffle else list(range(len(stories)))
    for i in range(0, len(order), size):
        chunk = [stories[j] for j in order[i : i + size]]
        n = max(len(w) for w, _, _ in chunk)
        ids = torch.zeros(len(chunk), n, dtype=torch.long)
        pad = torch.ones(len(chunk), n, dtype=torch.bool)
        for r, (w, _, _) in enumerate(chunk):
            ids[r, : len(w)] = torch.tensor([vocab[x] for x in w])
            pad[r, : len(w)] = False
        yield chunk, ids, pad


def evaluate(model, test, vocab):
    model.eval()
    right = {True: [0, 0], False: [0, 0]}
    words, start = 0, time.perf_counter()
    with torch.no_grad():
        for chunk, ids, pad in batches(test, vocab, 1, False, None):  # one story at a time, like the network
            logits = model(ids, pad)
            for r, (w, at, held) in enumerate(chunk):
                pred = logits[r, at - 1].argmax().item()  # predicting word `at` from words < at
                right[held][0] += pred == vocab[w[at]]
                right[held][1] += 1
                words += len(w)
    secs = time.perf_counter() - start
    pct = lambda h: 100.0 * right[h][0] / max(1, right[h][1])
    return pct(True), pct(False), 1e6 * secs / words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stories")
    ap.add_argument("--sizes", default="tiny,small,medium")
    ap.add_argument("--epochs", default="1,3,10,30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(1)  # the bit-vector network runs on one thread
    train, test = load(args.stories)
    words = sorted({w for s, _, _ in train + test for w in s})
    vocab = {w: i for i, w in enumerate(words)}
    ctx = max(len(s) for s, _, _ in train + test)
    checkpoints = sorted(int(e) for e in args.epochs.split(","))
    train_words = sum(len(s) for s, _, _ in train)
    print(f"{args.stories}: {len(train)} train / {len(test)} test stories, vocab {len(vocab)}, max length {ctx}")
    for name in args.sizes.split(","):
        layers, d, heads = SIZES[name]
        torch.manual_seed(args.seed)
        gen = torch.Generator().manual_seed(args.seed)
        model = GPT(len(vocab), layers, d, heads, ctx)
        params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        train_secs, epoch = 0.0, 0
        for target in checkpoints:
            model.train()
            start = time.perf_counter()
            while epoch < target:
                for _, ids, pad in batches(train, vocab, args.batch, True, gen):
                    logits = model(ids[:, :-1], pad[:, :-1])
                    y = ids[:, 1:].masked_fill(pad[:, 1:], -100)
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                epoch += 1
            train_secs += time.perf_counter() - start
            held, seen, infer_us = evaluate(model, test, vocab)
            print(
                f"  {name:6} L={layers} d={d} h={heads}: {params:,} params ({4 * params / 1e6:.2f} MB fp32); "
                f"{epoch:2} passes ({epoch * train_words:,} training words): held-out {held:5.1f}%  seen {seen:5.1f}%; "
                f"train {1e6 * train_secs / (epoch * train_words):.1f} µs/word, inference {infer_us:.1f} µs/word",
                flush=True,
            )


if __name__ == "__main__":
    main()
