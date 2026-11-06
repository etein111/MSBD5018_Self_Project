"""Phase 2 — fine-tune a BERT/RoBERTa sequence-classification head on MNLI.

Trains from scratch (only pretrained weights + a *new* 3-way head, no external
fine-tuned checkpoint), then evaluates on the SAME local MNLI dev jsonl files
used by Phase 1 prompting, so prompt vs fine-tune is directly comparable.

Usage (from Self_Project/):
    python -m src.finetune --model bert-base-uncased --out results/nli_bert_finetune.json
    python -m src.finetune --model roberta-base --out results/nli_roberta_finetune.json
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding,
                          get_linear_schedule_with_warmup)

from .data import load_mnli_all
from .utils import accuracy, get_device, save_json


class PairDataset(Dataset):
    """Tokenized (premise, hypothesis) pairs + integer labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings  # dict of list[int]
        self.labels = labels        # list[int]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item


def load_train_pairs(model_name: str, train_size: int, max_len: int):
    """Return (dataset, tokenizer) from an MNLI train slice, label 0/1/2 only."""
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Slice a bit more than needed: ~10% of MNLI train rows have label == -1
    # (hidden / no consensus) and must be dropped.
    raw = load_dataset("nyu-mll/multi_nli", split=f"train[:{train_size + 4000}]")
    rows = [(e["premise"], e["hypothesis"], int(e["label"]))
            for e in raw if int(e["label"]) in (0, 1, 2)]
    rows = rows[:train_size]

    premises = [r[0] for r in rows]
    hyps = [r[1] for r in rows]
    labels = [r[2] for r in rows]
    encodings = tokenizer(premises, hyps, truncation=True, max_length=max_len)
    return PairDataset(encodings, labels), tokenizer


def evaluate(model, tokenizer, device, data_dir, max_len, batch_size=32):
    """Evaluate on the local matched/mismatched dev jsonl (same as Phase 1)."""
    df = load_mnli_all(data_dir)
    out = []
    model.eval()
    for split in ("matched", "mismatched"):
        sub = df[df["split"] == split]
        enc = tokenizer(list(sub["premise"]), list(sub["hypothesis"]),
                        truncation=True, max_length=max_len, padding=True,
                        return_tensors="pt")
        preds = []
        with torch.no_grad():
            for i in range(0, len(sub), batch_size):
                batch = {k: v[i:i + batch_size].to(device) for k, v in enc.items()}
                logits = model(**batch).logits
                preds.extend(logits.argmax(dim=-1).cpu().tolist())
        acc = accuracy(sub["label"].tolist(), preds)
        out.append({"split": split, "n": int(len(sub)), "accuracy": acc})
    return out


def train(args):
    device = args.device or get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset, tokenizer = load_train_pairs(args.model, args.train_size, args.max_len)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    warmup = int(0.06 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    print(f"[{args.model}] train_size={args.train_size} steps/epoch={len(loader)} "
          f"epochs={args.epochs} device={device}")

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
        avg_loss = running / len(loader)
        evals = evaluate(model, tokenizer, device, args.data_dir, args.max_len)
        line = f"  epoch {epoch}: loss={avg_loss:.4f} | " + \
               " | ".join(f"{e['split']}={e['accuracy']:.4f}" for e in evals)
        print(line)
        history.append({"epoch": epoch, "loss": avg_loss, "dev": evals})

    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="bert-base-uncased | roberta-base")
    ap.add_argument("--train-size", type=int, default=15000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data/multi_nli")
    ap.add_argument("--out", default=None, help="results JSON path")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    history = train(args)
    final = history[-1]["dev"]  # report the final-epoch model

    results = [{
        "model": args.model, "method": "finetune", "split": r["split"],
        "verbalizer": "-", "prefix": "-", "k": "-", "n": r["n"],
        "accuracy": r["accuracy"], "train_size": args.train_size,
        "epochs": args.epochs, "lr": args.lr, "max_len": args.max_len,
        "history": history,
    } for r in final]

    if args.out:
        save_json(results, args.out)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
