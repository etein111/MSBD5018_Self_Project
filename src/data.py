"""Data loading & cleaning for MNLI, wiki_bio hallucination, and CrowS-Pairs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .utils import LABEL2ID

MNLI_FILES = {
    "matched": "dev_matched_sampled-1.jsonl",
    "mismatched": "dev_mismatched_sampled-1.jsonl",
}


def load_mnli_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append({
                "premise": d["sentence1"],
                "hypothesis": d["sentence2"],
                "gold_label": d["gold_label"],
                "genre": d.get("genre"),
                "pairID": d.get("pairID"),
            })
    return pd.DataFrame(rows)


def load_mnli_split(data_dir, split: str = "matched") -> pd.DataFrame:
    """Load one MNLI dev split, dropping examples without a consensus label."""
    df = load_mnli_jsonl(Path(data_dir) / MNLI_FILES[split])
    df = df[df["gold_label"].isin(LABEL2ID)].copy()  # drop gold_label == "-"
    df["label"] = df["gold_label"].map(LABEL2ID)
    return df.reset_index(drop=True)


def load_mnli_all(data_dir) -> pd.DataFrame:
    """Load both matched + mismatched dev splits with a 'split' column."""
    frames = []
    for split in ("matched", "mismatched"):
        df = load_mnli_split(data_dir, split)
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_few_shot_demos(k: int = 4) -> list[dict]:
    """Load k label-balanced (premise, hypothesis, label) demos from MNLI train.

    Uses HF streaming so we never download the full 392k training set. Labels in
    HF `multi_nli` are 0/1/2 = entailment/neutral/contradiction, matching
    ``utils.LABEL2ID``.
    """
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/multi_nli", split="train", streaming=True)
    per_label = k // 3 + (1 if k % 3 else 0)
    buckets = {0: [], 1: [], 2: []}
    for ex in ds:
        lab = int(ex["label"])
        if len(buckets[lab]) < per_label:
            buckets[lab].append({
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "label": lab,
            })
        if all(len(v) >= per_label for v in buckets.values()):
            break

    demos = []
    for i in range(per_label):
        for lab in (0, 1, 2):
            if i < len(buckets[lab]):
                demos.append(buckets[lab][i])
    return demos[:k]
