"""Shared utilities: device selection, label maps, metrics, JSON I/O."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# NLI label <-> id (entailment=0, neutral=1, contradiction=2)
LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Hallucination annotation string -> binary id (0=Factual, 1=Non-Factual)
HALLU_BINARY = {"accurate": 0, "minor_inaccurate": 1, "major_inaccurate": 1}


def get_device() -> str:
    """Pick the best available torch device (MPS on Apple Silicon)."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def precision_recall_f1(y_true, y_pred, pos_label: int = 1) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score

    return {
        "precision": float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
    }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path) as f:
        return json.load(f)
