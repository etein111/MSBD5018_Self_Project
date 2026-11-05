"""2.2 hallucination detection: apply the NLI models to wiki_bio_gpt3_hallucination.

Usage:
    python -m src.hallucination --model google/flan-t5-base
    python -m src.hallucination --model gpt2 --rule contradiction_only
"""
from __future__ import annotations

import argparse

import pandas as pd

from .nli_eval import load_scorer
from .prompts import VERBALIZERS, build_prompt
from .utils import accuracy, get_device, precision_recall_f1, save_json

# annotation string -> binary label (0=Factual, 1=Non-Factual)
ANNOTATION_TO_BINARY = {"accurate": 0, "minor_inaccurate": 1, "major_inaccurate": 1}


def load_hallucination_dataset() -> pd.DataFrame:
    """Load & flatten wiki_bio_gpt3_hallucination to sentence-level rows.

    NOTE: column names vary by dataset version; this introspects the available
    splits. If it fails, print `ds[0].keys()` to check the field names.
    """
    from datasets import load_dataset

    name = "potsawee/wiki_bio_gpt3_hallucination"
    ds = None
    for split in ("test", "train", "validation"):
        try:
            ds = load_dataset(name, split=split)
            break
        except Exception:
            continue
    if ds is None:
        ds = next(iter(load_dataset(name).values()))  # first split of a DatasetDict

    rows = []
    for ex in ds:
        premise = ex["wiki_bio_text"]
        sents = ex["gpt3_sentences"]
        anns = ex["annotation"]
        if isinstance(sents, str):  # some versions store one string; naive split
            sents = [s.strip() for s in sents.split(". ") if s.strip()]
        for s, a in zip(sents, anns):
            rows.append({"premise": premise, "hypothesis": s, "annotation": a})

    df = pd.DataFrame(rows)
    df["label"] = df["annotation"].map(ANNOTATION_TO_BINARY)
    return df.dropna(subset=["label"]).reset_index(drop=True)


def nli_to_hallucination(nli_label: int, rule: str = "entailment_factual") -> int:
    """Map an NLI label (0/1/2) to hallucination (0=Factual, 1=Non-Factual)."""
    if rule == "entailment_factual":
        return 0 if nli_label == 0 else 1  # entailment -> factual
    if rule == "contradiction_only":
        return 1 if nli_label == 2 else 0  # only contradiction -> non-factual
    raise ValueError(f"unknown rule: {rule}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--verbalizer", default="label_words", choices=list(VERBALIZERS))
    ap.add_argument("--rule", default="entailment_factual",
                    choices=["entailment_factual", "contradiction_only"])
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    scorer = load_scorer(args.model, device)
    verbalizer = VERBALIZERS[args.verbalizer]

    df = load_hallucination_dataset()
    if args.max_samples:
        df = df.head(args.max_samples)

    preds, golds = [], []
    for row in df.itertuples():
        prompt = build_prompt(row.premise, row.hypothesis, method="zero_shot",
                              verbalizer=verbalizer, model_type=scorer.model_type)
        nli_label = scorer.predict(prompt, verbalizer)
        preds.append(nli_to_hallucination(nli_label, args.rule))
        golds.append(int(row.label))

    acc = accuracy(golds, preds)
    prf = precision_recall_f1(golds, preds, pos_label=1)  # Non-Factual = positive
    result = {"model": args.model, "rule": args.rule, "verbalizer": args.verbalizer,
              "n": len(golds), "accuracy": acc, **prf}
    print(result)
    if args.out:
        save_json(result, args.out)


if __name__ == "__main__":
    main()
