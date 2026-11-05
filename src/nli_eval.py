"""2.1 main evaluation: run a model + prompt method over MNLI, report accuracy.

Usage (from Self_Project/):
    python -m src.nli_eval --model gpt2 --method zero_shot --split matched
    python -m src.nli_eval --model google/flan-t5-base --method few_shot --k 4 --split both
    python -m src.nli_eval --model google/flan-t5-base --method cot --cot-samples 3 --split both
    python -m src.nli_eval --model gpt2 --method zero_shot --verbalizer yes_maybe_no --split matched
    python -m src.nli_eval --model google/flan-t5-base --method zero_shot --prefix emotion --split matched
"""
from __future__ import annotations

import argparse

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from .data import load_few_shot_demos, load_mnli_all
from .prompts import EMOTION_PREFIX, ROLE_PREFIX, VERBALIZERS, build_prompt
from .scorer import Scorer
from .utils import accuracy, get_device, save_json

PREFIXES = {"": "", "emotion": EMOTION_PREFIX, "role": ROLE_PREFIX}


def load_scorer(model_name: str, device: str) -> Scorer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "t5" in model_name.lower() or "bart" in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model_type = "seq2seq"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model_type = "causal"
    return Scorer(model, tokenizer, model_type, device)


def evaluate_split(scorer, df, method, verbalizer, demos, prefix, cot_samples):
    preds, golds = [], []
    for row in df.itertuples():
        prompt = build_prompt(row.premise, row.hypothesis, method=method,
                              verbalizer=verbalizer, model_type=scorer.model_type,
                              demos=demos, prefix=prefix)
        if method == "cot":
            pred = scorer.predict_cot(prompt, verbalizer, num_samples=cot_samples)
            if pred is None:  # fall back to zero-shot verbalizer scoring
                p0 = build_prompt(row.premise, row.hypothesis, method="zero_shot",
                                  verbalizer=verbalizer, model_type=scorer.model_type)
                pred = scorer.predict(p0, verbalizer)
        else:
            pred = scorer.predict(prompt, verbalizer)
        preds.append(pred)
        golds.append(int(row.label))
    return preds, golds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. gpt2 or google/flan-t5-base")
    ap.add_argument("--method", required=True, choices=["zero_shot", "few_shot", "cot"])
    ap.add_argument("--split", default="both", choices=["matched", "mismatched", "both"])
    ap.add_argument("--k", type=int, default=4, help="few-shot demo count")
    ap.add_argument("--verbalizer", default="label_words", choices=list(VERBALIZERS))
    ap.add_argument("--prefix", default="", choices=list(PREFIXES), help="emotion/role ablation")
    ap.add_argument("--cot-samples", type=int, default=3)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--data-dir", default="data/multi_nli")
    ap.add_argument("--out", default=None, help="results JSON path")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    scorer = load_scorer(args.model, device)
    verbalizer = VERBALIZERS[args.verbalizer]
    prefix = PREFIXES[args.prefix]

    demos = load_few_shot_demos(args.k) if args.method == "few_shot" else None

    df = load_mnli_all(args.data_dir)
    splits = ["matched", "mismatched"] if args.split == "both" else [args.split]

    results = []
    for split in splits:
        sub = df[df["split"] == split]
        if args.max_samples:
            sub = sub.head(args.max_samples)
        preds, golds = evaluate_split(scorer, sub, args.method, verbalizer, demos, prefix, args.cot_samples)
        acc = accuracy(golds, preds)
        r = {"model": args.model, "method": args.method, "split": split,
             "verbalizer": args.verbalizer, "prefix": args.prefix, "k": args.k,
             "n": len(golds), "accuracy": acc}
        results.append(r)
        print(f"[{args.model}] {args.method} | split={split} | n={r['n']} | acc={acc:.4f}")

    if args.out:
        save_json(results, args.out)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
