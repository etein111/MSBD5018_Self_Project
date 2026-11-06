"""Aggregate results/*.json into comparison tables (NLI + hallucination + bias)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("results")


def _load(pattern: str) -> list[dict]:
    rows = []
    for p in sorted(RESULTS_DIR.glob(pattern)):
        data = json.loads(p.read_text())
        if isinstance(data, list):
            rows.extend(data)
        else:
            rows.append(data)
    return rows


def print_nli_table():
    rows = _load("nli_*.json")
    if not rows:
        print("(no nli_*.json results yet)")
        return
    print("\n## NLI prompting comparison\n")
    print("| model | method | k | verbalizer | prefix | split | n | samp | temp | acc |")
    print("|-------|--------|---|------------|--------|-------|---|------|------|-----|")
    gen = {"cot", "rar", "step_back"}
    for r in sorted(rows, key=lambda r: (r.get("model", ""), r.get("method", ""),
                                          r.get("verbalizer", ""), r.get("split", ""),
                                          r.get("cot_samples", 0))):
        k = r.get("k") if r.get("method") == "few_shot" else "-"
        samp = r.get("cot_samples", "-") if r.get("method") in gen else "-"
        temp = f"{r.get('cot_temperature'):.1f}" if (r.get("method") in gen
                                                     and r.get("cot_temperature") is not None) else "-"
        print(f"| {r.get('model')} | {r.get('method')} | {k} | {r.get('verbalizer')} | "
              f"{r.get('prefix') or '-'} | {r.get('split')} | {r.get('n')} | {samp} | {temp} | "
              f"{r.get('accuracy', 0):.4f} |")


def print_hallucination_table():
    rows = _load("hallu_*.json")
    if not rows:
        print("\n(no hallu_*.json results yet)")
        return
    print("\n## Hallucination detection\n")
    print("| model | verbalizer | rule | n | accuracy | precision | recall | f1 |")
    print("|-------|------------|------|---|----------|-----------|--------|----|")
    for r in sorted(rows, key=lambda r: (r.get("model", ""), r.get("verbalizer", ""),
                                          r.get("rule", ""))):
        print(f"| {r.get('model')} | {r.get('verbalizer', '-')} | {r.get('rule', '-')} | "
              f"{r.get('n')} | {r.get('accuracy', 0):.4f} | {r.get('precision', 0):.4f} | "
              f"{r.get('recall', 0):.4f} | {r.get('f1', 0):.4f} |")


def print_bias_table():
    rows = _load("bias*.json")
    if not rows:
        print("\n(no bias_*.json results yet)")
        return
    print("\n## Bias (CrowS-Pairs pseudo-log-likelihood, race-color)\n")
    print("| model | bias_type | n | stereo_wins | stereo_preference_ratio |")
    print("|-------|-----------|-----|-------------|--------------------------|")
    for r in sorted(rows, key=lambda r: r.get("model", "")):
        print(f"| {r.get('model')} | {r.get('bias_type')} | {r.get('n')} | "
              f"{r.get('stereo_wins')} | {r.get('stereo_preference_ratio', 0):.4f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["nli", "hallucination", "bias", "all"], default="all")
    args = ap.parse_args()

    if args.task in ("nli", "all"):
        print_nli_table()
    if args.task in ("hallucination", "all"):
        print_hallucination_table()
    if args.task in ("bias", "all"):
        print_bias_table()


if __name__ == "__main__":
    main()
