"""3 bias evaluation: pseudo-log-likelihood on CrowS-Pairs (race-color).

Usage:
    python -m src.crows_pairs --models bert-base-uncased roberta-base --bias-type race-color --n 80
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .utils import get_device, save_json


CROWS_PAIRS_URL = (
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/"
    "data/crows_pairs_anonymized.csv"
)


def load_crows_pairs(bias_type: str = "race-color", n: int = 80, seed: int = 0):
    """Return a list of {"sent_more","sent_less","stereo_antistereo","bias_type"}.

    datasets>=5 no longer supports script-based datasets (crows_pairs is one),
    so we read the original CSV from the CrowS-Pairs repo and filter locally.
    """
    import pandas as pd

    df = pd.read_csv(CROWS_PAIRS_URL)
    df = df[df["bias_type"] == bias_type].reset_index(drop=True)
    if n is not None:
        df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    return df[["sent_more", "sent_less", "stereo_antistereo", "bias_type"]].to_dict("records")


def sentence_pseudo_log_likelihood(text: str, model, tokenizer, device: str) -> float:
    """PLL(s) = sum over non-special tokens of log P(token | masked context).

    One token is masked at a time, matching the CrowS-Pairs pseudo-log-likelihood.
    """
    ids = tokenizer.encode(text, add_special_tokens=True)
    skip = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id,
            tokenizer.mask_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id}
    skip = {t for t in skip if t is not None}

    total = 0.0
    for i, tok in enumerate(ids):
        if tok in skip:
            continue
        masked = list(ids)
        masked[i] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(torch.tensor([masked], device=device)).logits[0, i]
        lp = torch.log_softmax(logits, dim=0)
        total += lp[tok].item()
    return total


def evaluate(model_name: str, bias_type: str = "race-color", n: int = 80,
             seed: int = 0, device: str = None) -> dict:
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()

    ds = load_crows_pairs(bias_type, n, seed)
    stereo_wins = 0
    pairs = []
    for ex in ds:
        pll_more = sentence_pseudo_log_likelihood(ex["sent_more"], model, tokenizer, device)
        pll_less = sentence_pseudo_log_likelihood(ex["sent_less"], model, tokenizer, device)
        # stereo_antistereo tells which of sent_more/sent_less is the stereotyping one
        if ex["stereo_antistereo"] == "stereo":
            stereo_pll, anti_pll = pll_more, pll_less
        else:
            stereo_pll, anti_pll = pll_less, pll_more
        stereo_wins += int(stereo_pll > anti_pll)
        pairs.append({"sent_more": ex["sent_more"], "sent_less": ex["sent_less"],
                      "stereo_antistereo": ex["stereo_antistereo"],
                      "pll_more": pll_more, "pll_less": pll_less,
                      "delta": stereo_pll - anti_pll})

    ratio = stereo_wins / len(pairs) if pairs else 0.0
    return {"model": model_name, "bias_type": bias_type, "n": len(pairs),
            "stereo_wins": stereo_wins, "stereo_preference_ratio": ratio, "pairs": pairs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["bert-base-uncased", "roberta-base"])
    ap.add_argument("--bias-type", default="race-color")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    results = []
    for m in args.models:
        r = evaluate(m, args.bias_type, args.n, args.seed, device)
        results.append(r)
        print(f"[{m}] n={r['n']} stereo_preference={r['stereo_preference_ratio']:.4f} "
              f"({r['stereo_wins']}/{r['n']})")
        top = sorted(r["pairs"], key=lambda p: abs(p["delta"]), reverse=True)[:3]
        for p in top:
            print(f"    more: {p['sent_more']}")
            print(f"    less: {p['sent_less']}")

    if args.out:
        save_json(results, args.out)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
