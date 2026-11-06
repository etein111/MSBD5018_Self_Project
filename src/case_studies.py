"""Extract real input/output case studies for the report (report/case_studies.md).

Runs the same pipelines as the eval scripts, but prints the actual prompt text,
model output and gold label for a handful of hand-selected examples.
"""
from __future__ import annotations

import json
from pathlib import Path

from .data import load_mnli_split
from .hallucination import ANNOTATION_TO_BINARY, load_hallucination_dataset
from .nli_eval import load_scorer
from .prompts import VERBALIZERS, build_prompt
from .scorer import parse_label
from .utils import ID2LABEL, get_device


def nli_cases(data_dir: str, scorer_fl, scorer_gpt2) -> list[dict]:
    df = load_mnli_split(data_dir, "matched")
    rows = []
    for lab in (0, 1, 2):  # one per label
        r = df[df["label"] == lab].iloc[0]
        prompt = build_prompt(r["premise"], r["hypothesis"], method="zero_shot",
                              verbalizer=VERBALIZERS["label_words"],
                              model_type="seq2seq")
        rows.append({
            "premise": r["premise"], "hypothesis": r["hypothesis"],
            "gold": ID2LABEL[lab],
            "flan_zero": ID2LABEL[scorer_fl.predict(prompt, VERBALIZERS["label_words"])],
            "gpt2_zero": ID2LABEL[scorer_gpt2.predict(
                build_prompt(r["premise"], r["hypothesis"], method="zero_shot",
                             verbalizer=VERBALIZERS["label_words"], model_type="causal"),
                VERBALIZERS["label_words"])],
        })
    return rows


def hallucination_cases(scorer_gpt2, scorer_fl_ymn) -> list[dict]:
    df = load_hallucination_dataset()
    out = []
    for lab in (0, 1):
        r = df[df["label"] == lab].iloc[0]
        prompt_g = build_prompt(r["premise"], r["hypothesis"], method="zero_shot",
                                verbalizer=VERBALIZERS["label_words"], model_type="causal")
        prompt_f = build_prompt(r["premise"], r["hypothesis"], method="zero_shot",
                                verbalizer=VERBALIZERS["yes_maybe_no"], model_type="seq2seq")
        g_nli = scorer_gpt2.predict(prompt_g, VERBALIZERS["label_words"])
        f_nli = scorer_fl_ymn.predict(prompt_f, VERBALIZERS["yes_maybe_no"])
        out.append({
            "premise": r["premise"][:400], "hypothesis": r["hypothesis"],
            "gold": "factual" if lab == 0 else "non-factual",
            "gpt2_label_words_nli": ID2LABEL[g_nli],
            "gpt2_pred": "factual" if g_nli == 0 else "non-factual",
            "flan_ymn_nli": ID2LABEL[f_nli],
            "flan_pred": "factual" if f_nli == 0 else "non-factual",
        })
    return out


def bias_cases() -> list[dict]:
    rows = json.loads(Path("results/bias.json").read_text())
    # use RoBERTa's top-3 pairs by |delta| (same as the eval prints)
    r = next(x for x in rows if x["model"] == "roberta-base")
    top = sorted(r["pairs"], key=lambda p: abs(p["delta"]), reverse=True)[:3]
    return [{"sent_more": p["sent_more"], "sent_less": p["sent_less"],
             "stereo": p["stereo_antistereo"]} for p in top]


def main():
    device = get_device()
    scorer_fl = load_scorer("google/flan-t5-base", device)
    scorer_gpt2 = load_scorer("gpt2", device)
    scorer_fl_ymn = scorer_fl  # same model, different verbalizer at call site

    data = {
        "nli": nli_cases("data/multi_nli", scorer_fl, scorer_gpt2),
        "hallucination": hallucination_cases(scorer_gpt2, scorer_fl_ymn),
        "bias": bias_cases(),
    }
    Path("report/case_studies.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
