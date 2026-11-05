"""Prompt templates & verbalizers for 3-way NLI (entailment/neutral/contradiction)."""
from __future__ import annotations

# Verbalizer options: label_id -> surface token(s) the model should emit.
# The "label_words" verbalizer uses the canonical label strings; the others are
# ablations. NOTE: for zero-shot, "yes_maybe_no"/"abc" only make sense when the
# few-shot demos (or an instruction) define the mapping for the model.
VERBALIZERS = {
    "label_words": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "yes_maybe_no": {0: "yes", 1: "maybe", 2: "no"},
    "abc": {0: "A", 1: "B", 2: "C"},
}

# Cheap one-line prefix ablations (Tier 1/Tier 2).
EMOTION_PREFIX = "Take a deep breath and think carefully about the relationship.\n"
ROLE_PREFIX = "You are an expert in natural language inference.\n"


def _options(verbalizer: dict) -> str:
    return ", ".join(verbalizer[i] for i in sorted(verbalizer))


def _example(premise: str, hypothesis: str, verbalizer: dict, label=None) -> str:
    """Format one premise/hypothesis pair.

    If ``label`` is None this is the query (ends right before the model should
    emit the label token); otherwise it is a few-shot demonstration.
    """
    head = f"Premise: {premise}\nHypothesis: {hypothesis}\nRelationship ({_options(verbalizer)}):"
    if label is None:
        return head
    return head + f" {verbalizer[label]}"


def build_prompt(premise, hypothesis, method="zero_shot", verbalizer=None,
                 model_type="causal", demos=None, prefix="") -> str:
    """Build the prompt string for one NLI example.

    method:    "zero_shot" | "few_shot" | "cot"
    demos:     list of {"premise", "hypothesis", "label"} (used by few_shot)
    prefix:    optional instruction prepended (e.g. EMOTION_PREFIX / ROLE_PREFIX)
    model_type: reserved for model-specific templates (currently unused).
    """
    verbalizer = verbalizer or VERBALIZERS["label_words"]
    parts = []
    if prefix:
        parts.append(prefix)

    if method == "cot":
        parts.append(
            f"Premise: {premise}\nHypothesis: {hypothesis}\n"
            f"Determine whether the hypothesis is entailment, neutral, or contradiction.\n"
            f"Let's think step by step."
        )
        return "\n".join(parts)

    if method == "few_shot":
        for d in demos or []:
            parts.append(_example(d["premise"], d["hypothesis"], verbalizer, d["label"]))
        parts.append("")
    parts.append(_example(premise, hypothesis, verbalizer))
    return "\n".join(parts)
