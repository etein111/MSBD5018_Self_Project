# MSBD5018 Individual Project — Evaluation of Language Models

Evaluates local language models on three tasks:

1. **NLI** (3-way entailment/neutral/contradiction) — primary method is **prompting**
   (zero/few-shot, CoT, self-consistency, RaR, step-back) on GPT-2 + Flan-T5, with a
   **verbalizer ablation**; secondary comparison is **fine-tuning** BERT/RoBERTa.
2. **Hallucination detection** — reuses the NLI models unchanged (`entailment → factual`).
3. **Race bias** — CrowS-Pairs pseudo-log-likelihood on BERT/RoBERTa.

Report: `report/main.tex` (LaTeX, Overleaf) with figures in `report/figures/` and case
studies in `report/case_studies.json`. Final numbers are summarized in `RESULTS.md`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.backends.mps.is_available())"   # expect True
```

Model weights are cached in the project-local `.hf_cache/` (set by `src/__init__.py` via
`HF_HOME`) so they don't mix with other projects. Run everything from `Self_Project/`.

## Run

### 2.1 NLI — prompting (primary)

```bash
# zero-shot / few-shot (full dev set)
python -m src.nli_eval --model gpt2 --method zero_shot --split both
python -m src.nli_eval --model google/flan-t5-base --method zero_shot --split both
python -m src.nli_eval --model google/flan-t5-base --method few_shot --k 4 --split both

# verbalizer ablation (the biggest lever)
python -m src.nli_eval --model google/flan-t5-base --method few_shot --k 4 \
    --verbalizer yes_maybe_no --split matched

# chain-of-thought + advanced prompts (n=300 subset)
python -m src.nli_eval --model google/flan-t5-base --method cot --split matched
python -m src.nli_eval --model google/flan-t5-base --method cot --cot-samples 3 --split matched  # self-consistency
python -m src.nli_eval --model google/flan-t5-base --method rar --split matched
python -m src.nli_eval --model google/flan-t5-base --method step_back --split matched
```

### 2.1 NLI — fine-tuning (secondary comparison)

```bash
python -m src.finetune --model bert-base-uncased --train-size 15000 --epochs 3
python -m src.finetune --model roberta-base --train-size 15000 --epochs 3
```

### 2.2 Hallucination detection

```bash
python -m src.hallucination --model google/flan-t5-base --verbalizer yes_maybe_no
python -m src.hallucination --model gpt2
```

### 3 Race bias

```bash
python -m src.crows_pairs --models bert-base-uncased roberta-base --bias-type race-color --n 80
```

### Figures & case studies

```bash
python -m src.plot          # -> report/figures/*.png
python -m src.case_studies  # -> report/case_studies.json
```

## Notes

- Sampling uses fixed seeds; pass `--out results/xxx.json` to persist results.
- The key finding: the **verbalizer** (label surface tokens) matters more than the prompt
  method — single-token `yes/maybe/no` lifts Flan-T5 few-shot from 0.450 → 0.633 on
  MNLI-matched and rescues hallucination F1 from 0.00 → 0.877.
