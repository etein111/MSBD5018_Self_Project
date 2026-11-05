# MSBD5018 Individual Project — Evaluation of Language Models

Week-1 scope: NLI prompting (GPT-2 + Flan-T5) + hallucination detection + race bias (BERT/RoBERTa).
See `PROJECT_GUIDE.md` for the full plan and `PROJECT_GUIDE.md#5` for the prompt-engineering menu.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.backends.mps.is_available())"   # expect True
```

## Run

```bash
# 2.1 NLI (multi-method comparison)
python -m src.nli_eval --model gpt2 --method zero_shot --split both
python -m src.nli_eval --model google/flan-t5-base --method zero_shot --split both
python -m src.nli_eval --model google/flan-t5-base --method few_shot --k 4 --split both
python -m src.nli_eval --model google/flan-t5-base --method cot --cot-samples 3 --split matched
python -m src.nli_eval --model gpt2 --method zero_shot --verbalizer yes_maybe_no --split matched
python -m src.nli_eval --model google/flan-t5-base --method zero_shot --prefix emotion --split matched

# 2.2 hallucination detection
python -m src.hallucination --model google/flan-t5-base
python -m src.hallucination --model gpt2 --rule contradiction_only

# 3 race bias (CrowS-Pairs pseudo-log-likelihood)
python -m src.crows_pairs --models bert-base-uncased roberta-base --bias-type race-color --n 80
```

## Notes

- Run from `Self_Project/` so relative imports and the `data/multi_nli` default path resolve.
- First run of each model downloads weights; datasets are fetched from Hugging Face.
- All sampling uses a fixed seed; pass `--out results/xxx.json` to persist results.
