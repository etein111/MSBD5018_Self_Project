"""Model wrappers: token-level scoring + free-form generation.

Covers causal LMs (GPT-2) and seq2seq LMs (T5/Flan-T5) for the NLI task.
"""
from __future__ import annotations

import torch


def parse_label(text: str):
    """Extract an NLI label id from free-form model output, or None if unclear.

    Order matters: contradiction/neutral are checked before entailment so that
    phrases like "does not entail" are not mis-read as entailment.
    """
    t = text.lower()
    if "contradiction" in t or "contradict" in t:
        return 2
    if "neutral" in t:
        return 1
    if "entailment" in t or "entailed" in t or "entail" in t:
        return 0
    return None


class Scorer:
    def __init__(self, model, tokenizer, model_type: str, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.model_type = model_type  # "causal" | "seq2seq"
        self.device = device

    # ---- causal LM: score P(target | prompt) ----
    def _score_tokens_causal(self, prompt: str, target_tokens) -> list[float]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        scores = []
        for tstr in target_tokens:
            tids = self.tokenizer.encode(tstr, add_special_tokens=False)
            if not tids:
                scores.append(-1e9)
                continue
            input_ids = prompt_ids + tids
            with torch.no_grad():
                logits = self.model(torch.tensor([input_ids], device=self.device)).logits[0]
            total = 0.0
            for j, tid in enumerate(tids):
                prev_idx = len(prompt_ids) + j - 1  # logit that predicts this token
                lp = torch.log_softmax(logits[prev_idx], dim=0)
                total += lp[tid].item()
            scores.append(total)
        return scores

    # ---- seq2seq LM: score the first decoded token ----
    def _score_tokens_seq2seq(self, prompt: str, target_tokens) -> list[float]:
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        start_id = getattr(self.model.config, "decoder_start_token_id", 0) or 0
        dec = torch.tensor([[start_id]], device=self.device)
        with torch.no_grad():
            out = self.model(**enc, decoder_input_ids=dec)
        lp = torch.log_softmax(out.logits[0, 0], dim=0)  # first generated token
        scores = []
        for tstr in target_tokens:
            tids = self.tokenizer.encode(tstr, add_special_tokens=False)
            # single-token verbalizer assumption (holds for our token choices);
            # multi-token verbalizers would need constrained generation instead.
            scores.append(lp[tids[0]].item() if tids else -1e9)
        return scores

    def score_labels(self, prompt: str, verbalizer: dict) -> dict:
        """Return {label_id: log prob} for each verbalizer surface token."""
        ids = sorted(verbalizer)
        tokens = [verbalizer[i] for i in ids]
        if self.model_type == "causal":
            scores = self._score_tokens_causal(prompt, tokens)
        else:
            scores = self._score_tokens_seq2seq(prompt, tokens)
        return {i: s for i, s in zip(ids, scores)}

    def predict(self, prompt: str, verbalizer: dict) -> int:
        probs = self.score_labels(prompt, verbalizer)
        return max(probs, key=probs.get)

    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 1.0,
                 num_samples: int = 1) -> list[str]:
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outs = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            num_return_sequences=num_samples,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        return self.tokenizer.batch_decode(outs[:, prompt_len:], skip_special_tokens=True)

    def predict_cot(self, prompt: str, verbalizer: dict, num_samples: int = 1,
                    temperature: float = 1.0, max_new_tokens: int = 64):
        """CoT: generate reasoning, parse a label, majority-vote over samples.

        Returns a label id, or None if no sample yields a parseable label
        (the caller should fall back to zero-shot verbalizer scoring).
        """
        texts = self.generate(prompt, max_new_tokens=max_new_tokens,
                              temperature=temperature, num_samples=num_samples)
        parsed = [p for p in (parse_label(t) for t in texts) if p is not None]
        if not parsed:
            return None
        return max(set(parsed), key=parsed.count)
