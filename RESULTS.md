# Phase 1 实验结果汇总 — NLI / 幻觉检测 / 偏见评估

> 复现：所有结果 JSON 在 `results/`，运行 `python -m src.report --task all` 可重新生成汇总表。

## 1. 实验设置

| 项目 | 内容 |
|------|------|
| 环境 | Apple Silicon (MPS)、Python 3.12、torch 2.15.0、transformers 5.22.1、datasets 5.1.0 |
| 模型 | gpt2 (124M, 因果)、google/flan-t5-base (248M, seq2seq)、bert-base-uncased (110M)、roberta-base (125M) |
| 数据 | MNLI dev matched (2460) + mismatched (2464)；wiki_bio_gpt3_hallucination (1908 句)；CrowS-Pairs race-color (80 对) |
| 打分方式 | 零/少样本用 verbalizer 秩分类（整序列 log-prob + 长度归一化）；CoT 用生成 + 多数投票 + 解析 |

**随机基线**：NLI 三分类 = 0.333；幻觉检测二分类 = 0.5；偏见 = 0.5（无偏）。

## 2. NLI 核心对比（label_words verbalizer）

| 方法 | gpt2 (m/mm) | flan-t5-base (m/mm) |
|------|-------------|---------------------|
| zero-shot | 0.356 / 0.352 | 0.365 / 0.367 |
| few-shot k=4 | 0.315 / 0.323 | **0.450 / 0.450** |
| CoT (n=300) | 0.393 / 0.340 | 0.360 / 0.313 |

## 3. Verbalizer 消融（few-shot k=4，matched）

| verbalizer | gpt2 | flan-t5-base |
|------------|------|--------------|
| label_words（entailment/neutral/contradiction，多子词） | 0.315 | 0.450 |
| A/B/C（单 token） | 0.363 | 0.493 |
| **yes/maybe/no（单 token）** | 0.366 | **0.633** |

## 4. 前缀消融（zero-shot，matched）

| 前缀 | gpt2 | flan-t5-base |
|------|------|--------------|
| 无 | 0.356 | 0.365 |
| emotion（"take a deep breath…"） | 0.340 | 0.363 |
| role（"You are an expert…"） | 0.339 | 0.363 |

## 5. 幻觉检测（wiki_bio_gpt3_hallucination，1908 句）

| 模型 | 规则 | accuracy | precision | recall | F1 |
|------|------|----------|-----------|--------|-----|
| gpt2 | entailment_factual | 0.559 | 0.725 | 0.636 | 0.678 |
| gpt2 | contradiction_only | 0.559 | 0.725 | 0.636 | 0.678 |
| flan-t5-base | entailment_factual | 0.270 | 0.000 | 0.000 | 0.000 |

## 6. 种族偏见（CrowS-Pairs race-color，80 对）

| 模型 | 刻板偏好比（0.5=无偏） |
|------|------------------------|
| bert-base-uncased | 0.625（50/80） |
| roberta-base | 0.638（51/80） |

## 7. 关键结论

1. **Verbalizer 是最大杠杆**。把多子词的 `entailment/neutral/contradiction` 换成单 token 的 `yes/maybe/no`，flan-t5 few-shot 从 0.45 → **0.63**（+18pt）。根因：子词切分（T5 把 `entailment` 切成 4 个子词，`neutral`/`contradiction` 各 1 个）与长度归一化带来的干扰，单 token 完全规避。
2. **few-shot 只对指令微调模型有效**。flan-t5 0.365→0.450（+8.5pt）；gpt2 0.356→0.315（-4pt），小因果 LM 被 demo 带偏。
3. **CoT 不稳定且不优于 few-shot**（且仅 n=300，误差 ±3%）。
4. **emotion/role 前缀无效**，gpt2 上反而略降。
5. **幻觉检测 gpt2 可用（F1 0.68）、flan-t5 退化（全预测 factual）**；zero-shot NLI 作为幻觉检测器整体偏弱。
6. **BERT/RoBERTa 均测得明显种族偏见**（刻板偏好比约 0.63）。

## 8. 局限与说明（写报告时需注明）

- **flan few8（0.362）不可信**：8 个 demo 超出 flan-t5 512-token 上下文，截断产生伪影；真实可比的是 few4（0.450）。
- **flan-t5 幻觉检测退化（0.270）**：zero-shot + 多子词 verbalizer 下全预测 entailment，属方法局限而非实现 bug（gpt2 同条件下 0.559 正常）。
- **CoT 仅 300 条/划分**（生成式推理慢），结论需谨慎。
- 幻觉检测的两个规则（entailment_factual 与 contradiction_only）在 gpt2 上结果一致（均为 0.559），说明 gpt2 在该任务上几乎不预测 `neutral`。

## 9. 复现

```bash
source .venv/bin/activate   # 自动设置 HF_HOME 到项目 .hf_cache
python -m src.nli_eval --model gpt2 --method zero_shot --split both --out results/nli_gpt2_zero.json
python -m src.nli_eval --model google/flan-t5-base --method few_shot --k 4 --split both --out results/nli_flan_few4.json
python -m src.nli_eval --model google/flan-t5-base --method few_shot --k 4 --verbalizer yes_maybe_no --split matched --out results/nli_flan_few4_ymn.json
python -m src.hallucination --model gpt2 --out results/hallu_gpt2.json
python -m src.crows_pairs --models bert-base-uncased roberta-base --bias-type race-color --n 80 --out results/bias.json
python -m src.report --task all
```

完整跑批脚本见 `results/run_phase1_core.sh`、`run_phase1_ablations.sh`、`run_phase1_part23.sh`。
