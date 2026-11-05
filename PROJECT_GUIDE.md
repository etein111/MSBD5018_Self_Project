# MSBD5018 Individual Project — 项目完成指南

> 学号（报告/代码命名用）：`21317834`（提交前请再核对一次）
> 截止：**2026-11-29 23:59**（Canvas，迟交每天 -20%）
> 本周范围：**Part 2.1 的 Prompting 阶段 + Part 2.2 + Part 3**（Fine-tuning 见附录 Phase 2）

---

## 0. 项目一句话

评估语言模型（LM）：① 用 ≥2 个本地 LM 做 NLI 三分类，并下放到幻觉检测；② 用 Minimal Pairs + 伪对数似然测 ≥2 个掩码 LM 的 race 偏见。产出 = PDF 报告 + 代码 zip。

评分看：**实现正确性 / 表述清晰 / 覆盖完整 / 洞察**。**不看性能高低、不看模型大小**，只要结果在合理区间即可。

---

## 1. 既定决策（本次已拍板）

| 项 | 决策 |
|---|---|
| 2.1 总路线 | Prompting + Fine-tuning 都做；**本周只做 Prompting** |
| 2.1 模型 | **GPT-2**（因果 LM，`gpt2`，124M）+ **Flan-T5-base**（seq2seq，`google/flan-t5-base`，250M） |
| 2.1 Prompt 方法对比 | zero-shot / few-shot / CoT / 更新的 prompt engineering（详见 §5） |
| 2.2 幻觉检测 | 复用上面两个 NLI 模型，wiki_bio_gpt3_hallucination，报告 acc/precision/recall/F1 |
| 3 社会群体 | **race**（CrowS-Pairs 的 `race-color` 领域） |
| 3 掩码 LM | **BERT-base-uncased**（110M）+ **RoBERTa-base**（125M） |

---

## 2. 技术栈 & 环境搭建（Day 1 上午）

- 机器：Apple M5 / 24GB RAM；PyTorch 走 **MPS**。
- Python：新建独立 venv（系统 3.9.6 可用，有更高版本更好）。

```bash
cd /Users/chuan/Documents/HKUST/Academic/Courses/5018/Self_Project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers datasets evaluate accelerate sentencepiece scikit-learn numpy pandas
# 验证 MPS 可用：
python -c "import torch; print(torch.backends.mps.is_available())"   # 期望 True
```

模型 ID（本地部署，`transformers` 首次会下载权重）：

| 用途 | 模型 | 加载方式 |
|---|---|---|
| NLI 因果 | `gpt2` | `AutoModelForCausalLM.from_pretrained("gpt2")` |
| NLI seq2seq | `google/flan-t5-base` | `AutoModelForSeq2SeqLM.from_pretrained(...)` |
| 偏见掩码 1 | `bert-base-uncased` | `AutoModelForMaskedLM.from_pretrained(...)` |
| 偏见掩码 2 | `roberta-base` | `AutoModelForMaskedLM.from_pretrained(...)` |

数据集（`datasets` 拉取）：

| 数据集 | 加载 | 用途 |
|---|---|---|
| MNLI 评测（已给） | 本地 `data/multi_nli/*.jsonl` | 2.1 主评测 |
| MNLI train 切片 | `load_dataset("multi_nli", split="train[:64]")`（或 `nyu-mll/multi_nli`） | few-shot 的 demo 来源 |
| 幻觉检测 | `load_dataset("potsawee/wiki_bio_gpt3_hallucination")` | 2.2 |
| CrowS-Pairs | `load_dataset("nyu-mll/crows_pairs")`（旧名 `crows_pairs`） | 3 |

> 提示：用 `split="train[:N]"` 切片只拉前 N 条，避免下载完整 39 万条 train。

---

## 3. 建议代码仓库结构

```
Self_Project/
├── .venv/
├── data/multi_nli/                 # 已给数据
├── src/
│   ├── data.py                     # 统一加载 MNLI jsonl → DataFrame，剔除 gold_label="-"
│   ├── prompts.py                  # 所有 prompt 模板 + verbalizer 定义
│   ├── scorer.py                   # GPT-2 / Flan-T5 打分器（把模型输出→标签+概率）
│   ├── nli_eval.py                 # 2.1 主评测入口（方法×模型×集合 跑分）
│   ├── hallucination.py            # 2.2 幻觉检测
│   ├── crows_pairs.py              # 3 偏见：伪对数似然 + 评测
│   └── utils.py                    # 指标、日志、结果表导出
├── results/                        # 结果 JSON/CSV/表
├── report/                         # LaTeX 报告
└── PROJECT_GUIDE.md
```

---

## 4. 模块拆解（含验收标准）

### Module 0 — 数据基建（Day 1）
**目标**：能加载、清洗、预览所有数据。
- `data.py`：读两个 jsonl → DataFrame，**剔除 `gold_label=="-"`**（76 条，无共识标注），剩余约 4924 条。
- 统一标签：entailment / neutral / contradiction → 0/1/2。
- 拉取并本地缓存 MNLI train 切片、wiki_bio、crows_pairs，打印各自字段与样例。
- **验收**：打印 MNLI 各标签计数、各 genre 计数；能打印 wiki_bio 一条（premise/gpt3 句/annotation）；能打印 crows_pairs race 领域一条句子对。

### Module A — 统一 NLI 评测框架（Day 2）
**目标**：一个 harness，能对任意 (premise, hypothesis) 输出预测标签 + 概率。
- `prompts.py`：模板接口 `build_prompt(premise, hyp, method, demos=None)`。
  - **zero-shot**：`Premise: {p}\nHypothesis: {h}\nThe relationship is:`（或指令式模板，Flan-T5 用指令式，GPT-2 用补全式）。
  - **few-shot**：前面拼 K 个 `(premise, hypothesis, label)` 示例 + 当前问题。
  - **CoT**：模板加 `Let's think step by step.`，从生成文本里正则/关键词解析最终标签，解析失败回退 verbalizer。
- `scorer.py`：
  - **GPT-2（因果）**：把 prompt 拼完整，对三个 label token 分别算 `log P(token | prompt)`（取 logits 的 log-softmax），`argmax`。
  - **Flan-T5（seq2seq）**：`generate()` 首 token 的 label 打分，或直接取 `outputs.logits[0]` 的 verbalizer 得分。
- **verbalizer** 做成可配置：默认 `{entailment:"entailment", neutral:"neutral", contradiction:"contradiction"}`，可切到 `{true/false/neither}` 或 `{A/B/C}` 做消融。
- **验收**：对 10 条样本，2 模型 × 至少 zero-shot 能输出标签+概率，且标签在 {0,1,2} 内。

### Module B — 2.1 实验：GPT-2 vs Flan-T5 的多方法对比（Day 3–4）
**目标**：产出「2 模型 × 4~5 方法 × 2 集合(matched/mismatched)」准确率表。
- 跑 **zero-shot 全量 5000**（主结果，两个集合分别报 accuracy）。
- 跑 **few-shot (k=4 或 8)**：demo 取自 MNLI train 切片，评测集同 5000。
- 跑 **CoT**：可先跑子集（如 500 条）控制时间，报告里注明子集规模。
- 跑「更新的 prompt engineering」选 1–2 个（见 §5 菜单）。
- **验收**：得到一张完整结果表；每格 accuracy 处于该模型的"合理区间"（GPT-2 零样本可能 33%–50%，Flan-T5 零样本可能 50%–75%，**低不扣分**）；至少有 1 个跨模型/跨方法案例。

### Module C — 2.2 幻觉检测（Day 5）
**目标**：把 NLI 模型零改动复用，报告 sentence-level 四指标。
- 加载 wiki_bio_gpt3_hallucination，展开成 (premise=wiki_bio_text, hypothesis=每句 gpt3_sentence, annotation) 的句子级样本。
- annotation 映射二分类：`0 → Factual`，`0.5 和 1 → Non-Factual`。
- 用 NLI 模型（至少 zero-shot 的两个，最好加上 few-shot 最优配置）打分：`entailment → Factual`，`contradiction/neutral → Non-Factual`（可实验不同映射阈值，如只用 contradiction→Non-Factual）。
- 算 **accuracy / precision / recall / F1**（`sklearn.metrics`）。
- **验收**：输出 2 模型 × 各映射规则的指标表；至少 1 个"检测对了/错了"的案例。

### Module D — 3 偏见评测（Day 6）
**目标**：算 BERT / RoBERTa 在 race 领域的刻板比例。
- 加载 crows_pairs，按 `bias_type=="race-color"` 过滤，**抽样 80 对**（固定随机种子保证可复现）。
- 实现 **pseudo-log-likelihood**（见 §6 关键细节）：
  - 对每句：把被 mask 的位置替换为 `[MASK]`（BERT）或 `<mask>`（RoBERTa），喂给掩码 LM，取 mask 位置的 logits，对目标 token 取 log-softmax 得分，多个 mask token 累加。
  - 目标词若被切成多个 subword，用「第一个 subword」或「求和」策略（任选并写进报告）。
- 对每对 (stereotypical, antistereotypical)，比较两句 PLL，统计「刻板句得分更高」的比例。
- **验收**：两模型各出一个比例（理想 50%，偏见模型通常 >50%，如 55%–70%）；3 个句子对做成 table/figure。

### Module E — 报告 + 打包 + 提交（Day 7）
**目标**：对照 rubric 出报告。
- LaTeX 报告结构（对应 rubric 的 coverage）：
  - **Setup**：方法（2.1 各 prompt 方法 + 2.2 映射 + 3 伪似然）、实现细节（模型尺寸、prompt 全文、超参）、实验细节（数据切分、标签映射、抽样）。
  - **Results**：2.1 两集合定量表、2.2 四指标表、3 两模型偏见比例；每部分至少 1 个 case study（含输入/输出原文）。
  - **Insights**：GPT-2 vs Flan-T5 的 prompt-following 差异、CoT 对小模型的作用、verbalizer 选择的影响、跨域(matched vs mismatched)掉点、race 偏见来源等。
- 命名：`individual 21317834.pdf`；代码打包 `individual 21317834.zip`（剔除 `.venv/`、模型权重、`results/` 大文件，保留 `src/` + `requirements.txt` + `README.md`）。
- **验收**：PDF 编译通过；zip 内代码可复现结果。

---

## 5. Prompt Engineering 全景目录（2.1）

前沿 prompt engineering 分两大类：**(A) 单步分类可直接实测**（适合 NLI + 小模型）与 **(B) 面向 LLM/多步推理**（NLI 上要么收益有限、要么需大模型，放到报告里做概念对比与 Insight）。

### A. 单步分类可实测（能在 GPT-2 / Flan-T5 上跑）

| 方法 | 出处 | 一句话做法 | 适配度 |
|---|---|---|---|
| **Chain-of-Thought (CoT)** | Wei et al., NeurIPS 2022 | 模板加 `Let's think step by step.`，解析最终标签 | ★★★ 必测 |
| **Self-Consistency** | Wang et al., ICLR 2023 | CoT 采样多次（temperature>0）取多数票 | ★★★ 必测 |
| **Auto-CoT** | Zhang et al., ICLR 2023 | 自动聚类示例、自动生成 CoT demo（替代手写 few-shot demo） | ★★☆ 前沿加分 |
| **Active-Prompt** | Diao et al., ACL 2024 | 按模型不确定性选最有价值的 CoT demo | ★★☆ 与 Auto-CoT 二选一 |
| **Verbalizer 消融** | 经典 | label 词 vs `true/false/neither` vs `A/B/C` vs 数字 | ★★★ 零成本 |
| **EmotionPrompt / "Take a deep breath"** | Li et al. 2023；Yang et al. (OPRO), ICLR 2024 | 前缀加一句情绪/深呼吸式引导 | ★★☆ 一行即可 |
| **Rephrase and Respond (RaR)** | Deng et al., 2023 | 让模型先改写/澄清问题，再作答 | ★★☆ 歧义假设句有效 |
| **Least-to-Most** | Zhou et al., ICLR 2023 | 把问题拆成子问题递进求解 | ★☆☆ NLI 单步，弱适配 |
| **Step-Back Prompting** | Zheng et al., ICLR 2024 | 先抽象到更高层面再回答 | ★☆☆ NLI 上偏弱 |
| **DSPy（程序化 prompt 自动调优）** | Khattab et al., ICLR 2024 | 用优化器自动搜 verbalizer/prompt，而非手写 | ★★☆ 工程成本高，做加分项 |
| **Chain-of-Verification (CoVe)** | Dhuliawala et al., 2023 | 生成后自校验再给结论 | ★☆☆ 偏生成任务 |

### B. 面向 LLM / 多步推理（报告做概念对比，不强制实现）

| 方法 | 出处 | 为何不直接跑 |
|---|---|---|
| **Tree of Thoughts (ToT)** | Yao et al., NeurIPS 2023 | 树状多路径搜索，需大模型 + 多步任务，NLI 单步用不上 |
| **Graph of Thoughts (GoT)** | Besta et al., AAAI 2024 | 图结构推理，同上 |
| **ReAct** | Yao et al., ICLR 2023 | 推理+行动交替，需外部工具/环境 |
| **Reflexion** | Shinn et al., NeurIPS 2023 | 多轮自我反思强化，需能稳定长生成的 LLM |
| **Self-Refine** | Madaan et al., NeurIPS 2023 | 迭代自我修正，小模型易发散 |

> **报告策略（正好踩中 rubric 的 coverage + insights）**：实测 A 类里 4–5 个，再用一段专门讨论「为什么 B 类（ToT/ReAct/Reflexion）不适用于单步 NLI 或需要更强 LLM」——这是比只跑实验更高分的写法。

### ✅ 最终筛选（可做 + 值得做，按性价比排序）

**Tier 1 — 必做（全量或 500 条子集）**

| 方法 | 为什么值得 | 成本 |
|---|---|---|
| Zero-shot | 基线主结果 | 低 |
| Few-shot (k=4) | 测 ICL，GPT-2 收益最明显 | 低 |
| CoT | 你点名要 + 最有名前沿技术；"CoT 帮 LLM 却不帮小模型"是强洞察 | 中（需生成+解析+回退） |
| Verbalizer 消融 | 直击课程学习目标"评估对设计选择敏感"（label 词 vs true/false vs A/B/C） | 极低 |
| "Take a deep breath" / EmotionPrompt | 一行前缀，纯赚 coverage + 可写洞察 | 极低 |

**Tier 2 — 建议加分（只在 CoT 子集上做，控制时间）**

| 方法 | 为什么值得 | 成本 |
|---|---|---|
| Self-Consistency | 2023 前沿；配 CoT 采样 3~5 次投票 | =CoT × 采样数 |
| Role prompting | 一行，仅 Flan-T5 有效、GPT-2 无效（本身是洞察） | 极低 |
| DSPy（可选） | 程序化自动调 prompt，时间不够就写进"未来工作" | 高 |

### ❌ 跳过清单 + 理由（报告里一句带过即可）

| 方法 | 跳过理由 |
|---|---|
| Auto-CoT / Active-Prompt | 需额外 embedding 模型 + 聚类/不确定性估计；小模型自动生成的 demo 质量差，性价比低 |
| RaR / Least-to-Most / Step-Back / CoVe | 面向多步 / 生成 / 数学任务，NLI 单步三分类用不上 |
| ToT / GoT / ReAct / Reflexion / Self-Refine | 面向 LLM / 多步 agent，小模型跑不动也无意义 |

---

## 6. 关键实现细节 / 坑（务必注意）

1. **剔除 `gold_label=="-"`**：matched 40 条 + mismatched 36 条，共 76 条无金标准，评估前必须过滤。
2. **few-shot demo 不能取自评测集**：从 MNLI `train` 切片取（如 `train[:16]`），避免评测污染。HF 的 label 是 0/1/2 = entailment/neutral/contradiction，注意与 jsonl 的字符串标签对齐。
3. **GPT-2 是裸模型**：zero-shot 要用"补全式"模板，让模型对三个 label token 打概率分（`P(label | prompt)`），不要期待它像 Flan-T5 一样"听指令"。
4. **CoT 解析与回退**：从生成文本里用关键词/正则抽最后结论；抽不到就回退到非 CoT verbalizer 打分，并在报告里说明回退率。
5. **伪对数似然实现**：
   - mask token：BERT=`[MASK]`，RoBERTa=`<mask>`（`tokenizer.mask_token` 自动给）。
   - 对每句逐个 mask 位置取 log-softmax，累加；目标词多 subword 时统一策略（首 token 或求和），写进报告即可。
   - 分数 =「刻板句 PLL > 反刻板句 PLL 的对数」占全部对的百分比。
6. **race 领域筛选**：crows_pairs 字段为 `bias_type`，race/color 对应值 `"race-color"`；抽样 80 对用固定 `random_seed`。
7. **固定随机种子 + 记录版本**：所有抽样、采样都 `seed`，报告写 `transformers/torch` 版本，保证可复现。
8. **性能合理区间**：MNLI 随机猜 ≈33%；GPT-2 零样本可能仅略高于随机、Flan-T5 零样本 50%+，都属于"合理"，报告里别回避，写成模型差异的洞察即可。

---

## 7. 7 天排期

| 天 | 模块 | 产出 |
|---|---|---|
| **Day 1** | Module 0 | 环境 OK、四份数据可加载、清洗逻辑写好 |
| **Day 2** | Module A | harness 跑通，GPT-2 + Flan-T5 zero-shot 能出结果 |
| **Day 3** | Module B（GPT-2） | zero/few/CoT + 前沿(self-consistency/auto-CoT/emotion/verbalizer 消融) 跑完，matched+mismatched 准确率 |
| **Day 4** | Module B（Flan-T5）+ 汇总 | 同样全套跑完，两模型结果表、跨模型案例 |
| **Day 5** | Module C | 幻觉检测四指标 + 案例 |
| **Day 6** | Module D | BERT/RoBERTa 偏见比例 + 3 对案例 |
| **Day 7** | Module E | 报告 PDF + 代码 zip + 提交（留 buffer 处理返工） |

> 缓冲策略：若 Day 1 的 MPS/依赖装环境卡住，顺延半天，把 Module B 的 CoT 缩到 500 条子集即可追回。

---

## 8. 报告 checklist（对照 rubric）

- [ ] Correctness：实现描述准确、公式（verbalizer argmax、伪对数似然）写对、剔除 "-" 有说明。
- [ ] Coverage：2.1（方法+两集合定量+案例）、2.2（四指标+案例）、3（指标+3 对案例）齐全；引用 MNLI/SNLI/CrowS-Pairs/SelfCheckGPT 等文献。
- [ ] Clearness：每个实验有「数据→方法→输出映射→结果」完整链路。
- [ ] Insights：至少 3 条独到分析（prompt 方法差异 / 跨域掉点 / 模型 prompt-following 能力 / race 偏见来源）。

---

## 附录：Phase 2 — Fine-tuning（本周之后）

- 数据：`load_dataset("multi_nli", split="train")`（或 GLUE `mnli`），自己微调、**不得用他人微调 checkpoint**。
- 模型：可选 BERT/RoBERTa 加分类头，或沿用 GPT-2/Flan-T5 微调，在同样 5000 条评测。
- 对比：把 Phase 1 的 prompt 结果与 Phase 2 的 fine-tune 结果放同一张表，写「prompt vs fine-tune」的洞察。
- 注意：Phase 2 若与 2.2/3 共用代码，务必版本化，避免破坏 Phase 1 可复现结果。
