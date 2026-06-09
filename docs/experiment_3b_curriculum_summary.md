# 24点游戏 3B 课程冷启动 + GRPO 复跑总结

更新时间：2026-06-07

> 说明：本文档记录的是 3B 历史预实验归档。题目指定主线已经更新为 Qwen2.5-1.5B-Instruct，远端补跑流程见 `docs/runbook.md` 和 `scripts/run_15b_mainline.sh`。该结果不作为正式主实验或模型大小对照。

## 1. 这次解决的问题

上一轮 Qwen2.5-0.5B GRPO 出现了策略塌缩：模型稳定输出 `<answer>24</answer>`，格式正确率 100%，但真实解题率 0%。直接换 Qwen2.5-3B 做 GRPO 也失败了，输出变成无意义重复文本，格式和解题都为 0%。

本轮采用更稳的两阶段方案：

1. 3B 模型课程 SFT 冷启动：先教模型输出“可验证表达式”或 `NO_SOLUTION`。
2. 从 SFT adapter 继续短 GRPO：用验证器奖励对候选表达式做强化学习微调。

## 2. 新增/修改代码

- `game24/solver.py`
  - 新增 Fraction 精确 24 点求解器，用于生成课程数据。
- `game24/sft_warmup.py`
  - 新增 SFT 冷启动训练脚本。
  - 支持合成可解题、合成 OOD 可解题、不可解拒答题。
- `game24/grpo_warm.py`
  - 新增从 SFT adapter 继续 GRPO 的训练脚本。
  - 准确率奖励权重 0.95，格式奖励权重 0.05。
- `game24/quick_eval.py`
  - 新增 greedy 快速评估。
- `game24/bestof_eval.py`
  - 新增 verifier-based best-of-N 评估。
- `game24/utils.py`
  - `extract_answer_from_completion` 支持将 `NO_SOLUTION` / `unsolvable` 识别为无答案，用于幻觉检测。

## 3. 3B 课程 SFT 结果

本地结果目录：

```text
output/game24-sft-3b-curriculum
```

训练配置：

| 项目 | 值 |
| --- | ---: |
| 模型 | Qwen2.5-3B-Instruct |
| 训练样本 | 3175 |
| epoch | 2 |
| step | 6350 |
| 学习率 | 8e-5 |
| 训练时间 | 1514.5 秒 |
| final loss | 0.0803 |

评估结果：

| 评估方式 | ID 解题率 | OOD 解题率 | 不可解幻觉率 | 格式正确率 |
| --- | ---: | ---: | ---: | ---: |
| greedy, n=60 | 10.0% | 26.7% | 58.3% | 100% |
| best-of-8, n=30 | 43.3% | 43.3% | 0.0% | 100% |

## 4. SFT 后短 GRPO 结果

本地结果目录：

```text
output/game24-grpo-3b-curriculum-short
```

训练配置：

| 项目 | 值 |
| --- | ---: |
| 初始化 adapter | `output/game24-sft-3b-curriculum/final_model` |
| GRPO prompt 数 | 50 |
| epoch | 1 |
| num_generations | 4 |
| 学习率 | 8e-7 |
| max_new_tokens | 96 |
| 训练时间 | 734.6 秒 |

评估结果：

| 评估方式 | ID 解题率 | OOD 解题率 | 不可解幻觉率 | 格式正确率 |
| --- | ---: | ---: | ---: | ---: |
| greedy, n=30 | 6.7% | 30.0% | 43.3% | 100% |
| best-of-8, n=30 | 40.0% | 53.3% | 0.0% | 100% |

## 5. 结论

这次已经解决了“小模型只学习输出模板”的核心问题：新模型不再输出固定 `<answer>24</answer>`，而是能生成使用输入数字的表达式，并且能在不可解题上输出 `NO_SOLUTION`。

当前最稳的使用方式是 verifier-based best-of-8：

- 如果重视 ID：SFT adapter 最好，ID best-of-8 为 43.3%。
- 如果重视 OOD：短 GRPO adapter 最好，OOD best-of-8 为 53.3%。
- 两者在 best-of-8 下不可解幻觉率均为 0%。

短 GRPO 没有全面提升 ID，说明继续 RL 时需要更长训练、更高候选数，或加入 reference KL / SFT loss 正则，避免对分布内能力造成轻微退化。不过相较上一轮 0% 解题率与 100% 幻觉率，本轮已经形成可交付的改进实验闭环。
