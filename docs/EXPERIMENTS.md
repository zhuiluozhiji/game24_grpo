# 24点 GRPO 实验记录

## 目标

本项目使用可验证奖励训练语言模型求解 24 点游戏。核心目标不是只生成固定格式，而是输出满足以下条件的表达式：

1. 每个给定数字恰好使用一次。
2. 只使用基本四则运算和括号。
3. 表达式计算结果等于 24。
4. 对不可解样本应输出 `NO_SOLUTION`，避免幻觉式编造。

## 评估设置

评估脚本会构造三个 split：

| Split | 含义 |
| --- | --- |
| ID | 与训练数据同分布的 held-out 可解题。 |
| OOD | 数字范围或组合方式更偏离训练分布的可解题。 |
| Unsolvable | 不可解题，用于检查模型是否乱编表达式。 |

主要指标：

| 指标 | 含义 |
| --- | --- |
| Solve rate | 表达式合法且等于 24 的比例。 |
| Format rate | 是否符合 `<think>...</think><answer>...</answer>` 格式。 |
| Hallucination rate | 对不可解题仍输出无效表达式的比例，越低越好。 |

解码方式：

| 方式 | 含义 |
| --- | --- |
| Greedy | 每题确定性生成一次，直接验证该答案。 |
| Best-of-8 | 每题采样 8 个候选，使用验证器选择第一个正确候选；若不可解题中出现拒答候选，则优先视为拒答。 |

## 实验 A：0.5B 直接 GRPO

目的：验证最小模型直接使用格式奖励 + 准确率奖励是否能学会 24 点。

复现命令见 `docs/RUNNING.md` 第 4 节。

原始配置：

| 项 | 值 |
| --- | --- |
| 基础模型 | Qwen2.5-0.5B-Instruct |
| LoRA rank | 16 |
| 训练样本 | 1089 |
| Epochs | 3 |
| Learning rate | 2e-5 |
| GRPO `num_generations` | 4 |
| GRPO `beta` | 0.04 |
| 训练耗时 | 17734.04 秒 |
| 输出目录 | `output/game24-grpo-full` |

结果摘要：

| Split | Solve rate | Format rate | Hallucination rate |
| --- | ---: | ---: | ---: |
| ID | 0.0% | 高 | - |
| OOD | 0.0% | 高 | - |
| Unsolvable | 0.0% | 高 | 100.0% |

结论：0.5B 直接 GRPO 主要学到了输出标签模板，无法稳定学到算术搜索策略。这个失败结果促成后续改动：换更大模型，并加入 SFT warm-up。

## 实验 B：3B 课程 SFT

目的：解决“小模型只学到输出模板”的问题。先用可验证表达式训练模型生成真实表达式，并加入不可解样本训练拒答行为。

复现命令见 `docs/RUNNING.md` 第 5 节。

原始配置：

| 项 | 值 |
| --- | --- |
| 基础模型 | Qwen2.5-3B-Instruct |
| 训练样本 | 3175 |
| Epochs | 2 |
| Learning rate | 8e-5 |
| Synthetic curriculum | 开启 |
| Unsolvable limit | 300 |
| 训练耗时 | 1514.48 秒 |
| 输出目录 | `output/game24-sft-3b-curriculum` |

Greedy 评估，输出文件 `output/game24-sft-3b-curriculum/quick_eval_60.json`：

| Split | N | Solve rate | Format rate | Hallucination rate |
| --- | ---: | ---: | ---: | ---: |
| ID | 60 | 10.0% | 100.0% | - |
| OOD | 60 | 26.7% | 100.0% | - |
| Unsolvable | 60 | 0.0% | 100.0% | 58.3% |

Best-of-8 评估，输出文件 `output/game24-sft-3b-curriculum/bestof8_eval_30.json`：

| Split | N | Solve rate | Format rate | Hallucination rate |
| --- | ---: | ---: | ---: | ---: |
| ID | 30 | 43.3% | 100.0% | - |
| OOD | 30 | 43.3% | 100.0% | - |
| Unsolvable | 30 | 0.0% | 100.0% | 0.0% |

结论：3B + 课程 SFT 明显突破了模板坍塌。Greedy 已能解出一部分题，best-of-8 说明模型候选中经常包含正确表达式；不可解题在 best-of-8 设置下能找到拒答候选，幻觉率降到 0。

## 实验 C：3B SFT 后短程 GRPO

目的：在 SFT adapter 上继续做短程 GRPO，观察可验证奖励是否进一步提升候选质量。

复现命令见 `docs/RUNNING.md` 第 6 节。

原始配置：

| 项 | 值 |
| --- | --- |
| 基础模型 | Qwen2.5-3B-Instruct |
| 初始 adapter | `output/game24-sft-3b-curriculum/final_model` |
| 训练样本 | 50 |
| Epochs | 1 |
| Learning rate | 8e-7 |
| `num_generations` | 4 |
| `max_new_tokens` | 96 |
| 训练耗时 | 734.58 秒 |
| 输出目录 | `output/game24-grpo-3b-curriculum-short` |

Greedy 评估，输出文件 `output/game24-grpo-3b-curriculum-short/quick_eval_30.json`：

| Split | N | Solve rate | Format rate | Hallucination rate |
| --- | ---: | ---: | ---: | ---: |
| ID | 30 | 6.7% | 100.0% | - |
| OOD | 30 | 30.0% | 100.0% | - |
| Unsolvable | 30 | 0.0% | 100.0% | 43.3% |

Best-of-8 评估，输出文件 `output/game24-grpo-3b-curriculum-short/bestof8_eval_30.json`：

| Split | N | Solve rate | Format rate | Hallucination rate |
| --- | ---: | ---: | ---: | ---: |
| ID | 30 | 40.0% | 100.0% | - |
| OOD | 30 | 53.3% | 100.0% | - |
| Unsolvable | 30 | 0.0% | 100.0% | 0.0% |

结论：短程 GRPO 没有稳定提升 greedy，但提升了 OOD best-of-8，从 43.3% 到 53.3%。这说明当前 GRPO 主要改善候选池覆盖率，而不是单次确定性输出。若继续实验，建议扩大 GRPO 样本数并加入更强的 reference/KL 控制，防止 SFT 能力被短程更新扰动。

## 交付包中的结果文件

本复现包保留轻量 JSON 结果，便于核对数值：

- `results/game24-grpo-full/`
- `results/game24-sft-3b-curriculum/`
- `results/game24-grpo-3b-curriculum-short/`

完整 LoRA adapter 和 tokenizer 大文件未放入 zip，可按 `docs/RUNNING.md` 中命令重新生成到对应 `output/` 目录。
