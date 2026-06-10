# 24点 GRPO 实验记录

## 目标

本项目使用可验证奖励训练语言模型求解 24 点游戏。核心目标不是只生成固定格式，而是输出满足以下条件的表达式：

1. 每个给定数字恰好使用一次。
2. 只使用基本四则运算和括号。
3. 表达式计算结果等于 24。
4. 对不可解样本应输出 `NO_SOLUTION`，避免幻觉式编造。

## 评估设置

评估脚本会构造以下 split：

| Split | 含义 |
| --- | --- |
| ID | 与训练数据同分布的 held-out 可解题。 |
| Official OOD | `test-time-compute/game-of-24` 官方测试集，排除 hard split 后评估。 |
| ToT hard 900-1000 | Tree of Thoughts 论文常用的 100 道难题，取 official dataset indices 900:1000。 |
| Unsolvable | 不可解题，用于检查模型是否乱编表达式。 |
| Countdown | `Jiayi-Pan/Countdown-Tasks-3to4`，3-4 数字凑任意目标数，用于加分项。 |

主要指标：

| 指标 | 含义 |
| --- | --- |
| Solve rate | 表达式合法且等于 24 的比例。 |
| Format rate | 是否符合 `<think>...</think><answer>...</answer>` 格式。 |
| Hallucination rate | 对不可解题仍输出无效表达式的比例，越低越好。 |
| Error counts | `format_error`、`number_mismatch`、`wrong_value`、`invalid_expression`、`refusal_or_no_answer`、`hallucination` 等错误类型计数。 |
| Difficulty buckets | official `game-of-24` 的 `solved_rate` / `rank` 聚合，用于分析 hard split 难度。 |

解码方式：

| 方式 | 含义 |
| --- | --- |
| Greedy | 每题确定性生成一次，直接验证该答案。 |
| Best-of-8 | 每题采样 8 个候选，使用验证器选择第一个正确候选；若不可解题中出现拒答候选，则优先视为拒答。 |
| Best-of-N sweep | 主线跑完后补跑 best-of-1/4/8/16，只改变测试时采样候选数，不改变模型和训练。 |

说明：Countdown 使用同一套验证器，但目标值来自每条样本的 `target` 字段，不固定为 24。

## 正式实验 A：1.5B 题目主线，已完成远端复跑

目的：严格对齐作业要求中的 Backbone 模型 `Qwen2.5-1.5B-Instruct`，形成最终报告主实验。

复现命令见 `docs/runbook.md` 第 4 节，或 `docs/RUNNING.md` 第 4 节。完整复跑记录见 `docs/experiment_15b_mainline_summary.md`。

计划配置：

| 项 | 值 |
| --- | --- |
| 基础模型 | Qwen2.5-1.5B-Instruct |
| 实验阶段 | base eval -> SFT warm-up -> GRPO continuation |
| SFT 学习率 | 8e-5 |
| GRPO 学习率 | 8e-7 |
| GRPO prompt 数 | 默认 300，可用 `GRPO_SAMPLES` 调整 |
| `num_generations` | 默认 8，可用 `GRPO_GENERATIONS` 调整 |
| 评估规模 | ID/OOD/unsolvable 默认 200，ToT hard 默认 100 |

| 模型阶段 | 解码 | ID | Official OOD | ToT hard 900-1000 | Unsolvable 幻觉率 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1.5B base | greedy | 0.0% | 2.0% | 1.0% | 100.0% |
| 1.5B base | best-of-8 | 1.5% | 3.0% | 1.0% | 80.0% |
| 1.5B SFT | greedy | 4.0% | 11.0% | 10.0% | 4.0% |
| 1.5B SFT | best-of-8 | 22.0% | 31.0% | 13.0% | 0.0% |
| 1.5B SFT+GRPO | greedy | 7.5% | 14.5% | 12.0% | 34.0% |
| 1.5B SFT+GRPO | best-of-8 | 27.5% | 43.5% | 32.0% | 1.0% |

结论：最佳主线配置为 `1.5B SFT+GRPO + best-of-8`。相比 SFT best-of-8，Official OOD 从 31.0% 提升到 43.5%，ToT hard 从 13.0% 提升到 32.0%，不可解幻觉率保持在 1.0%。

## 正式实验 B：Countdown 加分项，待远端补跑

目的：参考 TinyZero，将 24 点任务扩展为 3-4 数字凑任意目标数，验证 target-aware prompt、reward 和 verifier 的泛化性。

复现命令见 `docs/runbook.md` 第 6 节，或 `docs/RUNNING.md` 第 5 节。

计划配置：

| 项 | 值 |
| --- | --- |
| 数据集 | `Jiayi-Pan/Countdown-Tasks-3to4` |
| 输入 | 3-4 个数字 + 每条样本自己的 target |
| 输出 | 使用每个数字一次、达到 target 的表达式 |
| 实验阶段 | Countdown SFT -> Countdown GRPO continuation |
| 默认训练样本 | 3000 SFT + 300 GRPO |
| 默认评估规模 | 200 |

结果待补：

| 模型阶段 | 解码 | Countdown solve rate | Format rate |
| --- | --- | ---: | ---: |
| 1.5B Countdown SFT | greedy | 待跑 | 待跑 |
| 1.5B Countdown SFT | best-of-8 | 待跑 | 待跑 |
| 1.5B Countdown SFT+GRPO | greedy | 待跑 | 待跑 |
| 1.5B Countdown SFT+GRPO | best-of-8 | 待跑 | 待跑 |

## 正式实验 C：Verifier-based test-time compute，待远端补跑

目的：控制模型和训练阶段不变，只改变测试时采样候选数，分析验证器能否把候选池中的正确表达式筛出来。这比 0.5B/1.5B/3B 模型大小对照更干净，因为唯一变量是 test-time compute。

复现命令见 `docs/runbook.md` 第 5 节。

计划表格：

| 模型阶段 | 解码 | ID | Official OOD | ToT hard 900-1000 |
| --- | --- | ---: | ---: | ---: |
| 1.5B base | greedy | 待跑 | 待跑 | 待跑 |
| 1.5B base | best-of-1 | 待跑 | 待跑 | 待跑 |
| 1.5B base | best-of-4 | 待跑 | 待跑 | 待跑 |
| 1.5B base | best-of-8 | 待跑 | 待跑 | 待跑 |
| 1.5B base | best-of-16 | 待跑 | 待跑 | 待跑 |
| 1.5B SFT | greedy | 待跑 | 待跑 | 待跑 |
| 1.5B SFT | best-of-1 | 待跑 | 待跑 | 待跑 |
| 1.5B SFT | best-of-4 | 待跑 | 待跑 | 待跑 |
| 1.5B SFT | best-of-8 | 待跑 | 待跑 | 待跑 |
| 1.5B SFT | best-of-16 | 待跑 | 待跑 | 待跑 |
| 1.5B SFT+GRPO | greedy | 待跑 | 待跑 | 待跑 |
| 1.5B SFT+GRPO | best-of-1 | 待跑 | 待跑 | 待跑 |
| 1.5B SFT+GRPO | best-of-4 | 待跑 | 待跑 | 待跑 |
| 1.5B SFT+GRPO | best-of-8 | 待跑 | 待跑 | 待跑 |
| 1.5B SFT+GRPO | best-of-16 | 待跑 | 待跑 | 待跑 |

## 正式实验 D：GRPO 超参稳定性对照，已完成远端补跑

目的：回应 RL 训练不稳定的问题。当前主线 `8e-7/s300/g8` 提升了 OOD 与 hard split 的 best-of-8 solve rate，但 greedy 下不可解幻觉率升高。因此补两组低学习率对照，观察是否能在保住 OOD/hard 提升的同时降低 hallucination。

复现命令见 `docs/runbook.md` 第 4.1 节。

| GRPO 配置 | Greedy ID | Greedy OOD | Greedy hard | Greedy unsolvable hallucination | Best-of-8 ID | Best-of-8 OOD | Best-of-8 hard | Best-of-8 unsolvable hallucination |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `8e-7/s300/g8` 当前主线 | 7.5% | 14.5% | 12.0% | 34.0% | 27.5% | 43.5% | 32.0% | 1.0% |
| `3e-7/s300/g8` | 7.0% | 13.0% | 11.0% | 17.0% | 22.5% | 39.0% | 25.0% | 0.0% |
| `3e-7/s600/g8` | 6.5% | 15.0% | 11.0% | 25.0% | 28.5% | 41.5% | 31.0% | 2.0% |

结论：低学习率对照确实降低了 greedy 不可解幻觉率，`3e-7/s300` 从主线 34.0% 降到 17.0%，`3e-7/s600` 降到 25.0%。但主线 `8e-7/s300/g8` 仍有最高的 Official OOD best-of-8 43.5% 与 ToT hard best-of-8 32.0%。`3e-7/s600` 的 hard best-of-8 31.0% 接近主线，但不可解 best-of-8 hallucination 为 2.0%，略高于主线 1.0%。因此最终主结果仍建议保留 `8e-7/s300/g8`，低学习率结果作为稳定性对照写入分析。

## 正式分析 E：Hard split 难度与错误类型

评估 JSON 会保留并汇总 official `game-of-24` 中的 `rank` 与 `solved_rate`。最终报告建议补两张分析表：

1. 普通 official OOD 与 ToT hard 900-1000 的平均官方 `solved_rate`、平均 `rank`、模型 solve rate。
2. 按官方 `solved_rate` 分桶：`0-25% / 25-50% / 50-75% / 75-100%`，观察模型是否在低 solved-rate 题目上更容易失败。

错误类型用 `error_counts` 汇总，重点解释 `wrong_value`、`number_mismatch`、`format_error` 和 unsolvable 上的 `hallucination`。

## 历史预实验归档

下面结果保留用于记录探索过程，不作为正式主实验，也不作为严格模型大小对照。原因是 0.5B、1.5B、3B 的训练协议不同，不能把差异归因于模型规模。

### 归档 A：0.5B 直接 GRPO

目的：验证最小模型直接使用格式奖励 + 准确率奖励是否能学会 24 点。

复现脚本仍保留为 `scripts/run_old_05b_grpo.sh`，但不建议作为最终主线运行。

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

### 归档 B：3B 课程 SFT

目的：解决“小模型只学到输出模板”的问题。先用可验证表达式训练模型生成真实表达式，并加入不可解样本训练拒答行为。

复现脚本仍保留为 `scripts/run_3b_curriculum.sh`，但不建议作为最终主线运行。

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

### 归档 C：3B SFT 后短程 GRPO

目的：在 SFT adapter 上继续做短程 GRPO，观察可验证奖励是否进一步提升候选质量。

复现脚本仍保留为 `scripts/run_3b_curriculum.sh`，但不建议作为最终主线运行。

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

## 新增代码功能

- `game24/utils.py`
  - prompt 和 validator 支持任意 target。
- `game24/solver.py`
  - 新增通用 `solve_target`，保留 `solve_24` 兼容旧代码。
- `game24/data.py`
  - 支持 official `game-of-24`、ToT hard split 900:1000、Countdown 数据标准化。
- `game24/rewards.py`
  - accuracy reward 支持每条样本独立 target。
- `game24/quick_eval.py` / `game24/bestof_eval.py`
  - 支持 base model eval、hard split、Countdown、错误类型统计和 official 难度聚合。
- `game24/eval_analysis.py`
  - 聚合官方 `rank`、`solved_rate` 和 solved-rate 分桶。
- `game24/plot_metrics.py`
  - 从 metrics JSON 生成训练曲线。
- `game24/summarize_eval.py`
  - 汇总评估 JSON，方便填最终报告表格。

## 交付包中的结果文件

本复现包保留轻量 JSON 结果，便于核对数值：

- `results/game24-grpo-full/`
- `results/game24-sft-3b-curriculum/`
- `results/game24-grpo-3b-curriculum-short/`
- 新实验补跑后会写入 `output/game24-15b-*` 与 `output/countdown-*`

完整 LoRA adapter 和 tokenizer 大文件未放入 zip，可按 `docs/RUNNING.md` 中命令重新生成到对应 `output/` 目录。
