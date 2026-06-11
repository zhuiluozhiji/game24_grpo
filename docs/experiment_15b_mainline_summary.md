# 24点游戏实验4：1.5B题目主线完整复跑总结

更新时间：2026-06-09

本文档记录正式主线实验：`Qwen2.5-1.5B-Instruct` 的 base eval、SFT warm-up、SFT 后 GRPO continuation，以及 greedy / verifier-based best-of-8 在 ID、official OOD、ToT hard 900-1000 和不可解集上的结果。

## 1. 远端环境

| 项目 | 值 |
| --- | --- |
| 服务器目录 | `/data/ysf/game24_grpo` |
| Conda 环境 | `game24_grpo` |
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 4.51.3 |
| TRL | 0.15.2 |
| GPU | CUDA_VISIBLE_DEVICES=4 |
| 基础模型 | `/data/ysf/models/Qwen2.5-1.5B-Instruct` |
| 模型来源 | ModelScope 下载到用户目录 |

远端账号不能访问 `/data` 下其他用户目录，因此本次没有复用其他人的模型缓存，而是把模型下载到 `/data/ysf/models/Qwen2.5-1.5B-Instruct`。

## 2. 复现命令

```bash
source /data/ysf/miniconda3/etc/profile.d/conda.sh
conda activate game24_grpo
export PYTHONPATH=/data/ysf/game24_grpo

N_EVAL=200 N_HARD=100 BEST_OF=8 GRPO_SAMPLES=300 GRPO_GENERATIONS=8 \
  bash scripts/run_15b_mainline.sh /data/ysf/models/Qwen2.5-1.5B-Instruct 4
```

由于远端不能稳定访问 Hugging Face datasets，官方 `test-time-compute/game-of-24` CSV 已缓存为 `dataset_cache/game24_official.csv`，代码会优先读取该文件。该缓存包含 1362 条官方样本，其中 ToT hard split 使用 indices 900:1000，共 100 条。

## 3. 训练摘要

| 阶段 | 训练样本 | Epoch | Step | 学习率 | 训练时间 | 最后指标 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT warm-up | 3175 | 2 | 6350 | 8e-5 | 1176.0 秒 | final loss 0.0323 |
| SFT+GRPO | 300 prompts | 1 | 300 | 8e-7 | 6633.0 秒 | final avg reward 0.0412 |

SFT 输出目录：`output/game24-sft-15b-curriculum`。

GRPO 输出目录：`output/game24-grpo-15b-curriculum`。

## 4. 主结果

| 模型阶段 | 解码 | ID solve | Official OOD solve | ToT hard solve | 不可解幻觉率 | Format rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1.5B base | greedy | 0.0% | 2.0% | 1.0% | 100.0% | 48%-49% |
| 1.5B base | best-of-8 | 1.5% | 3.0% | 1.0% | 80.0% | 37%-46% |
| 1.5B SFT | greedy | 4.0% | 11.0% | 10.0% | 4.0% | 100.0% |
| 1.5B SFT | best-of-8 | 22.0% | 31.0% | 13.0% | 0.0% | 100.0% |
| 1.5B SFT+GRPO | greedy | 7.5% | 14.5% | 12.0% | 34.0% | 100.0% |
| 1.5B SFT+GRPO | best-of-8 | 27.5% | 43.5% | 32.0% | 1.0% | 100.0% |

结论：SFT 先解决了基础模型格式不稳和不可解题幻觉严重的问题；GRPO continuation 继续提升了可验证表达式的候选池质量。最佳交付配置是 `1.5B SFT+GRPO + best-of-8`，在 official OOD 上达到 43.5%，在 ToT hard 900-1000 上达到 32.0%，不可解幻觉率保持在 1.0%。

## 5. 错误类型观察

`1.5B SFT+GRPO + best-of-8` 的主要剩余错误：

| Split | 主要错误 |
| --- | --- |
| ID | `wrong_value=110`, `refusal_or_no_answer=35` |
| Official OOD | `wrong_value=70`, `refusal_or_no_answer=42`, `number_mismatch=1` |
| ToT hard | `wrong_value=56`, `refusal_or_no_answer=12` |
| Unsolvable | `refused=99`, `wrong_value=1` |

这说明模型已经较少违反数字使用约束，主要瓶颈变成算式搜索质量：输出格式稳定，但仍常给出值不等于 24 的表达式。

## 6. 本地结果文件

关键结果已从远端拉回本地：

| 路径 | 内容 |
| --- | --- |
| `results/game24-15b-base/` | base greedy 与 best-of-8 评估 JSON |
| `results/game24-sft-15b-curriculum/` | SFT metrics、summary、评估 JSON、loss 曲线 |
| `results/game24-grpo-15b-curriculum/` | GRPO metrics、summary、评估 JSON、reward/solve 曲线 |
| `results/logs/run_15b_mainline.log` | 远端完整运行日志 |
| `results/logs/summary_15b_mainline.log` | 汇总指标文本 |

训练曲线：

- `results/game24-sft-15b-curriculum/plots/sft_loss.png`
- `results/game24-grpo-15b-curriculum/plots/grpo_metrics_rewards.png`
- `results/game24-grpo-15b-curriculum/plots/grpo_metrics_solved.png`

远端绘图阶段曾因环境中缺少 `pyparsing` 失败；安装补齐后已重新生成曲线。本地 `game24/requirements.txt` 已显式加入该依赖。

## 7. GRPO 超参稳定性补跑

伙伴新增建议要求补跑低学习率 GRPO 对照，用于观察 RL 训练稳定性。该实验不重跑 SFT，而是从同一个 `output/game24-sft-15b-curriculum/final_model` 继续训练。

| GRPO 配置 | Greedy ID | Greedy OOD | Greedy hard | Greedy unsolvable hallucination | Best-of-8 ID | Best-of-8 OOD | Best-of-8 hard | Best-of-8 unsolvable hallucination |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `8e-7/s300/g8` 主线 | 7.5% | 14.5% | 12.0% | 34.0% | 27.5% | 43.5% | 32.0% | 1.0% |
| `3e-7/s300/g8` | 7.0% | 13.0% | 11.0% | 17.0% | 22.5% | 39.0% | 25.0% | 0.0% |
| `3e-7/s600/g8` | 6.5% | 15.0% | 11.0% | 25.0% | 28.5% | 41.5% | 31.0% | 2.0% |

稳定性结论：降低学习率能缓解 greedy 下不可解样本 hallucination，但没有带来更强的 best-of-8 OOD/hard 表现。`3e-7/s600` 的 ToT hard best-of-8 31.0% 接近主线 32.0%，但 OOD 仍低于主线，且 best-of-8 不可解 hallucination 略高。因此最终主线仍采用 `8e-7/s300/g8`，低学习率两组作为 ablation 支撑“GRPO 超参会影响 hallucination 与候选池质量”的分析。

## 8. Countdown 加分项补跑

伙伴新增建议中的 Countdown 3-4 数字任意目标加分项已完成。远端不能稳定访问 Hugging Face datasets，因此将 `Jiayi-Pan/Countdown-Tasks-3to4` 缓存为 `dataset_cache/countdown_tasks_3to4.parquet`，`game24/data.py` 会优先读取该本地缓存。

```bash
source /data/ysf/miniconda3/etc/profile.d/conda.sh
conda activate game24_grpo
export PYTHONPATH=/data/ysf/game24_grpo

bash scripts/run_countdown_bonus.sh /data/ysf/models/Qwen2.5-1.5B-Instruct 4
```

训练摘要：

| 阶段 | 训练样本 | Epoch | Step | 学习率 | 训练时间 | 最后指标 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Countdown SFT | 3000 | 1 | 3000 | 8e-5 | 569.7 秒 | final loss 0.0483 |
| Countdown SFT+GRPO | 300 prompts | 1 | 300 | 8e-7 | 7443.9 秒 | final avg reward 0.0491 |

| 模型阶段 | 解码 | Countdown solve | Format rate | 主要错误 |
| --- | --- | ---: | ---: | --- |
| 1.5B Countdown SFT | greedy | 10.5% | 100.0% | `wrong_value=179` |
| 1.5B Countdown SFT | best-of-8 | 42.0% | 100.0% | `wrong_value=114`, `invalid_expression=2` |
| 1.5B Countdown SFT+GRPO | greedy | 11.0% | 100.0% | `wrong_value=178` |
| 1.5B Countdown SFT+GRPO | best-of-8 | 40.5% | 100.0% | `wrong_value=119` |

Countdown 结果说明同一套 target-aware prompt、verifier 和 reward 可以迁移到任意目标数任务。best-of-8 显著高于 greedy，说明 verifier-based test-time compute 在 Countdown 上同样有效；但这组小规模 GRPO 未超过 SFT best-of-8，因此最终报告中应把 Countdown 定位为加分项框架迁移验证。
