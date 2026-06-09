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
