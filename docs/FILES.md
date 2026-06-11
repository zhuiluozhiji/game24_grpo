# 文件说明

## 核心代码

| 文件 | 用途 |
| --- | --- |
| `game24/config.py` | 1.5B/通用配置 dataclass，包括模型、LoRA、数据、GRPO、评估参数。 |
| `game24/config_3b.py` | 归档 3B 实验配置参考，不属于正式主线。 |
| `game24/utils.py` | AST 安全表达式求值、答案提取、解合法性验证。 |
| `game24/data.py` | 数据加载、缓存读取、prompt 格式化、ID/OOD/ToT hard/不可解 split 构造，并支持 Countdown。 |
| `game24/eval_analysis.py` | 评估诊断辅助函数，聚合官方 solved-rate/rank 难度分桶。 |
| `game24/rewards.py` | 原始格式奖励和准确率奖励。 |
| `game24/rewards_v2.py` | 后续实验的奖励函数变体。 |
| `game24/train.py` | 旧版 GRPO 训练主逻辑。 |
| `game24/run_experiment.py` | 旧版一键训练 + 评估入口。 |
| `game24/evaluate.py` | 旧版多 split 评估入口。 |
| `game24/sft_warmup.py` | 课程 SFT warm-up，正式主线用于 1.5B，也支持 Countdown。 |
| `game24/grpo_warm.py` | 从 SFT adapter 继续短程 GRPO。 |
| `game24/quick_eval.py` | Greedy 快速评估，支持 24 点、ToT hard split 和 Countdown。 |
| `game24/bestof_eval.py` | Verifier-based best-of-N 评估，支持 24 点、ToT hard split 和 Countdown。 |
| `game24/solver.py` | 枚举生成课程样本表达式，支持任意 target。 |
| `game24/plot_metrics.py` | 从 SFT/GRPO JSON 日志绘制 loss、reward、accuracy 曲线。 |
| `game24/summarize_eval.py` | 汇总评估 JSON，输出 solve rate、format rate、hallucination rate、错误类型计数和难度分桶。 |

## 数据与结果

| 路径 | 用途 |
| --- | --- |
| `dataset_cache/train.json` | `nlile/24-game` 训练缓存，避免复现时必须联网下载数据。 |
| `dataset_cache/game24_official.csv` | `test-time-compute/game-of-24` 官方评估集缓存，用于 official OOD 和 ToT hard 900-1000。 |
| `dataset_cache/countdown_tasks_3to4.parquet` | `Jiayi-Pan/Countdown-Tasks-3to4` 本地缓存，用于远端无法稳定访问 Hugging Face datasets 时复现 Countdown 加分项。 |
| `results/game24-15b-base/` | 正式 1.5B base greedy / best-of-8 评估结果。 |
| `results/game24-sft-15b-curriculum/` | 正式 1.5B SFT warm-up 训练摘要、评估结果和 loss 曲线。 |
| `results/game24-grpo-15b-curriculum/` | 正式 1.5B SFT+GRPO 训练摘要、评估结果和 reward/solve 曲线。 |
| `results/game24-grpo-15b-lr3e-7-s300/` | GRPO 稳定性对照：3e-7 学习率、300 prompts、g8 的评估结果和曲线。 |
| `results/game24-grpo-15b-lr3e-7-s600/` | GRPO 稳定性对照：3e-7 学习率、600 prompts、g8 的评估结果和曲线。 |
| `results/countdown-sft-15b/` | Countdown SFT 训练摘要、greedy/best-of-8 评估结果和 loss 曲线。 |
| `results/countdown-grpo-15b/` | Countdown SFT+GRPO 训练摘要、greedy/best-of-8 评估结果和 reward/solve 曲线。 |
| `results/logs/summary_15b_mainline.log` | 实验4主线汇总指标。 |
| `results/logs/run_15b_mainline.log` | 实验4远端完整运行日志。 |
| `results/logs/run_grpo_ablation.log` | GRPO 超参稳定性对照远端完整运行日志。 |
| `results/logs/run_countdown_bonus.log` | Countdown 加分项远端完整运行日志。 |
| `results/game24-grpo-full/` | 归档 0.5B direct GRPO 预实验结果，不属于正式主线。 |
| `results/game24-sft-3b-curriculum/` | 归档 3B 课程 SFT 预实验结果，不属于正式主线。 |
| `results/game24-grpo-3b-curriculum-short/` | 归档 3B SFT 后短程 GRPO 预实验结果，不属于正式主线。 |

## 文档与脚本

| 文件 | 用途 |
| --- | --- |
| `README.md` | 复现包快速入口。 |
| `docs/runbook.md` | 远端完整实验操作指南，包括命令顺序、日志保存、结果汇总和打包。 |
| `docs/RUNNING.md` | 服务器环境搭建、完整重跑命令、参数解释。 |
| `docs/EXPERIMENTS.md` | 实验设计、结果表、结论。 |
| `docs/experiment_15b_mainline_summary.md` | 实验4：Qwen2.5-1.5B-Instruct 主线远端完整复跑总结。 |
| `docs/FILES.md` | 包内文件说明。 |
| `docs/todo.md` | 后续待办与亮点状态，不放完整运行命令。 |
| `scripts/run_3b_curriculum.sh` | 归档脚本：复现 3B SFT + GRPO 预实验。 |
| `scripts/run_old_05b_grpo.sh` | 归档脚本：复现旧版 0.5B GRPO 预实验。 |
| `scripts/run_15b_mainline.sh` | 复现题目指定 1.5B 主线实验：base、SFT、GRPO、hard split 评估和曲线。 |
| `scripts/run_countdown_bonus.sh` | 复现 Countdown 3-4 数字任意目标加分项。 |
| `scripts/run_ttc_sweep.sh` | 主线跑完后补 best-of-1/4/8/16 verifier-based test-time compute 评估。 |
| `scripts/run_grpo_ablation.sh` | 基于 1.5B SFT adapter 补跑低学习率 GRPO 稳定性对照。 |
| `scripts/run_grpo_ablation.sh` | 主线跑完后补低学习率 GRPO 稳定性对照：`3e-7/s300` 和 `3e-7/s600`。 |
