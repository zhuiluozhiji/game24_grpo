# 文件说明

## 核心代码

| 文件 | 用途 |
| --- | --- |
| `game24/config.py` | 0.5B/通用配置 dataclass，包括模型、LoRA、数据、GRPO、评估参数。 |
| `game24/config_3b.py` | 3B 实验配置参考。 |
| `game24/utils.py` | AST 安全表达式求值、答案提取、解合法性验证。 |
| `game24/data.py` | 数据加载、缓存读取、prompt 格式化、ID/OOD/不可解 split 构造。 |
| `game24/rewards.py` | 原始格式奖励和准确率奖励。 |
| `game24/rewards_v2.py` | 后续实验的奖励函数变体。 |
| `game24/train.py` | 旧版 GRPO 训练主逻辑。 |
| `game24/run_experiment.py` | 旧版一键训练 + 评估入口。 |
| `game24/evaluate.py` | 旧版多 split 评估入口。 |
| `game24/sft_warmup.py` | 3B 课程 SFT warm-up，用于解决模板坍塌。 |
| `game24/grpo_warm.py` | 从 SFT adapter 继续短程 GRPO。 |
| `game24/quick_eval.py` | Greedy 快速评估。 |
| `game24/bestof_eval.py` | Verifier-based best-of-N 评估。 |
| `game24/solver.py` | 枚举生成课程样本表达式。 |

## 数据与结果

| 路径 | 用途 |
| --- | --- |
| `dataset_cache/train.json` | `nlile/24-game` 训练缓存，避免复现时必须联网下载数据。 |
| `results/game24-grpo-full/` | 0.5B 直接 GRPO 的轻量结果 JSON。 |
| `results/game24-sft-3b-curriculum/` | 3B 课程 SFT 的 summary 和评估 JSON。 |
| `results/game24-grpo-3b-curriculum-short/` | 3B SFT 后短程 GRPO 的 summary 和评估 JSON。 |

## 文档与脚本

| 文件 | 用途 |
| --- | --- |
| `README.md` | 复现包快速入口。 |
| `docs/RUNNING.md` | 服务器环境搭建、完整重跑命令、参数解释。 |
| `docs/EXPERIMENTS.md` | 实验设计、结果表、结论。 |
| `docs/FILES.md` | 包内文件说明。 |
| `scripts/run_3b_curriculum.sh` | 复现 3B SFT + GRPO + 评估的 shell 脚本。 |
| `scripts/run_old_05b_grpo.sh` | 复现旧版 0.5B GRPO 基线的 shell 脚本。 |
