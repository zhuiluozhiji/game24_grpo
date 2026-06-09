# 24点 GRPO 强化学习复现包

本包包含 24 点游戏强化学习实验的核心代码、数据缓存、轻量结果文件和复现文档。目标是让其他人在服务器 Miniconda 环境中复现以下实验：

1. Qwen2.5-0.5B-Instruct 直接 GRPO 基线。
2. Qwen2.5-3B-Instruct 课程 SFT warm-up。
3. 3B SFT adapter 上的短程 GRPO continuation。
4. Greedy 与 best-of-8 评估。

建议阅读顺序：

1. `docs/RUNNING.md`：从环境创建到完整重跑命令。
2. `docs/EXPERIMENTS.md`：实验目的、结果表、结论。
3. `docs/FILES.md`：代码和结果文件说明。

最快复现 3B 主实验：

```bash
conda create -n game24-grpo python=3.10 -y
conda activate game24-grpo
pip install -r game24/requirements.txt
export PYTHONPATH=$PWD
export QWEN_3B=/path/to/Qwen2.5-3B-Instruct

bash scripts/run_3b_curriculum.sh "$QWEN_3B" 0
```

生成结果会写入 `output/`，交付包自带的历史轻量结果保存在 `results/`。
