# 24点 GRPO 强化学习复现包

本包包含 24 点游戏强化学习实验的核心代码、数据缓存、轻量结果文件和复现文档。正式实验主线只保留题目指定的 Qwen2.5-1.5B-Instruct，并补充 Countdown 加分项。

1. Qwen2.5-1.5B-Instruct 主线：base / SFT warm-up / GRPO continuation。
2. Official game-of-24 OOD 与 ToT hard split 900-1000 评估。
3. Countdown 3-4 数字任意目标加分项。
4. Greedy、best-of-N sweep、错误类型统计、hard split 难度分桶与训练曲线绘图。

早期 0.5B direct GRPO 和 3B SFT/GRPO 结果保留在 `results/` 中作为历史预实验归档，不作为正式主实验或模型大小对照。

建议阅读顺序：

1. `docs/runbook.md`：远端完整实验操作指南。
2. `docs/RUNNING.md`：从环境创建到完整重跑命令。
3. `docs/experiment_15b_mainline_summary.md`：实验4（1.5B题目主线）完整远端复跑结果。
4. `docs/EXPERIMENTS.md`：实验目的、结果表、结论。
5. `docs/todo.md`：后续待办与亮点状态。
6. `docs/FILES.md`：代码和结果文件说明。

最快复现 1.5B 主线：

```bash
conda create -n game24-grpo python=3.10 -y
conda activate game24-grpo
pip install -r game24/requirements.txt
export PYTHONPATH=$PWD
export QWEN_15B=/path/to/Qwen2.5-1.5B-Instruct

bash scripts/run_15b_mainline.sh "$QWEN_15B" 0
```

生成结果会写入 `output/`，交付包自带的历史轻量结果保存在 `results/`。

Countdown 加分项：

```bash
bash scripts/run_countdown_bonus.sh "$QWEN_15B" 0
```

已归档的 Countdown 结果保存在 `results/countdown-sft-15b/` 和 `results/countdown-grpo-15b/`。在 200 条测试样本上，SFT greedy 为 10.5%，SFT best-of-8 为 42.0%；SFT+GRPO greedy 为 11.0%，SFT+GRPO best-of-8 为 40.5%。远端无法稳定联网时会优先读取 `dataset_cache/countdown_tasks_3to4.parquet`。

主线跑完后补 verifier-based test-time compute：

```bash
BEST_OF_LIST="1 4 8 16" bash scripts/run_ttc_sweep.sh "$QWEN_15B" 0
```
