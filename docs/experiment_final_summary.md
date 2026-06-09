# 24点游戏 GRPO 强化学习实验总结

更新时间：2026-06-07

> 说明：本文档记录的是旧版 Qwen2.5-0.5B direct GRPO 失败基线，不是最终作业总结。当前代码已经新增 1.5B 主线、official hard split、Countdown 加分项、错误统计和曲线绘图；最终报告应优先参考 `docs/runbook.md`、`docs/todo.md`、`docs/RUNNING.md` 和 `docs/EXPERIMENTS.md` 中的新增实验流程。

## 1. 实验状态

本次已完成 24 点游戏强化学习的端到端流程：

- 数据加载与训练/测试划分
- Qwen2.5-0.5B-Instruct + LoRA 初始化
- GRPO/RLVR 训练
- LoRA adapter 保存
- 分布内、分布外、不可解样本三类评估
- 训练与评估 JSON 结果落盘

远程完整实验目录：

```text
/data/ysf/game24_grpo/output/game24-grpo-full
```

本地已同步结果目录：

```text
output/game24-grpo-full
```

## 2. 训练配置

| 项目 | 值 |
| --- | --- |
| 基座模型 | Qwen2.5-0.5B-Instruct |
| 训练方法 | GRPO + RLVR |
| 参数高效微调 | LoRA |
| LoRA rank | 16 |
| 训练样本数 | 1089 |
| epoch | 3 |
| 总 step | 3267 |
| num_generations | 4 |
| learning rate | 2e-5 |
| KL beta | 0.04 |
| 训练耗时 | 17734 秒，约 4.93 小时 |
| 最终平均 reward | 0.51 |

训练摘要文件：

```text
output/game24-grpo-full/training_summary.json
```

## 3. 评估结果

| 测试集 | 样本数 | 解题成功率 | 格式正确率 | 幻觉率 |
| --- | ---: | ---: | ---: | ---: |
| 分布内 solvable 测试集 | 200 | 0.0% | 100.0% | - |
| 合成 OOD solvable 测试集 | 200 | 0.0% | 100.0% | - |
| 合成 unsolvable 幻觉检测集 | 100 | 0.0% | 100.0% | 100.0% |

评估摘要文件：

```text
output/game24-grpo-full/experiment_summary.json
output/game24-grpo-full/evaluation_results.json
```

## 4. 关键观察

模型已经稳定学会了输出格式：

```text
<think>...</think><answer>...</answer>
```

但策略塌缩为输出：

```text
<think> ( ) </think> <answer>24</answer>
```

因此格式奖励达到 100%，但表达式没有使用输入数字，无法通过 `validate_solution`，真实解题成功率为 0%。在不可解测试集上，模型仍然输出 `<answer>24</answer>`，因此幻觉率为 100%。

## 5. 本轮代码修复

为补齐全流程评估，已修改：

- `game24/data.py`
  - 优先读取 `dataset_cache/train.json`，缺失时再加载 Hugging Face 数据集。
  - 当源数据没有不可解样本时，自动合成 deterministic unsolvable 测试集。
  - 当 OOD 数据集无法联网加载时，自动合成 deterministic OOD solvable 测试集。
- `game24/rewards.py`
  - 将准确率奖励改为严格二值：只有表达式等于 24 且每个输入数字各用一次才给 1.0，否则给 -0.1。
  - 这个修复用于避免模型通过 `<answer>24</answer>` 获得错误的部分奖励。

## 6. 结论

实验流程已跑通，产物完整，可作为课程项目的失败案例/消融分析基础。当前结果说明：仅靠格式奖励 + 稀疏准确率奖励，在小模型和短训练下容易先学到输出模板，再陷入无效答案捷径。后续提升应优先考虑降低格式奖励权重、加入可验证中间过程约束、使用 SFT warm-up 或表达式搜索器生成少量冷启动样本。
