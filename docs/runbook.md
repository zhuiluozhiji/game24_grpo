# 远端完整实验操作指南

目标：补齐高分作业需要的完整实验闭环。正式主线只保留题目指定的 `Qwen2.5-1.5B-Instruct`，并完成 official hard split、Countdown 加分项、训练曲线和错误类型统计。

说明：仓库中保留了过去已经跑过的 0.5B direct GRPO 和 3B SFT/GRPO 结果，但它们不再作为正式主实验或模型大小对照。原因是训练方式不一致，不能严谨归因于模型规模。它们只作为历史预实验归档，用于说明探索过程，不建议写入最终主结果表。

所有命令默认在项目根目录执行。

## 0. 建议使用 tmux 并保存日志

```bash
tmux new -s game24
mkdir -p output/logs
```

每个长实验都建议用 `tee` 保存日志，例如：

```bash
bash scripts/run_15b_mainline.sh "$QWEN_15B" 0 2>&1 | tee output/logs/run_15b_mainline.log
```

## 1. 环境准备

```bash
conda create -n game24-grpo python=3.10 -y
conda activate game24-grpo

pip install -r game24/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121

export PYTHONPATH=$PWD
```

如果服务器已经有 CUDA/PyTorch 环境，可以跳过 `pip install torch`，但仍需安装 `game24/requirements.txt`。

## 2. 获取并配置模型路径

下面的 `export QWEN_15B=/path/to/...` 不是下载命令，而是告诉脚本“模型已经在哪里”。如果远端还没有 `Qwen2.5-1.5B-Instruct`，先用以下任一方式获取。

### 方式 A：服务器能访问 Hugging Face，直接使用模型名

这种方式最简单，`transformers` 会在首次运行时自动下载到 Hugging Face cache：

```bash
export QWEN_15B=Qwen/Qwen2.5-1.5B-Instruct
```

缺点是第一次训练/评估时会边跑边下载，容易因为网络中断失败。

### 方式 B：提前用 Hugging Face 下载到本地目录

```bash
pip install -U huggingface_hub
mkdir -p /data/models

huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir /data/models/Qwen2.5-1.5B-Instruct

export QWEN_15B=/data/models/Qwen2.5-1.5B-Instruct
```

### 方式 C：国内服务器用 ModelScope 下载

```bash
pip install -U modelscope
mkdir -p /data/models

modelscope download --model Qwen/Qwen2.5-1.5B-Instruct \
  --local_dir /data/models/Qwen2.5-1.5B-Instruct

export QWEN_15B=/data/models/Qwen2.5-1.5B-Instruct
```

### 最终确认环境变量

无论用哪种方式，正式实验只需要确认：

```bash
export QWEN_15B=/path/to/Qwen2.5-1.5B-Instruct
```

## 3. 先做轻量检查

```bash
python -m compileall game24
bash -n scripts/run_15b_mainline.sh
bash -n scripts/run_countdown_bonus.sh
bash -n scripts/run_ttc_sweep.sh
bash -n scripts/run_grpo_ablation.sh
```

检查 solver、validator、target-aware reward：

```bash
python - <<'PY'
from game24.solver import solve_target
from game24.utils import validate_solution
from game24.rewards import accuracy_reward

expr = solve_target([2, 3, 7], 13)
print(expr)
print(validate_solution([2, 3, 7], expr, 13))
print(accuracy_reward([f"<think>x</think><answer>{expr}</answer>"], [[2, 3, 7]], targets=[13]))
PY
```

## 4. 必跑：1.5B 题目主线

这一步是最终报告的主实验，覆盖：

- 1.5B base model greedy/best-of-N baseline。
- 1.5B SFT warm-up。
- 1.5B SFT 后 GRPO continuation。
- ID、official OOD、ToT hard split 900-1000、unsolvable。
- 训练曲线生成。

默认推荐规模：

```bash
N_EVAL=200 N_HARD=100 BEST_OF=8 GRPO_SAMPLES=300 GRPO_GENERATIONS=8 \
  bash scripts/run_15b_mainline.sh "$QWEN_15B" 0 2>&1 | tee output/logs/run_15b_mainline.log
```

如果显存或时间不够，先用小规模冒烟：

```bash
N_EVAL=30 N_HARD=30 BEST_OF=4 GRPO_SAMPLES=50 GRPO_GENERATIONS=4 \
  bash scripts/run_15b_mainline.sh "$QWEN_15B" 0 2>&1 | tee output/logs/run_15b_smoke.log
```

主实验预期输出：

```text
output/game24-15b-base/
output/game24-sft-15b-curriculum/
output/game24-grpo-15b-curriculum/
```

重点保留文件：

```text
quick_eval_*.json
bestof*_eval_*.json
sft_summary.json
sft_metrics.json
grpo_summary.json
grpo_metrics.json
plots/*.png
```

## 4.1 推荐补跑：GRPO 超参稳定性对照

这一步用于回应“强化学习训练不稳定，需要监控并必要时调整超参数”。它不重跑 SFT，不覆盖第 4 节主线的 `output/game24-grpo-15b-curriculum/`，而是基于已经训练好的 SFT adapter 继续跑两组更保守的 GRPO 对照：

| 输出目录 | GRPO_LR | GRPO_SAMPLES | GRPO_GENERATIONS | 目的 |
| --- | ---: | ---: | ---: | --- |
| `output/game24-grpo-15b-lr3e-7-s300/` | 3e-7 | 300 | 8 | 单独验证降低学习率是否能降低 hallucination |
| `output/game24-grpo-15b-lr3e-7-s600/` | 3e-7 | 600 | 8 | 在低学习率下增加训练步数，看能否保住或提升 OOD/hard |

推荐在第 4 节主线跑完后执行：

```bash
N_EVAL=200 N_HARD=100 BEST_OF=8 GRPO_GENERATIONS=8 \
  bash scripts/run_grpo_ablation.sh "$QWEN_15B" 0 2>&1 | tee output/logs/run_grpo_ablation.log
```

小规模冒烟只跑低学习率 300 samples：

```bash
GRPO_ABLATIONS="lr3e-7-s300:3e-7:300" N_EVAL=30 N_HARD=30 BEST_OF=4 \
  bash scripts/run_grpo_ablation.sh "$QWEN_15B" 0 2>&1 | tee output/logs/run_grpo_ablation_smoke.log
```

判断重点：

- `Official OOD` 和 `ToT hard` 的 greedy / best-of-8 solve rate 是否接近或超过主线 GRPO。
- `unsolvable` 的 greedy hallucination 是否低于主线 GRPO 的 34.0%。
- `best-of-8` hallucination 是否继续保持在低水平。
- `grpo_metrics.json` 和 `plots/` 中 reward、accuracy、solved 曲线是否更平稳。

## 5. 推荐补跑亮点：Verifier-based test-time compute

这一步不重新训练，只对已经得到的 base、SFT、SFT+GRPO 三个阶段做多档 best-of-N 评估。它用于回答：同一个 1.5B 模型和同一套训练下，采样候选数增加是否能显著提升 solve rate。

推荐在第 4 节主线跑完后执行：

```bash
N_EVAL=200 N_HARD=100 BEST_OF_LIST="1 4 8 16" \
  bash scripts/run_ttc_sweep.sh "$QWEN_15B" 0 2>&1 | tee output/logs/run_ttc_sweep.log
```

小规模冒烟：

```bash
N_EVAL=30 N_HARD=30 BEST_OF_LIST="1 4 8" \
  bash scripts/run_ttc_sweep.sh "$QWEN_15B" 0 2>&1 | tee output/logs/run_ttc_sweep_smoke.log
```

说明：

- Greedy 结果来自 `quick_eval_*.json`，是确定性单次生成。
- `best-of-1` 来自 `bestof_eval.py --best_of 1`，是采样 1 个候选，不等同于 greedy。
- 最终报告中可以画 `greedy / best-of-1 / best-of-4 / best-of-8 / best-of-16` 的 solve-rate 曲线。

## 6. 必跑加分项：Countdown 任意目标

这一步对应题目中的加分项：“参考 TinyZero，用 3-4 数字凑任意目标数”。

推荐规模：

```bash
COUNTDOWN_TRAIN=3000 COUNTDOWN_EVAL=200 BEST_OF=8 GRPO_SAMPLES=300 GRPO_GENERATIONS=8 \
  bash scripts/run_countdown_bonus.sh "$QWEN_15B" 0 2>&1 | tee output/logs/run_countdown_bonus.log
```

小规模冒烟：

```bash
COUNTDOWN_TRAIN=300 COUNTDOWN_EVAL=30 BEST_OF=4 GRPO_SAMPLES=50 GRPO_GENERATIONS=4 \
  bash scripts/run_countdown_bonus.sh "$QWEN_15B" 0 2>&1 | tee output/logs/run_countdown_smoke.log
```

预期输出：

```text
output/countdown-sft-15b/
output/countdown-grpo-15b/
```

## 7. 汇总结果

跑完后，先用汇总脚本把主要 JSON 结果打印出来：

```bash
python game24/summarize_eval.py \
  output/game24-15b-base/quick_eval_200.json \
  output/game24-15b-base/bestof8_eval_200.json \
  output/game24-sft-15b-curriculum/quick_eval_200.json \
  output/game24-sft-15b-curriculum/bestof8_eval_200.json \
  output/game24-grpo-15b-curriculum/quick_eval_200.json \
  output/game24-grpo-15b-curriculum/bestof8_eval_200.json \
  output/countdown-sft-15b/quick_eval_200.json \
  output/countdown-sft-15b/bestof8_eval_200.json \
  output/countdown-grpo-15b/quick_eval_200.json \
  output/countdown-grpo-15b/bestof8_eval_200.json \
  2>&1 | tee output/logs/summary_eval.log
```

如果你用的是小规模实验，把文件名里的 `200` 改成对应的 `30`。

汇总脚本会同时打印：

- `solve_rate`、`format_rate`、`hallucination_rate`；
- `error_counts` 错误类型；
- official split 中的平均 `solved_rate`、平均 `rank` 和 solved-rate 分桶表现。

补跑 best-of-N sweep 后，可以单独汇总：

```bash
python game24/summarize_eval.py \
  output/game24-15b-base/bestof*_eval_200.json \
  output/game24-sft-15b-curriculum/bestof*_eval_200.json \
  output/game24-grpo-15b-curriculum/bestof*_eval_200.json \
  2>&1 | tee output/logs/summary_ttc_sweep.log
```

补跑 GRPO 超参稳定性对照后，可以单独汇总：

```bash
python game24/summarize_eval.py \
  output/game24-grpo-15b-curriculum/quick_eval_200.json \
  output/game24-grpo-15b-curriculum/bestof8_eval_200.json \
  output/game24-grpo-15b-lr3e-7-s300/quick_eval_200.json \
  output/game24-grpo-15b-lr3e-7-s300/bestof8_eval_200.json \
  output/game24-grpo-15b-lr3e-7-s600/quick_eval_200.json \
  output/game24-grpo-15b-lr3e-7-s600/bestof8_eval_200.json \
  2>&1 | tee output/logs/summary_grpo_ablation.log
```

## 8. 单独补图

主脚本会自动调用 `plot_metrics.py`。如果中途失败或想重新生成图：

```bash
python game24/plot_metrics.py output/game24-sft-15b-curriculum
python game24/plot_metrics.py output/game24-grpo-15b-curriculum
python game24/plot_metrics.py output/game24-grpo-15b-lr3e-7-s300
python game24/plot_metrics.py output/game24-grpo-15b-lr3e-7-s600
python game24/plot_metrics.py output/countdown-sft-15b
python game24/plot_metrics.py output/countdown-grpo-15b
```

图像会保存到各 run 目录的 `plots/` 下。

## 9. 最终报告建议表格

最终报告至少整理这几张表：

1. 24 点主任务总表：
   `base 1.5B / SFT 1.5B / SFT+GRPO 1.5B`。
2. 每个模型的 greedy 与 best-of-8：
   `ID / official OOD / ToT hard 900-1000 / unsolvable hallucination`。
3. Countdown 加分项：
   `SFT / SFT+GRPO` 在 Countdown test 上的 greedy 与 best-of-8。
4. Test-time compute：
   `greedy / best-of-1 / best-of-4 / best-of-8 / best-of-16` 的 solve rate 曲线。
5. GRPO 超参稳定性对照：
   `8e-7/s300`、`3e-7/s300`、`3e-7/s600` 的 OOD、hard、hallucination 对比。
6. 错误类型统计：
   `format_error / number_mismatch / wrong_value / invalid_expression / refusal_or_no_answer / hallucination`。
7. Hard split 难度分析：
   官方 `solved_rate` 分桶下的模型 solve rate，以及 ToT hard 与普通 OOD 的对比。
8. 训练曲线：
   SFT loss、GRPO reward、GRPO accuracy、solved per group。

## 10. 打包结果

```bash
tar -czf output/game24_results_$(date +%Y%m%d_%H%M).tar.gz \
  output/game24-15b-base \
  output/game24-sft-15b-curriculum \
  output/game24-grpo-15b-curriculum \
  output/game24-grpo-15b-lr3e-7-s300 \
  output/game24-grpo-15b-lr3e-7-s600 \
  output/countdown-sft-15b \
  output/countdown-grpo-15b \
  output/logs
```

## 11. 历史预实验归档

以下结果已经保留在仓库中，但不作为正式主实验，也不作为严格模型大小对照：

```text
results/game24-grpo-full/
results/game24-sft-3b-curriculum/
results/game24-grpo-3b-curriculum-short/
```

使用方式：

- 可以在报告附录或“预实验探索”中一句话说明：早期 direct GRPO 和 3B warm-up 结果帮助我们确定正式方案。
- 不建议把 0.5B、1.5B、3B 放进同一张主对比表，因为训练协议不同。
- 不建议得出“模型越大越好”的结论。

## 12. 结果判断重点

- Greedy 是单次输出能力，通常偏低。
- Best-of-8 是候选池命中率，表示 8 个候选中是否至少有一个被验证器判定正确。
- 如果 GRPO 没有提升 ID greedy，但提升 OOD best-of-8，可以解释为：短程 GRPO 主要改善候选覆盖，而不是稳定单次输出。
- 如果 unsolvable hallucination 下降，这是重要亮点。
- Countdown 只要能跑通并给出独立表格，就是明确加分项完成。
