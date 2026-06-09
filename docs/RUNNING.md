# 24点 GRPO 实验复现指南

本文档面向服务器复现，假定服务器已经安装 Miniconda，并且可以访问 GPU。所有命令均在项目根目录执行。

## 1. 解压与创建环境

```bash
unzip game24_grpo_repro_package.zip -d game24_grpo_repro_package
cd game24_grpo_repro_package

conda create -n game24-grpo python=3.10 -y
conda activate game24-grpo

pip install -r game24/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
export PYTHONPATH=$PWD
```

如果服务器 PyTorch/CUDA 版本已经由集群统一安装，可跳过上面的 `pip install torch`，只安装 `game24/requirements.txt`。

## 2. 模型与数据准备

如果服务器可以访问 Hugging Face，可以直接使用模型名，`transformers` 会自动下载：

```bash
export QWEN_15B=Qwen/Qwen2.5-1.5B-Instruct
```

更推荐提前下载到本地目录，避免训练中反复联网。Hugging Face 下载方式：

```bash
pip install -U huggingface_hub
mkdir -p /data/models

huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir /data/models/Qwen2.5-1.5B-Instruct

export QWEN_15B=/data/models/Qwen2.5-1.5B-Instruct
```

国内服务器可以用 ModelScope：

```bash
pip install -U modelscope
mkdir -p /data/models

modelscope download --model Qwen/Qwen2.5-1.5B-Instruct \
  --local_dir /data/models/Qwen2.5-1.5B-Instruct

export QWEN_15B=/data/models/Qwen2.5-1.5B-Instruct
```

本包包含 `dataset_cache/train.json`，训练脚本会优先使用该缓存。若需要重新下载 `nlile/24-game`，删除该缓存并保证服务器可访问 Hugging Face datasets。

## 3. 快速检查

正式训练前先做不加载模型的轻量检查：

```bash
python -m compileall game24
bash -n scripts/run_15b_mainline.sh
bash -n scripts/run_countdown_bonus.sh
bash -n scripts/run_ttc_sweep.sh
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

## 4. 复现题目指定 1.5B 主线

该脚本会依次跑 base 评估、SFT warm-up、SFT 评估、GRPO continuation、GRPO 评估，并覆盖 official OOD 与 ToT hard split 900-1000。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_15b_mainline.sh "$QWEN_15B" 0
```

可用环境变量控制规模：

```bash
N_EVAL=200 N_HARD=100 BEST_OF=8 GRPO_SAMPLES=300 GRPO_GENERATIONS=8 \
  bash scripts/run_15b_mainline.sh "$QWEN_15B" 0
```

输出位置：

- `./output/game24-15b-base/`
- `./output/game24-sft-15b-curriculum/`
- `./output/game24-grpo-15b-curriculum/`

## 5. 复现 Countdown 加分项

该脚本使用 `Jiayi-Pan/Countdown-Tasks-3to4`，任务形式为 3-4 个数字凑任意目标数。验证器、奖励、prompt 均读取每条样本的 `target` 字段。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_countdown_bonus.sh "$QWEN_15B" 0
```

可用环境变量控制规模：

```bash
COUNTDOWN_TRAIN=3000 COUNTDOWN_EVAL=200 BEST_OF=8 GRPO_SAMPLES=300 \
  bash scripts/run_countdown_bonus.sh "$QWEN_15B" 0
```

输出位置：

- `./output/countdown-sft-15b/`
- `./output/countdown-grpo-15b/`

## 6. 补跑 test-time compute sweep

主线跑完后，可以只做评估、不重新训练，比较同一模型在不同候选数下的 solve rate。

```bash
N_EVAL=200 N_HARD=100 BEST_OF_LIST="1 4 8 16" \
  bash scripts/run_ttc_sweep.sh "$QWEN_15B" 0
```

说明：

- Greedy 是 `quick_eval.py` 的确定性单次输出。
- `best-of-1` 是采样一个候选，不等同于 greedy。
- `best-of-4/8/16` 用同一验证器从多个候选中选择第一个正确表达式。

## 7. 历史预实验归档

仓库保留了早期 0.5B direct GRPO 和 3B SFT/GRPO 结果：

```text
results/game24-grpo-full/
results/game24-sft-3b-curriculum/
results/game24-grpo-3b-curriculum-short/
```

这些结果不作为正式主实验，也不作为模型大小对照，因为训练协议不同。对应复现脚本仍保留在 `scripts/run_old_05b_grpo.sh` 和 `scripts/run_3b_curriculum.sh`，仅供归档或排查使用。

## 8. 参数含义

| 参数 | 所属脚本 | 含义 |
| --- | --- | --- |
| `--output_dir` | train/SFT/GRPO | 保存 adapter、tokenizer、summary、metrics 的目录。 |
| `--model_name` | `sft_warmup.py` | SFT 加载的基础模型路径或 Hugging Face 名称。 |
| `--base_model` | eval/GRPO | 评估或 GRPO continuation 加载的基础模型。 |
| `--init_adapter` | `grpo_warm.py` | GRPO continuation 的初始 LoRA adapter，通常为 SFT 输出的 `final_model`。 |
| `--model_path` | eval | 待评估的 LoRA adapter 路径。 |
| `--epochs` | train/SFT/GRPO | 训练轮数。 |
| `--max_train_samples` | SFT/GRPO | 限制训练样本数，便于短实验。 |
| `--learning_rate` | train/SFT/GRPO | LoRA 参数学习率。SFT 可更大，GRPO continuation 建议较小。 |
| `--synthetic` | `sft_warmup.py` | 启用程序生成的课程样本，补充真实数据。 |
| `--unsolvable_limit` | `sft_warmup.py` | 加入不可解样本数量，用于学习拒答，降低幻觉。 |
| `--num_generations` | GRPO | 每个 prompt 采样多少个候选，用组内相对奖励计算优势。 |
| `--beta` | 旧版 GRPO | KL 约束强度。越大越保守。 |
| `--max_new_tokens` | eval/GRPO | 每个答案最多生成 token 数。 |
| `--temperature` | best-of/GRPO | 采样温度。越高候选更多样，但错误也可能更多。 |
| `--n` | eval | 每个 split 评估多少道题。 |
| `--n_hard` | eval | ToT hard split 900-1000 评估多少道题。 |
| `--best_of` | `bestof_eval.py` | 每道题采样多少个候选，再由验证器选择正确候选。 |
| `--task` | SFT/GRPO/eval | `game24` 或 `countdown`，SFT 还支持 `mixed`。 |
| `--synthetic_target` | `sft_warmup.py` | 合成课程样本的目标值，默认 24。 |

## 9. 重要指标解释

- ID 解题率：in-distribution solve rate，在训练分布同源的 held-out 可解题上，答案表达式通过验证器的比例。
- OOD 解题率：out-of-distribution solve rate，在数字范围或构造方式不同的可解题上，答案表达式通过验证器的比例。
- Greedy：确定性解码，`do_sample=False`，每题只生成 1 个答案，反映模型单次输出能力。
- Best-of-8：每题采样 8 个候选，使用规则验证器挑出第一个正确答案；它反映“模型候选池里是否包含正确解”，通常高于 greedy。
- Hard split 难度分桶：official `game-of-24` 自带 `rank` 与 `solved_rate`，评估 JSON 的 `difficulty` 字段会按 solved-rate 区间汇总模型表现。
