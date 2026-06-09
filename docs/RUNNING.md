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

推荐使用本地模型路径，避免训练中反复联网下载：

```bash
export QWEN_05B=/data/ysf/.cache/modelscope/Qwen/Qwen2___5-0___5B-Instruct
export QWEN_3B=/data/ysf/.cache/modelscope/Qwen/Qwen2___5-3B-Instruct
```

如果使用 Hugging Face 名称，可将上面变量替换为：

```bash
export QWEN_05B=Qwen/Qwen2.5-0.5B-Instruct
export QWEN_3B=Qwen/Qwen2.5-3B-Instruct
```

本包包含 `dataset_cache/train.json`，训练脚本会优先使用该缓存。若需要重新下载 `nlile/24-game`，删除该缓存并保证服务器可访问 Hugging Face datasets。

## 3. 快速连通性测试

先跑一个小实验，确认环境、CUDA、模型加载、数据加载均正常：

```bash
CUDA_VISIBLE_DEVICES=0 python game24/run_experiment.py \
  --quick \
  --output_dir ./output/game24-quick-rerun
```

输出位置：

- `./output/game24-quick-rerun/final_model/`：LoRA adapter 与 tokenizer 文件。
- `./output/game24-quick-rerun/training_summary.json`：训练摘要。
- `./output/game24-quick-rerun/training_metrics.json`：逐步训练指标。

## 4. 复现旧版 0.5B GRPO 基线

旧版 `run_experiment.py` 的训练阶段读取 `game24/config.py` 中的 `ModelConfig.model_name`。在服务器上运行前，请确认该字段指向 `$QWEN_05B` 对应的本地路径或 Hugging Face 名称。

```bash
CUDA_VISIBLE_DEVICES=0 python game24/run_experiment.py \
  --epochs 3 \
  --learning_rate 2e-5 \
  --beta 0.04 \
  --num_generations 4 \
  --output_dir ./output/game24-grpo-full-rerun \
  --base_model "$QWEN_05B"
```

预期输出：

- `./output/game24-grpo-full-rerun/final_model/`
- `./output/game24-grpo-full-rerun/training_summary.json`
- `./output/game24-grpo-full-rerun/training_metrics.json`
- `./output/game24-grpo-full-rerun/evaluation_results.json`
- `./output/game24-grpo-full-rerun/experiment_summary.json`

该实验在原始记录中失败：模型主要学会 `<think>...</think><answer>...</answer>` 输出模板，ID/OOD 解题率均为 0。

## 5. 复现 3B 课程 SFT

这是用于解决“小模型只学到输出模板”的关键改动：先用可验证表达式和部分不可解样本做课程式 SFT warm-up，让模型先学会表达式求解和拒答行为，再接 GRPO。

```bash
CUDA_VISIBLE_DEVICES=0 python game24/sft_warmup.py \
  --model_name "$QWEN_3B" \
  --output_dir ./output/game24-sft-3b-curriculum \
  --epochs 2 \
  --learning_rate 8e-5 \
  --synthetic \
  --unsolvable_limit 300
```

输出位置：

- `./output/game24-sft-3b-curriculum/final_model/`
- `./output/game24-sft-3b-curriculum/sft_summary.json`
- `./output/game24-sft-3b-curriculum/sft_metrics.json`

复现本实验后，运行 greedy 评估：

```bash
CUDA_VISIBLE_DEVICES=0 python game24/quick_eval.py \
  --base_model "$QWEN_3B" \
  --model_path ./output/game24-sft-3b-curriculum/final_model \
  --output_file ./output/game24-sft-3b-curriculum/quick_eval_60.json \
  --n 60 \
  --max_new_tokens 96
```

运行 best-of-8 评估：

```bash
CUDA_VISIBLE_DEVICES=0 python game24/bestof_eval.py \
  --base_model "$QWEN_3B" \
  --model_path ./output/game24-sft-3b-curriculum/final_model \
  --output_file ./output/game24-sft-3b-curriculum/bestof8_eval_30.json \
  --n 30 \
  --best_of 8 \
  --max_new_tokens 96 \
  --temperature 0.9
```

## 6. 复现 3B SFT 后短程 GRPO

```bash
CUDA_VISIBLE_DEVICES=0 python game24/grpo_warm.py \
  --base_model "$QWEN_3B" \
  --init_adapter ./output/game24-sft-3b-curriculum/final_model \
  --output_dir ./output/game24-grpo-3b-curriculum-short \
  --epochs 1 \
  --max_train_samples 50 \
  --learning_rate 8e-7 \
  --num_generations 4 \
  --max_new_tokens 96 \
  --temperature 0.9
```

输出位置：

- `./output/game24-grpo-3b-curriculum-short/final_model/`
- `./output/game24-grpo-3b-curriculum-short/grpo_summary.json`
- `./output/game24-grpo-3b-curriculum-short/grpo_metrics.json`

评估：

```bash
CUDA_VISIBLE_DEVICES=0 python game24/quick_eval.py \
  --base_model "$QWEN_3B" \
  --model_path ./output/game24-grpo-3b-curriculum-short/final_model \
  --output_file ./output/game24-grpo-3b-curriculum-short/quick_eval_30.json \
  --n 30 \
  --max_new_tokens 96

CUDA_VISIBLE_DEVICES=0 python game24/bestof_eval.py \
  --base_model "$QWEN_3B" \
  --model_path ./output/game24-grpo-3b-curriculum-short/final_model \
  --output_file ./output/game24-grpo-3b-curriculum-short/bestof8_eval_30.json \
  --n 30 \
  --best_of 8 \
  --max_new_tokens 96 \
  --temperature 0.9
```

## 7. 参数含义

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
| `--best_of` | `bestof_eval.py` | 每道题采样多少个候选，再由验证器选择正确候选。 |

## 8. 重要指标解释

- ID 解题率：in-distribution solve rate，在训练分布同源的 held-out 可解题上，答案表达式通过验证器的比例。
- OOD 解题率：out-of-distribution solve rate，在数字范围或构造方式不同的可解题上，答案表达式通过验证器的比例。
- Greedy：确定性解码，`do_sample=False`，每题只生成 1 个答案，反映模型单次输出能力。
- Best-of-8：每题采样 8 个候选，使用规则验证器挑出第一个正确答案；它反映“模型候选池里是否包含正确解”，通常高于 greedy。
