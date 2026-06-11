# Verifier-based test-time compute sweep 总结

更新时间：2026-06-11

本文档记录伙伴建议中的主线 best-of-N sweep。实验控制模型与训练阶段不变，只改变测试时每题采样候选数 `best_of=1/4/8/16`，用验证器选择第一个正确表达式。主线 best-of-8 已在实验4中跑出，本次远端补跑缺失的 best-of-1、best-of-4、best-of-16。

## 1. 复现命令

```bash
source /data/ysf/miniconda3/etc/profile.d/conda.sh
conda activate game24_grpo
export PYTHONPATH=/data/ysf/game24_grpo

N_EVAL=200 N_HARD=100 BEST_OF_LIST="1 4 16" \
  bash scripts/run_ttc_sweep.sh /data/ysf/models/Qwen2.5-1.5B-Instruct 4
```

远端日志保存为 `output/logs/run_ttc_sweep.log`，本地归档为 `results/logs/run_ttc_sweep.log`。

## 2. 主指标

| 模型阶段 | 解码 | ID | Official OOD | ToT hard 900-1000 | Unsolvable 幻觉率 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1.5B base | best-of-1 | 0.5% | 1.0% | 0.0% | 96.0% |
| 1.5B base | best-of-4 | 1.0% | 2.5% | 1.0% | 90.0% |
| 1.5B base | best-of-8 | 1.5% | 3.0% | 1.0% | 80.0% |
| 1.5B base | best-of-16 | 4.0% | 8.5% | 5.0% | 64.0% |
| 1.5B SFT | best-of-1 | 3.5% | 7.5% | 3.0% | 24.0% |
| 1.5B SFT | best-of-4 | 14.5% | 14.0% | 14.0% | 1.0% |
| 1.5B SFT | best-of-8 | 22.0% | 31.0% | 13.0% | 0.0% |
| 1.5B SFT | best-of-16 | 33.0% | 47.5% | 37.0% | 0.0% |
| 1.5B SFT+GRPO | best-of-1 | 4.5% | 10.5% | 2.0% | 50.0% |
| 1.5B SFT+GRPO | best-of-4 | 15.0% | 25.5% | 14.0% | 7.0% |
| 1.5B SFT+GRPO | best-of-8 | 27.5% | 43.5% | 32.0% | 1.0% |
| 1.5B SFT+GRPO | best-of-16 | 44.0% | 65.5% | 49.0% | 0.0% |

## 3. 结论

`SFT+GRPO + best-of-16` 是当前最强评估配置，Official OOD 达到 65.5%，ToT hard 900-1000 达到 49.0%。相比 best-of-8，OOD 继续提升 22.0 个百分点，ToT hard 继续提升 17.0 个百分点，并且不可解 hallucination 降到 0.0%。

结果说明 verifier-based test-time compute 能有效利用候选池：模型单次输出仍常见 `wrong_value` 或拒答，但增大候选数后，验证器可以筛出更多合法且等于 24 的表达式。最终报告应把 best-of-N sweep 作为核心亮点，而不是只报告 greedy。
