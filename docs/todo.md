# 后续待办与亮点状态

本文档只记录还需要做的实验、分析和报告整理。远端按顺序执行的命令见 `docs/runbook.md`。

## 当前正式主线

- 保留：`Qwen2.5-1.5B-Instruct` 的 base -> SFT warm-up -> SFT+GRPO 主线。
- 保留：Countdown 3-4 数字任意目标加分项。
- 归档：0.5B direct GRPO 和 3B SFT/GRPO 只作为历史预实验，不进入正式主结果表，也不做模型大小对照。

## 亮点实现状态

| 亮点 | 当前状态 | 还需要做什么 |
| --- | --- | --- |
| Verifier-based test-time compute | 已有 `bestof_eval.py`，新增 `scripts/run_ttc_sweep.sh` 支持 best-of-1/4/8/16 sweep。 | 主线跑完后执行 sweep，并在报告中画 greedy/best-of-N solve-rate 曲线。 |
| 错误类型分析 | 已实现。`quick_eval.py` 和 `bestof_eval.py` 输出 `error_counts`，`summarize_eval.py` 可汇总。 | 跑完后整理各 split 的 `format_error`、`number_mismatch`、`wrong_value`、`invalid_expression`、`hallucination`。 |
| Hard split 难度分析 | 已补充。评估输出 `difficulty`，包含官方 `solved_rate`、`rank` 和 solved-rate 分桶表现。 | 比较 official OOD 与 ToT hard 900-1000；重点看低 solved-rate 桶是否更难。 |
| SFT vs GRPO 作用分析 | 已由主线覆盖：base、SFT、SFT+GRPO 三阶段。 | 比较 greedy 与 best-of-N 下 SFT/GRPO 是否提升，说明 GRPO 对候选覆盖和幻觉的影响。 |
| GRPO 超参稳定性对照 | 已新增 `scripts/run_grpo_ablation.sh`，默认跑 `3e-7/s300` 和 `3e-7/s600`。 | 主线跑完后补跑，比较 OOD/hard solve rate 与 unsolvable hallucination。 |
| Countdown 泛化加分项 | 已有 `scripts/run_countdown_bonus.sh`，reward、prompt、validator 均支持 per-example target。 | 远端跑出独立结果表，即使准确率不高也可作为框架迁移验证。 |

## 必须补跑

1. 1.5B 主线完整实验。
2. Countdown 加分项实验。
3. 主线 best-of-N sweep：`best-of-1/4/8/16`。
4. GRPO 超参稳定性对照：`3e-7/s300`、`3e-7/s600`。
5. 结果汇总：`summarize_eval.py` 输出的主指标、错误类型、难度分桶。
6. 训练曲线：SFT loss、GRPO reward/accuracy。

## 等第一轮结果后再决定

这些不是现在必须加入的新主线，避免实验发散。

1. 如果 `3e-7/s300` 和 `3e-7/s600` 都明显低于当前主线：
   - 保留当前 `8e-7/s300` 作为最终主结果。
   - 报告中解释低学习率更保守但没有带来更好候选覆盖。

2. 如果 SFT 本身很弱：
   - 增加 SFT 样本数或 epoch。
   - 增加 synthetic curriculum 的占比。

3. 如果 unsolvable hallucination 很高：
   - 增加 SFT 中不可解样本。
   - 报告中单独分析模型是否倾向于强行输出表达式。

4. 如果 best-of-8 已明显高于 greedy：
   - best-of-16 结果应保留为重要亮点。
   - 报告解释为候选池覆盖提升，但单次稳定输出仍不足。

## 最终报告应回答的问题

- SFT 是否主要提升格式正确率和表达式合法性？
- GRPO 是否提升 solve rate，还是只提升 best-of-N 候选命中率？
- ToT hard split 是否明显难于普通 official OOD？
- 错误主要来自算错、数字使用错误，还是格式错误？
- 不可解样本上模型是否仍 hallucinate？
- Countdown 是否证明同一套 verifier/reward 框架可迁移到任意 target？
