# Failure Taxonomy — Handoff

完整的 paired-comparison failure taxonomy pipeline 跑完了，结果 + 代码 + 数据都在当前目录。对 528 个 (task, setting) 三元组做了 raw / workflow / skill 三臂对比标注，产出了定量的失败模式分析报告。

## 方法

不做独立标注，直接做 paired 对比：对每个 (benchmark, setting, task)，把三个 arm 的 agent trajectory 同时送给 Claude Sonnet，让它用 v1 共享 taxonomy（12 个 mode）打标，同时输出 net_effect（fixed / regressed / unchanged / mixed）和 skill_mechanism（procedural_anchor / failure_warning / knowledge_injection / none / counterproductive）。这种 paired 设计直接产出因果叙述，不需要事后推断 contingency。

12 个 mode 是对 240 个独立 raw label 做 LLM 聚合 + 人工审定冻结的 v1 taxonomy，参见 `outputs/canonical_mode_map.v1.json`。

## 跑出来的结果

| 指标 | 数字 |
|------|------|
| 分析的 (task, setting) 三元组 | 528 |
| 成功率 raw | 59.1% |
| 成功率 workflow | 55.9% |
| 成功率 skill | 61.9% |
| skill vs raw 均值 delta | +2.84pp（CI [-2.27, +7.95]，**不显著**） |
| workflow vs raw 均值 delta | -3.22pp（CI [-8.14, +2.08]，**不显著**） |
| **skill vs workflow** | **+6.06pp（CI [+0.76, +11.36]，显著）** |

## 关键发现

1. **skill 对 workflow 的优势是唯一统计显著的**：+6.06pp，CI 不含 0。直接跟 raw 比不显著——因为 raw 成功率本身就比 workflow 高，workflow 拖累了基线。

2. **skill_guidance_misapplied_or_ignored 是 skill 最大负担**：net delta -48（skill arm 引入了 52 个新的这类 case，但只 fixed 4 个）。说明写好 skill.md 还不够——agent 读了但没按 skill 意图执行的问题很突出。

3. **environment_infrastructure_failure 是 skill 最大贡献**：net delta +27（skill 完全消除了这类 failure）。

4. **timeout_budget_exhaustion 是 workflow 最大负担**：workflow 比 raw 多引入了 46 个 timeout case——workflow 注入让 prompt 更长、context 更重，agent 慢了一圈。skill 比 workflow 在这项上 fixed 27 个。

5. **mechanism 以 procedural_anchor 为主**：347/528 skill cases 是靠 procedural_anchor 帮到 agent 的，说明 skill 的主要价值是给 agent 提供步骤模板，而不是知识注入（24 个）或 failure warning（64 个）。

## 主产出

- `outputs/report_v1.md` — 完整报告（成功率 / bootstrap CI / mode 频次 / paired delta / mechanism / per-setting / per-benchmark 分解）
- `outputs/report_v1_tables.json` — 同上，raw JSON 格式（用于绘图）
- `outputs/pair_labels_v1.jsonl` — 528 条 paired label，每条含三个 arm 的 mode、evidence、net_effect、mechanism
- `outputs/canonical_mode_map.v1.json` — v1 taxonomy 12 个 mode 定义

## 下一步建议

1. **skill_guidance_misapplied_or_ignored 是最值得深挖的 failure mode**：52 个新 case 里看 agent 具体在哪一步偏离 skill 意图——是读了 skill 但没 follow，还是 skill 跟 task 对不齐？
2. **0s5f setting（全 failure workflow）下 skill_vs_raw regression 最多（28 个 regressed）**：说明 failure-heavy workflow 反而会拖累 skill——可能是 skill 跟 failure 案例的"过度告警"效应。
3. **terminalbench2 在 skill_vs_raw 上 regression 最多（47 个 regressed vs 28 个 fixed）**：跟 skillsbench / terminalbenchpro 相比这个 benchmark 的 skill 质量更值得审查。

## 复现命令

```bash
source venv312/bin/activate

# Stage 1: manifest（~1min，本地）
python 01_build_manifest.py

# Stage 2: raw sample label（240 trial，PARALLEL=4，~30min）
PARALLEL=4 SAMPLE_N=240 python 02_sample_label.py

# Stage 3: aggregate modes → v1 canonical taxonomy（~10min）
python 03_aggregate_modes.py

# Stage 4: paired compare（528 triples，PARALLEL=2，分批跑绕 OAuth 限额）
BATCH_LIMIT=600 PARALLEL=2 python 04_pair_compare.py

# Stage 5: aggregate report（本地，<1s）
python 05_report.py
```

04_pair_compare.py 有 kill switch（连续 5 次 + 累计 10 次失败自动停），缓存到 `outputs/_pair_cache/`，断了重跑会跳过已完成的 triple。

详细设计和 prompt 结构在 PIPELINE.md 里。
