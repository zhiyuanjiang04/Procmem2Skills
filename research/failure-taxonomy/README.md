# Failure Mode Taxonomy Pipeline

按 PLAN.md 跑出 6 组实验（3 bench × 7 setting × 3 arm）的失败模式分类，回答两个问题：
- skill 减少了哪类失败
- workflow memory 为什么会拖累，skill 什么时候 work / fail

## 当前状态（完成）

| 阶段 | 状态 |
|------|------|
| 01 build-manifest | done — 8135 trial 索引 |
| 02 sample-label | done — 240 raw label |
| 03 aggregate-modes | done — v1 canonical taxonomy（12 mode）冻结 |
| 04 paired-compare | done — 528 (task, setting) triple 全部标注完 |
| 05 report | done — report_v1.md + report_v1_tables.json |

## 文件

| 文件 | 说明 |
|------|------|
| `01_build_manifest.py` | 扫描提取后的 pm2s-traces，输出统一 manifest |
| `02_sample_label.py` | stratified 抽样 + Claude Sonnet LLM-as-judge 打 raw label |
| `03_aggregate_modes.py` | 聚合 raw label → v1 canonical taxonomy |
| `04_pair_compare.py` | paired 对比标注（3 arm 同时，不做独立标注） |
| `05_report.py` | aggregate report（success rate / mode delta / mechanism / CI） |
| `outputs/manifest.jsonl` | 8135 行，每行一个 trial |
| `outputs/pair_labels_v1.jsonl` | 528 条 paired label |
| `outputs/report_v1.md` | **主产出**：完整分析报告 |
| `outputs/canonical_mode_map.v1.json` | v1 taxonomy 12 个 mode 定义 |
| `outputs/_pair_cache/` | 单 triple 缓存（resume 用） |

## 数据来源

- HF dataset：`Zhiyuanjiang/pm2s-traces-tmp`（gated）
- 8GB tarball 选择性解压：codex.txt + result.json + instruction.md + SKILL.md + solve.sh + task.toml（解压后 1.8GB）

## 跑法

```bash
source venv312/bin/activate
python 01_build_manifest.py
PARALLEL=4 SAMPLE_N=240 python 02_sample_label.py
python 03_aggregate_modes.py
BATCH_LIMIT=600 PARALLEL=2 python 04_pair_compare.py
python 05_report.py
```

详细参数说明在 PIPELINE.md 里。04_pair_compare.py 有自动断点续跑（缓存到 `outputs/_pair_cache/`）和 OAuth 限额 kill switch。

## 模型与认证

- 全程 Claude Sonnet 4.6
- `claude -p` headless 走 OAuth（Max 订阅，不消耗 API credit）
- 关键参数：`--tools ""`、`--no-session-persistence`、stdin 输入

## LLM 输出 schema

每个 trial 标注为：

```json
{
  "freeform_reasoning_short": "1-2 sentences summarizing what happened",
  "primary_mode_candidate": "noun phrase, e.g. 'missing python dependency'",
  "secondary_factors": ["..."],
  "evidence_spans": [{"source": "codex.txt", "quote": "..."}],
  "skill_effect_judgment": "helps | neutral | hurts | not_applicable",
  "skill_effect_reason": "...",
  "capability_vs_knowledge": "knowledge_missing | knowledge_present_but_misused | capability_limit | environmental"
}
```

## 核心结论

- skill vs workflow 成功率差 **+6.06pp**（bootstrap 95% CI [+0.76, +11.36]，唯一显著对比）
- skill 最大贡献：消除 `environment_infrastructure_failure`（net +27）
- skill 最大负担：`skill_guidance_misapplied_or_ignored`（net -48）
- workflow 最大负担：`timeout_budget_exhaustion`（比 raw 多 46 个）
- mechanism：347/528 靠 `procedural_anchor`，`knowledge_injection` 仅 24 个

详见 `outputs/report_v1.md` 和 HANDOFF.md。
