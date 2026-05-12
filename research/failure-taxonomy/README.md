# Failure Mode Taxonomy Pipeline

按 PLAN.md 跑出 6 组实验（3 bench × 7 setting × 3 arm）的失败模式分类，回答两个问题：
- skill 减少了哪类失败
- workflow memory 为什么会拖累，skill 什么时候 work / fail

## 当前状态

| 阶段 | 状态 |
|------|------|
| 01 build-manifest | done — 8135 trial 索引，含 result.json + codex.txt + instruction.md + SKILL.md 路径 |
| 02 sample-label | demo 跑通（5 trial × Sonnet via OAuth），label 质量验证 OK |
| 03 aggregate-modes | 待写 |
| 04 apply-map-full | 待写 |
| 05 report | 待写 |

## 文件

| 文件 | 说明 |
|------|------|
| `01_build_manifest.py` | 扫描提取后的 pm2s-traces，输出统一 manifest |
| `02_sample_label.py` | stratified 抽样 + Claude Sonnet LLM-as-judge 打 raw label |
| `outputs/manifest.jsonl` | 8135 行，每行一个 trial（含 reward/exception/agent/路径） |
| `outputs/labels_raw.jsonl` | LLM 标注结果（demo 5 trial，全量 240 跑完后扩展） |
| `outputs/_label_cache/` | 单 trial 缓存（resume 用） |

## 数据来源

- HF dataset：`Zhiyuanjiang/pm2s-traces-tmp`（gated）
- 8GB tarball 选择性解压：codex.txt + result.json + instruction.md + SKILL.md + solve.sh + task.toml（解压后 1.8GB）

## 跑法

```bash
# 1. build manifest（解压完后跑一次，<1 分钟）
python 01_build_manifest.py

# 2. 抽样 + 标注（240 sample 并行 4 个，~30 分钟）
PARALLEL=4 SAMPLE_N=240 python 02_sample_label.py

# 后续 03/04/05 还在写
```

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

## 已确认的下一步

03_aggregate_modes：把 raw label 聚类 → 每聚类 LLM 摘 canonical name → 人工审 v1 schema 冻结

04_apply_map_full：用冻结 schema 跑全量 8135 trial

05_report：算 paired delta（skill - raw）+ bootstrap CI，stratified by failure mode
