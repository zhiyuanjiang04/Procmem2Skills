# Failure Taxonomy Pipeline

## Goal

回答 "skill 减少了哪类失败 / 引入了哪类新问题"，用 LLM 在 paired trace 上对比标注产出因果叙述，再 aggregate 成数字。

## Inputs

| 文件 | 内容 |
|------|------|
| `outputs/manifest.jsonl` | 8135 trial 索引（benchmark / setting / arm / task / status / reward + 路径） |
| `outputs/canonical_mode_map.v1.json` | v1 taxonomy 12 个 mode 的定义（共享词汇表，不强制 cover 但让 LLM 引用） |
| `pm2s-traces/extracted/.../codex.txt` | agent trajectory |
| `pm2s-traces/extracted/.../instruction.md` | 任务原文（workflow arm 会自动包含注入的 workflow） |
| `pm2s-traces/extracted/.../SKILL.md` | 注入的 skill 内容（仅 skill arm） |
| `pm2s-traces/extracted/.../result.json` | 成败 + reward + exception |

## Pipeline

### Stage 1: build-manifest（已完成）

扫描 extracted/，每个 trial 一条记录。脚本 `01_build_manifest.py`。

### Stage 2: sample-label + aggregate-modes（已完成）

stratified 抽 240 trial 调 LLM 打 raw label，再聚合成 v1 canonical taxonomy（12 mode）。脚本 `02_sample_label.py` + `03_aggregate_modes.py`。

### Stage 3: paired-compare（v1 主标注阶段）

**关键变化**：不再独立标每条 trial，而是按 (task, setting) 凑齐多 arm 的 trial 一起送 LLM 对比。

#### 配对策略

对每个 (benchmark, setting) 组合：
1. 找该 setting 下 skill arm 和 workflow arm 都跑过的 task 集合（取交集）
2. 对每个 task，从 raw 池里找同任务的 baseline trial（如果存在）
3. 每个 arm 挑 1 条代表 trial（最新或固定 seed 随机）

产出 triples 列表：`[(task_name, setting, raw_trial?, workflow_trial?, skill_trial?), ...]`

raw 可能不存在（raw 跑的是 benchmark 全量，5s0f/skill arm 只跑子集）。如果 raw 缺失，记录 (workflow, skill) 二元对。

#### LLM prompt 结构

```
你正在对比同一任务在 3 个 condition 下的 agent trajectory。

任务原文（instruction.md）:
<...>

共享词汇表（v1 canonical mode taxonomy）:
- skill_guided_success: ...
- workflow_guided_success: ...
- algorithmic_logic_error: ...
（全部 12 个）

[ARM=raw] status=failure reward=0
codex.txt 节选（首尾各 N 行）: ...
result.json: ...

[ARM=workflow] status=success reward=1
（instruction.md 包含注入的 workflow）
codex.txt 节选: ...

[ARM=skill] status=success reward=1
注入的 SKILL.md:
<...>
codex.txt 节选: ...

请输出 JSON。
```

#### LLM 输出 schema (v1 paired label)

```json
{
  "task_name": "<task>",
  "benchmark": "<bench>",
  "setting": "<setting>",
  "per_arm": {
    "raw": {
      "mode": "<one of v1 12 modes>",
      "status": "success | failure",
      "evidence_quote": "verbatim <=160 chars from codex.txt"
    },
    "workflow": {...},
    "skill": {...}
  },
  "deltas": {
    "workflow_vs_raw": {
      "what_changed": "1-2 sentences",
      "net_effect": "fixed | regressed | unchanged | mixed",
      "fixed_mode": ["mode names that workflow eliminated"],
      "introduced_mode": ["mode names that workflow newly caused"]
    },
    "skill_vs_raw": {...},
    "skill_vs_workflow": {...}
  },
  "skill_mechanism": "knowledge_injection | procedural_anchor | failure_warning | none | counterproductive",
  "skill_mechanism_reason": "one sentence",
  "workflow_mechanism": "...",
  "confidence": "high | medium | low"
}
```

字段说明:

- `per_arm[arm].mode`: 在 v1 12 mode 里选一个
- `deltas.X_vs_Y.net_effect`: 4 选 1，描述 X 相对 Y 的总影响
- `deltas.X_vs_Y.fixed_mode`: Y 里出现的 mode，X 没有了
- `deltas.X_vs_Y.introduced_mode`: X 里出现的 mode，Y 里没有
- `skill_mechanism`: skill 为什么起作用
  - `knowledge_injection`: skill 提供了 agent 不知道的具体知识
  - `procedural_anchor`: skill 提供步骤模板让 agent 不偏离
  - `failure_warning`: skill 警告了某个 pitfall，agent 因此避开
  - `none`: skill 内容没起作用，agent 没读或没用
  - `counterproductive`: skill 把 agent 带偏了

### Stage 4: aggregate-report（待写）

把所有 paired label 汇总成可读统计：

- **Mode-level paired delta**: 每个 v1 mode 在 (raw, workflow, skill) 下的频次
- **Net-effect 分布**: skill_vs_raw 的 net_effect 在 fixed / regressed / unchanged / mixed 上的比例
- **Mechanism 分布**: skill_mechanism 出现频率（knowledge_injection 占多少 / procedural_anchor 占多少 / ...）
- **Per-setting breakdown**: 上面三组按 setting (5s0f...0s5f) 切片，看 workflow 比例对 skill 影响的趋势
- **Bootstrap CI**: per-task paired delta 做 1000-iter bootstrap 算置信区间
- **Case studies**: 每类典型代表 trial 各挑 1-2 个，整理 evidence

## 配对范围与成本

| 范围 | triples 数 | LLM 调用成本 |
|------|----------|------------|
| 完整：3 个 benchmark × 6 settings × 所有共有 task | ~1700 | ~$170 |
| 中等：3 个 benchmark × 2 极端 setting (5s0f, 0s5f) × 所有共有 task | ~560 | ~$56 |
| Pilot：skillsbench 5s0f 单一组合 | 24 | ~$2.4 |

Stage 3 实际跑：先 pilot 24 triple 验证 prompt 质量 → 中等范围 560 triple 拿主数据 → 看是否需要补全量。

## 模型

全程 Claude Sonnet 4.6 via `claude -p` headless OAuth。同 stage 1-2。

## 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/pair_labels_pilot.jsonl` | Stage 3 pilot 输出 (24 triples) |
| `outputs/pair_labels_v1.jsonl` | Stage 3 主输出 |
| `outputs/_pair_cache/<task>_<setting>.json` | 单 triple 缓存 |
| `outputs/report_v1.md` | Stage 4 aggregate report |
| `outputs/report_v1_tables.json` | Stage 4 raw 数据 |

## 跑法

```bash
# Stage 3 pilot
PARALLEL=2 PILOT=1 python 04_pair_compare.py

# Stage 3 main
PARALLEL=4 SETTINGS=5s0f,0s5f python 04_pair_compare.py

# Stage 4 report
python 05_report.py
```

## 已确认的设计决策

1. **对比标注 over 独立标注**：直接产出因果叙述而非 contingency 推断
2. **v1 12-mode 作为共享词汇**：不强制 LLM 用某个 mode，但提供一致的概念锚点
3. **net_effect + fixed/introduced mode**：双轨记录 — 一个 categorical 总结，一个 list 详情
4. **mechanism 字段单独列**：区分"skill 起作用"和"为什么起作用"，便于 paper 的因果叙事
