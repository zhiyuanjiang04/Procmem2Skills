# ProcMem2Skills（中文）

`procmem2skills` 是一个统一的轨迹蒸馏框架：把 agent 在真实任务中的执行轨迹，转成可复用、可组合、可迁移的标准 skills。

核心链路：

`trajectory -> workflow -> cluster -> dedup -> atomicize -> skill package`

## 1. 项目目标

这个项目聚焦三个问题：

- 怎么把“执行经验”从文本记忆变成“可执行能力单元”。
- 怎么把重叠 workflow 去冗余后得到跨任务可泛化的原子 skill。
- 怎么在真实 benchmark/harness/agent 中验证 skill 对弱模型有迁移增益。

对应到实现上，当前不是知识图谱路线，而是统一技能接口路线：收集轨迹、蒸馏技能、注入执行、再评估闭环。

## 2. 当前能力总览

已接入 benchmark（导入/蒸馏链路）：

- `mind2web`
- `webarena`
- `alfworld`
- `terminal-bench`

已支持两种 memory 模式：

- `offline`：批量离线轨迹蒸馏
- `online`：增量更新 skill repository

已支持双通道技能产物（默认）：

- `--success`：执行 playbook  
  文件结构：`SKILL.md + scripts/apply.sh + scripts/verify.sh`
- `--failure`：诊断/恢复 skill  
  文件结构：`SKILL.md + scripts/recover.sh + scripts/verify.sh`

已支持两类技能选择逻辑：

- `agent-first`（默认）：让 agent 在候选 skill 中先做选择
- `vector`：向量/词向量检索 fallback

## 3. 统一架构

### 3.1 Distillation Pipeline

1. 轨迹导入：将各 benchmark 原始格式映射到统一 `Trajectory` schema。  
2. 轨迹切分：按工具切换、成功信号等规则切 segment。  
3. workflow 归纳：从 segment 生成 workflow candidate。  
4. 聚类与去重：支持 lexical 与 embedding-dbscan。  
5. 原子化：提取 atomic skills。  
6. 技能打包：写成标准 skill 目录，可被 agent runtime 消费。  

### 3.2 Success/Failure 分流

- 蒸馏阶段优先使用成功轨迹生成主技能。
- 失败轨迹和失败信号用于恢复技能（failure channel）。
- LLM skill-creator 模式下，先做“轨迹整合”，再生成 success/failure 双技能。

### 3.3 运行时注入

- `skill-aware` 自定义 agent：每步检索 skill，支持 `agent-first` 选择。
- `native` agent（重点支持 codex/opencode）：通过 prompt template 注入技能。
- native 注入策略：优先 success skill，遇到失败信号再触发 failure skill。

### 3.4 Workflow 输出契约（最新）

为了同时满足“可聚类、信息不丢、可直接注入 context”，workflow induction 产物采用双层结构：

- `workflow.steps`：仅 action steps（用于聚类、去重、atomic skills 蒸馏）
- `workflow.metadata`：保留完整轨迹信息（用于分析、回放、上下文注入）

当前关键字段：

- `event_trace`：压缩版事件轨迹（分析友好）
- `event_trace_full`：完整事件原文（不截断，信息保真）
- `context_payload`：面向 agent 注入的结构化上下文
  - `objective` / `trigger` / `preconditions`
  - `steps`（action 视角）
  - `timeline`（全事件时序视角）
  - `verification` / `failure_modes`
- `information_coverage`：覆盖统计（action/non-action/result/success_signal 等）
- `cluster_reservation`：聚类预留特征
  - `step_tokens` / `step_operation_signatures` / `step_bigrams`
  - `action_families` / `tool_signature`
  - `objective_token` / `verification_tokens` / `failure_tokens`

说明：

- 单个 segment 内完整信息由 `event_trace_full + context_payload.timeline` 保证。
- 整个 episode 的完整语义由多个 segment workflow 组合得到。

### 3.5 全量 Workflow Induction 导出（task 聚合单文件）

按照"先汇总 rollout 再处理"的流程，workflow induction 现在提供两层入口：

- 函数模块：`src/procmem2skills/inducer/workflow_export.py`
- 批处理入口：`scripts/server/run_full_workflow_induction.py`

运行示例：

```bash
PYTHONPATH=src python3 scripts/server/run_full_workflow_induction.py \
  --input experiments/<bench>/<exp>/imported/live-trajectories.jsonl \
  --output experiments/<bench>/<exp>/workflows-by-task.json
```

输出 JSON 契约（顶层 key 为 task，value 为该 task 的多次尝试列表）：

```json
{
  "<task_id>": [
    {
      "episode_id": "...",
      "attempt_index": 1,
      "status": "success",
      "workflows": [ ... ]
    },
    {
      "episode_id": "...",
      "attempt_index": 2,
      "status": "failure",
      "workflows": [ ... ]
    }
  ]
}
```

规则：

- success / failure 都会执行 induction 并保留
- error 轨迹会被丢弃（不进入导出文件）
- 每次运行只产出一个按 task 聚合的 JSON，便于后续 context injection / clustering / dedup

## 4. 快速开始（本地）

### 4.1 安装

要求：`python>=3.10`。

```bash
uv venv .venv
uv pip install -e . --python .venv/bin/python
```

### 4.2 CLI 可用命令

```bash
./.venv/bin/procmem2skills --help
```

核心命令：

- `list-benchmarks`
- `import-benchmark`
- `distill-offline`
- `update-online`
- `run-mock-live`
- `summarize-taxonomy`
- `evaluate-replay-transfer`

### 4.3 最小可跑示例

```bash
./.venv/bin/procmem2skills import-benchmark mind2web \
  examples/raw-mind2web.json \
  tmp/imported/mind2web.jsonl

./.venv/bin/procmem2skills distill-offline \
  tmp/imported/mind2web.jsonl \
  tmp/skills/mind2web
```

### 4.4 Workflow->Skill 聚合模式（新增）

`distill-offline` / `update-online` / server harbor 入口现在支持两种补充设置：

- `--workflow-aggregation-mode per-task`
  - 对每个 task 的 workflows 单独聚合并产出技能。
  - 默认给 skill id 加 task 前缀，避免不同 task 同名 skill 冲突。
- `--workflow-aggregation-mode global-dbscan-qwen`
  - 对全量 workflows 统一走 `DBSCAN + Qwen embedding + dedup + skill generation`。
  - 自动强制 `embedding-dbscan`，默认模型 `Qwen/Qwen3-Embedding-0.6B`，并启用 embedding strict。

示例：

```bash
./.venv/bin/procmem2skills distill-offline \
  tmp/imported/terminal-bench.jsonl \
  tmp/skills/tb-per-task \
  --workflow-aggregation-mode per-task

./.venv/bin/procmem2skills distill-offline \
  tmp/imported/terminal-bench.jsonl \
  tmp/skills/tb-global-qwen \
  --workflow-aggregation-mode global-dbscan-qwen \
  --cluster-embedding-base-url http://127.0.0.1:8000/v1
```

### 4.5 LLM Skill Creator 风格控制（codex / cc / opencode）

skill 生成新增风格参数：

- `--skill-creator-agent-style codex|cc|claude-code|opencode`
- `--skill-creator-system-prompt "<your extra system constraints>"`
- `--skill-generation-strict-llm`（无 key / 生成失败时直接报错，不回退 heuristic）

示例：

```bash
./.venv/bin/procmem2skills distill-offline \
  tmp/imported/terminal-bench.jsonl \
  tmp/skills/tb-codex \
  --workflow-aggregation-mode global-dbscan-qwen \
  --skill-generation-mode llm-agent \
  --skill-creator-model openai/gpt-5.3-codex \
  --skill-creator-agent-style codex \
  --skill-generation-strict-llm
```

### 4.6 Controlled Workflow-Mix Study（固定 s/f 配比）

新增统一入口：

- `scripts/server/study/controlled/run.py`（推荐，分层短路径）
- `scripts/server/run_controlled_workflow_skill_study.py`（兼容别名）

用途：

- 从 `task -> attempts` 的 grouped workflows 中做 task-level 资格筛选。
- 按固定条件（如 `m=5,n=0` / `m=4,n=1`）对每个 task 随机抽样 workflows。
- 产出标准化 `atomic_skills.json`，并可直接调用 `skill-creator`（LLM）落地技能仓库。

关键筛选参数（对应“先筛任务，再注入固定数量 workflow”）：

- `--minimum-success-pool-size N`
- `--minimum-failure-pool-size N`
  - 例如设置 `--minimum-success-pool-size 5 --minimum-failure-pool-size 5`，即可只保留至少 `5s+5f` 的 task。
- `--require-counts-for-all-conditions`
  - 保证同一 task 可覆盖整套对比条件，而不是只覆盖其中一部分。

示例（先做筛选与抽样，不触发 LLM）：

```bash
PYTHONPATH=src ./.venv-py312/bin/python scripts/server/study/controlled/run.py \
  --workflow-input experiments/skillsbench/<exp>/workflows-by-task.json \
  --output-dir experiments/skillsbench/<exp>/controlled-mix \
  --mix 5 0 --mix 4 1 --mix 3 2 \
  --minimum-success-pool-size 5 \
  --minimum-failure-pool-size 5 \
  --skip-materialize
```

示例（统一 prompt + codex skill-creator 生成技能）：

```bash
PYTHONPATH=src ./.venv-py312/bin/python scripts/server/study/controlled/run.py \
  --workflow-input experiments/skillsbench/<exp>/workflows-by-task.json \
  --output-dir experiments/skillsbench/<exp>/controlled-mix \
  --mix 5 0 --mix 4 1 --mix 3 2 \
  --minimum-success-pool-size 5 \
  --minimum-failure-pool-size 5 \
  --skill-creator-model openai/gpt-5.3-codex \
  --skill-creator-agent-style codex
```

补充：

- 支持单条件写法：`--success-count m --failure-count n`
- 默认批量条件：`(5,0) (4,1) (3,2) (2,3) (1,4) (0,5)`（未传 `--mix`/单条件时启用）
- 默认 `minimum-mixed-success-count=0`、`minimum-mixed-failure-count=0`（允许 0 条）

## 5. 服务器统一入口（推荐）

服务器侧只用两个入口脚本：

```bash
bash scripts/server/setup_benchmark.sh <target>
bash scripts/server/run_benchmark_smoke.sh <target>
```

`setup_benchmark.sh` targets：

- `core`
- `mind2web`
- `alfworld`
- `terminal-bench`
- `skillsbench`
- `webarena`
- `all`

`run_benchmark_smoke.sh` targets：

- `mind2web`
- `alfworld`
- `terminal-bench`
- `terminal-bench-harbor`
- `skillsbench-harbor`
- `webarena`
- `all`

建议顺序：

```bash
bash scripts/server/setup_benchmark.sh core
bash scripts/server/setup_benchmark.sh alfworld
bash scripts/server/setup_benchmark.sh terminal-bench
bash scripts/server/setup_benchmark.sh skillsbench
bash scripts/server/setup_benchmark.sh webarena

bash scripts/server/run_benchmark_smoke.sh mind2web
bash scripts/server/run_benchmark_smoke.sh alfworld
bash scripts/server/run_benchmark_smoke.sh terminal-bench
bash scripts/server/run_benchmark_smoke.sh terminal-bench-harbor
bash scripts/server/run_benchmark_smoke.sh skillsbench-harbor
bash scripts/server/run_benchmark_smoke.sh webarena
```

### 5.1 服务器脚本使用索引（速查）

短路径入口（推荐）：

- `python scripts/server/study/controlled/run.py --help`
- `python scripts/server/study/induce/workflows.py --help`
- `python scripts/server/study/formal/run.py --help`
- `python scripts/server/study/live/skills.py --help`
- `python scripts/server/study/live/terminal.py --help`
- `python scripts/server/study/transfer/run.py --help`


| 脚本 | 用途 | 推荐用法 |
| --- | --- | --- |
| `scripts/server/setup_benchmark.sh` | 准备本地/服务器 benchmark 依赖环境 | `bash scripts/server/setup_benchmark.sh <target>` |
| `scripts/server/run_benchmark_smoke.sh` | 运行各 benchmark 的最小 smoke 检查 | `bash scripts/server/run_benchmark_smoke.sh <target>` |
| `scripts/server/run_formal_experiment.sh` | 统一正式实验入口（四类 benchmark） | `bash scripts/server/run_formal_experiment.sh --benchmark <name> --experiment-id <id> ...` |
| `scripts/server/run_terminal_bench_harbor_experiment.sh` | Terminal-Bench 单阶段 Harbor live 实验（可 bootstrap skills） | `bash scripts/server/run_terminal_bench_harbor_experiment.sh --experiment-id <id> --model <model> ...` |
| `scripts/server/run_skillsbench_harbor_experiment.sh` | SkillsBench Harbor live 实验（registry 优先，local path 回退） | `bash scripts/server/run_skillsbench_harbor_experiment.sh --experiment-id <id> --model <model> ...` |
| `scripts/server/run_terminal_bench_transfer_study.sh` | Terminal-Bench 强模型->弱模型迁移三阶段实验 | `bash scripts/server/run_terminal_bench_transfer_study.sh --experiment-id <id> --strong-model <m1> --weak-model <m2> ...` |
| `scripts/server/run_terminal_bench_transfer_matrix_parallel.sh` | 多弱模型并行 matrix 迁移实验 | `bash scripts/server/run_terminal_bench_transfer_matrix_parallel.sh --experiment-prefix <prefix> --strong-model <m1> --weak-model <m2> ...` |
| `history/code/scripts/server/legacy/*` | 归档脚本（历史参考，不进入主流程） | 仅查阅，不建议执行 |

说明：

- `scripts/server/common.sh` 是公共函数库（环境准备、python 选择、slug 规范化），不直接执行。
- 已归档脚本位于 `history/code/scripts/server/legacy/`：`check_capabilities.sh`、`build_terminal_bench_metadata_tree.py`、`generate_benchmark_trees.py`。
- 在服务器上优先使用 `setup_benchmark.sh` + `run_benchmark_smoke.sh` 验证环境，再进入 formal/transfer 实验。
- 调参建议：优先改 Python 入口参数（`run_* .py`），shell 仅保留薄封装与参数转发，避免难维护的复杂 shell 逻辑。

## 6. 正式实验入口

统一正式实验接口：

```bash
bash scripts/server/run_formal_experiment.sh \
  --benchmark terminal-bench \
  --experiment-id tb-formal-v1 \
  --memory-mode offline \
  --input-path <path-to-data-or-job-dir>
```

支持 benchmark：

- `mind2web`
- `webarena`
- `alfworld`
- `terminal-bench`

支持 memory mode：

- `offline`
- `online`

产物默认落在：

`experiments/<benchmark>/<experiment-id>/`

## 7. Terminal-Bench / SkillsBench 专项实验

### 7.1 Terminal-Bench Harbor live wrapper

```bash
bash scripts/server/run_terminal_bench_harbor_experiment.sh \
  --experiment-id tb-live-demo \
  --bootstrap-input examples/raw-terminal-bench.json \
  --model anthropic/claude-sonnet-4 \
  --n-tasks 1 \
  --dry-run
```

说明：

- `--dataset` 支持 Harbor registry 中的不同数据集/版本（例如 `terminal-bench@2.0`、`terminal-bench-sample@2.0`）。
- 额外 Harbor 原生参数可直接透传（脚本未声明的参数会自动附加到 `harbor run` 命令），便于随时调参。
- 并行默认值已调整为 `--n-concurrent 8`；资源较紧时建议回退到 `5`。

去掉 `--dry-run` 后会执行：

1. bootstrap 轨迹蒸馏技能库  
2. 启动 Harbor job（native 或 skill-aware）  
3. 导入结果为统一 trajectory  

### 7.2 SkillsBench Harbor wrapper

```bash
bash scripts/server/run_skillsbench_harbor_experiment.sh \
  --experiment-id sb-live-demo \
  --model openai/gpt-5.3-codex \
  --n-tasks 1 \
  --dry-run
```

说明：

- 默认 `--source-mode auto`：优先使用 Harbor registry `--dataset skillsbench`，找不到时回退到本地 `benchmarks/skillsbench/tasks`。
- 可显式指定：
  - registry：`--source-mode dataset --dataset skillsbench`
  - local path：`--source-mode path --skillsbench-path <path>`
- 默认导入映射使用 `terminal-bench` importer（Harbor trial 结构兼容路径），并保持统一 manifest/runbook 产物。
- 并行默认值同样为 `--n-concurrent 8`。

### 7.3 强弱模型迁移实验（默认 codex-native）

```bash
bash scripts/server/run_terminal_bench_transfer_study.sh \
  --experiment-id tb-transfer-v1 \
  --strong-model anthropic/claude-opus-4.5 \
  --weak-model anthropic/claude-sonnet-4.5 \
  --task-name build-cython-ext
```

当前默认参数是：

- `--strong-agent-mode native`
- `--weak-agent-mode native`
- `--skill-agent-mode native`
- `--strong-native-agent codex`
- `--weak-native-agent codex`
- `--skill-native-agent codex`

即优先走“native + codex + skill prompt 注入”路径。

### 7.4 全量任务与分片并行

`run_terminal_bench_transfer_study.sh` 现在支持直接拉取 `terminal-bench@2.0` 全任务并按分片运行：

```bash
bash scripts/server/run_terminal_bench_transfer_study.sh \
  --experiment-id tb-full-shard0 \
  --strong-model openai/gpt-5.3-codex \
  --weak-model openai/gpt-5.1 \
  --all-tasks \
  --task-shard-count 4 \
  --task-shard-index 0 \
  --strong-attempts 1 \
  --weak-attempts 1 \
  --skill-attempts 1 \
  --skill-generation-mode llm-agent \
  --skill-creator-model openai/gpt-5.3-codex \
  --base-url https://openrouter.ai/api/v1
```

### 7.5 Harbor 命名规范（新）

为避免 Harbor 侧 job 名称和本地结果目录混乱，live/transfer 两条链路都启用了统一命名：

- `experiment-id` 会标准化为小写 slug（仅保留 `a-z0-9-`）
- Harbor job 名采用结构化格式：`<benchmark>-<phase>-<experiment>-<model>`
- benchmark 前缀约定：
  - terminal-bench -> `tb`
  - skillsbench -> `sb`
- 默认长度控制：
  - `experiment-id` 最长 80
  - Harbor `job_name` 最长 96
- 每次运行会在 manifest 中保留：
  - `requested_experiment_id`（原始输入）
  - `experiment_id`（标准化后）
  - `harbor_job_name` 或 `phase_job_names`
- `harbor-jobs/` 下会创建规范别名（symlink）指向实际 Harbor job 目录，便于人工检索与二次脚本处理
- `run_terminal_bench_transfer_matrix_parallel.sh` 会自动规范化：
  - `--experiment-prefix` 先 slug 化再截断
  - 每个弱模型生成 `<prefix>-to-<weak-model-slug>` 形式的 canonical `experiment-id`

规范化规则（与代码一致）：

```text
lowercase -> 非 [a-z0-9] 连续字符替换为 - -> 去掉首尾 - -> 长度截断
空字符串兜底为 experiment
```

推荐模板（便于团队协作检索）：

- terminal-bench live：`tb-live-<model>-<subset>-<yyyymmdd>`
- skillsbench live：`sb-live-<model>-<subset>-<yyyymmdd>`
- transfer：`tb-transfer-<strong>-to-<weak>-<subset>-<yyyymmdd>`
- matrix 子实验：`<canonical-prefix>-to-<weak-model-slug>`

示例：

- 输入 `TB Full/Shard0` -> `tb-full-shard0`
- 输入 `claude-sonnet-4.5` -> `claude-sonnet-4-5`
- 组合后 `tb-full-shard0-to-claude-sonnet-4-5`
- 服务器脚本统一复用 `scripts/server/common.sh` 中的 `normalize_slug`/`truncate_slug`，避免命名规则漂移

如果要同时跑多个弱模型，使用并行矩阵脚本：

```bash
bash scripts/server/run_terminal_bench_transfer_matrix_parallel.sh \
  --experiment-prefix tb-full-parallel-shard0 \
  --strong-model openai/gpt-5.3-codex \
  --weak-model openai/gpt-5.1 \
  --weak-model openai/gpt-5.2 \
  --weak-model anthropic/claude-sonnet-4.5 \
  --parallel-jobs 3 \
  -- \
  --all-tasks \
  --task-shard-count 4 \
  --task-shard-index 0 \
  --strong-attempts 1 \
  --weak-attempts 1 \
  --skill-attempts 1 \
  --skill-generation-mode llm-agent \
  --skill-creator-model openai/gpt-5.3-codex \
  --base-url https://openrouter.ai/api/v1
```

## 8. Skill Failure Study（对应 ProcMem2Skills 失败归因文档）

新增统一入口：

```bash
./.venv/bin/procmem2skills run-skill-failure-study \
  <trajectory-jsonl> \
  <skill-repository-dir> \
  <output-report-json> \
  --retrieval-methods page-index,qwen3-embedding \
  --injection-strategies no-skill,direct-inline,claude-style-progressive \
  --pool-sizes 50,500,5000 \
  --qwen3-embedding-local-base-url http://127.0.0.1:8000/v1 \
  --qwen3-embedding-model Qwen/Qwen3-Embedding-0.6B \
  --qwen3-embedding-strict \
  --procedural-dbscan-eps 0.35 \
  --procedural-dbscan-min-samples 2 \
  --self-generated-modes all-procedural-memories,success-only-memories,skills-plus-procedural-memory \
  --split-modes in-task,cross-task-holdout \
  --benchmarks terminal-bench,alfworld \
  --agents codex,claude-code,opencode
```

参数轴和文档需求对齐关系：

- `--retrieval-methods`: page-index / qwen3-embedding（保留 context-injection, embedding-based 兼容）
  - `qwen3-embedding` 为真实 embedding 检索：优先尝试本地 endpoint，再回退远端 endpoint
- `--qwen3-embedding-local-base-url`: 本地 embedding 服务地址（优先）
- `--qwen3-embedding-base-url`: 远端 fallback 地址
- `--qwen3-embedding-model`: qwen embedding 模型名
- `--qwen3-embedding-strict`: 严格模式；embedding 不可用时直接失败，不回退 lexical
- `--injection-strategies`:
  - `no-skill`: 无技能注入基线
  - `direct-inline`: SkillRL 风格，命中技能后直接整段注入 prompt
  - `claude-style-progressive`: Claude/Codex/OpenCode 风格，先候选卡片再按需展开
- `--pool-sizes`: noisy skill pool（50 / 500 / 5000）
- `--self-generated-modes`: all procedural / success only / skills + procedural memory
  - `all-procedural-memories`: 全量轨迹先归纳 workflows，按 benchmark 形成静态 procedural memory（含 success/failure 标记），按 task 直接注入上下文
  - `success-only-memories`: 仅成功轨迹归纳 procedural memory，按 task 直接注入上下文
  - `skills-plus-procedural-memory`: 先从成功轨迹归纳 procedural memory，再聚类/去重并蒸馏 atomic skills；检索时优先拼接 task-procedural + atomic skills（可与外部 skills 一起）
  - procedural memory 去重默认走 `DBSCAN + qwen embedding`（真实 embedding）
- `--split-modes`: in-task 与 cross-task generalization
- `--benchmarks`, `--agents`: benchmark 与 agent 过滤
- `--terminal-bench-datasets`: terminal-bench 数据集版本过滤（如 `terminal-bench@2.0,terminal-bench-sample@2.0`）
- `--terminal-bench-parameter-keys`: terminal-bench 参数字段注入/分析维度选择
- 默认 `--batch-rollout-required`：强制先聚合全任务 rollout，再 distill atomic skills

输出报告中会给出：

- 每个实验 cell 的 `hit@k`、失败类别分布、trace 级明细
- 每个 cell 的 `injection_strategy`、`retrieved_skill_count`、progressive 展开统计
- 失败类别：unable retrieve / pick wrong / pick related but fail / agent misuse / skill error / misled by noise
- 总体摘要：cell 数量与平均 hit@k

## 9. 目录结构（精简版）

```text
docs/                          研究文档与方法说明
examples/                      小样本输入与示例轨迹
scripts/server/                服务器实验入口脚本（active）
scripts/server/study/          短路径入口（推荐日常使用）
history/code/scripts/server/legacy/  归档脚本（历史参考，默认不发布）
history/results/               历史实验结果归档
src/procmem2skills/
  adapters/                    benchmark profile 和适配层
  importers/                   原始数据 -> Trajectory
  segmenter/                   trajectory 切分
  inducer/                     workflow 归纳
  miner/                       聚类/去重/原子化
  packager/                    skill 生成与落盘
  runtime/                     检索、选择与在线更新
  integrations/                Harbor/native agent 集成
  evaluation/                  replay/live/transfer 评估
tests/                         单元测试
```

## 10. 文档导航

- [项目蓝图](./docs/project-blueprint.md)
- [研究计划](./docs/research-plan.md)
- [Benchmark 分析](./docs/benchmark-analysis.md)
- [统一架构](./docs/unified-architecture.md)
- [服务器部署说明](./docs/server-benchmark-deployment.md)
- [正式实验接口说明](./docs/formal-experiment-interface.md)

## 11. 已知边界

- 本地默认只保证离线链路与 mock/live dry-run；完整 live benchmark 依赖服务器环境。
- `terminal-bench` live 依赖 Docker 与 Harbor 任务环境。
- `webarena` live 依赖 BrowserGym + Playwright + Chromium + self-hosted 站点环境变量。
- LLM skill-creator 需要 `OPENROUTER_API_KEY` 或 `OPENAI_API_KEY`。

## 12. GitHub 开源准备（code-only）

为便于开源仓库初始化，默认策略是“仅提交代码与必要文档，不提交实验结果、数据和废弃内容”。

- 结果目录：`experiments/` 忽略
- 数据目录：`benchmarks/`、`data/`、`datasets/` 忽略
- 历史代码：`history/code/` 忽略
- 历史结果：`history/results/` 忽略
- 本地缓存与虚拟环境目录忽略

如果需要单独共享某次实验结果，建议另建 release asset 或独立对象存储，不直接入主仓库。

## 13. 参考

- Agent Workflow Memory: <https://github.com/zorazrw/agent-workflow-memory>
- Agent Skills 规范: <https://agentskills.io/skills>
- Anthropic Skills 文章: <https://www.anthropic.com/engineering/skills-to-improve-agentic-coding>

## 14. 开源协作规范

- [LICENSE](./LICENSE)
- [贡献指南](./CONTRIBUTING.md)
- [行为准则](./CODE_OF_CONDUCT.md)
- [安全策略](./SECURITY.md)
