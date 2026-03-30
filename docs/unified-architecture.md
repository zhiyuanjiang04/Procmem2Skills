# Unified Architecture

## 核心立场

`procmem2skills` 不走 memory knowledge graph 路线。

我们不把记忆建模为实体、关系和图谱推理问题，也不把重点放在知识库 schema 设计上。项目的中心是一个统一流水线：

`task execution -> trajectory -> workflow -> atomic skill -> runtime reuse`

这里的重点有三个：

- 记忆来源是 `真实任务执行`
- 记忆单元是 `可验证的原子技能`
- 泛化目标是 `跨 harness / agent / task`

## 为什么不是知识图谱

知识图谱路线往往强调：

- 实体抽取
- 关系链接
- 图结构检索
- 静态知识组织

而本项目要解决的是：

- 如何从多步执行中提取程序性模式
- 如何消除重叠 workflow 的冗余
- 如何把经验沉淀成可执行技能包
- 如何让这些技能在后续执行中真正被调用

因此我们的统一层不是 `entity-relation graph`，而是 `event-segment-workflow-skill` 四层过程表示。

## 统一表示

### 1. Event

所有 benchmark 都先映射到统一事件结构：

- `observation`
- `action`
- `tool`
- `result`
- `state_delta`
- `artifacts`
- `success_signal`

### 2. Segment

把长轨迹切成局部子过程，用于消除长程噪声并暴露稳定边界。

### 3. Workflow

把 segment 归纳成结构化流程模板，保留：

- objective
- trigger
- preconditions
- steps
- verification
- failure_modes

### 4. Atomic Skill

把 workflow 进一步压缩成最小可复用能力单元，并打包为标准 skills 目录：

- `SKILL.md`
- `references/`
- `scripts/`
- `assets/`

## 两条评测路径

## 路径 A：离线蒸馏后在线消费

1. 在 benchmark 上跑 agent，收集离线轨迹。
2. 用统一流水线抽取 workflow 和 atomic skills。
3. 按 `skill-creator` 规范兼容的格式打包成 skills。
4. 在后续在线评测中，把这些 skills 作为外部 memory repository 注入 agent。

这条路径适合：

- 构建 bootstrap skill 库
- 做与 AWM 的公平对比
- 控制变量分析 skill 粒度和上下文开销

对应 CLI：

```bash
procmem2skills distill-offline trajectories.jsonl skills/
```

## 路径 B：在线增量更新

1. agent 完成一个 episode。
2. 若任务成功且满足阈值，就把 trajectory 追加到 archive。
3. 对 archive 重跑蒸馏，得到更新后的 skill repository。
4. 后续 episode 继续消费最新 skills。

这条路径适合：

- 持续学习实验
- 跨任务 curriculum
- 观察 skill repository 的增长和稳定性

对应 CLI：

```bash
procmem2skills update-online latest-trajectories.jsonl archive/all.jsonl skills/
```

## Benchmark 接入策略

统一架构把 benchmark 接入拆成两层：

### 第一层：Replay / Import

把 benchmark 原始数据或 trace 导入统一 trajectory schema：

```bash
procmem2skills import-benchmark mind2web raw.json imported.jsonl
procmem2skills import-benchmark webarena traces.json imported.jsonl
```

这一层适合本地开发和 CPU-only 验证，因为它不要求 live harness、Docker 或 GPU。

### 第二层：Live Harness

当环境允许时，再把真实 benchmark 运行时接到统一 recorder：

- WebArena / WebArena-Verified
- ALFWorld interactive env
- Terminal-Bench harness

这层的职责是产生原始 episode，不改变上游的 `segment -> workflow -> cluster -> atomic skill` 流水线。

为保证本地开发时也能验证在线闭环，当前代码还提供了一个 `mock interactive adapter + live runner` 组合，用于检查：

- skill retrieval 是否被注入到决策阶段
- action -> step result -> trajectory recording 是否闭环
- 在线产生的 trajectory 是否能继续流回增量更新路径

## 跨 Harness / Agent / Task 泛化

统一架构的泛化不是一句空话，而是三层明确目标：

### 跨 Harness

不同 benchmark 只在 adapter 层不同。只要能映射到统一事件 schema，就能进入同一条蒸馏管线。

### 跨 Agent

trajectory 顶层显式记录 `agent` 和 `harness`，skill provenance 也保留这些来源。因此可以比较：

- 同一 harness 上不同 agent 的 skill overlap
- 同一 skill 是否能被不同 agent 消费

### 跨 Task

workflow 和 atomic skill 的目标就是把具体任务表层差异剥离掉，只保留可复用的程序模式。

## 当前代码落点

代码已经提供：

- `src/procmem2skills/models.py`
  统一 schema
- `src/procmem2skills/adapters/`
  四个 benchmark 的 profile 和 raw step normalizer
- `src/procmem2skills/importers/`
  四个 benchmark 的 replay/import 接口
- `src/procmem2skills/segmenter/heuristics.py`
  基于边界信号的切分器
- `src/procmem2skills/inducer/workflow.py`
  workflow induction
- `src/procmem2skills/miner/clustering.py`
  CPU-friendly workflow normalization、clustering 与 dedup
- `src/procmem2skills/miner/atomic_skills.py`
  rule-based atomic skill mining
- `src/procmem2skills/evaluation/runner.py`
  统一 live execution runner
- `src/procmem2skills/recorder/live.py`
  在线 episode recorder
- `src/procmem2skills/packager/skill_writer.py`
  标准 skill 目录打包
- `src/procmem2skills/runtime/update.py`
  在线增量更新

下一阶段要补的不是图谱，而是：

- benchmark 真实 adapter 接入
- 更强的 clustering / dedup
- skill retrieval 和在线调用反馈
