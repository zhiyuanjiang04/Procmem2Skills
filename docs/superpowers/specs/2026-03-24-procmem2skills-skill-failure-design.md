# ProcMem2Skills Skill Failure Study Design

## 1. 输入文档与目标

输入文档：`Documents/PM2Skills/ProcMem2Skills.pdf`（2026-03-24 提取）。

文档核心目标：回答 “skills 在什么情况下有效，什么情况下失败”，并将失败拆解为 `curation + retrieval + usage` 三类链路问题，随后在多 benchmark、多 agent、多检索配置下做可复现实验。

## 2. 需求逐条拆解（细粒度）

### 2.1 研究问题拆解

- R1: 技能失败归因框架必须覆盖：
  - R1.1 `unable to retrieve related skills`
  - R1.2 `pick related skills but fail to use`
    - R1.2.a `agent misuse of related skills`
    - R1.2.b `error inside skills themselves`
  - R1.3 `misled by noisy skills`
- R2: 失败归因需要落地为可计算标签与统计口径（不是只写文字结论）。

### 2.2 Benchmark 需求拆解

- B1: 面向 hundreds of tasks 的评测规模。
- B2: Terminal-Bench / Skills-Bench（harbor 体系）优先。
- B3: ALFWorld 作为相关工作常见环境。
- B4: 可继续扩展其他任务源（结构上可插拔）。

### 2.3 Agent 需求拆解

- A1: 需要支持直接使用 skill 的 agent。
- A2: 至少兼容 Claude Code / Codex / Opencode 这类模式（在本仓库语义上体现为 harbor native/skill-aware 兼容）。

### 2.4 实验设置需求拆解

- E1: Rollout -> Analyze traces 的闭环。
- E2: Self-generated skills 组：
  - E2.1 执行一次任务形成 memory
  - E2.2 重复执行 n 次形成 procedural memories
  - E2.3 比较 “全部 procedural memories / 仅成功 memories / skills + procedural memory”
- E3: Retrieve from skills pool 组：
  - E3.1 pool size = 50 / 500 / 5000
  - E3.2 retrieval method = page index / context injection / embedding-based
  - E3.3 trace 分析维度：选错 skill、选对但不会用、skill 本身错误、噪声误导
- E4: Cross-task generalization 组：
  - E4.1 从相似任务轨迹生成 skills
  - E4.2 评估 unseen tasks 上的迁移增益

## 3. 现有代码能力映射（结合真实库代码）

### 3.1 已覆盖能力

- 导入与 benchmark 适配：
  - `src/procmem2skills/importers/terminal_bench.py`
  - `src/procmem2skills/importers/alfworld.py`
  - `src/procmem2skills/adapters/terminal_bench.py`
  - `src/procmem2skills/adapters/alfworld.py`
- 蒸馏主链路（trajectory -> workflow -> cluster -> skills）：
  - `src/procmem2skills/evaluation/pipeline.py`
  - `src/procmem2skills/miner/clustering.py`
- 运行时检索与 skill 元数据索引：
  - `src/procmem2skills/runtime/retrieval.py`
- Harbor / Terminal-Bench 真实运行和回放：
  - `src/procmem2skills/integrations/harbor_terminal_experiment.py`
  - `src/procmem2skills/integrations/harbor_transfer_study.py`
- 失败信号抽取：
  - `src/procmem2skills/analysis/failure.py`

### 3.2 关键缺口

- G1: 缺少一个“直接对应 PDF 研究问题”的统一实验矩阵层。
- G2: 缺少对 `pool size x retrieval method` 的结构化 sweep 与统一输出。
- G3: 缺少对 `R1` 分类标签（unable/pick-wrong/pick-right-but-fail/noisy）的统一规则与聚合。
- G4: 缺少 cross-task holdout 泛化实验的统一入口（当前更多是 task-level transfer study，尚未抽象为复用框架）。

## 4. 设计决策

### 4.1 新增模块

新增 `src/procmem2skills/research/skill_failure_study.py`，提供四层结构：

- Layer 1: StudyConfig / ExperimentCell（实验配置）
- Layer 2: SkillPoolBuilder（50/500/5000 + noise injection）
- Layer 3: RetrievalExecutor（page-index / context-injection / embedding-based）
- Layer 4: FailureAttributor（将每条 case 归因为 R1 类别）

### 4.2 与现有库代码的耦合点

- 使用 `SkillDistillationPipeline` 构造 task-level oracle skills（对应 self-generated skills 设定）。
- 使用 `SkillIndex.search(scope="metadata")` 实现 page-index。
- 使用 `SkillIndex.search(scope="fulltext")` 近似 embedding-based（当前仓库已有 fulltext vector 语义）。
- 使用 `build_failure_analysis_from_trajectories` 和 `extract_failure_signals_from_text` 汇聚失败信号。
- 使用 `load_trajectories` / `write_trajectories` 复用 I/O 管线。

### 4.3 实验输出

- cell 级摘要（method + pool_size + split）：
  - case 数、hit@k、oracle coverage
  - 失败分类分布（R1.1/R1.2/R1.2.a/R1.2.b/R1.3）
- trace 级明细：
  - query、oracle skills、retrieved skills、分类标签、failure signals
- 总结报告：
  - 按 method 与 pool size 聚合
  - 按 task 聚合
  - cross-task holdout 对比

## 5. 实验变量定义

- 自变量：
  - retrieval method: `page-index | context-injection | embedding-based`
  - skill pool size: `50 | 500 | 5000`
  - skill memory mode: `all-procedural | success-only | skills+procedural`
  - split mode: `in-task | cross-task-holdout`
- 因变量：
  - retrieval hit@k
  - oracle recall
  - 失败分类占比
  - cross-task 相对变化

## 6. 失败分类规则（实现口径）

- `unable-to-retrieve-related-skills`:
  - oracle skills 非空，但 retrieved 与 oracle 交集为空，且无明显 noise 竞争证据。
- `misled-by-noisy-skills`:
  - oracle skills 非空，交集为空，且 top retrieved 全为 noise/padding skills。
- `pick-related-skills-but-fail-to-use`:
  - retrieved 命中 oracle，但该 trajectory 最终失败。
- `agent-misuse-of-related-skills`:
  - 命中 oracle 且失败，同时命令轨迹显示未执行/偏离主要 skill 操作。
- `error-inside-skills-themselves`:
  - 命中 oracle 且失败，失败信号与 skill 里的 failure/verification 线索高度重合。

注：`agent misuse` 与 `skill error` 目前是启发式判定，后续可接入 Harbor trace 更细粒度字段提升准确率。

## 7. 非目标（本轮）

- 不在本轮实现完整线上 Harbor 大规模调度器。
- 不在本轮引入外部向量数据库；先用现有 `SkillIndex` metadata/fulltext 两路。
- 不在本轮改变现有 distillation 算法主体。

## 8. 验证策略

- 单元测试优先验证：
  - pool 扩展与 noise 注入
  - 三种 retrieval strategy 行为一致性
  - 失败分类规则可复现
  - cross-task split 逻辑稳定
- smoke 级验证：
  - 小样本 trajectories 跑通完整 report 产物生成。
