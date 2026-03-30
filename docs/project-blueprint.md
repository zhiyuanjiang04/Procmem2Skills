# procmem2skills 项目蓝图

## 1. 一句话讲清楚项目

我们要研究的是：能否把真实 agent 在工具环境中的成功执行轨迹，蒸馏成一组标准化、可执行、可组合的原子 skills，并让这些 skills 作为 memory 反过来提升 agent 在新任务中的成功率、效率和泛化能力。

如果要进一步压缩成一句论文式表述：

> 将长程执行经验从“文本 workflow memory”重构为“可组合 atomic skill memory”，使 memory 单元与真实 agent 的运行时接口对齐。

## 2. 从 idea 到问题定义

### Step 1: 你已经有的直觉

你的原始想法有一个很强的核心：

- 成功轨迹本身包含了高价值 procedural knowledge。
- 这些知识如果只作为长文本保存在上下文里，复用效率不高。
- 更合理的做法是把轨迹中的高频模式沉淀成结构化能力单元。

这说明你的研究问题不是“要不要做 agent memory”，而是：

`memory 的最佳知识单元应该是什么？`

### Step 2: 为什么 AWM 是好的起点

`agent-workflow-memory` 的价值在于，它已经把“从成功轨迹中归纳程序性记忆”这件事走通了第一步。它至少证明了两点：

- 成功 episode 可以被抽象成更短、更稳定的 workflow。
- 把 workflow 注回后续任务，能改善 agent 在相关任务上的表现。

这让你的工作不需要从零开始证明“程序性记忆有用”，而是可以把重点放在：

`workflow 这种记忆单元是否足够好？`

### Step 3: 详细看源码后，AWM 的边界更像什么

基于仓库实现，可以更准确地把 AWM 的边界描述为：

- 它主要把过去成功经验归纳成文本化 workflow，然后在新任务时作为额外上下文注入模型。
- 它面向的是网页/文本 action 环境下的 agent 设定，而不是通用 CLI coding agent。
- 它的 workflow 更像“整段经验摘要”，而不是“最小可复用能力单元”。

这些判断有源码支撑：

- 在 `mind2web/memory.py` 中，workflow memory 是从 `workflow_path` 读取的文本，并被拼接到模型输入消息里。
- 在 `webarena/run.py` 中，系统依赖 BrowserGym/WebArena 的浏览器动作环境，说明其运行时主要面向网页代理。
- 在 `webarena/induce_prompt.py` 一类归纳脚本中，workflow 的生成本质上仍是对完整成功过程做总结，而不是对重叠子过程做系统性的原子分解。

所以，对 AWM 更严谨的批评不是“它完全不是 agent”，而是：

`它验证了 workflow memory 的价值，但没有把 memory 单元设计成适合真实工具型 agent 复用的 skills。`

### Step 4: 你的方案真正补上的缺口

你的方案比 AWM 多做了四步关键变换：

1. 从真实工具执行中采集 trajectory，而不是只处理特定 action benchmark。
2. 从 trajectory 中归纳 workflow，但不把 workflow 当终点。
3. 对 workflow 做聚类、去重和原子化，得到更细粒度的能力片段。
4. 把能力片段打包成标准 skills，让 memory 直接进入 agent 运行时。

因此你的项目不只是“再做一个 memory 方法”，而是提出了一个新的 memory interface：

`Textual Workflow Memory -> Executable Skill Memory`

## 3. 最终问题陈述

### 研究目标

构建一个系统，把真实 agent 的成功执行轨迹自动蒸馏成标准 skills，并验证这种 skillized memory 是否优于：

- 无 memory 的基线 agent
- 原始 trajectory 检索
- AWM 风格的 workflow text memory
- 不做原子化的 coarse workflow skill

### 核心假设

- `H1`: skillized memory 能提升 held-out 任务成功率。
- `H2`: 原子 skill 比整段 workflow 在分布外任务上更容易组合泛化。
- `H3`: 以 skill 为单位的记忆检索，比直接注入整段 workflow 更节省上下文成本。
- `H4`: 带有脚本、引用资料和资产文件的 skill，比纯文本摘要更容易稳定复用。

## 4. 系统架构

```mermaid
flowchart LR
    A[Real Agent Execution] --> B[Trajectory Recorder]
    B --> C[Trajectory Segmenter]
    C --> D[Workflow Inducer]
    D --> E[Cluster and Deduplicate]
    E --> F[Atomic Skill Miner]
    F --> G[Skill Packager]
    G --> H[Skill Repository]
    H --> I[Runtime Skill Retrieval]
    I --> J[Next Agent Run]
```

### 4.1 Trajectory Recorder

目标是记录“真实可执行过程”，而不是只保留最终答案。最小事件 schema 应包含：

- 时间戳
- 任务描述
- 当前子目标
- 工具调用
- 工具输入
- 工具输出
- 关键环境状态
- 失败与回退
- 最终产物 diff

### 4.2 Trajectory Segmenter

一条长轨迹必须先被切成稳定的功能片段，否则后面聚类会被路径、文件名和局部噪声污染。

建议先做两级切分：

- 规则切分：按工具切换、目标切换、错误恢复、文件集变化切段。
- LLM 精修：让模型判断边界是否对应一个完整子过程。

### 4.3 Workflow Inducer

每个 segment 先归纳成“局部 workflow 候选”，而不是立即生成 skill。这个中间层很重要，因为：

- workflow 保留了程序步骤信息；
- 但还没有被过早固定成最终 skill 形式；
- 便于后续聚类对齐和冗余消除。

workflow 候选建议包含：

- 目标
- 触发条件
- 前置条件
- 核心步骤
- 常见失败
- 可验证完成信号

### 4.4 Cluster and Deduplicate

这一层决定项目是否真的能泛化。建议使用双视角表示：

- 语义视角：segment/workflow 摘要 embedding
- 程序视角：工具序列、命令模板、文件操作模式、产物类型

做法建议是：

1. 先按任务域和工具类型做粗分桶。
2. 再在桶内做 embedding clustering。
3. 对 cluster 内 workflow 做模板对齐，消除路径名、变量名、仓库名等表层差异。
4. 建立 overlap graph，识别经常共同出现但可独立复用的子过程。

### 4.5 Atomic Skill Miner

这一步是你相对 AWM 的真正创新点。核心不是“把 workflow 切得越碎越好”，而是找到：

`最小但仍可独立触发、独立验证、独立复用的能力单元`

判断一个候选能否成为 atomic skill，可以用三个标准：

- 独立触发：它有明确前提，不依赖完整原始任务才能启动。
- 独立执行：它内部步骤可以直接指导 agent 或调用脚本完成。
- 独立验证：它有清晰的完成信号和失败边界。

### 4.6 Skill Packager

输出应符合标准 skill 目录形式，至少包含：

- `SKILL.md`
- `references/`
- `scripts/`
- `assets/`

为了兼顾研究可追踪性与标准兼容性，可以采用下面的做法：

- 对 agent 暴露的部分保持标准 skills 格式。
- 对研究内部元数据单独放在 `references/PROVENANCE.md` 或生成清单中。

### 4.7 Runtime Skill Retrieval

运行时不要把整个 skill 正文全量塞进上下文，而应该做两级检索：

- 一级：只看 skill metadata 和短描述，召回候选 skill。
- 二级：按需加载对应 `SKILL.md`、脚本、引用资料。

这和 skills 的设计初衷是一致的：把长程能力存储到外部文件系统，而不是全部挤在 prompt 内。

## 5. 技术选型

## 5.1 运行时 agent

首选：支持 shell、文件系统和多文件编辑的真实 coding agent 运行时。

推荐落地方式：

- 主运行时：Claude Code SDK 或等价的 CLI tool-using agent。
- 原因：你的研究目标就是把 memory 直接沉淀成 skills，而不是停留在 benchmark action predictor。

如果你暂时不想绑定单一供应商，可以把运行时抽象成统一事件接口，但第一版实验最好只选一个主平台，先把闭环跑通。

## 5.2 数据与中间产物存储

推荐：

- 原始轨迹：`JSONL`
- 分析表：`DuckDB`
- 技能仓库：文件系统目录

原因：

- JSONL 易记录事件流；
- DuckDB 很适合做离线挖掘、聚类输入和 ablation 分析；
- skill 本身天然就是目录结构。

## 5.3 Schema 与配置

推荐：

- `Python 3.10+`
- `Pydantic` 定义 episode、segment、workflow、skill candidate schema
- `Typer` 或 `argparse` 做离线流水线 CLI

## 5.4 表示学习与聚类

MVP 推荐：

- 文本表示：`sentence-transformers` 的通用 embedding 模型
- 聚类算法：`HDBSCAN`
- 重排序：cluster 内再做 pairwise similarity + 规则约束

原因：

- 你无法预先知道 skill 簇数量；
- HDBSCAN 对噪声点更友好；
- 第一版先把“能自动形成高质量簇”做出来，比追求最复杂图挖掘更重要。

V2 再考虑：

- 结合代码/命令特征的多模态表示
- 频繁子图挖掘
- 基于 DAG 的 skill 组合发现

## 5.5 Workflow 与 Skill 生成

推荐模式：

- 规则负责结构约束和变量归一化；
- LLM 负责摘要、模板抽象、前置条件提取和 skill 文本生成。

不要一开始就把全部归纳工作交给 LLM。否则你很难回答：

- skill 为什么被这样切分；
- 冗余为什么真的减少了；
- 泛化为什么来自结构而不是 prompt luck。

## 5.6 评测任务

评测一定要选“真实工具使用”任务，而不是只看静态 action prediction。

建议分两层：

- 主评测：真实代码仓库上的 CLI 任务
- 补充评测：受控的合成任务，用来精确测 skill 复用和组合

最重要的是任务集要包含：

- 重复出现的局部子过程
- 不同表层形式但相同底层程序模式
- 需要组合多个子技能的新任务

## 6. 你应该怎么讲这个 idea

如果你要对导师、合作者或评审解释，可以直接用下面这段逻辑：

1. AWM 证明了从成功轨迹中归纳 workflow memory 是有效的。
2. 但 workflow 仍然太粗，它本质上还是面向上下文注入的文本记忆。
3. 对真实工具型 agent 来说，更自然的记忆接口应该是 skills。
4. 真正关键的问题不是“能不能从轨迹生成 workflow”，而是“能不能把重叠 workflow 进一步压缩成可组合的原子 skills”。
5. 如果能做到，memory 就不只是 retrieval context，而会变成 agent 的长期能力库。

## 7. MVP 范围

第一阶段不要追求“全自动、全环境、全 benchmark”。最合理的 MVP 是：

- 只支持一个真实 CLI agent 运行时
- 只支持一个任务域
- 只做成功轨迹的离线蒸馏
- 只生成文本加轻量脚本 skill
- 只验证离线技能库对新任务成功率和效率的影响

这版只要闭环跑通，你的研究命题就已经成立了一半。

## 8. 最重要的消融实验

至少做下面四组对比：

- `No Memory`
- `Trajectory Retrieval`
- `Workflow Memory`
- `Atomic Skill Memory`

如果资源够，再加：

- `Atomic Skill Memory without scripts`
- `Atomic Skill Memory without clustering`
- `Atomic Skill Memory without decomposition`

## 9. 主要风险

- 轨迹质量不稳定：如果成功轨迹本身噪声大，后续归纳会被污染。
- skill 切分过细：会导致检索过载和组合成本上升。
- skill 切分过粗：会退化回 workflow memory。
- 运行时触发不准：召回错 skill 可能反而干扰 agent。
- 评测任务不合适：如果任务之间没有可复用子结构，就很难体现方法优势。

## 10. 当前最合适的项目定位

最稳的定位不是“做一个比 AWM 更复杂的 memory 系统”，而是：

> 提出一种面向真实工具型 agent 的 executable procedural memory 框架，并证明 atomic skill 是比 coarse workflow 更好的记忆单元。

## 参考资料

- Agent Workflow Memory 仓库：<https://github.com/zorazrw/agent-workflow-memory>
- `mind2web/memory.py`：<https://raw.githubusercontent.com/zorazrw/agent-workflow-memory/main/mind2web/memory.py>
- `webarena/run.py`：<https://raw.githubusercontent.com/zorazrw/agent-workflow-memory/main/webarena/run.py>
- `webarena/induce_prompt.py`：<https://raw.githubusercontent.com/zorazrw/agent-workflow-memory/main/webarena/induce_prompt.py>
- Agent Skills 规范：<https://agentskills.io/skills>
- Anthropic Engineering 博文：<https://www.anthropic.com/engineering/skills-to-improve-agentic-coding>
