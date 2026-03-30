# 研究计划

## 题目

**ProcMem2Skills: Distilling Agent Trajectories into Atomic Skills for Executable Procedural Memory**

## 1. 研究背景

随着 agent 在编程、运维和复杂工具使用场景中的能力提升，长期记忆逐渐成为限制 agent 持续改进的重要瓶颈。现有工作如 Agent Workflow Memory 表明，把成功执行过程归纳为 workflow memory 并在后续任务中检索使用，能够提升相关任务表现。然而，现有 workflow memory 仍主要以文本摘要形式存在，更适合做提示增强，而不是作为真实 agent 的可执行能力单元。

与此同时，近期的 Agent Skills 体系表明，程序性知识可以以外部文件化、按需加载、带脚本和参考资料的形式组织，并直接服务于真实工具型 agent 的执行过程。这为 procedural memory 的新接口提供了现实基础。

本研究据此提出：将 agent 成功轨迹蒸馏为 workflow 后，进一步通过聚类、去重和原子化，生成一组标准化 atomic skills，使长期记忆从文本 workflow 转变为可执行 skill memory。

## 2. 研究问题

本研究围绕以下问题展开：

- `RQ1`: 真实 agent 的成功轨迹能否被稳定蒸馏为可复用的 atomic skills？
- `RQ2`: 与 coarse workflow memory 相比，atomic skill memory 是否更有利于组合泛化？
- `RQ3`: 标准化 skill 目录形式是否能让长期记忆更自然地接入真实 agent 运行时？
- `RQ4`: 带脚本和引用资料的 skill 是否比纯文本 workflow 更能提升执行稳定性与效率？

## 3. 核心假设

- `H1`: 相比无记忆基线，atomic skill memory 能提高新任务成功率。
- `H2`: 相比 trajectory retrieval 和 workflow memory，atomic skill memory 在分布变化任务上具有更好的泛化能力。
- `H3`: 相比直接注入整段 workflow，skill 级两段式检索能降低上下文开销。
- `H4`: 包含 `SKILL.md`、`references/`、`scripts/` 和 `assets/` 的 skill 包，比纯文本记忆更容易稳定复现高价值子过程。

## 4. 方法概述

本方法包括六个阶段。

### 4.1 真实轨迹采集

在真实工具型 agent 环境中记录完整执行轨迹，包括任务说明、工具调用、命令参数、工具输出、文件变更、错误恢复和最终结果。轨迹采集对象不局限于网页 action，而强调 shell、文件系统和脚本执行等真实操作。

### 4.2 轨迹切分

将长轨迹切分为多个候选子过程。切分首先依据工具切换、目标切换、错误恢复和文件集变化等规则触发，再由 LLM 对边界进行修正，以提升片段的一致性和可解释性。

### 4.3 Workflow 归纳

对每个片段生成中间层 workflow 表示，保留目标、前置条件、核心步骤、失败模式和完成信号。该层既保留 procedural structure，又避免过早固化为最终 skill。

### 4.4 聚类与去重

基于 workflow 摘要、工具序列、命令模板和产物类型构建联合表示，对候选 workflow 进行聚类。聚类后进一步对变量、路径、仓库名等表层差异进行归一化，去除重复和近重复模板。

### 4.5 原子化

对 cluster 内 workflow 建立重叠图，识别能够独立触发、独立执行和独立验证的最小能力单元，并将其提炼为 atomic skill。该步骤是本研究相对于 workflow memory 的核心创新。

### 4.6 Skill 打包与运行时接入

将 atomic skill 输出为标准 skill 目录，至少包含：

- `SKILL.md`
- `references/`
- `scripts/`
- `assets/`

运行时采用两级检索：先基于短描述召回候选 skill，再按需加载 skill 正文和附属资源，使 skill memory 与真实 agent 的执行接口直接对齐。

## 5. 系统设计

### 5.1 模块划分

- `Recorder`: 记录 episode 级事件流
- `Segmenter`: 识别轨迹边界并生成 segment
- `Inducer`: 从 segment 归纳 workflow
- `Miner`: 完成聚类、去重和原子化
- `Packager`: 生成标准 skill 目录
- `Runtime`: 在新任务中检索并加载 skill

### 5.2 中间表示

本研究使用四层中间表示：

- `Trajectory`: 原始事件流
- `Segment`: 有边界的局部子过程
- `Workflow`: 结构化程序模板
- `Skill`: 可执行、可组合的长期记忆单元

这样的分层可以显式区分“观察到的过程”“归纳出的流程”和“最终可运行能力”，避免把所有问题都交给单次 LLM 生成。

## 6. 技术选型

### 6.1 实现语言与框架

- `Python 3.10+`
- `Pydantic`：定义轨迹、workflow 和 skill schema
- `Typer`：构建离线处理 CLI
- `DuckDB`：分析轨迹、聚类输入和实验结果

### 6.2 运行时环境

选择支持 shell、文件系统和多文件编辑的真实 agent 运行时作为主实验平台，以确保输出 skill 能直接在真实任务中被消费。第一阶段仅绑定一个主平台，优先完成完整闭环。

### 6.3 聚类与模板归一化

- 文本表示：通用 embedding 模型
- 聚类：`HDBSCAN`
- 模板归一化：基于规则的路径、变量、命令参数抽象
- LLM 辅助：仅用于语义摘要、前置条件抽取和 skill 文本生成

该设计强调结构约束与 LLM 归纳协同，而非完全依赖生成模型做黑箱归纳。

## 7. 实验设计

### 7.1 对比方法

- `Baseline`: 无记忆 agent
- `Traj-RAG`: 直接检索相似轨迹
- `Workflow-Memory`: AWM 风格的 workflow 文本记忆
- `Workflow-Skill`: 不做原子化，直接把整段 workflow 打包成 skill
- `ProcMem2Skills`: 完整方法

### 7.2 评测指标

- 任务成功率
- 平均完成步数
- 平均工具调用数
- 上下文 token 开销
- 技能触发准确率
- 技能复用率
- 组合深度
- 错误恢复成功率

### 7.3 任务设置

任务集应覆盖：

- 可重复出现的高频子过程
- 表层差异大但底层模式相同的任务
- 需要组合多个 skill 的新任务

为保证可信度，建议同时使用：

- 真实代码仓库中的 CLI 任务
- 小规模可控合成任务

前者衡量真实收益，后者用于精确观察 skill 复用与组合。

### 7.4 关键消融

- 去掉聚类，仅做 workflow 到 skill 的直接转换
- 去掉原子化，仅保留 coarse workflow skill
- 去掉脚本和引用资料，仅保留 `SKILL.md`
- 去掉两级检索，改为全量注入 skill 文本

## 8. 预期贡献

本研究预期贡献包括：

- 提出面向真实工具型 agent 的 executable procedural memory 框架
- 给出一条从 trajectory 到 standard skills 的自动蒸馏流水线
- 证明 atomic skill 是比 coarse workflow 更优的长期记忆单元
- 提供一套评估 skill memory 泛化、复用和上下文效率的实验范式

## 9. 计划安排

### 第一阶段：问题闭环与数据基线

- 搭建真实 agent 轨迹采集接口
- 定义事件 schema 和中间表示
- 构建首批成功轨迹数据集

### 第二阶段：workflow 归纳与聚类

- 完成轨迹切分
- 完成 workflow 归纳
- 完成聚类与重复模板识别

### 第三阶段：原子化与 skill 生成

- 设计 atomic skill 判定标准
- 实现 skill 打包器
- 构建第一版 skill repository

### 第四阶段：在线检索与实验评测

- 接入运行时检索模块
- 执行基线对比与消融实验
- 分析技能复用、组合和失败案例

## 10. 风险与缓解

- 如果轨迹噪声过大：增加成功过滤与后验验证，只蒸馏高质量 episode。
- 如果聚类质量不稳定：增加任务域分桶和规则特征，降低单纯 embedding 聚类的不确定性。
- 如果原子化过细：引入最小支持度、独立验证信号和触发稳定性约束。
- 如果运行时检索误召回：采用两级召回与 rerank，并记录 skill 使用反馈。

## 11. 结论

本研究旨在把 procedural memory 从“文本化 workflow 提示增强”推进到“标准化、可执行、可组合的长期 skill 库”。如果验证成功，这将为真实工具型 agent 提供一种更自然的能力积累机制，也为 workflow memory 与 runtime skill system 之间建立清晰桥梁。

## 参考资料

- Agent Workflow Memory: <https://github.com/zorazrw/agent-workflow-memory>
- AWM `mind2web/memory.py`: <https://raw.githubusercontent.com/zorazrw/agent-workflow-memory/main/mind2web/memory.py>
- AWM `webarena/run.py`: <https://raw.githubusercontent.com/zorazrw/agent-workflow-memory/main/webarena/run.py>
- AWM `webarena/induce_prompt.py`: <https://raw.githubusercontent.com/zorazrw/agent-workflow-memory/main/webarena/induce_prompt.py>
- Agent Skills 规范: <https://agentskills.io/skills>
- Anthropic Engineering, *Equipping Agents with Skills to Improve Performance*: <https://www.anthropic.com/engineering/skills-to-improve-agentic-coding>
