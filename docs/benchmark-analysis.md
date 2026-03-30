# Benchmark 调研与通用方案分析

## 1. 调研结论

如果把 `procmem2skills` 的目标定义为：

> 从 agent 真实执行轨迹中蒸馏 atomic skills，并验证 skill memory 能否跨任务、跨环境提升 agent 表现

那么四个 benchmark 不应该被等价看待，而应该被分层使用：

- `Mind2Web`：最适合做离线轨迹归纳、workflow induction 和泛化切分验证。
- `WebArena`：最适合做网页多步任务下的在线闭环验证。
- `ALFWorld`：最适合做低噪声、强结构环境中的 skill 原子化与组合分析。
- `Terminal-Bench`：最适合验证 skills 是否真的能进入真实 CLI agent 运行时并提升最终端到端表现。

换句话说，这四个 benchmark 能形成一条互补链路，而不是四个完全平行的测试集。

## 2. 四个 Benchmark 的定位

## 2.1 WebArena

官方 WebArena 将自己定义为“a standalone, self-hostable web environment for building autonomous agents”，并提供可程序化验证任务正确性的评测环境。官方仓库还说明当前 canonical 实现建议结合 AgentLab/BrowserGym 做并行实验，并保留了标准 `run.py` 评测入口。

从项目角度看，WebArena 的关键价值有三点：

- 它是 `真实交互式环境`，agent 必须一步步观察、点击、输入并改变外部状态。
- 它是 `长程任务环境`，任务往往跨页面、跨站点、跨工具。
- 它有 `功能性评测`，不只是动作匹配，而是检查任务结果是否真的满足目标。

这使它非常适合验证：

- workflow memory 是否能降低步数；
- atomic skills 是否比 coarse workflow 更容易组合；
- skill retrieval 是否真的改善了在线决策。

但 WebArena 也有明显成本：

- 环境部署重；
- 页面状态复杂；
- 观测和动作空间都带有网页环境特有噪声。

因此它适合作为 `主在线 benchmark`，但不适合作为唯一开发环境。

## 2.2 WebArena-Verified

`WebArena-Verified` 是 2025 年后出现的官方 verified release。它的重要变化不是换了任务本身，而是提供了：

- fully audited benchmark
- offline evaluation via network trace replay
- deterministic scoring

这对我们尤其关键，因为它让我们能把“agent 在线执行”和“memory 方法评估”部分解耦。

对 `procmem2skills` 来说，`WebArena-Verified` 的价值在于：

- 可以降低 live web 环境波动对实验结论的污染；
- 可以保留 agent 响应和网络 trace，便于做 trajectory-level 对照分析；
- 更适合做大规模 ablation。

因此更稳的策略不是在 `WebArena` 和 `WebArena-Verified` 中二选一，而是：

- 用 `WebArena` 做在线能力验证；
- 用 `WebArena-Verified` 做可复现实验和误差归因。

## 2.3 Mind2Web

官方 README 将 Mind2Web 定义为“the first dataset for developing and evaluating generalist agents for the web”，包含：

- `2000+` open-ended tasks
- `137` websites
- `31` domains
- crowdsourced action sequences

它的数据结构不是“给你一个持续在线环境”，而是“给你人类 demonstration 与页面快照”，包含：

- task instruction
- action sequence
- raw HTML / cleaned HTML
- operation type
- positive / negative candidate elements
- trace、HAR、video、snapshot 等原始回放信息

这意味着 Mind2Web 对我们非常重要，但使用方式必须准确：

- 它非常适合做 `offline trajectory mining`
- 它非常适合做 `workflow abstraction`
- 它非常适合做 `cross-task / cross-website / cross-domain` 泛化检验
- 它不应该被误当成与 WebArena 完全同型的在线 benchmark

也就是说，Mind2Web 在本项目里更像 `技能蒸馏数据源`，而不是唯一的在线 agent 测试场。

## 2.4 ALFWorld

ALFWorld 的官方定义是：它包含与 ALFRED 对齐的 interactive TextWorld environments，让 agent 先在抽象文本空间中学习高层策略，再迁移到 embodied tasks。

官方配置文件还给出了六类核心任务：

- Pick & Place
- Examine in Light
- Clean & Place
- Heat & Place
- Cool & Place
- Pick Two & Place

ALFWorld 对本项目的价值非常高，原因是它在结构上比网页环境干净很多：

- 动作空间是离散文本命令；
- 任务模板有限但可组合；
- 成功条件明确；
- 子过程天然可枚举。

这使它成为检验“atomic skill 是否真的存在”的理想环境。比如：

- `find object`
- `open container`
- `take object`
- `clean object`
- `navigate to receptacle`
- `place object`

这些步骤非常接近我们想要抽出的原子 skills。

所以，ALFWorld 最适合承担的角色是：

- skill 原子化算法的开发环境
- 组合泛化实验环境
- 去重与模板归一化方法的 sanity check 环境

## 2.5 Terminal-Bench

Terminal-Bench 的官方 README 直接把自己定义成“the benchmark for testing AI agents in real terminal environments”，并强调它由两部分组成：

- dataset of tasks
- execution harness connected to a terminal sandbox

每个任务至少包含：

- 英文 instruction
- test script
- oracle solution

这与我们的方向高度一致，因为我们的目标不是只在 benchmark 上提高动作精度，而是让 agent 在真实工具环境中学会可复用 skills。

Terminal-Bench 对本项目有两个独特意义：

- 它最接近 skills 的真实消费场景，因为 agent 本来就在 terminal 中执行。
- 它天然支持“脚本化验证”，这与 `skill -> script -> verifier` 的设计完全兼容。

如果只选一个最能体现 `procmem2skills` 价值的 benchmark，优先级其实是 `Terminal-Bench`。

## 3. 四者的共性

虽然四个 benchmark 表面差异很大，但对本项目来说，它们共享一套更底层的结构：

### 3.1 都可以表达成 Agent-Environment Loop

四者都可抽象为：

`instruction -> observation -> action -> environment transition -> new observation -> ... -> evaluator`

区别只是 observation 和 action 的具体形式不同。

### 3.2 都存在可重复的子过程

这是本项目成立的前提。如果任务完全没有重复模式，就无从蒸馏 skill。

四个 benchmark 中都能看到复用子过程：

- WebArena：搜索、定位实体、筛选信息、提交表单、跨站点核对信息
- Mind2Web：定位 DOM 元素、填写输入框、切换页面、选择候选项
- ALFWorld：找物体、开容器、操作工具、搬运、放置
- Terminal-Bench：定位文件、运行测试、解释错误、修改配置、重跑验证

### 3.3 都有可记录的执行轨迹

只是轨迹来源不同：

- WebArena：在线 agent 交互日志
- Mind2Web：人类 demonstration + raw traces
- ALFWorld：环境状态与文本动作历史
- Terminal-Bench：shell history、stdout/stderr、文件 diff、test result

这意味着 `trajectory` 可以成为统一的数据入口。

### 3.4 都可以定义“完成信号”

这是 skill 原子化的核心，因为 skill 必须可验证。

- WebArena：任务 evaluator 或页面/数据库状态
- Mind2Web：标注动作、候选元素、轨迹一致性
- ALFWorld：环境 reward / success flag
- Terminal-Bench：test script、产物存在性、命令返回值

## 4. 四者的关键差异

## 4.1 Observation 模态不同

- WebArena：DOM、a11y tree、URL、截图、网络状态
- Mind2Web：HTML 快照、候选元素、trace、screenshot
- ALFWorld：文本房间描述和物体状态
- Terminal-Bench：终端输出、文件内容、目录结构、测试结果

因此不能假设 skill 表示只靠一种 modality 就够。

## 4.2 Action 语法不同

- WebArena / Mind2Web：点击、输入、选择、导航
- ALFWorld：离散自然语言命令
- Terminal-Bench：shell 命令、脚本调用、文件编辑

因此真正的统一层不应是“动作 token”，而应是：

`tool-conditioned operation`

## 4.3 Environment 可控性不同

- Mind2Web 偏静态回放数据
- ALFWorld 高度可控
- WebArena 中等可控但部署复杂
- Terminal-Bench 高度真实但任务噪声大

这决定了实验设计要分为：

- 低噪声结构验证
- 高真实性最终验证

## 4.4 Evaluation 粒度不同

- Mind2Web 更偏 step-level / action-level
- WebArena 偏 task-level functional correctness
- ALFWorld 既可 step-level 也可 episode-level
- Terminal-Bench 偏 end-to-end execution success

因此我们的统一评测要至少保留两层指标：

- `process metrics`
- `outcome metrics`

## 5. 如何实现通用

真正的“通用”不是用一套 prompt 同时跑四个 benchmark，而是建立一套 benchmark-agnostic 的 memory pipeline。

## 5.1 统一抽象一：事件级轨迹 Schema

建议把所有 benchmark 的数据都转换成统一事件结构：

```text
Trajectory
  episode_id
  benchmark
  task_id
  instruction
  events[]

Event
  step_id
  timestamp
  observation
  thought(optional)
  action
  tool
  tool_args
  result
  state_delta
  artifacts
  success_signal(optional)
```

其中 benchmark-specific 的字段都进入 `artifacts` 或 `state_delta`，不要污染顶层 schema。

## 5.2 统一抽象二：Benchmark Adapter

每个 benchmark 只负责把自己的环境封装成同一套接口：

```python
class BenchmarkAdapter(Protocol):
    def reset(self, task_id: str) -> Observation: ...
    def step(self, action: Action) -> StepResult: ...
    def is_done(self) -> bool: ...
    def score(self) -> dict: ...
    def export_artifacts(self) -> dict: ...
```

需要两类 adapter：

- `InteractiveAdapter`
- `ReplayAdapter`

对应关系如下：

- WebArena: `InteractiveAdapter`
- WebArena-Verified: `ReplayAdapter` + `InteractiveAdapter`
- Mind2Web: `ReplayAdapter`
- ALFWorld: `InteractiveAdapter`
- Terminal-Bench: `InteractiveAdapter`

这一步是通用化的关键，因为它把“环境差异”隔离在 adapter 层，而让上层 skill pipeline 保持一致。

## 5.3 统一抽象三：Skill Candidate 表示

无论来自网页、家庭环境还是 terminal，候选 skill 都应该映射到同一套结构：

- `intent`
- `preconditions`
- `trigger`
- `tool pattern`
- `state transition`
- `verification`
- `failure recovery`
- `artifacts used`

这样聚类时可以同时利用：

- 语义信息
- 工具信息
- 状态转移信息
- 验证信息

## 5.4 统一抽象四：两阶段蒸馏

不要直接从原始轨迹一步生成 skills，而应该固定为：

`trajectory -> segment -> workflow -> atomic skill`

原因是不同 benchmark 的原始轨迹噪声差异太大：

- WebArena 有网页噪声
- Mind2Web 有人类 demonstration 偏差
- ALFWorld 有模板化语言
- Terminal-Bench 有 shell 输出噪声和长尾错误恢复

只有先做中间层 workflow，后面的聚类、去重和原子化才有可解释性。

## 6. 技术选型建议

## 6.1 语言与数据层

- `Python 3.10+`
- `Pydantic`：统一 schema
- `DuckDB`：存 episode、segment、workflow、cluster 元数据
- `JSONL`：存原始轨迹

原因是你需要同时支持离线挖掘和在线回放分析。

## 6.2 环境层

- WebArena / Mind2Web：`Playwright` 生态优先
- ALFWorld：直接用官方环境
- Terminal-Bench：直接接官方 harness
- 环境隔离：`Docker`

这里不要试图从第一天就统一所有运行方式；应该统一 `adapter interface`，而不是统一底层执行器。

## 6.3 表示与聚类

- 文本表示：通用 embedding 模型
- 结构特征：工具序列、动作模板、状态转移模式、验证器类型
- 聚类：`HDBSCAN`
- 近重复去除：规则归一化 + pairwise rerank

这套设计对四类 benchmark 都成立。

## 6.4 Skill 输出格式

标准输出保持 skill 目录：

- `SKILL.md`
- `references/`
- `scripts/`
- `assets/`

其中 benchmark-specific 的环境知识不要直接写死到正文，而应：

- 抽象成前置条件和触发条件
- 把环境细节放进 `references/`
- 把可执行补丁/验证封装进 `scripts/`

这样 skills 才有机会跨 benchmark 迁移。

## 7. 推荐的实验编排

最稳的路线不是四个 benchmark 同时硬上，而是三阶段推进：

### 阶段一：ALFWorld

目标：

- 验证原子化算法是否能抽出稳定 skill
- 验证 skill 组合是否优于 coarse workflow

原因：

- 动作空间干净
- success signal 明确
- 调试成本低

### 阶段二：Mind2Web + WebArena

目标：

- 用 Mind2Web 做离线 workflow/skill 挖掘
- 用 WebArena 做在线闭环验证

原因：

- 二者与 AWM 直接对齐，便于公平比较
- 可以直接证明你相对 AWM 的改进点

### 阶段三：Terminal-Bench

目标：

- 验证 skill memory 是否在真实 terminal agent 场景中仍然有效
- 验证标准 skill 目录是否真的能成为运行时 memory interface

原因：

- 这是最贴近你最终研究定位的 benchmark

## 8. 最终建议

如果现在就问“哪个 benchmark 最重要”，答案不是唯一的：

- 若目标是 `对齐 AWM 并证明你的方法更强`：优先 `Mind2Web + WebArena`
- 若目标是 `把方法做成真能运行的 agent memory system`：优先 `Terminal-Bench`
- 若目标是 `先把原子 skill 挖掘算法做扎实`：优先 `ALFWorld`

因此当前最合理的项目组织方式是：

1. 用 `ALFWorld` 打磨 skill mining。
2. 用 `Mind2Web + WebArena` 建立与 AWM 的主对比。
3. 用 `Terminal-Bench` 证明方法对真实 CLI agent 成立。

## 参考资料

- WebArena 官方站点：<https://webarena.dev/>
- WebArena 官方仓库：<https://github.com/web-arena-x/webarena>
- WebArena-Verified 官方仓库：<https://github.com/ServiceNow/webarena-verified>
- Mind2Web 官方仓库：<https://github.com/OSU-NLP-Group/Mind2Web>
- ALFWorld 官方仓库：<https://github.com/alfworld/alfworld>
- ALFWorld 官方站点：<https://alfworld.github.io/>
- Terminal-Bench 官方仓库：<https://github.com/harbor-framework/terminal-bench>
- Terminal-Bench 官方站点：<https://www.tbench.ai/>
- Agent Workflow Memory 官方仓库：<https://github.com/zorazrw/agent-workflow-memory>
