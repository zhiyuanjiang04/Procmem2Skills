# Formal Experiment Interface

## 目标

这份文档定义一套统一的正式实验接口，用同一个 runner 覆盖四个 benchmark：

- `Mind2Web`
- `WebArena`
- `ALFWorld`
- `Terminal-Bench`

统一接口只负责两件事：

1. 把 benchmark 侧采集到的真实轨迹统一导入为 `Trajectory`
2. 在同一目录结构下运行 `offline distillation` 或 `online update`

也就是说，benchmark 自己的环境部署、agent 执行和官方评测方式仍然保持 benchmark-native；而 `procmem2skills` 统一负责轨迹接入、skills 蒸馏、taxonomy 和实验产物管理。

## 统一命令

正式实验入口：

```bash
bash scripts/server/run_formal_experiment.sh \
  --benchmark <mind2web|webarena|alfworld|terminal-bench> \
  --experiment-id <name> \
  --memory-mode <offline|online> \
  --input-path <raw-input-or-result-dir>
```

如果先只想看计划，不实际执行：

```bash
bash scripts/server/run_formal_experiment.sh ... --dry-run
```

说明：

- `Mind2Web`、导入式 `WebArena`、导入式 `Terminal-Bench`、回放式 `ALFWorld` 都是纯离线处理 recorded trajectories，不需要任何模型 API key。
- 只有 benchmark-native 的 live agent 运行阶段才需要模型配置；如果你走 `OpenRouter`，统一使用你指定的 `OpenRouter` key，不要混用其他私有 key。

如果是 `ALFWorld`，也可以直接让本仓库实时采一条或多条 text-only 轨迹：

```bash
bash scripts/server/run_formal_experiment.sh \
  --benchmark alfworld \
  --experiment-id alfworld-live-demo \
  --memory-mode offline \
  --collect-live \
  --alfworld-split train \
  --alfworld-task-types 1 \
  --alfworld-num-games 1 \
  --alfworld-max-steps 40
```

## 产物结构

每次实验都会写入：

```text
experiments/<benchmark>/<experiment-id>/
  raw/                 benchmark 原始输入或实时采集输出
  imported/
    trajectories.jsonl
  memory/
    archive.jsonl      # 仅 online 模式使用
  skills/
  analysis/
    taxonomy.json
  manifest.json
  RUNBOOK.md
```

`manifest.json` 会记录本次实验的：

- benchmark
- experiment id
- memory mode
- 输入路径
- 导入轨迹数
- skills 数量
- taxonomy 输出路径

## 模式说明

### `offline`

统一执行：

`benchmark trajectory -> import -> distill-offline -> skills`

适合：

- 离线 bootstrapping
- 大规模历史轨迹蒸馏
- benchmark 间对比

### `online`

统一执行：

`benchmark trajectory stream -> import -> update-online -> growing skill repo`

适合：

- 在线增量更新
- 比较 static skill repo 与 dynamic skill repo
- 分析 skill accumulation 曲线

## Benchmark-Specific Guide

## 1. Mind2Web

依据其 README，Mind2Web 本质上是离线数据集，不需要 live harness。正式实验最自然的方式是直接使用官方 `train/`, `test_task/`, `test_website/`, `test_domain/` JSON。

### 前置

- 下载官方数据
- 目录中至少有 `train`, `test_task`, `test_website`, `test_domain`

### 正式实验

```bash
bash scripts/server/run_formal_experiment.sh \
  --benchmark mind2web \
  --experiment-id m2w-train0-offline \
  --memory-mode offline \
  --input-path /raid/zhiyuan/procmem2skills-legacy-20260315/awm-skill-induction/awm_source/mind2web/data/train/train_0.json
```

如果想直接跑整个 data root：

```bash
bash scripts/server/run_formal_experiment.sh \
  --benchmark mind2web \
  --experiment-id m2w-full-online \
  --memory-mode online \
  --input-path /path/to/mind2web/data
```

runner 会递归读取 JSON 文件。

## 2. WebArena

依据 WebArena/BrowserGym README，真实实验要先满足：

- `browsergym`
- `browsergym-webarena`
- `playwright install chromium`
- 完整的 WebArena 站点环境变量：
  - `WA_SHOPPING`
  - `WA_SHOPPING_ADMIN`
  - `WA_REDDIT`
  - `WA_GITLAB`
  - `WA_WIKIPEDIA`
  - `WA_MAP`
  - `WA_HOMEPAGE`

### 2.1 官方/README 侧采集

你需要先用 BrowserGym 或 AWM 的 `run.py` 生成真实结果目录。

一个最直接的 BrowserGym/AWM 风格命令是：

```bash
cd /path/to/webarena
python run.py --task_name "webarena.0" --max_steps 15
```

它会生成类似：

```text
results/webarena.0/
  exp_args.pkl
  experiment.log
  summary_info.json
  step_0.pkl.gz
  step_1.pkl.gz
  ...
```

本仓库现在已经支持直接导入这种 `BrowserGym exp_dir` 结构，不需要你再手工转成示例 JSON。

### 2.2 统一正式实验

```bash
bash scripts/server/run_formal_experiment.sh \
  --benchmark webarena \
  --experiment-id wa-task0-offline \
  --memory-mode offline \
  --input-path /path/to/webarena/results/webarena.0
```

如果你有一个结果根目录，里面有多个 `webarena.*` 子目录，也可以直接指向根目录：

```bash
bash scripts/server/run_formal_experiment.sh \
  --benchmark webarena \
  --experiment-id wa-batch-online \
  --memory-mode online \
  --input-path /path/to/webarena/results
```

## 3. ALFWorld

ALFWorld 的 README/任务结构说明它既适合离线导入，也适合在本仓库里直接做 text-only live collection。

### 3.1 直接 live 采集

这是目前四个 benchmark 里本仓库支持最完整的“从环境到技能”闭环：

```bash
bash scripts/server/run_formal_experiment.sh \
  --benchmark alfworld \
  --experiment-id alf-live-clean-place \
  --memory-mode offline \
  --collect-live \
  --alfworld-split train \
  --alfworld-task-types 3 \
  --alfworld-num-games 2 \
  --alfworld-max-steps 50
```

### 3.2 用已有 raw trajectory

```bash
bash scripts/server/run_formal_experiment.sh \
  --benchmark alfworld \
  --experiment-id alf-offline-replay \
  --memory-mode online \
  --input-path /path/to/alfworld/raw.json
```

## 4. Terminal-Bench

依据 Terminal-Bench README 和 Harbor README，真实 live 运行链路是：

- 安装 Docker
- 安装 `harbor`
- 用 `harbor run --dataset terminal-bench@2.0 ...` 启动真实任务

### 4.1 官方/README 侧 live 采集

Harbor README 的标准命令：

```bash
export OPENROUTER_API_KEY=<YOUR_OPENROUTER_KEY>
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
harbor run --dataset terminal-bench@2.0 \
  --agent claude-code \
  --model anthropic/claude-opus-4-1 \
  --n-concurrent 4
```

如果你的 agent wrapper 使用的不是 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 这组变量，就把同一个 OpenRouter key 与 base URL 映射到它要求的变量名即可。

如果你使用 Harbor 产出的 trial/job 目录，或者已有旧实验里的 `ATIF` bundle，本仓库可以直接导入，不需要你再做额外转换。

### 4.2 统一正式实验

如果你已经按 Harbor 官方教程跑出了 job 目录，继续使用统一 importer：

导入 Harbor/ATIF 结果目录：

```bash
bash scripts/server/run_formal_experiment.sh \
  --benchmark terminal-bench \
  --experiment-id tb-live-offline \
  --memory-mode offline \
  --input-path /path/to/terminal-bench/jobs
```

或者用单个任务目录：

```bash
bash scripts/server/run_formal_experiment.sh \
  --benchmark terminal-bench \
  --experiment-id tb-build-cython-online \
  --memory-mode online \
  --input-path /path/to/build-cython-ext__trial
```

### 4.3 真正用 skills 驱动 Harbor agent

上面的 `run_formal_experiment.sh` 负责的是“导入 Harbor 结果并统一蒸馏/分析”。

如果你要按 `harbor run` 的官方方式直接发起真实 `Terminal-Bench` 任务，同时让 agent 在执行时检索本仓库蒸馏出的 skills，使用新的 live wrapper：

```bash
bash scripts/server/run_terminal_bench_harbor_experiment.sh \
  --experiment-id tb-live-skill-agent \
  --bootstrap-input /path/to/previous/terminal-bench/jobs \
  --model anthropic/claude-sonnet-4 \
  --n-tasks 1 \
  --dry-run
```

去掉 `--dry-run` 之后，它会按下面的顺序执行：

1. 从 `--bootstrap-input` 导入旧轨迹并蒸馏 `bootstrap-skills/`
2. 调用 Harbor 官方 CLI：
   `harbor run --agent-import-path procmem2skills.integrations.harbor_terminal_agent:SkillAwareTerminalAgent ...`
3. 将 Harbor job 目录重新导入为统一 `Trajectory`
4. 在实验目录下写出 `harbor-manifest.json`、`HARBOR_RUNBOOK.md` 和 `imported/live-trajectories.jsonl`

这个 wrapper 默认读取你设置好的 `OpenRouter/OpenAI-compatible` 环境变量：

```bash
export OPENROUTER_API_KEY=<YOUR_OPENROUTER_KEY>
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

如果你已经有现成 skill repo，也可以直接传：

```bash
bash scripts/server/run_terminal_bench_harbor_experiment.sh \
  --experiment-id tb-live-existing-skills \
  --skill-repository /path/to/skills \
  --model anthropic/claude-opus-4-1
```

## 推荐正式实验矩阵

建议最少覆盖下面四组：

### Group A: Offline Bootstrap

- `Mind2Web + offline`
- `WebArena + offline`
- `ALFWorld + offline`
- `Terminal-Bench + offline`

目的：

- 比较不同 benchmark 上 skill 数量、cluster 数量、taxonomy 结构
- 检查 atomic skills 的 benchmark 迁移能力

### Group B: Online Update

- `Mind2Web + online`
- `WebArena + online`
- `ALFWorld + online`
- `Terminal-Bench + online`

目的：

- 对比 static repo 与 streaming repo
- 观察 skill repo 的增长和去重稳定性

### Group C: Live Collection Sanity

- `ALFWorld live + offline`
- `WebArena official run -> unified import`
- `Terminal-Bench harbor run -> unified import`

目的：

- 验证 runner 不是只处理人造示例，而是真的能吃 benchmark-native 输出

## 建议执行顺序

1. `Mind2Web offline`
2. `ALFWorld live`
3. `Terminal-Bench offline/live-import`
4. `WebArena official run -> import`

原因：

- Mind2Web 最稳，适合先打通大批量离线实验
- ALFWorld 最容易先做真实闭环
- Terminal-Bench 最接近真实 CLI agent memory 消费
- WebArena 部署最重，最后做最划算

## Dry Run 示例

在真正启动前，先生成计划：

```bash
bash scripts/server/run_formal_experiment.sh \
  --benchmark terminal-bench \
  --experiment-id tb-plan-only \
  --memory-mode offline \
  --input-path /path/to/jobs \
  --dry-run
```

这会输出：

- 解析后的输入路径
- 目标实验目录
- 计划生成的 imported / skills / taxonomy / manifest 路径

先 dry-run，再真正执行，是推荐流程。
