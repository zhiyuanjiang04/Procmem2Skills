# Server Benchmark Deployment

## 目标

这份文档只讨论一种约束下的部署方式：

- 所有安装、缓存、浏览器、虚拟环境、脚本产物都放在项目目录内部
- 不使用 `sudo`
- 不修改 `~/.bashrc`
- 不依赖系统级包升级

当前服务器能力的只读检查结果：

- `python3`: 3.10.12
- `git`: 可用
- `docker`: 可用
- `node`: 12.22.9
- `npm`: 8.5.1
- `java`: 11
- `playwright`: 不在系统 `PATH`
- `chromium/google-chrome`: 不在系统 `PATH`

## 2026-03-15 实测结论

- `Mind2Web`: 已用真实 `train_0.json` 跑通导入与蒸馏。
- `ALFWorld` text-only: 已在项目内安装 `alfworld`、下载数据到 `.cache/alfworld`，并成功跑出 1 条真实环境轨迹，随后完成 `import -> distill`。
- `Terminal-Bench` offline rollout: 已兼容旧实验目录中的真实 `ATIF` 轨迹格式，并成功完成 `import -> distill`。
- `Harbor`: 官方 `harbor` 包在服务器默认 Python `3.10.12` 上不可安装；改用项目内 `uv` 管理的 Python `3.12` 和 `.venv-py312` 后安装成功，并可列出 `terminal-bench@2.0` / `terminal-bench-sample@2.0` 数据集。
- `WebArena` / `BrowserGym`: 在项目内 Python `3.12` 环境中安装成功，Chromium 也已下载到 `.playwright`，并成功注册 `812` 个 `browsergym/webarena.*` 环境；但 `env.reset()` 会因缺失 `WA_SHOPPING` 等站点环境变量而失败，因此当前真实瓶颈是 self-hosted WebArena 站点部署，而不是 Python 包或浏览器本身。

## 成本分级

### Tier 1: Mind2Web

部署成本最低，原因：

- 不需要 Docker
- 不需要浏览器
- 不需要 GPU
- 离线 benchmark，可以先只做 data import + workflow/skill induction

主要成本：

- 获取数据集
- 把原始 JSON 导入统一 trajectory schema

适合第一批测试：

- importer 是否兼容真实数据字段
- `trajectory -> workflow -> cluster -> atomic skill` 是否能跑通
- 跨 website / domain 的 skill 聚类是否合理

### Tier 2: ALFWorld Text-Only

成本低到中等，原因：

- 不需要浏览器
- 不需要 Docker
- text-only 版本不要求视觉环境
- 对“原子 skill 是否存在”特别友好

主要成本：

- 安装 `alfworld`
- 下载 ALFWorld 数据资源
- 可能遇到 `textworld`/原生依赖兼容问题

适合第二批测试：

- 在线 trajectory recorder
- skill retrieval 是否改善分步决策
- 原子化和组合泛化

### Tier 3: Terminal-Bench

成本中等，原因：

- 最贴近真实 CLI agent
- 服务器已有 Docker，这是一个关键前提
- 但需要 Harbor 或等价 harness，以及任务镜像/环境管理

主要成本：

- 安装 harness
- 拉起 Docker 工作负载
- agent 执行成本和结果管理

适合第三批测试：

- 技能能否真正作为 CLI agent memory interface
- 在线增量更新是否在真实工具环境下成立

### Tier 4: WebArena

成本最高，原因：

- 需要 BrowserGym / Playwright / Chromium
- 需要 self-hosted WebArena 网站环境
- 通常还要依赖 Docker 或额外服务编排
- 调试面最广，失败点最多

主要成本：

- 浏览器安装
- BrowserGym/WebArena Python 依赖
- WebArena 站点部署与维护
- 网络/端口/状态重置

适合最后测试：

- 对齐 AWM 论文场景
- 证明方法不仅适用于 CLI/text benchmark，也适用于 web agents

## 推荐顺序

建议按下面的顺序推进：

1. `Mind2Web`
2. `ALFWorld` text-only
3. `Terminal-Bench`
4. `WebArena`

原因很直接：

- 先用 `Mind2Web` 验证离线蒸馏
- 再用 `ALFWorld` 验证在线 skill usage 与组合
- 再用 `Terminal-Bench` 验证真实 CLI 场景
- 最后再承担 `WebArena` 的高部署成本

## 当前脚本布局

配套脚本位于：

- `scripts/server/check_capabilities.sh`
- `scripts/server/setup_benchmark.sh`
- `scripts/server/run_benchmark_smoke.sh`
- `scripts/server/alfworld_collect_smoke.py`
- `scripts/server/run_terminal_bench_harbor_experiment.sh`
- `scripts/server/run_terminal_bench_transfer_study.sh`
- `scripts/server/run_formal_experiment.sh`

说明：

- `setup_benchmark.sh` 内置了之前的环境准备子步骤（core/mind2web/alfworld/terminal-bench/webarena）。
- `run_benchmark_smoke.sh` 内置了各 benchmark 的 smoke 子步骤，不再拆成多个一次性小脚本。
- 所有脚本默认都把缓存和工具链放在项目目录里。

## 统一运行方式

现在建议统一用两个入口：

```bash
bash scripts/server/setup_benchmark.sh <target>
bash scripts/server/run_benchmark_smoke.sh <target>
```

`setup_benchmark.sh` 支持：

- `core`
- `mind2web`
- `alfworld`
- `terminal-bench`
- `webarena`
- `all`

`run_benchmark_smoke.sh` 支持：

- `mind2web`
- `alfworld`
- `terminal-bench`
- `terminal-bench-harbor`
- `webarena`
- `all`

推荐使用顺序：

```bash
bash scripts/server/setup_benchmark.sh core
bash scripts/server/setup_benchmark.sh alfworld
bash scripts/server/setup_benchmark.sh terminal-bench
bash scripts/server/setup_benchmark.sh webarena

bash scripts/server/run_benchmark_smoke.sh mind2web
bash scripts/server/run_benchmark_smoke.sh alfworld
bash scripts/server/run_benchmark_smoke.sh terminal-bench
bash scripts/server/run_benchmark_smoke.sh terminal-bench-harbor
bash scripts/server/run_benchmark_smoke.sh webarena
```

这组 smoke 的含义分别是：

- `Mind2Web`: 离线导入与蒸馏链路可用
- `ALFWorld`: text-only 真实环境采样、导入与蒸馏链路可用
- `Terminal-Bench`: 真实历史 rollout 的离线导入与蒸馏链路可用
- `terminal-bench-harbor`: Harbor 安装、数据集发现、自定义 skill-aware agent import path 和 live wrapper `dry-run` 可用
- `WebArena`: 先验证离线 `import -> distill`，再验证 BrowserGym、Playwright、Chromium 和 env 注册；若缺少 self-hosted 站点环境变量，会在 reset 阶段暴露
