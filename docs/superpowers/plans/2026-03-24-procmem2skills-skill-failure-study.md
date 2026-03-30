# Skill Failure Study Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 `procmem2skills` 上新增一个与 `ProcMem2Skills.pdf` 对齐的实验框架，系统评估 skill 失败来源（curation/retrieval/usage）并输出可复现报告。

**Architecture:** 通过新增 `research` 层对已有 importer/distillation/retrieval/failure-analysis 模块做编排，形成实验矩阵（pool size x retrieval method x task split），并把结果落盘为 JSON 报告。框架以离线可复现为第一优先，后续可接 Harbor live trace 扩展。

**Tech Stack:** Python 3.10+, pydantic, typer, unittest, existing procmem2skills modules

---

### Task 1: 新增研究框架数据模型与矩阵生成器

**Files:**
- Create: `src/procmem2skills/research/__init__.py`
- Create: `src/procmem2skills/research/skill_failure_study.py`
- Test: `tests/test_skill_failure_study.py`

- [ ] **Step 1: 写失败测试（矩阵与配置）**

```python
def test_build_experiment_cells_expands_methods_and_pool_sizes(self):
    ...
```

- [ ] **Step 2: 运行单测确认失败**

Run: `./.venv/bin/python -m unittest tests.test_skill_failure_study.SkillFailureStudyTest.test_build_experiment_cells_expands_methods_and_pool_sizes`
Expected: FAIL（模块或函数不存在）

- [ ] **Step 3: 实现最小数据模型与 cell 生成逻辑**

```python
class RetrievalMethod(str, Enum): ...
class SkillFailureStudyConfig(BaseModel): ...
def build_experiment_cells(config): ...
```

- [ ] **Step 4: 重新运行单测确认通过**

Run: same as Step 2
Expected: PASS

- [ ] **Step 5: 清理调试输出**

删除临时 `print(...)`、注释掉的 debug 分支。

### Task 2: 实现 skill pool 扩展与 retrieval 策略执行

**Files:**
- Modify: `src/procmem2skills/research/skill_failure_study.py`
- Test: `tests/test_skill_failure_study.py`

- [ ] **Step 1: 写失败测试（pool 扩容 + 三种 retrieval）**

```python
def test_expand_index_to_pool_size_adds_noise_skills(self):
    ...
def test_retrieve_skills_supports_page_context_embedding_modes(self):
    ...
```

- [ ] **Step 2: 运行单测确认失败**

Run: `./.venv/bin/python -m unittest tests.test_skill_failure_study.SkillFailureStudyTest.test_expand_index_to_pool_size_adds_noise_skills tests.test_skill_failure_study.SkillFailureStudyTest.test_retrieve_skills_supports_page_context_embedding_modes`
Expected: FAIL

- [ ] **Step 3: 实现最小可用逻辑**

```python
def expand_index_to_pool_size(...): ...
def retrieve_skills(...): ...
```

- [ ] **Step 4: 重新运行测试确认通过**

Run: same as Step 2
Expected: PASS

- [ ] **Step 5: 清理调试代码**

去掉调试常量、临时 mock 分支。

### Task 3: 实现失败归因与报告汇总

**Files:**
- Modify: `src/procmem2skills/research/skill_failure_study.py`
- Test: `tests/test_skill_failure_study.py`

- [ ] **Step 1: 写失败测试（分类规则）**

```python
def test_classify_case_detects_unable_to_retrieve(self):
    ...
def test_classify_case_detects_misled_by_noise(self):
    ...
def test_classify_case_detects_pick_right_but_fail_to_use(self):
    ...
```

- [ ] **Step 2: 运行失败测试**

Run: `./.venv/bin/python -m unittest tests.test_skill_failure_study.SkillFailureStudyTest.test_classify_case_detects_unable_to_retrieve tests.test_skill_failure_study.SkillFailureStudyTest.test_classify_case_detects_misled_by_noise tests.test_skill_failure_study.SkillFailureStudyTest.test_classify_case_detects_pick_right_but_fail_to_use`
Expected: FAIL

- [ ] **Step 3: 实现归因器与聚合器**

```python
def classify_skill_failure_case(...): ...
def summarize_experiment_cells(...): ...
```

- [ ] **Step 4: 运行测试确认通过**

Run: same as Step 2
Expected: PASS

- [ ] **Step 5: 清理调试代码**

删除多余中间字段和未使用 helper。

### Task 4: 实现端到端 runner + CLI 接入

**Files:**
- Modify: `src/procmem2skills/research/skill_failure_study.py`
- Modify: `src/procmem2skills/cli.py`
- Test: `tests/test_skill_failure_study.py`

- [ ] **Step 1: 写失败测试（runner 产出报告）**

```python
def test_run_skill_failure_study_writes_json_report(self):
    ...
```

- [ ] **Step 2: 运行失败测试**

Run: `./.venv/bin/python -m unittest tests.test_skill_failure_study.SkillFailureStudyTest.test_run_skill_failure_study_writes_json_report`
Expected: FAIL

- [ ] **Step 3: 实现 runner 与 CLI 命令**

```python
def run_skill_failure_study(...): ...
@app.command("run-skill-failure-study")
def run_skill_failure_study_cmd(...): ...
```

- [ ] **Step 4: 运行测试确认通过**

Run: same as Step 2
Expected: PASS

- [ ] **Step 5: 清理调试代码**

确保 CLI 输出仅保留必要信息。

### Task 5: 全量回归验证

**Files:**
- Verify: `tests/test_skill_failure_study.py`
- Verify: 现有关键回归测试（`tests/test_skill_retrieval.py`、`tests/test_harbor_transfer_study.py`）

- [ ] **Step 1: 执行新增测试全集**

Run: `./.venv/bin/python -m unittest tests.test_skill_failure_study`
Expected: PASS

- [ ] **Step 2: 执行关键回归**

Run: `./.venv/bin/python -m unittest tests.test_skill_retrieval tests.test_harbor_transfer_study`
Expected: PASS

- [ ] **Step 3: 清理遗留调试代码并复测**

Run: same as Step 1 + Step 2
Expected: PASS
