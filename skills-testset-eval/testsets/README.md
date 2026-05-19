# Skill-selection testset (retrieval-isolation eval)

This directory holds a static evaluation harness that **mimics how harbor's
`terminus-2-skills` agent presents skills to the LLM**, so we can measure
skill-selection behavior without spinning up containers or running tasks
end-to-end.

## What harbor actually does (from source analysis)

Reference: `skillsbench_repo/libs/terminus_agent/agents/terminus_2/`.

1. `harbor_terminus_2_skills.py::setup` walks `/root/.claude/skills` and
   `/root/.terminus/skills` inside the container, reads each `SKILL.md`
   frontmatter, and builds `_skills_metadata = [{name, description, location}]`.
2. `_build_skill_prompt_xml` formats this into:
   ```
   <available_skills>
     <skill name="git">desc</skill>
     <skill name="docker">desc</skill>
   </available_skills>
   ```
3. The block is injected before `Task Description:` in
   `prompt-templates/terminus-xml-plain.txt`. The system prompt explicitly
   instructs:
   > If skills are available, you MUST check if any skill matches the task
   > BEFORE executing commands. To load a skill, respond with ONLY:
   > `<tool_call name="skill"><name>skill-name</name></tool_call>`
4. After the LLM responds, `_handle_skill_tool_calls_xml` extracts skill
   names with regex
   `<tool_call\s+name="skill">\s*<name>([^<]+)</name>\s*</tool_call>`,
   loads the SKILL.md text, prepends "Loaded skill: ..." and re-prompts.

The first agent turn — the **selection decision** — is the only thing this
harness evaluates. We stop at step 4.

## What this harness does

```
task_description ─► Qwen-3-Embedding-0.6B ─► FAISS top-K (44 787 skills)
                                                 │
                                                 ▼
                                  build_prompt() — same XML format as harbor
                                                 │
                                                 ▼
                                       Claude (CLI or SDK)
                                                 │
                                                 ▼
                            parse_selected_skills() — same regex as harbor
                                                 │
                                                 ▼
                              {selected_skills: [...]}  → JSONL
```

No docker, no terminal, no execution. Pure retrieval-isolation methodology
(consistent with `feedback_retrieval_isolation.md`).

## Files

- `skill_selection_eval/prompt.py`  — replicates `<available_skills>` block + skill-system instructions
- `skill_selection_eval/parser.py`  — same regex as `_handle_skill_tool_calls_xml`
- `skill_selection_eval/retrieve.py` — Qwen top-K from prebuilt FAISS index
- `skill_selection_eval/run.py`     — orchestrator (CLI or SDK backend)

## Input format

JSONL, one task per line:
```json
{"task_id": "nginx-config", "task_description": "...", "gt_skills": ["nginx", ...]}
```
`gt_skills` is optional (present for SkillsBench tasks, absent for terminal-bench).

## Pilot

```bash
cd procmem2skills
python -m testsets.skill_selection_eval.run \
    --tasks testsets/data/pilot_tasks.jsonl \
    --out   testsets/runs/2026-05-02-pilot.jsonl \
    --k 5 --backend cli --model claude-sonnet-4-5 --limit 20
```

## Open items (Hanwen sync)

- Source on disk for terminal-bench task descriptions (TB 2.0). Currently
  the `tb2.0` adapter expects a `terminal-bench-2.0/` directory to be
  cloned manually.
- Whether to align candidate `name` field with on-disk skill directory
  slugs (current code uses `slug` from the corpus — fine for a static eval
  but would need verification before any later harbor integration).
- For SkillsBench (87 tasks, static GT), candidate pool is ambiguous: top-K
  from the full 44 787 ClawHub corpus, or a held-out pool that *includes*
  the GT skill? Need to decide before computing Recall@K.
