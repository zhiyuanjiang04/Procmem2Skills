from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from procmem2skills.models import AtomicSkill
from procmem2skills.packager.llm_skill_creator import GeneratedSkillArtifact, LLMSkillCreator
from procmem2skills.packager.skill_writer import SkillWriter


@dataclass(frozen=True)
class SkillGenerationConfig:
    mode: str = "llm-agent"
    model: str | None = None
    base_url: str | None = None
    timeout_sec: int = 120
    max_retries: int = 1
    strict_llm: bool = False
    skill_creator_agent_style: str = "codex"
    skill_creator_system_prompt: str | None = None
    failure_analysis: dict | None = None
    failure_analysis_by_task: Mapping[str, dict] | None = None


def standard_llm_generation_config(
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout_sec: int = 120,
    max_retries: int = 1,
    skill_creator_agent_style: str = "codex",
    skill_creator_system_prompt: str | None = None,
    failure_analysis: dict | None = None,
    failure_analysis_by_task: Mapping[str, dict] | None = None,
) -> SkillGenerationConfig:
    return SkillGenerationConfig(
        mode="llm-agent",
        model=model,
        base_url=base_url,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        strict_llm=True,
        skill_creator_agent_style=skill_creator_agent_style,
        skill_creator_system_prompt=skill_creator_system_prompt,
        failure_analysis=failure_analysis,
        failure_analysis_by_task=failure_analysis_by_task,
    )


def materialize_skill_repository(
    *,
    skills: list[AtomicSkill],
    output_dir: Path,
    writer: SkillWriter | None = None,
    generation: SkillGenerationConfig | None = None,
) -> tuple[list[Path], dict]:
    writer = writer or SkillWriter()
    generation = generation or SkillGenerationConfig(mode="heuristic")
    requested_mode = (generation.mode or "heuristic").strip().lower()
    artifacts: dict[str, list[GeneratedSkillArtifact]] = {}
    metadata = {
        "requested_mode": requested_mode,
        "effective_mode": "heuristic",
        "llm_model": None,
        "llm_agent_style": generation.skill_creator_agent_style,
        "strict_llm": bool(generation.strict_llm),
        "llm_generated": 0,
        "llm_generated_variants": 0,
        "llm_fallback": 0,
    }

    if requested_mode == "llm-agent":
        model = generation.model or _default_skill_creator_model() or "openai/gpt-5.3-codex"
        metadata["llm_model"] = model
        if model and _has_any_api_key():
            creator = LLMSkillCreator(
                model=model,
                base_url=generation.base_url,
                timeout_sec=generation.timeout_sec,
                max_retries=generation.max_retries,
                agent_style=generation.skill_creator_agent_style,
                custom_system_prompt=generation.skill_creator_system_prompt,
            )
            for skill in skills:
                failure_context = _select_failure_context(
                    skill=skill,
                    failure_analysis=generation.failure_analysis,
                    failure_analysis_by_task=generation.failure_analysis_by_task,
                )
                try:
                    variants = creator.compose_skill_variants(
                        skill,
                        failure_context=failure_context,
                    )
                except Exception:
                    metadata["llm_fallback"] += 1
                    if generation.strict_llm:
                        raise
                else:
                    if not variants:
                        metadata["llm_fallback"] += 1
                        if generation.strict_llm:
                            raise RuntimeError(f"LLM skill creator returned no variants for skill={skill.skill_id}")
                        continue
                    artifacts[skill.skill_id] = variants
                    metadata["llm_generated"] += 1
                    metadata["llm_generated_variants"] += len(variants)
            metadata["effective_mode"] = "llm-agent" if artifacts else "heuristic"
            if generation.strict_llm and skills and not artifacts:
                raise RuntimeError("strict llm skill generation requested but no skills were generated")
        else:
            metadata["effective_mode"] = "heuristic"
            metadata["reason"] = "missing model or API key for llm-agent mode"
            if generation.strict_llm:
                raise RuntimeError("strict llm skill generation requested but model or API key is missing")

    written = writer.write_repository(skills, output_dir, generated_artifacts=artifacts)
    metadata["written_skill_dirs"] = len(written)
    return written, metadata


def materialize_skill_repository_standard_llm(
    *,
    skills: list[AtomicSkill],
    output_dir: Path,
    writer: SkillWriter | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_sec: int = 120,
    max_retries: int = 1,
    skill_creator_agent_style: str = "codex",
    skill_creator_system_prompt: str | None = None,
    failure_analysis: dict | None = None,
    failure_analysis_by_task: Mapping[str, dict] | None = None,
) -> tuple[list[Path], dict]:
    written, metadata = materialize_skill_repository(
        skills=skills,
        output_dir=output_dir,
        writer=writer,
        generation=standard_llm_generation_config(
            model=model,
            base_url=base_url,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            skill_creator_agent_style=skill_creator_agent_style,
            skill_creator_system_prompt=skill_creator_system_prompt,
            failure_analysis=failure_analysis,
            failure_analysis_by_task=failure_analysis_by_task,
        ),
    )
    metadata["standardized_llm_flow"] = "llm-agent-strict"
    return written, metadata


def _default_skill_creator_model() -> str | None:
    return (
        os.environ.get("PROCMEM_SKILL_CREATOR_MODEL")
        or os.environ.get("OPENROUTER_SKILL_MODEL")
        or os.environ.get("OPENAI_SKILL_MODEL")
    )


def _has_any_api_key() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def _select_failure_context(
    *,
    skill: AtomicSkill,
    failure_analysis: dict | None,
    failure_analysis_by_task: Mapping[str, dict] | None,
) -> dict | None:
    context: dict[str, object] = {}
    if isinstance(failure_analysis, dict) and failure_analysis:
        context["global_failure_analysis"] = failure_analysis

    if failure_analysis_by_task:
        task_context: dict[str, dict] = {}
        for task_name in skill.task_origins:
            payload = failure_analysis_by_task.get(task_name)
            if not isinstance(payload, dict):
                continue
            if int(payload.get("failures") or 0) <= 0:
                continue
            task_context[task_name] = payload

        if not task_context:
            ranked = sorted(
                (
                    (task_name, payload)
                    for task_name, payload in failure_analysis_by_task.items()
                    if isinstance(payload, dict) and int(payload.get("failures") or 0) > 0
                ),
                key=lambda item: int(item[1].get("failures") or 0),
                reverse=True,
            )
            for task_name, payload in ranked[:3]:
                task_context[task_name] = payload

        if task_context:
            context["task_failure_analysis"] = task_context

    return context or None
