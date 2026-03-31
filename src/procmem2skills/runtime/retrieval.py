from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SkillSearchHit(BaseModel):
    skill_id: str
    score: float
    matched_terms: list[str]


class SkillBundle(BaseModel):
    skill_id: str
    body: str
    references: dict[str, str] = Field(default_factory=dict)
    scripts: list[str] = Field(default_factory=list)


class SkillCard(BaseModel):
    skill_id: str
    name: str
    description: str


@dataclass(frozen=True)
class SkillRecord:
    skill_id: str
    body: str
    name: str
    description: str
    metadata_vector: Counter[str]
    fulltext_vector: Counter[str]


class SkillIndex:
    def __init__(self, records: dict[str, SkillRecord], repo_dir: Path | None = None) -> None:
        self.records = records
        self.repo_dir = repo_dir

    @classmethod
    def from_repository(cls, repo_dir: Path) -> "SkillIndex":
        records: dict[str, SkillRecord] = {}
        if not repo_dir.exists():
            return cls(records, repo_dir=repo_dir)
        for skill_dir in sorted(_iter_skill_dirs(repo_dir)):
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.is_file():
                continue
            body = skill_md.read_text(encoding="utf-8")
            frontmatter = _parse_frontmatter(body)
            relative_id = skill_dir.relative_to(repo_dir).as_posix()
            name = str(frontmatter.get("name") or skill_dir.name)
            description = str(frontmatter.get("description") or _first_nonempty_line(body) or skill_dir.name)
            metadata_text = f"{name}\n{description}"
            records[relative_id] = SkillRecord(
                skill_id=relative_id,
                body=body,
                name=name,
                description=description,
                metadata_vector=_vectorize(metadata_text),
                fulltext_vector=_vectorize(body),
            )
        return cls(records, repo_dir=repo_dir)

    def cards(self, skill_ids: list[str] | None = None) -> list[SkillCard]:
        if skill_ids is None:
            selected = sorted(self.records.values(), key=lambda item: item.skill_id)
        else:
            selected = [self.records[skill_id] for skill_id in skill_ids if skill_id in self.records]
        return [SkillCard(skill_id=item.skill_id, name=item.name, description=item.description) for item in selected]

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scope: Literal["metadata", "fulltext"] = "metadata",
    ) -> list[SkillSearchHit]:
        query_vector = _vectorize(query)
        if not query_vector:
            return []
        hits: list[SkillSearchHit] = []
        for skill_id, record in self.records.items():
            target_vector = record.metadata_vector if scope == "metadata" else record.fulltext_vector
            score = _cosine_similarity(query_vector, target_vector)
            if score <= 0:
                continue
            matched_terms = sorted(set(query_vector) & set(target_vector))
            hits.append(SkillSearchHit(skill_id=skill_id, score=score, matched_terms=matched_terms))
        return sorted(hits, key=lambda hit: (-hit.score, hit.skill_id))[:top_k]

    def load_bundle(self, skill_id: str) -> SkillBundle:
        record = self.records.get(skill_id)
        body = record.body if record else ""
        references = {}
        scripts = []
        if self.repo_dir is None:
            return SkillBundle(skill_id=skill_id, body=body)
        skill_dir = self.repo_dir / skill_id
        references_dir = skill_dir / "references"
        scripts_dir = skill_dir / "scripts"
        if references_dir.is_dir():
            for path in sorted(references_dir.iterdir()):
                if path.is_file():
                    references[path.name] = path.read_text(encoding="utf-8")
        if scripts_dir.is_dir():
            scripts = sorted(path.name for path in scripts_dir.iterdir() if path.is_file())
        return SkillBundle(skill_id=skill_id, body=body, references=references, scripts=scripts)


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2]


def _vectorize(text: str) -> Counter[str]:
    return Counter(_tokenize(text))


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = 0.0
    for token, left_weight in left.items():
        dot += float(left_weight) * float(right.get(token, 0))
    if dot <= 0:
        return 0.0
    left_norm = math.sqrt(sum(float(weight) * float(weight) for weight in left.values()))
    right_norm = math.sqrt(sum(float(weight) * float(weight) for weight in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _parse_frontmatter(markdown: str) -> dict[str, str]:
    lines = markdown.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}
    payload: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if not sep:
            continue
        payload[key.strip()] = value.strip()
    return payload


def _first_nonempty_line(markdown: str) -> str:
    for line in markdown.splitlines():
        text = line.strip()
        if text and not text.startswith("---"):
            return text
    return ""


def _iter_skill_dirs(repo_dir: Path):
    for skill_md in repo_dir.rglob("SKILL.md"):
        parent = skill_md.parent
        parts = parent.relative_to(repo_dir).parts
        if any(part.startswith(".") for part in parts):
            continue
        yield parent
