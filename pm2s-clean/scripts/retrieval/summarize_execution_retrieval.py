#!/usr/bin/env python3
"""Parse real execution trajectories for skill usage precision/recall."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_ROOT,
    agent_model_slug,
    candidate_setting_from_path,
    f1_score,
    normalize_slug,
    retrieval_run_dir,
    write_json,
    write_jsonl,
    write_latest_pointer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize real execution success and parsed skill usage.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=None, help="Run root containing comparison_report.json.")
    parser.add_argument("--comparison-report", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--agent", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_index(path: Path) -> dict[str, dict[str, set[str]]]:
    payload = load_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"candidate manifest missing rows: {path}")
    out: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        task = str(row.get("task_name") or "").strip()
        names: set[str] = set()
        gt: set[str] = set()
        for item in row.get("neighbors") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("neighbor_task_name") or item.get("skill_slug") or "").strip()
            slug = str(item.get("skill_slug") or name).strip()
            for value in {name, slug, normalize_slug(name), normalize_slug(slug)}:
                if value:
                    names.add(value.lower())
            if item.get("role") == "gt":
                for value in {name, slug, normalize_slug(name), normalize_slug(slug)}:
                    if value:
                        gt.add(value.lower())
        if task:
            out[task] = {"candidate_names": names, "gt_names": gt}
    return out


def read_text(path: Path, limit: int = 2_000_000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(text) > limit:
        return text[:limit]
    return text


def extract_skill_names_from_text(text: str) -> list[str]:
    patterns = [
        r"activate_skill\s*\{[^}]*[\"']name[\"']\s*:\s*[\"']([^\"']+)[\"']",
        r"<activated_skill\s+name=[\"']([^\"']+)[\"']",
        r"<skill[^>]*name=[\"']([^\"']+)[\"']",
        r"skills/([A-Za-z0-9._-]+)/SKILL\.md",
        r"\.claude/skills/([A-Za-z0-9._-]+)/SKILL\.md",
        r"\.gemini/skills/([A-Za-z0-9._-]+)/SKILL\.md",
    ]
    out: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I | re.S):
            name = re.sub(r"\s+", " ", match.group(1)).strip()
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                out.append(name)
    return out


def extract_skill_names_from_json(value: Any) -> list[str]:
    out: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if str(node.get("name") or "").strip() and any("skill" in str(k).lower() for k in node.keys()):
                out.append(str(node.get("name")).strip())
            if str(node.get("tool_name") or node.get("type") or "").lower() == "activate_skill":
                args = node.get("arguments") or node.get("args") or node.get("input") or {}
                if isinstance(args, dict) and str(args.get("name") or "").strip():
                    out.append(str(args.get("name")).strip())
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
        elif isinstance(node, str) and ("skill" in node.lower() or "activate_skill" in node):
            out.extend(extract_skill_names_from_text(node))

    walk(value)
    seen: set[str] = set()
    cleaned: list[str] = []
    for name in out:
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            cleaned.append(name)
    return cleaned


def trial_task_name(trial_dir: Path, result: dict[str, Any] | None = None) -> str:
    if result and str(result.get("task_name") or "").strip():
        return str(result.get("task_name")).strip()
    name = trial_dir.name
    return name.split("__", 1)[0] if "__" in name else name


def trial_success(result: dict[str, Any] | None, trial_dir: Path) -> bool | None:
    if result:
        rewards = (((result.get("verifier_result") or {}).get("rewards") or {}) if isinstance(result, dict) else {})
        reward = rewards.get("reward")
        if reward is not None:
            try:
                return float(reward) >= 1.0
            except Exception:
                pass
    reward_txt = trial_dir / "verifier" / "reward.txt"
    if reward_txt.is_file():
        try:
            return float(reward_txt.read_text(encoding="utf-8").strip()) >= 1.0
        except Exception:
            return None
    return None


def collect_trial_dirs(run_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for result_json in run_root.rglob("result.json"):
        if "/harbor-jobs/" not in str(result_json):
            continue
        trial_dir = result_json.parent
        if trial_dir.name.startswith("_"):
            continue
        if not (trial_dir / "agent").is_dir() and not (trial_dir / "verifier").is_dir():
            continue
        dirs.append(trial_dir)
    return sorted(set(dirs))


def normalize_set(names: list[str]) -> set[str]:
    out: set[str] = set()
    for name in names:
        for value in {name, normalize_slug(name)}:
            if value:
                out.add(value.lower())
    return out


def metrics(picked: set[str], gt: set[str]) -> tuple[bool, float, float]:
    if not picked:
        return False, 0.0, 0.0
    inter = picked & gt
    precision = len(inter) / len(picked) if picked else 0.0
    recall = len(inter) / len(gt) if gt else 0.0
    return bool(inter), precision, recall


def report_value(report: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in report and report[key] not in (None, ""):
            return report[key]
    return None


def main() -> int:
    args = parse_args()
    report_path = args.comparison_report
    run_root = args.run_root
    if report_path is None:
        if run_root is None:
            raise RuntimeError("pass --run-root or --comparison-report")
        report_path = run_root / "comparison_report.json"
    if run_root is None:
        run_root = report_path.parent
    if not report_path.is_file():
        raise RuntimeError(f"missing comparison report: {report_path}")

    idx = candidate_index(args.candidate_manifest.resolve())
    report = load_json(report_path)
    setting = candidate_setting_from_path(args.candidate_manifest.resolve())
    benchmark = args.benchmark.strip() or str(setting.get("benchmark") or report_value(report, "benchmark", "benchmark_output") or "skillsbench")
    agent = args.agent.strip() or str(report_value(report, "agent") or "agent")
    model = args.model.strip() or str(report_value(report, "model") or "model")
    run_id = args.run_id.strip() or str(report.get("run_id") or run_root.name)
    setting.setdefault("noise_mode", "unknown")
    setting.setdefault("pool_size", 0)
    setting.setdefault("seed", 0)
    if args.output_dir:
        out_dir = args.output_dir.resolve()
        setting_dir = out_dir.parent
    else:
        setting_dir = retrieval_run_dir(
            root=args.root,
            benchmark=benchmark,
            agent=agent,
            model=model,
            arm="real_execution",
            noise_mode=str(setting["noise_mode"]),
            pool_size=setting["pool_size"],
            seed=setting["seed"],
            run_id=".",
        ).parent
        out_dir = setting_dir / normalize_slug(run_id, default="run")

    rows: list[dict[str, Any]] = []
    for trial_dir in collect_trial_dirs(run_root.resolve()):
        result = None
        result_path = trial_dir / "result.json"
        if result_path.is_file():
            try:
                result = load_json(result_path)
            except Exception:
                result = None
        task = trial_task_name(trial_dir, result)
        allowed = idx.get(task, {"candidate_names": set(), "gt_names": set()})
        picked_names: list[str] = []
        for text_path in sorted((trial_dir / "agent").glob("*.txt")) if (trial_dir / "agent").is_dir() else []:
            picked_names.extend(extract_skill_names_from_text(read_text(text_path)))
        for json_path in sorted((trial_dir / "agent").rglob("*.json")) if (trial_dir / "agent").is_dir() else []:
            try:
                picked_names.extend(extract_skill_names_from_json(load_json(json_path)))
            except Exception:
                pass
        picked_set = normalize_set(picked_names)
        if allowed["candidate_names"]:
            picked_set = picked_set & allowed["candidate_names"]
        hit, precision, recall = metrics(picked_set, allowed["gt_names"])
        f1 = f1_score(precision, recall)
        rows.append(
            {
                "benchmark": benchmark,
                "agent": agent,
                "model": model,
                "agent_model": agent_model_slug(agent, model),
                "arm": "real_execution",
                "noise_mode": setting["noise_mode"],
                "pool_size": setting["pool_size"],
                "seed": setting["seed"],
                "task_name": task,
                "trial_name": result.get("trial_name") if result else trial_dir.name,
                "trial_dir": str(trial_dir),
                "success": trial_success(result, trial_dir),
                "picked_skills": sorted(picked_set),
                "gt_skills": sorted(allowed["gt_names"]),
                "hit_gt": hit,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    total = len(rows)
    success_rows = [r for r in rows if r["success"] is True]
    picked_rows = [r for r in rows if r["picked_skills"]]
    precision = sum(float(r["precision"]) for r in rows) / total if total else 0.0
    recall = sum(float(r["recall"]) for r in rows) / total if total else 0.0
    f1 = f1_score(precision, recall)
    hit = sum(1 for r in rows if r["hit_gt"])
    summary = {
        "benchmark": benchmark,
        "agent": agent,
        "model": model,
        "agent_model": agent_model_slug(agent, model),
        "arm": "real_execution",
        "noise_mode": setting["noise_mode"],
        "pool_size": setting["pool_size"],
        "seed": setting["seed"],
        "comparison_report": str(report_path.resolve()),
        "raw_run_root": str(run_root.resolve()),
        "candidate_manifest": str(args.candidate_manifest.resolve()),
        "run_id": run_id,
        "task_count": report.get("task_count"),
        "trial_total": total,
        "total": total,
        "success_total": len(success_rows),
        "success_rate": len(success_rows) / total if total else 0.0,
        "parsed_skill_trial_count": len(picked_rows),
        "parsed_skill_rate": len(picked_rows) / total if total else 0.0,
        "hit_gt_total": hit,
        "hit": hit,
        "hit_at_1": hit / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_when_skill_parsed": sum(float(r["precision"]) for r in picked_rows) / len(picked_rows) if picked_rows else 0.0,
        "recall_when_skill_parsed": sum(float(r["recall"]) for r in picked_rows) / len(picked_rows) if picked_rows else 0.0,
        "f1_when_skill_parsed": f1_score(
            sum(float(r["precision"]) for r in picked_rows) / len(picked_rows) if picked_rows else 0.0,
            sum(float(r["recall"]) for r in picked_rows) / len(picked_rows) if picked_rows else 0.0,
        ),
        "results": str(out_dir / "trial_skill_usage.jsonl"),
    }
    write_jsonl(out_dir / "trial_skill_usage.jsonl", rows)
    write_json(out_dir / "summary.json", summary)
    write_latest_pointer(setting_dir, out_dir / "summary.json", run_id)
    print(f"trials={total} success={summary['success_total']} parsed_skill_trials={len(picked_rows)}")
    print(f"precision={summary['precision']:.4f} recall={summary['recall']:.4f} f1={summary['f1']:.4f}")
    print(f"summary={out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
