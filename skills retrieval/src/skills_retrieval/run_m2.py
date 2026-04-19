"""Plan 2: full-scale collapse-curve sweep across all SkillsBench tasks.

88 tasks × {random, hard_neg_semantic} × N ∈ {1, 5, 50, 200, 1000} × 3 seeds × 2 probes
= 5,280 API calls. Use --skip-existing to resume a partial run.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
from pathlib import Path

import numpy as np

from .aggregate import aggregate_by_condition, load_per_trial
from .cli_driver import CLIDriver
from .config import PoolSpec, RunConfig, TrialRecord
from .data import Corpus, load_tasks
from .metrics import score_trial
from .plots import plot_collapse_curve, plot_selection_aware_divergence
from .pool_builder import build_pool
from .preflight import will_fit
from .prompt import render_awareness_prompt, render_pool_block, render_selection_prompt

MODEL_CONTEXT_LIMIT = 200_000


def _all_task_ids_in(tasks_path: Path) -> list[str]:
    ids: list[str] = []
    with Path(tasks_path).open() as f:
        for line in f:
            ids.append(json.loads(line)["task_id"])
    return ids


def _split_prompt(full_prompt: str) -> tuple[str, str]:
    """Split full_prompt into (before_pool, format_instruction) like run_m1."""
    parts = full_prompt.split("Available skills", 1)
    before_pool = parts[0]
    after_pool = parts[1] if len(parts) > 1 else ""
    fmt_match = after_pool.find("\n\nRespond with EXACTLY")
    if fmt_match != -1:
        format_instruction = after_pool[fmt_match:].lstrip("\n")
    else:
        format_instruction = "Respond per the protocol above."
    return before_pool, format_instruction


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-meta", default="data/embeddings/skill_metadata.jsonl")
    p.add_argument("--corpus-emb", default="data/embeddings/skill_embeddings.npy")
    p.add_argument("--corpus-desc", default="data/processed/skill_corpus.jsonl")
    p.add_argument("--tasks", default="data/selection_collapse/skillsbench/tasks.jsonl")
    p.add_argument("--task-embeds", default="skills retrieval/pools/tasks_gt_embeddings_full.npz")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--task-ids", nargs="*", default=None,
                   help="If omitted, runs all tasks in --tasks file.")
    p.add_argument("--pool-sizes", nargs="+", type=int, default=[1, 5, 50, 200, 1000])
    p.add_argument("--strategies", nargs="+", default=["random", "hard_neg_semantic"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--label", default="plan2-m2")
    p.add_argument("--out-dir", default=None, help="Override output dir (for resume).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip (pool_id, probe) pairs whose parsed/ file already exists.")
    args = p.parse_args()

    task_ids = args.task_ids or _all_task_ids_in(Path(args.tasks))

    corpus = Corpus.from_paths(
        Path(args.corpus_meta),
        Path(args.corpus_emb),
        descriptions_path=Path(args.corpus_desc),
    )
    tasks_by_id = {t.task_id: t for t in load_tasks(Path(args.tasks))}
    tasks = [tasks_by_id[tid] for tid in task_ids]

    embeds = np.load(args.task_embeds, allow_pickle=False)
    embed_task_ids = embeds["task_ids"].tolist()
    task_emb_by_id = dict(zip(embed_task_ids, embeds["task_embeddings"]))
    gt_offsets = embeds["gt_offsets"]
    gt_ids_all = embeds["gt_ids"].tolist()
    gt_emb_all = embeds["gt_embeddings"]
    gt_index_of_task_id = {tid: i for i, tid in enumerate(embed_task_ids)}

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
        out_dir = Path("skills retrieval/runs") / f"{ts}-{args.label}"
    for sub in ("raw", "parsed", "pools", "metrics", "figures"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    (out_dir / "skipped.jsonl").touch()

    run_cfg = RunConfig(
        label=args.label, model=args.model,
        task_ids=task_ids, strategies=args.strategies,
        pool_sizes=args.pool_sizes, seeds=args.seeds,
        max_concurrency=args.concurrency,
    )
    (out_dir / "config.json").write_text(run_cfg.model_dump_json(indent=2))

    driver = CLIDriver(model=args.model, max_concurrency=args.concurrency)

    async def run_pool(task, spec: PoolSpec):
        parsed_dir = out_dir / "parsed"
        pool_json = out_dir / "pools" / f"{spec.pool_id}.json"

        # Build pool (cheap; also needed for scoring even on pure resume)
        t_idx = gt_index_of_task_id[task.task_id]
        gt_start, gt_end = int(gt_offsets[t_idx]), int(gt_offsets[t_idx + 1])
        task_gt_ids = gt_ids_all[gt_start:gt_end]
        task_gt_embs = gt_emb_all[gt_start:gt_end]
        gt_bodies = task.gt_skill_bodies[: len(task_gt_ids)]
        gt_entries = [
            (gid, gid.rsplit("_", 1)[-1], body, emb)
            for gid, body, emb in zip(task_gt_ids, gt_bodies, task_gt_embs)
        ]
        pool = build_pool(
            spec, task, corpus,
            task_embedding=task_emb_by_id[task.task_id],
            gt_entries=gt_entries,
        )
        pool_block = render_pool_block(pool, representation="card")
        pool_json.write_text(json.dumps({
            "spec": spec.model_dump(),
            "display_ids": pool.display_ids,
            "id_map": pool.id_map,
            "gt_display_ids": pool.gt_display_ids,
        }, indent=2))

        probe_records: list[TrialRecord] = []
        for probe in ["awareness", "selection"]:
            target = parsed_dir / f"{spec.pool_id}__{probe}.json"
            if args.skip_existing and target.exists():
                probe_records.append(TrialRecord.model_validate_json(target.read_text()))
                continue

            full_prompt = (render_awareness_prompt(task.instruction, pool)
                           if probe == "awareness"
                           else render_selection_prompt(task.instruction, pool))
            if not will_fit(full_prompt, MODEL_CONTEXT_LIMIT, run_cfg.context_safety_margin):
                with (out_dir / "skipped.jsonl").open("a") as f:
                    f.write(json.dumps({
                        "pool_id": spec.pool_id, "probe": probe,
                        "reason": "context_overflow"
                    }) + "\n")
                continue

            before_pool, format_instruction = _split_prompt(full_prompt)
            user_prompt = before_pool + format_instruction
            rec = await driver.run_one(
                pool_id=spec.pool_id, probe=probe,
                system_prompt="You are a retrieval subject in a controlled study.",
                pool_block=pool_block,
                user_prompt=user_prompt,
            )
            probe_records.append(rec)
            (out_dir / "raw" / f"{spec.pool_id}__{probe}.txt").write_text(rec.raw_response)
            target.write_text(rec.model_dump_json(indent=2))

        return spec.pool_id, probe_records, {
            "id_map": pool.id_map,
            "gt_display_ids": pool.gt_display_ids,
        }

    coros = []
    for task in tasks:
        for strategy in args.strategies:
            for n in args.pool_sizes:
                for seed in args.seeds:
                    spec = PoolSpec(task_id=task.task_id, strategy=strategy, n=n, seed=seed)
                    coros.append(run_pool(task, spec))

    print(f"Dispatching {len(coros)} pool-level tasks ({len(coros)*2} API calls max) "
          f"with concurrency={args.concurrency}...", flush=True)
    results = await asyncio.gather(*coros)

    per_trial: list[dict] = []
    for pool_id, recs, pool_map in results:
        trial_row: dict = {"pool_id": pool_id}
        for rec in recs:
            parsed = {"extracted_ids": rec.extracted_ids,
                      "format_status": rec.format_status, "flags": rec.flags}
            scored = score_trial(parsed, pool_map, probe=rec.probe)
            for k, v in scored.items():
                if isinstance(v, (int, float)):
                    trial_row[f"{rec.probe}.{k}"] = v
            trial_row[f"{rec.probe}.format_status"] = rec.format_status
        per_trial.append(trial_row)

    (out_dir / "metrics" / "per_trial.jsonl").write_text(
        "\n".join(json.dumps(r) for r in per_trial)
    )

    rows = load_per_trial(out_dir / "metrics" / "per_trial.jsonl")
    agg = aggregate_by_condition(rows, n_boot=1000, seed=0)
    summary_json = {f"{s}__n{n}": v for (s, n), v in agg.items()}
    (out_dir / "metrics" / "summary_by_condition.json").write_text(
        json.dumps(summary_json, indent=2)
    )

    plot_collapse_curve(agg, out_dir / "figures" / "collapse_curve_recall5.pdf",
                        metric="awareness_recall5")
    plot_collapse_curve(agg, out_dir / "figures" / "collapse_curve_mrr.pdf",
                        metric="awareness_mrr")
    plot_collapse_curve(agg, out_dir / "figures" / "collapse_curve_selection.pdf",
                        metric="selection_top1")
    plot_selection_aware_divergence(agg, out_dir / "figures" / "selection_aware_divergence.pdf")

    print(f"Done. Output under {out_dir}", flush=True)
    print(json.dumps({k: v["awareness_recall5"]["mean"] for k, v in summary_json.items()}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
