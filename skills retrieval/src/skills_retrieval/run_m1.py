"""M1 reproduction: 5 pilot tasks × {random, hard_neg_semantic} × N ∈ {1,5,50,200} × 3 seeds × 2 probes.

Writes raw/parsed/metrics under skills retrieval/runs/<timestamp>-plan1-m1/.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
from pathlib import Path

import anthropic
import numpy as np

from .config import PoolSpec, RunConfig, TrialRecord
from .data import Corpus, load_tasks
from .driver import Driver
from .metrics import score_trial, aggregate_metrics
from .pool_builder import build_pool
from .prompt import render_awareness_prompt, render_pool_block, render_selection_prompt
from .preflight import will_fit

MODEL_CONTEXT_LIMIT = 200_000


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-meta", default="data/embeddings/skill_metadata.jsonl")
    parser.add_argument("--corpus-emb", default="data/embeddings/skill_embeddings.npy")
    parser.add_argument("--tasks", default="data/selection_collapse/skillsbench/tasks.jsonl")
    parser.add_argument("--task-embeds", default="skills retrieval/pools/tasks_gt_embeddings.npz")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--task-ids", nargs="+", default=["sb_000", "sb_003", "sb_004", "sb_006", "sb_007"])
    parser.add_argument("--pool-sizes", nargs="+", type=int, default=[1, 5, 50, 200])
    parser.add_argument("--strategies", nargs="+", default=["random", "hard_neg_semantic"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--label", default="plan1-m1")
    args = parser.parse_args()

    corpus = Corpus.from_paths(Path(args.corpus_meta), Path(args.corpus_emb))
    tasks_all = {t.task_id: t for t in load_tasks(Path(args.tasks))}
    tasks = [tasks_all[tid] for tid in args.task_ids]
    embeds = np.load(args.task_embeds, allow_pickle=False)
    task_emb_by_id = dict(zip(embeds["task_ids"].tolist(), embeds["task_embeddings"]))
    gt_offsets = embeds["gt_offsets"]
    gt_ids = embeds["gt_ids"].tolist()
    gt_emb = embeds["gt_embeddings"]

    ts = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
    out_dir = Path("skills retrieval/runs") / f"{ts}-{args.label}"
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "parsed").mkdir(parents=True, exist_ok=True)
    (out_dir / "pools").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "skipped.jsonl").touch()

    run_cfg = RunConfig(
        label=args.label, model=args.model,
        task_ids=args.task_ids, strategies=args.strategies,
        pool_sizes=args.pool_sizes, seeds=args.seeds,
    )
    (out_dir / "config.json").write_text(run_cfg.model_dump_json(indent=2))

    client = anthropic.AsyncAnthropic()
    driver = Driver(client=client, model=args.model, max_concurrency=run_cfg.max_concurrency)

    async def run_pool(task, spec: PoolSpec):
        t_idx_in_gt = args.task_ids.index(task.task_id)
        gt_start, gt_end = int(gt_offsets[t_idx_in_gt]), int(gt_offsets[t_idx_in_gt + 1])
        task_gt_ids = gt_ids[gt_start:gt_end]
        task_gt_embs = gt_emb[gt_start:gt_end]
        gt_entries = [
            (gid, gid.rsplit("_", 1)[-1], body, emb)
            for gid, body, emb in zip(task_gt_ids, task.gt_skill_bodies, task_gt_embs)
        ]
        pool = build_pool(spec, task, corpus, task_embedding=task_emb_by_id[task.task_id], gt_entries=gt_entries)
        pool_block = render_pool_block(pool, representation="card")
        (out_dir / "pools" / f"{spec.pool_id}.json").write_text(json.dumps({
            "spec": spec.model_dump(),
            "display_ids": pool.display_ids,
            "id_map": pool.id_map,
            "gt_display_ids": pool.gt_display_ids,
        }, indent=2))

        probe_records: list[TrialRecord] = []
        for probe in ["awareness", "selection"]:
            full_prompt = render_awareness_prompt(task.instruction, pool) if probe == "awareness" else render_selection_prompt(task.instruction, pool)
            if not will_fit(full_prompt, MODEL_CONTEXT_LIMIT, run_cfg.context_safety_margin):
                with (out_dir / "skipped.jsonl").open("a") as f:
                    f.write(json.dumps({"pool_id": spec.pool_id, "probe": probe, "reason": "context_overflow"}) + "\n")
                continue
            user_prompt = full_prompt.split("Available skills")[0] + "Respond per the protocol above."
            rec = await driver.run_one(
                pool_id=spec.pool_id, probe=probe,
                system_prompt="You are a retrieval subject in a controlled study.",
                pool_block=pool_block,
                user_prompt=user_prompt,
            )
            probe_records.append(rec)
            (out_dir / "raw" / f"{spec.pool_id}__{probe}.txt").write_text(rec.raw_response)
            (out_dir / "parsed" / f"{spec.pool_id}__{probe}.json").write_text(rec.model_dump_json(indent=2))
        return spec.pool_id, probe_records, pool

    coros = []
    for task in tasks:
        for strategy in args.strategies:
            for n in args.pool_sizes:
                for seed in args.seeds:
                    spec = PoolSpec(task_id=task.task_id, strategy=strategy, n=n, seed=seed)
                    coros.append(run_pool(task, spec))

    results = await asyncio.gather(*coros)

    per_trial: list[dict] = []
    for pool_id, recs, pool in results:
        pool_map = {"id_map": pool.id_map, "gt_display_ids": pool.gt_display_ids}
        trial_row: dict = {"pool_id": pool_id}
        for rec in recs:
            parsed = {"extracted_ids": rec.extracted_ids, "format_status": rec.format_status, "flags": rec.flags}
            scored = score_trial(parsed, pool_map, probe=rec.probe)
            trial_row.update({f"{rec.probe}.{k}": v for k, v in scored.items() if isinstance(v, (int, float))})
            trial_row[f"{rec.probe}.format_status"] = rec.format_status
        per_trial.append(trial_row)

    (out_dir / "metrics" / "per_trial.jsonl").write_text("\n".join(json.dumps(r) for r in per_trial))
    flat = []
    for r in per_trial:
        flat.append({
            "awareness_recall5": r.get("awareness.awareness_recall5", 0),
            "awareness_top1": r.get("awareness.awareness_top1", 0),
            "awareness_mrr": r.get("awareness.awareness_mrr", 0.0),
            "selection_top1": r.get("selection.selection_top1", 0),
            "parse_fail": max(r.get("awareness.parse_fail", 0), r.get("selection.parse_fail", 0)),
        })
    summary = aggregate_metrics(flat)
    (out_dir / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. Summary → {out_dir / 'metrics' / 'summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
