"""Evaluate clustering quality and generate markdown report."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CLUSTERS_DIR = DATA / "clusters"
SUMMARY = DATA / "sweep_summary.json"
TASKS = DATA / "tasks.jsonl"
REPORT = ROOT / "eval_report.md"


def load_tasks():
    return {json.loads(l)["task_id"]: json.loads(l) for l in open(TASKS)}


def main():
    tasks = load_tasks()
    summary = json.load(open(SUMMARY))

    lines = []
    lines.append("# Clustering Sweep Report\n")
    lines.append(f"Embeddings: Qwen3-Embedding-0.6B (1024-dim, L2 normalized)  ")
    lines.append(f"Algorithm: DBSCAN with cosine distance, min_samples=2  ")
    lines.append(f"Tasks: 241 from terminal-bench/original-tasks\n")

    # Sweep table
    lines.append("## Threshold Sweep\n")
    lines.append("| eps | clusters | noise | noise% | max | median | category purity |")
    lines.append("|-----|---------:|------:|-------:|----:|-------:|----------------:|")
    for r in summary:
        lines.append(
            f"| {r['eps']:.2f} | {r['n_clusters']} | {r['n_noise']} | "
            f"{r['noise_rate']*100:.1f}% | {r['max_cluster']} | {r['median_cluster']:.1f} | "
            f"{r['avg_category_purity']:.2f} |"
        )
    lines.append("")

    # Sweet spot analysis
    lines.append("## Observation\n")
    lines.append(
        "Two regimes separated by a sharp transition at eps≈0.45:\n"
        "- **eps ≤ 0.40**: many small clusters, very high noise (71-98%), high purity (0.84-0.88). "
        "Tight semantic groups, but most tasks remain isolated.\n"
        "- **eps ≥ 0.50**: chain merging collapses everything into one giant cluster. "
        "At eps=0.50, the largest cluster contains 103 tasks (43% of all tasks) spanning "
        "software-engineering, security, file-operations, model-training all at once. "
        "Purity drops accordingly.\n"
        "\n"
        "Sweet spot is **eps=0.40**: 26 clusters, max size 16, average category purity 0.84. "
        "Beyond this, DBSCAN's transitive linking pulls in unrelated tasks.\n"
    )

    # Detailed look at eps=0.40
    chosen = "0.40"
    lines.append(f"## Cluster Contents at eps={chosen}\n")
    data = json.load(open(CLUSTERS_DIR / f"eps_{chosen}.json"))
    clusters = data["clusters"]
    sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))
    lines.append(f"{data['n_clusters']} clusters, {data['n_noise']} noise ({data['n_noise']/241*100:.1f}%)\n")
    for cid, members in sorted_clusters:
        cats = Counter(tasks[t].get("category", "?") or "?" for t in members)
        cat_str = ", ".join(f"{c}×{n}" for c, n in cats.most_common(3))
        lines.append(f"### Cluster {cid} ({len(members)} tasks) — {cat_str}\n")
        for t in members:
            instr = tasks[t]["instruction"].replace("\n", " ").strip()[:120]
            lines.append(f"- `{t}` — {instr}...")
        lines.append("")

    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT}")
    print(f"  {len(lines)} lines, {len(sorted_clusters)} clusters detailed at eps={chosen}")


if __name__ == "__main__":
    main()
