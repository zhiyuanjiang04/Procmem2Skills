"""Post-process llm_clusters.json: drop singleton clusters into unclustered.

Refine often outputs split decisions that produce 1-task "clusters". A skill
needs at least 2 tasks to be considered useful, so move all single-member
groups to unclustered for the headline numbers.

Writes llm_clusters_clean.json next to llm_clusters.json (does not overwrite).
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "outputs" / "llm_clusters.json"
DST = ROOT / "outputs" / "llm_clusters_clean.json"


def main():
    d = json.loads(SRC.read_text())
    multi = []
    extra_unclustered = []
    for c in d["clusters"]:
        if len(c.get("member_ids", [])) >= 2:
            multi.append(c)
        else:
            extra_unclustered.extend(c.get("member_ids", []))

    final_unclustered = sorted(set(d.get("unclustered", []) + extra_unclustered))

    out = {
        "model": d.get("model"),
        "n_input_tasks": d.get("n_input_tasks"),
        "n_final_clusters": len(multi),
        "n_unclustered": len(final_unclustered),
        "tasks_in_multi_clusters": sum(len(c["member_ids"]) for c in multi),
        "fraction_clustered": round(
            sum(len(c["member_ids"]) for c in multi) / d.get("n_input_tasks", 241), 3
        ),
        "total_cost_usd": d.get("total_cost_usd"),
        "post_processing_note": (
            "Singleton final_groups (1-task clusters from split decisions) moved to unclustered. "
            "A skill requires at least 2 tasks to be useful."
        ),
        "clusters": multi,
        "unclustered": final_unclustered,
    }
    DST.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"saved → {DST}")
    print(f"  {out['n_final_clusters']} multi-clusters, "
          f"{out['n_unclustered']} unclustered")
    print(f"  tasks in multi-clusters: {out['tasks_in_multi_clusters']} "
          f"({out['fraction_clustered']*100:.1f}% of total)")


if __name__ == "__main__":
    main()
