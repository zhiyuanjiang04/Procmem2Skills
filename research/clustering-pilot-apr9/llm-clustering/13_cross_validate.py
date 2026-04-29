"""Cross-validate LLM clustering against DBSCAN eps=0.40.

Reports:
  - For each DBSCAN cluster: whether LLM kept it together, split it, or scattered it
  - Agreement statistics (Adjusted Rand Index, V-measure if sklearn available)
  - LLM-only clusters that DBSCAN didn't capture
"""
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).parent
LLM_FILE = ROOT / "outputs" / "llm_clusters.json"
DBSCAN_FILE = ROOT.parent / "data" / "clusters" / "eps_0.40.json"
OUT_FILE = ROOT / "outputs" / "cross_validation.md"


def load_assignment(clusters_data: dict, key: str = "clusters") -> dict[str, str]:
    """Map task_id → cluster_id (or '_NOISE' / '_UNCLUST')."""
    assign = {}
    if isinstance(clusters_data.get(key), list):
        for c in clusters_data[key]:
            cid = c.get("id") or c.get("cluster_id") or "?"
            for tid in c.get("member_ids", []):
                assign[tid] = cid
    elif isinstance(clusters_data.get(key), dict):
        # DBSCAN format: {"clusters": {"0": [...]}, "noise": [...]}
        for cid, members in clusters_data[key].items():
            for tid in members:
                assign[tid] = f"D{cid}"
    for tid in clusters_data.get("noise", []):
        assign[tid] = "_NOISE"
    for tid in clusters_data.get("unclustered", []):
        assign[tid] = "_UNCLUST"
    return assign


def per_dbscan_cluster_breakdown(dbscan_assign: dict, llm_assign: dict) -> list[dict]:
    """For each non-noise DBSCAN cluster, see how its members map in LLM clustering."""
    by_d_cluster = {}
    for tid, dcid in dbscan_assign.items():
        if dcid == "_NOISE":
            continue
        by_d_cluster.setdefault(dcid, []).append(tid)

    out = []
    for dcid, tids in sorted(by_d_cluster.items(), key=lambda x: -len(x[1])):
        llm_targets = [llm_assign.get(t, "_MISSING") for t in tids]
        counts = Counter(llm_targets)
        biggest = counts.most_common(1)[0]
        # how unified is this DBSCAN cluster in LLM space?
        unity = biggest[1] / len(tids)
        out.append({
            "dbscan_cluster": dcid,
            "size": len(tids),
            "members": tids,
            "llm_distribution": dict(counts),
            "unity": round(unity, 3),
        })
    return out


def pairwise_agreement(dbscan_assign: dict, llm_assign: dict, only_non_noise: bool = True) -> dict:
    """For all task pairs, agreement on 'same cluster?' between DBSCAN and LLM."""
    common = sorted(set(dbscan_assign.keys()) & set(llm_assign.keys()))
    if only_non_noise:
        common = [t for t in common if dbscan_assign[t] != "_NOISE" and llm_assign[t] != "_UNCLUST"]
    same_dbscan_same_llm = 0
    same_dbscan_diff_llm = 0
    diff_dbscan_same_llm = 0
    diff_dbscan_diff_llm = 0
    for a, b in combinations(common, 2):
        s_d = dbscan_assign[a] == dbscan_assign[b]
        s_l = llm_assign[a] == llm_assign[b]
        if s_d and s_l: same_dbscan_same_llm += 1
        elif s_d and not s_l: same_dbscan_diff_llm += 1
        elif not s_d and s_l: diff_dbscan_same_llm += 1
        else: diff_dbscan_diff_llm += 1
    n = len(common)
    n_pairs = n * (n - 1) // 2
    # Rand Index = agreement / total pairs
    agreement = same_dbscan_same_llm + diff_dbscan_diff_llm
    rand_index = agreement / n_pairs if n_pairs else 0.0
    return {
        "n_tasks_compared": n,
        "n_pairs": n_pairs,
        "rand_index": round(rand_index, 4),
        "same_in_both": same_dbscan_same_llm,
        "same_dbscan_only": same_dbscan_diff_llm,
        "same_llm_only": diff_dbscan_same_llm,
        "different_in_both": diff_dbscan_diff_llm,
    }


def llm_extra_clusters(llm_data: dict, dbscan_assign: dict) -> list[dict]:
    """LLM clusters whose members were noise in DBSCAN — what DBSCAN missed."""
    extras = []
    for c in llm_data.get("clusters", []):
        n_dbscan_noise = sum(1 for t in c["member_ids"] if dbscan_assign.get(t) == "_NOISE")
        if n_dbscan_noise >= 2:
            extras.append({
                "llm_cluster": c["id"],
                "skill_concept": c["skill_concept"],
                "size": len(c["member_ids"]),
                "n_dbscan_noise": n_dbscan_noise,
                "fraction_noise": round(n_dbscan_noise / len(c["member_ids"]), 3),
                "members": c["member_ids"],
            })
    extras.sort(key=lambda x: -x["fraction_noise"])
    return extras


def main():
    if not LLM_FILE.exists():
        raise SystemExit(f"missing {LLM_FILE}; run 09 + 10 first")
    if not DBSCAN_FILE.exists():
        raise SystemExit(f"missing {DBSCAN_FILE}; ensure DBSCAN clusters are present")

    llm_data = json.loads(LLM_FILE.read_text())
    dbscan_data = json.loads(DBSCAN_FILE.read_text())

    llm_assign = load_assignment(llm_data)
    dbscan_assign = load_assignment(dbscan_data)

    breakdown = per_dbscan_cluster_breakdown(dbscan_assign, llm_assign)
    agreement = pairwise_agreement(dbscan_assign, llm_assign)
    extras = llm_extra_clusters(llm_data, dbscan_assign)

    # Write markdown report
    lines = []
    lines.append("# LLM Clustering vs DBSCAN eps=0.40\n")
    lines.append(f"LLM clusters: {len(llm_data['clusters'])}, "
                 f"unclustered: {len(llm_data.get('unclustered', []))}")
    lines.append(f"DBSCAN multi-task clusters: {len(breakdown)}\n")

    lines.append("## Pairwise agreement\n")
    lines.append(f"- Tasks compared (non-noise & non-unclustered in both): "
                 f"**{agreement['n_tasks_compared']}**")
    lines.append(f"- Total pairs: {agreement['n_pairs']}")
    lines.append(f"- **Rand Index: {agreement['rand_index']}**")
    lines.append(f"- Same cluster in both: {agreement['same_in_both']}")
    lines.append(f"- Together in DBSCAN only: {agreement['same_dbscan_only']}")
    lines.append(f"- Together in LLM only: {agreement['same_llm_only']}")
    lines.append(f"- Apart in both: {agreement['different_in_both']}\n")

    lines.append("## How each DBSCAN cluster maps into LLM clustering\n")
    lines.append("`unity` = fraction of cluster members that ended up in the same LLM cluster.")
    lines.append("1.0 means LLM kept them all together; lower means LLM split them.\n")
    lines.append("| DBSCAN | size | unity | LLM distribution |")
    lines.append("|--------|-----:|------:|------------------|")
    for b in breakdown:
        dist_str = ", ".join(f"{k}×{v}" for k, v in sorted(b["llm_distribution"].items(), key=lambda x: -x[1]))
        lines.append(f"| {b['dbscan_cluster']} | {b['size']} | {b['unity']} | {dist_str} |")

    lines.append("\n## LLM clusters DBSCAN missed (≥2 DBSCAN-noise members)\n")
    if not extras:
        lines.append("None.")
    else:
        for e in extras[:30]:
            lines.append(f"### LLM {e['llm_cluster']} — \"{e['skill_concept']}\"")
            lines.append(f"  size={e['size']}, DBSCAN-noise members: {e['n_dbscan_noise']} ({e['fraction_noise']*100:.0f}%)")
            lines.append(f"  members: {', '.join(e['members'])}\n")

    OUT_FILE.write_text("\n".join(lines))
    print(f"saved → {OUT_FILE}")
    print(f"  Rand Index: {agreement['rand_index']}")
    print(f"  DBSCAN clusters analyzed: {len(breakdown)}")
    print(f"  LLM-extra clusters (≥2 noise members): {len(extras)}")


if __name__ == "__main__":
    main()
