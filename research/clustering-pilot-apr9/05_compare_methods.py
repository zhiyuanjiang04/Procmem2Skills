"""Compare DBSCAN, HDBSCAN, and Agglomerative clustering on the same embeddings.

Metrics reported:
  - n_clusters / noise_rate / max_cluster_size
  - category purity (top-1 category fraction per cluster, averaged)
  - tag jaccard (mean pairwise jaccard of tag sets within each cluster)
  - silhouette score (where defined)
"""
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean, median

import hdbscan
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).parent
DATA = ROOT / "data"
EMB_FILE = DATA / "embeddings.npy"
IDS_FILE = DATA / "task_ids.json"
TASKS_FILE = DATA / "tasks.jsonl"
OUT_FILE = DATA / "method_comparison.json"


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def cluster_metrics(labels, task_ids, id_to_meta, dist_matrix):
    clusters = {}
    for tid, lab in zip(task_ids, labels):
        clusters.setdefault(int(lab), []).append(tid)
    noise_label_tids = clusters.pop(-1, [])
    singletons = [c for c in clusters.values() if len(c) == 1]
    multi = [c for c in clusters.values() if len(c) >= 2]
    # for fair cross-method comparison: effective noise = explicit noise + singleton tasks
    n_effective_noise = len(noise_label_tids) + len(singletons)
    n_multi_clusters = len(multi)
    sizes = sorted([len(v) for v in multi], reverse=True)

    purities, tag_jaccards = [], []
    for tids in multi:
        cats = [id_to_meta[t].get("category", "") or "?" for t in tids]
        purities.append(Counter(cats).most_common(1)[0][1] / len(cats))
        tag_sets = [set(id_to_meta[t].get("tags", []) or []) for t in tids]
        pair_scores = [jaccard(a, b) for a, b in combinations(tag_sets, 2)]
        if pair_scores:
            tag_jaccards.append(mean(pair_scores))

    # silhouette only if there are at least 2 clusters and not too noisy
    sil = None
    valid_mask = np.array(labels) != -1
    if valid_mask.sum() > 2 and len(set(np.array(labels)[valid_mask])) >= 2:
        try:
            sil = float(silhouette_score(
                dist_matrix[valid_mask][:, valid_mask],
                np.array(labels)[valid_mask],
                metric="precomputed",
            ))
        except Exception:
            sil = None

    return {
        "n_multi_clusters": n_multi_clusters,
        "n_effective_noise": n_effective_noise,
        "effective_noise_rate": round(n_effective_noise / len(task_ids), 3),
        "max_cluster": sizes[0] if sizes else 0,
        "median_cluster": float(median(sizes)) if sizes else 0,
        "category_purity": round(mean(purities), 3) if purities else 0.0,
        "tag_jaccard": round(mean(tag_jaccards), 3) if tag_jaccards else 0.0,
        "silhouette": round(sil, 3) if sil is not None else None,
    }


def main():
    emb = np.load(EMB_FILE)
    task_ids = json.load(open(IDS_FILE))
    rows = [json.loads(l) for l in open(TASKS_FILE)]
    id_to_meta = {r["task_id"]: r for r in rows}
    print(f"loaded {emb.shape}")

    sim = np.clip(emb @ emb.T, -1.0, 1.0)
    dist = (1.0 - sim).astype(np.float64)
    np.fill_diagonal(dist, 0.0)

    results = {"dbscan": [], "hdbscan": [], "agglomerative": []}

    print("\n=== DBSCAN ===")
    print(f"{'eps':>5} {'clu':>4} {'noise%':>7} {'max':>4} {'med':>5} {'pur':>5} {'jacc':>5} {'sil':>6}")
    for eps in [0.30, 0.35, 0.40, 0.45, 0.50]:
        labels = DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit_predict(dist)
        m = cluster_metrics(labels, task_ids, id_to_meta, dist)
        m["param"] = {"eps": eps}
        results["dbscan"].append(m)
        sil = f"{m['silhouette']:.3f}" if m['silhouette'] is not None else "  -  "
        print(f"{eps:>5.2f} {m['n_multi_clusters']:>4} {m['effective_noise_rate']*100:>6.1f}% "
              f"{m['max_cluster']:>4} {m['median_cluster']:>5.1f} {m['category_purity']:>5.2f} "
              f"{m['tag_jaccard']:>5.2f} {sil:>6}")

    print("\n=== HDBSCAN ===")
    print(f"{'mcs':>5} {'clu':>4} {'noise%':>7} {'max':>4} {'med':>5} {'pur':>5} {'jacc':>5} {'sil':>6}")
    for mcs in [2, 3, 4, 5, 8]:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            metric="precomputed",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(dist)
        m = cluster_metrics(labels, task_ids, id_to_meta, dist)
        m["param"] = {"min_cluster_size": mcs}
        results["hdbscan"].append(m)
        sil = f"{m['silhouette']:.3f}" if m['silhouette'] is not None else "  -  "
        print(f"{mcs:>5} {m['n_multi_clusters']:>4} {m['effective_noise_rate']*100:>6.1f}% "
              f"{m['max_cluster']:>4} {m['median_cluster']:>5.1f} {m['category_purity']:>5.2f} "
              f"{m['tag_jaccard']:>5.2f} {sil:>6}")

    print("\n=== Agglomerative (average linkage, distance threshold) ===")
    print(f"{'thr':>5} {'clu':>4} {'noise%':>7} {'max':>4} {'med':>5} {'pur':>5} {'jacc':>5} {'sil':>6}")
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=thr,
            metric="precomputed",
            linkage="average",
        ).fit_predict(dist)
        m = cluster_metrics(labels, task_ids, id_to_meta, dist)
        m["param"] = {"distance_threshold": thr}
        results["agglomerative"].append(m)
        sil = f"{m['silhouette']:.3f}" if m['silhouette'] is not None else "  -  "
        print(f"{thr:>5.2f} {m['n_multi_clusters']:>4} {m['effective_noise_rate']*100:>6.1f}% "
              f"{m['max_cluster']:>4} {m['median_cluster']:>5.1f} {m['category_purity']:>5.2f} "
              f"{m['tag_jaccard']:>5.2f} {sil:>6}")

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
