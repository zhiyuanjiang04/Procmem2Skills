"""DBSCAN clustering with eps threshold sweep."""
import json
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).parent
EMB = ROOT / "outputs" / "embeddings.npy"
IDS = ROOT / "outputs" / "task_ids.json"
TASKS = ROOT / "outputs" / "tasks.jsonl"
OUT_DIR = ROOT / "outputs" / "clusters"
SUMMARY = ROOT / "outputs" / "sweep_summary.json"

EPS_VALUES = [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
MIN_SAMPLES = 2


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    emb = np.load(EMB)
    task_ids = json.load(open(IDS))
    rows = [json.loads(l) for l in open(TASKS)]
    id_to_meta = {r["task_id"]: r for r in rows}
    print(f"loaded embeddings {emb.shape}, {len(task_ids)} task ids")

    # cosine distance = 1 - cosine similarity (vectors are L2-normalized so dot = cos)
    sim = emb @ emb.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)

    summary = []
    for eps in EPS_VALUES:
        labels = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, metric="precomputed").fit_predict(dist)
        clusters = {}
        for tid, lab in zip(task_ids, labels):
            clusters.setdefault(int(lab), []).append(tid)
        noise = clusters.pop(-1, [])
        n_clusters = len(clusters)
        sizes = sorted([len(v) for v in clusters.values()], reverse=True)

        # category purity per cluster
        purities = []
        for tids in clusters.values():
            cats = [id_to_meta[t].get("category", "") or "unknown" for t in tids]
            top = Counter(cats).most_common(1)[0][1]
            purities.append(top / len(cats))
        avg_purity = float(np.mean(purities)) if purities else 0.0

        row = {
            "eps": eps,
            "n_clusters": n_clusters,
            "n_noise": len(noise),
            "noise_rate": len(noise) / len(task_ids),
            "max_cluster": sizes[0] if sizes else 0,
            "median_cluster": median(sizes) if sizes else 0,
            "avg_category_purity": round(avg_purity, 3),
        }
        summary.append(row)
        print(
            f"eps={eps:.2f}  clusters={n_clusters:3d}  noise={len(noise):3d} "
            f"({row['noise_rate']*100:4.1f}%)  max={row['max_cluster']:3d} "
            f"med={row['median_cluster']:.1f}  purity={avg_purity:.2f}"
        )

        # save full cluster contents
        out = {
            "eps": eps,
            "n_clusters": n_clusters,
            "n_noise": len(noise),
            "clusters": {str(k): v for k, v in sorted(clusters.items())},
            "noise": noise,
        }
        with open(OUT_DIR / f"eps_{eps:.2f}.json", "w") as f:
            json.dump(out, f, indent=2)

    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary → {SUMMARY}")
    print(f"per-eps clusters → {OUT_DIR}")


if __name__ == "__main__":
    main()
