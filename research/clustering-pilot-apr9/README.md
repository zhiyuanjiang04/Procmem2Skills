# Clustering Pilot - Apr 9

Pilot for clustering Terminal-Bench task descriptions with Qwen3-Embedding-0.6B and
DBSCAN. Trying to see if "cluster similar tasks → distill one skill per cluster" is a
viable structure for the ProcMem2Skills pipeline.

## Files

| File | Purpose |
|------|---------|
| `01_extract_tasks.py` | Parse task.yaml from terminal-bench/original-tasks, output tasks.jsonl |
| `02_embed_tasks.py` | Run Qwen3-Embedding-0.6B on instructions, save embeddings.npy |
| `03_cluster.py` | DBSCAN sweep over 9 eps values (0.20–0.70), per-cluster category purity |
| `04_eval.py` | Generate eval_report.md (sweep table + cluster contents at chosen eps) |
| `eval_report.md` | Sweep table and cluster contents at eps=0.40 |

## Reproduce

```bash
git clone https://github.com/laude-institute/terminal-bench.git
pip install pyyaml torch transformers scikit-learn numpy

python 01_extract_tasks.py
python 02_embed_tasks.py     # downloads Qwen3-Embedding-0.6B (~1.2 GB) on first run
python 03_cluster.py
python 04_eval.py
```

`data/` is excluded from the repo (regenerable).

## Findings

Tested 241 tasks from `terminal-bench/original-tasks`. Sweep results:

| eps  | clusters | noise%  | max | category purity |
|------|---------:|--------:|----:|----------------:|
| 0.30 |       18 |  84.2%  |   3 |   0.86 |
| 0.40 |       26 |  71.0%  |  16 |   0.84 |
| 0.45 |       25 |  57.7%  |  40 |   0.80 |
| 0.50 |       22 |  37.8%  | 103 |   0.78 |
| 0.55 |       13 |  23.2%  | 153 |   0.67 |
| 0.70 |        1 |   0.0%  | 241 |   0.20 |

DBSCAN flips behavior between eps=0.45 and 0.50. Below 0.45 you get small tight
clusters (2-16 tasks each) but most tasks (71%) stay as singletons. At 0.50 a
single bridging task is enough to chain-merge unrelated groups — the largest
cluster jumps to 103 tasks and mixes software-engineering, security,
file-operations, model-training all together.

eps=0.40 is the most usable point. 26 clusters, max size 16, category purity 0.84.
Examples of what comes out:

- 3 maze tasks together (`blind-maze-explorer-5x5/algorithm`, `interactive-maze-game`)
- 3 board-game-from-image tasks together (`chess-best-move`, `gomoku-planner`, `regex-chess`)
- 3 security/extract tasks together (`crack-7z-hash`, `extract-safely`, `git-leak-recovery`)
- 3 CSV→JSON/parquet tasks together
- 16 ML/data tasks together (HF inference, word2vec, speech-to-text, play-zork) — broader but not unreasonable

The 71% singleton rate is the main concern. Two possible reasons: (a) the
instruction text is genuinely diverse, or (b) the noise inside instructions
("You are given...", "Your task is to...") drowns out the semantic core.
Worth trying instruction summarization (LLM → 5 keywords → embed) before
giving up on density-based clustering.

DBSCAN single-linkage is fragile here. HDBSCAN or agglomerative with average-linkage
would likely give a smoother curve without the 0.45→0.50 collapse.

See `eval_report.md` for the full 26 clusters at eps=0.40.
