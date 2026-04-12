# Clustering Pilot - Apr 9

Pilot exploration for clustering Terminal-Bench task descriptions with Qwen3-Embedding-0.6B
and DBSCAN. Goal: see if "cluster similar tasks → distill one skill per cluster" is a viable
pipeline structure for ProcMem2Skills.

## Files

| File | Purpose |
|------|---------|
| `01_extract_tasks.py` | Parse task.yaml from terminal-bench/original-tasks, output tasks.jsonl |
| `02_embed_tasks.py` | Run Qwen3-Embedding-0.6B on instructions, save embeddings.npy |
| `03_cluster.py` | DBSCAN sweep over 9 eps values (0.20–0.70), per-cluster category purity |
| `04_eval.py` | Generate eval_report.md with sweep table + cluster contents at chosen eps |
| `eval_report.md` | Detailed cluster contents at the recommended eps=0.40 |
| `progress_apr9.md` | Weekly sync update for the team |

## Reproduce

```bash
# from terminal-bench checkout in sibling directory
git clone https://github.com/laude-institute/terminal-bench.git
pip install pyyaml torch transformers scikit-learn numpy

python 01_extract_tasks.py
python 02_embed_tasks.py     # downloads Qwen3-Embedding-0.6B (~1.2 GB) on first run
python 03_cluster.py
python 04_eval.py
```

`data/embeddings.npy` is not committed (~1 MB but regenerable, kept out for hygiene).

## Headline Result

Sharp transition at eps≈0.45. eps=0.40 gives 26 small clusters with 0.84 average category
purity but 71% noise. Past eps=0.45 chain merging collapses clusters into one giant
group (eps=0.50 → max cluster of 103 tasks). See progress_apr9.md and eval_report.md
for details.
