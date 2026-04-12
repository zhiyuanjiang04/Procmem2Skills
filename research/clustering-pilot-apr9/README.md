# Clustering Pilot - Apr 9

Clustering 241 task descriptions from `terminal-bench/original-tasks` with
Qwen3-Embedding-0.6B.

## Files

| File | Purpose |
|------|---------|
| `01_extract_tasks.py` | Parse task.yaml from terminal-bench/original-tasks → tasks.jsonl |
| `02_embed_tasks.py` | Run Qwen3-Embedding-0.6B on instructions, save embeddings.npy |
| `03_cluster.py` | DBSCAN sweep, write per-eps cluster contents |
| `05_compare_methods.py` | DBSCAN vs HDBSCAN vs Agglomerative on same embeddings |
| `06_text_variants.py` | full / first_sentence / stripped / title_only text variants |

## Outputs

| File | Content |
|------|---------|
| `outputs/tasks.jsonl` | 241 task records (instruction + category + tags) |
| `outputs/sweep_summary.json` | DBSCAN sweep summary table |
| `outputs/clusters/eps_*.json` | Full cluster membership per DBSCAN eps |
| `outputs/method_comparison.json` | DBSCAN/HDBSCAN/Agglomerative metrics |
| `outputs/text_variants_comparison.json` | Text variant metrics |

## Reproduce

```bash
git clone https://github.com/laude-institute/terminal-bench.git
pip install pyyaml torch transformers scikit-learn hdbscan numpy

python 01_extract_tasks.py
python 02_embed_tasks.py     # downloads Qwen3-Embedding-0.6B (~1.2 GB) on first run
python 03_cluster.py
python 05_compare_methods.py
python 06_text_variants.py
```

`embeddings.npy` is regenerable and not included here.
