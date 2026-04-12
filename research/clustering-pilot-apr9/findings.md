# Clustering Findings

What was tried, what came out, and what it suggests for ProcMem2Skills.

Embedding model: Qwen3-Embedding-0.6B (1024-dim, last-token pool, L2 normalized).
Tasks: 241 from `terminal-bench/original-tasks`.

## 1. Algorithm comparison (DBSCAN vs HDBSCAN vs Agglomerative)

Same embeddings (full instruction text), three clustering algorithms swept across
their natural threshold parameter. Reporting only multi-task clusters; singletons
are counted as effective noise.

### DBSCAN

| eps  | multi-cl | noise% | max | purity | tag jacc | silhouette |
|------|---------:|-------:|----:|-------:|---------:|-----------:|
| 0.30 |       18 |  84.2% |   3 |   0.86 |     0.64 |      0.635 |
| 0.40 |       26 |  71.0% |  16 |   0.84 |     0.57 |      0.383 |
| 0.45 |       25 |  57.7% |  40 |   0.80 |     0.43 |      0.197 |
| 0.50 |       22 |  37.8% | 103 |   0.78 |     0.40 |      0.063 |

Clean small clusters at low eps, but 71-98% of tasks stay as singletons.
The chain merge between eps=0.45 and 0.50 is where unrelated groups
get connected through bridging tasks — max cluster jumps 40 → 103.

### HDBSCAN

| min_cluster_size | multi-cl | noise% | max | purity |
|-----------------:|---------:|-------:|----:|-------:|
|                2 |        2 |   7.9% | 219 |   0.60 |
|                3 |        3 |  53.9% | 102 |   0.66 |
|                4 |        2 |  57.7% |  96 |   0.61 |
|                5 |        0 | 100.0% |   0 |      - |

HDBSCAN does not work on this data. With min_cluster_size=2 it dumps everything
into one 219-task cluster. With ≥5 it returns no clusters at all. The instruction
embeddings don't have the clean density gradients HDBSCAN needs — embeddings are
spread fairly uniformly in cosine space without obvious dense regions.

### Agglomerative (average linkage)

| threshold | multi-cl | noise% | max | purity | tag jacc |
|----------:|---------:|-------:|----:|-------:|---------:|
|      0.40 |       29 |  73.9% |   4 |   0.85 |     0.54 |
|      0.45 |       39 |  63.5% |   6 |   0.78 |     0.44 |
|      0.50 |       54 |  47.7% |  10 |   0.74 |     0.32 |
|      0.55 |       56 |  35.7% |  13 |   0.71 |     0.26 |

Agglomerative average-linkage gives the smoothest tradeoff. At thr=0.50 it produces
54 multi-task clusters (twice as many as DBSCAN's best) with 48% noise, max cluster
size 10, and category purity 0.74. Importantly, average linkage avoids the chain-merge
collapse that DBSCAN's single linkage suffers from — max cluster size grows
gradually instead of jumping.

**Algorithm pick:** Agglomerative average-linkage @ thr ≈ 0.50 looks like the right
default if we go ahead with embedding-based clustering on instruction text.

## 2. Text variant comparison

Holding the algorithm fixed (Agglomerative average-linkage), changing what text
gets embedded.

| variant | description | mean text length |
|---------|-------------|-----------------:|
| full | original `instruction` from task.yaml | 1198 |
| first_sentence | only the first sentence | 116 |
| stripped | remove "You are given / Your task is to / etc." | 895 |
| title_only | just the kebab-case task name with dashes → spaces | 18 |

At thr=0.45:

| variant | multi-cl | noise% | max | purity | tag jacc |
|---------|---------:|-------:|----:|-------:|---------:|
| full           | 19 | 58.1% | 64 | 0.76 | 0.36 |
| first_sentence | 31 | 54.8% | 38 | 0.72 | 0.24 |
| stripped       | 17 | 61.8% | 54 | 0.73 | 0.27 |
| title_only     | 49 | 27.8% | 15 | 0.63 | 0.22 |

`title_only` is the surprise: 49 multi-clusters with only 28% noise and max
cluster size 15. That's 2.5× more grouped tasks than `full`, half the noise,
and no chain merging. Purity drops (0.63 vs 0.76) but absolute purity is still
above random.

The other variants (`first_sentence`, `stripped`) only marginally help — the
boilerplate isn't the main issue, the long natural-language descriptions just
have low semantic density compared to the curated task names.

**Reading:** task names in terminal-bench are essentially human-written semantic
tags. They cluster well because they're already condensed. Long instructions
dilute that signal across embedding space.

## 3. How to talk about cluster quality

Five metrics in play, none of them good enough alone:

| metric | what it measures | weakness |
|--------|------------------|----------|
| effective noise rate | fraction of tasks NOT in any multi-cluster | trivially solved by linking everything |
| n multi-clusters | number of actual groups | can be inflated by tiny pairs |
| max cluster size | signal of chain merging | a single 100-task cluster can dominate |
| category purity | top category fraction per cluster | terminal-bench categories are coarse (60+ tasks per category) |
| tag jaccard | mean pairwise tag-set overlap | tags are sparse, many tasks have ≤2 tags |
| silhouette | geometric tightness vs separation | doesn't reflect semantic correctness |

**For ProcMem2Skills the question we actually care about is**: "can a single skill
serve all tasks in this cluster?" That's a downstream question — neither purity nor
silhouette answers it directly. Two ways forward:

- **LLM-as-judge**: feed each cluster's task descriptions to a model and ask
  "would one skill handle all of these?". Slow and expensive but the only metric
  aligned with our actual goal. Worth running on the top-20 clusters.

- **Solution-overlap**: each terminal-bench task has `solution.sh`. Within a
  cluster, count overlap of called tools/commands. Cheap to compute, more
  semantically grounded than embedding distance.

For the writeup I'd report all five quantitative metrics for the algorithm
comparison, plus an LLM-judge sample on the chosen final method to validate
that "category purity 0.74" actually corresponds to "humans agree these belong
together".

## What the next step should be

Two parallel paths worth running:

1. **Use task names instead of instructions for the first-pass cluster, then use
   the instructions only for refinement.** Task names give the global structure;
   instructions can split clusters that are too coarse.

2. **Build the LLM-as-judge evaluation harness now**, because (a) we'll need it
   for the final paper anyway, and (b) it's the only way to compare clustering
   methods on something that matches the actual downstream goal.
