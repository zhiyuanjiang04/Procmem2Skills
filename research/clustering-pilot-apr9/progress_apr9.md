# 进展 - Apr 9

## 这周做的

跑了 terminal-bench original-tasks 里 241 个任务的描述聚类。流程是 task.yaml
里的 instruction 字段抽出来，用 Qwen3-Embedding-0.6B 算 embedding（1024
维，L2 normalize），然后 DBSCAN 扫了 9 个 eps 阈值从 0.20 到 0.70。

代码在 clustering/ 下，4 个脚本：抽取、embedding、聚类、出报告。

## 看到的

阈值扫描有个明显的拐点在 0.45 左右：

| eps  | clusters | noise%  | max | purity |
|------|---------:|--------:|----:|-------:|
| 0.30 |       18 |  84.2%  |   3 |   0.86 |
| 0.40 |       26 |  71.0%  |  16 |   0.84 |
| 0.45 |       25 |  57.7%  |  40 |   0.80 |
| 0.50 |       22 |  37.8%  | 103 |   0.78 |
| 0.55 |       13 |  23.2%  | 153 |   0.67 |
| 0.70 |        1 |   0.0%  | 241 |   0.20 |

eps=0.40 之前 cluster 都是 2-3 个任务的小群组，纯度很高但孤立点占 71%。
过了 0.45 之后立刻链式合并——eps=0.50 时最大 cluster 一口气吃掉 103 个任务
(43% 的任务全在一个 cluster 里)，里面 software-engineering、security、
file-operations、model-training 全混在一起，purity 掉到 0.78 但其实已经
没意义了。

eps=0.40 的小 cluster 看起来都很合理，举几个：
- 3 个迷宫任务都在一起（blind-maze-explorer-5x5/algorithm + interactive-maze-game）
- 3 个棋类视觉任务在一起（chess-best-move + gomoku-planner + regex-chess）
- 3 个 security/extract 任务在一起（crack-7z-hash + extract-safely + git-leak-recovery）
- 3 个 CSV/JSON 数据处理在一起
- 16 个 ML/data 任务在一起（HF model、word2vec、speech-to-text、play-zork
  之类——这个稍微有点宽但还能接受）

eval_report.md 里有 26 个 cluster 的完整内容。

## 卡住或不确定的

1. 71% 孤立点太多了。要么任务描述本身差异就大，要么 DBSCAN 不适合这种
   长尾分布的数据，可能需要换 HDBSCAN 或者 Agglomerative。

2. 0.45→0.50 的链式合并是 DBSCAN 本身的问题，single-linkage 一旦有个桥接
   任务把两个语义不同的群连起来就崩了。如果要更稳的方案可能要换算法。

3. instruction 是自然语言描述，里面有不少废话（"You are given..."、"Your
   task is to..."），可能影响 embedding。试试先用 LLM 摘要成关键词再
   embedding 应该会更紧。

## 下周

- 用 HDBSCAN 重跑一遍对比，看链式合并问题能不能解决
- 试一下 instruction 摘要：每个任务让 LLM 输出 5 个关键词，再 embedding
- 跟 Zhiyuan 对一下 pipeline 那边的接入点：聚类结果是直接喂给 skill-creator，
  还是要再做一层"cluster 内代表性轨迹挑选"
