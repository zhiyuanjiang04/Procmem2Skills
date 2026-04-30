# LLM Clustering — Handoff

完整的两层 hierarchical LLM 聚类 pipeline 跑完了，结果 + 代码 + 中间产物都在当前目录。完全跳过 embedding，从任务描述到最终 cluster 都是用 Claude Sonnet 4.6 API 跑的。

## 方法

第一层粗聚类把 241 个任务按 60 个一批喂给 LLM，每批输出 15 个左右的候选 cluster（5 batches 共产出 59 个候选）。第二层对每个候选单独再调一次，这次喂完整 instruction + solution.sh，让 LLM 判 keep / split / remove_some，同时为 cluster 内每个任务单独标 applicable: YES/PARTIAL/NO + reason。最后再起一个独立的 judge（不复用前面的 context），读 final cluster + skill_concept + 任务证据，投票 ACCEPT / BORDERLINE / REJECT。Judge 结果做了人工抽样校对，cluster 划分跟 skill_concept 匹配度看起来没什么问题。

## 跑出来的结果

- 25 个 multi-task cluster 覆盖 61 个任务，剩下 180 unclustered（74.7%）
- Refine 阶段 33/59 是 split——粗聚类太松，细化时把不该一起的拆开了
- Judge: 8 ACCEPT / 17 BORDERLINE / 0 REJECT
- 跟 DBSCAN eps=0.40 的 Rand Index 0.9256，但有个细节：DBSCAN 觉得是同 cluster 的 76 对里 75 对被 LLM 拆了（LLM 严格得多），同时 LLM 在 DBSCAN 标 noise 的任务里多发现了 15 个 cluster

## 主产出

- `outputs/llm_clusters_clean.json` — 主结果。25 个 final cluster，每个有 skill_concept、member_ids、reasoning、per-task task_judgments
- `outputs/llm_judge.json` — 独立 judge 的 verdict、verdict_reason、weakest_link
- `outputs/cross_validation.md` — 跟 DBSCAN eps=0.40 的对比报告
- `outputs/llm_coarse.json` — 第一层粗聚类（59 候选 cluster，含每个 cluster 来自哪个 batch）
- `outputs/llm_clusters.json` — 第二层细化原始结果（含单例 group，未做 post-process）
- `outputs/_*_cache/` — 每个 LLM 调用的中间结果，resume 用

## 下一步建议

25 个 cluster 走 cluster-level skill，剩下 180 个走 per-task skill 的 hybrid 路径。要选 pilot 的话从 8 个 ACCEPT 里挑，比如 B1_C7.1（迷宫 DFS）、B2_C1.1（PEFT LoRA）、B4_C1.1（reproduce open-source bug）这几个 skill_concept 都很 concrete，适合先验证 cluster-level skill 涨点能不能逼近 per-task skill。

180 unclustered 这个数字跟之前 DBSCAN 跑出来的 71% noise 一致，是 terminal-bench 任务集本身决定的——多数任务真的没有"明显能共用 skill 的同伴"。不建议强行降这个数字，hybrid 路径反而比"硬聚 100% 但 cluster 都不准"实用。

## 复现命令

```bash
source venv312/bin/activate

python 08_prepare_llm_input.py
python 09_llm_coarse_cluster.py
PARALLEL=4 python 10_llm_refine_clusters.py
python 12_postprocess.py
PARALLEL=4 python 11_llm_judge.py
python 13_cross_validate.py
```

每步都有缓存到 `outputs/_*_cache/`，跑挂了重跑会跳过已完成的。

详细方法、prompt 设计、参数选择都在 README.md 里。
