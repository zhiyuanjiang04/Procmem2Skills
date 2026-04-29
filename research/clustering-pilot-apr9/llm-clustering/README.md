# LLM Clustering Pipeline

完全跳过 embedding，用 Claude Sonnet 4.6 对 241 个 terminal-bench 任务做两层 hierarchical 聚类，每个 cluster 输出一个候选 `skill_concept`，再用独立 LLM judge 验证质量。

## 跑下来的实际结果

| 阶段 | 数字 |
|------|------|
| 输入任务 | 241 |
| Coarse 阶段 multi-cluster | 59（5 batches × 12-15 个） |
| Refine 后真正 ≥2 任务的 cluster | **25** |
| 在 multi-cluster 里的任务总数 | 61（25.3% 总任务） |
| Unclustered | 180（74.7%） |
| 独立 judge 投票 | 8 ACCEPT / 17 BORDERLINE / 0 REJECT |
| vs DBSCAN eps=0.40 的 Rand Index | **0.9256** |
| LLM 多发现的 cluster（DBSCAN 全 noise） | 15 |
| 总成本 | $5.62 |

## 为什么

之前用 DBSCAN/HDBSCAN/Agglomerative 在 Qwen embedding 上聚类，所有定量指标（noise rate、cluster 数、category purity、silhouette）都跟"一个 skill 能不能 cover"这个真实目标对不齐。LLM 看到任务描述 + solution.sh 之后能直接判断 procedure 相似性。

## 设计

两层 hierarchical + 独立验证：
- **第一层**（粗）：5 个 batch，每 batch 60 个任务的精简卡片送给 LLM，输出 ~15 个候选 cluster。LLM 单次最多处理 ~60 任务，再大就超时
- **第二层**（细）：每个粗 cluster 一次 LLM 调用，输入 cluster 内任务的完整 instruction + solution.sh，决定 keep / split / remove_some
- **独立 judge**：另起一次 LLM 调用，看 final cluster + skill_concept + 任务证据，投票 ACCEPT / BORDERLINE / REJECT

## 文件

| 脚本 | 作用 | 调用次数 |
|------|------|---------|
| `08_prepare_llm_input.py` | 抽取 task card（goal + tools + category） | 0（纯本地） |
| `09_llm_coarse_cluster.py` | 第一层粗聚类（5 batches，concat） | 5 |
| `10_llm_refine_clusters.py` | 第二层细化（PARALLEL=N 并行） | ~59（每 coarse cluster 一次） |
| `11_llm_judge.py` | 独立 judge 评分 | ~25（每 final multi-cluster 一次） |
| `12_postprocess.py` | 把单例 cluster 归到 unclustered | 0 |
| `13_cross_validate.py` | 跟 DBSCAN eps=0.40 交叉验证 | 0 |

## 产出

| 文件 | 说明 |
|------|------|
| `outputs/task_cards.jsonl` | 241 个精简任务卡片 |
| `outputs/llm_coarse.json` | 第一层粗聚类（59 cluster，含 batch source 信息） |
| `outputs/llm_clusters.json` | 第二层细化原始结果（含 95 final_groups，包括单例） |
| `outputs/llm_clusters_clean.json` | **主产出**：post-process 后 25 个 multi-cluster + 180 unclustered |
| `outputs/llm_judge.json` | 独立 judge 投票（25 个 cluster 的 verdict） |
| `outputs/cross_validation.md` | LLM 聚类 vs DBSCAN 的对比报告 |
| `outputs/_coarse_batches/` | 粗聚类中间结果（resume cache） |
| `outputs/_refine_cache/` | 细化中间结果（resume cache，PARALLEL 安全） |
| `outputs/_judge_cache/` | judge 中间结果 |

## 跑法

```bash
# 1. 装环境（venv312 已有 transformers/torch/sklearn）
source venv312/bin/activate

# 2. 抽 task card（本地，~5s）
python 08_prepare_llm_input.py

# 3. 第一层粗聚类（5 batches，~30 分钟，会缓存到 _coarse_batches/）
python 09_llm_coarse_cluster.py

# 4. 第二层细化（默认串行；推荐 PARALLEL=4 加速到 ~45 分钟）
PARALLEL=4 python 10_llm_refine_clusters.py

# 5. Post-process 单例 → unclustered（本地，<1s）
python 12_postprocess.py

# 6. 独立 judge（PARALLEL=4，~10 分钟）
PARALLEL=4 python 11_llm_judge.py

# 7. 跟 DBSCAN 交叉验证（本地，<1s）
python 13_cross_validate.py
```

所有步骤都有缓存：跑挂了重跑会跳过已完成的 cluster。

## 模型与认证

- 全程 **Claude Sonnet 4.6**
- 通过 `claude -p` headless CLI 调用，走 **Claude Code Max OAuth**（订阅额度，不消耗 API credit）
- 关键参数：`--tools ""`（禁用 tool-use，单 turn 生成）、`--no-session-persistence`、`input=prompt`（stdin 传输大 prompt 必须）

## 输入数据

- `clustering/data/tasks.jsonl`：241 个任务的 instruction + 元数据（之前抽好的）
- `tb-work/terminal-bench/original-tasks/<task_id>/solution.sh`：每个任务的 oracle 解

## 关键发现

1. **Refine 阶段 33/59 决策是 split**——说明 LLM 在 60 任务一次的视角下倾向于过度归并。Refine 看到完整 instruction + solution 后就把那些不该一起的拆开
2. **Rand Index 0.9256** 跟 DBSCAN 高度一致——LLM 没有"任性"地推翻 embedding 的判断，只是补足了 DBSCAN 没抓到的 pattern
3. **15 个 LLM-only cluster** 是 DBSCAN 完全标 noise 的——典型例子：broken-python + conda-env-conflict-resolution，adaptive-rejection-sampler + bn-fit-modify
4. **Judge 0 REJECT** 但 17/25 BORDERLINE——说明大部分 cluster"勉强能用"但有 weakest_link，每个 cluster 的 verdict_reason 里指出了具体哪个任务最不贴合 skill_concept
