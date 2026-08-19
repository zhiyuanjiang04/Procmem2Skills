# ClawHub Noise Pool — 交接说明

修正版 skill retrieval 实验的 noise skills pool。从 ClawHub 爬的真实 skill 原件（不是 description-only stub），对应 `skill_retrieval_experiment.md` 第一步 3.2 的要求。

## 这是什么

- 从 ClawHub registry 按 `downloads` 降序取 top 10000，下载完整 skill 原件。
- 每个 skill 一个目录，含 `SKILL.md` 正文 + 附带脚本/资源 + `metadata.json`。
- 实际落地 9998 个（2 个是 slug 为乱码的 spam skill，下载失败，已剔除）。

## 目录结构

```
skill_retrieval/
├── noise_pool/
│   └── {slug}/
│       ├── SKILL.md         # 完整正文（含 frontmatter）
│       ├── *.py / *.md ...  # skill 自带的附件，原样保留
│       └── metadata.json
├── clawhub_full_manifest.json   # 全量 39982 个 slug 的索引（slug/ver/downloads）
├── download_noise.py            # 下载脚本（可复现、可 resume、可改 N/排序扩样）
├── quick_quality_check.py       # BM25 质检脚本
├── download_errors.log          # 失败/警告日志
└── download_dedup.log           # 内容去重日志（本次 0 条）
```

## metadata.json 字段

```json
{
  "skill_name": "<slug>",
  "source": "clawhub",
  "url": "https://clawhub.ai/api/v1/download?slug=<slug>",
  "description": "...",        // 取自 SKILL.md frontmatter 的 description；没写则 fallback 取正文首段
  "download_time": 1780897xxx, // unix 秒
  "hash": "<sha256 of zip>",   // 内容指纹，用于去重
  "downloads": 73306,
  "version": "1.0.5"
}
```

`description` 就是 retrieval 实验里要 embed 的那一段。用 `yaml.safe_load` 解析 frontmatter，已正确处理块标量（`description: |` 多行）。

## 数据情况（跑之前先知道这几点）

- **空描述 1.3%**（127 条）：其中 123 条是 skill 本身没带 SKILL.md（见 `download_errors.log` 的 `NO_SKILL_MD`）。构 pool 时建议过滤掉空描述的。
- **中文 skill 占 35.9%**（3594 条）：SkillsBench task 全英文，qwen 语义检索下中文 skill 基本进不了英文 task 的 similar top，实际只会当 random/dissimilar 噪声。**如果想让 similar/random/dissimilar 三种 noise 的语言基线一致，建议先过滤成纯英文子集**。需要的话用 metadata 里 description 是否含 CJK 一筛即可。
- **0 内容重复**：top 10000 按 zip sha256 去重，没有近重复。
- downloads 区间 410 到 177779（median 475）。

## 快速质检结论（BM25 粗筛，非 qwen）

跑 `quick_quality_check.py` 的结论：常见域 task（文档处理、旅行、数据分析）在 pool 里有大量真实可信的语义近邻，能构成有意义的 similar distractor（例：invoice 任务命中 pdf-invoice-parser / extract-pdf-compdf / pdf-analysis）。专业小众 task（GLM 湖泊模拟、Flink、Azure BGP）pool 里没有对口 skill，similar distractor 对这类 task 偏弱，属数据层面的客观情况。BM25 是词面匹配，长 description 会被通用词带偏，换 qwen 语义检索会明显更准。

## 扩样 / 换样

想换数量或换排序键，改 `download_noise.py`：

```bash
# 改取样数（默认 10000，按 downloads 降序）
python3 download_noise.py 20000

# 想随机抽 / 按别的键排序，改 main() 里的 recs.sort(...)；
# 全量 39982 个 slug 在 clawhub_full_manifest.json，无需重新翻页。
```

脚本是 resume 的（已存在 `metadata.json` 的 slug 会跳过），断了重跑不会重复下载。注意非交互 shell 要用带 pyyaml 的解释器（本机是 `/opt/homebrew/bin/python3.12`）。

## 数据来源

ClawHub registry（`https://clawhub.ai`，OpenClaw 的公开 skill 库，类 npm）。全程免 auth 只读。
- 列表：`GET /api/v1/skills?limit=200&sort=createdAt`，`nextCursor` 游标翻页。
- 下载：`GET /api/v1/download?slug=<slug>`，返回 zip。
