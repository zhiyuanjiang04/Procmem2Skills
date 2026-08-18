window.siteContent = {
  title: "Demystifying Agent Skills: Why They Work - Until They Don't",
  question: "Aggregate success tells us that a skill helped. It does not tell us why, when, or whether the agent used the right one.",
  authors: [
    { name: "Zhiyuan Jiang", aff: "Princeton University", mark: "1,*" },
    { name: "Fangrui Huang", aff: "Stanford University", mark: "3,*" },
    { name: "Hanwen Xing", aff: "University of Southern California", mark: "4" },
    { name: "Xander Wu", aff: "Stanford University", mark: "3" },
    { name: "Yipeng Gao", aff: "University of Southern California", mark: "4" },
    { name: "Rui Cao", aff: "Johns Hopkins University", mark: "5" },
    { name: "Mengdi Wang", aff: "Princeton University", mark: "1,dagger" },
    { name: "Shilong Liu", aff: "Princeton University", mark: "1,dagger" },
    { name: "Yijiang Li", aff: "UC San Diego", mark: "2,dagger" }
  ],
  affiliations: [
    "1 Princeton University",
    "2 UC San Diego",
    "3 Stanford University",
    "4 University of Southern California",
    "5 Johns Hopkins University"
  ],
  abstract: "Skills are structured packages of procedural experience for LLM agents. We study not only whether they improve task success, but also how representation, outcome annotation, retrieval, and invocation shape their behavior. Across controlled experiments and contrastive trajectory analysis, we find that skills work best when noisy traces become compact procedural anchors, while retrieval mismatch, brittle assumptions, and insufficient adaptation create clear failure boundaries.",
  findings: [
    {
      eyebrow: "Representation",
      value: "61.9% vs 55.9%",
      title: "Skill vs Workflow Memory",
      body: "Matched downstream success when the same source experience is packaged in two different procedural forms.",
      evidence: [{ label: "Skill", value: "61.9%", width: 61.9, tone: "skill" }, { label: "Workflow", value: "55.9%", width: 55.9, tone: "workflow" }],
      note: "+6.06 points for Skill over Workflow Memory."
    },
    {
      eyebrow: "Mechanism",
      value: "65.7% vs 4.5%",
      title: "Procedural anchoring vs knowledge injection",
      body: "Share of skill mechanisms in the taxonomy, showing what the agent behavior actually changed.",
      evidence: [{ label: "Procedural anchor", value: "65.7%", width: 65.7, tone: "skill" }, { label: "Knowledge injection", value: "4.5%", width: 4.5, tone: "orange" }],
      note: "Skill cases are primarily about stabilizing action, not adding facts."
    },
    {
      eyebrow: "Retrieval",
      value: "29.6% -> 3.3%",
      title: "Actual-use precision, k=5 -> k=100",
      body: "Exact ground-truth skill use during independent execution experiments as the candidate pool grows.",
      evidence: [{ label: "k=5", value: "29.6%", width: 29.6, tone: "skill" }, { label: "k=100", value: "3.3%", width: 3.3, tone: "orange" }],
      note: "Downstream success stays comparatively stable: 36.4% -> 39.3%."
    }
  ],
  taxonomy: [
    { id: "SC1", title: "Procedural anchoring", body: "The agent follows a reusable sequence or verification routine that makes execution more reliable." },
    { id: "SC2", title: "Execution and verification failures", body: "The artifact is present, but setup, implementation, runtime, or final checking still breaks down." },
    { id: "SC3", title: "Invocation, applicability, and boundary failures", body: "The guidance is ignored, over-applied, mismatched to the task, or limited by an external bottleneck." }
  ],
  modes: [
    "Procedural anchoring", "Environment setup", "Output compliance", "Service lifecycle",
    "Shell execution", "Algorithmic implementation", "Runtime validation", "Failure warning",
    "No meaningful use", "Counterproductive guidance", "Applicability mismatch", "External boundary"
  ],
  resources: {
    paper: "https://arxiv.org/",
    code: "https://github.com/zhiyuanjiang04/Procmem2Skills",
    template: "https://github.com/nerfies/nerfies.github.io"
  },
  bibtex: "@article{jiang2026demystifying,\n  title   = {Demystifying Agent Skills: Why They Work - Until They Don't},\n  author  = {Jiang, Zhiyuan and Huang, Fangrui and Xing, Hanwen and Wu, Xander and Gao, Yipeng and Cao, Rui and Wang, Mengdi and Liu, Shilong and Li, Yijiang},\n  year    = {2026},\n  note    = {Preprint}\n}"
};
