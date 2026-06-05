#!/usr/bin/env python3
"""Generate experiment design diagram for SkillsBench Eval Pipeline.

Style reference: 122.png (RQ1-RQ2 two-panel layout with flowchart boxes).
Output: High-res PNG suitable for NotebookLM import.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Colors matching 122.png style
C_BG = "#FFFFFF"
C_HEADER_A = "#4A6FA5"  # blue header
C_HEADER_B = "#6B8E5A"  # green header
C_HEADER_C = "#C06040"  # orange header
C_BOX_DB = "#E8EEF6"    # light blue (data)
C_BOX_PROC = "#FFF3E0"  # light orange (process)
C_BOX_METRIC = "#E8F5E9" # light green (metric)
C_BOX_CONFIG = "#F3E5F5" # light purple (config)
C_BOX_SKILL = "#FFF9C4"  # light yellow (skill)
C_BOX_ALERT = "#FFEBEE"  # light red (alert/finding)
C_ARROW = "#555555"
C_TEXT = "#222222"


def rounded_box(ax, x, y, w, h, text, color, fontsize=7, bold=False, text_color="#222"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor="#999", linewidth=0.8)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=text_color,
            wrap=True, multialignment="center")


def header_box(ax, x, y, w, h, text, color):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                         facecolor=color, edgecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")


def arrow(ax, x1, y1, x2, y2, text="", curved=False):
    style = "Simple,tail_width=0.3,head_width=4,head_length=3"
    if curved:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                connectionstyle="arc3,rad=0.15",
                                arrowstyle=style, color=C_ARROW, linewidth=0.8)
    else:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle=style, color=C_ARROW, linewidth=0.8)
    ax.add_patch(arrow)
    if text:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.015, text, ha="center", va="bottom",
                fontsize=5.5, color="#666", style="italic")


def main():
    fig, axes = plt.subplots(3, 1, figsize=(10, 16),
                              gridspec_kw={"height_ratios": [1.1, 0.7, 0.5]})
    fig.patch.set_facecolor(C_BG)

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("auto")
        ax.axis("off")

    # =================================================================
    # PANEL A: Overall Pipeline Design
    # =================================================================
    ax = axes[0]
    header_box(ax, 0.02, 0.92, 0.96, 0.07,
               "SkillsBench Prefill-Context Evaluation Pipeline", C_HEADER_A)

    # -- Data source --
    rounded_box(ax, 0.05, 0.78, 0.2, 0.10,
                "SkillsBench\nTask Repository\n(88 tasks)", C_BOX_DB, 7, True)
    rounded_box(ax, 0.30, 0.78, 0.2, 0.10,
                "ClawHub\nSkill Corpus\n(44,787 skills)", C_BOX_DB, 7, True)

    # Filtering
    rounded_box(ax, 0.55, 0.78, 0.20, 0.10,
                "Task Filtering\nHost-executable\nDocker lightweight", C_BOX_PROC, 7)
    arrow(ax, 0.25, 0.83, 0.55, 0.83)

    # Result of filtering
    rounded_box(ax, 0.78, 0.78, 0.18, 0.10,
                "50 Eval Tasks\n3 Batches", C_BOX_SKILL, 7, True)
    arrow(ax, 0.75, 0.83, 0.78, 0.83)

    # -- Two evaluation tracks --
    # Track 1: Selection Eval
    rounded_box(ax, 0.05, 0.58, 0.42, 0.14,
                "Stage 1: Skill Selection Eval\n\n"
                "Task description + Candidate pool\n"
                "Model picks best skill(s)\n"
                "Metrics: Hit@1, Recall, Refusal",
                C_BOX_PROC, 7)
    ax.text(0.26, 0.725, "Prompt-and-Parse", ha="center",
            fontsize=6, color=C_HEADER_A, fontweight="bold")

    # Track 2: Execution Eval
    rounded_box(ax, 0.53, 0.58, 0.42, 0.14,
                "Stage 2: Prefill Execution Eval\n\n"
                "GT skills prefilled in system prompt\n"
                "Agent executes task (Bash/Read/Write/Edit)\n"
                "Validation: pytest test_outputs.py",
                C_BOX_PROC, 7)
    ax.text(0.74, 0.725, "Docker Sandbox", ha="center",
            fontsize=6, color=C_HEADER_A, fontweight="bold")

    # Arrows from data to tracks
    arrow(ax, 0.15, 0.78, 0.15, 0.72)
    arrow(ax, 0.40, 0.78, 0.60, 0.72)
    arrow(ax, 0.50, 0.83, 0.74, 0.72, "GT skills")

    # -- Factorial Design --
    rounded_box(ax, 0.05, 0.38, 0.90, 0.15,
                "", C_BOX_CONFIG, 7)
    ax.text(0.50, 0.52, "Factorial Design", ha="center",
            fontsize=8, fontweight="bold", color="#6A1B9A")

    # Config boxes inside
    configs = [
        ("Pool Size\n5 | 10 | 20 | 50 | 100", 0.07, 0.39, 0.17, 0.11),
        ("Noise Mode\neasy | hard | random", 0.26, 0.39, 0.17, 0.11),
        ("Seeds\n0 | 1 | 2 | 3 | 4", 0.45, 0.39, 0.14, 0.11),
        ("Model\nClaude Sonnet 4.6", 0.61, 0.39, 0.16, 0.11),
        ("Timeout\n900s wall limit", 0.79, 0.39, 0.14, 0.11),
    ]
    for text, x, y, w, h in configs:
        rounded_box(ax, x, y, w, h, text, "#EDE7F6", 6.5)

    arrow(ax, 0.26, 0.58, 0.26, 0.535)
    arrow(ax, 0.74, 0.58, 0.74, 0.535)

    # -- Three batches --
    ax.text(0.50, 0.34, "Three Evaluation Batches", ha="center",
            fontsize=8, fontweight="bold", color=C_TEXT)

    batch_data = [
        ("First20\n14 evaluable tasks\nseeds [0-4]\npools [5,10,20,50]\n2333+ trials",
         0.05, 0.12, 0.27, 0.20, "#E3F2FD"),
        ("Smoke10\n10 tasks\nseeds [0-4]\npools [5,10,20,50,100]\n720 trials",
         0.37, 0.12, 0.26, 0.20, "#FFF3E0"),
        ("Smoke20\n20 tasks\nseeds [0-4]\npools [5,10,20,50,100]\n1500 trials",
         0.68, 0.12, 0.27, 0.20, "#F1F8E9"),
    ]
    for text, x, y, w, h, color in batch_data:
        rounded_box(ax, x, y, w, h, text, color, 7)

    # Metrics at bottom
    metrics = [
        ("Pass Rate", 0.10, 0.03, 0.15, 0.06, C_BOX_METRIC),
        ("Timeout Rate", 0.28, 0.03, 0.15, 0.06, C_BOX_METRIC),
        ("Wall Time", 0.46, 0.03, 0.12, 0.06, C_BOX_METRIC),
        ("Error Taxonomy", 0.61, 0.03, 0.15, 0.06, C_BOX_METRIC),
        ("API Check", 0.79, 0.03, 0.12, 0.06, C_BOX_METRIC),
    ]
    for text, x, y, w, h, color in metrics:
        rounded_box(ax, x, y, w, h, text, color, 6.5, True)
    for _, bx, by, bw, bh, _ in batch_data:
        arrow(ax, bx + bw/2, by, bx + bw/2, 0.09)

    # =================================================================
    # PANEL B: Key Analysis Dimensions
    # =================================================================
    ax = axes[1]
    header_box(ax, 0.02, 0.88, 0.96, 0.10,
               "Analysis Dimensions & Findings", C_HEADER_B)

    # Four analysis columns
    analyses = [
        ("Selection vs\nExecution\nComparison",
         "Hit@1 >= 83%\nfor 47/50 tasks\nbut 19 tasks = 0%\nexec pass",
         0.03, C_BOX_DB, C_BOX_ALERT),
        ("Pool Size x\nNoise Mode\nGrid",
         "No monotonic\ndegradation\nFlat across\nall conditions",
         0.26, "#FFF3E0", "#FFF9C4"),
        ("Timeout\nDecomposition\n(4-metric)",
         "pass/total\nTO/total\npass/non-TO\ncompletion%",
         0.49, C_BOX_CONFIG, "#EDE7F6"),
        ("API Exhaustion\nVerification",
         "15 patterns scanned\n900 trials checked\nZERO API hits\nAll failures genuine",
         0.72, "#E8F5E9", "#C8E6C9"),
    ]

    for title, finding, x, c1, c2 in analyses:
        rounded_box(ax, x, 0.50, 0.22, 0.34, title, c1, 7, True)
        rounded_box(ax, x, 0.10, 0.22, 0.34, finding, c2, 6.5)
        arrow(ax, x + 0.11, 0.50, x + 0.11, 0.44)

    # =================================================================
    # PANEL C: Core Conclusion
    # =================================================================
    ax = axes[2]
    header_box(ax, 0.02, 0.78, 0.96, 0.18,
               "Core Conclusions", C_HEADER_C)

    conclusions = [
        ("1. Selection-Execution\nDecoupling",
         "Skill retrieval is near-ceiling.\n"
         "Bottleneck = task difficulty,\n"
         "not skill selection.",
         0.03, 0.08, 0.30, 0.62),
        ("2. No Pool/Noise\nEffect",
         "Pass rate flat across\n"
         "pool 5-100, noise easy/\n"
         "hard/random. Hard noise\n"
         "hurts selection but\n"
         "NOT execution.",
         0.36, 0.08, 0.28, 0.62),
        ("3. Data Integrity\nVerified",
         "No API exhaustion.\n"
         "50 disjoint tasks,\n"
         "4553 trials,\n"
         "all genuine results.",
         0.67, 0.08, 0.28, 0.62),
    ]
    for title, body, x, y, w, h in conclusions:
        rounded_box(ax, x, y, w, h, "", "#FFF8E1")
        ax.text(x + w/2, y + h - 0.08, title, ha="center", va="top",
                fontsize=7.5, fontweight="bold", color="#E65100")
        ax.text(x + w/2, y + h/2 - 0.06, body, ha="center", va="center",
                fontsize=6.5, color=C_TEXT, multialignment="center")

    plt.tight_layout(pad=0.5)
    out = "results/experiment_design_diagram.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=C_BG)
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
