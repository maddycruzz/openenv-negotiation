import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 8b scores — read live from baseline_results.json
with open("baseline_results.json") as f:
    results = json.load(f)

task_order = [
    "single-round-consensus",
    "multi-round-negotiation",
    "adversarial-information",
    "pediatric-meningitis",
    "opioid-overdose",
]

labels = [
    "Easy\nConsensus",
    "Medium\nNegotiation",
    "Hard\nAdversarial",
    "Hard\nMeningitis",
    "Hard\nOpioid",
]

scores_8b = {r["task_id"]: r["score"] for r in results}

# 70b reference scores (llama-3.3-70b-versatile, pre-tightened grader run)
scores_70b = {
    "single-round-consensus": 0.99,
    "multi-round-negotiation": 0.99,
    "adversarial-information": 0.6329,
    "pediatric-meningitis": 0.99,
    "opioid-overdose": 0.7606,
}

y_8b  = [scores_8b.get(t, 0)  for t in task_order]
y_70b = [scores_70b.get(t, 0) for t in task_order]

x = np.arange(len(task_order))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

bars_weak   = ax.bar(x - width/2, y_8b,  width, label="Weak Model (8B)",   color="#3b82f6", alpha=0.85)
bars_strong = ax.bar(x + width/2, y_70b, width, label="Strong Model (70B)", color="#10b981", alpha=0.85)

for bar in bars_weak:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            color="#3b82f6", fontsize=9, fontweight="bold")

for bar in bars_strong:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            color="#10b981", fontsize=9, fontweight="bold")

ax.set_xlabel("Task Difficulty", color="white", fontsize=12, labelpad=10)
ax.set_ylabel("Score", color="white", fontsize=12, labelpad=10)
ax.set_title("Social Agent Negotiation — Baseline Score Comparison",
             color="white", fontsize=14, fontweight="bold", pad=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, color="white", fontsize=10)
ax.set_ylim(0, 1.1)
ax.tick_params(axis="y", colors="white")
ax.spines["bottom"].set_color("#333")
ax.spines["left"].set_color("#333")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, color="#222", linewidth=0.7)
ax.set_axisbelow(True)

legend = ax.legend(facecolor="#161b22", edgecolor="#333",
                   labelcolor="white", fontsize=10, loc="upper right")

plt.tight_layout()
plt.savefig("reward_curve_final.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
print("Saved: reward_curve_final.png")
