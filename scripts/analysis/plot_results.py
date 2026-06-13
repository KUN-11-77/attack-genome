"""Attack Genome Tier C++ — 3 张核心图。
Usage: python plot_results.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/data3/khsong/cogatedrive/exp/figures")
OUT.mkdir(parents=True, exist_ok=True)

# parse 3 shard summaries
shards = []
for s in [0, 1, 2]:
    p = f"/data3/khsong/cogatedrive/outputs/ag_tier_cpp_shard{s}/summary.txt"
    with open(p) as f:
        text = f.read()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [x.strip() for x in line.split("|")]
        if len(parts) >= 3 and parts[0] in ["CNN", "DINO", "TF"]:
            attack = parts[1]
            rest = parts[2]
            if "asrs=" in rest:
                asrs_str = rest.split("asrs=")[1]
                shards.append({"planner": parts[0], "attack": attack,
                               "asrs_str": asrs_str, "shard": s})

print("parsed", len(shards), "rows from 3 shards", flush=True)
if not shards:
    print("ERROR: no rows parsed", flush=True)
    sys.exit(1)

df = pd.DataFrame(shards)
df["asrs_list"] = df["asrs_str"].apply(
    lambda s: [float(x.strip()) for x in s.strip("[]").split(",")])
agg = df.groupby(["planner", "attack"])["asrs_list"].apply(
    lambda lists: np.mean(lists, axis=0)).reset_index()

attacks_order = ["Rain", "Snow", "Dusk", "Dawn", "MotionBlur",
                 "DigitalNoise", "LightDust", "DappledLight",
                 "VintageStyle", "CarlaStyle"]
strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
planners = ["CNN", "DINO", "TF"]
colors = {"CNN": "#d62728", "DINO": "#1f77b4", "TF": "#2ca02c"}

# ===== Plot 1: ASR curves =====
fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
for i, atk in enumerate(attacks_order):
    ax = axes[i // 5][i % 5]
    for p in planners:
        sub = agg[(agg["planner"] == p) & (agg["attack"] == atk)]
        if len(sub) == 0:
            continue
        asrs = sub.iloc[0]["asrs_list"]
        ax.plot(strengths, asrs, "-o", color=colors[p], label=p,
                linewidth=2, markersize=6)
    ax.set_title(atk, fontsize=12)
    ax.set_xlabel("Strength")
    ax.set_ylabel("ASR")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    if i == 0:
        ax.legend(loc="upper left", fontsize=9)
fig.suptitle("Tier C++ ASR curves (49 scenes, 3 planners)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig1_asr_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig1_asr_curves.png", flush=True)

# ===== Plot 2: Gene importance heatmap =====
imp = pd.read_csv("/data3/khsong/cogatedrive/exp/failure_factor/feature_importance.csv",
                  index_col=0)
for c in ["min_imp", "mean_imp", "max_diff"]:
    if c in imp.columns:
        imp = imp.drop(columns=[c])
cols = sorted(imp.columns.tolist())
imp = imp[cols]
top20 = imp.sort_values(cols[0], ascending=False).head(20)  # sort by first planner

fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(top20.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(top20.columns)))
ax.set_xticklabels(top20.columns, fontsize=11)
ax.set_yticks(range(len(top20.index)))
ax.set_yticklabels(top20.index, fontsize=9)
for i in range(len(top20.index)):
    for j in range(len(top20.columns)):
        v = top20.values[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=7, color="black" if v < 0.04 else "white")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Feature importance", fontsize=10)
ax.set_title("Top-20 Gene Features (Random Forest importance)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT / "fig2_gene_importance_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig2_gene_importance_heatmap.png", flush=True)

# ===== Plot 3: Failure Basin =====
csv = pd.read_csv("/data3/khsong/cogatedrive/outputs/ag_tier_cpp_merged/per_sample_genes.csv")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, p in enumerate(planners):
    ax = axes[i]
    sub = csv[csv["planner"] == p]
    x = sub["lane_yellow_ratio"].values
    y = sub["edge_density"].values
    c = sub["success"].values
    fail = c == 1
    ok = c == 0
    ax.scatter(x[ok], y[ok], c="#1f77b4", s=4, alpha=0.4, label="clean")
    ax.scatter(x[fail], y[fail], c="#d62728", s=4, alpha=0.4, label="fail")
    ax.set_title("%s (fail=%.1f%%)" % (p, fail.mean() * 100), fontsize=12)
    ax.set_xlabel("lane_yellow_ratio")
    ax.set_ylabel("edge_density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
fig.suptitle("Failure Basin: lane_yellow_ratio vs edge_density", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig3_failure_basin.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig3_failure_basin.png", flush=True)

print("=== ALL DONE. Figures in", OUT)
for f in sorted(os.listdir(OUT)):
    p = OUT / f
    print("  %s: %d KB" % (f, p.stat().st_size // 1024))
