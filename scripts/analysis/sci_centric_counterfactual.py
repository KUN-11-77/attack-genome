"""SCI-centric Tier C2 analysis (local, CPU-only).

Uses existing per-sample counterfactual data from 3 planners to:
1. Bucket samples by their per-sample top1 SCI category (Road Structure / Color / Texture / etc.)
2. Compute flip rate per category per planner
3. Test if Road Structure interventions have higher flip rate than other categories

This is a proxy for Tier C2: instead of running new planner forward-passes,
we re-analyze the existing counterfactual data with an SCI lens.

Inputs:
  - d:/cogatedrive/exp/tierB_partial/failure_basin_{cnn,dino,tf}/counterfactual_per_sample.csv
  - d:/cogatedrive/exp/tierB_partial/merged_3pl.csv (for full gene vector)
Output:
  - d:/cogatedrive/exp/tierB_partial/sci_centric/sci_centric_summary.json
  - d:/cogatedrive/exp/tierB_partial/sci_centric/sci_centric_table.md
"""
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = Path("d:/cogatedrive/exp/tierB_partial")
OUT = BASE / "sci_centric"
OUT.mkdir(parents=True, exist_ok=True)

# SCI category mapping: which genes fall under which SCI
SCI_CATEGORIES = {
    "Road_Structure": ["edge_mean", "edge_density", "lane_line_count", "lane_line_density"],
    "Frequency": ["low_freq_ratio", "mid_freq_ratio", "high_freq_ratio", "spectral_centroid"],
    "Color": ["hue_mean", "hue_std", "sat_mean", "sat_std", "val_mean", "val_std", "colorfulness"],
    "Texture": ["lbp_entropy", "glcm_contrast", "lbp_uniformity"],
    "Illumination": ["shadow_ratio", "highlight_ratio", "luma_entropy", "luma_skew", "luma_mean"],
    "Luma": ["rms_contrast", "mean_luma", "std_luma", "dynamic_range", "mean_shift", "std_shift"],
    "Detection": ["vehicle_loss", "person_loss", "detection_loss", "conf_loss", "vehicle_loss_ratio"],
}

GENE_TO_CAT = {}
for cat, genes in SCI_CATEGORIES.items():
    for g in genes:
        GENE_TO_CAT[g] = cat


def categorize_top1(top1_gene: str) -> str:
    return GENE_TO_CAT.get(top1_gene, "Other")


def parse_top_genes(s: str) -> list:
    """Parse 'edge_mean' or 'edge_mean;lane_line_count' into a list."""
    if pd.isna(s):
        return []
    return [g.strip() for g in re.split(r"[,;\s]+", str(s)) if g.strip()]


def main():
    print("=" * 70)
    print("SCI-Centric Counterfactual Analysis (Proxy for Tier C2)")
    print("=" * 70)

    summary = {"planners": {}}
    for planner in ["cnn", "dino", "tf"]:
        cf_path = BASE / f"failure_basin_{planner}" / "counterfactual_per_sample.csv"
        if not cf_path.exists():
            print(f"  SKIP: {cf_path} not found")
            continue

        print(f"\n--- {planner.upper()} ---")
        df = pd.read_csv(cf_path)
        print(f"  Loaded {len(df)} samples")

        # Categorize per-sample top1 gene
        df["sci_cat_top1"] = df["top1_gene"].apply(categorize_top1)
        # If top3 exists, derive majority category
        if "top3_genes" in df.columns:
            df["sci_cat_top3"] = df["top3_genes"].apply(
                lambda s: max(
                    [categorize_top1(g) for g in parse_top_genes(s)] + ["None"],
                    key=lambda c: sum(1 for g in parse_top_genes(s) if GENE_TO_CAT.get(g) == c)
                ) if parse_top_genes(s) else "None"
            )

        # Bucket by top1 SCI category
        cat_counts = df["sci_cat_top1"].value_counts().to_dict()
        print(f"  Top1 SCI category distribution: {cat_counts}")

        # Compute flip rate per category
        # At K=10 (already in df as cf_prob_topk at K=10)
        # The 'flipped_topk' column is the K-specific flip indicator
        # But the per-sample CSV has multiple K rows; we need to filter
        # For per-sample CSV structure: k=10 rows have cf_prob_topk and flipped_topk
        # Check the K=10 only
        if "k" in df.columns:
            df_k10 = df[df["k"] == 10].copy()
        else:
            # If not, use all rows but they're all K=10 already
            df_k10 = df.copy()

        if len(df_k10) == 0:
            print(f"  WARNING: no K=10 rows found, using all")
            df_k10 = df.copy()

        print(f"  K=10 rows: {len(df_k10)}")
        results_by_cat = {}
        for cat in sorted(df_k10["sci_cat_top1"].unique()):
            sub = df_k10[df_k10["sci_cat_top1"] == cat]
            n = len(sub)
            if n == 0:
                continue
            flip_top = sub["flipped_topk"].mean() if "flipped_topk" in sub.columns else None
            flip_rand = sub["flipped_rand"].mean() if "flipped_rand" in sub.columns else None
            results_by_cat[cat] = {
                "n": int(n),
                "flip_topk": float(flip_top) if flip_top is not None else None,
                "flip_rand": float(flip_rand) if flip_rand is not None else None,
                "lift": (float(flip_top) - float(flip_rand)) if (flip_top is not None and flip_rand is not None) else None,
            }
            print(f"    {cat:20s} n={n:3d}  flip_topk={flip_top if flip_top is None else f'{flip_top:.3f}'}  "
                  f"flip_rand={flip_rand if flip_rand is None else f'{flip_rand:.3f}'}  "
                  f"lift={results_by_cat[cat]['lift']}")

        summary["planners"][planner] = {
            "n_total": int(len(df)),
            "n_k10": int(len(df_k10)),
            "cat_distribution": cat_counts,
            "flip_by_sci_category": results_by_cat,
        }

    # Cross-planner summary
    print("\n" + "=" * 70)
    print("Cross-Planner SCI Comparison")
    print("=" * 70)
    cross_table = []
    cats = set()
    for p, info in summary["planners"].items():
        cats.update(info["flip_by_sci_category"].keys())
    cats = sorted(cats)
    for cat in cats:
        row = {"SCI_Category": cat}
        for p in ["cnn", "dino", "tf"]:
            v = summary["planners"].get(p, {}).get("flip_by_sci_category", {}).get(cat, {})
            row[f"{p}_flip"] = v.get("flip_topk")
            row[f"{p}_lift"] = v.get("lift")
        cross_table.append(row)
    cross_df = pd.DataFrame(cross_table)
    print(cross_df.to_string(index=False))

    # Save outputs
    with open(OUT / "sci_centric_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUT / "sci_centric_table.md", "w") as f:
        f.write("# SCI-Centric Counterfactual Analysis\n\n")
        f.write("Proxy for Tier C2: per-sample top1 SCI category → flip rate at K=10\n\n")
        f.write("## Hypothesis\n\n")
        f.write("If Road Structure SCI is causally critical, samples whose top1 driver is in\n")
        f.write("Road Structure category should have HIGH flip rate across all 3 planners,\n")
        f.write("and this rate should exceed other categories (Color, Frequency, Texture).\n\n")
        f.write("## Per-Planner Results\n\n")
        for p, info in summary["planners"].items():
            f.write(f"### {p.upper()}\n\n")
            f.write(f"- Samples: {info['n_total']} (K=10 subset: {info['n_k10']})\n")
            f.write(f"- Top1 SCI category distribution:\n\n")
            f.write("| Category | Count |\n|---|---|\n")
            for cat, cnt in sorted(info["cat_distribution"].items(), key=lambda x: -x[1]):
                f.write(f"| {cat} | {cnt} |\n")
            f.write("\nFlip rates at K=10:\n\n")
            f.write("| SCI Category | n | Top-K flip | Random flip | Lift |\n|---|---|---|---|---|\n")
            for cat, r in sorted(info["flip_by_sci_category"].items(), key=lambda x: -(x[1]["lift"] or 0)):
                f.write(f"| {cat} | {r['n']} | "
                        f"{r['flip_topk']:.3f} | "
                        f"{r['flip_rand']:.3f} | "
                        f"{r['lift']:+.3f} |\n")
            f.write("\n")
        f.write("## Cross-Planner Comparison\n\n")
        f.write(cross_df.to_markdown(index=False) if hasattr(cross_df, "to_markdown") else cross_df.to_string(index=False))
        f.write("\n\n## Interpretation\n\n")
        # Road Structure verdict
        rs_flips = []
        for p in ["cnn", "dino", "tf"]:
            v = summary["planners"].get(p, {}).get("flip_by_sci_category", {}).get("Road_Structure", {})
            if v.get("flip_topk") is not None:
                rs_flips.append((p, v["flip_topk"], v["lift"]))
        if rs_flips:
            f.write("### Road Structure SCI Across Planners\n\n")
            f.write("| Planner | Flip@K=10 | Lift over random |\n|---|---|---|\n")
            for p, f1, lift in rs_flips:
                f.write(f"| {p.upper()} | {f1:.3f} | {lift:+.3f} |\n")
            f.write("\n")
            avg_flip = np.mean([f1 for _, f1, _ in rs_flips])
            avg_lift = np.mean([lift for _, _, lift in rs_flips])
            f.write(f"- **Mean flip@K=10 for Road Structure**: {avg_flip:.3f}\n")
            f.write(f"- **Mean lift over random**: {avg_lift:+.3f}\n\n")
            if avg_flip > 0.6 and avg_lift > 0.4:
                f.write("> **Verdict**: Road Structure SCI shows strong cross-planner counterfactual effect. ")
                f.write("This is consistent with H3 (SCI Causality) at the XGBoost layer, ")
                f.write("and supports the SCI Hypothesis: Road Structure is the strongest cross-planner shared failure driver.\n")
            elif avg_flip > 0.4:
                f.write("> **Verdict**: Road Structure SCI shows moderate cross-planner counterfactual effect.\n")
            else:
                f.write("> **Verdict**: Road Structure SCI shows weak cross-planner counterfactual effect.\n")

    print(f"\nOutputs:")
    print(f"  - {OUT / 'sci_centric_summary.json'}")
    print(f"  - {OUT / 'sci_centric_table.md'}")


if __name__ == "__main__":
    main()
