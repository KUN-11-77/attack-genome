"""核心规律发现：从 4-planner ASR 数据中找反直觉/定量/可预测的规律。

候选规律（按 [[attack-genome-law-hunting]] 用户猜想）:
    1. "结构信息守恒" — 攻击主要破坏 edge_density / lane_line_density，
       这两个 gene 的组合在所有 planner 上都能预测失败
    2. "颜色偏移不迁移" — Dusk/Dawn 只让 CNN 失败，DINO/TF/VLM 鲁棒
    3. "ASR 与 strength 单调" — 真实数据中 ASR(strength) 单调递增
    4. "跨 planner 共有失败子集" — 所有 planner 都失败的样本子集（Common Failure）
       的 gene 特征有什么共性？

输入: per_sample_genes.csv (含 success 列, planner 列, gene 字段)
输出:
    - pattern1_structure_conservation.csv
    - pattern2_color_robustness.csv
    - pattern3_asr_monotonicity.csv
    - pattern4_common_failure_genes.csv
    - patterns_summary.json (一键结论)

用法:
    python scripts/analysis/hunt_patterns.py \
        --csv outputs/ag_tier_b/merged/per_sample_genes.csv \
        --output-dir results/patterns
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


GENE_FIELDS = [
    "low_freq_ratio", "mid_freq_ratio", "high_freq_ratio", "spectral_centroid",
    "hue_mean", "hue_std", "sat_mean", "sat_std", "val_mean", "val_std", "colorfulness",
    "lbp_entropy", "glcm_contrast", "lbp_uniformity",
    "rms_contrast", "mean_luma", "std_luma", "dynamic_range", "mean_shift", "std_shift",
    "edge_mean", "edge_density", "road_luma_mean", "road_luma_std",
    "lane_line_count", "lane_line_density",
    "vehicle_loss", "person_loss", "detection_loss", "conf_loss", "vehicle_loss_ratio",
    "shadow_ratio", "highlight_ratio", "luma_entropy", "luma_skew", "luma_mean",
]

STRUCTURE_GENES = ["edge_density", "lane_line_density", "edge_mean", "lane_line_count"]
COLOR_GENES = ["hue_mean", "hue_std", "sat_mean", "sat_std", "colorfulness"]
FREQUENCY_GENES = ["low_freq_ratio", "mid_freq_ratio", "high_freq_ratio"]


def pattern1_structure_conservation(df: pd.DataFrame, out_dir: Path) -> Dict:
    """Pattern 1: 结构基因 (edge/lane) 在所有 planner 上是否都预测失败？

    假设: edge_density 越低，failure 率越高（在所有 planner 上）。
    """
    print("\n=== Pattern 1: Structure Gene Conservation ===")
    rows = []
    for p in df["planner"].unique():
        sub = df[df["planner"] == p]
        for g in STRUCTURE_GENES:
            if g not in sub.columns:
                continue
            # 把 gene 值分 5 个 bin，算每个 bin 的 success rate
            sub_b = sub.copy()
            sub_b[f"{g}_bin"] = pd.qcut(sub_b[g], q=5, duplicates="drop")
            bin_stats = sub_b.groupby(f"{g}_bin", observed=True)["success"].agg(["mean", "count"])
            # 单调性检验：Spearman
            bin_centers = [b.mid for b in bin_stats.index]
            rho, pval = stats.spearmanr(bin_centers, bin_stats["mean"].values)
            rows.append({
                "planner": p, "gene": g,
                "spearman_rho": rho, "spearman_p": pval,
                "bin_success_rates": bin_stats["mean"].round(3).tolist(),
            })
            print(f"  {p:>20s} | {g:>20s} | rho={rho:+.3f} p={pval:.3g}")
    df_p1 = pd.DataFrame(rows)
    df_p1.to_csv(out_dir / "pattern1_structure_conservation.csv", index=False)

    # 结论：跨 planner 一致的结构基因
    summary_gene = df_p1.groupby("gene")["spearman_rho"].agg(["mean", "std"]).sort_values("mean")
    print(f"\n  Cross-planner structure gene consensus (most negative = strongest predictor):")
    print(summary_gene.to_string())
    return {
        "summary": summary_gene.to_dict(),
        "rows": rows,
    }


def pattern2_color_robustness(df: pd.DataFrame, out_dir: Path) -> Dict:
    """Pattern 2: 颜色攻击 (Dusk/Dawn) 是否只在 CNN 上有效？

    假设: Dusk/Dawn 在 CNN 上 ASR=0.7，但在 DINO/TF/VLM 上 ASR<0.4
    """
    print("\n=== Pattern 2: Color Attack Robustness ===")
    color_atks = ["Dusk", "Dawn", "VintageStyle", "CarlaStyle"]
    struct_atks = ["Rain", "MotionBlur", "DigitalNoise"]
    rows = []
    for p in df["planner"].unique():
        for atk_type, atks in [("color", color_atks), ("structure", struct_atks)]:
            sub = df[(df["planner"] == p) & (df["attack"].isin(atks))]
            if len(sub) == 0:
                continue
            mean_asr = sub["success"].mean()
            rows.append({"planner": p, "attack_type": atk_type, "mean_success": mean_asr,
                         "n_samples": len(sub)})
            print(f"  {p:>20s} | {atk_type:>10s} | ASR={1-mean_asr:.3f}")
    df_p2 = pd.DataFrame(rows)
    df_p2.to_csv(out_dir / "pattern2_color_robustness.csv", index=False)

    # 结论：CNN vs others 在 color 上的差异
    pivot = df_p2.pivot(index="planner", columns="attack_type", values="mean_success")
    if "color" in pivot.columns and "structure" in pivot.columns:
        pivot["gap"] = pivot["structure"] - pivot["color"]
        print(f"\n  Planner's color vs structure gap (positive = color less harmful):")
        print(pivot[["color", "structure", "gap"]].round(3).to_string())
    return {"pivot": pivot.to_dict() if not pivot.empty else {}}


def pattern3_asr_monotonicity(df: pd.DataFrame, out_dir: Path) -> Dict:
    """Pattern 3: ASR(strength) 是否单调递增？

    假设: 真实数据中所有 attack 的 ASR 都随 strength 单调递增。
    """
    print("\n=== Pattern 3: ASR Monotonicity ===")
    rows = []
    for p in df["planner"].unique():
        for atk in df["attack"].unique():
            sub = df[(df["planner"] == p) & (df["attack"] == atk)].sort_values("strength")
            if len(sub) < 3:
                continue
            agg = sub.groupby("strength")["success"].mean()
            # 检验单调性：相邻强度间 success 不下降
            diffs = agg.diff().dropna()
            n_violations = int((diffs < 0).sum())
            n_total = len(diffs)
            monotone_rate = 1.0 - n_violations / n_total if n_total > 0 else 0.0
            rows.append({
                "planner": p, "attack": atk,
                "n_violations": n_violations, "n_total": n_total,
                "monotone_rate": monotone_rate,
            })
    df_p3 = pd.DataFrame(rows)
    df_p3.to_csv(out_dir / "pattern3_asr_monotonicity.csv", index=False)
    summary = df_p3.groupby("planner")["monotone_rate"].mean()
    print(f"\n  Monotonicity rate per planner (1.0 = perfectly monotonic):")
    print(summary.round(3).to_string())
    return {"summary": summary.to_dict()}


def pattern4_common_failure_genes(df: pd.DataFrame, out_dir: Path) -> Dict:
    """Pattern 4: 跨 planner 都失败的样本（共失效），gene 空间有聚集性吗？

    假设: Common failure 样本在 edge_density, lane_line_density, detection_loss 上
    显著高于 partial failure。
    """
    print("\n=== Pattern 4: Common Failure Gene Signature ===")
    # 找每个 (scene, attack, strength) 的 failure 集合
    pivot = df.pivot_table(
        index=["scene_token", "attack", "strength"],
        columns="planner", values="success", aggfunc="first",
    ).reset_index()
    # 4 个 planner 都 fail = success 全为 0
    success_cols = [c for c in pivot.columns if c in df["planner"].unique()]
    pivot["n_fail"] = (pivot[success_cols] == 0).sum(axis=1)
    pivot["n_planners"] = len(success_cols)

    # 0 失败 vs 4 失败
    common = pivot[pivot["n_fail"] == pivot["n_planners"]]["scene_token"].astype(str) + "_" + \
             pivot[pivot["n_fail"] == pivot["n_planners"]]["attack"] + "_" + \
             pivot[pivot["n_fail"] == pivot["n_planners"]]["strength"].astype(str)
    robust = pivot[pivot["n_fail"] == 0]["scene_token"].astype(str) + "_" + \
             pivot[pivot["n_fail"] == 0]["attack"] + "_" + \
             pivot[pivot["n_fail"] == 0]["strength"].astype(str)
    print(f"  common-failure (all 4 fail): {len(common)} samples")
    print(f"  robust (all 4 succeed):     {len(robust)} samples")

    # 取 gene 值
    df["key"] = df["scene_token"].astype(str) + "_" + df["attack"] + "_" + df["strength"].astype(str)
    use_cols = [c for c in GENE_FIELDS if c in df.columns]
    common_g = df[df["key"].isin(common)].groupby("key")[use_cols].first()
    robust_g = df[df["key"].isin(robust)].groupby("key")[use_cols].first()

    # t-test 每个 gene
    rows = []
    for g in use_cols:
        if g in common_g.columns and g in robust_g.columns:
            t, p = stats.ttest_ind(common_g[g], robust_g[g], equal_var=False)
            rows.append({
                "gene": g,
                "common_mean": float(common_g[g].mean()),
                "robust_mean": float(robust_g[g].mean()),
                "t_stat": t, "p_value": p,
                "abs_diff": abs(common_g[g].mean() - robust_g[g].mean()),
            })
    df_p4 = pd.DataFrame(rows).sort_values("p_value")
    df_p4.to_csv(out_dir / "pattern4_common_failure_genes.csv", index=False)
    print(f"\n  Top 10 most discriminative genes (common vs robust):")
    print(df_p4.head(10)[["gene", "common_mean", "robust_mean", "t_stat", "p_value"]].to_string(index=False))
    return {"rows": rows[:10]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Hunt Patterns in 4-planner Attack Genome ===")
    print(f"  CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  {len(df)} rows, {df['planner'].nunique()} planners, {df['attack'].nunique()} attacks, {df['scene_token'].nunique()} scenes")

    results = {}
    results["p1_structure"] = pattern1_structure_conservation(df, out_dir)
    results["p2_color"] = pattern2_color_robustness(df, out_dir)
    results["p3_monotonicity"] = pattern3_asr_monotonicity(df, out_dir)
    results["p4_common_failure"] = pattern4_common_failure_genes(df, out_dir)

    with open(out_dir / "patterns_summary.json", "w") as f:
        json.dump({k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv) for kk, vv in v.items()}
                   for k, v in results.items()}, f, indent=2, default=str)
    print(f"\nDONE. outputs → {out_dir}")


if __name__ == "__main__":
    main()
