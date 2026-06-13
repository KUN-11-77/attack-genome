"""Failure Factor Discovery — Gene → Planner 失效分析。

读取 ``per_sample_genes.csv``（长格式：每行一个 sample × planner），
训练 Random Forest / XGBoost 把 gene 字段映射到 planner 失效概率。

输出：
- 各 planner 的 gene feature importance
- 预测能力（5-fold CV AUC）
- 跨 planner 重要性对比表 → 找 "跨架构不变量"
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
ATTRIBUTE_FIELDS = ["curve", "high_traffic", "low_light", "night", "road_occlusion", "driving_command"]
META_FIELDS = ["strength"]


def load_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"  loaded {len(df)} rows from {csv_path}")
    return df


def train_per_planner(
    df: pd.DataFrame, planner_name: str, n_estimators: int = 200, n_folds: int = 5,
) -> dict:
    """对单个 planner 训练 RF 并评估 feature importance + CV AUC。"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    sub = df[df["planner"] == planner_name].copy()
    X = sub[GENE_FIELDS + META_FIELDS].values.astype(np.float32)
    y = sub["success"].values.astype(np.int32)
    print(f"  {planner_name}: {len(sub)} samples, {y.sum()} failures ({y.mean():.2%})")

    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=None, n_jobs=-1, random_state=42,
    )
    rf.fit(X, y)
    importance = dict(zip(GENE_FIELDS + META_FIELDS, rf.feature_importances_))

    # 5-fold CV AUC
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    auc = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return {
        "planner": planner_name,
        "n_samples": int(len(sub)),
        "failure_rate": float(y.mean()),
        "cv_auc_mean": float(auc.mean()),
        "cv_auc_std": float(auc.std()),
        "feature_importance": importance,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="path to per_sample_genes.csv")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Failure Factor Discovery ===")
    print(f"  CSV: {args.csv}")
    print(f"  OUT: {out_dir}")

    df = load_csv(args.csv)
    print(f"\n=== Train per-planner RF ===")
    results = []
    for planner in sorted(df["planner"].unique()):
        r = train_per_planner(df, planner, args.n_estimators, args.n_folds)
        results.append(r)
        print(f"  {planner}: AUC = {r['cv_auc_mean']:.3f} ± {r['cv_auc_std']:.3f}")

    # 保存
    with open(out_dir / "per_planner_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved per_planner_results.json")

    # 生成 cross-planner importance 表
    print(f"\n=== Cross-planner importance heatmap data ===")
    rows = []
    for r in results:
        for fname, imp in r["feature_importance"].items():
            rows.append({"planner": r["planner"], "feature": fname, "importance": imp})
    imp_df = pd.DataFrame(rows)
    imp_pivot = imp_df.pivot(index="feature", columns="planner", values="importance").fillna(0)
    imp_pivot.to_csv(out_dir / "feature_importance.csv")
    imp_pivot.to_csv(out_dir / "feature_importance.csv", index=True)
    print(f"  saved feature_importance.csv (shape {imp_pivot.shape})")

    # 跨 planner 重要性比较：找 "任何 planner 都重视" 的 gene（top-K by min importance）
    print(f"\n=== Top-10 'Cross-Representation Important' features (min imp across planners) ===")
    imp_pivot["min_imp"] = imp_pivot.min(axis=1)
    imp_pivot["mean_imp"] = imp_pivot.mean(axis=1)
    top10 = imp_pivot.sort_values("min_imp", ascending=False).head(10)
    print(top10[["min_imp", "mean_imp"]])
    top10.to_csv(out_dir / "top10_cross_rep_important.csv")

    # 跨 planner 重要性比较：找 "CNN 重视但 DINO 不重视" 的 gene（差异最大）
    planners = sorted(df["planner"].unique())
    if len(planners) >= 2:
        print(f"\n=== Feature importance gap (top-10 by |diff|) ===")
        imp_pivot["max_diff"] = imp_pivot[planners].max(axis=1) - imp_pivot[planners].min(axis=1)
        top_diff = imp_pivot.sort_values("max_diff", ascending=False).head(10)
        print(top_diff[["max_diff"] + planners])
        top_diff.to_csv(out_dir / "top10_planner_gap.csv")

    print(f"\nDONE. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
