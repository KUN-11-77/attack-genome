"""Failure Basin 几何分析 — 跨 planner 共享失效区域。

把每个 (scene, attack, strength) 样本投影到 gene 空间，标 success/fail，
用 DBSCAN 聚类出每个 planner 的失效 basin，再求 planner 间的交集
= Cross-Representation Failure Basin（CRFB）。

这是 [[attack-genome-law-hunting]] 中 Failure Basin 的实现。

输入: per_sample_genes.csv (long format, 含 33 gene + planner + success)
输出:
    - per_planner_basin.json (DBSCAN 簇中心 + 半径)
    - common_basin.csv (跨 planner 共享簇)
    - basin_3d.png (3D scatter 可视化)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def per_planner_basin(
    df: pd.DataFrame, planner: str, eps: float = 0.5, min_samples: int = 5,
) -> Dict:
    """对单个 planner 在 gene 空间上聚类其失败样本。"""
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    sub = df[df["planner"] == planner].copy()
    if len(sub) == 0:
        return {"planner": planner, "clusters": [], "n_failures": 0}

    use_cols = [c for c in GENE_FIELDS if c in sub.columns]
    X = sub[use_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    is_fail = sub["success"].values.astype(bool)
    X_fail = X_scaled[is_fail]
    n_fail = int(is_fail.sum())
    if n_fail < min_samples:
        return {"planner": planner, "clusters": [], "n_failures": n_fail}

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_fail)
    labels = db.labels_

    clusters = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue  # 噪声点
        mask = labels == cid
        center = X_fail[mask].mean(axis=0)
        radius = float(np.linalg.norm(X_fail[mask] - center, axis=1).mean())
        clusters.append({
            "cluster_id": int(cid),
            "center_unscaled": scaler.inverse_transform([center])[0].tolist(),
            "radius_scaled": radius,
            "n_samples": int(mask.sum()),
            "attack_dist": sub[is_fail].loc[mask, "attack"].value_counts().to_dict(),
        })
    return {"planner": planner, "clusters": clusters, "n_failures": n_fail}


def common_basin(
    per_planner: Dict[str, Dict], overlap_iou: float = 0.3,
) -> List[Dict]:
    """求跨 planner 共享的 basin 簇。"""
    from itertools import combinations

    planners = list(per_planner.keys())
    if len(planners) < 2:
        return []

    common: List[Dict] = []
    for a, b in combinations(planners, 2):
        A_centers = np.array([c["center_unscaled"] for c in per_planner[a]["clusters"]])
        B_centers = np.array([c["center_unscaled"] for c in per_planner[b]["clusters"]])
        if len(A_centers) == 0 or len(B_centers) == 0:
            continue
        # 配对：归一化后欧氏距离 < 0.5 视为共享
        A_norm = A_centers / (np.linalg.norm(A_centers, axis=1, keepdims=True) + 1e-8)
        B_norm = B_centers / (np.linalg.norm(B_centers, axis=1, keepdims=True) + 1e-8)
        sim = A_norm @ B_norm.T
        for i in range(len(A_centers)):
            for j in range(len(B_centers)):
                if sim[i, j] > 0.9:  # cos sim > 0.9
                    common.append({
                        "planner_pair": (a, b),
                        "center_a": A_centers[i].tolist(),
                        "center_b": B_centers[j].tolist(),
                        "cosine_sim": float(sim[i, j]),
                        "n_a": per_planner[a]["clusters"][i]["n_samples"],
                        "n_b": per_planner[b]["clusters"][j]["n_samples"],
                    })
    return common


def plot_basin_3d(df: pd.DataFrame, out_path: Path, gene_x: str = "edge_density",
                  gene_y: str = "lane_line_density", gene_z: str = "rms_contrast") -> None:
    """3D scatter: x=edge_density, y=lane_line_density, z=rms_contrast
    color=attack, marker=planner, alpha=success."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    planners = sorted(df["planner"].unique())
    markers = {"CNN": "o", "DINO": "s", "TF": "^", "CLIP": "D"}
    colors = {"Rain": "tab:blue", "MotionBlur": "tab:orange", "Snow": "tab:green",
              "DigitalNoise": "tab:red", "Dusk": "tab:purple", "Dawn": "tab:brown",
              "LightDust": "tab:pink", "DappledLight": "tab:gray",
              "VintageStyle": "tab:olive", "CarlaStyle": "tab:cyan"}

    for p in planners:
        sub = df[df["planner"] == p]
        for atk, color in colors.items():
            ss = sub[sub["attack"] == atk]
            if len(ss) == 0:
                continue
            # pandas 2.x: 转 numpy 后再 plot，避免 Multi-dim indexing
            x_vals = ss[gene_x].to_numpy()
            y_vals = ss[gene_y].to_numpy()
            z_vals = ss[gene_z].to_numpy()
            ax.scatter(
                x_vals, y_vals, z_vals,
                c=color, marker=markers.get(p, "o"),
                s=12, alpha=0.4,
                label=f"{p}-{atk}" if (p == planners[0]) else None,
            )
    ax.set_xlabel(gene_x); ax.set_ylabel(gene_y); ax.set_zlabel(gene_z)
    ax.set_title("Failure Basin — Gene Space (red=fail, alpha∝strength)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--min-samples", type=int, default=5)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Failure Basin Discovery ===")
    print(f"  CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  loaded {len(df)} rows, planners={sorted(df['planner'].unique())}")

    # Per-planner basin
    per_planner: Dict[str, Dict] = {}
    for planner in sorted(df["planner"].unique()):
        basin = per_planner_basin(df, planner, args.eps, args.min_samples)
        per_planner[planner] = basin
        print(f"  {planner}: {len(basin['clusters'])} clusters, n_failures={basin['n_failures']}")

    with open(out_dir / "per_planner_basin.json", "w") as f:
        json.dump(per_planner, f, indent=2)
    print(f"  saved per_planner_basin.json")

    # Common basin
    common = common_basin(per_planner)
    df_common = pd.DataFrame(common)
    if not df_common.empty:
        df_common.to_csv(out_dir / "common_basin.csv", index=False)
        print(f"  common basin: {len(df_common)} shared clusters → common_basin.csv")
    else:
        print(f"  no common basin found (cos_sim threshold=0.9)")

    # Plot
    if not args.no_plot:
        plot_basin_3d(df, out_dir / "basin_3d.png")

    print(f"\nDONE. outputs → {out_dir}")


if __name__ == "__main__":
    main()
