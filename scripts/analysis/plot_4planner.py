"""4-planner ASR 对比 + 4 层表征演化曲线 + Gene→ASR 热力图。

输入: per_sample_genes.csv (long format: scene × attack × strength × planner × gene × success)
输出:
    - fig1_4planner_asr.png: 4 个 planner 的 ASR-vs-strength 曲线（10 attack × 4 planner）
    - fig2_evolution_curve.png: 4 层表征演化 (s_c + k logistic 拟合)
    - fig3_gene_heatmap.png: Gene → ASR 相关热力图
    - fig4_vulnerability_atlas.png: Vulnerability Atlas 可视化
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


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
COLORS = {"CNN-GTRS": "#d62728", "DINO-GTRS": "#1f77b4",
          "TransFuser": "#2ca02c", "DiffusionDrive-DINO": "#ff7f0e",
          "ReCogDrive-VLM": "#9467bd"}
PLANNER_ORDER = ["CNN-GTRS", "DINO-GTRS", "TransFuser", "DiffusionDrive-DINO", "ReCogDrive-VLM"]


def logistic(s, k, s_c):
    return 1.0 / (1.0 + np.exp(-k * (s - s_c)))


def fig1_asr_curves(df: pd.DataFrame, out_path: Path) -> None:
    """4 planner × 10 attack 的 ASR-vs-strength 曲线。"""
    attacks = sorted(df["attack"].unique())
    planners = [p for p in PLANNER_ORDER if p in df["planner"].unique()]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    for i, atk in enumerate(attacks[:10]):
        ax = axes[i // 5][i % 5]
        for p in planners:
            sub = df[(df["attack"] == atk) & (df["planner"] == p)]
            if len(sub) == 0:
                continue
            agg = sub.groupby("strength")["success"].mean().sort_index()
            # pandas 2.x: 显式转 numpy
            x_vals = np.asarray(agg.index.tolist(), dtype=np.float32)
            y_vals = np.asarray(agg.values, dtype=np.float32)
            ax.plot(x_vals, y_vals, "-o", color=COLORS.get(p, "k"),
                    label=p, lw=2, ms=5)
        ax.set_title(atk, fontsize=11)
        ax.set_xlabel("Strength")
        if i % 5 == 0:
            ax.set_ylabel("ASR")
        ax.axhline(0.5, color="gray", ls=":", alpha=0.4)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.suptitle("Attack Genome: 4-planner ASR curves (Tier B)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def fig2_evolution_curve(df: pd.DataFrame, out_path: Path) -> None:
    """4 层表征演化曲线: s_c (相变点) + k (logistic 陡峭度)。"""
    planners = [p for p in PLANNER_ORDER if p in df["planner"].unique()]
    rows = []
    for p in planners:
        for atk in sorted(df["attack"].unique()):
            sub = df[(df["planner"] == p) & (df["attack"] == atk)]
            if len(sub) < 3:
                continue
            agg = sub.groupby("strength")["success"].mean().sort_index()
            # pandas 2.x: 显式转 numpy
            strengths = np.asarray(agg.index.tolist(), dtype=np.float32)
            asrs = np.asarray(agg.values, dtype=np.float32)
            try:
                if not np.all(asrs == asrs[0]) and len(strengths) >= 3:
                    popt, _ = curve_fit(logistic, strengths, asrs, p0=[5, 0.5], maxfev=5000)
                    rows.append({"planner": p, "attack": atk, "s_c": popt[1], "k": popt[0]})
            except Exception:
                pass
    if not rows:
        print("  no data for evolution curve")
        return
    evo_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # s_c bar
    s_c_pivot = evo_df.pivot(index="attack", columns="planner", values="s_c")
    s_c_pivot[planners].plot(kind="bar", ax=axes[0], color=[COLORS.get(p, "k") for p in planners])
    axes[0].set_title("Phase transition s_c (higher = more robust)")
    axes[0].set_ylabel("s_c")
    axes[0].axhline(0.5, color="gray", ls="--", alpha=0.5)
    # k bar
    k_pivot = evo_df.pivot(index="attack", columns="planner", values="k")
    k_pivot[planners].plot(kind="bar", ax=axes[1], color=[COLORS.get(p, "k") for p in planners])
    axes[1].set_title("Logistic steepness k (higher = sharper transition)")
    axes[1].set_ylabel("k")
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Cross-Representation Evolution: 4 layers", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")
    return evo_df


def fig3_gene_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Gene → ASR 相关热力图（每个 planner 一列）。"""
    planners = [p for p in PLANNER_ORDER if p in df["planner"].unique()]
    corr = pd.DataFrame(index=GENE_FIELDS, columns=planners, dtype=float)
    for p in planners:
        sub = df[df["planner"] == p]
        for g in GENE_FIELDS:
            if g in sub.columns:
                corr.loc[g, p] = sub[g].corr(sub["success"])
    fig, ax = plt.subplots(figsize=(8, 12))
    im = ax.imshow(corr.fillna(0).values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(planners)))
    ax.set_xticklabels(planners, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(GENE_FIELDS)))
    ax.set_yticklabels(GENE_FIELDS, fontsize=8)
    for i in range(len(GENE_FIELDS)):
        for j in range(len(planners)):
            v = corr.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Pearson correlation (gene vs success)")
    fig.suptitle("Gene → ASR correlation across 4 planners", fontsize=13, y=1.00)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def fig4_vulnerability_atlas(common_csv: Path, df: pd.DataFrame, out_path: Path) -> None:
    """Vulnerability Atlas: 跨 planner 共享失效区的 attribute 分布。"""
    if not common_csv.exists() or len(df) == 0:
        print(f"  no common basin data, skipping fig4")
        return
    common = pd.read_csv(common_csv)
    fig, ax = plt.subplots(figsize=(10, 6))
    for planner in df["planner"].unique():
        sub = df[df["planner"] == planner]
        for atk in sub["attack"].unique():
            ss = sub[sub["attack"] == atk]
            ax.scatter(
                ss.get("edge_density", np.zeros(len(ss))),
                ss.get("lane_line_density", np.zeros(len(ss))),
                c=ss["success"], cmap="RdYlGn_r",
                marker="o", s=15, alpha=0.3, vmin=0, vmax=1,
            )
    ax.set_xlabel("edge_density ↓ (structure lost)")
    ax.set_ylabel("lane_line_density ↓ (lane lost)")
    ax.set_title("Vulnerability Atlas: structure-loss gene space")
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 1)),
                 ax=ax, label="success=0 red, success=1 green")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--common-basin", default=None)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== 4-planner visualization ===")
    print(f"  CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  {len(df)} rows, planners={df['planner'].unique().tolist()}")
    print(f"  attacks={df['attack'].unique().tolist()}")
    print(f"  scenes={df['scene_token'].nunique()}")

    fig1_asr_curves(df, out / "fig1_4planner_asr.png")
    evo = fig2_evolution_curve(df, out / "fig2_evolution_curve.png")
    if evo is not None:
        evo.to_csv(out / "evolution_metrics.csv", index=False)
    fig3_gene_heatmap(df, out / "fig3_gene_heatmap.png")
    if args.common_basin:
        fig4_vulnerability_atlas(Path(args.common_basin), df, out / "fig4_vulnerability_atlas.png")
    print(f"DONE. → {out}")


if __name__ == "__main__":
    main()
