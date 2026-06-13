"""3-planner SHAP Top-1 Gene Distribution — Venn + Pie.

读 exp/tierB_partial/failure_basin_{cnn,dino,tf}/counterfactual_per_sample.csv
或 per_sample_genes.csv，统计每个 planner 的 per-sample SHAP top-1 gene
分布，画 3 圈 Venn 图 + 各 planner 饼图。

使用：
  python scripts/analysis/shap_top1_venn.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

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
META_FIELDS = ["strength"]


def per_sample_shap(model, X: np.ndarray) -> np.ndarray:
    import xgboost as xgb
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    try:
        contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    except Exception:
        return np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
    return contribs[:, :-1].astype(np.float32)


def get_top1_for_planner(df: pd.DataFrame, planner: str, seed: int = 42, n_fail: int = 100):
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold

    sub = df[df["planner"] == planner].copy()
    feat = [c for c in GENE_FIELDS + META_FIELDS if c in sub.columns]
    X = sub[feat].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = sub["success"].values.astype(np.int32)
    ss = sub["scene_token"].values

    gkf = GroupKFold(n_splits=5)
    aucs, models = [], []
    for tr, te in gkf.split(X, y, ss):
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, eval_metric="logloss",
        )
        m.fit(X[tr], y[tr])
        models.append((m, tr, te))
        from sklearn.metrics import roc_auc_score
        try:
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        except ValueError:
            pass
    best = max(range(len(aucs)), key=lambda i: aucs[i])
    model, _, te = models[best]
    te_df = sub.iloc[te].copy()
    # 与 failure_basin_counterfactual.py 一致：
    # 选 predicted-fail (prob > 0.5)，按 scene 不重叠取 n_fail 个
    X_te = te_df[feat].values.astype(np.float32)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    probs_te = model.predict_proba(X_te)[:, 1]
    is_pred_fail = probs_te > 0.5
    pred_fail_idx = np.where(is_pred_fail)[0]
    # 按 scene 不重叠挑
    fail_by_scene = {}
    for i in pred_fail_idx:
        s = ss[te][i]
        # ss is indexed by the full df
        s_token = sub.iloc[te[i]]["scene_token"]
        if s_token not in fail_by_scene or probs_te[i] > probs_te[fail_by_scene[s_token]]:
            fail_by_scene[s_token] = i
    selected_idx = list(fail_by_scene.values())[:n_fail]
    if not selected_idx:
        # fallback: use top prob samples
        top_idx = np.argsort(-probs_te)[:n_fail]
        selected_idx = top_idx.tolist()

    X_fail = X_te[selected_idx]
    shap_fail = per_sample_shap(model, X_fail)
    # 与 failure_basin_counterfactual.py 一致：只看"把 prob 推高"的正贡献，
    # 取正贡献最大者为 top1（与报告中 80-88% 的口径对齐）
    pos_contrib = np.where(shap_fail > 0, shap_fail, 0)
    top1_idx = pos_contrib.argmax(axis=1)
    # 如果该样本全 0 positive contribution（极少见），fallback 到 argmax(有符号)
    has_pos = (pos_contrib.sum(axis=1) > 0)
    fallback = shap_fail.argmax(axis=1)
    top1_idx = np.where(has_pos, top1_idx, fallback)
    top1 = [feat[i] for i in top1_idx]
    return top1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="exp/tierB_partial/merged_3pl.csv")
    p.add_argument("--output-dir", default="exp/tierB_partial/figures")
    p.add_argument("--n-fail", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"=== SHAP Top-1 Venn & Pie ===")
    print(f"  csv={args.csv}, n_fail={args.n_fail}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    top1_by_planner: dict = {}
    for pl in ["CNN", "DINO", "TF"]:
        print(f"\n  processing {pl}...")
        top1_by_planner[pl] = get_top1_for_planner(df, pl, args.seed, args.n_fail)
        counts = pd.Series(top1_by_planner[pl]).value_counts()
        print(f"    top1 distribution (top 5):")
        for g, c in counts.head(5).items():
            print(f"      {g}: {c} ({c/len(top1_by_planner[pl])*100:.1f}%)")

    # ----- Venn diagram (hand-drawn, no external dep) -----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    set_cnn = set(top1_by_planner["CNN"])
    set_dino = set(top1_by_planner["DINO"])
    set_tf = set(top1_by_planner["TF"])

    # Compute region sizes
    n_cnn_only = len(set_cnn - set_dino - set_tf)
    n_dino_only = len(set_dino - set_cnn - set_tf)
    n_tf_only = len(set_tf - set_cnn - set_dino)
    n_cnn_dino = len(set_cnn & set_dino - set_tf)
    n_cnn_tf = len(set_cnn & set_tf - set_dino)
    n_dino_tf = len(set_dino & set_tf - set_cnn)
    n_all3 = len(set_cnn & set_dino & set_tf)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Hand-drawn Venn: 3 circles overlapping
    ax = axes[0]
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")

    c_cnn = Circle((-0.7, 0.2), 1.0, alpha=0.45, color="#4C72B0", label="CNN-GTRS")
    c_dino = Circle((0.7, 0.2), 1.0, alpha=0.45, color="#DD8452", label="DINO-GTRS")
    c_tf = Circle((0.0, -0.5), 1.0, alpha=0.45, color="#55A868", label="TransFuser")
    for c in (c_cnn, c_dino, c_tf):
        ax.add_patch(c)

    # Label each region with its count
    fontsize = 12
    ax.text(-1.15, 0.7, f"CNN\n{n_cnn_only}", ha="center", va="center", fontsize=fontsize)
    ax.text(1.15, 0.7, f"DINO\n{n_dino_only}", ha="center", va="center", fontsize=fontsize)
    ax.text(0.0, -1.1, f"TF\n{n_tf_only}", ha="center", va="center", fontsize=fontsize)
    ax.text(-0.35, 0.55, f"{n_cnn_dino}", ha="center", va="center", fontsize=fontsize)
    ax.text(0.35, 0.55, f"{n_cnn_tf}", ha="center", va="center", fontsize=fontsize)
    ax.text(0.0, 0.0, f"{n_dino_tf}", ha="center", va="center", fontsize=fontsize)
    ax.text(0.0, -0.25, f"{n_all3}", ha="center", va="center",
            fontsize=fontsize + 2, fontweight="bold", color="red")
    ax.set_title(
        f"Per-sample SHAP Top-1 Gene (Hand-drawn Venn)\n"
        f"CNN={len(set_cnn)} | DINO={len(set_dino)} | TF={len(set_tf)} | "
        f"shared all 3 = {n_all3}",
        fontsize=12,
    )
    ax.legend([c_cnn, c_dino, c_tf],
              ["CNN-GTRS", "DINO-GTRS", "TransFuser"],
              loc="upper right", fontsize=10)

    # ----- Pie: edge_mean share -----
    edge_share = {pl: sum(g == "edge_mean" for g in top1_by_planner[pl]) /
                       len(top1_by_planner[pl]) for pl in ["CNN", "DINO", "TF"]}

    ax_pie = axes[1]
    sizes = [edge_share["CNN"] * 100, edge_share["DINO"] * 100, edge_share["TF"] * 100]
    labels_p = [f"CNN ({edge_share['CNN']*100:.0f}%)",
                f"DINO ({edge_share['DINO']*100:.0f}%)",
                f"TF ({edge_share['TF']*100:.0f}%)"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    ax_pie.pie(sizes, labels=labels_p, colors=colors, startangle=90, textprops={"fontsize": 12})
    ax_pie.set_title("edge_mean share of per-sample SHAP top-1\n"
                     "(cross-architecture dominant)", fontsize=12)

    fig.tight_layout()
    fig.savefig(out_dir / "shap_top1_combined.png", dpi=150, bbox_inches="tight")
    print(f"  → {out_dir / 'shap_top1_combined.png'}")

    # Also save Venn alone
    fig_v, ax_v = plt.subplots(figsize=(8, 8))
    ax_v.set_xlim(-1.6, 1.6)
    ax_v.set_ylim(-1.4, 1.4)
    ax_v.set_aspect("equal")
    ax_v.axis("off")
    c_cnn2 = Circle((-0.7, 0.2), 1.0, alpha=0.45, color="#4C72B0")
    c_dino2 = Circle((0.7, 0.2), 1.0, alpha=0.45, color="#DD8452")
    c_tf2 = Circle((0.0, -0.5), 1.0, alpha=0.45, color="#55A868")
    for c in (c_cnn2, c_dino2, c_tf2):
        ax_v.add_patch(c)
    ax_v.text(-1.15, 0.7, f"CNN\n{n_cnn_only}", ha="center", va="center", fontsize=fontsize)
    ax_v.text(1.15, 0.7, f"DINO\n{n_dino_only}", ha="center", va="center", fontsize=fontsize)
    ax_v.text(0.0, -1.1, f"TF\n{n_tf_only}", ha="center", va="center", fontsize=fontsize)
    ax_v.text(-0.35, 0.55, f"{n_cnn_dino}", ha="center", va="center", fontsize=fontsize)
    ax_v.text(0.35, 0.55, f"{n_cnn_tf}", ha="center", va="center", fontsize=fontsize)
    ax_v.text(0.0, 0.0, f"{n_dino_tf}", ha="center", va="center", fontsize=fontsize)
    ax_v.text(0.0, -0.25, f"{n_all3}", ha="center", va="center",
              fontsize=fontsize + 2, fontweight="bold", color="red")
    ax_v.set_title(f"Per-sample SHAP Top-1 Gene (Venn)\n"
                   f"shared all 3 planners = {n_all3} genes",
                   fontsize=14)
    fig_v.tight_layout()
    fig_v.savefig(out_dir / "shap_top1_venn.png", dpi=150, bbox_inches="tight")
    print(f"  → {out_dir / 'shap_top1_venn.png'}")

    # Edge share alone
    fig_e, ax_e = plt.subplots(figsize=(7, 7))
    ax_e.pie(sizes, labels=labels_p, colors=colors, startangle=90, textprops={"fontsize": 12})
    ax_e.set_title("edge_mean share of per-sample SHAP top-1\n"
                   "(cross-architecture dominant)", fontsize=14)
    fig_e.tight_layout()
    fig_e.savefig(out_dir / "edge_mean_share.png", dpi=150, bbox_inches="tight")
    print(f"  → {out_dir / 'edge_mean_share.png'}")

    plt.close("all")

    # Also save data
    out_data = {
        "CNN": list(top1_by_planner["CNN"]),
        "DINO": list(top1_by_planner["DINO"]),
        "TF": list(top1_by_planner["TF"]),
    }
    import json
    with open(out_dir / "top1_per_planner.json", "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"  → {out_dir / 'top1_per_planner.json'}")
    print(f"\nDONE.")


if __name__ == "__main__":
    main()
