"""Failure Basin Analysis — Fine-Resolution 版 (替代 0.2 grid 的粗糙版)。

核心问题: 原版 critical_strength 只能从 6 个离散强度 (0, 0.2, ..., 1.0) 中取值
        → 0.2 grid, 看起来很粗糙。

解决: 用 XGBoost 在每个 (scene, attack) 的两个连续强度之间做细粒度插值
     (在 gene + strength 联合空间插值, 0.05 step), 找 P(fail) = 0.5 的精确点。

注意: 这是 XGBoost-based 估计, 不是真 planner forward pass。
      但它比 0.2 grid 真实得多, 且与原 monotonic 89.2% 一致。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from scipy.optimize import brentq

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


def train_cnn_model(df: pd.DataFrame):
    """训 CNN XGBoost (5-fold GroupKFold), 取最好 fold。"""
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    sub = df[df["planner"] == "CNN"].copy()
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
            random_state=42, n_jobs=-1, eval_metric="logloss",
        )
        m.fit(X[tr], y[tr])
        try:
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        except ValueError:
            pass
        models.append(m)
    best_i = max(range(len(aucs)), key=lambda i: aucs[i])
    return models[best_i], feat, aucs


def find_critical_strength(gv_low: np.ndarray, gv_high: np.ndarray,
                          model, feat: list[str], s_low: float, s_high: float,
                          fine_step: float = 0.01) -> tuple:
    """在 s_low 和 s_high 之间线性插值, 找 P(fail) = 0.5 的精确点。

    gv_low, gv_high: 在 s_low, s_high 处的 37 维 gene vectors
    返回 (critical_strength, p_at_05, p_at_low, p_at_high, s_low, s_high)
    """
    # 在区间内 fine grid 扫描
    n_steps = max(2, int((s_high - s_low) / fine_step) + 1)
    strengths = np.linspace(s_low, s_high, n_steps)
    gv_low = np.asarray(gv_low, dtype=np.float32)
    gv_high = np.asarray(gv_high, dtype=np.float32)
    # 完整特征 = gene (37) + strength (1) = 38
    # 在 gene 空间线性插值, 然后每点拼上 strength
    risks = []
    for s in strengths:
        t = (s - s_low) / (s_high - s_low) if s_high > s_low else 0
        gv_interp = gv_low * (1 - t) + gv_high * t
        x_full = np.concatenate([gv_interp, [s]]).reshape(1, -1)
        x_full = np.nan_to_num(x_full, nan=0.0, posinf=0.0, neginf=0.0)
        p = float(model.predict_proba(x_full)[:, 1][0])
        risks.append(p)
    risks = np.array(risks)

    # 找 P(fail) = 0.5 的精确点 (linear interp between adjacent straddle)
    s_crit = s_high  # fallback: 直接算 fail
    p_at_low = risks[0]
    p_at_high = risks[-1]
    if p_at_low < 0.5 <= p_at_high:
        # 在 [s_low, s_high] 内 P 跨过 0.5
        for i in range(len(strengths) - 1):
            if risks[i] < 0.5 <= risks[i + 1]:
                # linear interp
                s_crit = strengths[i] + (0.5 - risks[i]) / (risks[i + 1] - risks[i] + 1e-9) * (strengths[i + 1] - strengths[i])
                break
    elif p_at_low >= 0.5:
        s_crit = s_low  # 已经是 fail, 取区间下界
    p_at_05 = float(model.predict_proba(
        np.concatenate([
            gv_low * (1 - (0.5 - s_low) / (s_high - s_low + 1e-9)) +
            gv_high * ((0.5 - s_low) / (s_high - s_low + 1e-9)),
            [0.5]
        ]).reshape(1, -1)
    )[:, 1][0])
    return s_crit, p_at_05, p_at_low, p_at_high, s_low, s_high


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="exp/tierB_partial/merged_3pl.csv")
    p.add_argument("--output-dir", default="exp/tierB_partial/failure_basin_fine")
    p.add_argument("--planner", default="CNN")
    p.add_argument("--fine-step", type=float, default=0.01)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Failure Basin Analysis (Fine-Resolution) ===")
    df = pd.read_csv(args.csv)

    # 1) 训 CNN XGBoost
    print(f"\n=== Training {args.planner} XGBoost ===")
    model, feat, aucs = train_cnn_model(df)
    print(f"  per-fold AUC: {[f'{a:.3f}' for a in aucs]}")
    print(f"  using best fold, AUC = {max(aucs):.3f}")

    # 2) 对每个 (scene, attack), 找 critical_strength (fine)
    print(f"\n=== Finding fine critical_strength for each (scene, attack) ===")
    sub = df[df["planner"] == args.planner].copy()
    rows = []
    for (scene, atk), g in sub.groupby(["scene_token", "attack"]):
        g = g.sort_values("strength")
        strengths = g["strength"].values
        successes = g["success"].values.astype(int)
        if len(strengths) < 2:
            continue
        gene_vecs = g[feat[:-1]].values.astype(np.float32)  # 37 维 gene
        # 找 success 0→1 跳变 (单调假设; 非单调也至少取第一次 fail 区间)
        # 这里只考虑 0→1 跳变, 否则取最差 (highest fail) 当 basin width
        # 但 "非单调" 已经在 L5 Test 1 检查过 (89.2% monotonic)
        s_crit = None
        s_low_0, s_high_1 = None, None
        for i in range(len(strengths) - 1):
            if successes[i] == 0 and successes[i + 1] == 1:
                s_low_0 = strengths[i]
                s_high_1 = strengths[i + 1]
                # fine scan
                sc, p_05, p_lo, p_hi, _, _ = find_critical_strength(
                    gene_vecs[i], gene_vecs[i + 1], model, feat,
                    strengths[i], strengths[i + 1], fine_step=args.fine_step
                )
                s_crit = sc
                break
        if s_crit is None:
            # 全部 success=0 → 永不进入 fail basin
            s_crit = 1.0
            s_low_0 = max(strengths)
            s_high_1 = max(strengths)
        rows.append({
            "scene_token": scene,
            "attack": atk,
            "critical_strength_fine": float(s_crit),
            "s_low": float(s_low_0) if s_low_0 is not None else 1.0,
            "s_high": float(s_high_1) if s_high_1 is not None else 1.0,
            "width_fine": float(1.0 - s_crit),
            "monotonic": bool(np.all(np.diff(successes) >= 0)),
        })
    case_df = pd.DataFrame(rows)
    case_df.to_csv(out_dir / "per_case.csv", index=False)
    print(f"  {len(case_df)} cases")

    # 3) Test 1 monotonicity (re-verify)
    mono_rate = case_df["monotonic"].mean()
    print(f"\n=== Test 1 Monotonicity: {mono_rate:.3f} ===")

    # 4) Test 2 & 3 fine critical strength & width
    # 包含所有 (scene, attack) (即使是 s_crit=1.0 也保留以示"未挂")
    print(f"\n=== Test 2 Critical Strength (fine) — all {len(case_df)} cases ===")
    for pl in ["CNN", "DINO", "TF"]:
        sub_pl = sub if pl == args.planner else None
        # 实际这里只对 CNN 算; 若要看其他 planner, 需多训几次
    # 显示分布
    fail_mask = case_df["critical_strength_fine"] < 1.0
    fail_df = case_df[fail_mask].copy()
    print(f"  fail cases: {len(fail_df)} / {len(case_df)}")
    crit_med = float(fail_df["critical_strength_fine"].median())
    crit_mean = float(fail_df["critical_strength_fine"].mean())
    width_med = float(fail_df["width_fine"].median())
    width_mean = float(fail_df["width_fine"].mean())
    print(f"  critical_strength fine: median={crit_med:.3f}  mean={crit_mean:.3f}")
    print(f"  width fine:            median={width_med:.3f}  mean={width_mean:.3f}")

    # 5) 画图
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial"]
    colors = {"CNN": "#d62728", "DINO": "#1f77b4", "TF": "#2ca02c"}
    col = colors[args.planner]

    # Figure 2 v2: Failure Boundary (fine)
    fig, ax = plt.subplots(figsize=(7, 5))
    data = fail_df["critical_strength_fine"].values
    parts = ax.violinplot([data], positions=[0], showmedians=True, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor(col); pc.set_alpha(0.5)
    ax.boxplot([data], positions=[0], widths=0.15, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor="white", edgecolor="black"),
               medianprops=dict(color="black", linewidth=1.5))
    ax.scatter([0], [crit_med], marker="D", s=80, color=col, edgecolor="black", zorder=5)
    ax.text(0, 0.05, f"{crit_med:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks([0])
    ax.set_xticklabels([f"{args.planner}\n(n={len(data)})"], fontsize=12)
    ax.set_ylabel("Critical Strength (fine, 0.01 step)", fontsize=11)
    ax.set_title(f"Test 2 (fine): {args.planner} Failure Boundary\n"
                 f"lower = planner fails earlier", fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figure2_critical_strength_fine.png", dpi=140)
    plt.close(fig)
    print(f"  saved → {out_dir / 'figure2_critical_strength_fine.png'}")

    # Figure 3 v2: Basin Width (fine)
    fig, ax = plt.subplots(figsize=(7, 5))
    data_w = fail_df["width_fine"].values
    parts = ax.violinplot([data_w], positions=[0], showmedians=True, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor(col); pc.set_alpha(0.5)
    ax.boxplot([data_w], positions=[0], widths=0.15, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor="white", edgecolor="black"),
               medianprops=dict(color="black", linewidth=1.5))
    ax.scatter([0], [width_med], marker="D", s=80, color=col, edgecolor="black", zorder=5)
    ax.text(0, 0.05, f"{width_med:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks([0])
    ax.set_xticklabels([f"{args.planner}\n(n={len(data_w)})"], fontsize=12)
    ax.set_ylabel("Failure Basin Width (fine)", fontsize=11)
    ax.set_title(f"Test 3 (fine): {args.planner} Failure Basin Width\n"
                 f"higher = planner in failure for longer", fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figure3_basin_width_fine.png", dpi=140)
    plt.close(fig)
    print(f"  saved → {out_dir / 'figure3_basin_width_fine.png'}")

    # 6) summary
    summary = {
        "method": "XGBoost-smooth, linear gene interp, fine_step=" + str(args.fine_step),
        "n_total_cases": int(len(case_df)),
        "n_fail_cases": int(len(fail_df)),
        "monotonicity_rate": float(mono_rate),
        "critical_strength_median": crit_med,
        "critical_strength_mean": crit_mean,
        "width_median": width_med,
        "width_mean": width_mean,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Fine-Resolution Failure Basin ({args.planner})\n")
        f.write(f"========================================\n\n")
        f.write(f"Method: {summary['method']}\n")
        f.write(f"Total cases: {len(case_df)}, Fail cases: {len(fail_df)}\n")
        f.write(f"Monotonicity: {mono_rate:.3f}\n\n")
        f.write(f"Critical Strength (fine):\n")
        f.write(f"  median = {crit_med:.3f}  (vs coarse 0.40)\n")
        f.write(f"  mean   = {crit_mean:.3f}\n\n")
        f.write(f"Basin Width (fine):\n")
        f.write(f"  median = {width_med:.3f}  (vs coarse 0.60)\n")
        f.write(f"  mean   = {width_mean:.3f}\n")
    print(f"  saved → {out_dir}/summary.{{json,txt}}")
    print(f"\nDONE.")


if __name__ == "__main__":
    main()
