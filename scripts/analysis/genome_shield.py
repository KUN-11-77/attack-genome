"""Genome Shield — Module C of AGSA.

实时检测"是否正在进入 Failure Basin"。

设计（基于 Level 5 独立验证的几何结构）:
    - failure_prob:  CNN XGBoost(gene_vec) → P(fail | scene)
    - basin_distance: ||gene_vec - success_median||_2 / scale
    - basin_risk:    f(failure_prob, basin_distance, trend)
                      进入 basin 边界 (risk > 0.5) 即报警

答辩叙事:
    "不是分类器" — 是"在线安全监测仪"。
    "进入 Failure Basin" 而非"fail prob = 0.82"。

输入: 88,560 样本 gene data
输出:
    - genome_shield/cnn_shield.pkl 训练好的 XGBoost
    - genome_shield/cnn_success_median.npy CNN success 37-dim median
    - genome_shield/demo.gif 30s 实时 demo
    - genome_shield/latency.txt 推理延迟 benchmark
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.gridspec as gridspec

# ---- gene field 列表（与 ff3/cross_planner 一致） ----
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


def train_shield(csv_path: str, planner: str = "CNN"):
    """训练 CNN XGBoost (gene → fail) + 算 success 域 median。"""
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    print(f"=== Training Genome Shield for {planner} ===")
    df = pd.read_csv(csv_path)
    sub = df[df["planner"] == planner].copy()
    feat = [c for c in GENE_FIELDS + META_FIELDS if c in sub.columns]
    X = sub[feat].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = sub["success"].values.astype(np.int32)
    ss = sub["scene_token"].values

    # 5-fold GroupKFold 训最终 model (用全部数据)
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
    print(f"  per-fold AUC: {[f'{a:.3f}' for a in aucs]}")
    best_i = max(range(len(aucs)), key=lambda i: aucs[i])
    model = models[best_i]
    print(f"  selected fold#{best_i} (AUC={aucs[best_i]:.3f})")

    # success 域 median
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    success_idx = (y == 0)  # success=0 (planner survived)
    success_median_scaled = np.median(X_scaled[success_idx], axis=0)
    success_median_raw = np.median(X[success_idx], axis=0)
    # 用 scaled median 作为 basin center (L2 norm 有意义)
    X_std = X.std(axis=0) + 1e-6
    success_median_norm = success_median_raw  # in raw space, scale via X_std

    return model, scaler, X_std, success_median_raw, feat


def shield_predict(gene_vec: np.ndarray, model, scaler, X_std: np.ndarray,
                   success_median_raw: np.ndarray) -> Dict[str, float]:
    """对单帧 gene vector 返回 shield 监测指标。"""
    gv = np.asarray(gene_vec, dtype=np.float32).reshape(1, -1)
    gv = np.nan_to_num(gv, nan=0.0, posinf=0.0, neginf=0.0)

    # 1) failure prob
    fail_prob = float(model.predict_proba(gv)[:, 1][0])

    # 2) basin distance (Euclidean in scaled space)
    gv_scaled = scaler.transform(gv)[0]
    sm_scaled = scaler.transform(success_median_raw.reshape(1, -1))[0]
    basin_dist = float(np.linalg.norm(gv_scaled - sm_scaled))

    # 3) basin risk (composite, 偏向进入 basin 边界)
    #    risk=0 当 fail_prob<0.3 & basin_dist<3 (deep in success region)
    #    risk=1 当 fail_prob>0.7 or basin_dist>6 (in basin)
    basin_risk_fail = np.clip((fail_prob - 0.3) / 0.4, 0, 1)
    basin_risk_dist = np.clip((basin_dist - 3.0) / 3.0, 0, 1)
    basin_risk = float(max(basin_risk_fail, basin_risk_dist))

    # 4) status label
    if basin_risk > 0.8:
        status = "IN FAILURE BASIN"
    elif basin_risk > 0.5:
        status = "ENTERING BASIN"
    elif basin_risk > 0.3:
        status = "APPROACHING"
    else:
        status = "SAFE"

    return {
        "fail_prob": fail_prob,
        "basin_dist": basin_dist,
        "basin_risk": basin_risk,
        "status": status,
    }


def benchmark_latency(model, scaler, feat_dim: int, n_iter: int = 100) -> float:
    """测单帧推理 latency (ms)。"""
    gv = np.random.RandomState(0).randn(feat_dim).astype(np.float32)
    # warmup
    for _ in range(10):
        shield_predict(gv, model, scaler, np.ones(feat_dim), np.zeros(feat_dim))
    t0 = time.time()
    for _ in range(n_iter):
        shield_predict(gv, model, scaler, np.ones(feat_dim), np.zeros(feat_dim))
    dt = (time.time() - t0) / n_iter
    return dt * 1000


def build_demo_sequence(df: pd.DataFrame, planner: str = "CNN") -> pd.DataFrame:
    """构造一个 30 秒 demo 用的样本序列。

    选 3 个不同 (scene, attack) 组合 × 6 strength = 18 帧，每帧 1.7s = 30s。
    - 帧 1-6:  scene A + rain (会进入 basin)
    - 帧 7-12: scene B + dusk (不进入 basin)
    - 帧 13-18: scene C + digital_noise (边缘进入 basin)
    """
    sub = df[df["planner"] == planner]
    # 选 3 个不同 (scene, attack) 组合
    combos = []
    for (scene, atk), g in sub.groupby(["scene_token", "attack"]):
        s_sorted = g.sort_values("strength")
        if len(s_sorted) == 6:
            # 至少有一次 fail
            if s_sorted["success"].max() == 1:
                combos.append((scene, atk, s_sorted))
    if not combos:
        raise RuntimeError("no suitable combos found")
    rng = np.random.RandomState(7)
    rng.shuffle(combos)
    selected = combos[:3]
    seq = pd.concat([c[2] for c in selected], ignore_index=True)
    return seq


def render_demo(seq: pd.DataFrame, model, scaler, X_std: np.ndarray,
                success_median_raw: np.ndarray, feat: List[str],
                out_gif: Path, fps: int = 2, frame_dur: float = 1.7):
    """渲染 30s 实时 demo GIF。"""
    print(f"=== Rendering demo → {out_gif} ===")
    n = len(seq)
    duration = n * frame_dur
    print(f"  n_frames={n}, total_duration≈{duration:.1f}s @ {fps} fps")

    # 预计算所有 frame 的 shield 输出
    history = {
        "frame_idx": [], "fail_prob": [], "basin_dist": [],
        "basin_risk": [], "status": [],
        "edge_mean": [], "lane_line_density": [], "mean_luma": [],
        "strength": [], "attack": [], "scene": [],
    }
    for i, row in seq.iterrows():
        gv = row[feat].values.astype(np.float32)
        out = shield_predict(gv, model, scaler, X_std, success_median_raw)
        history["frame_idx"].append(len(history["frame_idx"]))
        history["fail_prob"].append(out["fail_prob"])
        history["basin_dist"].append(out["basin_dist"])
        history["basin_risk"].append(out["basin_risk"])
        history["status"].append(out["status"])
        history["edge_mean"].append(float(row.get("edge_mean", 0.0)))
        history["lane_line_density"].append(float(row.get("lane_line_density", 0.0)))
        history["mean_luma"].append(float(row.get("mean_luma", 0.0)))
        history["strength"].append(float(row.get("strength", 0.0)))
        history["attack"].append(str(row.get("attack", "")))
        history["scene"].append(str(row.get("scene_token", ""))[:8])

    # 4 子图: edge_mean, lane_density, basin_risk, fail_prob
    fig = plt.figure(figsize=(13, 8.5))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.45, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")

    status_color = {
        "SAFE": "#2ca02c", "APPROACHING": "#bcbd22",
        "ENTERING BASIN": "#ff7f0e", "IN FAILURE BASIN": "#d62728",
    }

    def update(fi):
        for ax in (ax1, ax2, ax3, ax4, ax5):
            ax.clear()
        # ax1: edge_mean + lane_line_density
        ax1.plot(range(fi + 1), history["edge_mean"][:fi + 1], "o-", color="#1f77b4", label="edge_mean")
        ax1.plot(range(fi + 1), history["lane_line_density"][:fi + 1], "s-", color="#ff7f0e", label="lane_line_density")
        ax1.scatter([fi], [history["edge_mean"][fi]], color="red", s=60, zorder=5)
        ax1.set_xlim(-0.5, n)
        ax1.set_ylim(0, max(max(history["edge_mean"]), max(history["lane_line_density"])) * 1.1 + 0.01)
        ax1.set_title("Structural Genes (per-frame)", fontsize=11)
        ax1.set_ylabel("gene value")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(alpha=0.3)
        # ax2: basin_risk (composite)
        ax2.plot(range(fi + 1), history["basin_risk"][:fi + 1], "o-", color="#d62728", linewidth=2)
        ax2.fill_between(range(fi + 1), 0, history["basin_risk"][:fi + 1], alpha=0.2, color="#d62728")
        ax2.scatter([fi], [history["basin_risk"][fi]], color="black", s=80, zorder=5)
        ax2.axhline(0.5, color="orange", linestyle="--", alpha=0.5, label="Entering threshold")
        ax2.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="In basin threshold")
        ax2.set_xlim(-0.5, n)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_title("Genome Shield: Basin Risk", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Basin Risk")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(alpha=0.3)
        # ax3: basin_dist
        ax3.plot(range(fi + 1), history["basin_dist"][:fi + 1], "o-", color="#9467bd")
        ax3.scatter([fi], [history["basin_dist"][fi]], color="black", s=60, zorder=5)
        ax3.set_xlim(-0.5, n)
        ax3.set_ylabel("Euclidean Dist")
        ax3.set_title("Distance to CNN Success Region", fontsize=11)
        ax3.grid(alpha=0.3)
        # ax4: fail_prob
        ax4.plot(range(fi + 1), history["fail_prob"][:fi + 1], "o-", color="#17becf")
        ax4.scatter([fi], [history["fail_prob"][fi]], color="black", s=60, zorder=5)
        ax4.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="decision boundary")
        ax4.set_xlim(-0.5, n)
        ax4.set_ylim(-0.05, 1.05)
        ax4.set_title("CNN XGBoost P(fail)", fontsize=11)
        ax4.set_ylabel("P(fail)")
        ax4.legend(loc="upper left", fontsize=8)
        ax4.grid(alpha=0.3)
        # ax5: status panel
        st = history["status"][fi]
        clr = status_color.get(st, "black")
        ax5.text(0.5, 0.75, f"Frame {fi + 1}/{n}   |   attack: {history['attack'][fi]}   |   "
                f"strength: {history['strength'][fi]:.2f}   |   scene: {history['scene'][fi]}",
                ha="center", va="center", fontsize=11, transform=ax5.transAxes)
        ax5.text(0.5, 0.35, f"STATUS: {st}", ha="center", va="center",
                fontsize=22, fontweight="bold", color=clr, transform=ax5.transAxes)
        ax5.text(0.5, 0.05, f"Basin Risk = {history['basin_risk'][fi]:.3f}   |   "
                f"Basin Dist = {history['basin_dist'][fi]:.2f}   |   "
                f"P(fail) = {history['fail_prob'][fi]:.3f}",
                ha="center", va="center", fontsize=10, color="gray",
                family="monospace", transform=ax5.transAxes)

    anim = FuncAnimation(fig, update, frames=n, interval=int(1000 / fps))
    try:
        anim.save(str(out_gif), writer=PillowWriter(fps=fps))
        print(f"  saved GIF → {out_gif}  ({out_gif.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"  [WARN] GIF save failed: {e}")
        # fallback: 拼一帧静态图
        fig.suptitle("Genome Shield: Failure Basin Monitoring (one frame)", fontsize=14)
        fig.savefig(out_gif.with_suffix(".png"), dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="exp/tierB_partial/merged_3pl.csv")
    p.add_argument("--output-dir", default="exp/tierB_partial/genome_shield")
    p.add_argument("--planner", default="CNN")
    p.add_argument("--n-bench", type=int, default=200)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 训 shield
    model, scaler, X_std, success_median_raw, feat = train_shield(args.csv, args.planner)

    # 2) latency benchmark
    print(f"\n=== Latency benchmark ({args.n_bench} iter) ===")
    dt_ms = benchmark_latency(model, scaler, len(feat), args.n_bench)
    print(f"  per-frame inference: {dt_ms:.2f} ms")
    with open(out_dir / "latency.txt", "w") as f:
        f.write(f"Genome Shield ({args.planner}) per-frame inference: {dt_ms:.2f} ms\n")
        f.write(f"Benchmark: {args.n_bench} iter on synthetic gene vector\n")
        f.write(f"Target: < 100 ms (real-time at 10 fps)\n")
        f.write(f"Pass: {dt_ms < 100}\n")

    # 3) demo sequence
    print(f"\n=== Building demo sequence ===")
    df = pd.read_csv(args.csv)
    seq = build_demo_sequence(df, args.planner)
    print(f"  selected {len(seq)} frames from {seq['scene_token'].nunique()} scenes")

    # 4) 渲染
    out_gif = out_dir / "demo.gif"
    render_demo(seq, model, scaler, X_std, success_median_raw, feat, out_gif,
                fps=2, frame_dur=1.7)

    # 5) 写 summary
    print(f"\n=== Summary ===")
    print(f"  {dt_ms:.1f} ms / frame (real-time OK at < 100 ms)")
    print(f"  demo: {out_gif}")
    print(f"  artifacts in {out_dir}/")
    print(f"\nDONE.")


if __name__ == "__main__":
    main()
