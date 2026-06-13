"""跨 Planner 预测验证 — 证明 Gene → Failure 规律可外推。

这是 [[attack-genome-law-hunting]] 中的"预测验证"步骤。

设计:
    1. 把 500 场景 5-fold split（按 scene_token，保证场景不泄漏）
    2. 对每折：用源 planner 的 (gene → success) 训练 XGBoost
    3. 用训练好的模型预测目标 planner 在测试集上的 success
    4. 计算 R² / AUC / accuracy
    5. 同样可以反方向跑 (target → source)
    6. 跨多对 planner 对 (CNN↔DINO, CNN↔TF, DINO↔TF) 都跑

如果 R²>0.8 / AUC>0.85，说明:
    - 攻击迁移性是"基因驱动"的，不是"模型特有"的
    - 可以用便宜的 planner 预测贵的 planner
    - 论文级证据：存在"跨表征守恒因子"

输入: per_sample_genes.csv
输出:
    - cross_planner_r2.json (每对 planner 的 R²/AUC)
    - per_pair_predictions.csv (详细预测结果)
    - transfer_law_summary.txt
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
META_FIELDS = ["strength"]


def build_xy(df: pd.DataFrame, planner: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造 (X, y, scene_idx) - scene_idx 用于分组 CV。"""
    sub = df[df["planner"] == planner].copy()
    use_cols = [c for c in GENE_FIELDS + META_FIELDS if c in sub.columns]
    X = sub[use_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = sub["success"].values.astype(np.int32)
    # scene_token → 0..N-1（用 hash 简化；分组 CV 内部重映射）
    scenes = sub["scene_token"].values
    return X, y, scenes


def cross_planner_r2(
    df: pd.DataFrame, src: str, tgt: str, n_folds: int = 5, seed: int = 42,
) -> Dict:
    """src → tgt：训练 src 的 gene→success 模型，预测 tgt。

    关键：按 scene_token 对齐 src 和 tgt（不能用 idx，必须用 token join），
    因为不同 planner 的 row 顺序可能不同。
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, r2_score, accuracy_score
    import xgboost as xgb

    # 用 scene_token 对齐
    use_cols = [c for c in GENE_FIELDS + META_FIELDS if c in df.columns]
    # strength 既在 use_cols 也是 key，分开处理
    key_cols = ["scene_token", "attack", "strength"]
    feature_cols = [c for c in use_cols if c not in key_cols]
    df_src = df[df["planner"] == src].copy()
    df_src["y_src"] = df_src["success"]
    df_tgt = df[df["planner"] == tgt].copy()
    df_tgt["y_tgt"] = df_tgt["success"]
    # merge on key cols
    merged = df_src[key_cols + ["y_src"] + feature_cols].merge(
        df_tgt[key_cols + ["y_tgt"]],
        on=key_cols, how="inner",
    )
    n = len(merged)
    if n < n_folds * 5:
        return {"src": src, "tgt": tgt, "n_folds": n_folds, "n_samples": n,
                "auc_mean": float("nan"), "auc_std": float("nan"),
                "r2_mean": float("nan"), "r2_std": float("nan"),
                "acc_mean": float("nan"), "acc_std": float("nan"),
                "per_fold": {"aucs": [], "r2s": [], "accs": []},
                "predictions": []}

    Xs = merged[feature_cols].values.astype(np.float32)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    ys = merged["y_src"].values.astype(np.int32)
    yt = merged["y_tgt"].values.astype(np.int32)
    ss = merged["scene_token"].values

    gkf = GroupKFold(n_splits=n_folds)
    aucs, r2s, accs = [], [], []
    all_pred: List[dict] = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(Xs, ys, ss)):
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, eval_metric="logloss",
        )
        model.fit(Xs[train_idx], ys[train_idx])
        proba = model.predict_proba(Xs[test_idx])[:, 1]
        pred = (proba > 0.5).astype(int)
        try:
            auc = float(roc_auc_score(yt[test_idx], proba))
        except ValueError:
            auc = float("nan")
        try:
            r2 = float(r2_score(yt[test_idx], proba))
        except ValueError:
            r2 = float("nan")
        acc = float(accuracy_score(yt[test_idx], pred))
        aucs.append(auc); r2s.append(r2); accs.append(acc)
        for i, idx in enumerate(test_idx):
            all_pred.append({
                "fold": fold, "scene_token": ss[idx], "src": src, "tgt": tgt,
                "y_true_src": int(ys[idx]), "y_true_tgt": int(yt[idx]),
                "y_pred_proba": float(proba[i]),
            })

    return {
        "src": src, "tgt": tgt,
        "n_folds": n_folds, "n_samples": n,
        "auc_mean": float(np.nanmean(aucs)), "auc_std": float(np.nanstd(aucs)),
        "r2_mean": float(np.nanmean(r2s)), "r2_std": float(np.nanstd(r2s)),
        "acc_mean": float(np.nanmean(accs)), "acc_std": float(np.nanstd(accs)),
        "per_fold": {"aucs": aucs, "r2s": r2s, "accs": accs},
        "predictions": all_pred,
    }


def same_planner_cv(df: pd.DataFrame, planner: str, n_folds: int = 5) -> Dict:
    """同 planner 5-fold CV → baseline 上限。"""
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.metrics import make_scorer, roc_auc_score
    import xgboost as xgb

    X, y, scenes = build_xy(df, planner)
    gkf = GroupKFold(n_splits=n_folds)
    aucs = []
    for train_idx, test_idx in gkf.split(X, y, scenes):
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=42, n_jobs=-1, eval_metric="logloss",
        )
        m.fit(X[train_idx], y[train_idx])
        try:
            aucs.append(roc_auc_score(y[test_idx], m.predict_proba(X[test_idx])[:, 1]))
        except ValueError:
            pass
    return {
        "planner": planner, "n_folds": n_folds,
        "auc_mean": float(np.nanmean(aucs)), "auc_std": float(np.nanstd(aucs)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--planners", nargs="+", default=None,
                   help="默认跑所有 planner 之间的两两对")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Cross-Planner Prediction (Gene → Success) ===")
    print(f"  CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  loaded {len(df)} rows, {df['scene_token'].nunique()} scenes")
    print(f"  planners: {sorted(df['planner'].unique())}")

    planners = args.planners or sorted(df["planner"].unique())
    if len(planners) < 2:
        print("  need >= 2 planners for cross-prediction")
        return

    # 1. 同 planner baseline (上限)
    print(f"\n=== Same-planner CV (baseline upper bound) ===")
    baselines = {}
    for pl in planners:
        b = same_planner_cv(df, pl, args.n_folds)
        baselines[pl] = b
        print(f"  {pl}: AUC = {b['auc_mean']:.3f} ± {b['auc_std']:.3f}")

    # 2. 跨 planner 预测
    print(f"\n=== Cross-planner prediction (src → tgt) ===")
    from itertools import permutations
    results = []
    all_pred_rows: List[dict] = []
    for src, tgt in permutations(planners, 2):
        if src == tgt:
            continue
        r = cross_planner_r2(df, src, tgt, args.n_folds)
        results.append({
            "src": src, "tgt": tgt,
            "auc_mean": r["auc_mean"], "auc_std": r["auc_std"],
            "r2_mean": r["r2_mean"], "r2_std": r["r2_std"],
            "acc_mean": r["acc_mean"], "acc_std": r["acc_std"],
            "n_samples": r["n_samples"],
        })
        print(f"  {src:>5s} → {tgt:<5s}: AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f} "
              f"R²={r['r2_mean']:.3f}±{r['r2_std']:.3f}")
        all_pred_rows.extend(r["predictions"])

    # 3. 保存
    with open(out_dir / "cross_planner_r2.json", "w") as f:
        json.dump({
            "baselines": baselines,
            "cross_results": results,
            "n_folds": args.n_folds,
        }, f, indent=2)
    pd.DataFrame(all_pred_rows).to_csv(out_dir / "per_pair_predictions.csv", index=False)
    pd.DataFrame(results).to_csv(out_dir / "cross_planner_summary.csv", index=False)

    # 4. 人类可读 summary
    with open(out_dir / "transfer_law_summary.txt", "w") as f:
        f.write(f"# Cross-Planner Transfer Law Summary\n")
        f.write(f"# baseline: 同 planner 5-fold CV (gene→success)\n\n")
        for pl, b in baselines.items():
            f.write(f"  baseline {pl}: AUC = {b['auc_mean']:.3f} ± {b['auc_std']:.3f}\n")
        f.write(f"\n# Cross-planner transfer (用 src 模型预测 tgt success)\n")
        f.write(f"# 关键阈值: AUC>0.7 / R²>0.5 视为迁移成功\n\n")
        for r in results:
            mark = "✅" if r["auc_mean"] > 0.7 else ("⚠️" if r["auc_mean"] > 0.6 else "❌")
            f.write(
                f"  {mark} {r['src']:>5s} → {r['tgt']:<5s}: "
                f"AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f}  "
                f"R²={r['r2_mean']:.3f}±{r['r2_std']:.3f}\n"
            )
        f.write(f"\n  → 见 cross_planner_summary.csv 完整数据\n")
    print(f"  saved cross_planner_r2.json, per_pair_predictions.csv, transfer_law_summary.txt")
    print(f"\nDONE. outputs → {out_dir}")


if __name__ == "__main__":
    main()
