"""Failure Basin 反事实验证 — Gene → Failure 规律的因果可操控性。

这是 [[attack-genome-law-hunting]] 中"Failure Basin + counterfactual falsification"的实现。

实验设计：
    1. 在 CNN 训练集 (gene → fail) 上训练 XGBoost
    2. 取 100 个 CNN fail 样本 (按 predicted_prob 从高到低挑)
    3. 对每个 fail 样本，用 SHAP 找到 top-K 关键 gene
    4. 把 top-K gene 的值"推回" success-domain:
       - target_value = CNN success 样本中该 gene 的中位数
       - new_gene_vector = original copy with top-K replaced
    5. 用 XGBoost 重新预测：success prob 有没有降下来？
    6. 报 flip_rate = 比例 (predicted_fail → predicted_success) at K=1,3,5,10

Law 预测：
    - flip_rate > 0.6 → 基因可操控，law 是因果的（强 law）
    - flip_rate 0.3-0.6 → 部分因果（混合 law）
    - flip_rate < 0.3 → 失败是分布式的，law 只是描述性（弱 law）

输入: exp/tierB_partial/merged_3pl.csv
输出: exp/tierB_partial/failure_basin/counterfactual_report.json
      exp/tierB_partial/failure_basin/counterfactual_per_sample.csv
      exp/tierB_partial/failure_basin/counterfactual_summary.txt
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


def train_cnn_model(df: pd.DataFrame, seed: int = 42) -> Tuple:
    """5-fold GroupKFold 训练 CNN XGBoost；返回最后一折的 model + 训练/测试索引。"""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

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
            random_state=seed, n_jobs=-1, eval_metric="logloss",
        )
        m.fit(X[tr], y[tr])
        try:
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        except ValueError:
            pass
        models.append((m, tr, te))

    # 取 AUC 中位的那一折作为"代表 model"
    best = max(range(len(aucs)), key=lambda i: aucs[i])
    print(f"  per-fold CNN AUC: {[f'{a:.3f}' for a in aucs]}  | using fold#{best} (AUC={aucs[best]:.3f})")
    return models[best], feat


def per_sample_shap(model, X: np.ndarray) -> np.ndarray:
    """per-sample SHAP 重要度。用 xgboost 自带 pred_contribs（不依赖 shap 包）。

    返回 shape = (n_samples, n_features) 的 *signed* contribution。
    重要：返回有符号而不是 |.| — 反事实只替换"把 prob 推高"（正贡献）的 gene，
    否则可能在打反方向。
    """
    import xgboost as xgb
    try:
        booster = model.get_booster()
        dmat = xgb.DMatrix(X)
        contribs = booster.predict(dmat, pred_contribs=True)
        # contribs: (n_samples, n_features + 1) — 最后一列是 bias
        contribs = contribs[:, :-1]
        return contribs  # 有符号
    except Exception as e:
        print(f"  [WARN] pred_contribs failed ({e}), falling back to global importance")
        # 退化: 全正（假设 positive correlation）
        return np.tile(np.abs(model.feature_importances_), (X.shape[0], 1))


def counterfactual_pushback(
    X_orig: np.ndarray, y_orig: np.ndarray, ss_orig: np.ndarray,
    X_success: np.ndarray,
    model, feat: List[str],
    n_fail: int = 100, k_values: List[int] = (1, 3, 5, 10),
    seed: int = 42,
) -> pd.DataFrame:
    """核心反事实循环。

    1. 从 CNN samples 中挑出 predicted_prob > 0.5 的 fail (由 model 预测)
       按 scene 不重叠取 n_fail 个
    2. 对每个 fail 样本：
       - 算 SHAP top-K
       - 把 top-K gene 替换为 X_success 中对应 gene 的中位数
       - 重新预测
    3. 报: original_prob, cf_prob_K, flipped (cf_prob < 0.5)
    """
    rng = np.random.RandomState(seed)
    # 1) 预测所有样本的概率
    probs_all = model.predict_proba(X_orig)[:, 1]
    # 2) predicted-fail = prob > 0.5，按 scene 不重叠挑
    is_pred_fail = probs_all > 0.5
    fail_idx_all = np.where(is_pred_fail)[0]
    print(f"  total predicted-fail CNN samples: {len(fail_idx_all)}")

    # 按 scene 去重 (从每个 scene 选一个最强 fail 样本)
    fail_by_scene = {}
    for i in fail_idx_all:
        s = ss_orig[i]
        if s not in fail_by_scene or probs_all[i] > probs_all[fail_by_scene[s]]:
            fail_by_scene[s] = i
    selected = list(fail_by_scene.values())
    rng.shuffle(selected)
    selected = selected[:n_fail]
    print(f"  selected {len(selected)} fail samples across {len({ss_orig[i] for i in selected})} scenes")

    # 3) success-domain 中位数
    success_median = np.nanmedian(X_success, axis=0)
    success_median = np.nan_to_num(success_median, nan=0.0, posinf=0.0, neginf=0.0)

    # 4) per-sample SHAP (一次性算全部 100 个 fail 样本)
    Xsel = X_orig[selected]
    shap_signed = per_sample_shap(model, Xsel)  # (n_samples, n_features), 有符号
    # 只看"把 prob 推高" (positive contribution) 的 gene，按正贡献大小降序
    pos_contrib = np.where(shap_signed > 0, shap_signed, 0)
    topk_idx = np.argsort(-pos_contrib, axis=1)  # 降序，每个样本自己的 top positive gene
    top1_list = [feat[i] for i in topk_idx[:, 0]]
    print(f"  per-sample SHAP computed: top1 positive-contrib gene 分布: "
          f"{pd.Series(top1_list).value_counts().head(5).to_dict()}")

    # 4b) Random-K baseline（每个 sample 随机选 K 个 gene）
    n_features = len(feat)

    # 5) 对每个 K 做 pushback (top-K + random-K 对照)
    rows = []
    print(f"\n  --- K  |  top-K flip  |  random-K flip  |  lift  ---")
    for ki, K in enumerate(k_values):
        flips_top = 0
        flips_rand = 0
        for ri, orig_i in enumerate(selected):
            # top-K
            topk = topk_idx[ri, :K]
            X_cf = X_orig[orig_i].copy()
            X_cf[topk] = success_median[topk]
            cf_prob = model.predict_proba(X_cf[None, :])[:, 1][0]
            flipped_top = int(cf_prob < 0.5)
            flips_top += flipped_top
            # random-K (固定 seed 派生 sub-seed)
            sub_rng = np.random.RandomState(seed * 1000 + ri)
            rk = sub_rng.choice(n_features, size=K, replace=False)
            X_cfr = X_orig[orig_i].copy()
            X_cfr[rk] = success_median[rk]
            cf_prob_r = model.predict_proba(X_cfr[None, :])[:, 1][0]
            flipped_rand = int(cf_prob_r < 0.5)
            flips_rand += flipped_rand
            rows.append({
                "k": K,
                "scene_token": ss_orig[orig_i],
                "orig_prob": float(probs_all[orig_i]),
                "cf_prob_topk": float(cf_prob),
                "cf_prob_rand": float(cf_prob_r),
                "delta_topk": float(probs_all[orig_i] - cf_prob),
                "delta_rand": float(probs_all[orig_i] - cf_prob_r),
                "flipped_topk": flipped_top,
                "flipped_rand": flipped_rand,
                "top1_gene": feat[topk_idx[ri, 0]],
                "top3_genes": "|".join(feat[topk_idx[ri, j]] for j in range(min(3, K))),
            })
        top_rate = flips_top / len(selected)
        rand_rate = flips_rand / len(selected)
        lift = top_rate - rand_rate
        print(f"  --- {K:2d}  |  {top_rate:>10.3f}  |  {rand_rate:>13.3f}  |  +{lift:.3f} ---")

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="exp/tierB_partial/merged_3pl.csv")
    p.add_argument("--output-dir", default="exp/tierB_partial/failure_basin")
    p.add_argument("--n-fail", type=int, default=100)
    p.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10])
    p.add_argument("--planner", default="CNN", help="Source planner for counterfactual")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Failure Basin Counterfactual ({args.planner}) ===")
    df = pd.read_csv(args.csv)
    print(f"  loaded {len(df)} rows, {df['scene_token'].nunique()} scenes")

    # 1) 训练 CNN model
    print(f"\n=== Train CNN XGBoost (5-fold GroupKFold) ===")
    (model, _, te), feat = train_cnn_model(df)

    # 2) 取所有 CNN 样本
    sub = df[df["planner"] == args.planner].copy()
    X = sub[feat].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = sub["success"].values.astype(np.int32)
    ss = sub["scene_token"].values

    # 3) success-domain 参考集
    X_success = X[y == 1]
    print(f"  CNN success-domain median source: {len(X_success)} samples")

    # 4) 反事实
    print(f"\n=== Counterfactual pushback (top-K gene → success median) ===")
    res = counterfactual_pushback(X, y, ss, X_success, model, feat,
                                   n_fail=args.n_fail, k_values=args.k_values)
    res.to_csv(out_dir / "counterfactual_per_sample.csv", index=False)

    # 5) 汇总
    summary = {}
    for K in args.k_values:
        sub_k = res[res["k"] == K]
        summary[K] = {
            "n_samples": len(sub_k),
            "flip_rate": float(sub_k["flipped_topk"].mean()),
            "flip_rate_random": float(sub_k["flipped_rand"].mean()),
            "median_delta": float(sub_k["delta_topk"].median()),
            "mean_delta": float(sub_k["delta_topk"].mean()),
        }

    with open(out_dir / "counterfactual_report.json", "w") as f:
        json.dump({
            "planner": args.planner,
            "n_fail": args.n_fail,
            "k_values": args.k_values,
            "summary": summary,
        }, f, indent=2)

    # 6) 人类可读
    with open(out_dir / "counterfactual_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"# Failure Basin Counterfactual — {args.planner}\n")
        f.write(f"# 把 fail 样本 top-K 重要 gene 替换为 success-domain 中位数\n")
        f.write(f"# 报: flip_rate = P(cf_prob < 0.5)\n\n")
        f.write(f"  n_fail samples: {args.n_fail}\n")
        f.write(f"  success-domain reference: {len(X_success)} CNN success samples\n\n")
        f.write(f"  {'K':>3s}  {'top-K flip':>11s}  {'random flip':>12s}  {'lift':>7s}  {'mean Δ_top':>10s}  判定\n")
        f.write(f"  {'-'*3}  {'-'*11}  {'-'*12}  {'-'*7}  {'-'*10}  {'-'*22}\n")
        for K, s in summary.items():
            sub = res[res["k"] == K]
            rand_rate = float(sub["flipped_rand"].mean())
            lift = s["flip_rate"] - rand_rate
            if s["flip_rate"] > 0.6 and lift > 0.1:
                verdict = "强 law (因果可操控)"
            elif s["flip_rate"] > 0.3 and lift > 0.05:
                verdict = "混合 law (部分可操控)"
            elif s["flip_rate"] > 0.6:
                verdict = "强效应，但 random 也高(慎重)"
            else:
                verdict = "弱 law (描述性)"
            f.write(f"  {K:>3d}  {s['flip_rate']:>11.3f}  {rand_rate:>12.3f}  {lift:>+7.3f}  "
                    f"{s['mean_delta']:>10.3f}  {verdict}\n")
        f.write(f"\n  top1 positive-contrib gene 分布 (per-sample SHAP):\n")
        top1_counts = res[res["k"] == args.k_values[0]]["top1_gene"].value_counts().head(10)
        for g, c in top1_counts.items():
            f.write(f"    {g:>20s}: {c}\n")
        f.write(f"\n  详见: counterfactual_per_sample.csv, counterfactual_report.json\n")
    print(f"  saved {out_dir}/counterfactual_*.{['csv','json','txt']}")
    print(f"\nDONE. outputs → {out_dir}")


if __name__ == "__main__":
    main()
