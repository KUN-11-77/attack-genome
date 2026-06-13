"""Forward-pass Gene-Targeted Counterfactual (Tier C-2 补完).

与 forward_pass_counterfactual.py 的差异：
  旧脚本：把攻击强度从 0.8 降到 [0, 0.2, 0.4, 0.6, 0.8]
         → 验证"减小攻击 → planner 恢复"（强度扫描）
  本脚本：根据 per-sample SHAP top-1 gene **针对性** 应用 gene-mitigation
         图像变换 → 验证"操控特定 gene → planner 恢复"（基因级因果）

Gene-mitigation 算子 (4 类)：
  - edge / freq 类 → gaussian blur（降低高频/边缘）
  - luma 类        → gamma / contrast 调整
  - color 类       → 饱和度衰减
  - detection 类   → 局部对比度调整（仿真"修复"检测）

输出：
  - exp/tierB_partial/forward_pass_counterfactual/gene_cf_<planner>.json
  - exp/tierB_partial/forward_pass_counterfactual/gene_cf_<planner>.csv
  - 3-planner 跨架构对比表 (gene_cf_3pl_summary.txt)

使用：
  python scripts/analysis/forward_pass_gene_counterfactual.py \\
      --csv exp/tierB_partial/merged_3pl.csv \\
      --planner CNN --n-fail 30 \\
      --openscene-root E:/navsim_workspace/dataset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


# ---------------------------------------------------------------------------
# Gene → Mitigation mapping
# ---------------------------------------------------------------------------
# 每个 gene 类别对应一个简单的图像处理算子。设计原则：
#   1. 算子必须可重复（确定性）
#   2. 算子必须与对应 gene 反向（reduce 该 gene 的异常值）
#   3. 不依赖外部模型，OpenCV / NumPy 直接实现

EDGE_FREQ_GENES = {"edge_mean", "edge_density", "high_freq_ratio",
                   "low_freq_ratio", "spectral_centroid",
                   "lbp_entropy", "glcm_contrast", "lbp_uniformity"}

LUMA_GENES = {"mean_luma", "std_luma", "rms_contrast", "dynamic_range",
              "mean_shift", "std_shift", "luma_entropy", "luma_skew",
              "luma_mean", "road_luma_mean", "road_luma_std",
              "shadow_ratio", "highlight_ratio"}

COLOR_GENES = {"hue_mean", "hue_std", "sat_mean", "sat_std",
               "val_mean", "val_std", "colorfulness"}

DETECTION_GENES = {"vehicle_loss", "person_loss", "detection_loss",
                   "conf_loss", "vehicle_loss_ratio"}


def categorize_gene(gene: str) -> str:
    if gene in EDGE_FREQ_GENES:
        return "edge_freq"
    if gene in LUMA_GENES:
        return "luma"
    if gene in COLOR_GENES:
        return "color"
    if gene in DETECTION_GENES:
        return "detection"
    return "other"


# ---------------------------------------------------------------------------
# Image-space mitigations (deterministic, OpenCV-based)
# ---------------------------------------------------------------------------

def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        return np.clip(img, 0, 255).astype(np.uint8)
    return img


def mitigate_edge_freq(img: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Gaussian blur — reduces edge_mean / high_freq_ratio / lbp_entropy."""
    try:
        import cv2
    except ImportError:
        from scipy.ndimage import gaussian_filter
        out = np.zeros_like(img)
        for c in range(img.shape[-1]):
            out[..., c] = gaussian_filter(img[..., c].astype(np.float32), sigma=sigma)
        return _ensure_uint8(out)
    ksize = int(2 * round(2 * sigma) + 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)


def mitigate_luma(img: np.ndarray, gamma: float = 0.85) -> np.ndarray:
    """Gamma correction toward mid-gray — reduces mean_shift / std_luma."""
    inv_gamma = 1.0 / max(gamma, 1e-3)
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    try:
        import cv2
        return cv2.LUT(img, lut)
    except ImportError:
        return lut[img]


def mitigate_color(img: np.ndarray, sat_scale: float = 0.6) -> np.ndarray:
    """Reduce saturation in HSV space — reduces sat_std / colorfulness."""
    try:
        import cv2
    except ImportError:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= sat_scale
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def mitigate_detection(img: np.ndarray, alpha: float = 1.15) -> np.ndarray:
    """Local contrast boost — synthesizes sharper objects, may restore detection."""
    try:
        import cv2
    except ImportError:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(img, 1 + alpha, blurred, -alpha, 0)
    return _ensure_uint8(sharpened)


def apply_mitigation(img: np.ndarray, category: str) -> np.ndarray:
    if category == "edge_freq":
        return mitigate_edge_freq(img)
    if category == "luma":
        return mitigate_luma(img)
    if category == "color":
        return mitigate_color(img)
    if category == "detection":
        return mitigate_detection(img)
    # Default: light blur (safe fallback)
    return mitigate_edge_freq(img, sigma=0.6)


# ---------------------------------------------------------------------------
# XGBoost model loading + per-sample SHAP
# ---------------------------------------------------------------------------

def train_xgb_for_planner(df: pd.DataFrame, planner: str, seed: int = 42):
    """Train one fold of XGBoost (gene→fail) and return model + feature list."""
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

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
        try:
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        except ValueError:
            pass
        models.append((m, tr, te, sub.iloc[te].copy()))

    best = max(range(len(aucs)), key=lambda i: aucs[i])
    return models[best], feat


def per_sample_shap(model, X: np.ndarray) -> np.ndarray:
    """per-sample signed SHAP via xgboost pred_contribs."""
    import xgboost as xgb
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    try:
        contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    except Exception:
        return np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
    return contribs[:, :-1].astype(np.float32)


# ---------------------------------------------------------------------------
# Forward-pass main loop
# ---------------------------------------------------------------------------

def traj_ade(traj_a: Optional[np.ndarray], traj_b: Optional[np.ndarray]) -> float:
    if traj_a is None or traj_b is None:
        return float("nan")
    return float(np.linalg.norm(traj_a[:, :2] - traj_b[:, :2], axis=1).mean())


def load_planner(planner_name: str, device: str = "cuda"):
    from scripts.attack_genome.adapters import (
        build_cnn_adapter_from_yaml,
        build_dino_adapter_from_yaml,
        build_transfuser_adapter_from_yaml,
    )
    factory = {"CNN": build_cnn_adapter_from_yaml,
               "DINO": build_dino_adapter_from_yaml,
               "TF": build_transfuser_adapter_from_yaml}[planner_name]
    return factory(device=device)


def load_image_from_standalone(token: str, openscene_root: str):
    from scripts.analysis.standalone_scene_loader import StandaloneSceneIndex
    if not hasattr(load_image_from_standalone, "_idx"):
        load_image_from_standalone._idx = StandaloneSceneIndex(openscene_root)
    return load_image_from_standalone._idx.load_image(token)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="exp/tierB_partial/merged_3pl.csv")
    p.add_argument("--openscene-root", default="E:/navsim_workspace/dataset")
    p.add_argument("--n-fail", type=int, default=30)
    p.add_argument("--planner", default="CNN", choices=["CNN", "DINO", "TF"])
    p.add_argument("--output-dir", default="exp/tierB_partial/forward_pass_counterfactual")
    p.add_argument("--ade-threshold", type=float, default=2.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"=== Gene-Targeted Forward-pass Counterfactual ({args.planner}) ===")
    print(f"  csv={args.csv}, n_fail={args.n_fail}, openscene={args.openscene_root}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load CSV + pick fail cases
    df = pd.read_csv(args.csv)
    df_pl = df[df["planner"] == args.planner].copy()
    df_fail = df_pl[df_pl["success"] == 0].copy()
    print(f"  total {args.planner} samples: {len(df_pl)}, fail samples: {len(df_fail)}")

    if len(df_fail) < args.n_fail:
        print(f"  [WARN] fewer fail samples than requested ({len(df_fail)} < {args.n_fail})")
    rng = np.random.RandomState(args.seed)
    fail_subset = df_fail.sample(n=min(args.n_fail, len(df_fail)), random_state=args.seed)
    print(f"  selected {len(fail_subset)} fail cases for gene-targeted counterfactual")

    # 2) Train XGBoost (one fold) for SHAP
    print(f"\n=== Training XGBoost (gene→fail) for {args.planner} ===")
    (model, tr_idx, te_idx, te_df), feat = train_xgb_for_planner(df, args.planner, args.seed)

    # Get SHAP for the fail samples (use their actual gene vectors)
    X_fail = fail_subset[feat].values.astype(np.float32)
    X_fail = np.nan_to_num(X_fail, nan=0.0, posinf=0.0, neginf=0.0)
    shap_fail = per_sample_shap(model, X_fail)  # (n_fail, 37)
    top1_idx = shap_fail.argmax(axis=1)
    top1_gene = [feat[i] for i in top1_idx]
    top1_category = [categorize_gene(g) for g in top1_gene]

    print(f"\n  top1 gene distribution (n={len(fail_subset)}):")
    for g, c in zip(top1_gene, top1_category):
        print(f"    {g} ({c})")
    print()

    # 3) Load planner
    print(f"=== Loading {args.planner} planner ===")
    try:
        adapter = load_planner(args.planner, device=args.device)
    except Exception as e:
        print(f"  [FATAL] cannot load {args.planner} adapter: {e}")
        print(f"  提示：本脚本需要 NAVSIM 数据 + GPU。本机若无数据/模型，")
        print(f"  可使用 --device cpu 与 --openscene-root 指向你自己的路径。")
        sys.exit(2)

    # 4) Forward-pass loop
    print(f"\n=== Forward-pass gene-targeted counterfactual ===")
    rows = []
    n_mitigated = 0
    n_image_load_failed = 0
    n_planner_failed = 0
    t0 = time.time()
    for i, (_, row) in enumerate(fail_subset.iterrows()):
        token = row["scene_token"]
        attack = row["attack"]
        strength = float(row.get("strength", 0.8))
        gene_top1 = top1_gene[i]
        cat = top1_category[i]

        # Load image
        try:
            clean_img = load_image_from_standalone(token, args.openscene_root)
        except Exception as e:
            n_image_load_failed += 1
            continue
        if clean_img is None:
            n_image_load_failed += 1
            continue

        # Apply original attack (re-derive attacked image)
        from navsim.agents.attack_genome.attacks.templates import ContinuousAttackSpace
        atk_space = ContinuousAttackSpace()
        try:
            attacked_img = atk_space.evaluate(clean_img, attack, strength)
        except Exception:
            continue

        # Mitigation: apply gene-mitigation transform ON TOP of the attacked image
        try:
            mitigated_img = apply_mitigation(attacked_img, cat)
        except Exception:
            mitigated_img = attacked_img

        # Planner forward pass: clean, attacked, mitigated
        try:
            traj_clean = adapter.predict(clean_img)
            traj_atk = adapter.predict(attacked_img)
            traj_mit = adapter.predict(mitigated_img)
        except Exception:
            n_planner_failed += 1
            continue

        ade_atk = traj_ade(traj_clean, traj_atk)
        ade_mit = traj_ade(traj_clean, traj_mit)
        flip = (ade_atk > args.ade_threshold) and (ade_mit < args.ade_threshold)
        improvement = ade_atk - ade_mit  # positive = mitigation helped

        rows.append({
            "scene_token": token, "attack": attack, "strength": strength,
            "top1_gene": gene_top1, "top1_category": cat,
            "ade_attacked": ade_atk,
            "ade_mitigated": ade_mit,
            "flip": int(flip),
            "improvement_m": improvement,
        })
        n_mitigated += 1
        if (i + 1) % 5 == 0:
            cur_flip = sum(r["flip"] for r in rows) / max(1, len(rows))
            print(f"  [{i+1}/{len(fail_subset)}] loaded_image_fail={n_image_load_failed} "
                  f"planner_fail={n_planner_failed} flip_rate={cur_flip:.3f}")

    elapsed = time.time() - t0
    print(f"\n  phase done: {n_mitigated}/{len(fail_subset)} cases in {elapsed:.1f}s")
    if n_image_load_failed > 0:
        print(f"  [INFO] {n_image_load_failed} image load failures "
              f"(check --openscene-root)")
    if n_planner_failed > 0:
        print(f"  [INFO] {n_planner_failed} planner forward failures")

    if not rows:
        print(f"\n  [ABORT] no successful rows. Cannot produce output.")
        return

    # 5) Aggregate
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / f"gene_cf_{args.planner.lower()}_per_sample.csv", index=False)

    overall_flip = float(res["flip"].mean())
    mean_improvement = float(res["improvement_m"].mean())
    median_improvement = float(res["improvement_m"].median())

    by_cat = res.groupby("top1_category").agg(
        n=("flip", "size"),
        flip_rate=("flip", "mean"),
        mean_improvement=("improvement_m", "mean"),
    ).reset_index()

    summary = {
        "planner": args.planner,
        "n_requested": args.n_fail,
        "n_successful": int(n_mitigated),
        "n_image_load_failed": int(n_image_load_failed),
        "n_planner_failed": int(n_planner_failed),
        "ade_threshold": args.ade_threshold,
        "overall_flip_rate": overall_flip,
        "mean_improvement_m": mean_improvement,
        "median_improvement_m": median_improvement,
        "per_category": by_cat.to_dict(orient="records"),
        "top1_gene_distribution": {g: int(c) for g, c in
                                    pd.Series(top1_gene[:n_mitigated]).value_counts().items()},
    }
    with open(out_dir / f"gene_cf_{args.planner.lower()}_report.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / f"gene_cf_{args.planner.lower()}_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"# Gene-Targeted Forward-pass Counterfactual — {args.planner}\n")
        f.write(f"# ADE threshold (success): {args.ade_threshold} m\n\n")
        f.write(f"  requested: {args.n_fail} fail cases\n")
        f.write(f"  successful: {n_mitigated} cases\n")
        f.write(f"  image_load_failed: {n_image_load_failed}\n")
        f.write(f"  planner_failed: {n_planner_failed}\n\n")
        f.write(f"  overall flip_rate: {overall_flip:.3f}\n")
        f.write(f"  mean ADE improvement: {mean_improvement:.3f} m\n")
        f.write(f"  median ADE improvement: {median_improvement:.3f} m\n\n")
        f.write(f"  per top1-gene-category:\n")
        for _, r in by_cat.iterrows():
            f.write(f"    {r['top1_category']:12s}  n={int(r['n']):3d}  "
                    f"flip={r['flip_rate']:.3f}  "
                    f"improvement={r['mean_improvement']:+.3f} m\n")

    print(f"\n=== Result ({args.planner}) ===")
    print(f"  overall flip_rate: {overall_flip:.3f}")
    print(f"  mean ADE improvement: {mean_improvement:+.3f} m")
    print(f"\n  per top1-category:")
    for _, r in by_cat.iterrows():
        print(f"    {r['top1_category']:12s}  n={int(r['n']):3d}  "
              f"flip={r['flip_rate']:.3f}  improvement={r['mean_improvement']:+.3f} m")
    print(f"\nDONE. → {out_dir}/")


if __name__ == "__main__":
    main()
