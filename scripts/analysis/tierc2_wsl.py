"""Tier C2 WSL — 用本机 1883 navdream scenes + CNN forward pass 验证 gene causality。

优势:
    - 预计算 11 个攻击 jpg（不用 on-the-fly apply attack）
    - WSL constanteye_sec env 已有 nuplan/navsim/torch 全通
    - 本机 RTX 5060 GPU

实验流程:
    1. 训练 XGBoost (gene→fail) on 88,560 Tier B 样本 (CNN only)
    2. 在 1883 navdream scenes 中随机抽 20 个
    3. 每个 scene: 加载 clean + pre-attacked image
    4. 对 attacked image 算 37-dim gene vector
    5. XGBoost 预测 fail + 算 per-sample SHAP top-1
    6. 对 top-1 gene 做 image-level 干预 (mitigation)
    7. 跑 CNN forward pass on clean / attacked / mitigated → 3 个轨迹
    8. 算 ADE: fail = ADE(attacked) > threshold, flip = ADE(mitigated) < threshold
    9. 报 flip_rate

用法:
    /opt/conda/envs/constanteye_sec/bin/python scripts/analysis/tierc2_wsl.py \\
        --n 20 --planner CNN \\
        --csv /mnt/d/cogatedrive/exp/tierB_partial/merged_3pl.csv \\
        --navdream-root /mnt/e/navsim_workspace/dataset/navdream_benchmark_outputs \\
        --output-dir /mnt/d/cogatedrive/exp/tierC2_wsl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path("/mnt/d/cogatedrive")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# 37 gene fields (from forward_pass_gene_counterfactual.py)
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
    if gene in EDGE_FREQ_GENES: return "edge_freq"
    if gene in LUMA_GENES: return "luma"
    if gene in COLOR_GENES: return "color"
    if gene in DETECTION_GENES: return "detection"
    return "other"


def _ensure_uint8(img):
    if img.dtype != np.uint8:
        return np.clip(img, 0, 255).astype(np.uint8)
    return img


def mitigate_edge_freq(img, sigma=1.2):
    ksize = int(2 * round(2 * sigma) + 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)


def mitigate_luma(img, gamma=0.85):
    inv_gamma = 1.0 / max(gamma, 1e-3)
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, lut)


def mitigate_color(img, sat_scale=0.6):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= sat_scale
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def mitigate_detection(img, alpha=1.15):
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    return cv2.addWeighted(img, 1 + alpha, blurred, -alpha, 0).astype(np.uint8)


def apply_mitigation(img, category):
    if category == "edge_freq": return mitigate_edge_freq(img)
    if category == "luma":      return mitigate_luma(img)
    if category == "color":     return mitigate_color(img)
    if category == "detection": return mitigate_detection(img)
    return mitigate_edge_freq(img, sigma=0.6)


def traj_ade(a, b):
    if a is None or b is None: return float("nan")
    return float(np.linalg.norm(a[:, :2] - b[:, :2], axis=1).mean())


def load_planner(name, device="cuda"):
    from scripts.attack_genome.adapters import (
        build_cnn_adapter_from_yaml,
        build_dino_adapter_from_yaml,
        build_transfuser_adapter_from_yaml,
    )
    factory = {"CNN": build_cnn_adapter_from_yaml,
               "DINO": build_dino_adapter_from_yaml,
               "TF": build_transfuser_adapter_from_yaml}[name]
    return factory(device=device)


def train_xgb_one_fold(df, planner, seed=42):
    """Train one XGBoost fold for gene→fail prediction (per-sample SHAP ready)."""
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
    best = (None, None, None, 0.0, None)
    for tr, te in gkf.split(X, y, ss):
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, eval_metric="logloss",
        )
        m.fit(X[tr], y[tr])
        try:
            auc = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
        except ValueError:
            continue
        if auc > best[3]:
            best = (m, tr, te, auc, feat)
    return best[0], best[4]


def per_sample_shap(model, X):
    import xgboost as xgb
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    try:
        contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    except Exception:
        return np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
    return contribs[:, :-1].astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--planner", default="CNN", choices=["CNN", "DINO", "TF"])
    p.add_argument("--attack", default="Rain",
                   help="Pre-applied attack to use. One of: Rain, Snow, Dusk, Dawn, MotionBlur, "
                        "DigitalNoise, LightDust, DappledLight, VintageStyle, CarlaStyle")
    p.add_argument("--csv", default="/mnt/d/cogatedrive/exp/tierB_partial/merged_3pl.csv")
    p.add_argument("--navdream-root", default="/mnt/e/navsim_workspace/dataset/navdream_benchmark_outputs")
    p.add_argument("--navdream-cache", default="/tmp/navdream_idx.pkl")
    p.add_argument("--output-dir", default="/mnt/d/cogatedrive/exp/tierC2_wsl")
    p.add_argument("--ade-threshold", type=float, default=2.0)
    p.add_argument("--k", type=int, default=1,
                   help="Number of top genes to mitigate (sequential composition)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"=== Tier C2 WSL ({args.planner}) ===")
    print(f"  n={args.n}, attack={args.attack}, csv={args.csv}")
    print(f"  navdream_root={args.navdream_root}")
    print(f"  output_dir={args.output_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Train XGBoost (one fold) for SHAP
    print(f"\n=== Loading CSV + training XGBoost ({args.planner}) ===")
    df = pd.read_csv(args.csv)
    print(f"  rows: {len(df)}")
    model, feat = train_xgb_one_fold(df, args.planner, seed=args.seed)
    print(f"  XGBoost trained, {len(feat)} features")

    # 2) Load navdream index
    from scripts.analysis.navdream_scene_loader import NavDreamIndex
    idx = NavDreamIndex(args.navdream_root, cache_path=args.navdream_cache)
    print(f"  navdream index: {len(idx.scene_to_loc)} scenes")

    # 3) Load planner
    print(f"\n=== Loading {args.planner} planner ===")
    adapter = load_planner(args.planner, device=args.device)
    print(f"  adapter loaded: {type(adapter).__name__}")

    # 4) Pick 20 scenes that have the requested pre-applied attack
    print(f"\n=== Sampling {args.n} scenes with pre-applied {args.attack} ===")
    rng = np.random.RandomState(args.seed)
    candidate_tokens = []
    for token, info in idx.scene_to_loc.items():
        # the loader uses ATTACK_NAME_TO_FILE
        from scripts.analysis.navdream_scene_loader import ATTACK_NAME_TO_FILE
        file_key = ATTACK_NAME_TO_FILE.get(args.attack)
        if not file_key:
            print(f"  [ERR] unknown attack: {args.attack}")
            return
        atk_file = info["scene_dir"] / "F0" / f"{file_key}.jpg"
        if atk_file.exists():
            candidate_tokens.append(token)
    print(f"  candidate scenes with {args.attack}: {len(candidate_tokens)}")
    if len(candidate_tokens) < args.n:
        print(f"  [WARN] not enough candidates, using all {len(candidate_tokens)}")
        n = len(candidate_tokens)
    else:
        n = args.n
    sampled = rng.choice(candidate_tokens, size=n, replace=False)

    # 5) Forward-pass loop
    print(f"\n=== Forward-pass loop ({n} samples) ===")
    rows = []
    t0 = time.time()
    for i, token in enumerate(sampled):
        print(f"\n  [{i+1}/{n}] token={token[:8]}...")
        clean = idx.load_image(token, attack=None)
        atk = idx.load_image(token, attack=args.attack)
        if clean is None or atk is None:
            print(f"    [skip] image load fail (clean={clean is not None}, atk={atk is not None})")
            continue

        # Compute gene vector of attacked image
        from navsim.agents.attack_genome.genes.genome_pipeline import AttackGenomeExtractor
        extractor = AttackGenomeExtractor()
        rec = extractor(atk)  # __call__ method
        gene_dict = rec.features
        # XGBoost input: feat vector in feat order, with META (strength) at end
        x_vec = np.array([[float(gene_dict.get(f, 0.0)) for f in GENE_FIELDS] +
                          [0.8]], dtype=np.float32)  # strength=0.8 for fixed pre-applied
        x_vec = np.nan_to_num(x_vec, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict + SHAP
        prob = float(model.predict_proba(x_vec)[:, 1][0])
        shap = per_sample_shap(model, x_vec)[0]  # (37+1,)
        # top-1 gene: use GENE_FIELDS only (ignore strength index 37)
        shap_genes = shap[:len(GENE_FIELDS)]
        top1_idx = int(np.argmax(shap_genes))
        top1_gene = GENE_FIELDS[top1_idx]
        cat = categorize_gene(top1_gene)
        print(f"    gene top1: {top1_gene} (cat={cat}); XGBoost prob: {prob:.3f}")

        # Apply top-K gene mitigations (deduplicated by category)
        # IMPORTANT: dedupe categories — applying same category mitigation N times
        # destroys the image (e.g. 3x blur for 3 edge_freq genes)
        try:
            mit = atk.copy()
            top_k_idx = np.argsort(-shap_genes)[:args.k]
            top_k_genes = [GENE_FIELDS[int(i)] for i in top_k_idx]
            top_k_cats = sorted(set([categorize_gene(g) for g in top_k_genes]))
            for c in top_k_cats:
                mit = apply_mitigation(mit, c)
        except Exception as e:
            print(f"    [skip] mitigation failed: {e}")
            continue

        # Forward pass: clean / attacked / mitigated
        try:
            traj_clean = adapter.predict(clean)
            traj_atk = adapter.predict(atk)
            traj_mit = adapter.predict(mit)
        except Exception as e:
            print(f"    [skip] planner fail: {e}")
            continue

        ade_atk = traj_ade(traj_clean, traj_atk)
        ade_mit = traj_ade(traj_clean, traj_mit)
        is_fail = ade_atk > args.ade_threshold
        is_flip = is_fail and (ade_mit < args.ade_threshold)
        improvement = ade_atk - ade_mit

        rows.append({
            "scene_token": token, "attack": args.attack, "k": args.k,
            "xgb_prob": prob, "top1_gene": top1_gene, "top1_category": cat,
            "top_k_genes": "|".join(top_k_genes),
            "ade_attacked": ade_atk, "ade_mitigated": ade_mit,
            "is_fail": int(is_fail), "is_flip": int(is_flip),
            "improvement_m": improvement,
        })
        print(f"    K={args.k} top: {top_k_genes}")
        print(f"    ADE atk={ade_atk:.2f}, mit={ade_mit:.2f}, is_fail={int(is_fail)}, "
              f"is_flip={int(is_flip)}, improvement={improvement:+.2f}m")

    elapsed = time.time() - t0
    print(f"\n  phase done: {len(rows)}/{n} in {elapsed:.1f}s")

    if not rows:
        print("\n[ABORT] no successful rows")
        return

    # 6) Aggregate
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / f"tierc2_{args.planner.lower()}_{args.attack.lower()}_k{args.k}_per_sample.csv", index=False)

    n_total = len(res)
    n_fail = int(res["is_fail"].sum())
    n_flip = int(res["is_flip"].sum())
    flip_rate_overall = n_flip / n_total
    flip_rate_over_fail = (n_flip / n_fail) if n_fail > 0 else float("nan")
    mean_imp = float(res["improvement_m"].mean())

    by_cat = res.groupby("top1_category").agg(
        n=("is_flip", "size"),
        n_flip=("is_flip", "sum"),
        flip_rate=("is_flip", "mean"),
        mean_improvement=("improvement_m", "mean"),
    ).reset_index()

    summary = {
        "planner": args.planner,
        "attack": args.attack,
        "k": args.k,
        "n_requested": args.n,
        "n_successful": n_total,
        "n_fail": n_fail,
        "n_flip": n_flip,
        "ade_threshold": args.ade_threshold,
        "flip_rate_overall": flip_rate_overall,
        "flip_rate_over_fail": flip_rate_over_fail,
        "mean_improvement_m": mean_imp,
        "median_improvement_m": float(res["improvement_m"].median()),
        "per_category": by_cat.to_dict(orient="records"),
    }
    with open(out_dir / f"tierc2_{args.planner.lower()}_{args.attack.lower()}_k{args.k}_report.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Result ({args.planner}, {args.attack}, K={args.k}) ===")
    print(f"  n_successful: {n_total}")
    print(f"  n_fail (ADE>thr): {n_fail}")
    print(f"  n_flip: {n_flip}")
    print(f"  flip_rate (over total): {flip_rate_overall:.3f}")
    print(f"  flip_rate (over fail): {flip_rate_over_fail:.3f}")
    print(f"  mean ADE improvement: {mean_imp:+.3f} m")
    print(f"  median ADE improvement: {float(res['improvement_m'].median()):+.3f} m")
    print(f"\n  per top1-category:")
    for _, r in by_cat.iterrows():
        print(f"    {r['top1_category']:12s}  n={int(r['n']):3d}  flip={int(r['n_flip']):2d}/{int(r['n']):2d}  rate={r['flip_rate']:.3f}  improvement={r['mean_improvement']:+.3f} m")
    print(f"\nDONE. → {out_dir}/")


if __name__ == "__main__":
    main()
