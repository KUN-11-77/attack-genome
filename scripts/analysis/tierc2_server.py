"""Tier C2 server wrapper — 用 server 的 navsim/benchmark + manifest 跑 20-sample planner forward。

用法:
    /opt/anaconda3/bin/python scripts/analysis/tierc2_server.py \\
        --planner CNN --n-fail 20 \\
        --benchmark-root /data3/khsong/data/navsim/benchmark \\
        --manifest /data3/khsong/exp/attack_genome/tierC2/manifest.csv \\
        --output-dir /data3/khsong/exp/tierC2_local
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

REPO_ROOT = Path("/data3/khsong/cogatedrive")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

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


# Gene-mitigation (reuse from forward_pass_gene_counterfactual.py)
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
    try:
        import cv2
        ksize = int(2 * round(2 * sigma) + 1)
        return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)
    except ImportError:
        from scipy.ndimage import gaussian_filter
        out = np.zeros_like(img)
        for c in range(img.shape[-1]):
            out[..., c] = gaussian_filter(img[..., c].astype(np.float32), sigma=sigma)
        return _ensure_uint8(out)


def mitigate_luma(img, gamma=0.85):
    inv_gamma = 1.0 / max(gamma, 1e-3)
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    import cv2
    return cv2.LUT(img, lut)


def mitigate_color(img, sat_scale=0.6):
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= sat_scale
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def mitigate_detection(img, alpha=1.15):
    import cv2
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(img, 1 + alpha, blurred, -alpha, 0)
    return _ensure_uint8(sharpened)


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--planner", default="CNN", choices=["CNN", "DINO", "TF"])
    p.add_argument("--n-fail", type=int, default=20)
    p.add_argument("--benchmark-root", default="/data3/khsong/data/navsim/benchmark")
    p.add_argument("--manifest", default="/data3/khsong/exp/attack_genome/tierC2/manifest.csv")
    p.add_argument("--output-dir", default="/data3/khsong/exp/tierC2_local")
    p.add_argument("--ade-threshold", type=float, default=2.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-manifest", action="store_true",
                   help="If set, use --manifest to pick samples (overrides --n-fail random pick)")
    args = p.parse_args()

    print(f"=== Tier C2 server wrapper ({args.planner}) ===")
    print(f"  benchmark_root={args.benchmark_root}")
    print(f"  manifest={args.manifest}, n_fail={args.n_fail}")
    print(f"  output_dir={args.output_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load manifest or fallback to random from merged_3pl.csv
    if args.use_manifest and Path(args.manifest).exists():
        manifest = pd.read_csv(args.manifest)
        # Filter to common-fail (fail_cnn for CNN planner)
        col = f"fail_{args.planner.lower()}"
        if col not in manifest.columns:
            col = "fail_cnn"
        manifest = manifest[manifest[col] == 1].reset_index(drop=True)
        samples = manifest.head(args.n_fail)
        print(f"  manifest: {len(manifest)} common-fail pool, picked {len(samples)}")
    else:
        csv = REPO_ROOT / "exp" / "tierB_partial" / "merged_3pl.csv"
        df = pd.read_csv(csv)
        df_pl = df[(df["planner"] == args.planner) & (df["success"] == 0)]
        samples = df_pl.sample(n=min(args.n_fail, len(df_pl)), random_state=args.seed)
        print(f"  random from csv: {len(samples)} samples")

    # 2) Build loader
    from scripts.analysis.navsim_benchmark_loader import NavsimBenchmarkIndex
    idx = NavsimBenchmarkIndex(args.benchmark_root,
                               cache_path="/tmp/navsim_bench_idx.pkl")

    # 3) Load planner
    print(f"\n=== Loading {args.planner} planner ===")
    try:
        adapter = load_planner(args.planner, device=args.device)
        print(f"  loaded {args.planner} planner")
    except Exception as e:
        print(f"  [FATAL] planner load failed: {e}")
        sys.exit(2)

    # 4) Forward pass loop
    print(f"\n=== Forward-pass loop ===")
    rows = []
    n_loaded, n_atk_failed, n_planner_failed, n_mitigated = 0, 0, 0, 0
    t0 = time.time()
    for i, row in samples.iterrows():
        token = row["scene_token"]
        attack = row["attack"]
        strength = float(row.get("strength", 0.8))
        if "top1_gene" in row and isinstance(row["top1_gene"], str):
            top1_gene = row["top1_gene"]
        else:
            top1_gene = "edge_mean"
        cat = categorize_gene(top1_gene)
        print(f"\n  [{i+1}/{len(samples)}] token={token[:8]}... attack={attack} s={strength} top1={top1_gene}({cat})")

        clean_img = idx.load_clean_image(token)
        if clean_img is None:
            print(f"    [skip] image load fail")
            n_loaded += 1
            continue
        n_loaded += 1
        try:
            attacked_img = idx.apply_attack(clean_img, attack, strength)
        except Exception as e:
            print(f"    [skip] attack failed: {e}")
            n_atk_failed += 1
            continue
        try:
            mitigated_img = apply_mitigation(attacked_img, cat)
        except Exception:
            mitigated_img = attacked_img
        try:
            traj_clean = adapter.predict(clean_img)
            traj_atk = adapter.predict(attacked_img)
            traj_mit = adapter.predict(mitigated_img)
        except Exception as e:
            print(f"    [skip] planner fail: {e}")
            n_planner_failed += 1
            continue
        n_mitigated += 1

        ade_atk = traj_ade(traj_clean, traj_atk)
        ade_mit = traj_ade(traj_clean, traj_mit)
        flip = (ade_atk > args.ade_threshold) and (ade_mit < args.ade_threshold)
        improvement = ade_atk - ade_mit

        rows.append({
            "scene_token": token, "attack": attack, "strength": strength,
            "top1_gene": top1_gene, "top1_category": cat,
            "ade_attacked": ade_atk, "ade_mitigated": ade_mit,
            "flip": int(flip), "improvement_m": improvement,
        })
        print(f"    ADE atk={ade_atk:.2f}, mit={ade_mit:.2f}, flip={int(flip)}, improvement={improvement:+.2f}m")

    elapsed = time.time() - t0
    print(f"\n  phase done: {n_mitigated}/{len(samples)} cases in {elapsed:.1f}s")
    print(f"  image_loaded: {n_loaded}, attack_failed: {n_atk_failed}, planner_failed: {n_planner_failed}")

    if not rows:
        print("\n[ABORT] no successful rows")
        return

    # 5) Aggregate
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / f"tierc2_{args.planner.lower()}_per_sample.csv", index=False)

    overall_flip = float(res["flip"].mean())
    mean_imp = float(res["improvement_m"].mean())
    median_imp = float(res["improvement_m"].median())

    by_cat = res.groupby("top1_category").agg(
        n=("flip", "size"), flip_rate=("flip", "mean"),
        mean_improvement=("improvement_m", "mean"),
    ).reset_index()

    summary = {
        "planner": args.planner,
        "n_requested": len(samples),
        "n_successful": int(n_mitigated),
        "n_image_load_failed": int(n_loaded - n_mitigated - n_planner_failed),
        "n_planner_failed": int(n_planner_failed),
        "ade_threshold": args.ade_threshold,
        "overall_flip_rate": overall_flip,
        "mean_improvement_m": mean_imp,
        "median_improvement_m": median_imp,
        "per_category": by_cat.to_dict(orient="records"),
    }
    with open(out_dir / f"tierc2_{args.planner.lower()}_report.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Result ({args.planner}) ===")
    print(f"  overall flip_rate: {overall_flip:.3f}")
    print(f"  mean ADE improvement: {mean_imp:+.3f} m")
    print(f"  median ADE improvement: {median_imp:+.3f} m\n  per top1-category:")
    for _, r in by_cat.iterrows():
        print(f"    {r['top1_category']:12s}  n={int(r['n']):3d}  flip={r['flip_rate']:.3f}  improvement={r['mean_improvement']:+.3f} m")
    print(f"\nDONE. → {out_dir}/")


if __name__ == "__main__":
    main()
