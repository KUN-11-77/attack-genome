"""Forward-pass 反事实 (本地版) — 在真 planner 上验证 gene-causality。

关键修改：绕开 nuplan，使用 standalone pkl loader；
         使用 local E:\\navsim_workspace 19,478 scenes 自建数据。

设计：
    1. 从 19,478 池取 N scenes
    2. 对每个 scene: 加载 clean image, 跑 planner → traj_clean
    3. 取 3 个攻击 (rain, dusk, digital_noise) × 2 强度 (high, low):
       4. high (s=0.8): 期望大多数 fail → "fail pool"
       5. low (s=0.4): 期望大多数 success → "clean baseline"
    6. 对 fail cases, 反事实 scan 5 个 strength [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
       跑 planner at each → 找 s_c (critical strength, planner 翻成 success)
    7. 报:
       - baseline ASR @ s=0.8 (top-level fail rate)
       - flip_rate at each strength reduction
       - s_c distribution
       - 对比 XGBoost counterfactual (之前在 88,560 上 K=10 flip 86%)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

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


def build_index(openscene_root: str):
    from scripts.analysis.standalone_scene_loader import StandaloneSceneIndex
    print(f"\n=== Building scene index from {openscene_root} ===")
    t0 = time.time()
    idx = StandaloneSceneIndex(openscene_root)
    print(f"  done in {time.time() - t0:.1f}s, total scenes: {len(idx.scene_to_loc)}")
    return idx


def traj_ade(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
    if traj_a is None or traj_b is None:
        return float("nan")
    return float(np.linalg.norm(traj_a[:, :2] - traj_b[:, :2], axis=1).mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--openscene-root", default="E:/navsim_workspace/dataset")
    p.add_argument("--n-scenes", type=int, default=30)
    p.add_argument("--planner", default="CNN", choices=["CNN", "DINO", "TF"])
    p.add_argument("--output-dir", default="exp/tierB_partial/forward_pass_counterfactual")
    p.add_argument("--ade-threshold", type=float, default=2.0)
    p.add_argument("--attacks", nargs="+", default=["rain", "dusk", "digital_noise"])
    p.add_argument("--fail-strength", type=float, default=0.8)
    p.add_argument("--scan-strengths", nargs="+", type=float,
                   default=[0.0, 0.2, 0.4, 0.6, 0.8])
    args = p.parse_args()

    print(f"=== Forward-pass Counterfactual (LOCAL, {args.planner}) ===")
    print(f"  openscene_root = {args.openscene_root}")
    print(f"  n_scenes = {args.n_scenes}, attacks = {args.attacks}")
    print(f"  fail_strength = {args.fail_strength}, ADE_threshold = {args.ade_threshold} m")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Index
    idx = build_index(args.openscene_root)
    all_scenes = list(idx.scene_to_loc.keys())
    rng = np.random.RandomState(42)
    rng.shuffle(all_scenes)
    selected = all_scenes[:args.n_scenes]
    print(f"\n  selected {len(selected)} scenes")

    # 2) Load planner
    print(f"\n=== Loading {args.planner} planner ===")
    from scripts.attack_genome.adapters import (
        build_cnn_adapter_from_yaml,
        build_dino_adapter_from_yaml,
        build_transfuser_adapter_from_yaml,
    )
    factory = {"CNN": build_cnn_adapter_from_yaml,
               "DINO": build_dino_adapter_from_yaml,
               "TF": build_transfuser_adapter_from_yaml}[args.planner]
    t0 = time.time()
    adapter = factory(device="cuda")
    print(f"  loaded {args.planner} in {time.time() - t0:.1f}s")

    # 3) Attack space
    from navsim.agents.attack_genome.attacks.templates import ContinuousAttackSpace
    attack_space = ContinuousAttackSpace()

    # 4) Forward pass loop
    print(f"\n=== Forward-pass loop ===")
    rows: List[dict] = []
    fail_cases: List[dict] = []  # for counterfactual scan
    t_total = time.time()
    n_loaded = 0
    n_failed_at_high = 0
    for si, token in enumerate(selected):
        clean_img = idx.load_image(token)
        if clean_img is None:
            continue
        try:
            traj_clean = adapter.predict(clean_img)
        except Exception as e:
            print(f"  [skip] {token}: planner error: {e}")
            continue
        n_loaded += 1
        for atk in args.attacks:
            attacked = attack_space.evaluate(clean_img, atk, args.fail_strength)
            try:
                traj_atk = adapter.predict(attacked)
            except Exception:
                continue
            ade = traj_ade(traj_clean, traj_atk)
            is_fail = ade > args.ade_threshold
            if is_fail:
                n_failed_at_high += 1
                fail_cases.append({
                    "scene_token": token, "attack": atk, "ade_high": ade,
                })
            rows.append({
                "scene_token": token, "attack": atk,
                "strength": args.fail_strength,
                "ade": ade, "planner_fail": int(is_fail),
            })
        if (si + 1) % 5 == 0:
            print(f"  [{si + 1}/{len(selected)}] loaded={n_loaded} fail@{args.fail_strength}={n_failed_at_high}")

    print(f"\n  phase 1: {n_loaded} scenes loaded, {n_failed_at_high} fail cases")
    print(f"  elapsed: {time.time() - t_total:.1f}s")

    # 5) Counterfactual scan on fail cases
    print(f"\n=== Counterfactual scan on {len(fail_cases)} fail cases ===")
    cf_rows: List[dict] = []
    t_cf = time.time()
    for ci, fc in enumerate(fail_cases):
        token = fc["scene_token"]
        atk = fc["attack"]
        clean_img = idx.load_image(token)
        if clean_img is None:
            continue
        traj_clean = adapter.predict(clean_img)
        s_c_found = None
        ade_at_s_c = None
        for s in args.scan_strengths:
            attacked = attack_space.evaluate(clean_img, atk, s)
            try:
                traj_s = adapter.predict(attacked)
            except Exception:
                continue
            ade = traj_ade(traj_clean, traj_s)
            is_success = ade < args.ade_threshold
            cf_rows.append({
                "scene_token": token, "attack": atk,
                "strength": s, "ade": ade, "planner_success": int(is_success),
                "s_c_found": s_c_found,
            })
            if is_success and s_c_found is None:
                s_c_found = s
                ade_at_s_c = ade
        cf_rows.extend([])  # placeholder
        if (ci + 1) % 10 == 0:
            print(f"  [{ci + 1}/{len(fail_cases)}] s_c_found {sum(1 for r in cf_rows if r['s_c_found'] is not None)}")

    print(f"  phase 2 elapsed: {time.time() - t_cf:.1f}s")

    # 6) Aggregate
    if not cf_rows:
        print("\n  [ABORT] no counterfactual rows")
        return
    res = pd.DataFrame(cf_rows)
    res.to_csv(out_dir / f"forward_pass_{args.planner.lower()}_per_sample.csv", index=False)

    by_strength = res.groupby("strength")["planner_success"].mean()
    n_total_cases = len(fail_cases)
    n_flipped = 0
    n_critical_only = 0
    for fc in fail_cases:
        token = fc["scene_token"]; atk = fc["attack"]
        case_rows = res[(res["scene_token"] == token) & (res["attack"] == atk)]
        if (case_rows["planner_success"] == 1).any():
            n_flipped += 1
        if case_rows["planner_success"].sum() > 0:
            n_critical_only += 1

    summary = {
        "planner": args.planner,
        "n_scenes_loaded": n_loaded,
        "n_fail_cases_at_high": len(fail_cases),
        "n_flipped_at_some_lower_strength": n_flipped,
        "flip_rate": n_flipped / max(1, n_total_cases),
        "per_strength_flip": {float(s): float(r) for s, r in by_strength.items()},
        "ade_threshold": args.ade_threshold,
        "fail_strength": args.fail_strength,
    }
    with open(out_dir / f"forward_pass_{args.planner.lower()}_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / f"forward_pass_{args.planner.lower()}_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"# Forward-pass Counterfactual — {args.planner} (LOCAL)\n")
        f.write(f"# 攻击强度扫描：在 fail 样本上降强度，看 planner 何时翻成 success\n\n")
        f.write(f"  scenes loaded: {n_loaded}\n")
        f.write(f"  fail cases @ s={args.fail_strength}: {len(fail_cases)}\n")
        f.write(f"  ADE threshold (success 判定): {args.ade_threshold} m\n\n")
        f.write(f"  per-strength flip_rate (cf → 翻成 success 的比例):\n")
        for s, r in by_strength.items():
            f.write(f"    s={s:.1f}: {r:.3f}\n")
        f.write(f"\n  flip_rate (any lower strength): {n_flipped / max(1, n_total_cases):.3f}\n")
        f.write(f"\n  → 真实 planner 是否被攻击强度调控，验证 XGBoost counterfactual\n")
        f.write(f"  详见: forward_pass_{args.planner.lower()}_per_sample.csv\n")

    print(f"\n=== Result ===")
    for s, r in by_strength.items():
        print(f"  s={s:.1f}: flip_rate = {r:.3f}")
    print(f"  overall: {n_flipped}/{n_total_cases} = {n_flipped/max(1,n_total_cases):.3f}")
    print(f"\nDONE. → {out_dir}/")


if __name__ == "__main__":
    main()
