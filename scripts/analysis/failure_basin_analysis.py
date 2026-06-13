"""Failure Basin Analysis — Module A 的独立验证。

严格定义: 不做 "counterfactual"，做"empirical failure basin structure"。

3 个测试:
    Test 1 — Monotonicity: planner 在 strength 上升时是否单调退化？
        期望: "success success success fail fail fail" 模式占比 > 80%
    Test 2 — Critical Strength: 第一次进入 fail 的 strength
        期望: CNN < DINO < TF (中位数)
    Test 3 — Failure Basin Width: 1.0 - critical_strength
        期望: CNN > DINO > TF (中位数)

输入: exp/tierB_partial/merged_3pl.csv
输出: exp/tierB_partial/failure_basin/figure{1,2,3}.png
      exp/tierB_partial/failure_basin/summary.json
      exp/tierB_partial/failure_basin/per_case.csv
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def is_monotonic(success_seq: List[int]) -> bool:
    """strength↑ 时, success (1=fail, 0=survive) 是否单调不减。

    注: merged_3pl 中 success=1 = 攻击成功 = planner fail
        所以 strength 越大 → success 应该越向 1 偏 (单调不减)
    """
    for i in range(len(success_seq) - 1):
        if success_seq[i] > success_seq[i + 1]:
            return False
    return True


def is_strict_monotonic(success_seq: List[int]) -> bool:
    """完全 0,0,0,1,1,1 形式 (无来回跳)。"""
    if not is_monotonic(success_seq):
        return False
    # 至少有一次 0→1 跳变才算"边界存在"
    has_jump = any(success_seq[i] != success_seq[i + 1] for i in range(len(success_seq) - 1))
    return has_jump


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="exp/tierB_partial/merged_3pl.csv")
    p.add_argument("--output-dir", default="exp/tierB_partial/failure_basin")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Failure Basin Analysis ===")
    print(f"  CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  loaded {len(df)} rows, {df['scene_token'].nunique()} scenes, "
          f"{df['planner'].nunique()} planners")

    # 1) 构建 (scene, attack, planner) → [(strength, success), ...]
    rows: List[dict] = []
    for (scene, atk, pl), g in df.groupby(["scene_token", "attack", "planner"]):
        g = g.sort_values("strength")
        strengths = g["strength"].values.tolist()
        successes = g["success"].values.astype(int).tolist()
        if len(successes) < 2:
            continue
        # critical_strength = 第一次 success==1 (fail) 的 strength
        cs = None
        for s, sc in zip(strengths, successes):
            if sc == 1:
                cs = s
                break
        # 如果全部 success==0 (从未 fail), 视为 cs > 1.0 (out of range)
        if cs is None:
            cs = max(strengths) + 0.1  # 标记为 "never failed within range"
        rows.append({
            "scene_token": scene,
            "attack": atk,
            "planner": pl,
            "n_steps": len(successes),
            "monotonic": is_monotonic(successes),
            "strict_monotonic": is_strict_monotonic(successes),
            "critical_strength": cs,
            "width": 1.0 - min(cs, 1.0),  # 1.0 - critical_strength, clip to [0,1]
            "fail_at_max": successes[-1] == 1,
            "success_at_min": successes[0] == 0,
        })
    case_df = pd.DataFrame(rows)
    case_df.to_csv(out_dir / "per_case.csv", index=False)
    print(f"  total (scene, attack, planner) cases: {len(case_df)}")

    # ============== Test 1: Monotonicity ==============
    print(f"\n=== Test 1: Monotonicity ===")
    mono_stats = {}
    for pl in ["CNN", "DINO", "TF"]:
        sub = case_df[case_df["planner"] == pl]
        mono_rate = sub["monotonic"].mean()
        strict_rate = sub["strict_monotonic"].mean()
        mono_stats[pl] = {"monotonic": float(mono_rate), "strict_monotonic": float(strict_rate)}
        print(f"  {pl}: monotonic={mono_rate:.3f}  strict={strict_rate:.3f}")
    overall_mono = case_df["monotonic"].mean()
    overall_strict = case_df["strict_monotonic"].mean()
    print(f"  ALL:  monotonic={overall_mono:.3f}  strict={overall_strict:.3f}")

    # ============== Test 2 & 3: Critical Strength & Width ==============
    # 只保留"实际发生 fail" (critical_strength <= 1.0)
    fail_df = case_df[case_df["fail_at_max"]].copy()
    print(f"\n=== Test 2 & 3: Critical Strength & Width (n={len(fail_df)}) ===")
    crit_stats = {}
    width_stats = {}
    for pl in ["CNN", "DINO", "TF"]:
        sub = fail_df[fail_df["planner"] == pl]
        crit_med = float(sub["critical_strength"].median())
        crit_mean = float(sub["critical_strength"].mean())
        width_med = float(sub["width"].median())
        width_mean = float(sub["width"].mean())
        crit_stats[pl] = {"median": crit_med, "mean": crit_mean, "n": int(len(sub))}
        width_stats[pl] = {"median": width_med, "mean": width_mean, "n": int(len(sub))}
        print(f"  {pl}: critical_strength median={crit_med:.3f} mean={crit_mean:.3f}; "
              f"width median={width_med:.3f} mean={width_mean:.3f}")

    # 检查 CNN < DINO < TF 鲁棒性层级
    crit_meds = [crit_stats[p]["median"] for p in ["CNN", "DINO", "TF"]]
    width_meds = [width_stats[p]["median"] for p in ["CNN", "DINO", "TF"]]
    crit_order_correct = crit_meds[0] < crit_meds[1] < crit_meds[2]
    width_order_correct = width_meds[0] > width_meds[1] > width_meds[2]
    print(f"\n  CNN<DINO<TF (critical)  : {crit_order_correct}  → {crit_meds}")
    print(f"  CNN>DINO>TF (width)     : {width_order_correct}  → {width_meds}")

    # ============== 画图 ==============
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"CNN": "#d62728", "DINO": "#1f77b4", "TF": "#2ca02c"}

    # Figure 1: Monotonicity bar
    fig, ax = plt.subplots(figsize=(6, 4.2))
    pls = ["CNN", "DINO", "TF"]
    mono_vals = [mono_stats[p]["monotonic"] * 100 for p in pls]
    strict_vals = [mono_stats[p]["strict_monotonic"] * 100 for p in pls]
    x = np.arange(len(pls))
    w = 0.35
    ax.bar(x - w/2, mono_vals, w, label="monotonic (0→0→...→1)", color=[colors[p] for p in pls], alpha=0.7)
    ax.bar(x + w/2, strict_vals, w, label="strict (exactly one 0→1 jump)", color=[colors[p] for p in pls], alpha=0.95)
    ax.axhline(80, color="gray", linestyle="--", alpha=0.5, label="80% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(pls, fontsize=12)
    ax.set_ylabel("% of (scene, attack) cases", fontsize=11)
    ax.set_title("Test 1: Monotonicity of Planner Degradation", fontsize=12)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 105)
    for i, p in enumerate(pls):
        ax.text(i - w/2, mono_vals[i] + 1, f"{mono_vals[i]:.1f}%", ha="center", fontsize=9)
        ax.text(i + w/2, strict_vals[i] + 1, f"{strict_vals[i]:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "figure1_monotonicity.png", dpi=140)
    plt.close(fig)
    print(f"\n  saved → {out_dir}/figure1_monotonicity.png")

    # Figure 2: Critical Strength distribution (violin + box)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    data_per_pl = [fail_df[fail_df["planner"] == p]["critical_strength"].values for p in pls]
    parts = ax.violinplot(data_per_pl, positions=range(len(pls)), showmedians=True, widths=0.7)
    for pc, p in zip(parts["bodies"], pls):
        pc.set_facecolor(colors[p])
        pc.set_alpha(0.5)
    ax.boxplot(data_per_pl, positions=range(len(pls)), widths=0.15,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor="white", edgecolor="black"),
               medianprops=dict(color="black", linewidth=1.5))
    for i, p in enumerate(pls):
        med = crit_stats[p]["median"]
        ax.scatter([i], [med], marker="D", s=80, color=colors[p], edgecolor="black", zorder=5, label=f"median={med:.2f}" if i == 0 else None)
        ax.text(i, 0.04, f"{med:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(pls)))
    ax.set_xticklabels(pls, fontsize=12)
    ax.set_ylabel("Critical Strength (first strength causing fail)", fontsize=11)
    ax.set_title("Test 2: Failure Boundary per Planner\n(lower = planner fails earlier)", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figure2_critical_strength.png", dpi=140)
    plt.close(fig)
    print(f"  saved → {out_dir}/figure2_critical_strength.png")

    # Figure 3: Failure Basin Width distribution
    fig, ax = plt.subplots(figsize=(6, 4.5))
    data_per_pl_w = [fail_df[fail_df["planner"] == p]["width"].values for p in pls]
    parts = ax.violinplot(data_per_pl_w, positions=range(len(pls)), showmedians=True, widths=0.7)
    for pc, p in zip(parts["bodies"], pls):
        pc.set_facecolor(colors[p])
        pc.set_alpha(0.5)
    ax.boxplot(data_per_pl_w, positions=range(len(pls)), widths=0.15,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor="white", edgecolor="black"),
               medianprops=dict(color="black", linewidth=1.5))
    for i, p in enumerate(pls):
        med = width_stats[p]["median"]
        ax.text(i, 0.04, f"{med:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(pls)))
    ax.set_xticklabels(pls, fontsize=12)
    ax.set_ylabel("Failure Basin Width (= 1.0 - critical_strength)", fontsize=11)
    ax.set_title("Test 3: Failure Basin Width per Planner\n(higher = planner is in failure for longer)", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "figure3_basin_width.png", dpi=140)
    plt.close(fig)
    print(f"  saved → {out_dir}/figure3_basin_width.png")

    # ============== 写 summary ==============
    summary = {
        "n_cases": int(len(case_df)),
        "n_fail_cases": int(len(fail_df)),
        "test1_monotonicity": {
            "overall_monotonic": float(overall_mono),
            "overall_strict": float(overall_strict),
            "per_planner": mono_stats,
            "expectation": "monotonic > 0.80 (proves Failure Basin exists)",
            "passed": bool(overall_mono > 0.80),
        },
        "test2_critical_strength": {
            "per_planner": crit_stats,
            "expected_order": "CNN < DINO < TF",
            "order_correct": bool(crit_order_correct),
        },
        "test3_basin_width": {
            "per_planner": width_stats,
            "expected_order": "CNN > DINO > TF",
            "order_correct": bool(width_order_correct),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 人类可读
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Failure Basin Analysis — Summary\n")
        f.write("================================\n\n")
        f.write(f"Total cases (scene, attack, planner): {len(case_df)}\n")
        f.write(f"Cases that fail at any strength: {len(fail_df)}\n\n")
        f.write(f"Test 1 — Monotonicity (expect > 80%):\n")
        for pl in pls:
            f.write(f"  {pl}: monotonic={mono_stats[pl]['monotonic']:.3f}  strict={mono_stats[pl]['strict_monotonic']:.3f}\n")
        f.write(f"  → overall: {overall_mono:.3f}  {'✓' if overall_mono > 0.80 else '✗'}\n\n")
        f.write(f"Test 2 — Critical Strength (expect CNN<DINO<TF):\n")
        for pl in pls:
            f.write(f"  {pl}: median={crit_stats[pl]['median']:.3f}\n")
        f.write(f"  → order correct: {crit_order_correct}  {crit_meds}\n\n")
        f.write(f"Test 3 — Basin Width (expect CNN>DINO>TF):\n")
        for pl in pls:
            f.write(f"  {pl}: median={width_stats[pl]['median']:.3f}\n")
        f.write(f"  → order correct: {width_order_correct}  {width_meds}\n")

    print(f"\n  saved → {out_dir}/summary.{{json,txt}}")
    print(f"\nDONE.")


if __name__ == "__main__":
    main()
