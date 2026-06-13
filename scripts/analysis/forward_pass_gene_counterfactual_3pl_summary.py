"""3-planner Forward-pass Gene-Targeted Counterfactual 跨架构汇总.

读 gene_cf_{cnn,dino,tf}_report.json，输出对比表 (3-planner 报告).

使用：
  python scripts/analysis/forward_pass_gene_counterfactual_3pl_summary.py \\
      --input-dir exp/tierB_partial/forward_pass_counterfactual
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

PLANNERS = ["CNN", "DINO", "TF"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="exp/tierB_partial/forward_pass_counterfactual")
    p.add_argument("--output", default="exp/tierB_partial/forward_pass_counterfactual/"
                                     "gene_cf_3pl_summary.txt")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, dict] = {}
    for pl in PLANNERS:
        f = in_dir / f"gene_cf_{pl.lower()}_report.json"
        if not f.exists():
            print(f"  [SKIP] {pl}: missing {f}")
            continue
        with open(f, "r", encoding="utf-8") as fp:
            summaries[pl] = json.load(fp)

    if not summaries:
        print("  [FATAL] no per-planner reports found.")
        return

    # 1) overall flip rate
    rows_overall = []
    rows_per_cat = []
    for pl, s in summaries.items():
        rows_overall.append({
            "planner": pl,
            "n_successful": s["n_successful"],
            "n_requested": s["n_requested"],
            "overall_flip_rate": s["overall_flip_rate"],
            "mean_improvement_m": s["mean_improvement_m"],
            "median_improvement_m": s["median_improvement_m"],
        })
        for cat_row in s["per_category"]:
            rows_per_cat.append({
                "planner": pl,
                "category": cat_row["top1_category"],
                "n": int(cat_row["n"]),
                "flip_rate": float(cat_row["flip_rate"]),
                "mean_improvement_m": float(cat_row["mean_improvement"]),
            })

    df_overall = pd.DataFrame(rows_overall)
    df_per_cat = pd.DataFrame(rows_per_cat)

    # Pivot per-category table
    if not df_per_cat.empty:
        pv = df_per_cat.pivot_table(
            index="category", columns="planner",
            values="flip_rate", aggfunc="mean"
        ).round(3)
    else:
        pv = pd.DataFrame()

    # 2) Save CSVs
    df_overall.to_csv(out_path.parent / "gene_cf_3pl_overall.csv", index=False)
    df_per_cat.to_csv(out_path.parent / "gene_cf_3pl_per_category.csv", index=False)
    if not pv.empty:
        pv.to_csv(out_path.parent / "gene_cf_3pl_pivot.csv")

    # 3) Print & write summary
    print("=== 3-planner Gene-Targeted Forward Counterfactual ===\n")
    print("Overall:")
    print(df_overall.to_string(index=False))
    print("\nPer-category flip rate (XGBoost SHAP top-1 → mitigation):")
    if not pv.empty:
        print(pv.to_string())
    else:
        print("  (no per-category data)")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 3-planner Gene-Targeted Forward Counterfactual\n")
        f.write("# (independent of XGBoost — uses real planner forward pass)\n\n")
        f.write("## Overall\n\n")
        f.write(df_overall.to_string(index=False))
        f.write("\n\n## Per top1-gene-category flip rate\n\n")
        if not pv.empty:
            f.write(pv.to_string())
        else:
            f.write("(no per-category data)\n")
        f.write("\n\n## Interpretation\n\n")
        f.write("  - 较高的 overall_flip_rate 说明基因级 mitigation 确实能让真实 planner 恢复。\n")
        f.write("  - per-category 不一致说明不同 gene 类别的因果杠杆强度不同 ——\n")
        f.write("    这是 Failure Law 的执行层证据。\n")
        f.write("  - 若 3-planner 都显著 → Cross-Architecture Failure Law 在执行层也成立。\n")

    print(f"\nDONE. → {out_path}")


if __name__ == "__main__":
    main()
