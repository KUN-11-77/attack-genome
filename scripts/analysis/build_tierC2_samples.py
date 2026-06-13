"""Build Tier C2 manifest: select 20 common-failure samples for forward-pass validation.

Inputs:
  - /data3/khsong/exp/attack_genome/tierB_v2/{cnn,dino,tf}/shard{0-4}/per_sample_genes.csv

Output:
  - /data3/khsong/exp/attack_genome/tierC2/manifest.csv (20 rows)
  - /data3/khsong/exp/attack_genome/tierC2/manifest_summary.txt
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/data3/khsong/exp/attack_genome/tierB_v2")
OUT = Path("/data3/khsong/exp/attack_genome/tierC2")
OUT.mkdir(parents=True, exist_ok=True)

PLANNERS = ["cnn", "dino", "tf"]
SHARDS = [0, 1, 2, 3, 4]


def load_planner(planner: str) -> pd.DataFrame:
    """Load all 5 shards for one planner, concatenate."""
    dfs = []
    for shard in SHARDS:
        p = BASE / planner / f"shard{shard}" / "per_sample_genes.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["__shard__"] = shard
            dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def main():
    print("Loading per-planner data...")
    planner_dfs = {}
    for p in PLANNERS:
        d = load_planner(p)
        # Need: scene_token, attack, strength, success (0/1)
        d["planner"] = p
        # Mark failure: success=0 → fail=1
        d["fail"] = 1 - d["success"].astype(int)
        planner_dfs[p] = d
        print(f"  {p}: {len(d)} samples, {d['fail'].sum()} failures "
              f"({d['fail'].mean()*100:.1f}%)")

    # Inner join on (scene_token, attack, strength) to find common samples
    # For 3-planner comparison, we need samples that exist in all 3
    merged = None
    for p in PLANNERS:
        d = planner_dfs[p][["scene_token", "attack", "strength", "fail"]].copy()
        d = d.rename(columns={"fail": f"fail_{p}"})
        if merged is None:
            merged = d
        else:
            merged = merged.merge(d, on=["scene_token", "attack", "strength"],
                                  how="inner")
    print(f"\nInner-joined (all 3 planners): {len(merged)} common samples")

    # Find common failures (all 3 fail)
    common_fail = merged[
        (merged["fail_cnn"] == 1)
        & (merged["fail_dino"] == 1)
        & (merged["fail_tf"] == 1)
    ].copy()
    print(f"Common failures (all 3 fail): {len(common_fail)}")

    if len(common_fail) == 0:
        print("ERROR: No common failures found!")
        sys.exit(1)

    # Sample 20 common-failure scenarios
    # Prefer diverse attacks and scenes
    # Strategy: stratified sample by attack
    n_per_attack = max(1, 20 // common_fail["attack"].nunique())
    sampled = []
    for attack, group in common_fail.groupby("attack"):
        if len(group) <= n_per_attack:
            sampled.append(group)
        else:
            sampled.append(group.sample(n=n_per_attack, random_state=42))
    manifest = pd.concat(sampled, ignore_index=True)
    if len(manifest) > 20:
        manifest = manifest.sample(n=20, random_state=42).reset_index(drop=True)
    else:
        manifest = manifest.reset_index(drop=True)

    # Add counterfactual tier C1 results for each sample
    # (we'll use existing per-sample top-1 driver from XGBoost)
    # For now, we'll find the per-sample top1 from the gene data
    print(f"\nSelected {len(manifest)} samples for Tier C2:")
    print(manifest[["scene_token", "attack", "strength"]].to_string())

    # Save manifest
    manifest.to_csv(OUT / "manifest.csv", index=False)

    # Summary
    with open(OUT / "manifest_summary.txt", "w") as f:
        f.write(f"Tier C2 Manifest\n")
        f.write(f"================\n\n")
        f.write(f"Total common-failure pool: {len(common_fail)}\n")
        f.write(f"Selected: {len(manifest)}\n\n")
        f.write(f"Attack distribution:\n")
        f.write(manifest["attack"].value_counts().to_string())
        f.write(f"\n\nScenes selected: {manifest['scene_token'].nunique()}\n")
        f.write(f"\nSamples (scene, attack, strength):\n")
        for _, row in manifest.iterrows():
            f.write(f"  {row['scene_token']}  {row['attack']}  s={row['strength']}\n")

    print(f"\nManifest saved: {OUT / 'manifest.csv'}")
    print(f"Summary saved: {OUT / 'manifest_summary.txt'}")


if __name__ == "__main__":
    main()
