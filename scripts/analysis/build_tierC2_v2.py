"""Build Tier C2 manifest v2: select 20 common-failure samples (strength >= 0.2).

Filters out s=0.0 samples (those are baseline failures, not attack-induced).
"""
import os
import sys
import pandas as pd
from pathlib import Path

BASE = Path("/data3/khsong/exp/attack_genome/tierB_v2")
OUT = Path("/data3/khsong/exp/attack_genome/tierC2")
OUT.mkdir(parents=True, exist_ok=True)

PLANNERS = ["cnn", "dino", "tf"]
SHARDS = [0, 1, 2, 3, 4]


def load_planner(planner: str) -> pd.DataFrame:
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
        d["planner"] = p
        d["fail"] = (d["success"] == 0).astype(int)
        planner_dfs[p] = d
        fail_s_attack = d[d["strength"] >= 0.2]["fail"].mean()
        print(f"  {p}: {len(d)} samples, all-strength fail={d['fail'].mean()*100:.1f}%, "
              f"attack-strength (s>=0.2) fail={fail_s_attack*100:.1f}%")

    # Inner join on (scene_token, attack, strength)
    merged = None
    for p in PLANNERS:
        d = planner_dfs[p][["scene_token", "attack", "strength", "fail"]].copy()
        d = d.rename(columns={"fail": f"fail_{p}"})
        if merged is None:
            merged = d
        else:
            merged = merged.merge(d, on=["scene_token", "attack", "strength"], how="inner")
    print(f"\nInner-joined (all 3 planners): {len(merged)} common samples")

    # Common failures (all 3 fail)
    common_fail = merged[
        (merged["fail_cnn"] == 1)
        & (merged["fail_dino"] == 1)
        & (merged["fail_tf"] == 1)
    ].copy()
    print(f"Common failures (all 3 fail, all strengths): {len(common_fail)}")

    # Filter to attack-induced only (s >= 0.2)
    common_fail_attack = common_fail[common_fail["strength"] >= 0.2].copy()
    print(f"Common failures (s >= 0.2): {len(common_fail_attack)}")

    if len(common_fail_attack) == 0:
        print("ERROR: No common attack-induced failures found!")
        sys.exit(1)

    # Stratified sample by attack, 2 per attack → 20 total (10 attacks)
    n_per_attack = 2
    sampled = []
    for attack, group in common_fail_attack.groupby("attack"):
        if len(group) <= n_per_attack:
            sampled.append(group)
        else:
            sampled.append(group.sample(n=n_per_attack, random_state=42))
    manifest = pd.concat(sampled, ignore_index=True)
    if len(manifest) > 20:
        manifest = manifest.sample(n=20, random_state=42).reset_index(drop=True)
    else:
        manifest = manifest.reset_index(drop=True)

    print(f"\nSelected {len(manifest)} samples for Tier C2:")
    print(manifest[["scene_token", "attack", "strength"]].to_string())

    manifest.to_csv(OUT / "manifest.csv", index=False)
    with open(OUT / "manifest_summary.txt", "w") as f:
        f.write(f"Tier C2 Manifest v2 (s >= 0.2)\n")
        f.write(f"================================\n\n")
        f.write(f"Total common-failure pool (s >= 0.2): {len(common_fail_attack)}\n")
        f.write(f"Selected: {len(manifest)}\n\n")
        f.write(f"Attack distribution:\n{manifest['attack'].value_counts().to_string()}\n\n")
        f.write(f"Strength distribution:\n{manifest['strength'].value_counts().sort_index().to_string()}\n\n")
        f.write(f"Samples (scene, attack, strength):\n")
        for _, row in manifest.iterrows():
            f.write(f"  {row['scene_token']}  {row['attack']:14s}  s={row['strength']}\n")

    print(f"\nManifest: {OUT / 'manifest.csv'}")


if __name__ == "__main__":
    main()
