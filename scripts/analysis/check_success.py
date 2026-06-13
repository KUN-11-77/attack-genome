"""Diagnose tierB_v2 data: what does 'success' mean?"""
import pandas as pd

BASE = "/data3/khsong/exp/attack_genome/tierB_v2"
for p in ["cnn", "dino", "tf"]:
    d = pd.read_csv(f"{BASE}/{p}/shard0/per_sample_genes.csv")
    s0 = d[d["strength"] == 0.0]
    s1 = d[d["success"] == 1]
    print(f"{p}: total={len(d)}  success=0:{int((d['success']==0).sum())}  success=1:{int((d['success']==1).sum())}")
    print(f"  s=0.0 success rate: {s0['success'].mean():.3f}")
    if "ade" in d.columns:
        print(f"  ADE: success=0 -> {d[d['success']==0]['ade'].mean():.3f}, success=1 -> {d[d['success']==1]['ade'].mean():.3f}")
    else:
        print(f"  no ADE column, available cols: {[c for c in d.columns if 'ad' in c.lower() or 'fail' in c.lower() or 'succ' in c.lower()][:5]}")
