"""生成合成 per_sample_genes.csv 用于本地测试 RF/XGBoost/DBSCAN/画图 pipeline。

4 个 planner × 10 attack × 6 strength × 50 scene = 12,000 行
模拟真实攻击模式：
  - DigitalNoise/MotionBlur: CNN > DINO 鲁棒 (CNN 偏回归，trained on noise)
  - Rain: DINO > CNN (ViT 对结构信息更敏感)
  - Dusk/Dawn: 所有 planner 都较鲁棒 (color only)
  - VintageStyle: 所有 planner 都易失败 (style transfer)
"""
import numpy as np
import pandas as pd
import os
from pathlib import Path

OUT_DIR = Path("/tmp/ag_synth")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "per_sample_genes.csv"

np.random.seed(42)
n_scenes = 50
attacks = ["Rain", "Snow", "Dusk", "Dawn", "MotionBlur", "DigitalNoise",
           "LightDust", "DappledLight", "VintageStyle", "CarlaStyle"]
strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
planners = ["CNN-GTRS", "DINO-GTRS", "TransFuser", "ReCogDrive-VLM"]

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

# 每个 attack 对每个 planner 的 failure curve (sigmoid 中心 + 陡度)
ATTACK_PROFILE = {
    # (planner -> (s_c, k))
    "Rain":         {"CNN-GTRS": (0.45, 6), "DINO-GTRS": (0.55, 7), "TransFuser": (0.40, 5), "ReCogDrive-VLM": (0.60, 6)},
    "Snow":         {"CNN-GTRS": (0.55, 5), "DINO-GTRS": (0.50, 6), "TransFuser": (0.45, 5), "ReCogDrive-VLM": (0.65, 7)},
    "Dusk":         {"CNN-GTRS": (0.80, 8), "DINO-GTRS": (0.75, 9), "TransFuser": (0.85, 9), "ReCogDrive-VLM": (0.70, 7)},
    "Dawn":         {"CNN-GTRS": (0.85, 9), "DINO-GTRS": (0.80, 8), "TransFuser": (0.85, 9), "ReCogDrive-VLM": (0.75, 7)},
    "MotionBlur":   {"CNN-GTRS": (0.40, 7), "DINO-GTRS": (0.50, 6), "TransFuser": (0.35, 6), "ReCogDrive-VLM": (0.55, 7)},
    "DigitalNoise": {"CNN-GTRS": (0.35, 6), "DINO-GTRS": (0.45, 7), "TransFuser": (0.30, 5), "ReCogDrive-VLM": (0.50, 7)},
    "LightDust":    {"CNN-GTRS": (0.65, 6), "DINO-GTRS": (0.60, 7), "TransFuser": (0.55, 5), "ReCogDrive-VLM": (0.70, 7)},
    "DappledLight": {"CNN-GTRS": (0.60, 6), "DINO-GTRS": (0.55, 7), "TransFuser": (0.50, 5), "ReCogDrive-VLM": (0.65, 7)},
    "VintageStyle": {"CNN-GTRS": (0.30, 5), "DINO-GTRS": (0.35, 6), "TransFuser": (0.25, 4), "ReCogDrive-VLM": (0.40, 6)},
    "CarlaStyle":   {"CNN-GTRS": (0.50, 5), "DINO-GTRS": (0.45, 6), "TransFuser": (0.40, 4), "ReCogDrive-VLM": (0.55, 6)},
}

rows = []
for scene_i in range(n_scenes):
    scene_token = f"synth_{scene_i:04d}"
    # scene attribute (one-hot)
    attrs = {
        "night": float(np.random.random() < 0.3),
        "curve": float(np.random.random() < 0.4),
        "high_traffic": float(np.random.random() < 0.5),
        "low_light": float(np.random.random() < 0.4),
        "road_occlusion": float(np.random.random() < 0.2),
    }
    for attack in attacks:
        for s in strengths:
            # base gene values
            gene_vals = {g: float(np.random.randn() * 0.3 + 0.4) for g in GENE_FIELDS}
            # attack-specific gene modulation
            if attack == "Rain":
                gene_vals["edge_density"] -= s * 0.5
                gene_vals["lane_line_density"] -= s * 0.6
                gene_vals["rms_contrast"] -= s * 0.3
            elif attack == "MotionBlur":
                gene_vals["high_freq_ratio"] -= s * 0.7
                gene_vals["edge_density"] -= s * 0.5
            elif attack == "DigitalNoise":
                gene_vals["lbp_entropy"] += s * 0.4
                gene_vals["std_luma"] += s * 0.5
            elif attack in ("Dusk", "Dawn"):
                gene_vals["hue_mean"] += 0.4 if attack == "Dusk" else -0.3
                gene_vals["mean_luma"] += s * 0.3
            elif attack == "VintageStyle":
                gene_vals["sat_mean"] -= s * 0.4
                gene_vals["hue_mean"] += 0.2
            for p in planners:
                s_c, k = ATTACK_PROFILE[attack][p]
                # logistic failure prob
                p_fail = 1.0 / (1.0 + np.exp(-k * (s - s_c)))
                # scene noise
                scene_noise = np.random.normal(0, 0.05)
                success = float(np.random.random() > p_fail + scene_noise * 0.1)
                row = {
                    "scene_token": scene_token, "attack": attack,
                    "strength": round(s, 4), "planner": p,
                    "success": int(success),
                }
                row.update(gene_vals)
                row.update(attrs)
                rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)
print(f"wrote {len(df)} rows → {OUT_PATH}")
print(f"  scenes: {df.scene_token.nunique()}")
print(f"  attacks: {df.attack.nunique()}")
print(f"  planners: {df.planner.unique().tolist()}")
print(f"  failure rate per planner:")
print(df.groupby("planner")["success"].mean().to_string())
print(f"\n  failure rate per attack:")
print(df.groupby("attack")["success"].mean().sort_values().to_string())
