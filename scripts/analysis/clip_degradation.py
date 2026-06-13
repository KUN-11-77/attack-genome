"""CLIP feature degradation analysis — 4th representation.

For the SAME 50 tokens as Tier C++, extract CLIP (ViT-B-32) features
from clean and attacked images. Compute cosine similarity degradation.
This gives us a 4th fundamentally different supervision signal
(language-supervised) alongside CNN (ImageNet), DINO (self-supervised), TF (hybrid).

Output: clip_degradation.csv — one row per (scene, attack, strength)
with cosine_sim and degradation_score.
"""
import sys, os
sys.path.insert(0, "/data3/khsong/cogatedrive")
os.environ.setdefault("NAVSIM_EXP_ROOT", "/data3/khsong/cogatedrive/exp")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/nas/datasets/navsim")

import numpy as np
import torch
import open_clip
from tqdm import tqdm
from pathlib import Path

from scripts.attack_genome.navsim_loader import NavsimAttackSceneLoader
from navsim.agents.attack_genome.attacks.templates import ContinuousAttackSpace

# Load CLIP ViT-B-32 from local checkpoint (server offline)
print("Loading CLIP ViT-B-32 from local checkpoint...")
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="/data3/khsong/data/navsim/models/clip_vitb32.pth")
model = model.cuda().eval()

# Use same scene loader as Tier C++
print("Loading scenes...")
loader = NavsimAttackSceneLoader(openscene_root="/nas/datasets/navsim", max_tokens=50)
scenes = loader.collect()
print(f"  {len(scenes)} scenes loaded")

attack_space = ContinuousAttackSpace()

@torch.no_grad()
def clip_feature(image: np.ndarray) -> np.ndarray:
    """Extract CLIP image feature for an (H,W,3) uint8 image."""
    from PIL import Image
    pil = Image.fromarray(image)
    img_tensor = preprocess(pil).unsqueeze(0).cuda()
    feat = model.encode_image(img_tensor)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy()

results = []
print("Extracting CLIP features for all (scene, attack, strength)...")
total = len(scenes) * len(attack_space)
pbar = tqdm(total=total, desc="[CLIP]")

for s in scenes:
    clean_feat = clip_feature(s.image)
    for atk_name in attack_space.attack_names:
        for strength in attack_space.strengths:
            attacked = attack_space.evaluate(s.image, atk_name, strength)
            attacked_feat = clip_feature(attacked)
            cos_sim = float(np.dot(clean_feat, attacked_feat))
            results.append({
                "scene_token": s.scene_token,
                "attack": atk_name,
                "strength": strength,
                "cosine_sim": cos_sim,
                "degradation": 1.0 - cos_sim,
            })
            pbar.update(1)
pbar.close()

# Save
out_dir = Path("/data3/khsong/cogatedrive/exp/clip_analysis")
out_dir.mkdir(parents=True, exist_ok=True)

import pandas as pd
df = pd.DataFrame(results)
df.to_csv(out_dir / "clip_degradation.csv", index=False)
print(f"Saved {len(df)} rows to clip_degradation.csv")

# Summary: per-attack mean degradation at max strength
summary = df[df["strength"] == 1.0].groupby("attack")["degradation"].mean().sort_values(ascending=False)
print("\nCLIP feature degradation @ max strength (s=1.0):")
for atk, val in summary.items():
    print(f"  {atk:>15s}: {val:.4f}")

# Compare with planner ASR: compute CLIP "virtual ASR" (degradation > threshold)
thresholds = [0.05, 0.10, 0.15, 0.20]
print("\nCLIP virtual ASR (fraction of samples with degradation > threshold):")
for t in thresholds:
    clip_asr = (df["degradation"] > t).mean()
    print(f"  threshold={t:.2f}: virtual_ASR={clip_asr:.3f}")

print("\nDONE. CLIP analysis complete.")
