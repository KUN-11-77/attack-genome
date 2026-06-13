"""从 NAVSIM 抽 1 张真实 perception 图 (clear) + 1 张 (rain) 供 demo 用。"""
import sys
from pathlib import Path
sys.path.insert(0, "d:/cogatedrive")
from scripts.analysis.standalone_scene_loader import StandaloneSceneIndex
import numpy as np
import cv2

OUT_DIR = Path("d:/cogatedrive/comp/figures/perception")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Building scene index ...")
idx = StandaloneSceneIndex("E:/navsim_workspace/dataset")
print(f"  {len(idx.scene_to_loc)} scenes")

# 取 scene 058c2eb31a41564cb (test split, 第 1 个) 作为 clear
scene_clear = list(idx.scene_to_loc.keys())[0]
print(f"\nclear scene: {scene_clear}")
img_clear = idx.load_image(scene_clear)
print(f"  shape: {img_clear.shape}  mean: {img_clear.mean():.1f}")
cv2.imwrite(str(OUT_DIR / "scene_clear.jpg"), cv2.cvtColor(img_clear, cv2.COLOR_RGB2BGR))
print(f"  saved → {OUT_DIR / 'scene_clear.jpg'}")

# 用 attack template 加 rain 制作 rain 版本 (注意 attack 名字大写, input uint8)
from navsim.agents.attack_genome.attacks.templates import get_attack_template
rain = get_attack_template("Rain")
img_rain = rain(img_clear, 0.7)
print(f"\nrain scene: {scene_clear} @ s=0.7")
print(f"  shape: {img_rain.shape}  mean: {img_rain.mean():.1f}")
cv2.imwrite(str(OUT_DIR / "scene_rain_s07.jpg"), cv2.cvtColor(img_rain, cv2.COLOR_RGB2BGR))
print(f"  saved → {OUT_DIR / 'scene_rain_s07.jpg'}")

# 再做一张 dusk (亮度大幅下降)
dusk = get_attack_template("Dusk")
img_dusk = dusk(img_clear, 0.8)
print(f"\ndusk scene: {scene_clear} @ s=0.8")
print(f"  shape: {img_dusk.shape}  mean: {img_dusk.mean():.1f}")
cv2.imwrite(str(OUT_DIR / "scene_dusk_s08.jpg"), cv2.cvtColor(img_dusk, cv2.COLOR_RGB2BGR))
print(f"  saved → {OUT_DIR / 'scene_dusk_s08.jpg'}")

print("\nall 3 perception images saved")
