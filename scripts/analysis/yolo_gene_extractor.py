"""YOLOv8-based semantic gene extractor.
Replaces heuristic proxies (saturation bands) with real object detection.
Extracts: vehicle/pedestrian/traffic_light counts and confidence scores
for all 49 scenes x 10 attacks x 6 strengths.
"""
import sys, os
sys.path.insert(0, "/data3/khsong/cogatedrive")
os.environ.setdefault("NAVSIM_EXP_ROOT", "/data3/khsong/cogatedrive/exp")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/nas/datasets/navsim")

import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

from scripts.attack_genome.navsim_loader import NavsimAttackSceneLoader
from navsim.agents.attack_genome.attacks.templates import ContinuousAttackSpace

print("Loading YOLOv8n from local file...")
model = YOLO("/data3/khsong/data/navsim/models/yolov8n.pt")
model.to("cuda")

# COCO class mapping
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
PERSON_CLASSES = {0}  # person
TRAFFIC_CLASSES = {9}  # traffic light

print("Loading scenes...")
loader = NavsimAttackSceneLoader(openscene_root="/nas/datasets/navsim", max_tokens=50)
scenes = loader.collect()
print(f"  {len(scenes)} scenes loaded")

attack_space = ContinuousAttackSpace()

def detect(img):
    """Run YOLO on a single image, return per-class counts and mean conf."""
    results = model(img, verbose=False)
    if len(results) == 0 or results[0].boxes is None:
        return {"vehicle_n": 0, "vehicle_conf": 0.0,
                "person_n": 0, "person_conf": 0.0,
                "traffic_n": 0, "traffic_conf": 0.0,
                "total_n": 0, "total_conf": 0.0}
    boxes = results[0].boxes
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    vehicle_mask = np.isin(cls_ids, list(VEHICLE_CLASSES))
    person_mask = np.isin(cls_ids, list(PERSON_CLASSES))
    traffic_mask = np.isin(cls_ids, list(TRAFFIC_CLASSES))
    return {
        "vehicle_n": int(vehicle_mask.sum()),
        "vehicle_conf": float(confs[vehicle_mask].mean()) if vehicle_mask.any() else 0.0,
        "person_n": int(person_mask.sum()),
        "person_conf": float(confs[person_mask].mean()) if person_mask.any() else 0.0,
        "traffic_n": int(traffic_mask.sum()),
        "traffic_conf": float(confs[traffic_mask].mean()) if traffic_mask.any() else 0.0,
        "total_n": int(len(cls_ids)),
        "total_conf": float(confs.mean()) if len(confs) > 0 else 0.0,
    }

results = []
print("Running YOLO on all (scene, attack, strength)...")
total = len(scenes) * len(attack_space)
pbar = tqdm(total=total, desc="[YOLO]")

for s in scenes:
    # Detect clean first
    clean_det = detect(s.image)
    for atk_name in attack_space.attack_names:
        for strength in attack_space.strengths:
            attacked = attack_space.evaluate(s.image, atk_name, strength)
            att_det = detect(attacked)
            row = {
                "scene_token": s.scene_token,
                "attack": atk_name,
                "strength": strength,
                # Clean counts
                "clean_vehicle_n": clean_det["vehicle_n"],
                "clean_person_n": clean_det["person_n"],
                "clean_traffic_n": clean_det["traffic_n"],
                "clean_total_n": clean_det["total_n"],
                "clean_total_conf": clean_det["total_conf"],
                # Attacked counts
                "att_vehicle_n": att_det["vehicle_n"],
                "att_person_n": att_det["person_n"],
                "att_traffic_n": att_det["traffic_n"],
                "att_total_n": att_det["total_n"],
                "att_total_conf": att_det["total_conf"],
                # Degradation metrics (clean - attacked, positive = lost detections)
                "vehicle_loss": clean_det["vehicle_n"] - att_det["vehicle_n"],
                "person_loss": clean_det["person_n"] - att_det["person_n"],
                "traffic_loss": clean_det["traffic_n"] - att_det["traffic_n"],
                "total_loss": clean_det["total_n"] - att_det["total_n"],
                "conf_loss": clean_det["total_conf"] - att_det["total_conf"],
                # Relative loss (normalized by clean count)
                "vehicle_loss_ratio": (clean_det["vehicle_n"] - att_det["vehicle_n"]) / max(1, clean_det["vehicle_n"]),
                "person_loss_ratio": (clean_det["person_n"] - att_det["person_n"]) / max(1, clean_det["person_n"]),
                "total_loss_ratio": (clean_det["total_n"] - att_det["total_n"]) / max(1, clean_det["total_n"]),
            }
            results.append(row)
            pbar.update(1)
pbar.close()

import pandas as pd
out_dir = Path("/data3/khsong/cogatedrive/exp/yolo_genes")
out_dir.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(out_dir / "yolo_genes.csv", index=False)
print(f"Saved {len(df)} rows to yolo_genes.csv")

# Summary: per-attack mean detection loss @ max strength
summary = df[df["strength"] == 1.0].groupby("attack")[
    ["vehicle_loss", "person_loss", "total_loss", "conf_loss"]
].mean().sort_values("total_loss", ascending=False)
print("\nYOLO detection loss @ max strength (s=1.0):")
print(summary.round(3))

# How often does YOLO lose any detection?
loss_rate = (df["total_loss"] > 0).mean()
print(f"\nSamples with ANY detection loss: {loss_rate:.1%}")
# By attack at max strength
print("\nDetection loss rate per attack @ s=1.0:")
for atk_name in attack_space.attack_names:
    sub = df[(df["attack"] == atk_name) & (df["strength"] == 1.0)]
    rate = (sub["total_loss"] > 0).mean()
    print(f"  {atk_name:>15s}: {rate:.1%}  (avg loss: {sub['total_loss'].mean():.2f} objects)")

print("\nDONE. YOLO semantic genes extracted.")
