"""Run YOLO gene extraction LOCALLY, output CSV for server consumption."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ["OPENSCENE_DATA_ROOT"] = ROOT + "/dataset"

# Use WSL-mounted dataset if available, else local
OPENSCENE = "d:/cogatedrive/dataset"
if not os.path.isdir(OPENSCENE + "/sensor_blobs"):
    print("no local dataset, trying WSL mount...")
    OPENSCENE = "/mnt/e/navsim_workspace/dataset"

import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path
import pandas as pd

from scripts.attack_genome.navsim_loader import NavsimAttackSceneLoader
from navsim.agents.attack_genome.attacks.templates import ContinuousAttackSpace

print(f"Loading YOLOv8n...")
model = YOLO("yolov8n.pt")  # local file in current dir

VEHICLE_CLASSES = {2, 3, 5, 7}
PERSON_CLASSES = {0}
TRAFFIC_CLASSES = {9}

def detect(img):
    results = model(img, verbose=False)
    if len(results) == 0 or results[0].boxes is None:
        return {"vehicle_n": 0, "vehicle_conf": 0.0, "person_n": 0, "person_conf": 0.0,
                "traffic_n": 0, "traffic_conf": 0.0, "total_n": 0, "total_conf": 0.0}
    boxes = results[0].boxes
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    vm = np.isin(cls_ids, list(VEHICLE_CLASSES))
    pm = np.isin(cls_ids, list(PERSON_CLASSES))
    tm = np.isin(cls_ids, list(TRAFFIC_CLASSES))
    return {
        "vehicle_n": int(vm.sum()),
        "vehicle_conf": float(confs[vm].mean()) if vm.any() else 0.0,
        "person_n": int(pm.sum()),
        "person_conf": float(confs[pm].mean()) if pm.any() else 0.0,
        "traffic_n": int(tm.sum()),
        "traffic_conf": float(confs[tm].mean()) if tm.any() else 0.0,
        "total_n": int(len(cls_ids)),
        "total_conf": float(confs.mean()) if len(confs) > 0 else 0.0,
    }

print(f"Loading scenes from {OPENSCENE}...")
loader = NavsimAttackSceneLoader(openscene_root=OPENSCENE, max_tokens=10)
scenes = loader.collect()
print(f"  {len(scenes)} scenes loaded")

attack_space = ContinuousAttackSpace()
results = []
total = len(scenes) * len(attack_space)
pbar = tqdm(total=total, desc="[YOLO local]")

for s in scenes:
    clean_det = detect(s.image)
    for atk_name in attack_space.attack_names:
        for strength in attack_space.strengths:
            attacked = attack_space.evaluate(s.image, atk_name, strength)
            att_det = detect(attacked)
            row = {
                "scene_token": s.scene_token, "attack": atk_name, "strength": strength,
                "clean_vehicle_n": clean_det["vehicle_n"],
                "clean_person_n": clean_det["person_n"],
                "clean_total_n": clean_det["total_n"],
                "att_vehicle_n": att_det["vehicle_n"],
                "att_person_n": att_det["person_n"],
                "att_total_n": att_det["total_n"],
                "vehicle_loss": clean_det["vehicle_n"] - att_det["vehicle_n"],
                "person_loss": clean_det["person_n"] - att_det["person_n"],
                "detection_loss": clean_det["total_n"] - att_det["total_n"],
                "conf_loss": clean_det["total_conf"] - att_det["total_conf"],
                "vehicle_loss_ratio": (clean_det["vehicle_n"] - att_det["vehicle_n"]) / max(1, clean_det["vehicle_n"]),
            }
            results.append(row)
            pbar.update(1)
pbar.close()

df = pd.DataFrame(results)
out = "d:/cogatedrive/exp/yolo_genes_local.csv"
df.to_csv(out, index=False)
print(f"\nSaved {len(df)} rows to {out}")

# Quick summary
print("\n=== Detections on CLEAN images ===")
for i, s in enumerate(scenes[:3]):
    cd = detect(s.image)
    print(f"  Scene {i}: vehicles={cd['vehicle_n']}, persons={cd['person_n']}, total={cd['total_n']}")

print("\n=== Detection loss @ max strength ===")
s1 = df[df["strength"] == 1.0]
for atk in attack_space.attack_names:
    sub = s1[s1["attack"] == atk]
    print(f"  {atk:>15s}: vehicle_loss={sub['vehicle_loss'].mean():.2f}, detection_loss={sub['detection_loss'].mean():.2f}")

print("\nDONE. Upload yolo_genes_local.csv to server.")
