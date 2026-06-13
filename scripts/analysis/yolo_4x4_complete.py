"""YOLO 4×4 矩阵补全：DigitalNoise + Dusk/Dawn。

之前的 server 跑只输出了 Rain / MotionBlur / CarlaStyle / Snow，缺：
  - DigitalNoise (噪声像素破坏)
  - Dusk/Dawn (颜色偏移)

补全后形成 4×4 矩阵：
  - 4 attack 类型 × 6 strength × 50 scene × YOLO 推理

在服务器上跑（GPU 0, 单 GPU 足够，因为 YOLO 只有 200×50 = 10000 次推理）。
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.attack_genome.attacks.templates import ContinuousAttackSpace


# 4 attack types（用户设计：噪声 / 模糊 / 雨线 / 颜色偏移）
ATTACKS_4 = ["DigitalNoise", "MotionBlur", "Rain", "Dusk"]
# 注：Dusk/Dawn 一起跑（都属于颜色偏移），所以是 4 类（4 个独立 axis）
STRENGTHS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_SCENES = 50  # mini split default


def load_navigation_scenes(n: int = 50) -> List[Dict]:
    """从 mini NAVSIM 加载 N 个场景的 front camera 图像。"""
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.planning.training.dataset import NavsimDataset

    # 用 NAVSIM_DEVKIT_ROOT 解析路径
    sensor_config = SensorConfig.build_all_sensors(
        use_navsim_poses=True, with_bev=False
    )
    scene_filter = SceneFilter.random_split(
        num_scenes=n, seed=42
    )
    scene_loader = SceneLoader(
        openscene_root=os.environ.get("OPENSCENE_DATA_ROOT", "/nas/datasets/navsim"),
        data_root=os.environ.get("NAVSIM_DEVKIT_ROOT", str(REPO_ROOT)),
        split="mini",
        sensor_config=sensor_config,
        scene_filter=scene_filter,
    )
    scenes = list(scene_loader)
    return scenes


def yolo_inference(model: YOLO, image: np.ndarray) -> Dict:
    """单图 YOLO → {n_objects, mean_conf, n_vehicles, n_persons, n_traffic_lights}。"""
    results = model(image, verbose=False)[0]
    n_total = len(results.boxes)
    if n_total == 0:
        return {"n_objects": 0, "mean_conf": 0.0, "n_vehicles": 0, "n_persons": 0, "n_traffic_lights": 0}
    cls = results.boxes.cls.cpu().numpy().astype(int)
    conf = results.boxes.conf.cpu().numpy()
    # COCO classes: 2=car, 0=person, 9=traffic light
    return {
        "n_objects": int(n_total),
        "mean_conf": float(conf.mean()),
        "n_vehicles": int((cls == 2).sum()),
        "n_persons": int((cls == 0).sum()),
        "n_traffic_lights": int((cls == 9).sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--n-scenes", type=int, default=N_SCENES)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model", default="yolov8n.pt")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== YOLO 4×4 completion ===")
    print(f"  attacks: {ATTACKS_4}")
    print(f"  strengths: {STRENGTHS}")
    print(f"  n_scenes: {args.n_scenes}")
    print(f"  device: {args.device}")
    print(f"  output: {out_dir}")

    # 加载 YOLO
    print(f"\n[1/3] loading YOLO model {args.model} ...")
    model = YOLO(args.model)
    if args.device:
        model.to(args.device)

    # 加载场景
    print(f"\n[2/3] loading {args.n_scenes} NAVSIM scenes ...")
    scenes = load_navigation_scenes(args.n_scenes)
    print(f"  loaded {len(scenes)} scenes")

    # 攻击空间
    attack_space = ContinuousAttackSpace(
        attack_names=ATTACKS_4, strengths=STRENGTHS
    )

    # 推理
    print(f"\n[3/3] running 4×6×{len(scenes)} = {4*6*len(scenes)} YOLO inferences ...")
    rows = []
    t0 = time.time()
    n_done = 0
    total = 4 * 6 * len(scenes)
    for s_i, scene in enumerate(scenes):
        # 取 front camera
        agent_input = scene.get_agent_input()
        image = agent_input.cameras[-1].cam_f0.image  # (H, W, 3) uint8
        if image is None or image.size == 0:
            print(f"  scene {s_i}: empty image, skip")
            continue
        # clean baseline
        clean_stats = yolo_inference(model, image)
        for attack in ATTACKS_4:
            for s in STRENGTHS:
                attacked = attack_space.evaluate(image, attack, s)
                atk_stats = yolo_inference(model, attacked)
                # 写一行 loss
                rows.append({
                    "scene_token": scene.scene_token,
                    "attack": attack,
                    "strength": s,
                    "n_objects_clean": clean_stats["n_objects"],
                    "n_objects_attacked": atk_stats["n_objects"],
                    "n_vehicles_clean": clean_stats["n_vehicles"],
                    "n_vehicles_attacked": atk_stats["n_vehicles"],
                    "n_persons_clean": clean_stats["n_persons"],
                    "n_persons_attacked": atk_stats["n_persons"],
                    "conf_clean": clean_stats["mean_conf"],
                    "conf_attacked": atk_stats["mean_conf"],
                    "detection_loss": clean_stats["n_objects"] - atk_stats["n_objects"],
                    "vehicle_loss": clean_stats["n_vehicles"] - atk_stats["n_vehicles"],
                    "person_loss": clean_stats["n_persons"] - atk_stats["n_persons"],
                    "conf_loss": clean_stats["mean_conf"] - atk_stats["mean_conf"],
                    "vehicle_loss_ratio": (clean_stats["n_vehicles"] - atk_stats["n_vehicles"]) / max(clean_stats["n_vehicles"], 1),
                })
                n_done += 1
                if n_done % 100 == 0:
                    elapsed = time.time() - t0
                    speed = n_done / elapsed
                    eta = (total - n_done) / speed if speed > 0 else float("inf")
                    eta_str = f"{int(eta//60)}m{int(eta%60):02d}s" if eta != float("inf") else "?"
                    print(f"  [{n_done}/{total}] {100*n_done/total:.1f}% elapsed={int(elapsed//60)}m{int(elapsed%60):02d}s eta={eta_str}", flush=True)
    print(f"\n  done {n_done} in {int((time.time()-t0)//60)}m{int((time.time()-t0)%60):02d}s")

    # 落盘
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = out_dir / "yolo_4x4.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  saved → {csv_path}")
    # 重建 yolo_genes.csv（per_sample_genes 的格式），给 genome_pipeline.py 加载
    # 这里我们存 raw 矩阵，yolo_gene_extractor.py 后续负责 gene 计算
    yolo_gene_csv = out_dir / "yolo_raw_stats.csv"
    df.to_csv(yolo_gene_csv, index=False)
    print(f"  saved → {yolo_gene_csv}")

    # 简要 summary
    summary = df.groupby(["attack", "strength"])["detection_loss"].agg(["mean", "max"]).reset_index()
    print(f"\n=== Summary (per attack × strength) ===")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "yolo_4x4_summary.csv", index=False)


if __name__ == "__main__":
    main()
