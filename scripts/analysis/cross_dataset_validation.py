"""Cross-Dataset Validation (Tier C-1 跨数据集验证).

两种模式：
  --mode cross-scene   用 NAVSIM 同一数据集的不同 scene 子集 (如 test vs trainval)
                      作为 "cross-dataset" 代理。数据已就绪，无需新下载。
  --mode external      读 --image-dir 提供的任意外部图像目录（如 nuScenes mini
                      解码后的 camera frames），抽取 37-dim gene 后用
                      NAVSIM 训练的 XGBoost 预测 fail 概率。

设计目的：
  - 验证 §4.3 Cross-Architecture Failure Law 是否在跨数据集 / 跨 split 仍然成立
  - 验证 37-dim gene space 是否是"数据集无关的图像描述子"

输出：
  exp/tierB_partial/cross_dataset/<mode>_results.json
  exp/tierB_partial/cross_dataset/<mode>_summary.txt

使用：
  # 跨 scene 验证（默认 NAVSIM test split）
  python scripts/analysis/cross_dataset_validation.py --mode cross-scene \\
      --csv exp/tierB_partial/merged_3pl.csv

  # 跨数据集（外部图像目录）
  python scripts/analysis/cross_dataset_validation.py --mode external \\
      --image-dir datasets/navsim_mini/ \\
      --planner CNN --n-images 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
META_FIELDS = ["strength"]


# ---------------------------------------------------------------------------
# Gene extraction (从原始图像 → 37 维)
# ---------------------------------------------------------------------------

def extract_37d_gene(image: np.ndarray, yolo_model=None) -> Dict[str, float]:
    """从单张图像提取 37 维基因。

    实现与 scripts/attack_genome/extract_genome.py 兼容。
    如果 extract_genome 模块不可用，使用本地简化实现。
    """
    try:
        from scripts.attack_genome.extract_genome import extract_gene_from_image
        g = extract_gene_from_image(image, yolo_model=yolo_model)
        return g
    except Exception:
        pass
    return _extract_gene_local(image, yolo_model)


def _extract_gene_local(image: np.ndarray, yolo_model=None) -> Dict[str, float]:
    """本地 37-dim gene 提取（与 extract_genome.py 接口对齐）。"""
    out: Dict[str, float] = {f: 0.0 for f in GENE_FIELDS}
    try:
        import cv2
        if image.shape[-1] == 3 and image.dtype == np.uint8:
            bgr = image
        else:
            bgr = np.clip(image, 0, 255).astype(np.uint8)
            if bgr.shape[-1] == 4:
                bgr = bgr[..., :3]

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Frequency
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        r_norm = r / r.max()
        low = (r_norm < 0.1).sum()
        mid = ((r_norm >= 0.1) & (r_norm < 0.3)).sum()
        high = (r_norm >= 0.3).sum()
        total = max(1, low + mid + high)
        energy_low = mag[r_norm < 0.1].sum()
        energy_mid = mag[(r_norm >= 0.1) & (r_norm < 0.3)].sum()
        energy_high = mag[r_norm >= 0.3].sum()
        energy_total = max(1e-6, energy_low + energy_mid + energy_high)
        out["low_freq_ratio"] = float(energy_low / energy_total)
        out["mid_freq_ratio"] = float(energy_mid / energy_total)
        out["high_freq_ratio"] = float(energy_high / energy_total)
        out["spectral_centroid"] = float((r * mag).sum() / max(1.0, mag.sum()))

        # Color
        out["hue_mean"] = float(hsv[..., 0].mean())
        out["hue_std"] = float(hsv[..., 0].std())
        out["sat_mean"] = float(hsv[..., 1].mean())
        out["sat_std"] = float(hsv[..., 1].std())
        out["val_mean"] = float(hsv[..., 2].mean())
        out["val_std"] = float(hsv[..., 2].std())
        out["colorfulness"] = float(np.std(hsv[..., 1]) + np.std(hsv[..., 2]))

        # Texture (LBP proxy: gray histogram entropy)
        hist, _ = np.histogram(gray, bins=32, range=(0, 256))
        p = hist / max(1, hist.sum())
        p_safe = p[p > 0]
        out["lbp_entropy"] = float(-(p_safe * np.log2(p_safe)).sum())
        # GLCM contrast (simplified: gradient magnitude)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        out["glcm_contrast"] = float(np.sqrt(gx ** 2 + gy ** 2).mean())
        out["lbp_uniformity"] = float((p ** 2).sum())

        # Luma
        out["rms_contrast"] = float(gray.std() / max(1.0, gray.mean()))
        out["mean_luma"] = float(gray.mean())
        out["std_luma"] = float(gray.std())
        out["dynamic_range"] = float(gray.max() - gray.min())
        out["mean_shift"] = float(gray.mean() - 128.0)
        out["std_shift"] = float(gray.std() - 64.0)
        lh, _ = np.histogram(gray, bins=64, range=(0, 256))
        plh = lh / max(1, lh.sum())
        plh_safe = plh[plh > 0]
        out["luma_entropy"] = float(-(plh_safe * np.log2(plh_safe)).sum())
        out["luma_skew"] = float(((gray - gray.mean()) ** 3).mean() / max(1.0, gray.std() ** 3))
        out["luma_mean"] = float(gray.mean())

        # Structure: edge (Canny) + road region (bottom half)
        edges = cv2.Canny(gray, 80, 160)
        out["edge_mean"] = float(edges.mean())
        out["edge_density"] = float((edges > 0).mean())
        h2 = h // 2
        road_region = gray[h2:, :]
        out["road_luma_mean"] = float(road_region.mean())
        out["road_luma_std"] = float(road_region.std())

        # Lane line proxy: Hough lines on road region
        edges_road = cv2.Canny(road_region, 80, 160)
        try:
            lines = cv2.HoughLinesP(edges_road, 1, np.pi / 180,
                                    threshold=20, minLineLength=15, maxLineGap=10)
            out["lane_line_count"] = float(0 if lines is None else len(lines))
            out["lane_line_density"] = float(0 if lines is None else len(lines)) / max(1, edges_road.size / 1000)
        except Exception:
            out["lane_line_count"] = 0.0
            out["lane_line_density"] = 0.0

        # Shadow / highlight
        out["shadow_ratio"] = float((gray < 50).mean())
        out["highlight_ratio"] = float((gray > 220).mean())

        # Detection (YOLO if available)
        if yolo_model is not None:
            try:
                res = yolo_model(image, verbose=False)
                if hasattr(res, "__iter__") and len(res) > 0:
                    r0 = res[0]
                    boxes = r0.boxes
                    names = r0.names if hasattr(r0, "names") else {}
                    n_veh = 0
                    n_person = 0
                    n_total = 0
                    confs = []
                    if boxes is not None and len(boxes) > 0:
                        for b in boxes:
                            cls = int(b.cls[0].item()) if hasattr(b, "cls") else -1
                            conf = float(b.conf[0].item()) if hasattr(b, "conf") else 1.0
                            n_total += 1
                            confs.append(conf)
                            if cls in (1, 2, 3, 5, 7):  # car, truck, bus, etc.
                                n_veh += 1
                            if cls == 0:
                                n_person += 1
                    out["vehicle_loss"] = float(max(0, 5 - n_veh))
                    out["person_loss"] = float(max(0, 3 - n_person))
                    out["detection_loss"] = float(max(0, 10 - n_total))
                    out["conf_loss"] = float(1.0 - np.mean(confs)) if confs else 0.0
                    out["vehicle_loss_ratio"] = out["vehicle_loss"] / 5.0
            except Exception:
                out["vehicle_loss"] = 0.0
                out["person_loss"] = 0.0
                out["detection_loss"] = 0.0
                out["conf_loss"] = 0.0
                out["vehicle_loss_ratio"] = 0.0
        else:
            out["vehicle_loss"] = 0.0
            out["person_loss"] = 0.0
            out["detection_loss"] = 0.0
            out["conf_loss"] = 0.0
            out["vehicle_loss_ratio"] = 0.0

    except Exception as e:
        # 任何失败 → 返回零向量
        sys.stderr.write(f"  [WARN] gene extract failed: {e}\n")

    return out


# ---------------------------------------------------------------------------
# Mode A: cross-scene (在 NAVSIM 同一 dataset 内做 scene-stratified 验证)
# ---------------------------------------------------------------------------

def run_cross_scene(csv_path: str, n_holdout_scenes: int, seed: int) -> dict:
    """用 NAVSIM 同 dataset 的不同 scene 集做 cross-scene 验证。

    做法：
    - 合并数据 = 88,560 (3 planner × 492 scene × 60 attack-strength)
    - 随机抽 n_holdout_scenes 个 scene 作为 "holdout" (代理 cross-dataset)
    - 用其余 scene 训练 XGBoost → 在 holdout 上测 AUC
    - 跨 planner 也用同样切分
    """
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    df = pd.read_csv(csv_path)
    print(f"  loaded {len(df)} rows, {df['scene_token'].nunique()} unique scenes")

    rng = np.random.RandomState(seed)
    all_scenes = df["scene_token"].unique()
    rng.shuffle(all_scenes)
    holdout = set(all_scenes[:n_holdout_scenes].tolist())
    print(f"  holdout scenes: {len(holdout)}")

    df_train = df[~df["scene_token"].isin(holdout)].copy()
    df_test = df[df["scene_token"].isin(holdout)].copy()
    print(f"  train: {len(df_train)} rows, test: {len(df_test)} rows")

    results = []
    for pl in ["CNN", "DINO", "TF"]:
        sub_tr = df_train[df_train["planner"] == pl]
        sub_te = df_test[df_test["planner"] == pl]
        if len(sub_tr) == 0 or len(sub_te) == 0:
            continue
        feat = [c for c in GENE_FIELDS + META_FIELDS if c in sub_tr.columns]
        X_tr = np.nan_to_num(sub_tr[feat].values.astype(np.float32), nan=0.0)
        y_tr = sub_tr["success"].values.astype(np.int32)
        X_te = np.nan_to_num(sub_te[feat].values.astype(np.float32), nan=0.0)
        y_te = sub_te["success"].values.astype(np.int32)
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, eval_metric="logloss",
        )
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)[:, 1]
        if len(np.unique(y_te)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_te, proba))
        results.append({
            "planner": pl,
            "n_train": int(len(sub_tr)),
            "n_test": int(len(sub_te)),
            "test_fail_rate": float(1 - y_te.mean()),
            "auc": auc,
        })
        print(f"  {pl}: train={len(sub_tr)}, test={len(sub_te)}, "
              f"test_fail_rate={1-y_te.mean():.3f}, AUC={auc:.3f}")

    # 跨 planner transfer
    cross_results = []
    for src in ["CNN", "DINO", "TF"]:
        for tgt in ["CNN", "DINO", "TF"]:
            if src == tgt:
                continue
            sub_tr = df_train[df_train["planner"] == src]
            sub_te = df_test[df_test["planner"] == tgt]
            if len(sub_tr) == 0 or len(sub_te) == 0:
                continue
            feat = [c for c in GENE_FIELDS + META_FIELDS if c in sub_tr.columns]
            X_tr = np.nan_to_num(sub_tr[feat].values.astype(np.float32), nan=0.0)
            y_tr = sub_tr["success"].values.astype(np.int32)
            X_te = np.nan_to_num(sub_te[feat].values.astype(np.float32), nan=0.0)
            y_te = sub_te["success"].values.astype(np.int32)
            m = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed, n_jobs=-1, eval_metric="logloss",
            )
            m.fit(X_tr, y_tr)
            proba = m.predict_proba(X_te)[:, 1]
            if len(np.unique(y_te)) < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(y_te, proba))
            cross_results.append({
                "src": src, "tgt": tgt, "auc": auc,
                "n_train": int(len(sub_tr)),
                "n_test": int(len(sub_te)),
            })
            print(f"  {src} → {tgt}: AUC={auc:.3f}")

    aucs_within = [r["auc"] for r in results if not np.isnan(r["auc"])]
    aucs_cross = [r["auc"] for r in cross_results if not np.isnan(r["auc"])]
    summary = {
        "mode": "cross-scene",
        "n_holdout_scenes": int(n_holdout_scenes),
        "within_planner_auc": {
            "mean": float(np.mean(aucs_within)) if aucs_within else None,
            "per_planner": results,
        },
        "cross_planner_auc": {
            "mean": float(np.mean(aucs_cross)) if aucs_cross else None,
            "per_pair": cross_results,
        },
    }
    return summary


# ---------------------------------------------------------------------------
# Mode B: external (从外部图像目录)
# ---------------------------------------------------------------------------

def run_external(image_dir: str, csv_model: str, planner: str,
                 n_images: int, seed: int) -> dict:
    """读外部图像目录 + NAVSIM 训练的 XGBoost 模型 → 报跨数据集 fail 概率分布。

    注意：external 模式可能无法在 GPU 上跑 planner (无 ckpt)；
    这里只做 gene 抽取 + XGBoost 预测，看 gene 分布是否与 NAVSIM 一致。
    """
    # Find model
    import xgboost as xgb
    df_model = pd.read_csv(csv_model)
    df_pl = df_model[df_model["planner"] == planner].copy()
    feat = [c for c in GENE_FIELDS + META_FIELDS if c in df_pl.columns]
    X = np.nan_to_num(df_pl[feat].values.astype(np.float32), nan=0.0)
    y = df_pl["success"].values.astype(np.int32)
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1, eval_metric="logloss",
    )
    m.fit(X, y)
    print(f"  trained XGBoost on {len(df_pl)} NAVSIM-{planner} samples")

    # Read external images
    img_dir = Path(image_dir)
    img_files = sorted([
        p for p in img_dir.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    ])
    if len(img_files) == 0:
        return {"mode": "external", "error": f"no images found in {image_dir}"}
    rng = np.random.RandomState(seed)
    rng.shuffle(img_files)
    img_files = img_files[:n_images]
    print(f"  selected {len(img_files)} external images")

    try:
        import cv2
    except ImportError:
        return {"mode": "external", "error": "opencv not installed"}

    # Try load YOLO for detection gene
    yolo = None
    try:
        from ultralytics import YOLO
        yolo = YOLO("yolov8n.pt")
    except Exception:
        pass

    from navsim.agents.attack_genome.attacks.templates import ContinuousAttackSpace
    atk_space = ContinuousAttackSpace()
    attacks = ["rain", "dusk", "digital_noise"]
    strengths = [0.0, 0.2, 0.4, 0.6, 0.8]

    rows = []
    t0 = time.time()
    for ii, fp in enumerate(img_files):
        try:
            img = cv2.imread(str(fp))
            if img is None:
                continue
        except Exception:
            continue
        for atk in attacks:
            for s in strengths:
                try:
                    attacked = atk_space.evaluate(img, atk, s)
                except Exception:
                    attacked = img
                g = extract_37d_gene(attacked, yolo_model=yolo)
                g["strength"] = float(s)
                gvec = np.array([[g[c] for c in feat]], dtype=np.float32)
                gvec = np.nan_to_num(gvec, nan=0.0, posinf=0.0, neginf=0.0)
                p_fail = float(m.predict_proba(gvec)[0, 1])
                rows.append({
                    "image": str(fp), "attack": atk, "strength": s,
                    "pred_fail_prob": p_fail,
                    **{k: float(v) for k, v in g.items()},
                })
        if (ii + 1) % 5 == 0:
            print(f"  [{ii+1}/{len(img_files)}] elapsed={time.time()-t0:.1f}s")

    if not rows:
        return {"mode": "external", "error": "no rows produced"}

    res = pd.DataFrame(rows)
    res.to_csv(img_dir / "cross_dataset_predictions.csv", index=False)

    summary = {
        "mode": "external",
        "image_dir": str(image_dir),
        "n_images": int(len(img_files)),
        "n_rows": int(len(res)),
        "planner_used_for_xgb_training": planner,
        "mean_pred_fail_prob": float(res["pred_fail_prob"].mean()),
        "std_pred_fail_prob": float(res["pred_fail_prob"].std()),
        "per_attack": res.groupby("attack")["pred_fail_prob"].mean().to_dict(),
        "per_strength": res.groupby("strength")["pred_fail_prob"].mean().to_dict(),
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cross-scene", "external"], default="cross-scene")
    p.add_argument("--csv", default="exp/tierB_partial/merged_3pl.csv",
                   help="NAVSIM 训练数据 (用于 cross-scene 切分或 external 的 XGBoost 训练)")
    p.add_argument("--image-dir", default="",
                   help="external 模式下的图像目录 (e.g. datasets/navsim_mini/...)")
    p.add_argument("--n-holdout-scenes", type=int, default=100,
                   help="cross-scene 模式下 holdout 场景数")
    p.add_argument("--n-images", type=int, default=30,
                   help="external 模式下最多读多少张图")
    p.add_argument("--planner", default="CNN", choices=["CNN", "DINO", "TF"])
    p.add_argument("--output-dir", default="exp/tierB_partial/cross_dataset")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Cross-Dataset Validation (mode={args.mode}) ===")

    if args.mode == "cross-scene":
        summary = run_cross_scene(args.csv, args.n_holdout_scenes, args.seed)
        out_json = out_dir / "cross_scene_results.json"
        out_txt = out_dir / "cross_scene_summary.txt"
    else:
        if not args.image_dir:
            print("  [FATAL] --image-dir required for external mode")
            sys.exit(2)
        summary = run_external(args.image_dir, args.csv, args.planner,
                               args.n_images, args.seed)
        out_json = out_dir / "external_results.json"
        out_txt = out_dir / "external_summary.txt"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"# Cross-Dataset Validation — mode={args.mode}\n\n")
        f.write(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDONE. → {out_json}")


if __name__ == "__main__":
    main()
