"""Standalone scene loader — 完全绕开 nuplan / navsim.dataloader。

直接读 nuScenes pkl (test/trainval 拆 pkl) + sensor_blobs jpg，
重建 navsim_loader 的 stitch_for_gtrs (256×1024) 输出。

用法：
    from scripts.analysis.standalone_scene_loader import StandaloneSceneIndex
    idx = StandaloneSceneIndex("/e/navsim_workspace/dataset")
    image = idx.load_image("8462feefcf135a4a")  # (256, 1024, 3) uint8
"""
from __future__ import annotations
import pickle, sys, time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class StandaloneSceneIndex:
    """scene_token → (pkl_path, frame_idx, sensor_blobs_root)."""

    def __init__(self, openscene_root: str, splits: Tuple[str, ...] = ("test", "trainval", "mini")):
        self.root = Path(openscene_root)
        self.splits = splits
        self.scene_to_loc: Dict[str, Tuple[Path, int, Path]] = {}
        self._build()

    def _build(self) -> None:
        navsim_logs = self.root / "navsim_logs"
        sensor_blobs = self.root / "sensor_blobs"
        if not navsim_logs.is_dir() or not sensor_blobs.is_dir():
            raise FileNotFoundError(f"navsim_logs or sensor_blobs not under {self.root}")

        for split in self.splits:
            logs_dir = navsim_logs / split
            blobs_dir = sensor_blobs / split
            if not logs_dir.is_dir():
                continue
            n_pkl = 0
            for pkl_path in sorted(logs_dir.glob("*.pkl")):
                n_pkl += 1
                try:
                    with open(pkl_path, "rb") as f:
                        frames = pickle.load(f)
                except Exception:
                    continue
                if not isinstance(frames, list):
                    continue
                for fi, fr in enumerate(frames):
                    st = fr.get("scene_token")
                    if st and st not in self.scene_to_loc:
                        self.scene_to_loc[st] = (pkl_path, fi, blobs_dir)
            print(f"  [standalone] split={split:>9s}: {n_pkl} pkls → {len(self.scene_to_loc)} unique scenes total")

    def has(self, scene_token: str) -> bool:
        return scene_token in self.scene_to_loc

    def load_image(self, scene_token: str, target_h: int = 256, target_w: int = 1024) -> Optional[np.ndarray]:
        """读 3 cam (F0/L0/R0) jpg，按 stitch_for_gtrs 拼成 256×1024。"""
        if scene_token not in self.scene_to_loc:
            return None
        pkl_path, fi, blobs_dir = self.scene_to_loc[scene_token]
        try:
            with open(pkl_path, "rb") as f:
                frames = pickle.load(f)
            fr = frames[fi]
            cams = fr["cams"]
            # CAM_F0 是必须的
            f0_rel = cams["CAM_F0"]["data_path"]
            f0 = cv2.imread(str(blobs_dir / f0_rel), cv2.IMREAD_COLOR)
            if f0 is None:
                return None
            f0 = cv2.cvtColor(f0, cv2.COLOR_BGR2RGB)
            # L0 / R0 可选 (原 stitch_for_gtrs 在缺失时回退到只用 F0)
            l0, r0 = None, None
            if "CAM_L0" in cams:
                l0 = cv2.imread(str(blobs_dir / cams["CAM_L0"]["data_path"]), cv2.IMREAD_COLOR)
                if l0 is not None: l0 = cv2.cvtColor(l0, cv2.COLOR_BGR2RGB)
            if "CAM_R0" in cams:
                r0 = cv2.imread(str(blobs_dir / cams["CAM_R0"]["data_path"]), cv2.IMREAD_COLOR)
                if r0 is not None: r0 = cv2.cvtColor(r0, cv2.COLOR_BGR2RGB)

            if l0 is not None and r0 is not None and l0.size > 0 and r0.size > 0:
                l0_c = l0[28:-28, 416:-416]
                f0_c = f0[28:-28]
                r0_c = r0[28:-28, 416:-416]
                stitched = np.concatenate([l0_c, f0_c, r0_c], axis=1)
            else:
                stitched = f0[28:-28]
            return cv2.resize(stitched, (target_w, target_h), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"  [standalone] load_image err: {scene_token}: {e}")
            return None

    def load_gt_trajectory(self, scene_token: str, num_future: int = 8) -> Optional[np.ndarray]:
        """从相邻帧取 ego 位置, 算 (T, 3) 相对未来轨迹 (x, y, heading)。"""
        if scene_token not in self.scene_to_loc:
            return None
        pkl_path, fi, _ = self.scene_to_loc[scene_token]
        try:
            with open(pkl_path, "rb") as f:
                frames = pickle.load(f)
            if fi + num_future >= len(frames):
                return None
            fr0 = frames[fi]
            ego0_t = fr0["ego2global_translation"]
            ego0_R = fr0["ego2global_rotation"]
            traj = []
            for k in range(1, num_future + 1):
                fr = frames[fi + k]
                ego_t = fr["ego2global_translation"]
                # 相对位置
                d = ego_t - ego0_t
                # rotate 到 ego0 frame
                d_ego = ego0_R.T @ d  # (3,)  (inverse of ego0_R = ego0_R.T for rotation)
                traj.append([float(d_ego[0]), float(d_ego[1]), 0.0])
            return np.asarray(traj, dtype=np.float32)
        except Exception as e:
            return None


if __name__ == "__main__":
    # 简单测试
    root = "/e/navsim_workspace/dataset"
    t0 = time.time()
    idx = StandaloneSceneIndex(root)
    print(f"\nindex built in {time.time() - t0:.1f}s, total scenes: {len(idx.scene_to_loc)}")

    # 拿一个真实 scene token 测
    import pandas as pd
    df = pd.read_csv("d:/cogatedrive/exp/tierB_partial/merged_3pl.csv")
    for token in df["scene_token"].unique()[:3]:
        print(f"\n--- {token} ---")
        img = idx.load_image(token)
        if img is not None:
            print(f"  image: shape={img.shape} dtype={img.dtype} mean={img.mean():.1f}")
        else:
            print(f"  image: FAILED to load")
        traj = idx.load_gt_trajectory(token)
        if traj is not None:
            print(f"  traj: shape={traj.shape} last xy={traj[-1, :2].tolist()}")
