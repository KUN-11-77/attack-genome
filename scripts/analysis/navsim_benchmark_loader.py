"""Navsim/benchmark 场景 loader — 服务器用。

适配 /data3/khsong/data/navsim/benchmark/<log>/<16-char token>/{F0,L0,R0}/*.jpg 结构。

scene_token → (log_dir, F0/L0/R0 clean filename)。

攻击应用：通过 navsim.agents.attack_genome.attacks.templates.ContinuousAttackSpace 在
clean image 上 evaluate。强度由 csv 的 strength 字段决定。
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class NavsimBenchmarkIndex:
    """scene_token → scene_dir。"""

    def __init__(self, benchmark_root: str, cache_path: Optional[str] = None):
        self.root = Path(benchmark_root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"benchmark root not found: {self.root}")
        self.cache_path = Path(cache_path) if cache_path else None
        self.scene_to_dir: Dict[str, Path] = {}
        self._build()

    def _build(self) -> None:
        if self.cache_path and self.cache_path.exists():
            print(f"  [navsim_bench] loading cached index from {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                self.scene_to_dir = pickle.load(f)
            print(f"  [navsim_bench] cached: {len(self.scene_to_dir)} scenes")
            return

        t0 = time.time()
        n_logs = 0
        for log_dir in sorted(self.root.iterdir()):
            if not log_dir.is_dir():
                continue
            n_logs += 1
            for scene_dir in sorted(log_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                token = scene_dir.name
                if len(token) != 16:
                    continue
                # must have F0/L0/R0
                if (scene_dir / "F0").is_dir() and (scene_dir / "L0").is_dir() and (scene_dir / "R0").is_dir():
                    self.scene_to_dir[token] = scene_dir
        print(f"  [navsim_bench] scanned {n_logs} logs, {len(self.scene_to_dir)} scenes in {time.time() - t0:.1f}s")

        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.scene_to_dir, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  [navsim_bench] cached to {self.cache_path}")

    def has(self, scene_token: str) -> bool:
        return scene_token in self.scene_to_dir

    def _load_cam(self, scene_dir: Path, cam: str) -> Optional[np.ndarray]:
        cam_dir = scene_dir / cam
        if not cam_dir.is_dir():
            return None
        # find any jpg (clean)
        jpgs = sorted(cam_dir.glob("*.jpg"))
        if not jpgs:
            return None
        img = cv2.imread(str(jpgs[0]), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_clean_image(self, scene_token: str, target_h: int = 256, target_w: int = 1024) -> Optional[np.ndarray]:
        """读 clean stitched 256×1024 image。"""
        if scene_token not in self.scene_to_dir:
            return None
        scene_dir = self.scene_to_dir[scene_token]
        try:
            f0 = self._load_cam(scene_dir, "F0")
            l0 = self._load_cam(scene_dir, "L0")
            r0 = self._load_cam(scene_dir, "R0")
            if f0 is None:
                return None
            if l0 is not None and r0 is not None and l0.size > 0 and r0.size > 0:
                l0_c = l0[28:-28, 416:-416]
                f0_c = f0[28:-28]
                r0_c = r0[28:-28, 416:-416]
                stitched = np.concatenate([l0_c, f0_c, r0_c], axis=1)
            else:
                stitched = f0[28:-28]
            return cv2.resize(stitched, (target_w, target_h), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"  [navsim_bench] load err: {scene_token}: {e}")
            return None

    def apply_attack(self, img: np.ndarray, attack: str, strength: float):
        """通过 navsim.attack_genome ContinuousAttackSpace 应用攻击。"""
        from navsim.agents.attack_genome.attacks.templates import ContinuousAttackSpace
        atk_space = ContinuousAttackSpace()
        return atk_space.evaluate(img, attack, float(strength))


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "/data3/khsong/data/navsim/benchmark"
    cache = sys.argv[2] if len(sys.argv) > 2 else "/tmp/navsim_bench_idx.pkl"
    idx = NavsimBenchmarkIndex(root, cache_path=cache)
    print(f"\nindex: {len(idx.scene_to_dir)} scenes")
    # Test
    if len(idx.scene_to_dir) > 0:
        first = sorted(idx.scene_to_dir.keys())[0]
        img = idx.load_clean_image(first)
        print(f"  test load [{first}]: shape={None if img is None else img.shape}")
