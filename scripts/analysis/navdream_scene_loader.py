"""NavDream 场景 loader — 直接读 navdream_benchmark_outputs 下的预计算 attacked images。

结构：
    navdream_benchmark_outputs/<navdream_log>/<16-char scene_token>/
        F0/<clean_token>.jpg + 11 个 attacked .jpg
        L0/<clean_token>.jpg + 11 个 attacked .jpg
        R0/<clean_token>.jpg + 11 个 attacked .jpg

CSV 攻击名 → 文件名后缀：
    Rain         → heavy_rain
    Snow         → heavy_snow
    Dusk         → dusk_sunset
    Dawn         → dawn_sunrise
    MotionBlur   → motion_blur
    DigitalNoise → digital_noise
    LightDust    → light_dust
    DappledLight → dappled_light
    VintageStyle → vintage_photo
    CarlaStyle   → carla_toy_like

API：
    idx = NavDreamIndex("/mnt/e/navsim_workspace/dataset")
    img = idx.load_image(token)                # 256×1024 stitched clean
    img = idx.load_image(token, attack="Rain") # 256×1024 stitched attacked
    idx.has(token)
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


ATTACK_NAME_TO_FILE = {
    "Rain":         "heavy_rain",
    "Snow":         "heavy_snow",
    "Dusk":         "dusk_sunset",
    "Dawn":         "dawn_sunrise",
    "MotionBlur":   "motion_blur",
    "DigitalNoise": "digital_noise",
    "LightDust":    "light_dust",
    "DappledLight": "dappled_light",
    "VintageStyle": "vintage_photo",
    "CarlaStyle":   "carla_toy_like",
}


class NavDreamIndex:
    """scene_token → (log_dir, F0_clean, L0_clean, R0_clean, attack_files dict)."""

    def __init__(self, openscene_root: str, cache_path: Optional[str] = None):
        self.root = Path(openscene_root)
        self.bench_root = self.root / "navdream_benchmark_outputs"
        if not self.bench_root.is_dir():
            raise FileNotFoundError(f"navdream_benchmark_outputs not under {self.root}")
        self.cache_path = Path(cache_path) if cache_path else None
        self.scene_to_loc: Dict[str, Dict] = {}
        self._build()

    def _build(self) -> None:
        if self.cache_path and self.cache_path.exists():
            print(f"  [navdream] loading cached index from {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                self.scene_to_loc = pickle.load(f)
            print(f"  [navdream] cached index: {len(self.scene_to_loc)} scenes")
            return

        t0 = time.time()
        n_logs = 0
        for log_dir in sorted(self.bench_root.iterdir()):
            if not log_dir.is_dir():
                continue
            n_logs += 1
            for scene_dir in sorted(log_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                token = scene_dir.name
                if len(token) != 16:
                    continue
                f0_dir = scene_dir / "F0"
                l0_dir = scene_dir / "L0"
                r0_dir = scene_dir / "R0"
                if not (f0_dir.is_dir() and l0_dir.is_dir() and r0_dir.is_dir()):
                    continue
                f0_files = sorted([p.name for p in f0_dir.glob("*.jpg")])
                if not f0_files:
                    continue
                # The "clean" file is the one that is NOT a known attack name.
                # In the actual data, clean is the .jpg without an underscore-attack suffix.
                # We just take the first .jpg that's NOT a known attack.
                attack_keys = set(ATTACK_NAME_TO_FILE.values())
                clean_files = [f for f in f0_files if f.split(".jpg")[0] not in attack_keys]
                if not clean_files:
                    continue
                clean_name = clean_files[0]
                self.scene_to_loc[token] = {
                    "scene_dir": scene_dir,
                    "clean_name": clean_name,
                }
        print(f"  [navdream] scanned {n_logs} logs, {len(self.scene_to_loc)} scenes in {time.time() - t0:.1f}s")

        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.scene_to_loc, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  [navdream] cached index to {self.cache_path}")

    def has(self, scene_token: str) -> bool:
        return scene_token in self.scene_to_loc

    def _load_cam(self, scene_dir: Path, cam: str, file_name: str) -> Optional[np.ndarray]:
        path = scene_dir / cam / file_name
        if not path.exists():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _stitch(self, f0: np.ndarray, l0: Optional[np.ndarray], r0: Optional[np.ndarray],
                target_h: int = 256, target_w: int = 1024) -> np.ndarray:
        if l0 is not None and r0 is not None and l0.size > 0 and r0.size > 0:
            l0_c = l0[28:-28, 416:-416]
            f0_c = f0[28:-28]
            r0_c = r0[28:-28, 416:-416]
            stitched = np.concatenate([l0_c, f0_c, r0_c], axis=1)
        else:
            stitched = f0[28:-28]
        return cv2.resize(stitched, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def load_image(self, scene_token: str, attack: Optional[str] = None,
                   target_h: int = 256, target_w: int = 1024) -> Optional[np.ndarray]:
        """读 clean 或 attacked stitched 256×1024 image。"""
        if scene_token not in self.scene_to_loc:
            return None
        info = self.scene_to_loc[scene_token]
        scene_dir: Path = info["scene_dir"]
        if attack is None:
            f_name = info["clean_name"]
        else:
            file_key = ATTACK_NAME_TO_FILE.get(attack)
            if file_key is None:
                return None
            f_name = f"{file_key}.jpg"
            # 某些 attack 有 2 个变体 (e.g. dappled_light, dappled_light2),
            # 我们只取第一个。
        try:
            f0 = self._load_cam(scene_dir, "F0", f_name)
            l0 = self._load_cam(scene_dir, "L0", f_name)
            r0 = self._load_cam(scene_dir, "R0", f_name)
            if f0 is None:
                return None
            return self._stitch(f0, l0, r0, target_h, target_w)
        except Exception as e:
            print(f"  [navdream] load_image err: {scene_token}/{attack}: {e}")
            return None


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/e/navsim_workspace/dataset"
    cache = sys.argv[2] if len(sys.argv) > 2 else "/tmp/navdream_index_cache.pkl"
    idx = NavDreamIndex(root, cache_path=cache)
    print(f"\nindex built: {len(idx.scene_to_loc)} scenes")
    # Test
    import pandas as pd
    df = pd.read_csv("d:/cogatedrive/exp/tierB_partial/merged_3pl.csv")
    sample = df.sample(n=3, random_state=42)
    for _, row in sample.iterrows():
        token = row["scene_token"]
        attack = row["attack"]
        print(f"\n--- {token} (attack={attack}) ---")
        clean = idx.load_image(token)
        atk = idx.load_image(token, attack=attack)
        print(f"  clean: {'OK shape=' + str(clean.shape) if clean is not None else 'FAILED'}")
        print(f"  attack: {'OK shape=' + str(atk.shape) if atk is not None else 'FAILED'}")
