"""颜色基因（Color Gene）。

设计文档 3.2：一级基因。
特征：
    - HSV 偏移（H/S/V 均值）
    - 饱和度变化
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

from navsim.agents.attack_genome.genes.base import GeneExtractorBase, ensure_uint8


@dataclass
class ColorGeneExtractor(GeneExtractorBase):
    """HSV 颜色空间统计特征。"""

    name: str = "color"
    enabled: bool = True

    def extract(self, image: np.ndarray) -> Dict[str, float]:
        img = ensure_uint8(image)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # 仅统计非零像素，避免黑色背景对饱和度均值的干扰
        nonzero_v = v[v > 5]
        if nonzero_v.size == 0:
            nonzero_v = v.flatten()

        return {
            "hue_mean": float(h.mean()),
            "hue_std": float(h.std()),
            "sat_mean": float(s.mean() / 255.0),
            "sat_std": float(s.std() / 255.0),
            "val_mean": float(nonzero_v.mean() / 255.0),
            "val_std": float(nonzero_v.std() / 255.0),
            # 颜色丰富度：>阈值的饱和度像素占比
            "colorfulness": float(
                (s > 32).sum() / max(1, s.size)
            ),
        }
