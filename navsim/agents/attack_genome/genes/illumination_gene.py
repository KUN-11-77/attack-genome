"""光照基因（Illumination Gene）。

设计文档 3.3：二级基因。
特征：
    - 阴影覆盖率
    - 强光区域占比

实现方式：
    1. 灰度图统计低亮度像素占比 → 阴影覆盖
    2. 灰度图统计高亮度像素占比 → 强光
    3. 直方图偏度：攻击是否让图像整体更暗 / 更亮
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

from navsim.agents.attack_genome.genes.base import GeneExtractorBase, to_gray


@dataclass
class IlluminationGeneExtractor(GeneExtractorBase):
    """全局光照扰动统计。"""

    name: str = "illumination"
    enabled: bool = True
    shadow_thresh: int = 40
    highlight_thresh: int = 220

    def extract(self, image: np.ndarray) -> Dict[str, float]:
        gray = to_gray(image)
        h, w = gray.shape
        total = float(gray.size)

        shadow = float((gray < self.shadow_thresh).sum() / total)
        highlight = float((gray > self.highlight_thresh).sum() / total)

        # 全局亮度直方图熵
        hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=False)
        hist = hist.astype(np.float32) + 1e-8
        hist /= hist.sum()
        entropy = float(-(hist * np.log(hist)).sum())

        # 偏度（左右亮度分布）
        mean = float(gray.mean())
        std = float(gray.std() + 1e-8)
        skew = float(((gray - mean) ** 3).mean() / (std ** 3))

        return {
            "shadow_ratio": shadow,
            "highlight_ratio": highlight,
            "luma_entropy": entropy,
            "luma_skew": skew,
            "luma_mean": mean / 255.0,
        }
