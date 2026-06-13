"""对比度基因（Contrast Gene）。

设计文档 3.2：一级基因。
特征：
    - RMS Contrast
    - 全局均值差（用于反映攻击前后亮度偏移）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from navsim.agents.attack_genome.genes.base import GeneExtractorBase, to_gray


@dataclass
class ContrastGeneExtractor(GeneExtractorBase):
    """RMS 对比度与亮度统计。

    当提供 ``reference`` 时，额外计算 ``mean_shift``。
    """

    name: str = "contrast"
    enabled: bool = True
    reference: Optional[np.ndarray] = None

    def extract(self, image: np.ndarray) -> Dict[str, float]:
        gray = to_gray(image).astype(np.float32) / 255.0
        rms = float(np.sqrt(np.mean(gray ** 2)))
        mean = float(gray.mean())
        std = float(gray.std())
        # 动态范围
        dyn_range = float(gray.max() - gray.min())

        result: Dict[str, float] = {
            "rms_contrast": rms,
            "mean_luma": mean,
            "std_luma": std,
            "dynamic_range": dyn_range,
        }
        if self.reference is not None:
            ref_gray = to_gray(self.reference).astype(np.float32) / 255.0
            result["mean_shift"] = float(mean - ref_gray.mean())
            result["std_shift"] = float(std - ref_gray.std())
        return result
