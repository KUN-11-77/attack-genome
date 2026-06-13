"""频率基因（Frequency Gene）。

设计文档 3.2：一级基因。
特征：
    - 低频能量占比
    - 高频能量占比

实现思路：
    1. 灰度图做二维 FFT
    2. 用低通 / 高通圆环 mask 划分频段
    3. 分别累计能量并归一化
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

from navsim.agents.attack_genome.genes.base import GeneExtractorBase, to_gray


@dataclass
class FrequencyGeneExtractor(GeneExtractorBase):
    """频域能量分布特征。

    Parameters
    ----------
    low_cutoff_ratio:
        低频圆环的归一化半径阈值（相对图像短边的一半）。
    high_cutoff_ratio:
        高频圆环内半径阈值。
    """

    name: str = "frequency"
    low_cutoff_ratio: float = 0.15
    high_cutoff_ratio: float = 0.45
    enabled: bool = True

    def extract(self, image: np.ndarray) -> Dict[str, float]:
        gray = to_gray(image).astype(np.float32) / 255.0
        h, w = gray.shape
        # 去除直流分量后再做 FFT，避免恒定亮度主导
        gray = gray - gray.mean()
        spec = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.abs(spec) ** 2

        cy, cx = h / 2.0, w / 2.0
        # 频域半径网格（相对半短边）
        yy, xx = np.indices((h, w))
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / min(cy, cx)
        total = magnitude.sum() + 1e-8

        low_mask = radius <= self.low_cutoff_ratio
        high_mask = radius >= self.high_cutoff_ratio

        low_ratio = float(magnitude[low_mask].sum() / total)
        high_ratio = float(magnitude[high_mask].sum() / total)
        # 中频占比
        mid_ratio = float(
            magnitude[(~low_mask) & (~high_mask)].sum() / total
        )

        return {
            "low_freq_ratio": low_ratio,
            "mid_freq_ratio": mid_ratio,
            "high_freq_ratio": high_ratio,
            "spectral_centroid": float(
                (radius * magnitude).sum() / total
            ),
        }
