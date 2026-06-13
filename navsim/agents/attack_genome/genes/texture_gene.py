"""纹理基因（Texture Gene）。

设计文档 3.2：一级基因。
特征：
    - LBP 熵
    - GLCM 对比度
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

from navsim.agents.attack_genome.genes.base import GeneExtractorBase, to_gray


def _lbp_hist(gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """Uniform LBP 直方图。"""

    h, w = gray.shape
    if h <= 2 * radius + 1 or w <= 2 * radius + 1:
        return np.zeros(n_points + 2, dtype=np.float32)
    center = gray[radius:h - radius, radius:w - radius]
    codes = np.zeros_like(center, dtype=np.uint8)
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    for i, ang in enumerate(angles):
        dy = int(round(radius * np.sin(ang)))
        dx = int(round(radius * np.cos(ang)))
        # 用最近邻采样避免越界
        ys = np.clip(np.arange(radius, h - radius) + dy, 0, h - 1)
        xs = np.clip(np.arange(radius, w - radius) + dx, 0, w - 1)
        neighbor = gray[ys[:, None], xs[None, :]]
        codes |= ((neighbor >= center).astype(np.uint8) << i)
    hist, _ = np.histogram(codes, bins=range(0, 2 ** n_points + 1), density=False)
    hist = hist.astype(np.float32) + 1e-8
    hist /= hist.sum()
    return hist


def _lbp_entropy(hist: np.ndarray) -> float:
    return float(-(hist * np.log(hist + 1e-12)).sum())


def _glcm_contrast(gray: np.ndarray, distances=(1,), angles=(0,)) -> float:
    """粗略估计灰度共生矩阵的对比度，避免引入 skimage 依赖。"""

    g = (gray / 16).astype(np.uint8)  # 量化到 0-15 减小矩阵
    values = []
    for d in distances:
        for a in angles:
            shifted = np.roll(g, shift=(d * int(np.sin(a)), d * int(np.cos(a))), axis=(0, 1))
            diff = (g.astype(np.int16) - shifted.astype(np.int16)) ** 2
            values.append(diff.mean())
    if not values:
        return 0.0
    return float(np.mean(values))


@dataclass
class TextureGeneExtractor(GeneExtractorBase):
    """纹理统计特征（LBP 熵 + GLCM 对比度）。"""

    name: str = "texture"
    enabled: bool = True

    def extract(self, image: np.ndarray) -> Dict[str, float]:
        gray = to_gray(image)
        # 缩放到 256x256 控制计算量，基因属于粗粒度描述
        scale = 256
        if max(gray.shape) > scale:
            ratio = scale / max(gray.shape)
            gray = cv2.resize(gray, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        hist = _lbp_hist(gray)
        return {
            "lbp_entropy": _lbp_entropy(hist),
            "glcm_contrast": _glcm_contrast(gray),
            "lbp_uniformity": float((hist ** 2).sum()),
        }
