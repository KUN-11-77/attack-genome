"""结构基因（Structural Gene）。

设计文档 3.3：二级基因。
特征：
    - 道路可见度（基于 Sobel 边缘密度）
    - 车道线清晰度（基于 Hough 线段密度）

为了让结构基因在视觉外观被攻击后依然有意义，我们把度量分解为：
    1. 边缘密度：图像梯度强度均值，越高表示结构越丰富
    2. 道路占比：图像下半部分亮度/方差分布，反映地面纹理被保留
    3. 车道线显著度：使用 ``cv2.HoughLinesP`` 检测直线数量
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

from navsim.agents.attack_genome.genes.base import GeneExtractorBase, to_gray


@dataclass
class StructuralGeneExtractor(GeneExtractorBase):
    """结构特征，刻画场景几何是否被破坏。"""

    name: str = "structural"
    enabled: bool = True
    hough_threshold: int = 40
    hough_min_line_length: int = 30
    hough_max_line_gap: int = 10

    def extract(self, image: np.ndarray) -> Dict[str, float]:
        gray = to_gray(image)
        # 边缘强度
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        edge_mean = float(mag.mean() / 255.0)
        edge_density = float((mag > 30).sum() / mag.size)

        # 道路可见度：下半部分的方差 + 亮度方差
        h, w = gray.shape
        lower = gray[h // 2 :, :]
        road_luma_mean = float(lower.mean() / 255.0)
        road_luma_std = float(lower.std() / 255.0)

        # 车道线检测
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )
        line_count = 0 if lines is None else int(lines.shape[0])
        line_density = float(line_count) / float(h * w) * 1e4

        return {
            "edge_mean": edge_mean,
            "edge_density": edge_density,
            "road_luma_mean": road_luma_mean,
            "road_luma_std": road_luma_std,
            "lane_line_count": float(line_count),
            "lane_line_density": line_density,
        }
