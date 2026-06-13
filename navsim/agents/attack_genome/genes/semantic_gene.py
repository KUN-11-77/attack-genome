"""语义基因（Semantic Gene）。

设计文档 3.3：二级基因。
特征：
    - 车辆显著度
    - 行人显著度
    - 交通设施显著度

为了在没有真值标签的情况下也能估计这些显著度，使用颜色/形状的启发
式代理：把图像分成上 / 中 / 下三个水平带，对每个带估计显著性高的色
块占比。这是一种轻量代理，适合作为指纹特征。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

from navsim.agents.attack_genome.genes.base import GeneExtractorBase, ensure_uint8


def _vertical_band_ratio(
    img: np.ndarray, band_slice: slice, sat_thresh: int = 50, lum_low: int = 40, lum_high: int = 220
) -> float:
    """特定水平带内"显著色块"像素占比。"""

    band = img[band_slice, :, :]
    hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
    s = hsv[..., 1]
    v = hsv[..., 2]
    mask = (s > sat_thresh) & (v > lum_low) & (v < lum_high)
    return float(mask.sum() / max(1, mask.size))


@dataclass
class SemanticGeneExtractor(GeneExtractorBase):
    """语义显著度代理特征。"""

    name: str = "semantic"
    enabled: bool = True

    def extract(self, image: np.ndarray) -> Dict[str, float]:
        img = ensure_uint8(image)
        h, _, _ = img.shape

        # 车辆通常出现在图像中下部（前方车流）
        vehicle_proxy = _vertical_band_ratio(img, slice(int(h * 0.5), int(h * 0.85)))
        # 行人较矮，可出现在中下偏右
        pedestrian_proxy = _vertical_band_ratio(
            img, slice(int(h * 0.5), int(h * 0.95)), sat_thresh=80
        )
        # 交通设施（信号灯、标志）通常位于上方
        traffic_proxy = _vertical_band_ratio(
            img, slice(int(h * 0.05), int(h * 0.45)), sat_thresh=120
        )

        # 黄色车道线显著度（HSV 中黄 ≈ H in [20, 35]）
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        yellow_mask = (
            (h_ch >= 15) & (h_ch <= 40) & (s_ch > 100) & (v_ch > 80)
        )
        yellow_ratio = float(yellow_mask.sum() / max(1, yellow_mask.size))

        return {
            "vehicle_saliency": vehicle_proxy,
            "pedestrian_saliency": pedestrian_proxy,
            "traffic_saliency": traffic_proxy,
            "lane_yellow_ratio": yellow_ratio,
        }
