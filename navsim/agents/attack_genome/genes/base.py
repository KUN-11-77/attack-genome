"""基因提取器基类与公共工具。

基因提取器只依赖 numpy + opencv，避免被具体训练框架锁定，便于离线批
处理和单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np


ArrayImage = np.ndarray  # shape (H, W, 3), uint8 / float32


def ensure_uint8(image: ArrayImage) -> ArrayImage:
    """将图像统一到 ``uint8`` 范围。"""

    if image.dtype == np.uint8:
        return image
    arr = image.astype(np.float32)
    if arr.max() <= 1.0 + 1e-6:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def to_gray(image: ArrayImage) -> np.ndarray:
    """统一到单通道灰度 ``uint8``。"""

    img = ensure_uint8(image)
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


@dataclass
class GeneExtractorBase:
    """基因提取器基类。子类通过 ``name`` 标识输出字段。"""

    name: str
    enabled: bool = True
    params: Optional[Dict[str, Any]] = None

    def extract(self, image: ArrayImage) -> Dict[str, float]:
        """提取基因特征，返回 ``{key: scalar}``。"""

        raise NotImplementedError

    def __call__(self, image: ArrayImage) -> Dict[str, float]:
        if not self.enabled:
            return {}
        return dict(self.extract(image))
