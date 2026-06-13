"""10 个攻击模板的具体实现。

设计文档第 5 节列出的攻击类型。每个模板都是 ``AttackTemplate`` 的子
类，使用纯 OpenCV / NumPy 实现，便于在 GPU 受限环境下离线大规模生
成对抗样本。

强度约定 ``s in [0, 1]``：
    0.0  → 原图
    0.2  → 微弱扰动
    0.4  → 轻度扰动
    0.6  → 中度扰动
    0.8  → 强扰动
    1.0  → 极端扰动
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from navsim.agents.attack_genome.genes.base import ensure_uint8


ArrayImage = np.ndarray
StrengthFn = Callable[[ArrayImage, float], ArrayImage]


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------

def _blend(image: ArrayImage, modified: ArrayImage, strength: float) -> ArrayImage:
    """按强度将 ``modified`` 与 ``image`` 线性混合。"""

    if strength <= 0:
        return image
    if strength >= 1:
        return modified
    s = float(strength)
    # cv2.addWeighted 要求两个输入 dtype 一致
    if image.dtype != modified.dtype:
        modified = modified.astype(image.dtype)
    return cv2.addWeighted(image, 1.0 - s, modified, s, 0)


def _clip_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _hue_shift_hsv(image: ArrayImage, delta_h: float) -> ArrayImage:
    img = ensure_uint8(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(delta_h)) % 180
    hsv[..., 0] = np.clip(hsv[..., 0], 0, 179)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _value_scale(image: ArrayImage, factor: float) -> ArrayImage:
    img = ensure_uint8(image).astype(np.float32)
    img = img * factor
    return _clip_uint8(img)


# ----------------------------------------------------------------------
# 模板基类
# ----------------------------------------------------------------------


@dataclass
class AttackTemplate:
    """单个攻击模板的注册信息。"""

    name: str
    description: str
    apply: StrengthFn
    # 是否为外观类（设计文档 2.3 节区分 "adversarial appearance"）
    is_appearance: bool = True

    def __call__(self, image: ArrayImage, strength: float) -> ArrayImage:
        if strength <= 0:
            return image
        return ensure_uint8(self.apply(image, float(np.clip(strength, 0.0, 1.0))))


# ----------------------------------------------------------------------
# 具体模板
# ----------------------------------------------------------------------


def rain_template(image: ArrayImage, strength: float) -> ArrayImage:
    """雨线纹理：白/半透明短线 + 轻微运动模糊。"""

    h, w = image.shape[:2]
    n = int(800 * strength)
    overlay = image.copy()
    for _ in range(n):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(8, 16)
        cv2.line(
            overlay,
            (x, y),
            (x - 2, y + length),
            (220, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return _blend(image, overlay, strength)


def snow_template(image: ArrayImage, strength: float) -> ArrayImage:
    """雪花：白点 + 模糊光晕。"""

    h, w = image.shape[:2]
    n = int(600 * strength)
    overlay = image.copy()
    pts = np.random.randint(0, [w, h], size=(n, 2))
    for (x, y) in pts:
        cv2.circle(overlay, (int(x), int(y)), 2, (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
    return _blend(image, overlay, strength)


def dusk_template(image: ArrayImage, strength: float) -> ArrayImage:
    """黄昏：色温变暖（红 / 橙）+ 亮度下降。"""

    img = image.astype(np.float32)
    img[..., 0] *= 1.0 + 0.25 * strength  # R
    img[..., 1] *= 1.0 + 0.10 * strength  # G
    img[..., 2] *= 1.0 - 0.35 * strength  # B
    img *= 1.0 - 0.15 * strength
    return _clip_uint8(img)


def dawn_template(image: ArrayImage, strength: float) -> ArrayImage:
    """黎明：色温变冷（蓝 / 紫）+ 亮度变化小。"""

    img = image.astype(np.float32)
    img[..., 0] *= 1.0 + 0.05 * strength
    img[..., 1] *= 1.0 + 0.05 * strength
    img[..., 2] *= 1.0 + 0.30 * strength
    return _clip_uint8(img)


def motion_blur_template(image: ArrayImage, strength: float) -> ArrayImage:
    """运动模糊：水平方向线性模糊。"""

    k = int(2 + 14 * strength)
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    kernel /= kernel.sum() + 1e-8
    blurred = cv2.filter2D(image, -1, kernel)
    return _blend(image, blurred, strength)


def digital_noise_template(image: ArrayImage, strength: float) -> ArrayImage:
    """数字噪声：高斯噪声 + 椒盐噪声。"""

    sigma = 30.0 * strength
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    img = image.astype(np.float32) + noise
    noisy = _clip_uint8(img)
    if strength > 0.3:
        prob = 0.02 * strength
        mask = np.random.rand(*image.shape[:2])
        noisy[mask < prob / 2] = 0
        noisy[mask > 1 - prob / 2] = 255
    return _blend(image, noisy, strength)


def light_dust_template(image: ArrayImage, strength: float) -> ArrayImage:
    """轻度扬尘：暖黄色雾 + 轻度对比度下降。"""

    overlay = image.astype(np.float32)
    overlay[..., 0] = np.clip(overlay[..., 0] + 25 * strength, 0, 255)
    overlay[..., 1] = np.clip(overlay[..., 1] + 15 * strength, 0, 255)
    overlay[..., 2] = np.clip(overlay[..., 2] - 5 * strength, 0, 255)
    overlay = cv2.GaussianBlur(overlay, (5, 5), 0)
    return _blend(image, overlay, strength)


def dappled_light_template(image: ArrayImage, strength: float) -> ArrayImage:
    """斑驳光影：高频乘性噪声模拟树叶光斑。"""

    h, w = image.shape[:2]
    base = np.random.rand(h // 32, w // 32).astype(np.float32)
    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    base = 0.85 + 0.30 * base * strength
    img = image.astype(np.float32) * base[..., None]
    return _clip_uint8(img)


def vintage_style_template(image: ArrayImage, strength: float) -> ArrayImage:
    """复古风格：HSV 偏移到黄褐 + 降饱和 + 暗角。"""

    img = _hue_shift_hsv(image, delta_h=10 * strength)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= 1.0 - 0.45 * strength
    img = cv2.cvtColor(_clip_uint8(hsv), cv2.COLOR_HSV2RGB)

    h, w = image.shape[:2]
    yy, xx = np.indices((h, w))
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / max(cy, cx)
    vignette = 1.0 - 0.45 * strength * np.clip(r, 0, 1) ** 2
    img = img.astype(np.float32) * vignette[..., None]
    return _clip_uint8(img)


def carla_style_template(image: ArrayImage, strength: float) -> ArrayImage:
    """CARLA 风格：低对比 + 颜色微调 + 轻度噪声。"""

    img = image.astype(np.float32)
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = mean + (img - mean) * (1.0 - 0.35 * strength)
    img[..., 0] = np.clip(img[..., 0] + 5 * strength, 0, 255)
    img[..., 2] = np.clip(img[..., 2] - 5 * strength, 0, 255)
    img = _clip_uint8(img)
    noise = np.random.normal(0, 6 * strength, image.shape).astype(np.float32)
    img = _clip_uint8(img + noise)
    return _blend(image, img, strength)


# ----------------------------------------------------------------------
# 注册表
# ----------------------------------------------------------------------


ATTACK_TEMPLATES: Dict[str, AttackTemplate] = {
    "Rain": AttackTemplate(
        name="Rain",
        description="雨线纹理叠加。",
        apply=rain_template,
    ),
    "Snow": AttackTemplate(
        name="Snow",
        description="雪花颗粒 + 高斯模糊。",
        apply=snow_template,
    ),
    "Dusk": AttackTemplate(
        name="Dusk",
        description="暖色黄昏 + 亮度下降。",
        apply=dusk_template,
    ),
    "Dawn": AttackTemplate(
        name="Dawn",
        description="冷色黎明，色温偏蓝紫。",
        apply=dawn_template,
    ),
    "MotionBlur": AttackTemplate(
        name="MotionBlur",
        description="水平方向运动模糊。",
        apply=motion_blur_template,
    ),
    "DigitalNoise": AttackTemplate(
        name="DigitalNoise",
        description="高斯 + 椒盐混合噪声。",
        apply=digital_noise_template,
    ),
    "LightDust": AttackTemplate(
        name="LightDust",
        description="暖色扬尘雾 + 降饱和。",
        apply=light_dust_template,
    ),
    "DappledLight": AttackTemplate(
        name="DappledLight",
        description="乘性高频噪声模拟斑驳光影。",
        apply=dappled_light_template,
    ),
    "VintageStyle": AttackTemplate(
        name="VintageStyle",
        description="复古黄褐 + 暗角。",
        apply=vintage_style_template,
    ),
    "CarlaStyle": AttackTemplate(
        name="CarlaStyle",
        description="CARLA 渲染风格 + 轻噪声。",
        apply=carla_style_template,
    ),
}


def list_attack_names() -> List[str]:
    return list(ATTACK_TEMPLATES.keys())


def get_attack_template(name: str) -> AttackTemplate:
    if name not in ATTACK_TEMPLATES:
        raise KeyError(
            f"Unknown attack template: {name}. "
            f"Available: {list_attack_names()}"
        )
    return ATTACK_TEMPLATES[name]


# ----------------------------------------------------------------------
# 连续攻击空间
# ----------------------------------------------------------------------


DEFAULT_STRENGTHS: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


class ContinuousAttackSpace:
    """连续攻击空间组织器。

    按 (attack_name, strength) 笛卡尔积生成 (scene, attack) 评估点。
    设计文档第 5 节：500 场景 × 10 攻击 × 6 强度 = 30000 次评估。
    """

    def __init__(
        self,
        attack_names: Optional[Sequence[str]] = None,
        strengths: Sequence[float] = DEFAULT_STRENGTHS,
    ) -> None:
        if attack_names is None:
            attack_names = list_attack_names()
        self.attack_names: List[str] = list(attack_names)
        self.strengths: List[float] = list(strengths)

    def __len__(self) -> int:
        return len(self.attack_names) * len(self.strengths)

    def grid(self) -> List[Tuple[str, float]]:
        """生成 (attack, strength) 网格。"""

        return [
            (name, s) for name in self.attack_names for s in self.strengths
        ]

    def evaluate(
        self,
        image: ArrayImage,
        attack_name: str,
        strength: float,
    ) -> ArrayImage:
        template = get_attack_template(attack_name)
        return template(image, strength)

    def evaluate_grid(
        self, image: ArrayImage
    ) -> Dict[Tuple[str, float], ArrayImage]:
        return {key: self.evaluate(image, *key) for key in self.grid()}

    def total_evaluations(self, num_scenes: int) -> int:
        return num_scenes * len(self)

    @classmethod
    def default(cls) -> "ContinuousAttackSpace":
        return cls()
