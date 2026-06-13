"""Attack Genome 主配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple


def _default_trajectory_sampling() -> Any:
    """懒加载 ``TrajectorySampling``，允许在无 nuplan 环境导入。"""

    try:
        from nuplan.planning.simulation.trajectory.trajectory_sampling import (
            TrajectorySampling,
        )
    except ImportError:
        return None
    return TrajectorySampling(time_horizon=4, interval_length=0.5)


@dataclass
class AttackGenomeConfig:
    """Attack Genome 运行总配置。"""

    # 攻击模板与强度
    attack_names: Tuple[str, ...] = field(
        default_factory=lambda: (
            "Rain", "Snow", "Dusk", "Dawn", "MotionBlur",
            "DigitalNoise", "LightDust", "DappledLight",
            "VintageStyle", "CarlaStyle",
        )
    )
    strengths: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    # 表征族
    representations: Tuple[str, ...] = ("CNN", "DINO", "CLIP")

    # 攻击成功阈值（设计文档 6.1）
    ade_threshold: float = 1.0
    # 相变点目标 ASR（设计文档 6.3）
    s_c_target_asr: float = 0.5

    # 共同失效挖掘参数
    common_failure_eps: float = 0.5
    common_failure_min_samples: int = 5

    # Vulnerability Atlas
    vulnerability_top_k: int = 50
    vulnerability_focus_attributes: Tuple[str, ...] = (
        "night", "curve", "high_traffic", "low_light", "road_occlusion",
    )

    # 路径
    output_dir: str = "outputs/attack_genome"

    # 默认轨迹采样（懒加载）
    trajectory_sampling: Optional[Any] = field(
        default_factory=_default_trajectory_sampling
    )

    # 随机种子
    seed: int = 42
