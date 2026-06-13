"""Attack Genome 在 NAVSIM 中的 Agent 适配。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

from navsim.agents.attack_genome.attack_genome_config import AttackGenomeConfig
from navsim.agents.attack_genome.attack_genome_model import AttackGenomeModel

if TYPE_CHECKING:  # pragma: no cover
    from navsim.common.dataclasses import SensorConfig
    from navsim.agents.abstract_agent import AbstractAgent
    from navsim.planning.training.abstract_feature_target_builder import (
        AbstractFeatureBuilder,
        AbstractTargetBuilder,
    )


_AbstractAgentBase: Any = object
_SensorConfig: Any = None


def _get_abstract_agent_base() -> Any:
    """懒加载 ``AbstractAgent``，允许在没有 pytorch_lightning / nuplan
    的环境仅 import 纯 numpy 部分。"""

    global _AbstractAgentBase
    if _AbstractAgentBase is object:
        try:
            from navsim.agents.abstract_agent import AbstractAgent
            _AbstractAgentBase = AbstractAgent
        except Exception:
            _AbstractAgentBase = object
    return _AbstractAgentBase


def _get_sensor_config_cls() -> Any:
    """懒加载 ``SensorConfig``。"""

    global _SensorConfig
    if _SensorConfig is None:
        try:
            from navsim.common.dataclasses import SensorConfig
            _SensorConfig = SensorConfig
        except Exception:
            _SensorConfig = _FallbackSensorConfig
    return _SensorConfig


class _FallbackSensorConfig:
    """无 nuplan 时给 ``get_sensor_config`` 用的最小占位。"""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _safe_pl_callbacks() -> List[Any]:
    try:
        import pytorch_lightning as pl  # noqa: F401
        return []
    except ImportError:
        return []


class AttackGenomeAgent(_get_abstract_agent_base()):
    """Attack Genome 评估 agent（不训练，仅做推理 / 攻击评估）。

    继承 ``AbstractAgent`` 是为了无缝接入 ``SceneLoader / Dataset /
    MetricCacheLoader`` 的现有数据通路。

    在没有 ``pytorch_lightning`` / ``nuplan`` 的环境下，本类退化为
    普通 ``nn.Module``（``object`` 基类），其余功能不受影响。
    """

    def __init__(
        self,
        config: AttackGenomeConfig,
        lr: float = 1e-4,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: Optional[Any] = None,
    ) -> None:
        try:
            super().__init__(
                trajectory_sampling=trajectory_sampling or config.trajectory_sampling
            )
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self.model = AttackGenomeModel(config)
        self.reference_planner: Optional[Any] = None
        self.attack_context: Dict[str, Any] = {}

    def name(self) -> str:
        return "AttackGenomeAgent"

    def initialize(self) -> None:
        if self._checkpoint_path:
            state = torch.load(
                self._checkpoint_path, map_location=torch.device("cpu")
            )
            self.load_state_dict(state, strict=False)

    def get_sensor_config(self) -> Any:
        cls = _get_sensor_config_cls()
        return cls(
            cam_f0=[0, 1, 2, 3],
            cam_l0=[0, 1, 2, 3],
            cam_l1=[],
            cam_l2=[],
            cam_r0=[0, 1, 2, 3],
            cam_r1=[],
            cam_r2=[],
            cam_b0=[],
            lidar_pc=[],
        )

    def get_target_builders(self) -> List[Any]:
        return []

    def get_feature_builders(self) -> List[Any]:
        return []

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "camera_feature" in features:
            image = features["camera_feature"]
        elif "image" in features:
            image = features["image"]
        else:
            raise KeyError(
                "AttackGenomeAgent.forward expects `camera_feature` or `image`."
            )
        ctx = self.attack_context or {}
        attack_name = ctx.get("attack_name", "Rain")
        strength = float(ctx.get("strength", 0.0))
        reference = features.get("reference_image")
        out = self.model(
            image,
            attack_name=attack_name,
            strength=strength,
            reference=reference,
        )
        return {
            "trajectory": features.get("trajectory", torch.zeros(image.shape[0], 8, 3)),
            "genome_features": out.genome.features,
            "attack_name": attack_name,
            "strength": strength,
        }

    def compute_loss(self, features, targets, predictions):
        return torch.tensor(0.0, requires_grad=True)

    def get_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Union[torch.optim.Optimizer, Any]]]:
        return torch.optim.Adam(self.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[Any]:
        return _safe_pl_callbacks()
