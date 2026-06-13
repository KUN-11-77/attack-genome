"""Representation Family 评估器。

设计文档第 4 节：将 CNN Planner / DINO Planner / CLIP Planner 视作
``RepresentationFamily``。本模块提供：

    - :class:`PlannerAdapter`：把现有 NAVSIM ``AbstractAgent`` 适配
      为 ``(attacked_image) -> trajectory`` 接口。
    - :class:`RepresentationFamily`：组合一个或多个 planner，按相同
      攻击输入计算轨迹 → 评估 ASR / 相变点。
    - :func:`build_default_family`：根据环境与已注册 checkpoint 自动构
      建 CNN / DINO / CLIP 三个 planner。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from navsim.agents.attack_genome.evaluator.metrics import (
    ADE,
    DEFAULT_ADE_THRESHOLD,
    attack_success_from_ade,
    attack_success_rate,
    safety_phase_transition,
)


# ----------------------------------------------------------------------
# 轨迹相关
# ----------------------------------------------------------------------


@dataclass
class TrajectoryPair:
    """一次评估的预测 / 真值对。"""

    predicted: np.ndarray  # (T, 3) or (T, 2)
    gt: np.ndarray  # (T, 3) or (T, 2)
    ade: float = 0.0
    success: bool = False

    @classmethod
    def build(
        cls,
        predicted: np.ndarray,
        gt: np.ndarray,
        threshold: float = DEFAULT_ADE_THRESHOLD,
    ) -> "TrajectoryPair":
        ade = ADE(predicted, gt)
        return cls(
            predicted=predicted,
            gt=gt,
            ade=ade,
            success=attack_success_from_ade(ade, threshold=threshold),
        )


# ----------------------------------------------------------------------
# Planner 适配器
# ----------------------------------------------------------------------


PredictFn = Callable[[np.ndarray], np.ndarray]


class PlannerAdapter:
    """把 NAVSIM ``AbstractAgent`` 抽象为 ``(image) -> trajectory`` 函数。

    对于训练 / 推理分离的模型，外部可传入 ``predict`` 闭包，例如::

        adapter = PlannerAdapter(
            name="CNN-GTRS",
            representation="CNN",
            predict=lambda img: gtrs_agent(img)[..., :2],
        )
    """

    def __init__(
        self,
        name: str,
        representation: str,
        predict: Optional[PredictFn] = None,
        agent: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.representation = representation
        self._agent = agent
        self._predict = predict
        if predict is None and agent is None:
            raise ValueError(
                "PlannerAdapter requires either `predict` or `agent`."
            )

    def predict(self, image: np.ndarray) -> np.ndarray:
        if self._predict is not None:
            return self._predict(image)
        # 默认行为：调用 agent（需要外部预先注入 compute_trajectory 的封装）
        if hasattr(self._agent, "compute_trajectory_from_image"):
            return self._agent.compute_trajectory_from_image(image)
        raise RuntimeError(
            f"PlannerAdapter for {self.name} has no usable predict fn."
        )


# ----------------------------------------------------------------------
# Family 注册
# ----------------------------------------------------------------------


@dataclass
class RepresentationFamilyResult:
    """单次攻击评估的家族级结果。"""

    planner: str
    representation: str
    attack: str
    strength: float
    n: int
    asr: float
    mean_ade: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "planner": self.planner,
            "representation": self.representation,
            "attack": self.attack,
            "strength": self.strength,
            "n": self.n,
            "asr": self.asr,
            "mean_ade": self.mean_ade,
        }


@dataclass
class RepresentationFamily:
    """一组 planner 的集合。"""

    planners: List[PlannerAdapter] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_repr: Dict[str, PlannerAdapter] = {
            p.representation.upper(): p for p in self.planners
        }

    def add(self, planner: PlannerAdapter) -> None:
        self.planners.append(planner)
        self._by_repr[planner.representation.upper()] = planner

    def get(self, representation: str) -> PlannerAdapter:
        if representation.upper() not in self._by_repr:
            raise KeyError(
                f"Representation '{representation}' not in family. "
                f"Available: {list(self._by_repr)}"
            )
        return self._by_repr[representation.upper()]

    def representations(self) -> List[str]:
        return [p.representation.upper() for p in self.planners]

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------
    def evaluate(
        self,
        scenes: Sequence[Dict[str, np.ndarray]],
        attack: str,
        strength: float,
        threshold: float = DEFAULT_ADE_THRESHOLD,
    ) -> List[RepresentationFamilyResult]:
        """对每个 planner 评估 (scene, attack) ASR。

        scenes
            list of dicts::

                {
                    "image": np.ndarray,        # 当前帧（攻击前）
                    "attacked_image": np.ndarray,
                    "gt_trajectory": np.ndarray,
                }
        """

        results: List[RepresentationFamilyResult] = []
        for planner in self.planners:
            pairs: List[TrajectoryPair] = []
            for s in scenes:
                pred = planner.predict(s["attacked_image"])
                pairs.append(
                    TrajectoryPair.build(pred, s["gt_trajectory"], threshold)
                )
            n = len(pairs)
            asr = attack_success_rate([p.success for p in pairs])
            mean_ade = float(np.mean([p.ade for p in pairs])) if pairs else 0.0
            results.append(
                RepresentationFamilyResult(
                    planner=planner.name,
                    representation=planner.representation,
                    attack=attack,
                    strength=strength,
                    n=n,
                    asr=asr,
                    mean_ade=mean_ade,
                )
            )
        return results


class RepresentationFamilyRegistry:
    """全局注册表（单例），按 representation 名查询 planner。"""

    _FAMILIES: Dict[str, RepresentationFamily] = {}

    @classmethod
    def register(cls, name: str, family: RepresentationFamily) -> None:
        cls._FAMILIES[name] = family

    @classmethod
    def get(cls, name: str) -> RepresentationFamily:
        if name not in cls._FAMILIES:
            raise KeyError(f"Family '{name}' not registered.")
        return cls._FAMILIES[name]

    @classmethod
    def all_names(cls) -> List[str]:
        return list(cls._FAMILIES.keys())


def build_default_family(
    cnn_adapter: Optional[PlannerAdapter] = None,
    dino_adapter: Optional[PlannerAdapter] = None,
    clip_adapter: Optional[PlannerAdapter] = None,
) -> RepresentationFamily:
    """构造默认 CNN / DINO / CLIP 家族。允许部分为 None 留空。"""

    family = RepresentationFamily()
    if cnn_adapter is not None:
        family.add(cnn_adapter)
    if dino_adapter is not None:
        family.add(dino_adapter)
    if clip_adapter is not None:
        family.add(clip_adapter)
    return family
