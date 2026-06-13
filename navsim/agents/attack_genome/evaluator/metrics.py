"""评估指标。

设计文档第 6 节：
    - ADE（Average Displacement Error）
    - 攻击成功判定：``ADE > 1m``
    - ASR（Attack Success Rate）
    - 安全相变点 s_c：ASR 首次 ≥ 50% 时的攻击强度
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_ADE_THRESHOLD: float = 1.0  # 设计文档 6.1


def ADE(predicted: np.ndarray, gt: np.ndarray) -> float:
    """平均位移误差 (m)。

    Parameters
    ----------
    predicted, gt:
        shape ``(T, 2 or 3)``，会自动忽略第 3 维（heading）。
    """

    p = np.asarray(predicted, dtype=np.float32)[..., :2]
    g = np.asarray(gt, dtype=np.float32)[..., :2]
    if p.shape != g.shape:
        raise ValueError(
            f"Trajectory shapes must match: predicted={p.shape}, gt={g.shape}"
        )
    diff = p - g
    return float(np.linalg.norm(diff, axis=-1).mean())


def attack_success_from_ade(
    ade: float, threshold: float = DEFAULT_ADE_THRESHOLD
) -> bool:
    """``ADE > threshold`` → 攻击成功。"""

    return ade > threshold


def attack_success_rate(
    successes: Sequence[bool], total: Optional[int] = None
) -> float:
    n = total if total is not None else len(successes)
    if n == 0:
        return 0.0
    return float(np.count_nonzero(successes)) / float(n)


def asr_curve(
    strengths: Sequence[float],
    successes_by_strength: Dict[float, Sequence[bool]],
    threshold: float = DEFAULT_ADE_THRESHOLD,
) -> Dict[float, float]:
    """对每个强度计算 ASR。"""

    curve: Dict[float, float] = {}
    for s in strengths:
        succ = successes_by_strength.get(float(s), [])
        curve[float(s)] = attack_success_rate(succ)
    return curve


def safety_phase_transition(
    strengths: Sequence[float],
    asrs: Sequence[float],
    target_asr: float = 0.5,
) -> Optional[float]:
    """``s_c``：ASR 首次达到 ``target_asr`` 时的攻击强度。

    实现：单调性假设下用线性插值估计首达点。如果整条 ASR 曲线都未触
    及 ``target_asr``，返回 ``None``。
    """

    if not strengths or not asrs:
        return None
    strengths_sorted = sorted(zip(strengths, asrs))
    prev_s, prev_a = strengths_sorted[0]
    if prev_a >= target_asr:
        return float(prev_s)
    for s, a in strengths_sorted[1:]:
        if a >= target_asr:
            # 在 (prev_s, prev_a) → (s, a) 之间线性插值
            if a == prev_a:
                return float(s)
            ratio = (target_asr - prev_a) / (a - prev_a)
            return float(prev_s + ratio * (s - prev_s))
        prev_s, prev_a = s, a
    return None


@dataclass
class ASRCurve:
    strengths: List[float]
    asrs: List[float]
    s_c: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "strengths": list(self.strengths),
            "asrs": list(self.asrs),
            "s_c": self.s_c,
        }
