"""共同失效模式（Common Failure Modes）。

设计文档第 7 节：
    Step1: 收集共同失效样本 (CNN 失败 ∧ DINO 失败 ∧ CLIP 失败)
    Step2: 提取场景属性 + 攻击基因
    Step3: DBSCAN 聚类
    Step4: 生成 Common Failure Atlas
"""

from navsim.agents.attack_genome.failure_modes.common_failure import (
    CommonFailureAtlas,
    CommonFailureCluster,
    CommonFailureSample,
    common_failure_mining,
)

__all__ = [
    "CommonFailureAtlas",
    "CommonFailureCluster",
    "CommonFailureSample",
    "common_failure_mining",
]
