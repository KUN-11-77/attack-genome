"""Attack Genome 模型层。

本身不实现端到端 planner，而是把：
    1. 攻击模板（``ContinuousAttackSpace``）
    2. 基因提取器（``AttackGenomeExtractor``）
    3. 表征族评估器（``RepresentationFamily``）
封装为一个轻量 ``nn.Module``，方便在 ``pl.LightningModule`` 或
``AbstractAgent`` 中复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from navsim.agents.attack_genome.attack_genome_config import AttackGenomeConfig
from navsim.agents.attack_genome.attacks.templates import ContinuousAttackSpace
from navsim.agents.attack_genome.genes.genome_pipeline import (
    AttackGenomeExtractor,
    AttackGenomeRecord,
)


@dataclass
class AttackGenomeModelOutput:
    """``forward`` 的输出（仅用于链路验证，不参与训练 loss）。"""

    genome: AttackGenomeRecord
    attack_name: str
    strength: float


class AttackGenomeModel(nn.Module):
    """Attack Genome 统一模型。"""

    def __init__(self, config: AttackGenomeConfig) -> None:
        super().__init__()
        self.config = config
        self.attack_space = ContinuousAttackSpace(
            attack_names=list(config.attack_names),
            strengths=list(config.strengths),
        )
        self.genome_extractor = AttackGenomeExtractor()

    def forward(
        self,
        image: torch.Tensor,
        attack_name: str = "Rain",
        strength: float = 0.0,
        reference: Optional[torch.Tensor] = None,
    ) -> AttackGenomeModelOutput:
        """``image``：``(B, 3, H, W)`` 的 batch。

        返回的 ``genome.features`` 总是 batch 中第 0 张图的基因（用于
        链路验证）。批量提取请直接调用 ``self.genome_extractor``。
        """

        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError(
                f"AttackGenomeModel expects (B, 3, H, W); got {tuple(image.shape)}"
            )
        np_img = image[0].permute(1, 2, 0).detach().cpu().numpy()
        ref_np = (
            reference[0].permute(1, 2, 0).detach().cpu().numpy()
            if reference is not None
            else None
        )
        attacked = self.attack_space.evaluate(np_img, attack_name, strength)
        record = self.genome_extractor(
            attacked, reference=ref_np
        )
        record.attack_name = attack_name
        record.attack_strength = strength
        return AttackGenomeModelOutput(
            genome=record,
            attack_name=attack_name,
            strength=strength,
        )
