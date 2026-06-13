"""跨架构迁移性分析。

设计文档第 6.4 节要求计算：
    - Pearson 相关
    - KL 散度

输入是 (attack, strength) 上的 ASR / mean_ade 序列。我们提供：

    - :func:`pearson_correlation`
    - :func:`kl_divergence`
    - :func:`js_divergence`（对称版，更适合做 ASR 分布对比）
    - :class:`TransferabilityMatrix`：把所有 (repr_i, repr_j) 之间的
      Pearson / KL 计算成矩阵，方便后续分析哪些迁移性强。
"""

from navsim.agents.attack_genome.transferability.transfer_analysis import (
    TransferabilityMatrix,
    TransferabilityPair,
    js_divergence,
    kl_divergence,
    pearson_correlation,
    to_distribution,
)

__all__ = [
    "TransferabilityMatrix",
    "TransferabilityPair",
    "js_divergence",
    "kl_divergence",
    "pearson_correlation",
    "to_distribution",
]
