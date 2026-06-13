"""跨架构迁移性分析。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return 0.0
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def to_distribution(
    asr_values: Sequence[float], eps: float = 1e-8
) -> np.ndarray:
    """ASR → 离散分布。

    为了让 KL 散度有意义，把每个 ASR 看作概率质量（截断到 [0,1]），
    并做 1-bin 直方图。
    """

    p = np.clip(np.asarray(asr_values, dtype=np.float64), 0.0, 1.0) + eps
    p /= p.sum()
    return p


def kl_divergence(
    p: Sequence[float], q: Sequence[float], eps: float = 1e-8
) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(
    p: Sequence[float], q: Sequence[float], eps: float = 1e-8
) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


@dataclass
class TransferabilityPair:
    """一对表征之间的迁移性度量。"""

    repr_a: str
    repr_b: str
    pearson: float
    kl: float
    js: float
    n_points: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "repr_a": self.repr_a,
            "repr_b": self.repr_b,
            "pearson": self.pearson,
            "kl": self.kl,
            "js": self.js,
            "n_points": self.n_points,
        }


@dataclass
class TransferabilityMatrix:
    """把所有表征两两之间的迁移性度量矩阵化。"""

    representations: List[str]
    pearson_matrix: np.ndarray
    kl_matrix: np.ndarray
    js_matrix: np.ndarray
    pairs: List[TransferabilityPair] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "representations": list(self.representations),
            "pearson": self.pearson_matrix.tolist(),
            "kl": self.kl_matrix.tolist(),
            "js": self.js_matrix.tolist(),
            "pairs": [p.to_dict() for p in self.pairs],
        }

    @classmethod
    def from_asr_sequences(
        cls,
        sequences: Dict[str, Sequence[float]],
    ) -> "TransferabilityMatrix":
        """``sequences``：{representation: ASR 序列}。"""

        reps = list(sequences.keys())
        n = len(reps)
        pear = np.eye(n, dtype=np.float64)
        klm = np.zeros((n, n), dtype=np.float64)
        jsm = np.zeros((n, n), dtype=np.float64)
        pairs: List[TransferabilityPair] = []

        for i, a in enumerate(reps):
            for j, b in enumerate(reps):
                if i == j:
                    continue
                if j < i:
                    pear[i, j] = pear[j, i]
                    klm[i, j] = klm[j, i]
                    jsm[i, j] = jsm[j, i]
                    continue
                pa = np.asarray(sequences[a], dtype=np.float64)
                pb = np.asarray(sequences[b], dtype=np.float64)
                # 长度不匹配时截断到 min
                L = min(len(pa), len(pb))
                pa = pa[:L]
                pb = pb[:L]
                p = pearson_correlation(pa, pb)
                da = to_distribution(pa)
                db = to_distribution(pb)
                k = kl_divergence(da, db)
                js_v = js_divergence(da, db)
                # i/j 已经是 int；为了保持矩阵对称，使用显式 int 索引
                pear[i, j] = p
                pear[j, i] = p
                klm[i, j] = k
                klm[j, i] = k
                jsm[i, j] = js_v
                jsm[j, i] = js_v
                pairs.append(
                    TransferabilityPair(
                        repr_a=a,
                        repr_b=b,
                        pearson=p,
                        kl=k,
                        js=js_v,
                        n_points=L,
                    )
                )
        return cls(
            representations=reps,
            pearson_matrix=pear,
            kl_matrix=klm,
            js_matrix=jsm,
            pairs=pairs,
        )

    def top_migrating_pairs(
        self, top_k: int = 5, by: str = "pearson"
    ) -> List[TransferabilityPair]:
        if by not in {"pearson", "kl", "js"}:
            raise ValueError("`by` must be one of: pearson, kl, js")
        if by == "pearson":
            sorted_pairs = sorted(
                self.pairs, key=lambda p: -abs(p.pearson)
            )
        elif by == "kl":
            sorted_pairs = sorted(self.pairs, key=lambda p: p.kl)
        else:
            sorted_pairs = sorted(self.pairs, key=lambda p: p.js)
        return sorted_pairs[:top_k]
