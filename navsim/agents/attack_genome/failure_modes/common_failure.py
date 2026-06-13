"""共同失效模式挖掘。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from navsim.agents.attack_genome.genes.genome_pipeline import (
    AttackGenomeRecord,
    GENOME_FIELD_ORDER,
)


@dataclass
class CommonFailureSample:
    """单个 (scene, attack) 上的共同失效样本。"""

    scene_token: str
    attack: str
    strength: float
    # 每个表征在该样本上是否失效
    repr_failures: Dict[str, bool]
    # 攻击基因指纹
    genome: AttackGenomeRecord
    # 场景属性（可由外部注入，例如夜间 / 弯道 / 高车流）
    scene_attributes: Dict[str, Any] = field(default_factory=dict)

    def is_common_failure(self, required: Sequence[str]) -> bool:
        return all(self.repr_failures.get(r, False) for r in required)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，供 Failure Factor Discovery 训练使用。

        关键：保留 ``repr_failures``（每个 planner 的 success 标签）和
        ``genome.features``（完整 gene 字段），让下游 XGBoost/RF 能直接
        ``Gene × Planner → Success`` 训练。
        """
        return {
            "scene_token": self.scene_token,
            "attack": self.attack,
            "strength": float(self.strength),
            "repr_failures": dict(self.repr_failures),
            "genome": self.genome.to_dict() if self.genome is not None else {},
            "scene_attributes": dict(self.scene_attributes),
        }


@dataclass
class CommonFailureCluster:
    """DBSCAN 聚类得到的簇。"""

    cluster_id: int
    size: int
    centroid: np.ndarray
    # 该簇中出现频率最高的攻击 / 场景属性
    dominant_attacks: List[str]
    dominant_attributes: List[str]
    sample_tokens: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "centroid": self.centroid.tolist(),
            "dominant_attacks": self.dominant_attacks,
            "dominant_attributes": self.dominant_attributes,
            "sample_tokens": self.sample_tokens,
        }


@dataclass
class CommonFailureAtlas:
    """所有簇的集合。"""

    clusters: List[CommonFailureCluster]
    n_samples: int
    noise_samples: int
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_samples": self.n_samples,
            "noise_samples": self.noise_samples,
            "config": self.config,
            "clusters": [c.to_dict() for c in self.clusters],
        }


# ----------------------------------------------------------------------
# 核心实现
# ----------------------------------------------------------------------


def _zscore(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True) + 1e-8
    return (matrix - mean) / std


def _topk(values: Dict[str, int], k: int) -> List[str]:
    return [k_ for k_, _ in sorted(values.items(), key=lambda kv: -kv[1])[:k]]


def common_failure_mining(
    samples: Sequence[CommonFailureSample],
    *,
    representations: Sequence[str],
    eps: float = 0.5,
    min_samples: int = 5,
    top_k: int = 3,
    use_fields: Optional[Sequence[str]] = None,
) -> CommonFailureAtlas:
    """设计文档第 7 节挖掘流程。

    1. 收集所有 (repr_i 都失败) 的样本
    2. 用基因指纹 + 场景属性组成特征向量
    3. DBSCAN 聚类
    4. 输出簇中心和簇代表特征
    """

    # Step 1
    common_samples = [
        s for s in samples if s.is_common_failure(representations)
    ]
    n_total = len(samples)
    n_common = len(common_samples)
    if n_common == 0:
        return CommonFailureAtlas(
            clusters=[],
            n_samples=n_total,
            noise_samples=0,
            config={
                "eps": eps,
                "min_samples": min_samples,
                "representations": list(representations),
            },
        )

    # Step 2：构建特征矩阵
    if use_fields is None:
        use_fields = list(GENOME_FIELD_ORDER)
    genome_matrix = np.stack(
        [s.genome.to_vector(use_fields) for s in common_samples],
        axis=0,
    )
    # 收集所有出现过的 scene attribute key
    attr_keys: List[str] = sorted(
        {k for s in common_samples for k in s.scene_attributes.keys()}
    )
    attr_matrix = np.zeros((n_common, len(attr_keys)), dtype=np.float32)
    for i, s in enumerate(common_samples):
        for j, k in enumerate(attr_keys):
            v = s.scene_attributes.get(k, 0)
            try:
                attr_matrix[i, j] = float(v)
            except (TypeError, ValueError):
                attr_matrix[i, j] = 0.0

    feature = np.concatenate([genome_matrix, attr_matrix], axis=1)
    feature = _zscore(feature)

    # Step 3：DBSCAN
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "common_failure_mining 需要 scikit-learn，请先 pip install scikit-learn"
        ) from e

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(feature)

    # Step 4：聚合簇
    clusters: List[CommonFailureCluster] = []
    noise_count = int((labels == -1).sum())
    for cid in sorted(set(labels) - {-1}):
        idx = np.where(labels == cid)[0]
        cluster_samples = [common_samples[i] for i in idx]
        centroid = feature[idx].mean(axis=0)
        # 主导攻击
        attack_count: Dict[str, int] = {}
        for s in cluster_samples:
            attack_count[s.attack] = attack_count.get(s.attack, 0) + 1
        # 主导场景属性（值非 0）
        attr_count: Dict[str, int] = {}
        for s in cluster_samples:
            for k, v in s.scene_attributes.items():
                try:
                    if float(v) > 0:
                        attr_count[k] = attr_count.get(k, 0) + 1
                except (TypeError, ValueError):
                    continue
        clusters.append(
            CommonFailureCluster(
                cluster_id=int(cid),
                size=int(len(idx)),
                centroid=centroid.astype(np.float32),
                dominant_attacks=_topk(attack_count, top_k),
                dominant_attributes=_topk(attr_count, top_k),
                sample_tokens=[s.scene_token for s in cluster_samples],
            )
        )

    return CommonFailureAtlas(
        clusters=clusters,
        n_samples=n_total,
        noise_samples=noise_count,
        config={
            "eps": eps,
            "min_samples": min_samples,
            "representations": list(representations),
            "use_fields": list(use_fields),
            "attribute_keys": attr_keys,
        },
    )
