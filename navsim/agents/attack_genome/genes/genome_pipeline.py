"""Attack Genome 特征组合与序列化。

提供：
    - :class:`AttackGenomeRecord`：单张图像的完整基因指纹
    - :class:`AttackGenomeExtractor`：组合所有子基因提取器
    - :func:`extract_genome_from_array`：便捷入口
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from navsim.agents.attack_genome.genes.color_gene import ColorGeneExtractor
from navsim.agents.attack_genome.genes.contrast_gene import ContrastGeneExtractor
from navsim.agents.attack_genome.genes.frequency_gene import FrequencyGeneExtractor
from navsim.agents.attack_genome.genes.illumination_gene import IlluminationGeneExtractor
from navsim.agents.attack_genome.genes.semantic_gene import SemanticGeneExtractor
from navsim.agents.attack_genome.genes.structural_gene import StructuralGeneExtractor
from navsim.agents.attack_genome.genes.texture_gene import TextureGeneExtractor


# 所有一级 / 二级基因的字段顺序，用于聚类时向量化
GENOME_FIELD_ORDER: List[str] = [
    # frequency
    "low_freq_ratio", "mid_freq_ratio", "high_freq_ratio", "spectral_centroid",
    # color
    "hue_mean", "hue_std", "sat_mean", "sat_std", "val_mean", "val_std", "colorfulness",
    # texture
    "lbp_entropy", "glcm_contrast", "lbp_uniformity",
    # contrast
    "rms_contrast", "mean_luma", "std_luma", "dynamic_range",
    "mean_shift", "std_shift",
    # structural
    "edge_mean", "edge_density", "road_luma_mean", "road_luma_std",
    "lane_line_count", "lane_line_density",
    # semantic (heuristic proxy — see design.md §3.4.2)
    "vehicle_saliency", "pedestrian_saliency",
    "traffic_saliency", "lane_yellow_ratio",
    # illumination
    "shadow_ratio", "highlight_ratio", "luma_entropy", "luma_skew", "luma_mean",
]


@dataclass
class AttackGenomeRecord:
    """单张图像的完整基因指纹。"""

    scene_token: Optional[str] = None
    attack_name: Optional[str] = None
    attack_strength: Optional[float] = None
    # 字段顺序见 ``GENOME_FIELD_ORDER``；冗余保存以便扩展
    features: Dict[str, float] = field(default_factory=dict)

    def to_vector(self, fields: Optional[Sequence[str]] = None) -> np.ndarray:
        if fields is None:
            fields = GENOME_FIELD_ORDER
        vec = []
        for k in fields:
            v = self.features.get(k, 0.0)
            vec.append(float(v) if v is not None else 0.0)
        return np.asarray(vec, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_token": self.scene_token,
            "attack_name": self.attack_name,
            "attack_strength": self.attack_strength,
            "features": dict(self.features),
        }


class AttackGenomeExtractor:
    """Attack Genome 组合提取器。"""

    def __init__(self, extractors: Optional[Sequence[Any]] = None) -> None:
        if extractors is None:
            extractors = [
                FrequencyGeneExtractor(),
                ColorGeneExtractor(),
                TextureGeneExtractor(),
                ContrastGeneExtractor(),
                StructuralGeneExtractor(),
                SemanticGeneExtractor(),
                IlluminationGeneExtractor(),
            ]
        self.extractors = list(extractors)

    def __call__(
        self,
        image: np.ndarray,
        reference: Optional[np.ndarray] = None,
        record: Optional[AttackGenomeRecord] = None,
        yolo_context: Optional[Dict[str, Any]] = None,
    ) -> AttackGenomeRecord:
        if record is None:
            record = AttackGenomeRecord()
        # 把 reference 注入对比度基因以计算 mean_shift
        for ext in self.extractors:
            if isinstance(ext, ContrastGeneExtractor):
                ext.reference = reference
            feats = ext(image)
            record.features.update(feats)
        return record


def extract_genome_from_array(
    image: np.ndarray,
    *,
    scene_token: Optional[str] = None,
    attack_name: Optional[str] = None,
    attack_strength: Optional[float] = None,
    reference: Optional[np.ndarray] = None,
) -> AttackGenomeRecord:
    """便捷函数：图像 → AttackGenomeRecord。"""

    extractor = AttackGenomeExtractor()
    record = AttackGenomeRecord(
        scene_token=scene_token,
        attack_name=attack_name,
        attack_strength=attack_strength,
    )
    return extractor(image, reference=reference, record=record)
