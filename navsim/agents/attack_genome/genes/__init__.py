"""
Attack Genome 基因提取器集合。

按设计文档第 3 节：

- 一级基因：Frequency / Color / Texture / Contrast
- 二级基因：Structural / Semantic / Illumination

每个模块提供统一的 ``extract(image) -> dict`` 接口，方便组合成
完整基因指纹。
"""

from navsim.agents.attack_genome.genes.frequency_gene import FrequencyGeneExtractor
from navsim.agents.attack_genome.genes.color_gene import ColorGeneExtractor
from navsim.agents.attack_genome.genes.texture_gene import TextureGeneExtractor
from navsim.agents.attack_genome.genes.contrast_gene import ContrastGeneExtractor
from navsim.agents.attack_genome.genes.structural_gene import StructuralGeneExtractor
from navsim.agents.attack_genome.genes.semantic_gene import SemanticGeneExtractor
from navsim.agents.attack_genome.genes.illumination_gene import IlluminationGeneExtractor
from navsim.agents.attack_genome.genes.genome_pipeline import (
    AttackGenomeExtractor,
    AttackGenomeRecord,
    extract_genome_from_array,
)

__all__ = [
    "FrequencyGeneExtractor",
    "ColorGeneExtractor",
    "TextureGeneExtractor",
    "ContrastGeneExtractor",
    "StructuralGeneExtractor",
    "SemanticGeneExtractor",
    "IlluminationGeneExtractor",
    "AttackGenomeExtractor",
    "AttackGenomeRecord",
    "extract_genome_from_array",
]
