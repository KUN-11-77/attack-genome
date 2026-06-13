"""YOLO-based semantic gene extractor.

Replaces :class:`SemanticGeneExtractor` (heuristic color proxies)
with real YOLOv8 object detection. Reads from pre-computed CSV
(yolo_gene_extractor.py output), no model loaded at runtime.

Gene fields:
    vehicle_loss — number of vehicles lost under attack
    person_loss — number of persons lost under attack
    detection_loss — total objects lost under attack
    conf_loss — mean detection confidence drop under attack
    vehicle_loss_ratio — fraction of vehicles lost
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from navsim.agents.attack_genome.genes.base import GeneExtractorBase

_YOLO_CACHE: Optional[pd.DataFrame] = None
_YOLO_CSV_PATH: Optional[str] = None


def _load_cache(csv_path: str) -> pd.DataFrame:
    global _YOLO_CACHE, _YOLO_CSV_PATH
    if _YOLO_CACHE is not None and _YOLO_CSV_PATH == csv_path:
        return _YOLO_CACHE
    _YOLO_CACHE = pd.read_csv(csv_path)
    _YOLO_CSV_PATH = csv_path
    return _YOLO_CACHE


@dataclass
class YoloGeneExtractor(GeneExtractorBase):
    """YOLO 检测的退化 gene。只查表，不跑模型。"""

    name: str = "yolo"
    enabled: bool = True

    # Default path on server; overrideable by env var
    csv_path: str = field(
        default_factory=lambda: os.environ.get(
            "YOLO_GENE_CSV",
            "/data3/khsong/cogatedrive/exp/yolo_genes/yolo_genes.csv",
        )
    )

    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """查表返回。``image`` 参数忽略（预计算表中已有）。"""
        # YoloGeneExtractor 不直接处理 image; gene 值由外部注入
        # （通过 ``_build_common_failure_samples`` 的 scene_token/attack/strength 查表）
        return {}


def query_yolo_gene(
    scene_token: str,
    attack: str,
    strength: float,
    csv_path: Optional[str] = None,
) -> Dict[str, float]:
    """查预计算 YOLO gene 表。

    在 ``_build_common_failure_samples`` 中调用，用样本的 (scene, attack, strength)
    查表获取预计算好的检测退化指标。
    """
    if csv_path is None:
        csv_path = os.environ.get(
            "YOLO_GENE_CSV",
            "/data3/khsong/cogatedrive/exp/yolo_genes/yolo_genes.csv",
        )
    df = _load_cache(csv_path)

    row = df[
        (df["scene_token"] == scene_token)
        & (df["attack"] == attack)
        & (abs(df["strength"] - strength) < 0.001)
    ]
    if len(row) == 0:
        return {
            "vehicle_loss": 0.0,
            "person_loss": 0.0,
            "detection_loss": 0.0,
            "conf_loss": 0.0,
            "vehicle_loss_ratio": 0.0,
        }

    r = row.iloc[0]
    return {
        "vehicle_loss": float(r.get("vehicle_loss", 0)),
        "person_loss": float(r.get("person_loss", 0)),
        "detection_loss": float(r.get("total_loss", 0)),
        "conf_loss": float(r.get("conf_loss", 0)),
        "vehicle_loss_ratio": float(r.get("vehicle_loss_ratio", 0)),
    }
