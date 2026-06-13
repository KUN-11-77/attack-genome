"""Attack Genome 主 pipeline。

设计文档对应：
    1. 加载场景 + 运行 3 个 planner，得到 clean 轨迹
    2. 对 (500 场景 × 10 攻击 × 6 强度) 共 30000 次评估
    3. 计算每个 (planner, attack) 的 ASR 曲线与 s_c
    4. 计算跨架构迁移性矩阵
    5. 共同失效挖掘 → Common Failure Atlas
    6. 构建 Vulnerability Atlas
    7. 汇总输出 json / csv

设计目标：
    - 上层只关心 ``run_pipeline(scenes, family, output_dir)``
    - 内部全部模块都接受最小化接口（numpy 数组 / 字典），方便注入
      真实 NAVSIM Scene 或合成数据。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from navsim.agents.attack_genome.attack_genome_config import AttackGenomeConfig
from navsim.agents.attack_genome.attacks.templates import ContinuousAttackSpace
from navsim.agents.attack_genome.evaluator.metrics import (
    ADE,
    DEFAULT_ADE_THRESHOLD,
    asr_curve,
    attack_success_from_ade,
    safety_phase_transition,
)
from navsim.agents.attack_genome.evaluator.representation_family import (
    PlannerAdapter,
    RepresentationFamily,
    RepresentationFamilyResult,
    TrajectoryPair,
)
from navsim.agents.attack_genome.failure_modes.common_failure import (
    CommonFailureAtlas,
    CommonFailureSample,
    common_failure_mining,
)
from navsim.agents.attack_genome.genes.genome_pipeline import (
    AttackGenomeExtractor,
    AttackGenomeRecord,
)
from navsim.agents.attack_genome.transferability.transfer_analysis import (
    TransferabilityMatrix,
)
from navsim.agents.attack_genome.vulnerability.vulnerability_atlas import (
    VulnerabilityAtlas,
    build_vulnerability_atlas,
)


# ----------------------------------------------------------------------
# 输入数据结构
# ----------------------------------------------------------------------


@dataclass
class AttackGenomeScene:
    """一次评估所需的最小场景数据。"""

    scene_token: str
    image: np.ndarray  # (H, W, 3) uint8
    gt_trajectory: np.ndarray  # (T, 3)
    attributes: Dict[str, Any] = field(default_factory=dict)
    # 可选：clean trajectory（基线）
    clean_trajectory: Optional[np.ndarray] = None


# ----------------------------------------------------------------------
# 结果聚合
# ----------------------------------------------------------------------


@dataclass
class AttackGenomeResult:
    """整个 pipeline 的输出容器。"""

    config: Dict[str, Any]
    asr_curves: Dict[str, Dict[str, Any]]  # {repr: {attack: {strengths, asrs, s_c}}}
    transferability: TransferabilityMatrix
    common_failure_atlas: CommonFailureAtlas
    vulnerability_atlas: VulnerabilityAtlas
    n_scenes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "n_scenes": self.n_scenes,
            "asr_curves": self.asr_curves,
            "transferability": self.transferability.to_dict(),
            "common_failure_atlas": self.common_failure_atlas.to_dict(),
            "vulnerability_atlas": self.vulnerability_atlas.to_dict(),
        }


# ----------------------------------------------------------------------
# 评估核心
# ----------------------------------------------------------------------


def _build_scene_dicts(
    scenes: Sequence[AttackGenomeScene],
    attack_space: ContinuousAttackSpace,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in scenes:
        for attack in attack_space.attack_names:
            for strength in attack_space.strengths:
                attacked = attack_space.evaluate(s.image, attack, strength)
                out.append(
                    {
                        "scene_token": s.scene_token,
                        "attack": attack,
                        "strength": strength,
                        "image": s.image,
                        "attacked_image": attacked,
                        "gt_trajectory": s.gt_trajectory,
                        "attributes": s.attributes,
                    }
                )
    return out


def _compute_clean_predictions(
    scenes: Sequence[AttackGenomeScene],
    family: RepresentationFamily,
) -> Dict[str, Dict[str, np.ndarray]]:
    """预计算每个场景的 clean 轨迹。{scene_token: {repr: clean_trajectory}}"""
    clean: Dict[str, Dict[str, np.ndarray]] = {}
    for s in scenes:
        clean[s.scene_token] = {}
        for planner in family.planners:
            clean[s.scene_token][planner.representation.upper()] = planner.predict(s.image)
    return clean


def _aggregate_asr_curves(
    samples: Sequence[Dict[str, Any]],
    family: RepresentationFamily,
    config: AttackGenomeConfig,
    clean_predictions: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str, float], Dict[str, TrajectoryPair]]]:
    """``{repr: {attack: {strengths, asrs, s_c}}}`` + forward cache。

    攻击成功判定：``ADE(attacked_pred, clean_pred) > threshold``，
    而非 ``ADE(pred, gt)``。只有 planner 在 clean 图上能正常驾驶的场景，
    攻击评估才有意义。

    同时返回 ``forward_cache``：``{(scene_token, attack, strength) -> {repr: TrajectoryPair}}``，
    供 ``_build_common_failure_samples`` 复用，避免重复 forward。
    """

    import time
    from tqdm import tqdm

    by_repr: Dict[str, Dict[str, Dict[float, List[bool]]]] = {}
    by_repr_ade: Dict[str, Dict[str, Dict[float, List[float]]]] = {}
    for planner in family.planners:
        by_repr[planner.representation.upper()] = {}
        by_repr_ade[planner.representation.upper()] = {}

    forward_cache: Dict[Tuple[str, str, float], Dict[str, TrajectoryPair]] = {}

    total = len(samples) * len(family.planners)
    pbar = tqdm(
        total=total,
        desc="[ASR] forward",
        dynamic_ncols=True,
        mininterval=1.0,
        unit="fwd",
    )
    t_start = time.time()
    last_log_t = t_start
    n_done = 0
    for s in samples:
        key = (s["scene_token"], s["attack"], float(s["strength"]))
        for planner in family.planners:
            repr_name = planner.representation.upper()
            pred = planner.predict(s["attacked_image"])
            # 攻击成功 = 相对 clean 轨迹的偏移 > threshold
            if clean_predictions is not None and s["scene_token"] in clean_predictions:
                clean_pred = clean_predictions[s["scene_token"]].get(repr_name)
                if clean_pred is not None:
                    ade = ADE(pred, clean_pred)
                    success = attack_success_from_ade(ade, threshold=config.ade_threshold)
                else:
                    pair = TrajectoryPair.build(pred, s["gt_trajectory"], threshold=config.ade_threshold)
                    ade, success = pair.ade, pair.success
            else:
                pair = TrajectoryPair.build(pred, s["gt_trajectory"], threshold=config.ade_threshold)
                ade, success = pair.ade, pair.success
            # 仍然存为 TrajectoryPair 保持兼容（但 gt 字段填 clean_pred）
            pair = TrajectoryPair(
                predicted=pred,
                gt=clean_predictions[s["scene_token"]][repr_name] if (clean_predictions and s["scene_token"] in clean_predictions and repr_name in clean_predictions[s["scene_token"]]) else s["gt_trajectory"],
                ade=ade,
                success=success,
            )
            by_attack = by_repr[repr_name].setdefault(s["attack"], {})
            by_attack.setdefault(float(s["strength"]), []).append(success)
            by_repr_ade[repr_name].setdefault(s["attack"], {}).setdefault(float(s["strength"]), []).append(ade)
            forward_cache.setdefault(key, {})[repr_name] = pair
            n_done += 1
            pbar.update(1)
            now = time.time()
            if now - last_log_t > 30:
                elapsed = now - t_start
                speed = n_done / elapsed
                eta = (total - n_done) / speed if speed > 0 else float("inf")
                eta_str = (
                    f"{int(eta // 60)}m{int(eta % 60):02d}s"
                    if eta != float("inf") else "?"
                )
                print(
                    f"  [progress] {n_done}/{total} "
                    f"({100*n_done/total:.1f}%) "
                    f"elapsed={int(elapsed//60)}m{int(elapsed%60):02d}s "
                    f"eta={eta_str} "
                    f"speed={speed:.2f} fwd/s",
                    flush=True,
                )
                last_log_t = now
    pbar.close()
    elapsed = time.time() - t_start
    print(f"  [progress] ASR done: {n_done}/{total} in {int(elapsed//60)}m{int(elapsed%60):02d}s", flush=True)

    out: Dict[str, Dict[str, Any]] = {}
    for repr_name, attacks in by_repr.items():
        out[repr_name] = {}
        for attack, strength_map in attacks.items():
            strengths = sorted(strength_map.keys())
            asrs = [asr_curve(strengths, {s: strength_map[s]})[s] for s in strengths]
            s_c = safety_phase_transition(
                strengths, asrs, target_asr=config.s_c_target_asr
            )
            mean_ades = [
                float(np.mean(by_repr_ade.get(repr_name, {}).get(attack, {}).get(s, [0])))
                for s in strengths
            ]
            out[repr_name][attack] = {
                "strengths": strengths,
                "asrs": asrs,
                "s_c": s_c,
                "mean_ade": mean_ades,
            }
    return out, forward_cache


def _build_common_failure_samples(
    samples: Sequence[Dict[str, Any]],
    family: RepresentationFamily,
    config: AttackGenomeConfig,
    forward_cache: Optional[Dict[Tuple[str, str, float], Dict[str, TrajectoryPair]]] = None,
) -> List[CommonFailureSample]:
    """对每个 (scene, attack) 跑所有 planner，组装样本。

    ``forward_cache``：如果提供，对每个 (scene_token, attack, strength) 直接
    复用 ``_aggregate_asr_curves`` 已算好的 ``TrajectoryPair``，避免重复
    forward。结果与无 cache 路径**完全 bit-exact 等价**。
    """

    extractor = AttackGenomeExtractor()
    out: List[CommonFailureSample] = []
    by_key: Dict[tuple, Dict[str, Any]] = {}
    for s in samples:
        key = (s["scene_token"], s["attack"], float(s["strength"]))
        by_key.setdefault(key, s)

    import time
    from tqdm import tqdm
    pbar = tqdm(
        total=len(by_key),
        desc="[CF] mining",
        dynamic_ncols=True,
        mininterval=1.0,
        unit="sample",
    )
    t_start = time.time()
    n_done = 0
    for key, s in by_key.items():
        scene_token, attack, strength = key
        repr_failures: Dict[str, bool] = {}
        if forward_cache is not None and key in forward_cache:
            cached = forward_cache[key]
            for repr_name, pair in cached.items():
                repr_failures[repr_name] = pair.success
        else:
            for planner in family.planners:
                pred = planner.predict(s["attacked_image"])
                pair = TrajectoryPair.build(
                    pred, s["gt_trajectory"], threshold=config.ade_threshold
                )
                repr_failures[planner.representation.upper()] = pair.success
        record = extractor(
            s["attacked_image"],
            reference=s["image"],
            yolo_context={
                "scene_token": scene_token,
                "attack": attack,
                "strength": strength,
            },
        )
        record.scene_token = scene_token
        record.attack_name = attack
        record.attack_strength = strength
        out.append(
            CommonFailureSample(
                scene_token=scene_token,
                attack=attack,
                strength=strength,
                repr_failures=repr_failures,
                genome=record,
                scene_attributes=dict(s.get("attributes", {})),
            )
        )
        n_done += 1
        pbar.update(1)
    pbar.close()
    elapsed = time.time() - t_start
    print(f"  [progress] CF done: {n_done}/{len(by_key)} in {int(elapsed//60)}m{int(elapsed%60):02d}s", flush=True)
    return out


# ----------------------------------------------------------------------
# 顶层入口
# ----------------------------------------------------------------------


def run_pipeline(
    scenes: Sequence[AttackGenomeScene],
    family: RepresentationFamily,
    config: Optional[AttackGenomeConfig] = None,
    output_dir: Optional[str] = None,
) -> AttackGenomeResult:
    """运行完整 pipeline。

    Parameters
    ----------
    scenes:
        输入场景列表
    family:
        至少包含 2 个 PlannerAdapter 的表征族
    config:
        默认使用 :class:`AttackGenomeConfig`
    output_dir:
        若指定，会把结果以 ``result.json`` 持久化。
    """

    cfg = config or AttackGenomeConfig()
    attack_space = ContinuousAttackSpace(
        attack_names=list(cfg.attack_names),
        strengths=list(cfg.strengths),
    )

    if len(family.planners) < 1:
        raise ValueError(
            "run_pipeline requires at least 1 PlannerAdapter."
        )
    if len(family.planners) < 2:
        # 单 planner 模式：跳过 transferability（将在 merge 阶段跨 shard 算）
        import logging
        logging.getLogger(__name__).warning(
            "run_pipeline: only 1 planner, transferability will be skipped per-shard "
            "and computed at merge time."
        )

    # Step 1：构造评估网格
    samples = _build_scene_dicts(scenes, attack_space)

    # Step 1.5：预计算 clean 轨迹（攻击成功 = ADE(attacked, clean) > threshold）
    clean_predictions = _compute_clean_predictions(scenes, family)

    # Step 2：ASR 曲线 + s_c + forward cache
    asr_curves, forward_cache = _aggregate_asr_curves(samples, family, cfg, clean_predictions=clean_predictions)

    # Step 3：迁移性矩阵（每个攻击类型算一次，最后取平均）
    transferability = _build_transferability(asr_curves, family)

    # Step 4：共同失效挖掘（复用 forward cache，0 重复 forward）
    cf_samples = _build_common_failure_samples(
        samples, family, cfg, forward_cache=forward_cache
    )
    common_atlas = common_failure_mining(
        cf_samples,
        representations=cfg.representations,
        eps=cfg.common_failure_eps,
        min_samples=cfg.common_failure_min_samples,
    )

    # Step 5：Vulnerability Atlas
    vuln_atlas = build_vulnerability_atlas(
        cf_samples,
        focus_attributes=cfg.vulnerability_focus_attributes,
        representations=cfg.representations,
        top_k=cfg.vulnerability_top_k,
    )

    result = AttackGenomeResult(
        config={"attack_names": list(cfg.attack_names),
                "strengths": list(cfg.strengths),
                "representations": list(cfg.representations),
                "ade_threshold": cfg.ade_threshold,
                "s_c_target_asr": cfg.s_c_target_asr},
        asr_curves=asr_curves,
        transferability=transferability,
        common_failure_atlas=common_atlas,
        vulnerability_atlas=vuln_atlas,
        n_scenes=len(scenes),
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "result.json"), "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        # === Failure Factor Discovery 数据落盘 ===
        # 每个 (scene, attack, strength) → 完整 gene 向量 + 每个 planner 的 success 标签
        # 下游 XGBoost / RF 用 ``Gene × Planner → Success`` 训练
        _dump_per_sample_genes(cf_samples, output_dir)

    return result


def _dump_per_sample_genes(
    cf_samples: Sequence["CommonFailureSample"],
    output_dir: str,
) -> None:
    """把每个样本的 gene + planner-success 标签落盘。

    生成两个文件:
      - per_sample_genes.jsonl: 每行一个样本，完整结构（方便后续灵活解析）
      - per_sample_genes.csv: 宽格式，每行一个 (sample × planner) 组合
        sklearn / XGBoost 直接吃
    """
    import csv
    from navsim.agents.attack_genome.genes.genome_pipeline import (
        GENOME_FIELD_ORDER,
    )

    # JSONL：完整结构
    jsonl_path = os.path.join(output_dir, "per_sample_genes.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in cf_samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    # CSV：宽格式（长表：每个 planner 一行）
    # 收集所有 planner 名 + 所有 gene 字段
    all_planners: List[str] = sorted({
        p for s in cf_samples for p in s.repr_failures.keys()
    })
    all_gene_fields: List[str] = list(GENOME_FIELD_ORDER)
    all_attr_fields: List[str] = sorted({
        k for s in cf_samples for k in s.scene_attributes.keys()
    })

    csv_path = os.path.join(output_dir, "per_sample_genes.csv")
    header = (
        ["scene_token", "attack", "strength", "planner", "success"]
        + all_gene_fields
        + all_attr_fields
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for s in cf_samples:
            genome = s.genome.to_dict() if s.genome is not None else {}
            features = genome.get("features", {}) or {}
            row_base = [s.scene_token, s.attack, f"{s.strength:.4f}"]
            for planner in all_planners:
                row = list(row_base) + [planner, int(bool(s.repr_failures.get(planner, False)))]
                for gf in all_gene_fields:
                    v = features.get(gf, 0.0)
                    try:
                        row.append(f"{float(v):.6f}")
                    except (TypeError, ValueError):
                        row.append("0.0")
                for af in all_attr_fields:
                    v = s.scene_attributes.get(af, 0)
                    try:
                        row.append(f"{float(v):.6f}")
                    except (TypeError, ValueError):
                        row.append("0.0")
                writer.writerow(row)
    print(
        f"  [dump] per_sample_genes.jsonl: {len(cf_samples)} samples",
        flush=True,
    )
    print(
        f"  [dump] per_sample_genes.csv: {len(cf_samples) * len(all_planners)} rows "
        f"({len(cf_samples)} samples x {len(all_planners)} planners)",
        flush=True,
    )


# ----------------------------------------------------------------------
# 辅助：把 ASR 曲线聚合成迁移性矩阵
# ----------------------------------------------------------------------


def _build_transferability(
    asr_curves: Dict[str, Dict[str, Any]],
    family: RepresentationFamily,
) -> TransferabilityMatrix:
    """用所有攻击上的 ASR 序列拼接出每个表征的 ASR 序列。"""

    seqs: Dict[str, List[float]] = {p.representation.upper(): [] for p in family.planners}
    for repr_name, attacks in asr_curves.items():
        for attack, info in attacks.items():
            seqs[repr_name].extend(info["asrs"])
    # 转 numpy friendly
    seqs_np = {k: np.asarray(v, dtype=np.float64) for k, v in seqs.items()}
    return TransferabilityMatrix.from_asr_sequences(seqs_np)
