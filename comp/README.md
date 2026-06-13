# AGSA — Attack Genome Security Auditor

> **首个针对端到端自动驾驶模型的跨架构安全审计系统**
> 通过 37 维基因空间发现失效规律、跨架构评估风险、并在运行时检测进入 Failure Basin 的过程。

第十九届全国大学生信息安全竞赛（作品赛）· 自由赛道 · 2026.06

> **⭐ 2026-06-13 更新：5 层证据链完成**——新增 **Tier C2 执行层因果验证**（CNN 真 forward-pass 50% flip / DINO 0% flip 验证 OOD 设计意图）。详见 `TIER_C2_REPORT.md` 和 `figures/figure4_tierc2_full.png`。

---

## 1. 一句话定位

> **AGSA** is a cross-architecture security audit system for end-to-end autonomous driving. It discovers failure laws in a 37-dimensional gene space, evaluates cross-architecture risks, and detects in real time the process of a planner entering the *Failure Basin*.

## 2. 4 个可记忆数字

| Number | 含义 | 来源 |
|--------|------|------|
| **88,560** | real planner decisions | 3 planner × 492 scene × 10 attack × 6 strength |
| **89.2%** | monotonic failure trajectories | independent of XGBoost (L5 独立验证) |
| **0.876–0.939** | cross-architecture AUC | 6 Risk Auditor XGBoost models |
| **0.4 ms** | per-frame online detection | Genome Shield real-time monitor |

---

## 3. 三模块架构

```
AGSA
 ├── Module A · Genome Discovery   规律发现  (5 阶段证据链)
 ├── Module B · Risk Auditor         风险预测  (跨规划器 API)
 └── Module C · Genome Shield       实时防御  (在线监测)
```

| Module | 答什么问题 | 关键产物 |
|--------|-----------|---------|
| **A** | 为什么会失败？ | 5 阶段证据链 + per-sample SHAP top-K 因果杠杆 |
| **B** | 不升级会多大概率挂？ | 6 个跨 planner XGBoost + `audit_scene` API |
| **C** | 现在是不是在进入 basin？ | Basin Risk + 4 档状态实时监测 |

---

## 4. 5 阶段证据链（Module A 核心）

| Level | 答什么 | 关键数字 | 数据来源 |
|-------|--------|---------|---------|
| L1 Observation | 谁更脆弱？ | CNN 37.4% / DINO 9.4% / TF 5.0% | planner success/fail |
| L2 Pattern | 哪个 gene 决定？ | 3 planner AUC > 0.89 | gene→fail XGBoost |
| L3 Law | 跨架构守恒吗？ | 6/6 cross-planner AUC > 0.70 | src→tgt XGBoost |
| L4 Causality | 能否反推修复？ | K=10 flip 86%/71%/100% | SHAP top-K pushback |
| **L5 Failure Basin** | **Basin 几何存在吗？** | **89.2% monotonic, CNN 3× wide** | **planner 真实决策（独立）** |

**L5 完全脱离 XGBoost**，仅用 planner 真实 success 列独立验证 Failure Basin 几何，是整个项目最硬的证据。

---

## 5. 比赛包内容

| 文件 | 内容 | 大小 |
|------|------|------|
| `report.pdf` (22 页) / `report.tex` | 作品报告（依官方模板） | 625 KB |
| `agsa_defense.pptx` (31 页, 16:9) | 答辩 PPT，SOC 风格 | 565 KB |
| `figures/master_figure.png` | 首页总图 (AGSA SOC dashboard) | 168 KB |
| `figures/page2_genome_explorer.png` | Module A 可视化 | 104 KB |
| `figures/page3_risk_auditor.png` | Module B 可视化 | 101 KB |
| `figures/page4_failure_basin_map.png` | L5 核心证据图 | 85 KB |
| `figures/page5_executive_summary.png` | 4 大数字总结 | 108 KB |
| `figures/genome_shield_demo.gif` | Module C 30s 真实数据 demo | 506 KB |
| `README.md` | 本文件 | — |

---

## 6. 复现指南

### 6.1 环境

- **Python**: 3.10+
- **GPU**: 任意（XGBoost 训练可纯 CPU 跑）
- **磁盘**: ~10 GB (4 GB 模型 + 2 GB 数据 + 4 GB 脚本/图)

### 6.2 完整复现命令

```bash
# 0. 数据准备
# merged_3pl.csv (88,560 样本) 已存放在 ../exp/tierB_partial/

# 1. Module A · Genome Discovery
# 1.1 L1-L3 同 planner + 跨 planner 预测
python ../scripts/analysis/cross_planner_predict.py \
    --csv ../exp/tierB_partial/merged_3pl.csv \
    --output-dir ../exp/tierB_partial/cross_planner_3pl \
    --n-folds 5

# 1.2 L4 反事实 (3-planner)
for p in CNN DINO TF; do
    python ../scripts/analysis/failure_basin_counterfactual.py \
        --planner $p --n-fail 100 \
        --k-values 1 2 3 4 5 6 7 8 9 10 12 14 16 20 \
        --output-dir ../exp/tierB_partial/failure_basin_${p,,}
done

# 1.3 ★ L5 独立验证
python ../scripts/analysis/failure_basin_analysis.py
# 产出: 3 张图 + summary.json + summary.txt

# 2. Module B · Risk Auditor
python ../scripts/analysis/risk_auditor.py --mode train
python ../scripts/analysis/risk_auditor.py --mode audit \
    --scene 016d6a913efa5ff1 --attack DigitalNoise --planner CNN

# 3. Module C · Genome Shield
python ../scripts/analysis/genome_shield.py
# 产出: demo.gif (1300×850, 30s) + latency.txt

# 4. PPT 生成
python ../scripts/analysis/agsa_pptx.py
```

### 6.3 复现时间

| 步骤 | 预计时间 |
|------|---------|
| L1-L3 训练 | ~5 min (CPU) |
| L4 反事实 | ~9 min (3 planner, CPU) |
| L5 独立验证 | < 1 min (纯 pandas) |
| Risk Auditor 训练 | ~15 s (CPU) |
| Genome Shield 训练 + demo | ~30 s (CPU) |
| **合计** | **~15 min on CPU-only laptop** |

---

## 7. 数据与模型权重

| 项 | 来源 / 路径 |
|----|------------|
| NAVSIM OpenScene (test + trainval) | nuScenes 衍生公开数据集 |
| 10 种攻击模板 | `navsim/agents/attack_genome/attacks/templates.py` |
| 37 维 gene 提取器 | `navsim/agents/attack_genome/genes/` |
| 3 planner adapters | `scripts/attack_genome/adapters.py` |
| 4.2 GB 模型权重 | `models/navsim_ckpts/` (CNN 945 MB + VoVNet 323 MB + DINO 167 MB + TF 672 MB) |

---

## 8. 关键发现 (Strongest Claim)

> **Cross-Representation Failure Law is a causally manipulable phenomenon**:
> attack success is governed by a representation-invariant gene signature
> (dominated by edge density) whose manipulation **flips the planner verdict in 71-100%**
> of cases across 3 architectures,
> with the **same gene (edge density) being the top1 per-sample driver in 80-88% of failures**.

| 证据 | 数字 | 性质 |
|------|------|------|
| 跨 planner 形状守恒 | 6/6 AUC > 0.70 | 预测层 |
| 跨 planner 因果可操控 | K=10 flip 86%/71%/100% | 代理层 (SHAP) |
| Failure Basin 独立验证 | 89.2% monotonic, CNN 3× wide | **执行层（planner 真实决策）** |

---

## 9. 创新性

1. **基因空间视角**：把对抗攻击从"事件名"翻译为 37 维数值向量
2. **跨表征守恒**：6/6 跨 planner AUC > 0.70（失败的形状守恒）
3. **因果可操控**：SHAP top-K 反向 pushback 86% flip
4. **Failure Basin 几何**：89.2% monotonic + CNN 3× 宽 basin（L5 独立验证）

---

## 10. 实用价值（3 类用户）

- **自动驾驶厂商**：升级决策支持（CNN → TF 减 52.4% 风险）
- **安全评测机构**：失效归因报告（Top 5 驱动基因）
- **车端运维**：实时监测（0.4 ms/帧）+ 法规侧量化审计

---

## 11. 局限与未来

- 单数据集（NAVSIM）→ 需跨数据集验证（nuScenes 完整 / Waymo）
- L4 Causality 在 XGBoost 预测层 → L5 已用真实 planner 决策独立验证
- 37 维 gene → 可扩展（IR / Saliency / 语义嵌入）
- ONNX 导出 planner → 端到端 forward-pass 验证
- 自适应 attack recipe（联合扰动 SHAP top-K）

---

## 12. 答辩流程（90s 速记）

| 时间 | 步骤 |
|------|------|
| 0-30s | 播放 SOC dashboard 视频 (PPT Page 1)，风险从 12% → 82%，进入 Failure Basin |
| 30-60s | 打开 Genome Explorer (PPT Page 2)，展示 edge\_mean + lane\_density 是主导驱动 |
| 60-90s | 打开 Risk Auditor (PPT Page 3)，展示 CNN 82% → DINO 47% → TF 39%，给出升级建议 |

---

## 13. 引用

```bibtex
@misc{agsa2026,
  title  = {AGSA: Attack Genome Security Auditor for End-to-End Autonomous Driving},
  year   = {2026},
  note   = {National Information Security Competition (Undergraduate Track), 2026.06}
}
```

---

**License**: Internal research / competition use.
**Generated**: 2026.06.11 · 22-page report + 31-page PPT + 5 SOC figures + 30s real-data demo.
