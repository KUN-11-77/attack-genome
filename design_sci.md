# AutoDrive-SCI

## Discovering Safety-Critical Information in Autonomous Driving Systems

### 自动驾驶安全关键信息发现与风险审计系统

---

# 0. 与现有资产对接

> 本文是 **SCI 叙事版本**的研究设计文档。
> 原 Attack Genome 版见 [`design.md`](design.md)，
> 工程手册见 [`doc/attack_genome/README.md`](doc/attack_genome/README.md) 与
> [`doc/attack_genome/WSL_RUNBOOK.md`](doc/attack_genome/WSL_RUNBOOK.md)。

## 0.1 叙事重心迁移

| 旧叙事 (Attack Genome) | 新叙事 (SCI) |
|---|---|
| Attack Genome 是创新 | SCI Discovery 是创新 |
| 分析攻击为何迁移 | 发现共享安全关键信息 |
| Gene 是核心对象 | SCI Category 是核心对象 |
| 37 维 gene vector 是输出 | SCI Health Score 是输出 |

Attack Genome 降级为 **Layer 1 方法论**——它仍然是统一的攻击语义表示工具，但不再是故事主角。

## 0.2 已有可复用资产

同 [`design.md` §0.1](design.md)——所有代码、模型、数据已盘点。

## 0.3 SCI 框架与现有实现的映射

| SCI Layer | 内容 | 对应实现 | 状态 |
|---|---|---|---|
| Layer 1: Attack Genome | 统一攻击语义表示 | `navsim/agents/attack_genome/genes/` | ✅ |
| Layer 2: SCI Discovery | 跨架构共享失效驱动因子 | SHAP + per-sample top1 + cross-planner intersection | ✅ |
| Layer 3: SCI Law | 跨架构预测验证 | Cross-planner AUC > 0.70 (6/6 pairs) | ✅ |
| Layer 4: SCI Causality | Planner forward-pass 因果验证 | **Tier C2 — 待实现** | ❌ |
| Layer 5: SCI Monitor | 实时 SCI 健康度监测 | Genome Shield 升级 | ⚠️ 基础已有 |

---

# Abstract

随着自动驾驶系统从监督 CNN 发展到自监督 ViT 再到视觉-语言模型，一个根本安全问题长期被忽视：

**不同视觉架构的自动驾驶系统，是否共享同一类安全关键信息？**

现有研究聚焦于单模型鲁棒性评测或攻击设计，却缺乏对"跨架构共享失效机制"的系统研究。

本项目提出 AutoDrive-SCI。我们利用 NavDream 可控外观操控空间，在 CNN（GTRS-Dense VoV）、自监督 DINO（GTRS-Dense-DINO）和混合架构 TransFuser 三类视觉规划器上，系统分析攻击下的失效模式。

**核心发现**：三类架构虽然内部表征完全不同，但共享同一类 **Safety-Critical Information (SCI)**——以道路结构信息（Road Structure Information）为核心。当 SCI 被破坏时，所有 Planner 进入同一个 Failure Basin。

项目最终输出：
- **SCI Categories**（安全关键信息分类）
- **SCI Law**（跨架构守恒规律）
- **SCI Causality**（因果验证）
- **SCI Monitor**（实时风险审计系统）

为自动驾驶安全研究提供从"模型鲁棒性"到"信息鲁棒性"的新范式。

---

# 1. 问题定义

## 1.1 研究背景

自动驾驶视觉规划器的 backbone 正在经历快速演化：

```
CNN (ImageNet 监督)
  ↓
Self-Supervised ViT (DINO)
  ↓
Vision-Language Model
```

然而一个关键问题尚未被系统研究：**当视觉表征演化时，攻击迁移的根源是什么？**

传统解释：参数相似、数据集偏差、特征迁移。但这些解释不能说明为什么结构不同的架构会出现相似的失效模式。

## 1.2 核心科学问题

### Q1: 是否存在跨架构共享的安全关键信息 (SCI)？

不同视觉架构虽然学习方式不同，但都服务于同一个任务——从 2D 图像恢复 3D 道路几何。是否存在某些信息是所有架构都必须依赖的？

### Q2: 攻击迁移的根源是否是 SCI 共享？

攻击之所以能跨架构迁移，是否因为它们在破坏同一种 SCI？

### Q3: SCI 下降是否因果性地导致 Planner 失效？

相关性 ≠ 因果性。SCI 下降是否直接导致 Planner 行为退化？

---

# 2. Safety-Critical Information (SCI) 假说

## 2.1 核心假说

> **Safety-Critical Information Hypothesis**: 对于视觉自动驾驶系统，存在一组跨架构共享的安全关键信息 (SCI)。当 SCI 被破坏到阈值以下时，无论 Planner 架构如何，系统都会进入 Failure Basin。

## 2.2 形式化定义

- $P = \{p_1, p_2, ..., p_n\}$：Planner 集合（CNN, DINO, TransFuser, ...）
- $G: (scene, attack) \to \mathbb{R}^{37}$：Attack Genome 向量
- $\text{fail}(p, scene, attack) \in \{0, 1\}$：Planner $p$ 在给定条件下是否失效

**SCI Candidate**：满足以下条件的 gene 子集 $S \subset \{1..37\}$：

1. **高失效贡献度**：$S$ 中的 gene 在 per-sample SHAP 分析中是 top driver
2. **跨架构稳定**：$S$ 在多个 planner 的 top-K driver 中重叠率 > 70%
3. **跨架构可预测**：用 $P \setminus \{p_i\}$ 训练的 gene→fail 模型能在 $p_i$ 上 AUC > 0.70

## 2.3 当前 SCI Candidates（基于 88,560 样本）

| SCI Category | 对应 Gene | CNN rank | DINO rank | TF rank | 跨架构一致性 |
|---|---|---|---|---|---|
| **Road Structure** | edge_mean, edge_density, lane_line_count, lane_line_density | top1 (88%) | top1 (82%) | top1 (80%) | ★★★ |
| **Lane Geometry** | lane_line_count, lane_line_density | top5 | top5 | top3 | ★★★ |
| **Texture Stability** | lbp_entropy, glcm_contrast | top3 | top2 | top2 | ★★☆ |

**关键证据**：edge_mean 在 80-88% 的 failure 样本中是 per-sample top1 driver，跨 CNN/DINO/TF 一致。这不是统计巧合。

---

# 3. 威胁模型

## 攻击目标
诱导规划器产生危险轨迹偏移：偏离车道、错误转向、碰撞风险提升。

## 攻击能力
攻击者通过 LED 广告牌、投影系统、数字显示屏等对车辆视觉输入产生外观操控。

## 攻击约束
不能修改道路拓扑、篡改车辆内部系统、修改地图。只能影响视觉外观。

## 研究范围
Adversarial Appearance Manipulation（对抗性外观操控）。NavDream 生成的风格变化在此框架下被视为攻击模板。

---

# 4. SCI 系统架构

```
┌─────────────────────────────────┐
│ Layer 1: Attack Genome          │  统一攻击语义表示
│ 37-D gene vector extraction     │  ✅ 已完成
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Layer 2: SCI Discovery          │  跨架构共享失效因子挖掘
│ SHAP + Cross-Planner Intersect  │  ✅ 已完成
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Layer 3: SCI Law                │  跨架构预测验证
│ Cross-Planner AUC > 0.70        │  ✅ 已完成
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Layer 4: SCI Causality          │  Planner 因果验证
│ Forward-Pass Intervention       │  ❌ Tier C2
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│ Layer 5: SCI Monitor            │  实时风险审计
│ SCI Health + Failure Basin Risk │  ⚠️ 升级中
└─────────────────────────────────┘
```

---

# 5. Layer 1 — Attack Genome

## 统一攻击语义表示

将像素级攻击（Rain, Snow, Blur, Dust, Dusk, Dawn, Noise, ...）映射到统一的 37 维语义空间。这是 SCI 发现的方法论基础。

### 5.1 Gene 分类

| 类 | Gene 字段 | 物理含义 |
|---|---|---|
| **结构 (6)** | edge_mean, edge_density, road_luma_mean, road_luma_std, lane_line_count, lane_line_density | 道路几何完整性 |
| **频率 (4)** | low_freq_ratio, mid_freq_ratio, high_freq_ratio, spectral_centroid | 频域能量分布 |
| **颜色 (7)** | hue_mean, hue_std, sat_mean, sat_std, val_mean, val_std, colorfulness | 色彩信息 |
| **纹理 (3)** | lbp_entropy, glcm_contrast, lbp_uniformity | 纹理复杂度 |
| **亮度 (9)** | rms_contrast, mean_luma, std_luma, dynamic_range, mean_shift, std_shift, luma_entropy, luma_skew, luma_mean | 亮度信息 |
| **光照 (4)** | shadow_ratio, highlight_ratio | 局部光照 |
| **检测 (5)** | vehicle_loss, person_loss, detection_loss, conf_loss, vehicle_loss_ratio | 感知退化 |
| **元信息 (1)** | strength | 攻击强度 |

### 5.2 基因提取正确性

7 类基因全部用纯 numpy + opencv 实现。46 项测试覆盖 33 项数值正确性验证（含 FP32 确定性约束）。详见 [`design.md` §3.4](design.md)。

---

# 6. Layer 2 — SCI Discovery

## 发现安全关键信息

### 6.1 实验设置

| 参数 | 值 |
|---|---|
| Planner | CNN-GTRS (VoV), DINO-GTRS (ViT), TransFuser (ResNet+BEV) |
| 场景数 | 492 (NavDream OOD benchmark) |
| 攻击 | 10 types × 6 strengths = 60 configs |
| 总样本 | 88,560 (29,520/planner) |
| 分析工具 | XGBoost (5-fold GroupKFold by scene) + per-sample SHAP |

### 6.2 SCI Candidate 挖掘流程

```
Failure Samples (per planner)
  ↓
XGBoost: Gene → Fail (5-fold CV)
  ↓
Per-sample SHAP → Top-K driving genes
  ↓
Cross-planner intersection → SCI Candidates
```

### 6.3 关键发现：edge_mean 跨架构集中度

```
CNN  top1 driver: edge_mean  88% of fail samples
DINO top1 driver: edge_mean  82% of fail samples
TF   top1 driver: edge_mean  80% of fail samples
```

**反常信号**：正常情况下，三种完全不同的架构应该有不同的 top driver。80-88% 的集中度暗示这不是架构特定的现象，而是**任务特定的约束**——所有 Planner 都必须依赖道路结构信息。

---

# 7. Layer 3 — SCI Law

## 跨架构 SCI 守恒规律

### 7.1 跨架构共有/差异信息

**跨架构共有** (min importance > 0.03 across 3 planners):

| Gene | CNN | DINO | TF | SCI 分类 |
|---|---|---|---|---|
| edge_mean | 0.045 | 0.043 | 0.033 | Road Structure |
| lbp_entropy | 0.033 | 0.048 | 0.041 | Texture Stability |
| high_freq_ratio | 0.039 | 0.045 | 0.048 | Frequency |
| spectral_centroid | 0.034 | 0.042 | 0.038 | Frequency |

**架构差异** (max gap > 0.01):

| Gene | CNN | DINO | TF | 解释 |
|---|---|---|---|---|
| strength | 0.063 | 0.012 | 0.013 | CNN 对攻击强度敏感 |
| mean_shift | 0.064 | 0.027 | 0.031 | CNN 依赖亮度变化 |
| luma_entropy | 0.023 | 0.041 | 0.024 | DINO 单独依赖 |

**核心含义**：表征升级（CNN→DINO/TF）显著衰减对纹理/颜色类攻击的敏感度，但**无法消除对结构信息的依赖**——这就是表征演化的边界。

### 7.2 跨架构预测验证

**同 Planner 5-fold AUC（baseline）**：

| Planner | n | Fail % | CV AUC |
|---|---|---|---|
| CNN | 29,520 | 37.4% | 0.898 |
| DINO | 29,520 | 9.4% | 0.894 |
| TF | 29,520 | 5.0% | 0.943 |

**跨 Planner 迁移（gene→fail，6/6 > 0.70）**：

| src → tgt | AUC | Δ vs baseline |
|---|---|---|
| **CNN → DINO** | **0.798** | -0.085 |
| CNN → TF | 0.756 | -0.127 |
| DINO → CNN | 0.772 | -0.103 |
| DINO → TF | 0.704 | -0.171 |
| TF → CNN | 0.703 | -0.235 |
| TF → DINO | 0.707 | -0.231 |

**规律级含义**：
- 失败的"形状"（gene profile）在表征间守恒——CNN 训练的模型能在 TF（失败率仅 5%）上达到 AUC=0.756
- 失败幅度被表征压缩，失败形状不被压缩
- **这证明 SCI 是跨架构共享的，不是某个架构的特异现象**

---

# 8. Layer 4 — SCI Causality（核心未完成项）

## 从相关性到因果性

### 8.1 当前状态：XGBoost 层因果（Tier C1 ✅）

**反事实实验**（per-sample SHAP → gene replacement → XGBoost re-prediction）：

| Planner | K=10 flip rate | Random-K flip | Lift |
|---|---|---|---|
| CNN | 86% | 7% | +0.79 |
| DINO | 71% | 2% | +0.69 |
| TF | 100% | 19% | +0.81 |

**证据等级**：基因排序是真实因果信号（非噪声）。但仅限于 XGBoost 预测层。

### 8.2 缺失：Planner 层因果（Tier C2 ❌）

当前证据链：
```
图像 → Gene → XGBoost → Fail prediction  ✅ (Tier C1)
```

需要的证据链：
```
图像 → Gene → SCI Intervention → Planner forward pass → Trajectory recovery  ❌ (Tier C2)
```

### 8.3 Tier C2 实验设计

**核心命题**：SCI 下降因果性地导致 Planner 失效。

**实验方案**（最小可行版，20 样本）：

```
1. 选 20 个 CNN/DINO/TF 共同失效样本
2. 对每个样本，识别 per-sample top SCI driver
3. 在 NavDream 风格空间中搜索"SCI 恢复、其他 gene 近似"的邻居图
4. 对原图和邻居图分别跑 Planner forward pass
5. 比较 trajectory: ADE 是否显著下降
```

**成功标准**：
- 20 样本中 ≥ 15 个 trajectory 显著恢复 → SCI 因果成立
- 5-14 个 → 部分因果，需更大样本
- < 5 个 → SCI 可能仅是统计相关，需重新审视

**实验产物**：
- `exp/tierC2/manifest.csv`：20 样本 × 2 条件 × 3 planner = 120 forward passes
- `exp/tierC2/trajectory_comparison.png`：原图 vs SCI 恢复图的轨迹对比
- `exp/tierC2/causality_report.md`：因果验证结论

---

# 9. Layer 5 — SCI Monitor

## 实时安全关键信息审计

### 9.1 系统设计

```
实时驾驶视频
  ↓
Attack Genome 提取（每帧）
  ↓
SCI Health Score 计算
  ↓
Failure Basin Risk 预测
  ↓
Top Vulnerable SCI 定位
  ↓
Alert → Safety Fallback
```

### 9.2 输出指标

| 指标 | 含义 | 阈值 |
|---|---|---|
| **SCI Health Score** | 道路结构 / 车道几何 / 纹理稳定性的加权健康度 | < 0.3 → Critical |
| **Failure Basin Risk** | 当前 SCI 状态下 Planner 失效概率 | > 0.7 → Alert |
| **Top SCI Driver** | 当前最脆弱的 SCI 维度 | 用于解释和恢复 |

### 9.3 Demo 设计

**Page 1 — Live Monitor**
```
┌──────────────────────────────────────┐
│ Live Driving Scene                   │
│                                      │
│ SCI Health                           │
│ Road Structure   ████░░░░░░ 12%  ⚠  │
│ Lane Geometry    ██████░░░░ 41%     │
│ Texture          ████████░░ 73%     │
│                                      │
│ Failure Basin Risk  ██████████ 94%   │
│ Top Driver: Road Boundary Continuity │
└──────────────────────────────────────┘
```

**Page 2 — Cross-Planner Validation**
```
Same scene. Different architectures.
Same SCI degradation.

CNN    DINO   TF
 ↓       ↓      ↓
All enter Failure Basin when
Road Structure < threshold
```

**Page 3 — Conclusion**
> Different architectures.
> Same Safety-Critical Information.
> Attacks transfer because SCI transfers.

---

# 10. 实验执行计划

| Tier | 内容 | 状态 | 环境 |
|---|---|---|---|
| A | pytest smoke + 基因正确性 | ✅ 46 passed | 本地 5060 |
| B | 492 scene × 3 planner gene extraction + 同/跨 planner 预测 + 反事实 | ✅ 88,560 样本 | 服务器 6×3080Ti |
| **C2** | **SCI Causality: Planner forward-pass 验证** | **❌ 核心缺口** | 服务器 |
| C3 | Cross-dataset (nuScenes 50 scene smoke test) | ⬜ 论文补充 | 服务器 |
| D | Full 500 scene sweep + VLM planner | ⬜ 论文补充 | 服务器 |

---

# 11. 核心贡献

**Contribution 1 — SCI Discovery**: 首次发现自动驾驶视觉规划器存在跨架构共享的安全关键信息 (SCI)，以道路结构信息为核心。

**Contribution 2 — SCI Law**: 建立跨架构 SCI 守恒规律——失败形状在表征间守恒（跨 planner AUC > 0.70），失败幅度被表征压缩（9.4% vs 37.4%），结构信息依赖不随表征演化衰减。

**Contribution 3 — SCI Causality**: （待 Tier C2）证明 SCI 下降因果性地导致 Planner 失效，非仅统计相关。

**Contribution 4 — SCI Monitor**: 构建实时 SCI 健康度审计系统，实现攻击检测、风险预警和脆弱 SCI 定位。

---

# 12. 安全意义

**对攻击者**：发现攻击迁移的根源不是模型相似，而是 SCI 共享。攻击者可以通过破坏 SCI 实现跨架构攻击——不需要针对每个 Planner 设计独立攻击。

**对防御者**：安全分析应从"模型鲁棒性"转向"信息鲁棒性"。防御不应只换 backbone，应直接保护 SCI 维度（增强道路结构、车道几何的视觉完整性）。

**对评测者**：仅报告 ASR 不够——应同时报告 SCI-level 的攻击贡献度和跨架构 SCI 守恒度。

---

# 13. 答辩一句话

> 我们发现自动驾驶系统虽然采用完全不同的视觉架构，但共享同一类 Safety-Critical Information。攻击的本质不是欺骗某个模型，而是破坏这些被所有模型共同依赖的安全关键信息。因此，安全分析应从"模型鲁棒性"转向"信息鲁棒性"。

---

# 14. 服务器协作

同 [`design.md` §13](design.md)。路径约定不变。

**Tier C2 服务器运行**：
```bash
ssh khsong@10.13.74.231 -p 66
cd /data3/khsong/cogatedrive
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate attack_genome
bash scripts/attack_genome/run_tierC2.sh
```

---

# Appendix A — 论文骨架（SCI 版）

| Section | 内容 | 状态 |
|---|---|---|
| 1. Introduction | SCI 假说 + 表征演化边界问题 | 撰写中 |
| 2. Related Work | OOD robustness / Attack transfer / Failure diagnosis | 撰写中 |
| 3. SCI Hypothesis | 形式化定义 + 可证伪条件 | 撰写中 |
| 4. Attack Genome (Method) | 37-D gene space 作为 SCI 测量工具 | ✅ |
| 5.1 SCI Discovery | 跨架构共享 driver 挖掘 | ✅ |
| 5.2 SCI Law | 跨 planner AUC > 0.70, 6/6 | ✅ |
| 5.3 SCI Causality | 反事实 (XGBoost) + Forward-pass (Tier C2) | ⚠️ C1✅ C2❌ |
| 6. SCI Monitor | 实时审计系统 | ⚠️ 升级中 |
| 7. Discussion | 安全 implications (攻击/防御/评测) | ✅ |
| 8. Limitation | 单 dataset / XGBoost 预测层 / 37 维不完备 | ✅ |
| 9. Conclusion | SCI 是跨架构共享的，攻击迁移根源在此 | ✅ |

---

# Appendix B — 与 Attack Genome 原版的差异

| 维度 | Attack Genome 原版 | SCI 版 |
|---|---|---|
| 核心创新 | 攻击基因分析框架 | 安全关键信息发现 |
| 主角 | Attack Genome | SCI |
| 方法地位 | 创新本身 | Layer 1 工具 |
| 最强 claim | Cross-Representation Failure Law | SCI Hypothesis + evidence |
| 边缘的定位 | edge 是重要 gene | edge 是 Road Structure SCI 的测量代理 |
| 安全叙事 | 攻击分析 | 信息鲁棒性范式 |
| 国一记忆点 | 跨架构攻击迁移分析 | 不同架构共享同一类安全关键信息 |

---

**最后更新**：2026-06-12 — SCI 叙事重构完成，待 Tier C2 实验。
**下一里程碑**：Tier C2 — SCI Causality Planner forward-pass 验证（20 样本最小可行版）。
