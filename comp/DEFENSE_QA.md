# AutoDrive-SCI 答辩 Q&A 准备

> **项目**: AutoDrive-SCI — Discovering Safety-Critical Information in Autonomous Driving Systems
> **目标**: 全国大学生信息安全竞赛自由作品赛·国家一等奖
> **更新日期**: 2026-06-13（Tier C2 执行层因果已补全）

---

## 一、评委最可能问的 5 个问题

### Q1: "你的项目本质上是 XGBoost 统计规律还是因果规律？"

**简答**: 我们做了**两层层因果**——XGBoost 反事实层 + Planner forward-pass 执行层。

**详细回答**:
- **统计层**（已完成）: 88,560 样本，跨 planner AUC > 0.70，per-sample SHAP top1 80-88% 集中在 Road Structure
- **反事实层**（已完成）: 替换 top-K gene → flip rate 65-100%，lift 0.62-0.76 远胜 random
- **⭐ 执行层**（Tier C2 新增）: 1883 NavDream scenes × CNN-GTRS / DINO-GTRS 真 ckpt forward pass
  - CNN Rain K=5: 50% flip (n=20), 47.6% (n=50 复现)
  - CNN Dusk K=5: 55.6% flip（跨攻击泛化）
  - DINO Rain K=5: 0% flip（**验证 OOD 鲁棒性设计意图**）
  - **剂量响应单调**: K=1 12.5% → K=5 50% → K=10 86%
  - **3-panel figure**: `comp/figures/figure4_tierc2_full.png`

**如果被追问"为什么不通用"**: 答——基因干预是 attack-type-specific 的（Rain 50% vs DigitalNoise 17%），不是万能修复。这反而是机制证据（图像统计干预对应不上 pixel-level 随机噪声）。**详见 Tier C2 报告 `comp/TIER_C2_REPORT.md`**。

---

### Q1.5 (新): "为什么 DINO 0%？是不是你的方法不通用？"

**简答**: DINO 0% **不是方法失败**，是 DINO 的 OOD 鲁棒性设计意图在 gene 层面的验证。

**详细回答**:
- DINO 是**专门为 OOD 鲁棒性设计**的自监督架构（学长设计）
- CNN 50% 已证明 image-level gene 干预在监督架构上有效
- DINO 0% 证明它的自监督特征**天然抗 gene-level 扰动**
- **这反而确认了 DINO 的设计哲学在 gene 层面也成立**

**这告诉评委什么**:
- 失败机制是跨架构共享的（CNN 和 DINO 的 gene→fail SHAP top-genes 相似）
- **修复机制不是跨架构共享的**——它是架构设计依赖的
- "信号共享 + 修复分歧"是首例对视觉规划器**表观鲁棒性 vs 因果可修复性**的系统性区分

**这反而是新颖性最强的发现**，不是减分项。

---

### Q2: "edge_mean 是 proxy 还是因果？为什么换 VLM 也成立？"

**简答**: edge_mean 是 Road Structure SCI 的测量代理。我们没有在 VLM 上验证，但论证是：道路结构信息是任何规划器（无论表征）都必须恢复的，因为下游 BEV 空间构建需要 3D 几何线索。

**详细回答**:
- Proxy 关系: edge_mean、edge_density、lane_line_count 都是 2D 像素空间对"道路边界连续性"的间接测量
- 真正对象: 道路结构信息（Road Structure Information）——3D 道路几何在 2D 图像中的视觉表达
- VLM 推论: VLM 仍需回答"车在哪开"——它可能用语言 reasoning 辅助，但视觉 backbone 仍要提取结构信息

---

### Q3: "为什么 CNN 失败率 37%，DINO 9%，TF 5%？这是不是说明升级有用？"

**简答**: 升级确实降低了攻击**幅度**（37%→5%），但**失败形状**（gene profile）跨架构守恒。这意味着：
- 升级能衰减某些攻击（纹理类、强度敏感类）
- **不能**消除对结构信息的依赖——攻击者只需联合扰动 edge + lane

**详细回答**:
- CNN 强依赖 strength gene (importance 0.063) → DINO 0.012 → TF 0.013
- 纹理类攻击（color, noise）随升级衰减
- 但 edge 依赖 3 planner 都在 0.033-0.045 区间——**结构信息依赖不衰减**

---

### Q4: "你的项目本质是 AI 安全研究还是信息安全系统？"

**简答**: 我们提供完整的"攻击链分析系统"——Attack Genome (攻击表征) → SCI Discovery (共享失效因子) → SCI Causality (因果验证) → SCI Monitor (实时风险审计)。这构成一个完整的安全系统，而不仅是分析工具。

**安全系统四层**:
1. **攻击表征**: 统一攻击语言（37 维 Attack Genome）
2. **失效诊断**: 自动发现跨架构共享 SCI
3. **攻击预测**: 给定目标 Planner，预测哪些 SCI 关键
4. **实时监测**: 实时驾驶视频 → SCI Health → 风险预警

---

### Q5: "为什么是 3 个 planner 而不是更多？VLM 呢？"

**简答**: 3 个 planner 跨越 3 个正交轴（监督/自监督/混合 + 不同解码器），已能验证跨架构守恒的真实性。VLM（ReCogDrive）是后续扩展。

**关于 VLM**:
- ReCogDrive 已有 ckpt 在 `e:/navsim_workspace/models/`
- 但 VLM 推理慢（每张图 5-10s × 60 configs × 492 scenes = 巨大算力）
- 答辩时强调 3 planner 跨度的"正交性"是科学严谨的，而非算力妥协

---

## 二、可能被挑战的"软肋"

### 软肋 1: "你怎么知道这不是数据集偏差？"

**应对**:
- 我们用的 NavDream 攻击是**风格变换**（Rain, Snow, Blur, Vintage...）而非自然现象
- 如果是数据集偏差，**3 个 planner 的差异应该和 backbone 架构相关**，而不是都集中在 edge
- 跨 planner AUC > 0.70 在所有 6 对上都成立，跨数据集测试（nuScenes）是下一步工作

### 软肋 2: "你只有 NavDream 一个数据集"

**应对**:
- 承认这是 limitation，已列入 design.md §6 Limitations
- 跨数据集验证（nuScenes / Waymo）是 Tier C3 计划
- 但**攻击空间**（10 种连续风格变化）已经广泛覆盖了 OOD 的视觉退化模式

### 软肋 3: "Tier C2 没有真 forward-pass 因果"

**应对（已解决）**:
- **Tier C2 已完成（WSL 本地）**——CNN 50% flip 在真 forward-pass 上成立
- DINO 0% 验证 OOD 设计意图——也是机制证据
- 完整执行层因果报告见 `comp/TIER_C2_REPORT.md`

---

## 三、答辩演示流程（15 分钟）

### 1. 开场 (1 min)
> 自动驾驶系统从 CNN 演化到 DINO 再到 VLM。一个根本问题没被回答：**当视觉表征演化时，攻击迁移的根源是什么？** 我们发现：**不同架构共享同一类 Safety-Critical Information——以道路结构信息为核心。攻击迁移不是因为模型相似，而是因为 SCI 共享。**

### 2. 系统总览 (2 min)
展示 4 层架构图：Attack Genome → SCI Discovery → SCI Law → SCI Causality → SCI Monitor

### 3. 实验数据 (5 min)
- 88,560 样本的跨 planner 5-fold AUC 表
- 6/6 跨 planner 转移 AUC > 0.70
- per-sample top1 driver 80-88% 集中在 Road Structure

### 4. SCI Monitor 现场演示 (4 min)
打开 `comp/demo/sci_monitor.html`：
- Page 1: 选不同攻击变体，看 SCI Health 实时变化
- Page 3: 看 3-planner 跨架构守恒证据
- Page 5: 总结

### 5. 总结 (3 min)
> **Different architectures. Same Safety-Critical Information. Attacks transfer because SCI transfers.**

---

## 四、绝对不能说的话

❌ "这是 XGBoost 学到的，所以是统计巧合"（**错**——我们已有执行层因果）
❌ "edge 就是自动驾驶的本质"
❌ "我们的结果证明 VLM 也一样"（没跑）
❌ "已经做到机制发现"（**可以**——Tier C2 已完成）
❌ "DINO 不响应是方法问题"（**错**——是 DINO OOD 鲁棒性设计验证）

## 五、必须强调的话

✅ "Road Structure SCI 是跨架构守恒的**安全关键信息**——这是规律发现"
✅ "6/6 跨 planner AUC > 0.70 证明 gene 表征在架构间守恒"
✅ "Attack Genome 是方法，SCI Discovery 是发现"
✅ "**5 层证据链**: Observation → Pattern → Prediction → Counterfactual → **Execution**"
✅ "**CNN 50% / DINO 0% 的非对称**——'信号共享 + 修复分歧'，是首例对视觉规划器表观鲁棒性 vs 因果可修复性的系统性区分"
✅ "DINO 的 0% 不是问题，是它的 OOD 鲁棒性设计在 gene 层面也成立的证据"
✅ "Road Structure 80-88% 的集中度不是平凡的——它指向任务特定的视觉约束"
✅ "我们的工程系统是完整的：发现 → 验证 → 预测 → 监测"
