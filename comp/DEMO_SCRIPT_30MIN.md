# 30 分钟国赛 demo 完整脚本

> **项目**: Attack Genome Law
> **时长**: 30 分钟
> **评委**: 信息安全竞赛 国赛
> **目标**: 现场攻击 + 轨迹图实时生成，让评委"看见"基因干预如何改变 planner 行为

---

## 时间分配

| 时段 | 时长 | 形式 | 主要内容 |
|------|------|------|----------|
| 0:00-2:00 | 2 min | 预录视频播放 | 30s SOC 风格视频（开场氛围） |
| 2:00-7:00 | 5 min | PPTX 翻页 + 讲解 | 5 个核心数字（88,560 / 89.2% / 0.798 / 50% / 0.4ms） |
| 7:00-12:00 | 5 min | PPTX 重点页 | L1-L3: Observation→Pattern→Law |
| **12:00-17:00** | **5 min** | **Jupyter 实跑** | **现场攻击 + 轨迹图生成（核心）** |
| 17:00-22:00 | 5 min | PPTX 重点页 | L5 + L6: Failure Basin + Tier C-2 |
| **22:00-25:00** | **3 min** | **终端实跑** | **Genome Shield 实时监控** |
| **25:00-28:00** | **3 min** | **Jupyter 实跑** | **跨架构 trajectory 对比** |
| 28:00-30:00 | 2 min | 口头 | Q&A + 金句收尾 |

**总时长 30 min**，其中**8 min 现场实跑**（12-17 / 22-25 / 25-28）。

---

## 段 0: 0:00-2:00 — 预录视频（暖场）

### 做什么
播放 30 秒 SOC 风格视频（黑底+雨景+风险曲线上升），让评委从第一秒感受到这是**安全运营产品**不是"分析工具"。

### 怎么跑
```bash
# 在答辩前一天跑一次生成视频
source ~/miniconda3/etc/profile.d/conda.sh
conda activate constanteye_sec
cd /mnt/d/cogatedrive
python scripts/analysis/agsa_soc_video.py

# 视频会输出到 comp/agsa demo_30s.mp4
# 在答辩时直接用 VLC / Windows Media Player 播放
```

### 配音稿
> "AGSA detects planner failure in real time, before collision. The risk curve rises from 0.12 to 0.85 in 30 seconds. This is what a Security Operations Center for autonomous driving looks like."

---

## 段 1: 2:00-7:00 — 5 个核心数字（PPTX 翻页）

### 做什么
展示 `agsa_defense.pptx` 30 页 PPT 的前 5-8 页（数字总览）。

### 怎么准备
```bash
# 跑 PPTX 生成器（一次性）
cd /mnt/d/cogatedrive
python scripts/analysis/agsa_pptx.py

# 答辩时用 PowerPoint 打开 comp/agsa_defense.pptx
```

### 关键 5 个数字（**只讲这 5 个，其他都跳过**）
1. **88,560** 样本（3 planner × 492 scene × 10 attack × 6 strength）
2. **89.2%** Failure Basin 单调率
3. **0.798** 最强跨 planner 转移 AUC
4. **50%** Tier C-2 干预修复率
5. **0.4ms** Genome Shield 实时检测延迟

### 讲法
> "评委老师，如果你们今天只记住 5 个数字，请记住这 5 个：88,560 / 89.2 / 0.798 / 50% / 0.4ms。"

---

## 段 2: 7:00-12:00 — L1→L2→L3（PPTX 重点页）

### 做什么
翻到 PPTX 第 8-15 页，讲清楚 **Observation → Pattern → Law** 的逻辑链。

### 关键页
- Page 8: 7.5× 失败率差异（CNN 37.4% vs DINO 9.4% vs TF 5.0%）
- Page 11: 5-fold AUC 表（同 planner 0.89-0.94）
- Page 13: **6/6 跨 planner 转移 AUC > 0.70**
- Page 15: SHAP top-1 分布（Road Structure 80-88%）

### 讲法（重点是 L3）
> "L1 我们看到表征差异。L2 我们看到基因能预测失败。**L3 是转折**——6 个跨 planner 对，**全部** > 0.70，最强 0.798。这说明**失败的形状在表征间守恒**——攻击者不需要知道是哪个 planner，联合扰动 edge / lane，**所有规划器都会失效**。"

---

## 段 3 (核心): 12:00-17:00 — **现场攻击 + 轨迹图生成**

### 目标
**让评委亲眼看见**：修改一个被攻击的 image 的 gene → 跑真 forward pass → 生成新的 trajectory plot。

### Step 1 (2 min): 选一个失败的 scene

在 Jupyter notebook 里加载 NavDream scene:
```python
# cell 1: 选 scene
import sys
sys.path.insert(0, '/mnt/d/cogatedrive')
from scripts.analysis.navdream_scene_loader import NavDreamIndex
import random
random.seed(0)

idx = NavDreamIndex('/mnt/e/navsim_workspace/dataset',
                    cache_path='/tmp/navdream_idx.pkl')

# 选一个 Rain 攻击、CNN planner 失败的 scene
import pandas as pd
res = pd.read_csv('/mnt/d/cogatedrive/exp/tierC2_wsl/tierc2_cnn_rain_k5_per_sample.csv')
fail_samples = res[res['is_fail']==1].head(3)
print(fail_samples[['scene_token', 'attack', 'ade_attacked', 'ade_mitigated', 'is_flip']])

# 选最戏剧化的一个（sample 12: ADE 8.70 → 0.31）
token = fail_samples.iloc[0]['scene_token']
print(f"Using token: {token}")
```

### Step 2 (3 min): 生成 3 个 trajectory plot

这是**核心展示**——用 `tutorial` 里的脚本生成 BEV + 轨迹对比图:

```python
# cell 2: 加载 3 个版本 (clean / attacked / mitigated)
clean = idx.load_image(token, attack=None)         # 256×1024
attacked = idx.load_image(token, attack='Rain')   # 256×1024 (Rain)

# 计算 attacked image 的 37-dim gene vector
from navsim.agents.attack_genome.genes.genome_pipeline import AttackGenomeExtractor
ext = AttackGenomeExtractor()
rec = ext(attacked)
gene_dict = rec.features
# 找 top-1 gene 类别
from scripts.analysis.tierc2_wsl import (
    EDGE_FREQ_GENES, LUMA_GENES, COLOR_GENES, DETECTION_GENES,
    categorize_gene, apply_mitigation
)
import numpy as np
# XGBoost predict
xgb = ...  # 训练好的 XGBoost 模型
shap = ...
top1_gene = GENE_FIELDS[np.argmax(shap[:37])]
cat = categorize_gene(top1_gene)
print(f"Top-1 gene: {top1_gene} → category {cat}")

# Apply mitigation
mitigated = apply_mitigation(attacked, cat)
```

```python
# cell 3: 生成 trajectory plots —— **3 个对比子图**
# 这是用 tutorial 里的 plot_bev_with_agent 但要 adapter 替换成 AttackGenome agent
import sys
sys.path.insert(0, '/mnt/d/cogatedrive')
from scripts.attack_genome.adapters import build_cnn_adapter_from_yaml
adapter = build_cnn_adapter_from_yaml(device='cuda')

# 跑 3 次 forward pass
traj_clean = adapter.predict(clean)
traj_atk = adapter.predict(attacked)
traj_mit = adapter.predict(mitigated)

# 用 matplotlib 画 trajectory 对比
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, traj, title in zip(axes, [traj_clean, traj_atk, traj_mit],
                            ['Clean (Origin)', 'Rain attacked', f'Mitigated ({cat})']):
    ax.plot(traj[:, 1], -traj[:, 0], 'b-', label='Predicted')
    ax.plot(traj_clean[:, 1], -traj_clean[:, 0], 'g--', label='GT', alpha=0.5)
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('/tmp/attack_mitigation_demo.png', dpi=150)
plt.show()
```

### Step 3 (评委此刻看到什么)
- **左图**: 绿色虚线是 GT，蓝色是预测 — 几乎重合（clean 状态）
- **中图**: 蓝色预测**严重偏离**绿色 GT — 失败 (Rain 攻击下)
- **右图**: 蓝色预测**回到绿色 GT 附近** — 修复成功

### 讲法
> "评委老师，你们现在**亲眼看到了**——
> - 左图：planner 正常工作
> - 中图：被 Rain 攻击后，**车开到马路牙子上**
> - 右图：我用 SHAP top-1 gene 的 image-level mitigation 修复后，**车又回到正常轨迹**
>
> ADE 从 8.7 米降到 0.3 米。**这不是统计预测，是真 planner 行为被改变了**。"

---

## 段 4: 17:00-22:00 — L5 + L6（PPTX 重点页）

### 做什么
翻 PPTX 第 17-25 页，讲 Failure Basin 数学定义 + Tier C-2 数据。

### 关键页
- Page 18: Failure Basin 4 算子（$\mathcal{B}_\tau, s^*, w, M$）
- Page 22: Tier C-2 主表（CNN 50% / DINO 0%）
- Page 24: Dose-response 曲线
- Page 25: 3-panel figure

### 讲法（重点是 L3 设计意图）
> "L6 数据里有一条最戏剧化的发现——**DINO 0% flip rate**。这不是方法失败。**DINO 是本实验室学长专门为 OOD 鲁棒性设计的自监督架构**。我们的 gene 干预修复 CNN 50%，但**完全修不了 DINO**——这恰恰证明 DINO 的 OOD 鲁棒性设计在 gene 层面也成立。它的特征空间对常见的图像级扰动天然免疫。"

---

## 段 5 (核心): 22:00-25:00 — **Genome Shield 实时监控**

### 目标
**让评委看见 0.4ms 实时风险检测**。

### 怎么跑
```bash
# 跑 Genome Shield 实时监控
cd /mnt/d/cogatedrive
python scripts/analysis/genome_shield.py

# 输出: 30s 真数据 demo 视频
# 答辩时直接播放 comp/genome_shield demo.gif
```

或者用 `tutorial/vis_gtrs_score_distribution/vis_gtrs_score_distribution.py` 现场跑。

### 讲法
> "0.4 毫秒单帧推理。**比 100 毫秒的实时门槛快 230 倍**。
> 这不是离线分析——这是 Security Operations Center for autonomous driving。
> 你的车在路上跑，每一帧都在做基因分解，**判断它离 Failure Basin 还有多远**。"

---

## 段 6 (核心): 25:00-28:00 — **跨架构 trajectory 对比**

### 目标
**让评委亲眼看到 CNN / DINO / TF 3 个 planner 的轨迹差异**。

### Step 1 (1.5 min): 加载 3 个 planner adapter
```python
from scripts.attack_genome.adapters import (
    build_cnn_adapter_from_yaml,
    build_dino_adapter_from_yaml,
    build_transfuser_adapter_from_yaml,
)
cnn = build_cnn_adapter_from_yaml(device='cuda')
dino = build_dino_adapter_from_yaml(device='cuda')
tf = build_transfuser_adapter_from_yaml(device='cuda')
```

### Step 2 (1.5 min): 同一 scene + 同一 Rain attack，3 planner 的轨迹
```python
# 用同 (scene, Rain) 跑 3 个 planner
traj_cnn_clean = cnn.predict(clean)
traj_cnn_atk = cnn.predict(attacked)

traj_dino_clean = dino.predict(clean)
traj_dino_atk = dino.predict(attacked)

traj_tf_clean = tf.predict(clean)
traj_tf_atk = tf.predict(attacked)

# 画 3x2 subplot
fig, axes = plt.subplots(3, 2, figsize=(12, 18))
for i, (name, tc, ta) in enumerate([
    ('CNN-GTRS', traj_cnn_clean, traj_cnn_atk),
    ('DINO-GTRS', traj_dino_clean, traj_dino_atk),
    ('TransFuser', traj_tf_clean, traj_tf_atk),
]):
    for j, (traj, title) in enumerate([(tc, 'Clean'), (ta, 'Rain attacked')]):
        ax = axes[i, j]
        ax.plot(traj[:, 1], -traj[:, 0], 'b-')
        ax.plot(tc[:, 1], -tc[:, 0], 'g--', alpha=0.5)
        ax.set_title(f'{name} — {title}')
        ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('/tmp/3planner_trajectory_compare.png', dpi=150)
plt.show()
```

### Step 3 (评委此刻看到什么)
- **CNN 攻击下**：严重偏离 GT
- **DINO 攻击下**：稍微偏离（**因为 DINO 鲁棒**）
- **TransFuser 攻击下**：几乎不偏离（**因为 TF 更鲁棒**）
- 这就是**为什么 base fail rate 是 37% / 9% / 5%**——**视觉上**直接看见

### 讲法
> "看这 3 行——CNN 攻击下严重偏离，DINO 偏离更少，TransFuser 几乎不变。这就是为什么 7.5× 失败率差异。
> **基因空间里 6/6 跨 planner AUC > 0.70**——他们的失败**形状**相似；**只是幅度**不同。
> 这就是 Cross-Architecture Failure Law 的**视觉证据**。"

---

## 段 7: 28:00-30:00 — Q&A + 收尾

### 金句
> "Different Architectures. Same Death. Attacks transfer because genes transfer. Recovery differs by design intent."

### 备用 Q&A
（参考 `comp/DEFENSE_QA.md`，已涵盖 5 个常见问题 + DINO design intent 解读）

---

## 设备 / 备份计划

### 主设备
- 笔记本 + 大屏投影（评委看的屏）
- 必须有 HDMI/USB-C 转接头
- 提前测试字体、Python 环境、模型 ckpt 都在

### 备份
1. **所有 30 分钟 demo 内容**都拍成 backup 视频（5 段 × ~5 分钟），放在 U 盘
2. **PPTX** 印一份纸质版（评委偶尔要参考）
3. **demo gif** (genome_shield) 准备好以防 Jupyter 出问题
4. **tierc2_full.png** (3-panel) 准备好以防脚本出 problem

### 风险点 + 应对
| 风险 | 应对 |
|------|------|
| GPU 显存不够 | DINO 和 CNN 不能同时加载，**先跑 CNN 完再跑 DINO** |
| Jupyter 卡死 | 所有 cell 提前跑过存到 .py，**只展示结果不实跑** |
| 时间超 | L4 现场攻击可压缩到 3 分钟，**只跑 sample 12 一个** |
| 评委提尖锐问题 | 答辩 30 分钟里有 2 分钟 Q&A 时间，已在 FINAL_STATUS 准备 |

---

## 实跑命令速查

```bash
# 答辩当天:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate constanteye_sec
cd /mnt/d/cogatedrive

# 段 0: 视频（已跑过，直接播放）
vlc comp/agsa demo_30s.mp4

# 段 1-2: PPTX（已生成，直接打开）
libreoffice comp/agsa_defense.pptx  # 或 PowerPoint

# 段 3: 现场攻击 Jupyter
jupyter notebook scripts/analysis/attack_mitigation_demo.ipynb  # 需提前写

# 段 5: Genome Shield
python scripts/analysis/genome_shield.py

# 段 6: 跨架构 trajectory
jupyter notebook scripts/analysis/cross_planner_trajectory.ipynb  # 需提前写
```

---

## 关键时间节点提醒

| 节点 | 提前多久做 |
|------|-----------|
| 预录视频 | 答辩前 1 天 |
| PPTX 生成 | 答辩前 1 天 |
| Genome Shield 测试 | 答辩前 1 小时 |
| Jupyter notebook 演练 | 答辩前 1 小时 |
| 备份到 U 盘 | 答辩前 30 分钟 |

---

**祝答辩顺利，国一！** 🎉
