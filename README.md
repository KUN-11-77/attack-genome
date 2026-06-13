# Attack Genome Law

> **Cross-Architecture Failure Law of Vision-Based Planners: A Gene-Space Diagnosis**
>
> 全国大学生信息安全竞赛（作品赛）自由赛道 — 参赛项目

[![Status](https://img.shields.io/badge/Status-Tier_C2_Complete-brightgreen)]()
[![Target](https://img.shields.io/badge/Target-国赛一等奖-blue)]()
[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12%2BCUDA-orange)]()

---

## 🎯 项目核心命题

视觉端到端规划器在自然扰动下表现不稳定。我们把"攻击名称"换成 **37 维基因空间**，在 3 类视觉规划器上系统诊断了攻击成功/失败的因：

> **Cross-Architecture Failure Law**: 规划器失败机制是 **gene 介导** 的，跨 planner 共享；但 **修复机制是架构设计依赖** 的（CNN 可修，DINO 因 OOD 鲁棒性设计天然免疫）。

---

## 📊 关键发现（3 层叙事）

| Layer | 主张 | 证据 |
|-------|------|------|
| **L1** | 失败机制跨 planner 共享（gene→fail 信号）| Tier B 6/6 跨 planner AUC > 0.70 |
| **L2** | Image-level gene 干预能修复 CNN（执行层因果）| Tier C2：CNN Rain K=5 翻 50%（n=50 复现 47.6%）|
| **L3** | DINO 对同样干预完全免疫 → **验证 OOD 鲁棒性设计意图** | Tier C2：DINO 0% flip |

**单调剂量响应** (K=1: 12.5% → K=5: 50% → K=10: 86%) 证明 gene→fix 是定量因果。

---

## 🔬 实验数据速览

### Tier C2: 1883 NavDream scenes × CNN/DINO × 3 attacks × K=1,5,10

| 实验 | n | n_fail | n_flip | flip/fail | 含义 |
|------|---|--------|--------|-----------|------|
| CNN, Rain, K=1, n=20 | 20 | 8 | 1 | **12.5%** | 单 gene baseline |
| **CNN, Rain, K=5, n=20** | 20 | 8 | 4 | **50.0%** | ⭐ 主结果 |
| **CNN, Rain, K=5, n=50** | 50 | 21 | 10 | **47.6%** | ⭐ 复现确认 |
| CNN, Dusk, K=5, n=20 | 20 | 9 | 5 | **55.6%** | 跨攻击泛化 |
| CNN, DigitalNoise, K=5, n=20 | 20 | 6 | 1 | 16.7% | 边界案例（pixel-noise）|
| **DINO, Rain, K=5, n=20** | 20 | 5 | **0** | **0%** | ⭐ 验证 OOD 设计意图 |

### 3 planner × 88,560 样本 (Tier B 已完成)

| Planner | Fail rate | 跨 planner transfer AUC |
|---------|-----------|--------------------------|
| CNN-GTRS (VoVNet-99) | 37.4% | 6/6 对 > 0.70 |
| DINO-GTRS (DINOv2 ViT-L) | 9.4% | (vs CNN: 0.798) |
| TransFuser (ResNet-34 + BEV) | 5.0% | (vs CNN: 0.756) |

---

## 📁 仓库结构

```
attack-genome/
├── README.md                    # 本文件
├── ATACK_GENOME_LAW.md          # Tier B 完整 LAW 文档
├── tierc2_final.md              # Tier C2 完整报告 (执行层因果)
├── design.md                    # 项目设计文档
├── design_sci.md                # SCI 视角设计
├── IDEA_REPORT.md               # 早期 idea 调研
├── README_GTRS_UPSTREAM.md      # 上游 GTRS README（保留署名）
│
├── scripts/
│   ├── analysis/                # ⭐ 分析 + 实验脚本
│   │   ├── tierc2_wsl.py            # Tier C2 主实验
│   │   ├── navdream_scene_loader.py # NavDream scene loader
│   │   ├── forward_pass_gene_counterfactual.py
│   │   ├── plot_tierc2_full.py      # 3-panel 论文级图
│   │   ├── cross_planner_predict.py
│   │   ├── failure_basin_counterfactual.py
│   │   └── ...
│   ├── attack_genome/            # Planner adapter
│   ├── evaluation/
│   ├── training/
│   └── submission/
│
├── navsim/                      # NavSIM 源码 (13MB)
│   └── agents/
│       ├── attack_genome/        # ⭐ Attack Genome 核心
│       │   ├── genes/            # 37-dim gene extractors
│       │   ├── attacks/          # 10 种攻击实现
│       │   ├── evaluator/        # 评估 + PlannerAdapter
│       │   ├── failure_modes/    # 失败模式分类
│       │   ├── transferability/  # 迁移性分析
│       │   └── vulnerability/    # 漏洞图谱
│       ├── gtrs_dense/           # CNN-GTRS
│       ├── gtrs_dense_dino/      # DINO-GTRS
│       └── diffusiondrive/       # TransFuser
│
├── exp/
│   └── tierC2_wsl/              # ⭐ Tier C2 完整 outputs
│       ├── tierc2_cnn_rain_k{1,5,10}_*.csv
│       ├── tierc2_cnn_dusk_k5_*
│       ├── tierc2_cnn_digitalnoise_k5_*
│       ├── tierc2_dino_rain_k5_*
│       ├── tierc2_final.md
│       └── figures/
│           ├── dose_response.png
│           ├── dose_response.pdf
│           ├── tierc2_full.png   # ⭐ 3-panel 答辩图
│           └── tierc2_full.pdf
│
├── environment.yml
├── requirements.txt
├── AGENTS.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## 🚀 快速开始

### 环境准备

```bash
# 推荐 Python 3.10, CUDA 12.x
conda create -n constanteye_sec python=3.10
conda activate constanteye_sec

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install nuplan-devkit shapely geopandas rasterio aioboto3 retry
pip install xgboost opencv-python-headless timm
```

### 数据准备

- **NavDream scenes** (~1883 scenes, 11 预计算攻击) → `/path/to/navdream_benchmark_outputs/`
- **模型 ckpts** → `/path/to/models/navsim_ckpts/`:
  - `gtrs_dense_vov.ckpt` (CNN)
  - `gtrs_dino_epoch_47_from_scratch.ckpt` (DINO)
  - `transfuser_seed_0.ckpt` (TransFuser)

### 跑 Tier C2

```bash
# 主实验: CNN, Rain, K=5, n=20
python scripts/analysis/tierc2_wsl.py \
    --n 20 --k 5 --planner CNN --attack Rain \
    --navdream-root /path/to/navsim_workspace/dataset \
    --csv /path/to/exp/tierB_partial/merged_3pl.csv \
    --output-dir /path/to/exp/tierC2_wsl

# 画图
python scripts/analysis/plot_tierc2_full.py
```

---

## 🏆 国家级奖项目标

**项目定位**：全国大学生信息安全竞赛（作品赛）自由赛道
**预期奖项**：国家一等奖

| 维度 | 评估 |
|------|------|
| 选题前沿性 | 自动驾驶 + 对抗鲁棒性 + 因果推断（**最热**）|
| 技术深度 | CV + ML + 规划 + 因果科学（**深**）|
| 实验规模 | 88,560 + 1883 × 3 planner × 10 attack |
| 创新性 | **"信号共享 + 修复分歧"** — 跨架构因果二分，首例 |
| 完整性 | 3 层叙事 + 7 个独立实验 + 答辩 figure + Q&A |

---

## 📚 文档导航

- **[ATACK_GENOME_LAW.md](ATACK_GENOME_LAW.md)** — Tier B 完整 LAW 文档
- **[tierc2_final.md](exp/tierC2_wsl/tierc2_final.md)** — Tier C2 完整执行层因果报告
- **[design.md](design.md)** — 项目设计文档
- **[exp/tierC2_wsl/figures/](exp/tierC2_wsl/figures/)** — 答辩 figure (300dpi PNG + PDF)

---

## 🔬 复现与扩展

### 复现 Tier B (88,560 样本)

```bash
python scripts/analysis/cross_planner_predict.py \
    --csv exp/tierB_partial/merged_3pl.csv

python scripts/analysis/failure_basin_counterfactual.py \
    --planner CNN --n-fail 100 --k-values 1 5 10 20
```

### 未来方向

- **Tier C1**: 跨 dataset 验证（nuScenes / Waymo）
- **Tier C2+**: TransFuser K=5（完成 3-planner 闭环）
- **L4**: 与 diffusion-based counterfactual attack 连接

---

## 📝 引用

```bibtex
@misc{attack_genome_law_2026,
  title={Cross-Architecture Failure Law of Vision-Based Planners: A Gene-Space Diagnosis},
  author={CogateDrive},
  year={2026},
  note={National College Student Information Security Competition (Works Track)},
  howpublished={\url{https://github.com/KUN-11-77/attack-genome}}
}
```

---

## 📄 许可证

本项目代码遵循项目根目录的 `LICENSE`。

## 👥 致谢

- 指导老师 & 实验室同门
- 学长（DINO 架构设计者）的 OOD 鲁棒性设计哲学启发了 L3 验证
- 答辩评委老师们的批判性反馈

