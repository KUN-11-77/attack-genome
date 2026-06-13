# AutoDrive-AttackGenome

## Discovering Common Failure Modes Across Visual Representations for Autonomous Driving

### 自动驾驶视觉规划器跨架构攻击迁移与共同失效模式分析平台

---

# 0. 与现有项目的对接状态

> 本文是研究设计文档（做什么、为什么、目标是什么）。工程层
> 模块组织、运行命令、WSL/服务器手册统一写在
> [`doc/attack_genome/README.md`](doc/attack_genome/README.md) 与
> [`doc/attack_genome/WSL_RUNBOOK.md`](doc/attack_genome/WSL_RUNBOOK.md)，
> 不在本文件重复。运行前请先读这两份文档。

## 0.1 已有可复用资产（cogatedrive 仓库内）

| 模块 | 路径 | 状态 |
| --- | --- | --- |
| 基因提取（7 类） | [`navsim/agents/attack_genome/genes/`](navsim/agents/attack_genome/genes/) | 已实现 |
| 连续攻击空间 | [`navsim/agents/attack_genome/attacks/templates.py`](navsim/agents/attack_genome/attacks/templates.py) | 10 个内置模板 |
| 评估指标 | [`navsim/agents/attack_genome/evaluator/metrics.py`](navsim/agents/attack_genome/evaluator/metrics.py) | ADE / ASR / s_c |
| 迁移性矩阵 | [`navsim/agents/attack_genome/transferability/transfer_analysis.py`](navsim/agents/attack_genome/transferability/transfer_analysis.py) | Pearson / KL / JS |
| 共同失效挖掘 | [`navsim/agents/attack_genome/failure_modes/common_failure.py`](navsim/agents/attack_genome/failure_modes/common_failure.py) | DBSCAN |
| 脆弱图谱 | [`navsim/agents/attack_genome/vulnerability/vulnerability_atlas.py`](navsim/agents/attack_genome/vulnerability/vulnerability_atlas.py) | 已实现 |
| Planner 适配器 | [`scripts/attack_genome/adapters.py`](scripts/attack_genome/adapters.py) | CNN / DINO / VLM-proxy |
| NavSim mini 加载 | [`scripts/attack_genome/navsim_loader.py`](scripts/attack_genome/navsim_loader.py) | stitch_for_gtrs + GT |
| 冒烟测试 | [`tests/attack_genome/test_smoke.py`](tests/attack_genome/test_smoke.py) | 13 项已实现 |
| Demo pipeline | [`scripts/attack_genome/run_pipeline.py`](scripts/attack_genome/run_pipeline.py) | demo / real 两种模式 |

设计文档对应实现：

| 设计章节 | 实现位置 |
| --- | --- |
| 3 Attack Genome | `navsim/agents/attack_genome/genes/` |
| 4 Representation Family | `navsim/agents/attack_genome/evaluator/representation_family.py` |
| 5 Continuous Attack Space | `navsim/agents/attack_genome/attacks/templates.py` |
| 6.1 ADE/ASR | `navsim/agents/attack_genome/evaluator/metrics.py` |
| 6.3 s_c | `navsim/agents/attack_genome/evaluator/metrics.py::safety_phase_transition` |
| 6.4 Pearson/KL | `navsim/agents/attack_genome/transferability/transfer_analysis.py` |
| 7 Common Failure Modes | `navsim/agents/attack_genome/failure_modes/common_failure.py` |
| 8 Vulnerability Atlas | `navsim/agents/attack_genome/vulnerability/vulnerability_atlas.py` |

## 0.2 数据与模型落位（已盘点）

| 资产 | 位置 | 用途 |
| --- | --- | --- |
| NAVSIM logs (trainval + test) | `E:\navsim_workspace\dataset\navsim_logs\` | 真实场景 GT + sensor |
| NavDream 基准（10 风格 + PGD） | `E:\navsim_workspace\dataset\navdream\benchmark\` | 连续攻击模板素材 |
| Sensor blobs (mini + test) | `E:\navsim_workspace\dataset\sensor_blobs\` | 多相机图像 |
| Maps | `E:\navsim_workspace\dataset\maps\` | NUPlan maps |
| Metric cache | `E:\navsim_workspace\dataset\metric_cache\` | PDM 评分缓存 |
| GTRS-Dense (VoV) ckpt | `E:\navsim_workspace\models\navsim_ckpts\gtrs_dense_vov.ckpt` | CNN planner |
| GTRS-Dense-DINO ckpt | `E:\navsim_workspace\models\navsim_ckpts\gtrs_dino_epoch_47_from_scratch.ckpt` | DINO planner |
| DiffusionDrive ckpts | `E:\navsim_workspace\models\navsim_ckpts\` | 备用 planner |
| DINOv3 ViT-H/16+ | `E:\navsim_workspace\models\dinov3_vith16plus_pretrain_lvd1689m.pth` | DINO backbone |
| DD3D VoV | `E:\navsim_workspace\models\navsim_ckpts\gtrs_dd3d_det_final.pth` | VoV 预训练 |
| Qwen2-VL-7B | `E:\navsim_workspace\models\qwen2-vl-7b\qwen\` | VLM planner（B 阶段） |
| ResNet-34 (timm) | `E:\navsim_workspace\models\resnet34_timm\` | 备用 CNN backbone |
| 轨迹词表 | `E:\navsim_workspace\traj_final\8192.npy` & `16384.npy` | GTRS dense vocab |
| 现有 demo 结果 | `E:\navsim_workspace\exp\`（bev_traj_tutorial, trajectory_overlay, phase_b_pgd …） | Phase A/B/C 历史 |

## 0.3 算力布局（本地 + 服务器）

| 环境 | 设备 | 角色 | 备注 |
| --- | --- | --- | --- |
| 本地 Win11 | RTX 5060 (8GB) | 仅 Tier A（CPU 冒烟） | GTRS-VoV 装不下 |
| 服务器 | 6 × RTX 3080 Ti (12GB) | Tier B / C / D | 服务器 IP/端口见 `doc/attack_genome/WSL_RUNBOOK.md` §6 |

> 依据：本地 5060 跑完整 500 场景 × 10 攻击 × 6 强度 × 3 planner ≈ **15 小时**，
> 服务器 6×3080 Ti 串并行后估算 ≤ **2 小时**，因此真实 NavDream sweep
> 一律走服务器，本地只做纯 CPU 冒烟。

## 0.4 路径约定（WSL + 服务器）

| 用途 | 本地 (Windows) | WSL | 服务器 (khsong，**有 NAS 只读、无写权限**) |
| --- | --- | --- | --- |
| 代码 | `D:\cogatedrive` | `/mnt/d/cogatedrive` | `/data3/khsong/cogatedrive`（也可 symlink 到 `/home/khsong/cogatedrive`） |
| 数据 (NAVSIM) | `E:\navsim_workspace\dataset\` | `/mnt/e/navsim_workspace/dataset/` | **可读** `/nas/users/jbwang/navsim/benchmark/outputs/benchmark/`；**可写** `/data3/khsong/data/navsim/dataset/`（从本地或 NAS 同步过来） |
| 原始 openscene | `D:\navsim_data\openscene-v1.1\` | `/mnt/d/navsim_data/openscene-v1.1/` | `/data3/khsong/data/navsim/openscene-v1.1/`（**从 NAS 只读侧 rsync 一次**即可） |
| 模型 ckpt | `E:\navsim_workspace\models\` | `/mnt/e/navsim_workspace/models/` | `/data3/khsong/data/navsim/models/`（按 symlink 引用，不要复制） |
| 实验产物 | `E:\navsim_workspace\exp\attack_genome\` | `/mnt/e/navsim_workspace/exp/attack_genome/` | `/data3/khsong/exp/attack_genome/`（**写这里**） |
| Conda env | n/a | `/home/songkunhong/miniconda3/envs/constanteye_sec` | `/data3/khsong/envs/<env>`（按 `/data*/khsong/envs/...` 约定自建） |

> 路径在不同环境下不通用；任何写死 `/mnt/e/...` 的脚本都要靠
> `NAVSIM_EXP_ROOT` / `OPENSCENE_DATA_ROOT` / `NAVSIM_DEVKIT_ROOT`
> 环境变量覆盖；服务器脚本统一从 `/data3/khsong/cogatedrive/scripts/attack_genome/server_*.sh` 走。

### 0.4.1 服务器存储策略（NAS 只读 + 本地盘可写）

* **可读**：通过 `/nas/users/jbwang/navsim/benchmark/outputs/benchmark/`
  直接访问 jbwang 已有的 NAVSIM mini / navdream benchmark / sensor_blobs
  等数据，**不要复制**到本地盘；
* **可写**：所有**新生成的实验产物**、**本用户私有数据**写到
  `/data3/khsong/exp/attack_genome/`；
* **conda env** 装在 `/data3/khsong/envs/<env>`（约定 `/data*/khsong/envs/...`）；
* 服务器脚本的环境变量模板（**NAS 只读侧直接用绝对路径**）：

  ```bash
  # 读：直接挂 jbwang 的 NAS benchmark
  export NAVSIM_DEVKIT_ROOT=/data3/khsong/cogatedrive
  export OPENSCENE_DATA_ROOT=/nas/users/jbwang/navsim/benchmark/outputs/benchmark
  export NAVSIM_WORKSPACE_ROOT=/nas/users/jbwang/navsim/benchmark/outputs/benchmark
  # 写：落到本用户目录
  export NAVSIM_EXP_ROOT=/data3/khsong/exp/attack_genome
  ```

* **不要**往 `/nas/users/jbwang/...` 写任何东西——会直接报 `Permission denied`；
  如果发现脚本硬编码 NAS 写路径，统一改成 `/data3/khsong/exp/attack_genome/`。

---

# Abstract

随着自动驾驶系统逐渐从传统监督学习CNN发展到自监督视觉模型以及视觉语言模型，一个关键安全问题尚未得到系统研究：

**当视觉表征发生演化时，曾经有效的攻击是否仍然有效？**

现有工作主要关注单模型鲁棒性评测或攻击设计，却缺乏对攻击跨架构迁移性的系统研究。

本项目提出 AutoDrive-AttackGenome。

我们利用 NavDream 提供的可控外观操控空间，构建自动驾驶视觉攻击基因组（Attack Genome），并在 CNN（GTRS-Dense VoV）、自监督 DINO（GTRS-Dense-DINO）以及视觉语言（ReCogDrive / Qwen2-VL 代理）构成的表征家族上，系统分析攻击迁移行为、脆弱场景分布以及共同失效模式。

项目最终输出：

* Attack Genome（攻击基因组）
* Vulnerability Atlas（脆弱场景图谱）
* Common Failure Modes（共同失效模式）

为自动驾驶视觉安全评估与下一代防御设计提供依据。

---

# 1. 问题定义

## 1.1 研究背景

自动驾驶系统越来越依赖视觉模型完成环境理解与轨迹规划。

近年来视觉骨干经历明显演化：

CNN

↓

Self-Supervised Vision Model

↓

Vision-Language Model

然而目前尚不清楚：

* 模型升级是否能够消除已有攻击；
* 哪些攻击具有跨架构迁移能力；
* 是否存在所有视觉架构共享的失效模式。

## 1.2 核心科学问题

### Q1
不同视觉表征是否会显著改变攻击有效性分布？

### Q2
哪些攻击基因最能预测跨架构迁移行为？

### Q3
是否存在使所有视觉架构同时失效的共同失效模式？

---

# 2. 威胁模型

## 攻击目标
诱导规划器产生危险轨迹偏移：
* 偏离车道
* 错误转向
* 碰撞风险提升

## 攻击能力
攻击者能够通过：
* LED广告牌
* 投影系统
* 数字显示屏
* 可替换视觉纹理

对车辆视觉输入产生外观操控。

## 攻击约束
攻击者不能：
* 修改道路拓扑结构
* 篡改车辆内部系统
* 修改地图

只能影响视觉外观。

## 研究范围
本项目研究：
Adversarial Appearance Manipulation
即：对抗性外观操控。
NavDream生成的风格变化被视为攻击模板，而非自然天气现象。

---

# 3. Attack Genome

## 3.1 核心思想

传统工作关注：攻击类型
本项目关注：攻击基因
即：导致攻击产生效果的最小可解释单元。

## 3.2 一级基因
* **Frequency Gene**：低频能量占比、高频能量占比
* **Color Gene**：HSV偏移、饱和度变化
* **Texture Gene**：LBP熵、GLCM对比度
* **Contrast Gene**：RMS Contrast

## 3.3 二级基因
* **Structural Gene**：道路可见度、车道线清晰度
* **Semantic Gene**：车辆显著度、行人显著度、交通设施显著度
* **Illumination Gene**：阴影覆盖率、强光区域占比

## 3.4 基因提取正确性保障（Gene Extraction Correctness）

7 类基因全部用纯 numpy + opencv 实现（见
[`navsim/agents/attack_genome/genes/`](navsim/agents/attack_genome/genes/)），
不依赖训练框架，便于离线批处理和单元测试。下表列出每个基因的具体
算法与可验证的不变量。

| 基因 | 实现 | 数学定义 | 数值可验证不变量 |
| --- | --- | --- | --- |
| **Frequency** | `frequency_gene.py` | 2D FFT → 半径网格 → 低/中/高频能量占比 | ① `low + mid + high ≡ 1.0`；② 全零图 → `spectral_centroid = 0`；③ 直流分量减除后能量守恒 |
| **Color** | `color_gene.py` | RGB→HSV → H/S/V 通道统计 | ① 灰度图 `sat_mean ≈ 0`；② HSV 范围合法；③ `colorfulness ∈ [0,1]` |
| **Texture** | `texture_gene.py` | Uniform LBP 直方图熵 + 自实现 GLCM 对比度（量化到 16 级） | ① 单色图 `lbp_entropy ≈ 0`；② 噪声图 `lbp_entropy` 单调上升；③ `lbp_uniformity ∈ [1/n_bins, 1]` |
| **Contrast** | `contrast_gene.py` | RMS Contrast = `sqrt(mean(gray²))` + 亮度差 | ① 全黑图 `rms_contrast = 0`；② 全白图 `rms_contrast = 1`；③ `reference` 注入后 `mean_shift` 与已知亮度差一致 |
| **Structural** | `structural_gene.py` | Sobel 边缘密度 + Canny + HoughLinesP 车道线计数 | ① 平滑图 `edge_density ≈ 0`；② 渐变图 `edge_density` 随梯度幅值单调；③ Hough 线段数随阈值单调下降 |
| **Semantic** | `semantic_gene.py` | **启发式代理**：HSV 阈值 + 水平带分布 | ⚠️ **当前用色块比例作代理**，不是真值检测；详见 §3.4.2 |
| **Illumination** | `illumination_gene.py` | 阴影/高光占比 + 32-bin 直方图熵 + 偏度 | ① 全黑图 `shadow_ratio = 1`；② 全白图 `highlight_ratio = 1`；③ 偏度符合 `(mean−µ)^3 / σ^3` 解析公式 |

### 3.4.1 当前测试覆盖与缺口（必须诚实承认）

[`tests/attack_genome/test_smoke.py`](tests/attack_genome/test_smoke.py) 现有 13
个测试，但 `TestGenes` 仅有 2 项：

* `test_extract_returns_full_field_set`：仅检查 `GENOME_FIELD_ORDER`
  中所有字段都出现在 `record.features` 里，**不验证数值**；
* `test_extract_with_reference`：仅检查 `mean_shift` / `std_shift` 字
  段存在，**不验证数值**。

**结论：现冒烟测试只覆盖 API 形状，不覆盖数学正确性。** 这是当前
最大的可信度漏洞，必须在 Tier A.2 补齐。

### 3.4.2 Semantic Gene 是启发式代理，不是真值检测

[`semantic_gene.py:9-12`](navsim/agents/attack_genome/genes/semantic_gene.py#L9-L12)
明确写明"在没有真值标签的情况下使用颜色/形状的启发式代理"，
具体做法：

* `vehicle_saliency` / `pedestrian_saliency` / `traffic_saliency`：
  按水平带分块 + HSV 饱和度阈值计数显著色块；
* `lane_yellow_ratio`：HSV 中黄色 H∈[15,40] 像素占比。

**这一代理有以下已知局限，必须在论文 §X.X 明确披露**：

1. **对车体颜色**：浅灰/银色车辆会被低估，深色车辆与背景难以区分；
2. **对行人**：穿浅色衣服的行人会被显著高估；
3. **对攻击的鉴别力低**：局部对抗补丁可能不影响水平带统计；
4. **不提供空间精确度**：只输出比例，无 bbox/segmap。

**升级路径**（A→B 阶段）：

* 用 nuScenes / OpenScene 现成的 3D bbox 标签统计真值 saliency（仅
  适用于有真值标注的子集，如 navtrain）；
* 或者用冻结的 YOLOv8 / DINO 检测器输出做近似 saliency（代价是
  牺牲 0–5 FPS，需要 GPU）；
* 写作时把"proxy vs detector-based"作为 ablation 章节。

### 3.4.3 Tier A.2 — 基因提取正确性补强（开工顺序）

在 `tests/attack_genome/` 下新增 `test_genes_correctness.py`，对每个
基因至少写一个**带解析解**的数值断言：

```python
# 例：FrequencyGeneExtractor
def test_frequency_zero_image_gives_zero_centroid():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    rec = extract_genome_from_array(img, scene_token="z")
    assert rec.features["low_freq_ratio"] + \
           rec.features["mid_freq_ratio"] + \
           rec.features["high_freq_ratio"] == pytest.approx(1.0, abs=1e-3)
    assert rec.features["spectral_centroid"] == pytest.approx(0.0, abs=1e-6)

def test_frequency_constant_image_gives_zero_energy():
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    rec = extract_genome_from_array(img, scene_token="c")
    # 直流分量被减除，所有能量比应接近 0
    assert rec.features["spectral_centroid"] < 1e-3
```

每个基因至少 2 个测试：

| 基因 | 测试 1（极值图） | 测试 2（不变量） |
| --- | --- | --- |
| Frequency | 全黑 / 全白图 → 离心度 ≈ 0 | 三段能量和 ≡ 1 |
| Color | 灰度图 → sat_mean ≈ 0 | HSV 范围合法 |
| Texture | 单色图 → lbp_entropy ≈ 0 | noise 图 → lbp_uniformity 单调下降 |
| Contrast | 全黑 → rms=0；全白 → rms=1 | reference 注入后 mean_shift 与解析一致 |
| Structural | 平滑图 → edge_density ≈ 0 | 渐变图 → edge_density 随梯度单调 |
| Semantic | 全黑图 → 4 个比例 ≈ 0 | 注入高饱和色块 → colorfulness 单调上升 |
| Illumination | 全黑 → shadow_ratio=1 | 全白 → highlight_ratio=1；偏度解析公式核对 |

跑通后才能在论文里写"基因提取在 X 个数值不变量上经过验证"。

### 3.4.4 参考图不变量（Reference Invariant）

[ContrastGeneExtractor](navsim/agents/attack_genome/genes/contrast_gene.py)
接受 `reference` 参数，可用于验证整条管线：

* `extract(image, reference=image)` → `mean_shift = 0` 且 `std_shift = 0`；
* `extract(reference)` 走 single-image 路径（不写入 shift 字段）→ 与
  `extract(image, reference=image)` 的非 shift 字段完全相同。

这是端到端最便宜的不变量，可在 pipeline test 里一并验证。

---

# 4. Representation Family

本项目不预设模型能力排序。研究对象定义为 **Representation Family**。

| Planner | Backbone | Representation | ckpt 路径 |
| --- | --- | --- | --- |
| GTRS-Dense | VoV (DD3D 预训练) | Supervised CNN | `e:/navsim_workspace/models/navsim_ckpts/gtrs_dense_vov.ckpt` |
| GTRS-Dense-DINO | DINOv3 ViT-H/16+ | Self-Supervised | `e:/navsim_workspace/models/navsim_ckpts/gtrs_dino_epoch_47_from_scratch.ckpt` |
| ReCogDrive / VLM-proxy | DINOv3 → Qwen2-VL-7B | Vision-Language | `e:/navsim_workspace/models/qwen2-vl-7b/` |

> A 阶段 VLM 用 proxy 方案（同一 DINO 模型换 representation 名）；
> B 阶段再升级到真 ReCogDrive + Qwen2-VL-7B + flash_attn。

---

# 5. Continuous Attack Space

利用 NavDream 构建连续攻击空间。

攻击模板（10 个）：

* Rain
* Snow
* Dusk
* Dawn
* Motion Blur
* Digital Noise
* Light Dust
* Dappled Light
* Vintage Style
* CARLA Style

攻击强度：`s ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}`

实验规模：
* 全量：**500 场景 × 10 攻击 × 6 强度 × 3 planner = 90,000 次评估**（Tier D，服务器）
* 最小 demo：**50 场景 × 10 攻击 × 6 强度 × 3 planner = 9,000 次评估**（Tier C，服务器）
* 快速验证：**4 合成场景**（Tier A，本地纯 CPU）

---

# 6. 最小 Demo 分级与执行计划

> 这是当前阶段的实际执行表，按"是否能复现 design.md 的故事线"切分。
> 详见 [`doc/attack_genome/README.md`](doc/attack_genome/README.md) 与
> [`doc/attack_genome/WSL_RUNBOOK.md`](doc/attack_genome/WSL_RUNBOOK.md)。

| Tier | 内容 | 估时（本地 5060） | 估时（服务器 6×3080Ti） | 是否阻塞设计文档 |
| --- | --- | --- | --- | --- |
| A | pytest smoke + 合成 demo pipeline + 单图基因提取 | < 5 min | < 2 min | 是（必须通过） |
| B | 加载 3 个真实 planner 适配器（CNN/DINO/VLM-proxy）+ 1 张真实 NavDream 图过 pipeline | 装不下 GTRS-VoV | 5–10 min | 是 |
| C | 50 真实 NavDream 场景 × 10 攻击 × 6 强度 × 3 planner，输出 result.json | 估 6–8 h（且 VRAM 不够） | 1–2 h | 是 |
| D | 全量 500 场景 sweep | 15 h（基线） | 6–10 h | 否（论文补充） |

## 6.1 Tier A — 本地 5060 即可

```bash
# WSL 下
cd /mnt/d/cogatedrive

# A.1 — 现有冒烟（13 passed）
python -m pytest tests/attack_genome/test_smoke.py -v

# A.2 — 基因提取正确性补强（§3.4.3 新增，7 基因 × 2 测试 ≈ 14 项）
python -m pytest tests/attack_genome/test_genes_correctness.py -v

# A.3 — 合成数据 demo pipeline
python scripts/attack_genome/run_pipeline.py --mode demo \
    --num-scenes 4 --output outputs/ag_smoke

# A.4 — 单图基因提取
python scripts/attack_genome/extract_genome.py \
    --image assets/test.jpg \
    --attack Rain --strength 0.6 \
    --output outputs/ag_smoke/genome.json
```

> A.2 必须在 A.1 之后立即补；它是 §3.4.1 暴露的"测试只覆盖
> API 形状、不覆盖数学正确性"缺口的最小补丁。A.2 不过的话
> B / C 都先不要上服务器。

## 6.2 Tier B — 服务器

```bash
ssh khsong@10.13.74.231 -p 66
# 第一次：先改密码，按 GPU 群文档操作
cd /home/khsong/cogatedrive
bash scripts/attack_genome/server_smoke_planners.sh
```

输出：`outputs/ag_server/smoke_planner_report.md`

## 6.3 Tier C — 服务器最小真实数据 demo

```bash
# 服务器
bash scripts/attack_genome/server_run_mini_50.sh
# 内部：python scripts/attack_genome/run_pipeline.py --mode real \
#   --scenes-json /data3/khsong/exp/attack_genome/scenes_50.json \
#   --output /data3/khsong/exp/attack_genome/ag_mini_50
```

输出：
* `result.json`：ASR 曲线 + 迁移性矩阵 + 共同失效聚类 + 脆弱图谱
* `figures/`：BEV 轨迹对比、基因指纹热力、迁移性矩阵图

> 50 场景是"够让设计文档 story 站得住"的最小规模；继续扩到
> 500 场景属于 Tier D，作为论文补充实验。

## 6.4 Tier D — 全量 sweep（论文阶段）

在 Tier C 跑通后，把 `--num-scenes 500`、拆 6 卡并行；预计 6–10 h。

---

# 7. Evaluation

## 7.1 攻击成功定义
`ADE > 1m` 记为攻击成功。

## 7.2 攻击成功率
ASR — Attack Success Rate。

## 7.3 安全相变点
模型 ASR 达到 50% 时对应的攻击强度，记为 `s_c`。

## 7.4 跨架构迁移性
* Pearson Correlation
* KL Divergence
* JS Divergence

实现：`navsim/agents/attack_genome/transferability/transfer_analysis.py`

---

# 8. Common Failure Modes

## 定义
若同一 `(scene, attack)` 同时导致 CNN / DINO / CLIP 三种表征全部失效，
则定义为 Common Failure Mode（共同失效模式）。

## 挖掘流程
1. 收集共同失效样本
2. 提取场景属性 + 攻击基因
3. DBSCAN 聚类
4. 输出 `CommonFailureAtlas`

实现：`navsim/agents/attack_genome/failure_modes/common_failure.py`

---

# 9. Vulnerability Atlas

输出高风险场景地图，重点分析：
* 夜间
* 弯道
* 高车流
* 低照度
* 道路遮挡

输出：Top Vulnerable Scenarios + 对应攻击基因指纹。

实现：`navsim/agents/attack_genome/vulnerability/vulnerability_atlas.py`

---

# 10. 预期发现

本项目不预设结果。可能发现：

**Case A** 模型升级显著降低攻击风险

**Case B** 部分攻击具有跨架构迁移性

**Case C** 低频扰动主导共同失效

**Case D** 存在共享脆弱场景

任意结果均具有安全价值。

---

# 11. 核心贡献

**Contribution 1**：首次系统研究自动驾驶视觉攻击跨架构迁移行为。

**Contribution 2**：提出 Attack Genome，建立 `Attack → Gene → Failure` 分析框架。

**Contribution 3**：提出 Common Failure Modes 概念，发现视觉架构共享脆弱点。

**Contribution 4**：构建 Vulnerability Atlas，支持安全测试与防御设计。

---

# 12. 答辩总结

现有研究关注：某个模型是否会被攻击。
我们的研究关注：当视觉系统发生演化时，攻击是否仍然有效；哪些攻击
会失效；哪些攻击能够跨越不同视觉架构；以及哪些场景会成为所有视觉
系统共同的安全瓶颈。

这正是自动驾驶视觉安全从"单模型鲁棒性"走向"系统级安全规律研究"的关键一步。

---

# 13. 服务器与本地协作约定

## 13.1 服务器使用前置（按用户 2026-06-10 要求）
1. 第一次登陆服务器后**立即改密码**；
2. **先读** [GPU 服务器使用文档](https://docs.qq.com/doc/DSmpyQkFyeGhTVVdo)；
3. 6×RTX 3080 Ti（12GB）比本地 RTX 5060（8GB）有更大显存；
   但**Ampere vs Blackwell** 单卡推理吞吐低于本地，单卡不要当 5060 用；
4. **conda** 在 `/data*/khsong/`（约定 `/data3/khsong/envs/<env>`）；
   **代码**放 `/data3/khsong/cogatedrive`（也可 symlink 到 `/home/khsong/`）；
   **NAS 是只读**，可以直接读 jbwang 的
   `/nas/users/jbwang/navsim/benchmark/outputs/benchmark/`；
   **本用户写**的产物 / 数据落到 `/data3/khsong/exp/attack_genome/`；
5. 上传数据用 `rsync` 从本地 `D:\navsim_data` /
   `E:\navsim_workspace` 推到 `/data3/khsong/data/navsim/`（如果 NAS 上
   已有同名数据，直接读 NAS，不要复制）。

## 13.2 代码里的硬编码路径
* `scripts/attack_genome/adapters.py::_resolve_default_paths` 默认指向
  `/mnt/e/navsim_workspace` —— 服务器运行前需要导出
  `NAVSIM_WORKSPACE_ROOT=/nas/users/jbwang/navsim/benchmark/outputs/benchmark`
  并修改该函数或用环境变量覆盖；
* `scripts/attack_genome/adapters.py::_ensure_env` 设置 `NAVSIM_EXP_ROOT`、
  `NAVSIM_DEVKIT_ROOT`、`OPENSCENE_DATA_ROOT`；脚本入口
  应统一读这些变量，不要再写 `/data/jbwang/...` 老路径；
* 老脚本里**写** `/nas/users/jbwang/navsim/benchmark/outputs/benchmark/`
  的部分替换为 `/data3/khsong/exp/attack_genome/`，**读**的部分保留为
  `/nas/users/jbwang/navsim/benchmark/outputs/benchmark/`（或
  用 `NAVSIM_WORKSPACE_ROOT` 覆盖）。

## 13.3 跑实验前的最低 checklist
1. `conda activate <env>`（WSL：`constanteye_sec`；服务器：用户专属 env）
2. 确认 `nvidia-smi` 看得到卡（服务器看到 6 张 3080 Ti）
3. 确认 `/nas/users/jbwang/...`（只读侧）和
   `/data3/khsong/exp/attack_genome/`（可写侧）都已就位
4. 先跑 Tier A 通过再开 Tier B
5. Tier C 之后每次跑都写 `experiments/<date>_tierC_*/RESULT.md`，
   并把 result.json 和关键 figure 备份到 `/data3/khsong/exp/attack_genome/`

## 13.4 备份策略
* 关键代码改动 push 到 git（cogatedrive 仓库）；
* 每个 Tier 完成的 result.json / figures 同步到
  `/data3/khsong/exp/attack_genome/<tier>/`；
* 模型 ckpt **不要**复制，按 symlink 引用 NAS 已有
  `/nas/users/jbwang/.../navsim_ckpts/` 即可。