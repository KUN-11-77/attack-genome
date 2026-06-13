# Research Idea Report

**Direction**: Use `RESEARCH_BRIEF.md` as context. Focus on lightweight high-cognition end-to-end driving for NAVSIM.
**Generated**: 2026-04-09
**Status**: validated draft — literature survey + first-pass filtering + deeper novelty check + external reviewer criticism completed
**Ideas evaluated**: 10 generated → 5 survived filtering → 2 carried forward → 1 recommended + 1 diagnostic backup

## Landscape Summary

当前文献基本分成三条线。第一条是**轻量规划线**：DrivoR 用 per-camera visual registers 压缩多相机 token，SparseDriveV2 用 path/velocity factorization 把静态词表扩展到很强，Latent-WAM 用 compact scene tokens + 小模型拿到不错的 NAVSIM v2 / HUGSIM 结果。这条线已经证明“更轻”本身不是空白，但它们普遍缺少显式高阶 cognition。

第二条是**连续动作生成 + RL**：DiffusionDrive、DiffusionDriveV2、HAD 已经把 truncated diffusion、anchor-aware RL、hierarchical diffusion、metric-decoupled RL 推到比较前面。这里的主要风险是：如果新方案只是“扩散 planner + 多目标 RL”，会非常靠近 HAD / DiffusionDriveV2，难以讲清新意。

第三条是**高认知 VLA / world-model**：ReCogDrive、UniDriveVLA、ExploreVLA、DriveVA、AutoMoT 等工作都在增强 cognition、reasoning 或 world modeling，但普遍更重，且常常依赖在线大模型、dense world modeling 或视频生成。这与当前 brief 里的 non-goals（不走重 3D、不做在线 VLM 自回归）存在明显冲突。

综合来看，最有空间的缺口不是“再做一个更大的 cognition model”，而是：**把高阶 cognition 离线蒸馏到极少量 semantic/register bottleneck 中，再交给结构保持的 planner 与低成本多目标优化去使用。**

## Recommended Ideas (ranked)

### Idea 1: Counterfactual Semantic Registers — RECOMMENDED

- **Hypothesis**: NAVSIM 长尾失败的根因往往不是像素缺失，而是缺少“如果此刻不让行/不减速会发生什么”的 counterfactual decision variables；这些语义可以被离线蒸馏到极少量在线 registers 中。
- **Core idea**: 以 DrivoR 式 register encoder 为骨架，但不是仅压缩视觉，而是用离线 candidate trajectory + decomposed PDMS 子指标构造 counterfactual supervision，把 yield/proceed、stop-point、conflict-zone risk、traffic-light legality、progress affordability 等决策变量压进少量 semantic registers；下游用轻量 path-velocity planner 解码，而不是在线跑 VLM 或 diffusion。
- **Expected outcome**: 在不显著增加延迟的情况下，提高交叉口、行人冲突、红绿灯等长尾场景的 PDMS / EPDMS，尤其是 safety-related submetrics；同时 register 变化能对语义干预（如改灯态、移除行人）表现出一致响应。
- **Novelty**: 7.8/10 — closest work: DrivoR + ReCogDrive + CF-VLA + KnowDiffuser
- **Novelty check**: 没找到“把离线 counterfactual preference/distillation 压进 tiny online registers”这一直接近邻。最近相关工作多是在线 VLM/LLM reasoning、语言 token bridge，或更重的 semantic/world tokens，而不是极小 bottleneck + 轻量 planner。
- **Feasibility**: 中等实现复杂度；主要成本在离线 candidate 采样、metric cache、counterfactual variable 定义与 pairwise/listwise distillation。预估 5–7 周可做出完整最小闭环。
- **Risk**: MEDIUM
- **Contribution type**: method
- **Reviewer's likely objection**: “这会不会只是 DrivoR + 更好的 auxiliary supervision？你的 semantic registers 真的是语义瓶颈，而不是被重新命名的 hidden states 吗？”
- **Why we should do this**: 它最贴合当前 brief：轻量、非在线 VLM、非重 3D，同时有机会提出一个更可防守的中心命题——**tiny semantic bottleneck can carry counterfactual decision variables sufficient for long-tail planning improvement**。

### Idea 2: Path-Velocity Residual Manifold Planner — BACKUP

- **Hypothesis**: 与其用更密词表或 diffusion，不如先学一个 path-curvature / velocity-profile 的连续低维流形，再预测 latent code + 小残差；这能用更低算力达到相近的动作多样性与物理合理性。
- **Core idea**: 先从 expert trajectory 中学习 factorized manifold（可用小 VAE / PCA-style latent / codebook-free latent），在线 planner 只预测 latent code 与 structure-preserving residual；以 matched FLOPs 对比 regression、SparseDriveV2 风格大词表和小 diffusion planner。
- **Expected outcome**: 在相同 backbone 和近似算力下，取得比 dense vocabulary 更低的 memory / compute 成本，同时比 direct regression 更强的多模态表达。
- **Novelty**: 7.2/10 — closest work: SparseDriveV2 + HAD
- **Feasibility**: 很高；实现路径清晰，最适合 3 个月内快速出一条稳妥主线或 fallback。预估 3–5 周得到强 baseline。
- **Risk**: LOW
- **Contribution type**: method
- **Reviewer's likely objection**: “这会不会只是另一种 action representation trick？”
- **Why we should do this**: 它最安全，也最容易做出 clean ablation；即便最终不如主方向，也能形成很强的备选方案。

### Idea 3: Offline Hard-Negative Ranking Instead of RL — DIAGNOSTIC BACKUP

- **Hypothesis**: 许多 RL 增益其实来自“看见了有信息量的负样本”，而不一定来自真正的 policy optimization；如果成立，可以用离线 hard-negative ranking 替代高成本 RL。
- **Core idea**: 对每个 expert 样本生成局部 path/speed perturbations，用缓存的 decomposed metrics 打分，再加入 pairwise/listwise ranking loss，直接测试 “offline negatives 是否能替代 RL”。
- **Expected outcome**: 在接近零在线仿真成本下，拿到接近 RL 后处理的 safety / comfort 提升，并把“RL 是否必要”变成明确结论。
- **Novelty**: 5.8/10 — closest work: DiffusionDriveV2 + HAD + scoring-based planners
- **Novelty check**: 没找到标题层面非常接近的 arXiv 直接近邻，但它在机制上明显接近 scorer / ranking recipe，更像“field assumption test”而不是全新方法。
- **Feasibility**: 很高；2–3 周可出结果。
- **Risk**: LOW
- **Contribution type**: diagnostic / empirical finding / training ingredient
- **Reviewer's likely objection**: “这更像训练技巧或 ablation，不像完整方法论文；即便替代了 RL，也不能严格证明 RL 的增益主要来自 negative exposure。”
- **Why we should do this**: 它非常适合作为主 idea 的配套诊断，也可以在主线受阻时快速产出有价值的正/负结论；但不应再被当作首要主 paper 方向。

## Validation Update

- **External reviewer verdict**: Idea 1 明显比 Idea 3 更像 NeurIPS 级主线，但必须把“counterfactual preference distillation”收紧成**可检验的 semantic bottleneck claim**，否则会退化成“register + better supervision”。
- **What makes Idea 1 defensible**: 明确定义一小组 counterfactual decision variables；架构上强制 planner 只能通过 semantic registers 使用这些变量；加入 semantic intervention test，而不只报 benchmark 分数。
- **What makes Idea 3 useful**: 最好被 framing 成 diagnostic study + training recipe，而不是 standalone flagship method。

## Lower-Priority Ideas


| Idea                               | Why not ranked higher                          |
| ---------------------------------- | ---------------------------------------------- |
| Prototype Counterfactual Memory    | 方向有趣，但 retrieval/memory 设计容易把项目拉向系统工程，主贡献会被冲淡。 |
| Reward-Simplex Conditional Planner | 诊断价值高，但容易被 reviewer 视为 HAD/MDPO 的轻量变体。         |
| Metric-Aligned Register Slots      | 更适合做主方向的辅助模块或可解释性分析，不够像独立主 paper。              |


## Eliminated Ideas (for reference)


| Idea                                         | Reason eliminated                            |
| -------------------------------------------- | -------------------------------------------- |
| Ambiguity-Triggered Dual Policy              | 有效但偏小修小补，容易落成 small trick。                   |
| Temporal Saliency Registers                  | 更像 encoder 细化点，paper thesis 不够强。             |
| Invariant Semantic Intervention Distillation | 新颖但 annotation / intervention 设计太重，3 个月风险偏高。 |
| Uncertainty-Gated Safe Residual              | 工程上合理，但 novelty 较弱，像 safety patch。           |


## Suggested Execution Order

1. Start with **Counterfactual Semantic Registers** as the main idea.
2. Keep **Path-Velocity Residual Manifold Planner** as the main backup.
3. Use **Offline Hard-Negative Ranking** as the cheapest diagnostic branch and possible auxiliary training ingredient.

## Next Steps

- If continuing the pipeline, run `/research-refine` on **Counterfactual Semantic Registers**
- Convert Idea 3 into a diagnostic module/section rather than a standalone main thesis
- Keep Idea 2 as fallback if Idea 1 weakens during refinement

