# GTRS 子项目入口

## 项目默认上下文

- 研究类任务默认先阅读 `RESEARCH_BRIEF.md`，并将其视为问题定义、约束、non-goals 与已有尝试的主来源。
- 当前项目主要围绕 NAVSIM v1/v2 上的端到端自动驾驶研究展开。
- 本入口只保留稳定、长期有效的协作约束；动态研究进展、临时结论、实验状态不要写在这里。

## 协作与输出偏好

- 与用户沟通、总结和新增项目级说明优先使用中文。
- 输出保持简洁直接，避免重复用户已经明确写出的背景。
- 本入口保持短小稳定；动态内容分别写入 `RESEARCH_BRIEF.md`、`IDEA_REPORT.md`、`refine-logs/`、实验记录系统或 `research-wiki/`。

## 研究工作流偏好

- 先固定问题锚点、约束与 non-goals，再扩展到文献、idea、review 与 refine。
- 当前阶段的 `idea-discovery` 只负责文献调研、idea 生成、novelty 检查、批判性评审，以及可选的方法细化。
- 不要在 idea 阶段擅自引入 pilot 实验、实验执行或长跑任务逻辑；后续实验规划与执行交给专门的 skills。
- 优先收敛到 1 个主 idea + 1 个备选 idea，避免同时推进过多并行方向。

## 环境与实验约束

- 默认使用 `conda activate dinov3`；不要原地修改该环境。
- 涉及实验代码、脚本或长任务时，优先使用 git worktree 隔离工作。
- 实验产物根目录遵循 `NAVSIM_EXP_ROOT` 约定。
- 有意义的实验必须写入实验记录系统，并保留可读的结果摘要层。
- 任何涉及联网的部分，使用 `web access` skill，而不是内置 web search。
- 阅读任何 PDF 文件时，只用 `pdf` skill。
