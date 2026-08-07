# 开发交接约定

## 当前主交接文档

- [2026-08-07 阶段汇报：当前设计、实验和后续消融](REPORT_BRIEF_20260807.md)
- [2026-08-07 三数据集混合 Jury 实验](MIXED_JURY_EXPERIMENT_20260807.md)
- [当前 M3 工作流](M3_WORKFLOW.md)
- [总开发计划](DEVELOPMENT_PLAN.md)

阶段性实验长文和一次性诊断脚本只作为本机历史材料保留，不再作为主交接入口。真实运行的
raw artifacts、SQLite 和 manifests 保留在被 Git 忽略的 `runs/` / `artifacts/` 下。

## 当前可交接状态

- 正式 Target：`glm-4.6v-target-v2`，初始输出预算 4096 tokens，仅在空响应、纯 reasoning
  或截断时以 8192 tokens 兜底。
- Blind Grounding：`glm-4.6v-grounding-v2`，2048 tokens 起步，允许一次 4096-token
  合约修复；无文字图片必须返回可观察的场景描述，不能返回空 observations。
- 异构 Jury：Seed 2.0 Mini 负责 intent、DeepSeek V4 Flash 负责 compliance、Kimi K2.6
  负责 harm enablement、Grok 4.3 负责 oversensitivity、GLM 5.2 负责 arbitration。
- 三数据集 9 条历史实验最终产生 8 条结果和 1 条 batch failure；L0 1 条、L2 2 条、
  `review_required` 5 条。它是修复前基线，不应改写为 v2 整体指标。
- v2 使用同一个 MM-SafetyBench 历史失败样本回归：Target 从 3 次调用降为 1 次；
  Grounding 从 `unavailable` 变为一次调用得到 `complete`。后续 Jury 实际触发仲裁，但最终
  仍为 `review_required`，说明聚合语义还需单独验证。
- 目前仍没有 response-level 人工金标，不能报告准确率、F1 或统计显著性。

## 接手后的第一优先级

1. 使用冻结的 TargetResponse 对 9 条混合集重跑 Grounding/Jury，隔离 v2 Grounding 的整体
   收益，不重复支付 Target 成本。
2. 建立 4–6 条人工复核的退出集合，至少包含 2 条 L1、2 条 Compliance/Enablement 冲突和
   1 条预期仲裁。
3. 核验仲裁后 `review_required` 的确定性下限语义；不要通过放宽 fail-closed 规则掩盖冲突。

任何交接任务必须让接手者在不了解全部架构的情况下独立验证结果。

## 任务卡模板

```text
目标：
允许修改的目录：
禁止修改的契约：
输入 Fixture：
预期输出：
运行命令：
正式验收命令与预期结果：
已知问题：
```

## 交接前检查

1. 工作区不包含 API Key、原始有害媒体或本机绝对路径。
2. 涉及正式链路的变更必须通过 `scripts/run_e2e_acceptance.py`；静态检查必须通过。
3. 数据字段变更注明 Schema 版本，提示词变更注明 Prompt 版本。
4. 说明是否发生模型调用、样本量、模型名称、Token 和实际费用。
5. 提供未完成项和下一步，不用口头信息替代文档。

## Git 提交边界

应提交：`src/safejudge/` 核心实现、`config/` 版本化配置、
`scripts/run_e2e_acceptance.py`、README 和主交接/实验摘要文档。

不提交：`data/`、`runs/`、`artifacts/`、SQLite/checkpoint、provider raw payload、覆盖率文件、
本机模型、下载缓存，以及为单轮下载、抽样、汇总或诊断编写的一次性脚本。提交前使用
显式路径执行 `git add`，不要在混合工作区使用 `git add -A`。

## 受保护的架构契约

未经核心开发者确认，不修改以下内容：

- 统一多模态数据 Schema 的已有字段语义。
- 回答合规程度 `0/1/2` 的方向和判定边界；禁止改成问题风险标签。
- `TARGET` 被评测模型与 `JUDGE` 裁判模型的角色隔离。
- 主 Agent 到子 Agent 的单向、白名单通信规则。
- 子 Agent 无工具权限、无长期记忆写权限的隔离规则。
- 聚合结果、调用成本和来源追踪字段。
