# MultiAgent-SafeJudge

面向多模态模型回答的可复现安全评测框架。公开 Benchmark 先转换为统一数据协议，由
被测模型生成并冻结回答，再交给主协调流程进行 blind grounding、意图识别、三席位并行
裁判、确定性聚合与按需仲裁。

M3 核心链路已经跑通 20 条真实 API 异构 Jury，并完成一次来自 MM-SafetyBench、MOSSBench
和 Omni-SafetyBench 的 9 条混合实验。混合实验暴露的 Target 重试放大和 Grounding 空证据
问题已经用 v2 profile 修复，并在同一历史失败样本上完成真实回归。工程实现可以收尾，但
研究验收仍缺一组人工复核的 L1、冲突与仲裁定向实验。当前设计、已测结果和消融计划见
[阶段汇报](docs/REPORT_BRIEF_20260807.md)，工作流细节见
[M3 文档](docs/M3_WORKFLOW.md)，三数据集实验的逐样本解释见
[混合 Jury 实验报告](docs/MIXED_JURY_EXPERIMENT_20260807.md)。

## 本地检查

项目要求 Python 3.12+，推荐使用 `uv`：

```powershell
uv venv --python 3.12
uv sync --extra dev --extra orchestration
uv run safejudge doctor
uv run ruff check .
uv run mypy
```

`doctor` 只检查本地配置，不访问模型 API。复制 `.env.example` 为 `.env` 后填写真实服务
配置；不要提交密钥。

## 正式端到端验收

仓库只保留一个正式验收脚本。它固定执行：Canonical JSONL → 真实 Target → frozen
TargetResponse → blind 模型 Grounding → 主协调流程 → 异构 Jury 子裁判 → 聚合/仲裁 →
EvaluationResult 与验收报告。

```powershell
python scripts/run_e2e_acceptance.py `
  --input data/formal/m3-exit20-v1.jsonl `
  --media-root . `
  --target-profile glm-4.6v-target-v2 `
  --grounding-profile glm-4.6v-grounding-v2 `
  --jury-plan config/juries/m3-heterogeneous-v1.toml `
  --output-dir runs/formal-e2e `
  --limit 1 `
  --allow-unqualified-model
```

脚本只调用注册表中的 OpenRouter 或本地 OpenAI-compatible 真实模型，没有模拟 Provider、
dry-run 或静默降级。任一 Target、Grounding、Judge 失败，结果缺失或未 resolved 都以非零状态退出。
命令会产生真实调用费用，运行前应核对 profile、样本量和账户预算。

模型接口见 [M2 模型层](docs/MODEL_LAYER.md)，数据转换见
[数据集接入说明](docs/DATASETS.md)。

## 仓库边界

远端仓库只保存可维护的源码、版本化配置、正式端到端验收入口和结果摘要。Benchmark 原始
媒体、转换后的 JSONL、模型原始响应、SQLite、checkpoint、运行日志和一次性诊断脚本均不
提交。接手者应按数据集说明在本地准备 Canonical JSONL；`data/` 和 `runs/` 中的路径只是
运行约定，不代表仓库内置实验数据。
