# MultiAgent-SafeJudge

面向多模态模型回答的可复现安全评测框架。公开 Benchmark 先转换为统一数据协议，由
被测模型生成并冻结回答，再交给主协调流程进行 blind grounding、意图识别、并行分轴
裁判、确定性聚合与按需仲裁。默认正式配置由 DeepSeek V4 Flash 完整执行 Intent、31 小类路由、
Compliance、Enablement、Oversensitivity 和仲裁；Llama Guard 作为额外的逐小类安全
子 agent，独立记录命中的国标小类和 Constitution 规则，不参与投票或覆盖主聚合。
媒体与请求理解在上游完成；正式 DeepSeek + Llama Guard 配置中的下游子裁判仅接收版本化的
紧凑事实包、完整待评回答和对应 Constitution，不再接收原请求的全部长上下文。
当前正式实验只使用本机已有的 JailBreakV-28K testbench；另外三个 Adapter 仅保留作历史兼容，
不要求为本阶段下载额外数据集。

历史 M3 链路曾跑通 20 条真实 API 异构 Jury，并完成一次来自 MM-SafetyBench、MOSSBench
和 Omni-SafetyBench 的 9 条混合实验。混合实验暴露的 Target 重试放大和 Grounding 空证据
问题已经用 v2 profile 修复，并在同一历史失败样本上完成真实回归。历史设计和已测结果见
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
TargetResponse → blind 模型 Grounding → DeepSeek 完整主 Judge → Llama Guard 逐小类子 agent →
DeepSeek 结果聚合/仲裁 → EvaluationResult 与验收报告。

```powershell
python scripts/run_e2e_acceptance.py `
  --input data/formal/jailbreakv-balanced-5.jsonl `
  --media-root D:\JailBreakV_28K\JailBreakV_28K `
  --target-profile glm-4.6v-target-v2 `
  --grounding-profile glm-4.6v-grounding-v2 `
  --jury-plan config/juries/m3-deepseek-llamaguard-remote-v1.toml `
  --output-dir runs/formal-e2e `
  --limit 5 `
  --allow-unqualified-model
```

脚本只调用注册表中的 OpenRouter 或本地 OpenAI-compatible 真实模型，没有模拟 Provider、
dry-run 或静默降级。任一 Target、Grounding、Judge 失败，结果缺失或未 resolved 都以非零状态退出。
命令会产生真实调用费用，运行前应核对 profile、样本量和账户预算。

默认护栏计划通过 OpenRouter 调用 `meta-llama/llama-guard-4-12b`，不下载权重、不租用 GPU
服务器；另提供 DeepSeek 远程 + Llama Guard 本地 OpenAI-compatible 服务的混合配置。
Llama Guard 的原生 `safe/unsafe + S1..S14` 输出由版本化映射转换为 DeepSeek 已路由的国标小类；
不使用概率、置信度、Reason Code 或 RTG。详情见
[安全护栏模型接入](docs/GUARDRAIL_MODEL.md)。原来的纯 Kimi 计划
`config/juries/m3-single-judge-v1.toml` 和 Kimi + Llama Guard 计划继续保留，用于历史复现。
第一次跑全远程实验可直接使用 `scripts/run_remote_experiment.ps1`。

模型接口见 [M2 模型层](docs/MODEL_LAYER.md)，数据转换见
[数据集接入说明](docs/DATASETS.md)。

国标风险分类使用独立、版本化的 Taxonomy 配置。标准编号、版本、正式分类名称、条款定位、
官方来源及其与 Constitution Pack 的映射规范见
[标准来源与溯源约定](docs/standards/README.md)。仓库现已录入 GB/T 45654-2025 附录 A 的
5 个父类和 31 个叶子小类，并为每个叶子小类启用 Constitution 映射和多标签 Category Router；
该 Taxonomy 路由在评估命令和正式远程实验中默认开启，可用 `--no-taxonomy` 显式关闭；
国标正文因再分发许可不明确不提交，仅保留官方来源、条款定位和版本化映射。

## 仓库边界

远端仓库只保存可维护的源码、版本化配置、正式端到端验收入口和结果摘要。Benchmark 原始
媒体、转换后的 JSONL、模型原始响应、SQLite、checkpoint、运行日志和一次性诊断脚本均不
提交。接手者应按数据集说明在本地准备 Canonical JSONL；`data/` 和 `runs/` 中的路径只是
运行约定，不代表仓库内置实验数据。
