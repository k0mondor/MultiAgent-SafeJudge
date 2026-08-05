# MultiAgent-SafeJudge

面向多模态大模型的可复现安全评测框架。项目使用统一数据协议将公开 Benchmark
提交给被评测模型生成回答，再通过主从式 Multi-Agent 裁判工作流评估回答的
合规程度与过敏感性。

项目当前已完成 M2 模型适配层和 M3 LangGraph Multi-Agent 核心图，并使用 25 条真实
Benchmark 样本打通 M0–M2 离线桥接。M3 已具备三裁判并行、输入隔离、按需仲裁和
SQLite Checkpoint 失败恢复。开发路线、阶段验收标准以及协作方式见
[开发计划](docs/DEVELOPMENT_PLAN.md)。需要多人接力时遵循
[开发交接约定](docs/HANDOFF.md)。

首批公开 Benchmark、模态组合和转换命令见[数据集接入说明](docs/DATASETS.md)。
TARGET/JUDGE 协议、Fake/OpenRouter、缓存和费用策略见
[M2 模型适配层](docs/MODEL_LAYER.md)。
M3 图拓扑、prompt 扩写边界、本地并发和恢复方法见
[M3 Multi-Agent 核心图](docs/M3_WORKFLOW.md)。
本地并行裁判服务的安装、启动和 Pilot 命令见
[vLLM 文本裁判 Pilot](docs/VLLM_JUDGE.md)。

## 本地开发

项目要求 Python 3.12+，开发环境由 `.python-version` 固定为 Python 3.12。
推荐使用 `uv`：

```powershell
uv venv --python 3.12
uv sync --extra dev --extra orchestration
uv run safejudge doctor
uv run ruff check .
uv run mypy
uv run pytest
```

`doctor` 只检查本地配置，不访问网络或模型 API。复制 `.env.example` 为 `.env`
后可以覆盖开发配置；不要提交真实密钥。

## 本地被测模型

LM Studio、Ollama、vLLM 等本地推理服务可通过 OpenAI-compatible
`/v1/chat/completions` 接入。先在项目根目录的 `.env` 配置
`LOCAL_MODEL_BASE_URL` 和 `LOCAL_MODEL_ID`，再使用
`safejudge target run-jsonl --provider local`。首次运行建议添加 `--limit 1`；完整命令、
模态能力声明和输出位置见[M2 模型适配层](docs/MODEL_LAYER.md#本地-openai-compatible-模型)。

OpenRouter 使用同一批处理入口，将 Provider 改为 `--provider openrouter`，并在 `.env` 中
配置 `OPENROUTER_API_KEY` 和 `OPENROUTER_MODEL_ID`。

仓库还提供 Qwen2.5-Omni 的 NF4 量化和独立文本生成脚本。它们用于先验证本地权重，
不替代上面的 OpenAI-compatible 推理服务：

```powershell
python scripts/quantize_qwen_omni_nf4.py `
  --source C:\models\Qwen2.5-Omni-7B `
  --output C:\models\Qwen2.5-Omni-7B-NF4-Text

python scripts/test_qwen_omni_generate.py `
  --model C:\models\Qwen2.5-Omni-7B-NF4-Text `
  --device cuda `
  --prompt "请用一句话回答：你是谁？"
```

量化脚本保留原模型，使用 bitsandbytes NF4 双重量化，并关闭语音输出模块以降低体积；
图片、音频和视频输入所需的 Thinker 权重仍保留。脚本需要 PyTorch、Transformers、
Accelerate 和 bitsandbytes。当前本地验收产物为 6.25 GB，已从落盘目录重新加载并成功生成中文回答。
在 RTX 4060 Laptop 8 GB 上显式使用 CUDA 12.6 验收时，模型加载约 11.9 秒，短回答约
3.33 token/s，峰值分配显存约 6,240 MiB。`--device cuda` 会在 CUDA 不可用时直接报错，
不会静默退回 CPU。

已下载数据集的最小多模态矩阵可用同一个模型进程依次测试：

```powershell
& ".\.venv-omni\Scripts\python.exe" `
  "scripts\run_local_acceptance_matrix.py" `
  --model "C:\models\Qwen2.5-Omni-7B-NF4-Text" `
  --device cuda `
  --samples-per-group 1 `
  --max-new-tokens 48
```

脚本从 MM-SafetyBench、MOSSBench 和 Omni-SafetyBench 的 image/audio/video 子集各取
一条真实样本，复核媒体大小与 SHA-256，输出 `TargetResponse 1.1`、批次 Manifest、
SQLite 调用账本和内容寻址的原始响应 Artifact。
