# vLLM 文本裁判 Pilot

SafeJudge 把 vLLM 当作独立的 OpenAI-compatible 推理服务。LangGraph 同时发出三个子裁判
请求，`ModelInvoker.max_concurrency` 控制客户端并发，vLLM 负责服务端连续批处理、KV Cache
和 GPU 调度。所有子裁判共享一个模型实例，不会为每个 Agent 重复加载权重。

## 本机建议形态

当前 RTX 4060 Laptop 只有 8 GB 显存，因此采用分阶段运行：

1. 使用 Qwen2.5-Omni Target 生成并固化 `TargetResponse` JSONL。
2. 结束 Target 进程并释放显存。
3. 在 WSL2 Ubuntu 中启动 1.5B/3B 文本 Judge 的 vLLM 服务。
4. Windows SafeJudge 使用 `evaluate run-jsonl` 对固化回答评分。

不要同时在 8 GB 显存中常驻 Omni Target 和 Judge。vLLM 0.11.1 的 CUDA 12.6 wheel 与当前
驱动更接近；升级 vLLM 前必须重新核对 wheel CUDA 版本与 NVIDIA 驱动兼容性。

## WSL 环境

```bash
python3 -m pip install --user --upgrade uv

~/.local/bin/uv venv ~/.venvs/safejudge-vllm \
  --python 3.12 --seed --managed-python

~/.local/bin/uv pip install \
  --python ~/.venvs/safejudge-vllm/bin/python \
  'vllm==0.11.1' --torch-backend=cu126
```

从仓库根目录启动服务：

```bash
bash scripts/start_vllm_judge.sh
```

脚本默认使用 `Qwen/Qwen2.5-1.5B-Instruct`、端口 `8000`、服务模型名
`safejudge-judge`、`max-num-seqs=4` 和 82% 显存上限。首次启动会下载模型。可以覆盖：

```bash
SAFEJUDGE_VLLM_MODEL=/path/to/model \
SAFEJUDGE_VLLM_PORT=8000 \
bash scripts/start_vllm_judge.sh
```

若 WSL 无法直连 Hugging Face，可以先从 Qwen 官方 ModelScope 仓库在 Windows 下载，
再把 `SAFEJUDGE_VLLM_MODEL` 指向 `/mnt/c/models/...`。不要直接使用项目的
`Qwen2.5-Omni-7B-NF4-Text`：该目录是 Transformers 文本验收产物，不是完整的 vLLM
Omni 服务快照。

## SafeJudge 配置

```dotenv
JUDGE_MODEL_BASE_URL=http://127.0.0.1:8000/v1
JUDGE_MODEL_ID=safejudge-judge
JUDGE_MODEL_API_KEY=local-safejudge
JUDGE_MODEL_TIMEOUT_SECONDS=120
```

先检查服务：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/v1/models `
  -Headers @{ Authorization = "Bearer local-safejudge" }
```

然后对一条已经固化的回答运行 M3 Pilot：

```powershell
safejudge evaluate run-jsonl `
  --provider local `
  --input data/acceptance/canonical.jsonl `
  --target-responses runs/target/target-responses.jsonl `
  --output runs/vllm-pilot/evaluations.jsonl `
  --store runs/vllm-pilot/judge-calls.sqlite3 `
  --checkpoint runs/vllm-pilot/checkpoints.sqlite3 `
  --node-ledger runs/vllm-pilot/node-ledger.sqlite3 `
  --artifact-root artifacts/vllm-pilot `
  --judge-env-file runs/vllm-pilot/judge.env `
  --experiment-id vllm-judge-pilot `
  --run-id qwen-1p5b-pass-1 `
  --max-concurrency 3 `
  --max-sample-concurrency 1 `
  --limit 1
```

相同 `run-id` 和评测指纹再次执行时，会读取已完成 checkpoint；模型、Prompt、Rubric、
聚合策略或参数变化都会产生新的 `evaluation_key`，不会错误复用旧结论。

本地 Judge 调用会把 Pydantic 契约转换为 vLLM `response_format=json_schema`。证据文本
不是自由生成字段，而是从目标回答的连续原文候选中选择；随后运行时再次校验来源、
补齐 SHA-256 与 `start/end`。Schema 保证结构，代码保证证据可回查，Prompt 只承担语义。

## Pilot 验收

- `/v1/models` 返回 `safejudge-judge`。
- 三个 panel 调用可以同时在 vLLM 日志中出现。
- `evaluations.jsonl` 能通过 `EvaluationResult 1.2` 校验。
- 每条证据都包含来源 SHA-256 与可回查的 `start/end`。
- `node-ledger.sqlite3` 记录每个业务节点的状态、尝试次数、内容哈希、耗时与模型调用 ID。
- Manifest 记录模型、缓存命中、调用费用、仲裁数量和最终等级分布。
- 记录冷启动时间、峰值显存、每条平均延迟和第二次运行缓存情况。

## 2026-08-05 本机结果

- 环境：WSL2 Ubuntu、vLLM 0.11.1、PyTorch 2.9.0+cu126、RTX 4060 Laptop 8 GB。
- 模型：`Qwen2.5-1.5B-Instruct`，由本地 Windows 路径加载。
- 权重显存约 2.887 GiB，KV Cache 可用约 3.51 GiB；服务成功返回 `/v1/models`。
- 一条 MM-SafetyBench 固化回答完成 4 次裁判调用，三个 panel 确实同时进入 vLLM；
  运行产物写入 `runs/vllm-pilot/`。
- 同一 `run-id + evaluation_key` 重跑约 2.5 秒完成，模型调用数与节点账本行数均未增加。

1.5B 模型只通过了工程链路验收，不代表裁判准确率验收。该样本中出现了理由与标签
语义不一致、过敏感性误判等现象；进入正式实验前仍需 3B/7B 候选模型与人工金标准集
做准确率、稳定性和校准对比。
