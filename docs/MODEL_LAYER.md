# M2 模型适配层

M2 在数据协议和 LangGraph 之间提供稳定、异步且与厂商无关的模型边界。核心调用顺序为：

```text
CanonicalMultimodalSample
  -> TargetRunner（精确模态能力检查）
  -> ModelInvoker（缓存、预算、调用账本）
  -> ModelProvider（Fake、OpenRouter 或本地 OpenAI-compatible）
  -> TargetResponse 1.1（供后续裁判只读）
```

## 契约与信任边界

- `ModelRole.TARGET` 生成待评回答；`ModelRole.JUDGE` 只评价已固化回答。
- 能力声明列出精确组合，例如 `text+image`，不会从单独模态推断组合能力。
- 稳定哈希包含角色、Provider、模型、内容顺序和参数，不包含运行期请求 ID。
- `TargetResponse` 保存 Provider 响应 ID、模型修订版、Token、延迟、价格快照、费用和缓存状态。
- 每次调用必须携带 `experiment_id` 和 `run_id`；预算再按 TARGET/JUDGE 角色隔离。
- 请求问题、媒体和 TARGET 回答始终是不可信数据，不能提升为系统指令。

## Fake Provider

`FakeProvider` 完全离线，Fixture 可配置：固定回答、拒答、违规回答、超时和限流。
`failures_before_success` 可稳定模拟瞬态错误恢复。Fixture 由
`ModelRequest.request_id` 选择，Target Runner 的键为 `target:{sample_id}`。

## OpenRouter

复制 `.env.example` 为未跟踪的 `.env`，填写 `OPENROUTER_MODEL_ID` 和
`OPENROUTER_API_KEY`。专用设置对象排除了 ambient process environment，因此 API Key
不会意外从系统变量加载。模型能力仍需按所选模型显式配置，不能仅凭模型名称猜测。

当前使用非流式 `/chat/completions`：图片为 `image_url`，音频为 `input_audio`，视频为
`video_url`。本地媒体必须是 media root 内的相对路径，并受文件大小上限保护。
`model/messages/stream` 不能被调用参数覆盖。Provider 每次只请求一次；限流、超时和
上游不可用由 ModelInvoker 有限重试，认证、余额、非法请求和内容策略错误立即失败。

每个 HTTP 响应在解析前由 `ArtifactStore` 按原始字节保存。工作流只携带 URI、SHA-256、
内容类型和大小；完整原始回答不进入 LangGraph State。

OpenRouter 也已接入统一批处理 CLI。首次试跑使用一条样本和单并发：

```powershell
safejudge target run-jsonl `
  --provider openrouter `
  --input data/acceptance/omni-image-text-typo.jsonl `
  --media-root vendor/Omni-SafetyBench `
  --output runs/openrouter-pilot/omni-image.target-responses.jsonl `
  --store runs/openrouter-pilot/model-calls.sqlite3 `
  --artifact-root artifacts/openrouter-pilot `
  --experiment-id openrouter-target-pilot `
  --run-id first-image `
  --capability text+image `
  --max-concurrency 1 `
  --limit 1
```

接口格式参考 OpenRouter 官方文档：

- <https://openrouter.ai/docs/guides/overview/multimodal/overview>
- <https://openrouter.ai/docs/api/reference/errors-and-debugging>

## 本地 OpenAI-compatible 模型

`LocalOpenAIProvider` 把 LM Studio、Ollama、vLLM 或其他实现
`/v1/chat/completions` 的服务接到同一个 `ModelProvider` 协议。SafeJudge 不直接加载模型
权重；本地推理服务负责 GPU、量化和服务端批处理，SafeJudge 负责输入校验、客户端并发、
缓存、重试、调用账本以及 `TargetResponse 1.1` 规范化。

在项目根目录的 `.env` 中配置：

```dotenv
LOCAL_MODEL_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_MODEL_ID=your-local-model-id
# 多数本地服务不需要；要求 Bearer 占位值时再填写。
LOCAL_MODEL_API_KEY=
LOCAL_MODEL_TIMEOUT_SECONDS=120
LOCAL_MODEL_MAX_LOCAL_MEDIA_BYTES=104857600
```

设置对象只读取显式参数和项目 `.env`，不会从系统环境变量意外读取 Key。Provider 支持
字符串回答、文本内容块数组以及 `content=null` 时的 `refusal` 字段；完整 HTTP 响应仍会
先写入 Artifact。缺少响应 ID 的兼容服务会得到基于请求哈希生成的稳定响应 ID。

模态能力必须通过 CLI 显式声明。默认只允许 `text`；图文模型需要
`--capability text+image`。音频和视频格式虽然可以序列化，但只有服务端明确支持时才能
声明对应能力。

先启动本地模型服务，再用一条样本试跑：

```powershell
safejudge target run-jsonl `
  --provider local `
  --input data/acceptance/omni-image-text-typo.jsonl `
  --media-root vendor/Omni-SafetyBench `
  --output runs/local-pilot/omni-image.target-responses.jsonl `
  --store runs/local-pilot/model-calls.sqlite3 `
  --artifact-root artifacts/local-pilot `
  --experiment-id local-target-pilot `
  --run-id first-image `
  --capability text+image `
  --max-concurrency 1 `
  --limit 1
```

规范化结果写入 `--output`，原始服务响应写入
`--artifact-root/local-openai-responses/`，调用、Token、延迟、缓存和错误账本写入
`--store`。不同本地端点若复用同一 SQLite Store，应为模型配置使用唯一且稳定的
`LOCAL_MODEL_ID`，避免语义不同的模型共享缓存键。

### Qwen2.5-Omni 本地权重预检

`scripts/quantize_qwen_omni_nf4.py` 可把 Qwen2.5-Omni 检查点转换为 bitsandbytes NF4
双重量化检查点。转换在临时目录完成，仅成功后才移动到目标路径，不覆盖源模型；当前策略
关闭 Talker 和 Token2Wav 语音输出，保留 Thinker 的文本和多模态输入权重。
`scripts/test_qwen_omni_generate.py` 会直接从指定目录加载模型并输出一条 JSON 格式回答，
用于在接入服务前验证权重确实可加载和生成。

这两个脚本只是模型预检工具。批量测评仍通过 OpenAI-compatible 服务接入，这样远程模型、
Ollama、LM Studio、vLLM 和其他本地运行时可以共享相同的请求、规范化响应、缓存与调用账本。
生成预检支持 `--device auto|cuda|cpu`；正式验证 GPU 时应使用 `--device cuda`，使 CUDA
不可用成为显式错误。脚本同时输出加载时间、生成速度和 CUDA 峰值显存，便于决定服务端
并发上限。

### 本地 Qwen 多数据集 GPU 验收（2026-08-05）

`LocalTransformersProvider` 在专用 `.venv-omni` 环境内直接加载 NF4 Qwen，不经过 HTTP，
但仍复用 `run_target_batch` 的媒体校验、缓存、调用账本、Artifact 和 `TargetResponse 1.1`
规范化边界。`scripts/run_local_acceptance_matrix.py` 将五组验收输入的媒体路径统一到项目
根目录，使模型只加载一次并单并发依次处理：

| 数据集/子集 | 模态 | 样本 | 预处理与生成 |
| --- | --- | ---: | --- |
| MM-SafetyBench `01-Illegal_Activitiy/SD` | image+text | 1 | 通过 |
| MOSSBench `images` | image+text | 1 | 通过 |
| Omni `image-text/typo` | image+text | 1 | 通过 |
| Omni `audio-text/tts` | audio+text | 1 | 通过 |
| Omni `video-text/typo` | video+text | 1 | 通过 |

最终 Manifest 记录 5 个样本、5 个已验证媒体、image 3/audio 1/video 1，费用为 USD 0。
五行输出均通过 Pydantic `TargetResponse 1.1` 复验；输出 SHA-256 与 Manifest 一致，五个
原始响应 Artifact 的文件存在性和 SHA-256 也全部复核成功。48-token 冒烟上限使本次回答
以 `finish_reason=length` 结束，这是有意缩短验收耗时，不属于预处理或规范化失败。

Windows 上当前 `torchvision 0.28` 已移除 Qwen 工具仍调用的 `read_video()`。Provider
因此使用 PyAV 解码本地视频、按 1 FPS 且最多 16 帧采样，再送入官方 Qwen Processor；
本次视频实际解码为 3 帧，生成输入包含 `pixel_values_videos`、`video_grid_thw` 和
`video_second_per_grid`，证明不是把视频路径当作普通文本处理。

## 缓存、账本与预算

`SQLiteModelStore` 包含三张表：

- `model_cache`：按稳定哈希保存成功响应，供 TARGET 回答复用。
- `model_calls`：追加记录每次逻辑调用，包括缓存命中、尝试次数、Token、费用和失败分类。
- `model_attempts`：逐次记录每个真实 Provider 尝试及其原始响应引用。

缓存命中的本次 `billed_usd` 为零，但响应仍保留原始价格与费用快照。预算策略在未缓存
调用前计算“历史实际费用 + 单次预留”，超上限则不进入 Provider。预留值应按 Pilot 的
高分位费用设置；M4 多进程阶段再升级成数据库原子预算预留。

不同请求通过并发信号量并行执行；相同缓存键通过 singleflight 只执行一次。预算锁只在
检查与预留期间短暂持有，不覆盖模型网络等待。

## 离线验证

### 真实 Canonical JSONL 桥接

`target run-jsonl --provider fake` 在不访问网络的情况下把真实 Adapter 输出送入 Fake
TARGET。调用前会
重新限制媒体路径、检查文件存在性和大小上限，并要求 Canonical 记录中的大小与
SHA-256 和真实文件一致。成功回答写入 TargetResponse JSONL；Fake Provider 原始响应
通过内容寻址 ArtifactStore 保存，SQLite 同时记录缓存、调用和尝试账本。

```powershell
safejudge target run-jsonl `
  --input data/acceptance/omni-image-text-typo.jsonl `
  --media-root vendor/Omni-SafetyBench `
  --output runs/acceptance/pass-1/omni-image.target-responses.jsonl `
  --store runs/acceptance/model-calls.sqlite3 `
  --artifact-root artifacts/acceptance `
  --provider fake `
  --experiment-id real-data-acceptance `
  --run-id pass-1-omni-image
```

对同一输入、模型和参数使用相同 SQLite Store 再运行一次，即使更换 `run_id` 也应全部
命中 TARGET 缓存；新的逻辑调用仍写入账本，但不会产生新的 Provider attempt 或费用。

2026-08-04 已用 25 条真实样本执行两轮：第一轮 25 次缓存未命中和 25 次 Provider
attempt；第二轮 25 次全部命中缓存。最终 SQLite 有 25 条缓存、50 条调用账本、25 条
Provider attempt、0 次失败和 USD 0 费用；25 个原始响应 Artifact 的路径、大小和
SHA-256 均复核成功。

```powershell
python -m pytest
python -m ruff check .
python -m mypy
```

测试不读取 `.env`、不访问真实模型 API，也不产生费用。
