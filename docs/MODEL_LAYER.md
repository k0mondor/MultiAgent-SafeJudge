# M2 模型适配层

模型层只保留两个真实 Provider：OpenRouter 与本地 OpenAI-compatible 服务。调用路径为：

```text
CanonicalMultimodalSample
  -> TargetRunner（精确模态能力检查）
  -> ModelInvoker（缓存、预算、有限重试、调用账本）
  -> registered real ModelProvider
  -> TargetResponse 1.1（冻结后供裁判只读）
```

## 契约

- `TARGET` 生成待评回答，`JUDGE` 和 `GROUNDING` 执行各自固定角色。
- 所有入口必须显式选择 `config/models.toml` 中的 profile；不从模型名猜测能力。
- `TargetResponse` 保存响应 ID、模型修订版、Token、延迟、价格快照、费用和缓存状态。
- 原始 Provider 响应先按内容寻址写入 Artifact；工作流只引用 URI、SHA-256、类型和大小。
- 请求、媒体与 Target 文本均是不可信数据，不能提升为系统指令。

## Provider

OpenRouter 使用非流式 `/chat/completions`。图片、音频和视频分别编码为服务支持的多模态
内容块；本地媒体必须位于可信 media root 内，并通过文件大小限制。

`LocalOpenAIProvider` 用同一协议接入 vLLM、LM Studio、Ollama 等兼容服务。服务必须返回
标准响应 ID 和可解析文本；协议缺失直接失败，不再生成伪响应 ID。

两类 Provider 共用媒体编码、响应解析、Token 与 HTTP 状态映射。认证、余额、非法请求和
内容策略错误立即失败；仅限流、超时和上游暂时不可用执行有界重试。部分真实模型曾出现
reasoning-only 或截断，因此 profile 可以配置少量 `retry_parameters`；它不是跨 Provider
切换，也不会绕过内容策略。

## 注册表与配置

每个模型 profile 只处于 `candidate` 或 `approved`。正式运行默认拒绝 candidate；开发期
必须显式加 `--allow-unqualified-model`。profile 固定 Provider、模型 ID、角色、结构化输出
方式、调用参数与有限重试参数。

环境配置放在未跟踪的 `.env`：

```dotenv
OPENROUTER_API_KEY=
LOCAL_MODEL_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_MODEL_API_KEY=
```

## 缓存、账本与预算

`SQLiteModelStore` 保存稳定请求缓存、逻辑调用账本和每次真实 Provider attempt。缓存命中
不新增费用；预算在未缓存调用前检查。不同请求按信号量并行，相同缓存键通过 singleflight
只调用一次。Batch 保留样本级错误记录，正式验收脚本会在任何失败上整体返回失败。

模型调用层支持显式 `--cache-only`：缓存缺失在进入预算和 Provider 边界前抛出
`CacheMissError`。正式验收脚本的 `--replay` 会冻结并校验已有 TargetResponse 文件，再以该模式
执行 Grounding、Judge 和 Guardrail，因此重放不会产生新 Provider attempt 或费用。

## 验收入口

不再维护模拟 Provider 和旧 Schema 的离线测试入口。唯一正式检验是
[`scripts/run_e2e_acceptance.py`](../scripts/run_e2e_acceptance.py)，覆盖数据读取、媒体校验、
真实 Target、blind Grounding、单一 Judge 分轴裁判、聚合、结果和 manifests。命令见项目 README。
