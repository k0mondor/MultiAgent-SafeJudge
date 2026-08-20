# 安全护栏模型接入

## 当前正式配置

默认正式 Jury 计划为 `config/juries/m3-deepseek-llamaguard-remote-v1.toml`：

- DeepSeek V4 Flash：完整承担 Intent、31 小类路由、Compliance、Harm Enablement、
  Oversensitivity 和受限仲裁；
- Llama Guard 4 12B：在 DeepSeek 路由后，对每个国标叶子小类独立执行安全子 agent 判断；
- 确定性代码：把 Llama Guard 的原生 `S1..S14` 类目与已路由的 GB/T 45654-2025
  叶子小类交叉校验，并单独保存 `category_id`、`constitution_id` 和
  `triggered_rule_ids`。

Llama Guard 结果保存在 `category_results[].guardrail_verdict`，失败保存在
`category_results[].guardrail_failure`。它不替换 DeepSeek 的 Compliance，不参与异构投票，
也不改变 DeepSeek 的聚合结果；正式 Target 对比实验中不应在看到结果后更换该配置。

## 上下文隔离

两份 DeepSeek + Llama Guard 配置都固定使用 `subjudge_context_mode = "compact"`。信息边界如下：

- 上游 DeepSeek Intent 阶段读取原请求和 Grounding，完成媒体理解与请求理解；
- 上游 DeepSeek Category Router 读取原请求、完整 TargetResponse 和 31 小类 Taxonomy，执行多标签路由；
- 下游 DeepSeek Compliance/Enablement/Oversensitivity 子裁判不再读取原请求全文，只接收请求哈希、
  `request_intent`、`scope_status`、`requested_action`、`intent_basis`、受控 Grounding 观察、
  完整 TargetResponse 和当前小类对应的 Constitution；
- Llama Guard 使用同一紧凑事实包作为 user 消息，并把完整 TargetResponse 作为 assistant 消息。

因此压缩的是原问题的长上下文，不是待评回答，也不会丢掉 31 小类路由或 Constitution
归因。`request_snapshot_sha256` 只用于溯源，子裁判不得据此重建原请求。旧的纯 Kimi 配置未声明
该字段，继续按 `full` 运行以复现历史结果；切换上下文模式会进入 Jury fingerprint，属于新的
实验条件。

## 远程调用

当前模型 ID：

```text
meta-llama/llama-guard-4-12b
```

1. 在 <https://openrouter.ai/keys> 创建 API Key。
2. 从 `.env.example` 复制生成 `.env`。
3. 只填写：

```dotenv
OPENROUTER_API_KEY=sk-or-v1-你的密钥
```

4. 先以 `--limit 1 --allow-unqualified-model` 运行正式端到端脚本。真实请求会计费；
   原始响应保存在 Guardrail Artifact 目录，密钥不会进入 Manifest。

模型 Profile 是 candidate，完成真实 API 冒烟并检查原始输出前不要改成 approved。API 或输出
协议异常会形成可审计的 `guardrail_failure`；DeepSeek 主 Judge 的完整结果仍会照常保存。

## 离散输出和映射边界

Adapter 只接受官方原生协议：

```text
safe
```

或：

```text
unsafe
S7
```

不读取 logits，不生成连续风险分数，也不把解析失败伪装成“疑似”。命中时保存对应
Constitution 及真正触发的规则，未命中时 `triggered_rule_ids` 为空。`S1..S14` 与国标小类
不是等价分类体系；版本化交叉映射位于
`config/guardrails/llama-guard-4-gbt45654-v1.toml`，只用于验证 SafeJudge Router 已选中的
小类。修改映射会改变 Jury fingerprint，必须作为新的实验条件记录。

## 本地/远程组合

仓库提供两份配置：

- `m3-deepseek-llamaguard-remote-v1.toml`：DeepSeek 和 Llama Guard 都通过 OpenRouter 远程调用；
- `m3-deepseek-llamaguard-local-v1.toml`：DeepSeek 仍通过 OpenRouter，Llama Guard 连接本地或租用
  服务器上的 OpenAI-compatible 服务。

第二种配置仍不是完全离线，因为 DeepSeek 主 Judge 是远程模型。若 Llama Guard 服务运行在其他
GPU 服务器，把 `.env` 中的 `JUDGE_MODEL_BASE_URL` 改成该服务器的 `/v1` 地址即可。

## 可选本地部署

远程方案无需下载。如果以后使用大显存服务器自部署，官方权重和许可证入口为：

- <https://huggingface.co/meta-llama/Llama-Guard-4-12B>
- <https://www.llama.com/llama-downloads/>

该模型为 12B BF16，多模态本地加载不适合当前 8GB 笔记本显存。自部署可以用 vLLM 暴露
OpenAI-compatible Chat API，然后选择：

```text
config/juries/m3-deepseek-llamaguard-local-v1.toml
```

本地服务地址写入 `.env`：

```dotenv
JUDGE_MODEL_BASE_URL=http://127.0.0.1:8000/v1
JUDGE_MODEL_API_KEY=local-safejudge
```

不要把模型权重或真实密钥提交到 Git。
