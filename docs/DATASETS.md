# 数据集接入

## 第一阶段选择

| Adapter | 研究作用 | 当前模态组合 | 许可证/限制 |
| --- | --- | --- | --- |
| `mm-safetybench` | 图像相关越狱与回答违规 | SD+text、SD_TYPO+text、TYPO+text | CC BY-NC 4.0，研究使用 |
| `mossbench` | 良性请求的过敏感/错误拒答 | image+text | CC BY-SA 4.0；可作测试集，禁止训练 |
| `omni-safetybench` | 同一安全种子的跨模态差异 | image+text、audio+text、video+text | CC BY-NC 4.0 |

仓库不重新分发这些数据集的原始媒体。使用者应从官方来源取得数据并遵守各自条款。

## 通用输出

每个 Adapter 输出一行一个 `CanonicalMultimodalSample` 的 JSONL。旁边会生成
`*.manifest.json`，包含：

- Adapter 和 Schema 版本
- 输入、输出文件名及 SHA-256
- 样本数、数据集名、模态统计、请求意图统计
- 转换时间

Manifest 不记录开发者机器的绝对路径。

## 查看 Adapter

```powershell
uv run safejudge data adapters
```

## MM-SafetyBench

官方问题文件位于 `data/processed_questions/{scenario}.json`，媒体解压到 `data/imgs`。
默认每条问题生成 SD、SD_TYPO、TYPO 三个图文样本：

```powershell
uv run safejudge data convert `
  --adapter mm-safetybench `
  --input vendor/MM-SafetyBench/data/processed_questions/01-Illegal_Activitiy.json `
  --media-root vendor/MM-SafetyBench/data/imgs `
  --output data/processed/mm-safetybench/illegal-activity.jsonl `
  --hash-media
```

开发阶段只转换 SD 变体：

```powershell
uv run safejudge data convert `
  --adapter mm-safetybench `
  --input vendor/MM-SafetyBench/data/processed_questions/01-Illegal_Activitiy.json `
  --media-root vendor/MM-SafetyBench/data/imgs `
  --output data/processed/mm-safetybench/illegal-activity-sd.jsonl `
  --variant sd
```

## MOSSBench

Adapter 接受 JSON、JSONL 和 Hugging Face ImageFolder 的 `metadata.csv`。图像字段可以是
`image`、`file_name`、`image_path` 或包含 `path` 的对象。

```powershell
uv run safejudge data convert `
  --adapter mossbench `
  --input vendor/MOSSBench/images/metadata.csv `
  --media-root vendor/MOSSBench `
  --output data/processed/mossbench/oversensitivity.jsonl `
  --hash-media
```

MOSSBench 样本统一标记为良性请求；这里不代表被评测模型回答一定合规。过敏感裁判需要
结合目标模型是否拒答、警告或明显削弱回答得出结论。

## Omni-SafetyBench

官方仓库把不同模态 JSONL 放在同一个数据仓库，但字段集合不同，整库自动加载可能发生
Schema cast 错误。本项目直接转换单个 JSONL 子集，并要求第一阶段每条记录恰好包含
`image_path`、`audio_path`、`video_path` 之一。

```powershell
uv run safejudge data convert `
  --adapter omni-safetybench `
  --input vendor/Omni-SafetyBench/meta_files/dual-modal/audio-text/tts.jsonl `
  --media-root vendor/Omni-SafetyBench `
  --output data/processed/omni-safetybench/audio-text-tts.jsonl `
  --subset audio-text-tts `
  --hash-media
```

`parallel_seed_id` 保留官方 `id`，用于以后对齐相同种子的 image-text、audio-text 和
video-text 结果。第一阶段不转换 image-audio-text 或 video-audio-text。

## 只检查元数据

媒体还未下载时可以加 `--skip-media-verification` 检查字段映射。此模式不会证明媒体
存在，也不会生成媒体哈希，不能用于正式实验 Manifest。

## 真实小样本验收（M2 并行任务）

不要先下载或转换全量。每个数据集选一个官方子集，保留 5–20 个样本及其真实媒体，
然后使用 `--limit`、媒体校验和哈希生成验收产物：

```powershell
uv run safejudge data convert `
  --adapter <adapter> `
  --input <official-metadata-file> `
  --media-root <official-media-root> `
  --output data/acceptance/<dataset>.jsonl `
  --limit 10 `
  --hash-media
```

人工逐行抽查以下内容，并把结果记录在同目录 `ACCEPTANCE.md`：

1. `sample_id` 唯一，`source.original_id` 能回到官方记录。
2. 每个 `media.uri` 都是相对路径，拼接 media root 后确实存在且能人工打开。
3. `media_type`、MIME、SHA-256、大小和音视频时长字段与文件一致。
4. 文本、风险类别、攻击类型及 benign/harmful 意图没有错位。
5. Manifest 的样本数、模态数和输出哈希与 JSONL 一致。

验收顺序建议：MM-SafetyBench 图文 5 条、MOSSBench 图文 5 条、Omni-SafetyBench
image/audio/video-text 各 5 条。真实媒体仍不提交 Git。

### 本机验收结果（2026-08-04）

已完成以下真实小样本下载与转换，输出位于被 Git 忽略的 `data/acceptance/`：

| 数据集/子集 | 数量 | 结果 |
| --- | ---: | --- |
| MM-SafetyBench `01-Illegal_Activitiy/SD` | 5 | 通过 |
| MOSSBench `images` | 5 | 通过 |
| Omni-SafetyBench `image-text/typo` | 5 | 通过 |
| Omni-SafetyBench `audio-text/tts` | 5 | 通过 |
| Omni-SafetyBench `video-text/typo` | 5 | 通过 |

五组输出均复核了唯一 ID、真实媒体存在性、文件大小和 SHA-256，以及 Manifest 中的
输出哈希。Omni 三个子集使用相同的平行种子 `0..4`。图片已人工打开并与文本/关键词
对照；音频具有 MPEG 帧头，视频具有 MP4 `ftyp` 容器头。当前机器未安装 FFmpeg，
因此时长探测仍留到 M6 的媒体探测任务。
