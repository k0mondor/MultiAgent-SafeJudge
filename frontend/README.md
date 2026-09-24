# SafeJudge Explorer

本地答辩展示界面。采用 React、React Flow、shadcn/ui 和 Radix UI；不发起模型调用。

## 运行

在 frontend 目录执行：

```powershell
npm install
npm run dev
```

浏览器打开终端显示的本地地址。`npm run build` 进行类型检查与生产构建。

## 使用

- `/`：按阶段浏览评估流程。滚轮缩放画布，顶部阶段标签或箭头切换阶段；点击节点聚焦并打开详情。Esc 关闭详情。
- `/models`：模型配置演示，按角色过滤注册表中的 profile。选择保存到浏览器本地，不修改后端。
- `/cases`：精选冻结实验案例，支持图像放大、本地音频、回答原文、Intent/媒体理解/分类/裁判/聚合记录展开与仲裁评分对照。
- 全流程视图按四个阶段分区，点击分区或节点进入对应阶段；放大后分区标签淡出。
- 进入阶段时自动播放并保留蓝色连线；缩放和节点聚焦不重播。

注册表快照来自 `config/models.toml` 与 `config/aggregators/shifted-product-v1.toml`，保存在 `src/registry.json`。修改后端配置后须刷新该快照。它不包含密钥或端点地址。聚合版本初始展示 v1.3，不代表所有历史实验采用该版本。

`src/components/ui/` 的 button/card/sheet/tabs 下载自 shadcn/ui 官方 new-york-v4 registry（MIT），仅调整 cn 与 Radix 导入路径以及关闭按钮中文标签。核心 UI 采用实际 HTML/SVG 组件，生成预览图不参与页面渲染。

## 案例数据

`src/cases-data.json` 是四条真实评估记录的展示快照，来源为 `outputs/blind-threshold27-200-20260918/evaluations.jsonl`；媒体从验证样本按 SHA-256 校验后复制至 `public/cases/`。页面不发起推理。译文、摘要与高亮为展示整理，原始请求、冻结回答和中间输出可展开核对。无四轴记录时不展示评分，不补零。

嵌入媒体来自 Omni-SafetyBench（CC-BY-NC-4.0），保留用于非商业研究和答辩展示；来源与许可证见每个案例的实验信息。
