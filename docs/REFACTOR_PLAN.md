# 主框架渐进重构计划

目标是在不改变模型请求、Prompt、评估结果 Schema 和历史入口的前提下，先消除低风险基础设施
重复，再降低工作流主图的职责密度。

## 本轮范围

- [x] 统一文件 SHA-256、`FileDigest` 和原子写入实现。
- [x] Target/Evaluation batch 复用统一文件基础设施。
- [x] 将评分、冲突保守等级和不可变结果更新移出 LangGraph 主图。
- [x] 合并 Judge 与 Guardrail 的 Provider 构造入口。
- [x] 使用单元测试、完整测试、Ruff、MyPy 和缓存重放验证兼容性。

## 暂不处理

- [x] 经确认后移除 `--no-taxonomy` 旧全局裁判路径。
- 不删除历史 Schema 兼容和旧实验读取能力。
- 不改变 Prompt 拼装、版本或请求哈希。
- 不改变审计账本、缓存、checkpoint 和 manifest 字段。

## 后续候选

历史 Schema 迁移仍保留，后续只有在历史产物归档策略确认后才移动到独立适配层。
