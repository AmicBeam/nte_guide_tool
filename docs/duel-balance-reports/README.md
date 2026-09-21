# 平衡报告归档约定

此目录只保留规范，不保存单轮报告。训练、评估和平衡实验的报告与数据写入 `artifacts/rl-evals/<run-id>/`，由Git忽略，不提交或推送。

既有旧盖卡报告已迁至 `artifacts/rl-evals/reports/duel-balance-reports/`，保留原文件名。它们只用于历史追溯；旧三角看板不能证明V2已经平衡。

V2完整训练与报告按[完整训练、测试与报告要求](../duel-v2-full-training-report-spec.md)及[训练手册](../duel-v2-public-training.md)执行。报告必须写清模型／规则／构筑身份、样本分母、实际覆盖、失败和缺项，并关联原始评估资料。不得为了归档方便把报告重新复制回 `docs/`。

仅维护旧盖卡实验时，仍记录其三角配对、seed、样本数、胜率、登场指标、看板状态及问题归因；具体流程见[平衡迭代说明](../duel-balance-iteration.md)。
