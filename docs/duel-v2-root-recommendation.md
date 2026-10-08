# 有界根选点诊断与离线对照

本协议只研究冻结模型的推理选点，不训练、换牌、修改卡效或批准网站AI。单轮配置、原始记录、源码副本、模型、报告及录像只放 `artifacts/rl-evals/<run-id>/`。

## 默认算法与问题边界

现行 `recovery_search.choose_root` 按访问层筛选，再按 `log(prior)+Gumbel+completed_Q` 选择；搜索结束使用最大访问层。这与 [Mctx 官方策略的最终选点](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py)及其 [sequential halving](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/seq_halving.py)一致。不能仅凭低先验高Q动作未被选中，将其认定为实现错误。

请求模拟数可能在等访问分配轮次中途结束；服务的即时胜利检查还会减少树预算。访问数相差一次可能因此改变最终资格。恢复资格与改变先验／Q权重是两个问题，必须独立比较。Q是自适应搜索和采样隐藏世界上的估值，不是胜率；局面后续真实获胜也不证明当时已经有可靠终局证明。

## 实验接口

`recovery_search.search(..., root_trace=True)` 额外保存每次已完成模拟的访问层、根动作、回传值、访问数、平均Q和变换Q；默认关闭。轨迹不记录未完成模拟，不改变分配、随机流、默认动作或 `pi`。结合原先验／噪声可复算每轮候选资格和评分。轨迹含搜索私有资料，不能放入公开录像。

`rl/root_recommendation.py:recommend` 只读取搜索结果，提供三个预声明模式：

| 模式 | 最终资格与评分 |
| --- | --- |
| `legacy` | 原最大访问层和原合成评分 |
| `completed_sweep_gumbel` | 最近一次完整等访问分配轮次实际访问的候选，使用最新已完成的原合成评分；未完整的新轮次不淘汰其尚未访问的对手 |
| `completed_sweep_value` | 相同完整轮次候选；每个候选至少访问2次时，以 `(访问数×平均Q+根价值)/(访问数+1)` 排序，原合成分数只破精确同分；不足2次则用上一模式 |

完成一轮不表示Q准确，也不表示新增独立信息集。价值模式的一份根价值伪观测仅为降低低访问估值噪声的研究正则，不是统计置信界。第一轮也未完成时，两种实验模式均保留原动作；不将未访问动作的填充Q作为已验证线路。它们不把已淘汰的全部单访问动作重新放回决选，不承诺最高Q必然更好。

这些模式只用于推理末端消融，分配与 `pi=softmax(log prior+completed_Q)` 保持原值；不得把实验动作当成原Gumbel策略目标或不经验证直接用于训练。网站的公开即时胜利优先检查保留在这些推荐之前：只有不依赖未知响应、抽牌或随机性的证明才可优先，采样世界终局仍只是一份模拟结果。

## 冻结与复现

入口先核对显式提供的旧评估源码清单，复制到新目录，只覆盖本任务的追踪／推荐组件；数字模型、训练清单、规则源码和旧评估保持原字节。共享种子账本分配新范围，已看过的根只作诊断，不进入独立对局分母。

```bash
.venv/bin/python scripts/check_duel_v2_root_selection.py \
  --input artifacts/rl-evals/EXPLICIT_PREVIOUS_RUN \
  --output artifacts/rl-evals/NEW_RUN --phase prepare
.venv/bin/python artifacts/rl-evals/NEW_RUN/run.py \
  --output artifacts/rl-evals/NEW_RUN --phase diagnose
.venv/bin/python artifacts/rl-evals/NEW_RUN/run.py \
  --output artifacts/rl-evals/NEW_RUN --phase games
.venv/bin/python scripts/check_duel_v2_root_selection.py \
  --output artifacts/rl-evals/NEW_RUN --phase report
```

输入格式是含 `config.json`、`new-source-manifest.json`、`sources/new/`、`branch-selection.json` 和完整私有原局 `raw/` 的显式归档。默认真红对创生4个新根种子、创生内战2个、覆纹对快攻2个；每个种子交换先后手，分别执行三个模式，共48局、8个配对种子簇。A方使用所评模式，B方保持原搜索。双方模型与构筑冻结，沿用至多32次证明／树模拟、3秒、16候选、3步短终局检查，无探索噪声。总局数、进程数和时限启动前固定，默认4进程、1200秒；不为追求期望结论追加局数。不是完整五队验收。

真实动作继续用正式引擎，按保存动作逐步核验状态SHA再导出公开事件。记录每次决策的原／候选选择、实际模拟量、完成状态、证明检查与耗时；超时保留的部分搜索如实单列。动作改变后，对手从新局面重新决策，不重放已不合法的旧续局。

## 报告与采用边界

分别报告各对阵、先后手、模式的计划／尝试／正常完成／胜负平／异常／截断，按配对种子簇计算差异及不确定性。动作变化、已知局面机制启动、实际胜负与成本分开陈述；不能把几十个同局决策当成几十场独立证据。单访问高Q、没有完成第一轮、末轮预算不整齐、时限中断、价值错误与先验压制须保留各自诊断。

核验原与新源码清单、数字模型SHA及有限性、原局前缀重放、实际动作合法性、公开录像隐私与双方视角重建。独立对局前冻结有限候选与样本数；只作探索对照时不通过成绩自动挑选或批准模式。广泛强度／退化结论需要另一批未见种子、覆盖更多对手的预声明确认，及现行完整评估规范。小样本或区间宽时保留默认选点。

针对性验证：`.venv/bin/python -m unittest tests.test_duel_v2_root_recommendation tests.test_duel_v2_gumbel_search tests.test_duel_v2_learning_audits tests.test_duel_v2_serving_search`。
