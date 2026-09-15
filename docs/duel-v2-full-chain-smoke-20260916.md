# 双预组全链路短测验收记录 · 2026-09-16

- 结果：6 个训练阶段、34 个冻结评估单元完成，412 场评估正常结束；运行约 18.24 分钟。
- 本轮从零初始化 hidden=32 小模型，只验证流程可运行，不是前一小时模型的强度复测。
- 59 项相关自动化检查通过；报告打包、逐文件校验与公开示例录像导出通过。
- Mac 报告与附件：`artifacts/rl-evals/full-chain-20260916/report-v2/`。
- 最终包：`artifacts/rl-evals/full-chain-20260916/report-bundle-v2.zip`，SHA-256 `f0244da73e9318477727c0746e8cbac1c86fb84dab8a96ffad19b1cc06829f8c`；155 个文件校验通过。
- Windows 原始记录：`D:/git/nte_board_game/.rl-full-chain-20260916/artifacts/full-chain-smoke/`。逐局私有记录和其余公开录像保留在该目录；Mac 包包含报告、CSV、模型/checkpoint、候选和一局公开录像。
- GPU 终局计数微基准：128 环境、1000 次调用总计约 0.175 秒，即每次约 0.175 毫秒。这是独立调用成本，不是端到端训练降速比例。
- 预测胜率未提供校准模型；同动作机制效果属于关联记录，不是隔离的因果收益。两套候选在四场筛选中均未胜过原始配牌，按预定规则保留原始配牌。
- 未部署或替换网站模型，训练/评估进程已结束。下文是最终报告正文，数据附件不提交 Git。

---

# 双预组完整链路短测报告

这是有界短训练的可行性验证，样本不用于证明套牌最优或游戏平衡。

- 规则身份：`c1abaf1979bf377994a0c019e05e1bb4267098782562937d757c73baadc69d7b`。
- 训练标量与冻结评估分别列出；报告生成不批准上线。
- 预测胜率：未提供经校准的概率模型；附逐回合局势 CSV，原始价值分数不冒充胜率。
- 终结/环合效果附源事件；同动作内变化可能由多个效果共同造成，未声称因果胜率贡献。

## 训练与配牌阶段


### starter 阶段结果

本轮基础模型起点：从零初始化；hidden=32。这是短预算链路验证。

| 队伍 | 阶段 | 累计更新 | 本轮更新 | 正常终局 | 用时秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| starter | base | 4 | 4 | 123 | 10.46 |
| starter | cards | 1 | 1 | 8 | 2.07 |
| starter | adapted | 6 | 2 | 30 | 5.86 |

### starter 训练内先后手统计（出牌阶段）

| 阶段 | 对手 | 位置 | 胜/负/平 | 胜率 |
| --- | --- | --- | --- | ---: |
| base | rule | first | 5/24/0 | 17.2% |
| base | rule | second | 9/21/0 | 30.0% |
| base | frozen_models | first | 17/16/0 | 51.5% |
| base | frozen_models | second | 20/11/0 | 64.5% |
| adapted | rule | first | 0/10/0 | 0.0% |
| adapted | rule | second | 0/3/0 | 0.0% |
| adapted | frozen_models | first | 0/7/0 | 0.0% |
| adapted | frozen_models | second | 0/10/0 | 0.0% |

每次更新的累计分组统计在对应 `metrics.jsonl`；停止时未完成局另见 `position-results.json`。

### weave-rush 阶段结果

本轮基础模型起点：从零初始化；hidden=32。这是短预算链路验证。

| 队伍 | 阶段 | 累计更新 | 本轮更新 | 正常终局 | 用时秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| weave-rush | base | 4 | 4 | 208 | 11.51 |
| weave-rush | cards | 1 | 1 | 8 | 1.92 |
| weave-rush | adapted | 6 | 2 | 54 | 6.17 |

### weave-rush 训练内先后手统计（出牌阶段）

| 阶段 | 对手 | 位置 | 胜/负/平 | 胜率 |
| --- | --- | --- | --- | ---: |
| base | rule | first | 33/9/0 | 78.6% |
| base | rule | second | 36/14/0 | 72.0% |
| base | frozen_models | first | 48/7/0 | 87.3% |
| base | frozen_models | second | 57/4/0 | 93.4% |
| adapted | rule | first | 1/2/0 | 33.3% |
| adapted | rule | second | 6/3/0 | 66.7% |
| adapted | frozen_models | first | 14/3/0 | 82.4% |
| adapted | frozen_models | second | 21/4/0 | 84.0% |

每次更新的累计分组统计在对应 `metrics.jsonl`；停止时未完成局另见 `position-results.json`。

## 冻结评估矩阵

| 测试 | 场数 | 学习方先手胜率 | 学习方后手胜率 | 实际先手方胜率 | 平均全局回合 | 异常/截断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| starter-screen-original | 4 | 0.0% | 0.0% | 50.0% | 19.50 | 0/0 |
| starter-screen-candidate | 4 | 0.0% | 0.0% | 50.0% | 19.00 | 0/0 |
| weave-rush-screen-original | 4 | 0.0% | 0.0% | 50.0% | 18.50 | 0/0 |
| weave-rush-screen-candidate | 4 | 0.0% | 0.0% | 50.0% | 18.00 | 0/0 |
| starter-original-rule-starter | 2 | 0.0% | 0.0% | 50.0% | 12.50 | 0/0 |
| starter-original-rule-weave-rush | 2 | 0.0% | 0.0% | 50.0% | 8.50 | 0/0 |
| starter-original-rule-varied | 30 | 0.0% | 0.0% | 50.0% | 15.73 | 0/0 |
| starter-original-model-varied-starter | 30 | 73.3% | 86.7% | 43.3% | 30.63 | 0/0 |
| starter-original-model-varied-weave-rush | 30 | 73.3% | 93.3% | 40.0% | 30.43 | 0/0 |
| starter-original-mirror | 2 | 0.0% | 0.0% | 50.0% | 38.00 | 0/0 |
| weave-rush-original-rule-starter | 2 | 0.0% | 0.0% | 50.0% | 19.50 | 0/0 |
| weave-rush-original-rule-weave-rush | 2 | 0.0% | 0.0% | 50.0% | 11.50 | 0/0 |
| weave-rush-original-rule-varied | 30 | 13.3% | 13.3% | 50.0% | 19.67 | 0/0 |
| weave-rush-original-model-varied-starter | 30 | 93.3% | 100.0% | 46.7% | 16.20 | 0/0 |
| weave-rush-original-model-varied-weave-rush | 30 | 93.3% | 100.0% | 46.7% | 16.53 | 0/0 |
| weave-rush-original-mirror | 2 | 100.0% | 0.0% | 100.0% | 21.00 | 0/0 |
| cross-original | 2 | 0.0% | 0.0% | 50.0% | 28.50 | 0/0 |
| cross-original（覆纹快攻视角，同批对局） | 2 | 100.0% | 100.0% | 50.0% | 28.50 | 同上，勿重复计局 |
| starter-candidate-rule-starter | 2 | 0.0% | 0.0% | 50.0% | 16.50 | 0/0 |
| starter-candidate-rule-weave-rush | 2 | 0.0% | 0.0% | 50.0% | 10.50 | 0/0 |
| starter-candidate-rule-varied | 30 | 0.0% | 0.0% | 50.0% | 15.30 | 0/0 |
| starter-candidate-model-varied-starter | 30 | 40.0% | 46.7% | 46.7% | 32.37 | 0/0 |
| starter-candidate-model-varied-weave-rush | 30 | 53.3% | 60.0% | 46.7% | 31.77 | 0/0 |
| starter-candidate-mirror | 2 | 0.0% | 100.0% | 0.0% | 35.00 | 0/0 |
| weave-rush-candidate-rule-starter | 2 | 0.0% | 0.0% | 50.0% | 16.50 | 0/0 |
| weave-rush-candidate-rule-weave-rush | 2 | 0.0% | 0.0% | 50.0% | 12.50 | 0/0 |
| weave-rush-candidate-rule-varied | 30 | 0.0% | 0.0% | 50.0% | 17.03 | 0/0 |
| weave-rush-candidate-model-varied-starter | 30 | 73.3% | 93.3% | 40.0% | 24.47 | 0/0 |
| weave-rush-candidate-model-varied-weave-rush | 30 | 80.0% | 100.0% | 40.0% | 24.93 | 0/0 |
| weave-rush-candidate-mirror | 2 | 0.0% | 0.0% | 50.0% | 13.50 | 0/0 |
| cross-candidate | 2 | 0.0% | 0.0% | 50.0% | 20.50 | 0/0 |
| cross-candidate（覆纹快攻视角，同批对局） | 2 | 100.0% | 100.0% | 50.0% | 20.50 | 同上，勿重复计局 |
| starter-candidate-old-policy | 2 | 0.0% | 0.0% | 50.0% | 9.50 | 0/0 |
| starter-candidate-new-policy | 2 | 0.0% | 0.0% | 50.0% | 10.50 | 0/0 |
| weave-rush-candidate-old-policy | 2 | 0.0% | 0.0% | 50.0% | 9.50 | 0/0 |
| weave-rush-candidate-new-policy | 2 | 0.0% | 0.0% | 50.0% | 9.50 | 0/0 |

交叉互搏的首行以创生为学习方；第二行换算覆纹快攻视角，来自同批对局。

## 构筑调整与理由

下表只依据独立于最终测试集的筛选样本；小样本用于验证选择流程，不能据此声称最优。候选生成过程是配牌策略采样，不伪造逐张换牌的模型内心理由。

### starter

- 原始筛选胜率：0.0%；候选：0.0%；每种 4 场。
- 选择：original；理由：Candidate did not beat baseline; retain baseline。
- 全批候选及原始评分：`starter/cards/candidate-history.jsonl`。
- 候选适应训练单独完成；本表筛选使用冻结基础策略，最终矩阵另外展示适应后效果。

| 卡牌 | 原始张数 | 候选张数 |
| --- | ---: | ---: |
| J02 过往消息皆为利刃 | 2 | 0 |
| J03 优势初成 | 0 | 1 |
| J04 出手即致命 | 2 | 0 |
| J05 掌握底细的优越 | 0 | 1 |
| J06 知晓每一条秘密 | 0 | 2 |
| J07 使命必达 | 0 | 2 |
| J08 现实避难所 | 2 | 0 |
| N01 帮派初建成 | 0 | 1 |
| N02 第二位成员 | 2 | 1 |
| N03 要叫大姐头 | 2 | 0 |
| N04 不是闯祸精 | 2 | 0 |
| N05 小弟遍天下 | 0 | 1 |
| N06 因为是家人 | 0 | 2 |
| N07 心有猛虎 | 2 | 1 |
| N08 预备备 | 0 | 2 |
| Y01 重复囚徒困境 | 0 | 1 |
| Y03 策略特性：善良 | 2 | 1 |
| Y04 策略特性：可激怒 | 0 | 1 |
| Y05 策略特性：清晰 | 0 | 1 |
| Y06 策略特性：宽容 | 2 | 1 |
| Y08 错误的门 | 2 | 1 |
| Z01 初明凝视 | 0 | 2 |
| Z04 未决迷数 | 2 | 0 |
| Z07 休息日 | 2 | 1 |
| Z08 倾世之雨 | 1 | 2 |

| 采样版本 | 候选 | 先手胜/完成 | 后手胜/完成 | 筛选分 | 处理 |
| --- | --- | --- | --- | ---: | --- |
| 0 | tuned-bf2276e1a8bb | 0/1 | 0/1 | -1.0 | 本批未选中；不据此认定构筑更差 |
| 0 | tuned-a27ba19f61d7 | 1/1 | 0/1 | 0.0 | 本批未选中；不据此认定构筑更差 |
| 0 | tuned-e0b98f7115ee | 0/1 | 1/1 | 0.0 | 本批未选中；不据此认定构筑更差 |
| 0 | tuned-4a5602dae281 | 1/1 | 1/1 | 1.0 | 选入额外配对筛选 |

### weave-rush

- 原始筛选胜率：0.0%；候选：0.0%；每种 4 场。
- 选择：original；理由：Candidate did not beat baseline; retain baseline。
- 全批候选及原始评分：`weave-rush/cards/candidate-history.jsonl`。
- 候选适应训练单独完成；本表筛选使用冻结基础策略，最终矩阵另外展示适应后效果。

| 卡牌 | 原始张数 | 候选张数 |
| --- | ---: | ---: |
| B01 唷 队长！ | 2 | 1 |
| B02 招魂幡 | 2 | 1 |
| B03 恶诅不过一开口 | 0 | 1 |
| B05 敬此月色 | 0 | 1 |
| B07 判予秋 | 0 | 1 |
| B08 茶话会 | 2 | 1 |
| M01 特遣行动 | 2 | 1 |
| M02 二次补考 | 0 | 2 |
| M04 午休时间 | 2 | 1 |
| M07 美好的一天从睡醒开始 | 0 | 1 |
| M08 开始净空 | 2 | 1 |
| Y03 策略特性：善良 | 2 | 0 |
| Y04 策略特性：可激怒 | 0 | 1 |
| Y05 策略特性：清晰 | 0 | 1 |
| Z01 初明凝视 | 0 | 2 |
| Z03 奇异记叙 | 2 | 1 |
| Z05 默示赋命 | 2 | 0 |
| Z06 诓定解放 | 0 | 1 |

| 采样版本 | 候选 | 先手胜/完成 | 后手胜/完成 | 筛选分 | 处理 |
| --- | --- | --- | --- | ---: | --- |
| 0 | tuned-db60b875cff4 | 0/1 | 0/1 | -1.0 | 本批未选中；不据此认定构筑更差 |
| 0 | tuned-a89856ca7843 | 1/1 | 1/1 | 1.0 | 选入额外配对筛选 |
| 0 | tuned-da208466e9dd | 0/1 | 0/1 | -1.0 | 本批未选中；不据此认定构筑更差 |
| 0 | tuned-1d598710a9f4 | 0/1 | 0/1 | -1.0 | 本批未选中；不据此认定构筑更差 |

## 本轮选定配牌：逐卡与机制摘要

下表以最终变化规则 AI 测试为样本；全部对阵细分见 CSV。抽到仅计 draw 事件，见到还包括起手、换牌、检索与回收等原卡入手。成功使用只计原卡，复制/衍生来源另列。

| 套牌 | 位置 | 卡牌 | 张数 | 见到局数/总数 | 打出局数 | 打出时胜率 | 平均首次打出回合 | 使用局平均结束回合 |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| starter | first | N02 第二位成员 | 2 | 8/15 | 1 | 0.0% | 6.00 | 30.00 |
| starter | first | N03 要叫大姐头 | 2 | 9/15 | 1 | 0.0% | 15.00 | 20.00 |
| starter | first | N04 不是闯祸精 | 2 | 9/15 | 5 | 0.0% | 9.00 | 18.40 |
| starter | first | N07 心有猛虎 | 2 | 13/15 | 4 | 0.0% | 8.50 | 20.00 |
| starter | first | Z02 探悉天职 | 1 | 6/15 | 6 | 0.0% | 8.00 | 14.00 |
| starter | first | Z04 未决迷数 | 2 | 12/15 | 12 | 0.0% | 6.00 | 15.33 |
| starter | first | Z05 默示赋命 | 2 | 13/15 | 13 | 0.0% | 6.08 | 15.08 |
| starter | first | Z07 休息日 | 2 | 10/15 | 7 | 0.0% | 9.86 | 17.43 |
| starter | first | Z08 倾世之雨 | 1 | 6/15 | 5 | 0.0% | 8.60 | 14.00 |
| starter | first | J01 知晓与制衡 | 2 | 9/15 | 8 | 0.0% | 7.50 | 13.25 |
| starter | first | J02 过往消息皆为利刃 | 2 | 10/15 | 9 | 0.0% | 7.22 | 16.22 |
| starter | first | J04 出手即致命 | 2 | 11/15 | 11 | 0.0% | 3.91 | 15.09 |
| starter | first | J08 现实避难所 | 2 | 12/15 | 12 | 0.0% | 7.17 | 14.67 |
| starter | first | Y02 策略：「一报还一报」 | 2 | 10/15 | 9 | 0.0% | 6.56 | 15.33 |
| starter | first | Y03 策略特性：善良 | 2 | 12/15 | 12 | 0.0% | 6.00 | 15.17 |
| starter | first | Y06 策略特性：宽容 | 2 | 11/15 | 8 | 0.0% | 7.25 | 13.75 |
| starter | first | Y08 错误的门 | 2 | 14/15 | 13 | 0.0% | 6.08 | 14.00 |
| starter | second | N02 第二位成员 | 2 | 8/15 | 2 | 0.0% | 14.00 | 18.50 |
| starter | second | N03 要叫大姐头 | 2 | 8/15 | 3 | 0.0% | 18.67 | 23.67 |
| starter | second | N04 不是闯祸精 | 2 | 9/15 | 7 | 0.0% | 9.71 | 19.57 |
| starter | second | N07 心有猛虎 | 2 | 13/15 | 7 | 0.0% | 12.57 | 19.86 |
| starter | second | Z02 探悉天职 | 1 | 6/15 | 5 | 0.0% | 9.20 | 17.60 |
| starter | second | Z04 未决迷数 | 2 | 10/15 | 10 | 0.0% | 5.80 | 18.70 |
| starter | second | Z05 默示赋命 | 2 | 14/15 | 14 | 0.0% | 6.57 | 16.36 |
| starter | second | Z07 休息日 | 2 | 9/15 | 8 | 0.0% | 9.75 | 19.62 |
| starter | second | Z08 倾世之雨 | 1 | 8/15 | 7 | 0.0% | 9.14 | 16.86 |
| starter | second | J01 知晓与制衡 | 2 | 9/15 | 9 | 0.0% | 6.89 | 14.33 |
| starter | second | J02 过往消息皆为利刃 | 2 | 12/15 | 12 | 0.0% | 8.83 | 17.75 |
| starter | second | J04 出手即致命 | 2 | 12/15 | 12 | 0.0% | 6.33 | 17.08 |
| starter | second | J08 现实避难所 | 2 | 12/15 | 12 | 0.0% | 7.50 | 16.75 |
| starter | second | Y02 策略：「一报还一报」 | 2 | 9/15 | 9 | 0.0% | 9.11 | 18.44 |
| starter | second | Y03 策略特性：善良 | 2 | 13/15 | 13 | 0.0% | 6.77 | 16.08 |
| starter | second | Y06 策略特性：宽容 | 2 | 12/15 | 7 | 0.0% | 5.14 | 15.43 |
| starter | second | Y08 错误的门 | 2 | 12/15 | 12 | 0.0% | 6.33 | 16.58 |
| weave-rush | first | M01 特遣行动 | 2 | 10/15 | 7 | 28.6% | 9.29 | 20.57 |
| weave-rush | first | M03 全勤奖金 | 2 | 11/15 | 11 | 9.1% | 7.73 | 20.27 |
| weave-rush | first | M04 午休时间 | 2 | 10/15 | 10 | 10.0% | 7.80 | 18.90 |
| weave-rush | first | M08 开始净空 | 2 | 12/15 | 12 | 16.7% | 5.17 | 19.83 |
| weave-rush | first | B01 唷 队长！ | 2 | 12/15 | 12 | 8.3% | 5.17 | 19.42 |
| weave-rush | first | B02 招魂幡 | 2 | 12/15 | 11 | 18.2% | 6.09 | 18.91 |
| weave-rush | first | B06 于五色为白 | 2 | 12/15 | 12 | 16.7% | 7.50 | 20.50 |
| weave-rush | first | B08 茶话会 | 2 | 10/15 | 10 | 20.0% | 6.40 | 18.60 |
| weave-rush | first | Z03 奇异记叙 | 2 | 11/15 | 5 | 40.0% | 11.00 | 20.80 |
| weave-rush | first | Z04 未决迷数 | 2 | 14/15 | 11 | 9.1% | 11.91 | 20.09 |
| weave-rush | first | Z05 默示赋命 | 2 | 12/15 | 11 | 18.2% | 8.82 | 20.91 |
| weave-rush | first | Z08 倾世之雨 | 2 | 11/15 | 8 | 25.0% | 10.25 | 20.00 |
| weave-rush | first | Y01 重复囚徒困境 | 2 | 11/15 | 10 | 10.0% | 9.40 | 19.90 |
| weave-rush | first | Y03 策略特性：善良 | 2 | 12/15 | 12 | 8.3% | 6.83 | 18.92 |
| weave-rush | first | Y06 策略特性：宽容 | 2 | 11/15 | 11 | 9.1% | 6.45 | 19.73 |
| weave-rush | first | Y07 想做什么梦？ | 2 | 12/15 | 12 | 16.7% | 8.33 | 19.67 |
| weave-rush | second | M01 特遣行动 | 2 | 11/15 | 10 | 20.0% | 12.20 | 21.50 |
| weave-rush | second | M03 全勤奖金 | 2 | 10/15 | 10 | 10.0% | 7.80 | 20.90 |
| weave-rush | second | M04 午休时间 | 2 | 9/15 | 9 | 22.2% | 7.78 | 20.89 |
| weave-rush | second | M08 开始净空 | 2 | 12/15 | 12 | 16.7% | 5.00 | 21.08 |
| weave-rush | second | B01 唷 队长！ | 2 | 13/15 | 13 | 7.7% | 7.23 | 18.92 |
| weave-rush | second | B02 招魂幡 | 2 | 12/15 | 11 | 18.2% | 5.45 | 20.00 |
| weave-rush | second | B06 于五色为白 | 2 | 11/15 | 10 | 20.0% | 6.40 | 21.10 |
| weave-rush | second | B08 茶话会 | 2 | 10/15 | 10 | 20.0% | 6.80 | 19.50 |
| weave-rush | second | Z03 奇异记叙 | 2 | 9/15 | 6 | 33.3% | 11.00 | 23.17 |
| weave-rush | second | Z04 未决迷数 | 2 | 12/15 | 11 | 9.1% | 11.09 | 19.73 |
| weave-rush | second | Z05 默示赋命 | 2 | 12/15 | 11 | 18.2% | 7.27 | 21.27 |
| weave-rush | second | Z08 倾世之雨 | 2 | 11/15 | 8 | 25.0% | 11.00 | 21.88 |
| weave-rush | second | Y01 重复囚徒困境 | 2 | 12/15 | 12 | 16.7% | 11.83 | 21.08 |
| weave-rush | second | Y03 策略特性：善良 | 2 | 12/15 | 12 | 8.3% | 7.83 | 19.50 |
| weave-rush | second | Y06 策略特性：宽容 | 2 | 11/15 | 11 | 9.1% | 6.91 | 19.18 |
| weave-rush | second | Y07 想做什么梦？ | 2 | 12/15 | 12 | 16.7% | 8.83 | 20.08 |

### 终结与环合

| 测试 | 位置 | 角色 | 机制 | 次数 | 使用局数 | 使用局胜率 | 可用角色回合 | 首次使用回合 | 同动作对手生命变化/次 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| starter-original-rule-varied | first | nanali | ultimate | 0 | 0 | — | 24 | — | — |
| starter-original-rule-varied | first | nanali | harmony | 0 | 0 | — | 30 | — | — |
| starter-original-rule-varied | first | iloy | ultimate | 0 | 0 | — | 16 | — | — |
| starter-original-rule-varied | first | iloy | harmony | 1 | 1 | 0.0% | 28 | 11.0 | 0.0 |
| starter-original-rule-varied | first | zero | ultimate | 0 | 0 | — | 27 | — | — |
| starter-original-rule-varied | first | zero | harmony | 0 | 0 | — | 3 | — | — |
| starter-original-rule-varied | first | jiuyuan | ultimate | 0 | 0 | — | 22 | — | — |
| starter-original-rule-varied | first | jiuyuan | harmony | 0 | 0 | — | 32 | — | — |
| starter-original-rule-varied | second | nanali | ultimate | 0 | 0 | — | 31 | — | — |
| starter-original-rule-varied | second | nanali | harmony | 2 | 2 | 0.0% | 18 | 17.0 | -7.0 |
| starter-original-rule-varied | second | iloy | ultimate | 0 | 0 | — | 26 | — | — |
| starter-original-rule-varied | second | iloy | harmony | 3 | 3 | 0.0% | 20 | 10.666666666666666 | -2.0 |
| starter-original-rule-varied | second | zero | ultimate | 0 | 0 | — | 39 | — | — |
| starter-original-rule-varied | second | zero | harmony | 1 | 1 | 0.0% | 8 | 24.0 | -4.0 |
| starter-original-rule-varied | second | jiuyuan | ultimate | 0 | 0 | — | 43 | — | — |
| starter-original-rule-varied | second | jiuyuan | harmony | 0 | 0 | — | 21 | — | — |
| weave-rush-original-rule-varied | first | bohe | ultimate | 0 | 0 | — | 43 | — | — |
| weave-rush-original-rule-varied | first | bohe | harmony | 7 | 6 | 16.7% | 17 | 10.666666666666666 | -2.5714285714285716 |
| weave-rush-original-rule-varied | first | baicang | ultimate | 0 | 0 | — | 56 | — | — |
| weave-rush-original-rule-varied | first | baicang | harmony | 0 | 0 | — | 10 | — | — |
| weave-rush-original-rule-varied | first | zero | ultimate | 0 | 0 | — | 47 | — | — |
| weave-rush-original-rule-varied | first | zero | harmony | 2 | 2 | 0.0% | 6 | 16.0 | -3.5 |
| weave-rush-original-rule-varied | first | iloy | ultimate | 0 | 0 | — | 78 | — | — |
| weave-rush-original-rule-varied | first | iloy | harmony | 0 | 0 | — | 37 | — | — |
| weave-rush-original-rule-varied | second | bohe | ultimate | 0 | 0 | — | 49 | — | — |
| weave-rush-original-rule-varied | second | bohe | harmony | 5 | 4 | 25.0% | 15 | 10.0 | -4.0 |
| weave-rush-original-rule-varied | second | baicang | ultimate | 0 | 0 | — | 57 | — | — |
| weave-rush-original-rule-varied | second | baicang | harmony | 2 | 2 | 100.0% | 8 | 25.0 | -3.0 |
| weave-rush-original-rule-varied | second | zero | ultimate | 0 | 0 | — | 42 | — | — |
| weave-rush-original-rule-varied | second | zero | harmony | 1 | 1 | 0.0% | 2 | 14.0 | -3.0 |
| weave-rush-original-rule-varied | second | iloy | ultimate | 0 | 0 | — | 82 | — | — |
| weave-rush-original-rule-varied | second | iloy | harmony | 0 | 0 | — | 35 | — | — |

环合类型、提供者、受益者、终结结束及源效果事件保留在每局 telemetry；机会以存在合法发动动作的角色·全局回合去重，包含普通出击和战斗牌；同回合多次触发另计次数。

## 曲线与统计限制

- `turn-states.csv` 给出各回合仍在进行的样本量及平均生命、行动力、能量、环合值；没有概率预测。
- `card-metrics.csv` 包含抽到/见到/使用的分子分母、来源、平均次数和回合数。
- `training-win-rates.csv` 保留各次更新按对手类型和先后手的胜率序列，不能混入冻结评估。
- `generated-card-metrics.csv` 单列复制与衍生牌的获得、使用和胜负关联，不混入主牌组抽到率。
- `mechanism-metrics.csv` 包含按角色/先后手的机会、次数和同动作生命变化。跨动作/跨回合的净因果收益尚未归因，不将相邻效果全算作某机制贡献。
- `sample-replay.json` 是已记录对局的公开可播放录像；私有种子及隐藏牌库不在其中。
- 全部测试可执行不等于本轮样本量足以判定强度；未使用的卡牌/机制不判弱。
- 本轮不提供可靠预测胜率、长期收敛结论或机制的因果胜率贡献。
