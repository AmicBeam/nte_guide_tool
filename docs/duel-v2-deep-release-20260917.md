# 深度训练最终模型与配牌交付 · 2026-09-17

## 交付摘要

- 训练 01:55–06:42，约 4 小时 47 分钟，六阶段正常完成。纯胜负奖励、v3 起手学习、固定角色配牌训练与4轮局部搜索均生效。
- 最终冻结评估耗时约60分钟，34个常规单元1472局＋8个起手对照单元256局，共1728局完整结束。
- 两套均使用预先选中的局部配牌和随后适配的模型。新模型已通过真实报告准入、当前规则哈希、数值加载、角色顺序和实际配牌绑定校验；不是把训练末批采样牌表当最终构筑。
- 逐卡主表来自最终同模型同构筑镜像互搏，使用双方视角；规则AI逐卡数据不充当主表。预测胜率未校准，仍明确不可用。
- 更新项目高级人机的权重与 serving_build；普通官方预组和玩家构筑不变。未部署生产站点，部署等待用户验收。

## 两套对规则 AI 的先后手胜率

每行32局，先后手各16局；属于有限样本。

| 学习方 | 规则对手预组 | 先手 | 后手 |
| --- | --- | ---: | ---: |
| 创生 | 创生 | 87.50% | 100.00% |
| 创生 | 快攻 | 68.75% | 81.25% |
| 快攻 | 创生 | 100.00% | 100.00% |
| 快攻 | 快攻 | 100.00% | 81.25% |

## 同模型同构筑镜像互搏

按整场实际先后手统计，双方视角归并；每套32局，不能把两个视角当成64场独立对局。

| 套牌 | 先手胜率 | 后手胜率 | 平局率 |
| --- | ---: | ---: | ---: |
| 创生 | 40.62% | 59.38% | 0.00% |
| 快攻 | 62.50% | 37.50% | 0.00% |

## 交叉互搏

| 套牌 | 自己先手胜率 | 自己后手胜率 |
| --- | ---: | ---: |
| 创生 | 37.50% | 31.25% |
| 快攻 | 68.75% | 62.50% |

同一批32局的两个视角，不重复计局。创生对快攻仍处下风；功能与准入通过不等于牌组已平衡。

## 相对上一版高级人机的实际换牌

### 创生
- 换出：N08「预备备」×1、Y08「错误的门」×1、Y03「策略特性：善良」×1、Z04「未决迷数」×1、Z07「休息日」×1、J08「现实避难所」×1、J03「优势初成」×1。
- 换入：N07「心有猛虎」×1、Y02「策略：「一报还一报」」×2、Z05「默示赋命」×2、J01「知晓与制衡」×1、J02「过往消息皆为利刃」×1。
- 模型 SHA-256：`55cb64291951d97c2ea5a470ca6a446618defa67e50ce10c78c62bf2590fe921`。
- 配牌 SHA-256：`cd9c6ed53bb9f58a6886cd0517710d5ceea53dca0268158080c8bf6d259b2fe5`。
- 版本：resident_public_v3，适配阶段更新 572 次；开局策略 learned_joint_v1。

### 快攻
- 换出：M08「开始净空」×1、B08「茶花会」×2、Z05「默示赋命」×2。
- 换入：M02「二次补考」×1、B07「判予秋」×2、Z07「休息日」×2。
- 模型 SHA-256：`42e3a063f0a5d3b13f7593f75b48aebd2b10a5ce85ec908691a14583422d43ca`。
- 配牌 SHA-256：`727d0c11d3de614172aa7929f7390e41ee19514d6856b1a6121872f6e5a6e635`。
- 版本：resident_public_v3，适配阶段更新 587 次；开局策略 learned_joint_v1。

## 验证与产物

- 浏览器从大厅实际进入两套高级人机：创生房间 `97D92A` 的新模型主动换2张起手、首回合出牌出击并正常推进到第3回合；快攻房间 `4B711B` 成功开局并完成换牌。
- 项目加载、规则不匹配拒绝、实际高级人机 API、玩家自定义构筑、镜像配牌、完成对局与回放，以及报告打包回归共33项通过。v3 升级后修正了一处只识别 v2 的测试注入目标，未放宽加载规则校验。
- 打包修复：优先取 selected-deck.json，批准后的模型和配牌均与已评估结果按 SHA 绑定；回归覆盖“末批候选与最终局部配牌不同”的情况。
- Mac 完整报告：`artifacts/rl-evals/deep-20260917/report/report.md`。
- 起手对照：同目录 `opening-analysis/opening-report.md`；最终双方配牌：`final/starter/deck.json`、`final/weave-rush/deck.json`。
- 已验证公开例局：同目录 `selected-cross-replay.json`，为最终两套候选交叉互搏对局，通过公开录像隐私检查。
- 压缩包：`artifacts/rl-evals/deep-20260917/report-bundle.zip`，30,179,202字节，30,959文件逐项校验通过。
- 包 SHA-256：`4716b8796a376e1c374b5c7d4da52a7bc2c9ffc8232aafc4fe90463094a8b53b`。
- Windows 完整私有逐局记录：`D:/git/nte_board_game/.rl-deep-20260917/artifacts/deep-six-hour/`；不入 Git。

## 定时延迟说明

原定9点和12:10的调度规则遗漏 UTC 换算，实际触发时间晚8小时；13:40修正后的自动任务已触发并完成本次交付。训练本身按预算正常结束。此处如实保留过程，不将配置保存成功当作此前已执行。

---

## 完整训练与评估正文

# 双预组综合训练报告

这是本轮有界综合训练与冻结评估报告，单次结果不证明全局最优或游戏平衡。

- 规则身份：`16601946435bb6145eb0b5888bf7db2c4b0dee16ce4f8eeff2ea8a241478bdd9`。
- 训练标量与冻结评估分别列出；报告生成不批准上线。
- 预测胜率：未提供经校准的概率模型；附逐回合局势 CSV，原始价值分数不冒充胜率。
- 终结/环合效果附源事件；同动作内变化可能由多个效果共同造成，未声称因果胜率贡献。

## 训练与配牌阶段


### starter 阶段结果

本轮基础模型起点：D:\git\nte_board_game\.rl-outcome-n01-20260916\artifacts\outcome-n01-two-hour-20260916\starter\base\latest.pt；hidden=64。训练预算及停止原因见阶段记录。

| 队伍 | 阶段 | 累计更新 | 本轮更新 | 正常终局 | 用时秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| starter | base | 1062 | 1062 | 420280 | 3240.12 |
| starter | cards | 846 | 846 | 541440 | 3238.16 |
| starter | adapted | 572 | 572 | 233797 | 1620.19 |

### starter 训练内先后手统计（出牌阶段）

| 阶段 | 对手 | 位置 | 胜/负/平 | 胜率 |
| --- | --- | --- | --- | ---: |
| base | rule | first | 40224/1913/0 | 95.5% |
| base | rule | second | 40760/1418/0 | 96.6% |
| base | frozen_models | first | 103405/64709/0 | 61.5% |
| base | frozen_models | second | 106875/61259/0 | 63.6% |
| base | 阵容池/starter | first | 55609/28711/0 | 65.9% |
| base | 阵容池/starter | second | 57344/26640/0 | 68.3% |
| base | 阵容池/weave-rush | first | 50332/33795/0 | 59.8% |
| base | 阵容池/weave-rush | second | 50960/33154/0 | 60.6% |
| base | 阵容池/public-random | first | 37688/4116/0 | 90.2% |
| base | 阵容池/public-random | second | 39331/2883/0 | 93.2% |
| adapted | rule | first | 22704/696/0 | 97.0% |
| adapted | rule | second | 22847/461/0 | 98.0% |
| adapted | frozen_models | first | 63207/30471/24 | 67.5% |
| adapted | frozen_models | second | 65369/28126/18 | 69.9% |
| adapted | 阵容池/starter | first | 35519/11233/1 | 76.0% |
| adapted | 阵容池/starter | second | 36152/10751/1 | 77.1% |
| adapted | 阵容池/weave-rush | first | 28554/18115/23 | 61.2% |
| adapted | 阵容池/weave-rush | second | 29850/16766/17 | 64.0% |
| adapted | 阵容池/public-random | first | 21838/1819/0 | 92.3% |
| adapted | 阵容池/public-random | second | 22214/1070/0 | 95.4% |

每次更新的累计分组统计在对应 `metrics.jsonl`；停止时未完成局另见 `position-results.json`。对手类型与阵容池是同批对局的不同切分，不可相加。

### weave-rush 阶段结果

本轮基础模型起点：D:\git\nte_board_game\.rl-outcome-n01-20260916\artifacts\outcome-n01-two-hour-20260916\weave-rush\base\latest.pt；hidden=64。训练预算及停止原因见阶段记录。

| 队伍 | 阶段 | 累计更新 | 本轮更新 | 正常终局 | 用时秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| weave-rush | base | 1166 | 1166 | 610694 | 3240.04 |
| weave-rush | cards | 962 | 962 | 615680 | 3237.20 |
| weave-rush | adapted | 587 | 587 | 318590 | 1620.11 |

### weave-rush 训练内先后手统计（出牌阶段）

| 阶段 | 对手 | 位置 | 胜/负/平 | 胜率 |
| --- | --- | --- | --- | ---: |
| base | rule | first | 60154/861/8 | 98.6% |
| base | rule | second | 60435/821/1 | 98.7% |
| base | frozen_models | first | 185750/58381/297 | 76.0% |
| base | frozen_models | second | 183407/60753/291 | 75.0% |
| base | 阵容池/starter | first | 98589/23670/23 | 80.6% |
| base | 阵容池/starter | second | 98924/22879/14 | 81.2% |
| base | 阵容池/weave-rush | first | 88131/33580/267 | 72.3% |
| base | 阵容池/weave-rush | second | 85116/37209/267 | 69.4% |
| base | 阵容池/public-random | first | 59184/1992/15 | 96.7% |
| base | 阵容池/public-random | second | 59802/1486/11 | 97.6% |
| adapted | rule | first | 31461/272/0 | 99.1% |
| adapted | rule | second | 31689/209/1 | 99.3% |
| adapted | frozen_models | first | 106693/20698/142 | 83.7% |
| adapted | frozen_models | second | 106710/20585/130 | 83.7% |
| adapted | 阵容池/starter | first | 56699/7095/9 | 88.9% |
| adapted | 阵容池/starter | second | 57282/6595/5 | 89.7% |
| adapted | 阵容池/weave-rush | first | 50214/13295/128 | 78.9% |
| adapted | 阵容池/weave-rush | second | 49663/13810/124 | 78.1% |
| adapted | 阵容池/public-random | first | 31241/580/5 | 98.2% |
| adapted | 阵容池/public-random | second | 31454/389/2 | 98.8% |

每次更新的累计分组统计在对应 `metrics.jsonl`；停止时未完成局另见 `position-results.json`。对手类型与阵容池是同批对局的不同切分，不可相加。

## 冻结评估矩阵

| 测试 | 场数 | 学习方先手胜率 | 学习方后手胜率 | 实际先手方胜率 | 平均全局回合 | 异常/截断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| starter-screen-original | 32 | 93.8% | 100.0% | 46.9% | 17.59 | 0/0 |
| starter-screen-candidate | 32 | 87.5% | 100.0% | 43.8% | 19.00 | 0/0 |
| weave-rush-screen-original | 32 | 100.0% | 100.0% | 50.0% | 13.00 | 0/0 |
| weave-rush-screen-candidate | 32 | 100.0% | 100.0% | 50.0% | 14.75 | 0/0 |
| starter-original-rule-starter | 32 | 93.8% | 93.8% | 50.0% | 17.19 | 0/0 |
| starter-original-rule-weave-rush | 32 | 50.0% | 50.0% | 50.0% | 18.78 | 0/0 |
| starter-original-rule-varied | 64 | 90.6% | 100.0% | 45.3% | 18.94 | 0/0 |
| starter-original-model-varied-starter | 64 | 93.8% | 84.4% | 54.7% | 19.19 | 0/0 |
| starter-original-model-varied-weave-rush | 64 | 90.6% | 90.6% | 50.0% | 17.69 | 0/0 |
| starter-original-mirror | 32 | 31.2% | 37.5% | 46.9% | 20.50 | 0/0 |
| weave-rush-original-rule-starter | 32 | 87.5% | 93.8% | 46.9% | 13.44 | 0/0 |
| weave-rush-original-rule-weave-rush | 32 | 87.5% | 81.2% | 53.1% | 15.00 | 0/0 |
| weave-rush-original-rule-varied | 64 | 96.9% | 93.8% | 51.6% | 15.08 | 0/0 |
| weave-rush-original-model-varied-starter | 64 | 87.5% | 90.6% | 48.4% | 14.95 | 0/0 |
| weave-rush-original-model-varied-weave-rush | 64 | 100.0% | 96.9% | 51.6% | 13.25 | 0/0 |
| weave-rush-original-mirror | 32 | 56.2% | 75.0% | 40.6% | 16.38 | 0/0 |
| cross-original | 32 | 18.8% | 25.0% | 46.9% | 17.88 | 0/0 |
| cross-original（覆纹快攻视角，同批对局） | 32 | 75.0% | 81.2% | 46.9% | 17.88 | 同上，勿重复计局 |
| starter-candidate-rule-starter | 32 | 87.5% | 100.0% | 43.8% | 16.44 | 0/0 |
| starter-candidate-rule-weave-rush | 32 | 68.8% | 81.2% | 43.8% | 18.28 | 0/0 |
| starter-candidate-rule-varied | 64 | 90.6% | 96.9% | 46.9% | 17.61 | 0/0 |
| starter-candidate-model-varied-starter | 64 | 87.5% | 90.6% | 48.4% | 18.66 | 0/0 |
| starter-candidate-model-varied-weave-rush | 64 | 93.8% | 96.9% | 48.4% | 16.73 | 0/0 |
| starter-candidate-mirror | 32 | 37.5% | 56.2% | 40.6% | 19.19 | 0/0 |
| weave-rush-candidate-rule-starter | 32 | 100.0% | 100.0% | 50.0% | 12.09 | 0/0 |
| weave-rush-candidate-rule-weave-rush | 32 | 100.0% | 81.2% | 59.4% | 15.00 | 0/0 |
| weave-rush-candidate-rule-varied | 64 | 93.8% | 100.0% | 46.9% | 13.19 | 0/0 |
| weave-rush-candidate-model-varied-starter | 64 | 96.9% | 100.0% | 48.4% | 13.92 | 0/0 |
| weave-rush-candidate-model-varied-weave-rush | 64 | 100.0% | 100.0% | 50.0% | 12.23 | 0/0 |
| weave-rush-candidate-mirror | 32 | 75.0% | 50.0% | 62.5% | 15.91 | 0/0 |
| cross-candidate | 32 | 37.5% | 31.2% | 53.1% | 17.34 | 0/0 |
| cross-candidate（覆纹快攻视角，同批对局） | 32 | 68.8% | 62.5% | 53.1% | 17.34 | 同上，勿重复计局 |
| starter-candidate-old-policy | 32 | 100.0% | 87.5% | 56.2% | 19.16 | 0/0 |
| starter-candidate-new-policy | 32 | 93.8% | 100.0% | 46.9% | 17.56 | 0/0 |
| weave-rush-candidate-old-policy | 32 | 93.8% | 93.8% | 50.0% | 13.78 | 0/0 |
| weave-rush-candidate-new-policy | 32 | 100.0% | 93.8% | 53.1% | 13.72 | 0/0 |

交叉互搏的首行以创生为学习方；第二行换算覆纹快攻视角，来自同批对局。

## 构筑调整与理由

下表只依据独立于最终测试集的筛选样本；小样本用于验证选择流程，不能据此声称最优。候选生成过程是配牌策略采样，不伪造逐张换牌的模型内心理由。

### starter

- 原始筛选胜率：74.4%；候选：78.9%；每种 512 场。
- 选择：candidate；理由：Four-scenario paired local search; one-copy exhaustive, two-copy sampled。
- 全批候选及原始评分：`starter/cards/candidate-history.jsonl`。
- 筛选证据：`starter/candidate-selection.json`；最差阵容胜率：0.71484375；最弱位置胜率：0.671875。
- 候选适应训练单独完成；本表筛选使用冻结基础策略，最终矩阵另外展示适应后效果。

| 卡牌 | 原始张数 | 候选张数 |
| --- | ---: | ---: |
| J01 知晓与制衡 | 2 | 1 |
| J03 优势初成 | 0 | 1 |
| J07 使命必达 | 0 | 1 |
| J08 现实避难所 | 2 | 1 |
| N07 心有猛虎 | 2 | 1 |
| N08 预备备 | 0 | 1 |
| Y03 策略特性：善良 | 2 | 1 |
| Y07 想做什么梦？ | 0 | 2 |
| Y08 错误的门 | 2 | 1 |
| Z02 探悉天职 | 1 | 0 |
| Z03 奇异记叙 | 0 | 2 |
| Z04 未决迷数 | 2 | 1 |
| Z07 休息日 | 2 | 1 |
| Z08 倾世之雨 | 1 | 2 |

#### 分级候选比较（筛选集，不是独立留出）

完成：True；选择：local-25c08f389eee；Four-scenario paired local search; one-copy exhaustive, two-copy sampled。


局部搜索换牌过程（得分仅代表同轮共同筛选集）：

| 轮次 | 换出 | 换入 | 父配牌得分 | 选中得分 |
| --- | --- | --- | ---: | ---: |
| 1 | {'Y03': 1, 'J03': 1} | {'Y02': 1, 'J05': 1} | 77.93% | 79.10% |
| 2 | {'N08': 1, 'J05': 1} | {'N07': 1, 'J01': 1} | 78.32% | 79.10% |
| 3 | {} | {} | 79.10% | 79.10% |
| 4 | {'Z04': 1, 'J08': 1} | {'Z07': 1, 'J07': 1} | 77.73% | 78.91% |

局部初筛仅展示前 8 及基线；完整候选、父配牌和计数见 JSON。

| 层级 | 候选 | 对手阵容 | 位置 | 胜/完成 | 截断 |
| --- | --- | --- | --- | --- | ---: |
| 0 | original | starter | first | 6/8 | 0 |
| 0 | original | starter | second | 6/8 | 0 |
| 0 | original | weave-rush | first | 5/8 | 0 |
| 0 | original | weave-rush | second | 7/8 | 0 |
| 0 | serving | starter | first | 5/8 | 0 |
| 0 | serving | starter | second | 7/8 | 0 |
| 0 | serving | weave-rush | first | 6/8 | 0 |
| 0 | serving | weave-rush | second | 5/8 | 0 |
| 0 | seed-0 | starter | first | 7/8 | 0 |
| 0 | seed-0 | starter | second | 6/8 | 0 |
| 0 | seed-0 | weave-rush | first | 7/8 | 0 |
| 0 | seed-0 | weave-rush | second | 6/8 | 0 |
| 0 | last | starter | first | 6/8 | 0 |
| 0 | last | starter | second | 7/8 | 0 |
| 0 | last | weave-rush | first | 4/8 | 0 |
| 0 | last | weave-rush | second | 6/8 | 0 |
| 0 | tuned-50e7ff745fdb | starter | first | 6/8 | 0 |
| 0 | tuned-50e7ff745fdb | starter | second | 7/8 | 0 |
| 0 | tuned-50e7ff745fdb | weave-rush | first | 6/8 | 0 |
| 0 | tuned-50e7ff745fdb | weave-rush | second | 6/8 | 0 |
| 0 | tuned-bc7a897ad958 | starter | first | 7/8 | 0 |
| 0 | tuned-bc7a897ad958 | starter | second | 6/8 | 0 |
| 0 | tuned-bc7a897ad958 | weave-rush | first | 7/8 | 0 |
| 0 | tuned-bc7a897ad958 | weave-rush | second | 8/8 | 0 |
| 0 | tuned-dbef77e4755e | starter | first | 6/8 | 0 |
| 0 | tuned-dbef77e4755e | starter | second | 7/8 | 0 |
| 0 | tuned-dbef77e4755e | weave-rush | first | 7/8 | 0 |
| 0 | tuned-dbef77e4755e | weave-rush | second | 7/8 | 0 |
| 0 | tuned-1bfd06adcb43 | starter | first | 6/8 | 0 |
| 0 | tuned-1bfd06adcb43 | starter | second | 6/8 | 0 |
| 0 | tuned-1bfd06adcb43 | weave-rush | first | 6/8 | 0 |
| 0 | tuned-1bfd06adcb43 | weave-rush | second | 5/8 | 0 |
| 1 | original | starter | first | 22/32 | 0 |
| 1 | original | starter | second | 23/32 | 0 |
| 1 | original | weave-rush | first | 19/32 | 0 |
| 1 | original | weave-rush | second | 19/32 | 0 |
| 1 | tuned-bc7a897ad958 | starter | first | 25/32 | 0 |
| 1 | tuned-bc7a897ad958 | starter | second | 25/32 | 0 |
| 1 | tuned-bc7a897ad958 | weave-rush | first | 24/32 | 0 |
| 1 | tuned-bc7a897ad958 | weave-rush | second | 22/32 | 0 |
| 1 | tuned-dbef77e4755e | starter | first | 27/32 | 0 |
| 1 | tuned-dbef77e4755e | starter | second | 28/32 | 0 |
| 1 | tuned-dbef77e4755e | weave-rush | first | 23/32 | 0 |
| 1 | tuned-dbef77e4755e | weave-rush | second | 25/32 | 0 |
| 1 | seed-0 | starter | first | 25/32 | 0 |
| 1 | seed-0 | starter | second | 27/32 | 0 |
| 1 | seed-0 | weave-rush | first | 21/32 | 0 |
| 1 | seed-0 | weave-rush | second | 22/32 | 0 |
| 2 | original | starter | first | 103/128 | 0 |
| 2 | original | starter | second | 105/128 | 0 |
| 2 | original | weave-rush | first | 86/128 | 0 |
| 2 | original | weave-rush | second | 90/128 | 0 |
| 2 | tuned-dbef77e4755e | starter | first | 107/128 | 0 |
| 2 | tuned-dbef77e4755e | starter | second | 114/128 | 0 |
| 2 | tuned-dbef77e4755e | weave-rush | first | 86/128 | 0 |
| 2 | tuned-dbef77e4755e | weave-rush | second | 89/128 | 0 |
| local-1-screen | original | starter | first | 5/8 | 0 |
| local-1-screen | original | starter | second | 7/8 | 0 |
| local-1-screen | original | weave-rush | first | 5/8 | 0 |
| local-1-screen | original | weave-rush | second | 5/8 | 0 |
| local-1-screen | local-0f5f2b79fd4c | starter | first | 7/8 | 0 |
| local-1-screen | local-0f5f2b79fd4c | starter | second | 8/8 | 0 |
| local-1-screen | local-0f5f2b79fd4c | weave-rush | first | 8/8 | 0 |
| local-1-screen | local-0f5f2b79fd4c | weave-rush | second | 5/8 | 0 |
| local-1-screen | local-722a21a4ca87 | starter | first | 8/8 | 0 |
| local-1-screen | local-722a21a4ca87 | starter | second | 8/8 | 0 |
| local-1-screen | local-722a21a4ca87 | weave-rush | first | 8/8 | 0 |
| local-1-screen | local-722a21a4ca87 | weave-rush | second | 5/8 | 0 |
| local-1-screen | local-179d7436073e | starter | first | 7/8 | 0 |
| local-1-screen | local-179d7436073e | starter | second | 8/8 | 0 |
| local-1-screen | local-179d7436073e | weave-rush | first | 8/8 | 0 |
| local-1-screen | local-179d7436073e | weave-rush | second | 5/8 | 0 |
| local-1-screen | local-09eb0a3792dd | starter | first | 7/8 | 0 |
| local-1-screen | local-09eb0a3792dd | starter | second | 8/8 | 0 |
| local-1-screen | local-09eb0a3792dd | weave-rush | first | 8/8 | 0 |
| local-1-screen | local-09eb0a3792dd | weave-rush | second | 5/8 | 0 |
| local-1-screen | local-c785c387d2ba | starter | first | 8/8 | 0 |
| local-1-screen | local-c785c387d2ba | starter | second | 8/8 | 0 |
| local-1-screen | local-c785c387d2ba | weave-rush | first | 8/8 | 0 |
| local-1-screen | local-c785c387d2ba | weave-rush | second | 5/8 | 0 |
| local-1-screen | local-cc5131e654d0 | starter | first | 8/8 | 0 |
| local-1-screen | local-cc5131e654d0 | starter | second | 8/8 | 0 |
| local-1-screen | local-cc5131e654d0 | weave-rush | first | 6/8 | 0 |
| local-1-screen | local-cc5131e654d0 | weave-rush | second | 6/8 | 0 |
| local-1-screen | local-f39c25ac2f63 | starter | first | 8/8 | 0 |
| local-1-screen | local-f39c25ac2f63 | starter | second | 8/8 | 0 |
| local-1-screen | local-f39c25ac2f63 | weave-rush | first | 6/8 | 0 |
| local-1-screen | local-f39c25ac2f63 | weave-rush | second | 6/8 | 0 |
| local-1-screen | local-92dbe1d5d835 | starter | first | 8/8 | 0 |
| local-1-screen | local-92dbe1d5d835 | starter | second | 8/8 | 0 |
| local-1-screen | local-92dbe1d5d835 | weave-rush | first | 6/8 | 0 |
| local-1-screen | local-92dbe1d5d835 | weave-rush | second | 6/8 | 0 |
| local-1-final | original | starter | first | 96/128 | 0 |
| local-1-final | original | starter | second | 100/128 | 0 |
| local-1-final | original | weave-rush | first | 89/128 | 0 |
| local-1-final | original | weave-rush | second | 84/128 | 0 |
| local-1-final | tuned-dbef77e4755e | starter | first | 110/128 | 0 |
| local-1-final | tuned-dbef77e4755e | starter | second | 109/128 | 0 |
| local-1-final | tuned-dbef77e4755e | weave-rush | first | 90/128 | 0 |
| local-1-final | tuned-dbef77e4755e | weave-rush | second | 90/128 | 0 |
| local-1-final | local-722a21a4ca87 | starter | first | 110/128 | 0 |
| local-1-final | local-722a21a4ca87 | starter | second | 102/128 | 0 |
| local-1-final | local-722a21a4ca87 | weave-rush | first | 87/128 | 0 |
| local-1-final | local-722a21a4ca87 | weave-rush | second | 95/128 | 0 |
| local-1-final | local-c785c387d2ba | starter | first | 112/128 | 0 |
| local-1-final | local-c785c387d2ba | starter | second | 104/128 | 0 |
| local-1-final | local-c785c387d2ba | weave-rush | first | 88/128 | 0 |
| local-1-final | local-c785c387d2ba | weave-rush | second | 95/128 | 0 |
| local-1-final | local-06460c19f14e | starter | first | 110/128 | 0 |
| local-1-final | local-06460c19f14e | starter | second | 105/128 | 0 |
| local-1-final | local-06460c19f14e | weave-rush | first | 93/128 | 0 |
| local-1-final | local-06460c19f14e | weave-rush | second | 97/128 | 0 |
| local-1-final | local-09eb0a3792dd | starter | first | 112/128 | 0 |
| local-1-final | local-09eb0a3792dd | starter | second | 105/128 | 0 |
| local-1-final | local-09eb0a3792dd | weave-rush | first | 88/128 | 0 |
| local-1-final | local-09eb0a3792dd | weave-rush | second | 95/128 | 0 |
| local-1-final | local-0f5f2b79fd4c | starter | first | 109/128 | 0 |
| local-1-final | local-0f5f2b79fd4c | starter | second | 102/128 | 0 |
| local-1-final | local-0f5f2b79fd4c | weave-rush | first | 87/128 | 0 |
| local-1-final | local-0f5f2b79fd4c | weave-rush | second | 95/128 | 0 |
| local-1-final | local-179d7436073e | starter | first | 114/128 | 0 |
| local-1-final | local-179d7436073e | starter | second | 104/128 | 0 |
| local-1-final | local-179d7436073e | weave-rush | first | 86/128 | 0 |
| local-1-final | local-179d7436073e | weave-rush | second | 96/128 | 0 |
| local-1-final | local-92dbe1d5d835 | starter | first | 106/128 | 0 |
| local-1-final | local-92dbe1d5d835 | starter | second | 106/128 | 0 |
| local-1-final | local-92dbe1d5d835 | weave-rush | first | 84/128 | 0 |
| local-1-final | local-92dbe1d5d835 | weave-rush | second | 84/128 | 0 |
| local-1-final | local-cc5131e654d0 | starter | first | 106/128 | 0 |
| local-1-final | local-cc5131e654d0 | starter | second | 104/128 | 0 |
| local-1-final | local-cc5131e654d0 | weave-rush | first | 92/128 | 0 |
| local-1-final | local-cc5131e654d0 | weave-rush | second | 90/128 | 0 |
| local-2-screen | original | starter | first | 6/8 | 0 |
| local-2-screen | original | starter | second | 5/8 | 0 |
| local-2-screen | original | weave-rush | first | 6/8 | 0 |
| local-2-screen | original | weave-rush | second | 5/8 | 0 |
| local-2-screen | local-39f8b7f0dbc2 | starter | first | 8/8 | 0 |
| local-2-screen | local-39f8b7f0dbc2 | starter | second | 7/8 | 0 |
| local-2-screen | local-39f8b7f0dbc2 | weave-rush | first | 6/8 | 0 |
| local-2-screen | local-39f8b7f0dbc2 | weave-rush | second | 8/8 | 0 |
| local-2-screen | local-051403085756 | starter | first | 8/8 | 0 |
| local-2-screen | local-051403085756 | starter | second | 7/8 | 0 |
| local-2-screen | local-051403085756 | weave-rush | first | 6/8 | 0 |
| local-2-screen | local-051403085756 | weave-rush | second | 8/8 | 0 |
| local-2-screen | local-dcffd10e6417 | starter | first | 8/8 | 0 |
| local-2-screen | local-dcffd10e6417 | starter | second | 8/8 | 0 |
| local-2-screen | local-dcffd10e6417 | weave-rush | first | 5/8 | 0 |
| local-2-screen | local-dcffd10e6417 | weave-rush | second | 8/8 | 0 |
| local-2-screen | local-50906d5779b2 | starter | first | 8/8 | 0 |
| local-2-screen | local-50906d5779b2 | starter | second | 8/8 | 0 |
| local-2-screen | local-50906d5779b2 | weave-rush | first | 5/8 | 0 |
| local-2-screen | local-50906d5779b2 | weave-rush | second | 8/8 | 0 |
| local-2-screen | local-15ae41411775 | starter | first | 8/8 | 0 |
| local-2-screen | local-15ae41411775 | starter | second | 7/8 | 0 |
| local-2-screen | local-15ae41411775 | weave-rush | first | 6/8 | 0 |
| local-2-screen | local-15ae41411775 | weave-rush | second | 8/8 | 0 |
| local-2-screen | local-6ad435d04d8c | starter | first | 8/8 | 0 |
| local-2-screen | local-6ad435d04d8c | starter | second | 7/8 | 0 |
| local-2-screen | local-6ad435d04d8c | weave-rush | first | 6/8 | 0 |
| local-2-screen | local-6ad435d04d8c | weave-rush | second | 8/8 | 0 |
| local-2-screen | local-16cd51ab14ce | starter | first | 8/8 | 0 |
| local-2-screen | local-16cd51ab14ce | starter | second | 8/8 | 0 |
| local-2-screen | local-16cd51ab14ce | weave-rush | first | 6/8 | 0 |
| local-2-screen | local-16cd51ab14ce | weave-rush | second | 7/8 | 0 |
| local-2-screen | local-f5922a6fe20b | starter | first | 8/8 | 0 |
| local-2-screen | local-f5922a6fe20b | starter | second | 8/8 | 0 |
| local-2-screen | local-f5922a6fe20b | weave-rush | first | 6/8 | 0 |
| local-2-screen | local-f5922a6fe20b | weave-rush | second | 7/8 | 0 |
| local-2-final | original | starter | first | 103/128 | 0 |
| local-2-final | original | starter | second | 108/128 | 0 |
| local-2-final | original | weave-rush | first | 94/128 | 0 |
| local-2-final | original | weave-rush | second | 93/128 | 0 |
| local-2-final | local-06460c19f14e | starter | first | 104/128 | 0 |
| local-2-final | local-06460c19f14e | starter | second | 113/128 | 0 |
| local-2-final | local-06460c19f14e | weave-rush | first | 87/128 | 0 |
| local-2-final | local-06460c19f14e | weave-rush | second | 97/128 | 0 |
| local-2-final | local-051403085756 | starter | first | 103/128 | 0 |
| local-2-final | local-051403085756 | starter | second | 110/128 | 0 |
| local-2-final | local-051403085756 | weave-rush | first | 84/128 | 0 |
| local-2-final | local-051403085756 | weave-rush | second | 91/128 | 0 |
| local-2-final | local-15ae41411775 | starter | first | 108/128 | 0 |
| local-2-final | local-15ae41411775 | starter | second | 113/128 | 0 |
| local-2-final | local-15ae41411775 | weave-rush | first | 86/128 | 0 |
| local-2-final | local-15ae41411775 | weave-rush | second | 98/128 | 0 |
| local-2-final | local-16cd51ab14ce | starter | first | 103/128 | 0 |
| local-2-final | local-16cd51ab14ce | starter | second | 105/128 | 0 |
| local-2-final | local-16cd51ab14ce | weave-rush | first | 79/128 | 0 |
| local-2-final | local-16cd51ab14ce | weave-rush | second | 97/128 | 0 |
| local-2-final | local-39f8b7f0dbc2 | starter | first | 104/128 | 0 |
| local-2-final | local-39f8b7f0dbc2 | starter | second | 109/128 | 0 |
| local-2-final | local-39f8b7f0dbc2 | weave-rush | first | 81/128 | 0 |
| local-2-final | local-39f8b7f0dbc2 | weave-rush | second | 93/128 | 0 |
| local-2-final | local-50906d5779b2 | starter | first | 105/128 | 0 |
| local-2-final | local-50906d5779b2 | starter | second | 106/128 | 0 |
| local-2-final | local-50906d5779b2 | weave-rush | first | 81/128 | 0 |
| local-2-final | local-50906d5779b2 | weave-rush | second | 95/128 | 0 |
| local-2-final | local-6ad435d04d8c | starter | first | 104/128 | 0 |
| local-2-final | local-6ad435d04d8c | starter | second | 103/128 | 0 |
| local-2-final | local-6ad435d04d8c | weave-rush | first | 76/128 | 0 |
| local-2-final | local-6ad435d04d8c | weave-rush | second | 95/128 | 0 |
| local-2-final | local-dcffd10e6417 | starter | first | 102/128 | 0 |
| local-2-final | local-dcffd10e6417 | starter | second | 113/128 | 0 |
| local-2-final | local-dcffd10e6417 | weave-rush | first | 89/128 | 0 |
| local-2-final | local-dcffd10e6417 | weave-rush | second | 96/128 | 0 |
| local-2-final | local-f5922a6fe20b | starter | first | 102/128 | 0 |
| local-2-final | local-f5922a6fe20b | starter | second | 100/128 | 0 |
| local-2-final | local-f5922a6fe20b | weave-rush | first | 79/128 | 0 |
| local-2-final | local-f5922a6fe20b | weave-rush | second | 94/128 | 0 |
| local-3-screen | original | starter | first | 6/8 | 0 |
| local-3-screen | original | starter | second | 6/8 | 0 |
| local-3-screen | original | weave-rush | first | 6/8 | 0 |
| local-3-screen | original | weave-rush | second | 5/8 | 0 |
| local-3-screen | local-69b86343a744 | starter | first | 7/8 | 0 |
| local-3-screen | local-69b86343a744 | starter | second | 7/8 | 0 |
| local-3-screen | local-69b86343a744 | weave-rush | first | 5/8 | 0 |
| local-3-screen | local-69b86343a744 | weave-rush | second | 8/8 | 0 |
| local-3-screen | local-167daef5b964 | starter | first | 8/8 | 0 |
| local-3-screen | local-167daef5b964 | starter | second | 8/8 | 0 |
| local-3-screen | local-167daef5b964 | weave-rush | first | 5/8 | 0 |
| local-3-screen | local-167daef5b964 | weave-rush | second | 6/8 | 0 |
| local-3-screen | local-d60d56b5121e | starter | first | 7/8 | 0 |
| local-3-screen | local-d60d56b5121e | starter | second | 8/8 | 0 |
| local-3-screen | local-d60d56b5121e | weave-rush | first | 6/8 | 0 |
| local-3-screen | local-d60d56b5121e | weave-rush | second | 6/8 | 0 |
| local-3-screen | local-cc17b5a4184e | starter | first | 8/8 | 0 |
| local-3-screen | local-cc17b5a4184e | starter | second | 8/8 | 0 |
| local-3-screen | local-cc17b5a4184e | weave-rush | first | 5/8 | 0 |
| local-3-screen | local-cc17b5a4184e | weave-rush | second | 6/8 | 0 |
| local-3-screen | local-f1ee812eda54 | starter | first | 8/8 | 0 |
| local-3-screen | local-f1ee812eda54 | starter | second | 8/8 | 0 |
| local-3-screen | local-f1ee812eda54 | weave-rush | first | 5/8 | 0 |
| local-3-screen | local-f1ee812eda54 | weave-rush | second | 6/8 | 0 |
| local-3-screen | local-3df7f71666b2 | starter | first | 8/8 | 0 |
| local-3-screen | local-3df7f71666b2 | starter | second | 8/8 | 0 |
| local-3-screen | local-3df7f71666b2 | weave-rush | first | 5/8 | 0 |
| local-3-screen | local-3df7f71666b2 | weave-rush | second | 6/8 | 0 |
| local-3-screen | local-76de34b3cbf3 | starter | first | 7/8 | 0 |
| local-3-screen | local-76de34b3cbf3 | starter | second | 7/8 | 0 |
| local-3-screen | local-76de34b3cbf3 | weave-rush | first | 5/8 | 0 |
| local-3-screen | local-76de34b3cbf3 | weave-rush | second | 8/8 | 0 |
| local-3-screen | local-e6894f51216d | starter | first | 6/8 | 0 |
| local-3-screen | local-e6894f51216d | starter | second | 7/8 | 0 |
| local-3-screen | local-e6894f51216d | weave-rush | first | 7/8 | 0 |
| local-3-screen | local-e6894f51216d | weave-rush | second | 8/8 | 0 |
| local-3-final | original | starter | first | 104/128 | 0 |
| local-3-final | original | starter | second | 106/128 | 0 |
| local-3-final | original | weave-rush | first | 79/128 | 0 |
| local-3-final | original | weave-rush | second | 91/128 | 0 |
| local-3-final | local-15ae41411775 | starter | first | 105/128 | 0 |
| local-3-final | local-15ae41411775 | starter | second | 116/128 | 0 |
| local-3-final | local-15ae41411775 | weave-rush | first | 92/128 | 0 |
| local-3-final | local-15ae41411775 | weave-rush | second | 92/128 | 0 |
| local-3-final | local-e6894f51216d | starter | first | 104/128 | 0 |
| local-3-final | local-e6894f51216d | starter | second | 111/128 | 0 |
| local-3-final | local-e6894f51216d | weave-rush | first | 89/128 | 0 |
| local-3-final | local-e6894f51216d | weave-rush | second | 95/128 | 0 |
| local-3-final | local-140bed5ad388 | starter | first | 102/128 | 0 |
| local-3-final | local-140bed5ad388 | starter | second | 111/128 | 0 |
| local-3-final | local-140bed5ad388 | weave-rush | first | 87/128 | 0 |
| local-3-final | local-140bed5ad388 | weave-rush | second | 88/128 | 0 |
| local-3-final | local-167daef5b964 | starter | first | 106/128 | 0 |
| local-3-final | local-167daef5b964 | starter | second | 112/128 | 0 |
| local-3-final | local-167daef5b964 | weave-rush | first | 87/128 | 0 |
| local-3-final | local-167daef5b964 | weave-rush | second | 93/128 | 0 |
| local-3-final | local-3df7f71666b2 | starter | first | 108/128 | 0 |
| local-3-final | local-3df7f71666b2 | starter | second | 115/128 | 0 |
| local-3-final | local-3df7f71666b2 | weave-rush | first | 92/128 | 0 |
| local-3-final | local-3df7f71666b2 | weave-rush | second | 88/128 | 0 |
| local-3-final | local-414572849f6d | starter | first | 107/128 | 0 |
| local-3-final | local-414572849f6d | starter | second | 108/128 | 0 |
| local-3-final | local-414572849f6d | weave-rush | first | 94/128 | 0 |
| local-3-final | local-414572849f6d | weave-rush | second | 91/128 | 0 |
| local-3-final | local-59497ecab2d1 | starter | first | 106/128 | 0 |
| local-3-final | local-59497ecab2d1 | starter | second | 116/128 | 0 |
| local-3-final | local-59497ecab2d1 | weave-rush | first | 90/128 | 0 |
| local-3-final | local-59497ecab2d1 | weave-rush | second | 89/128 | 0 |
| local-3-final | local-6986484ee84f | starter | first | 106/128 | 0 |
| local-3-final | local-6986484ee84f | starter | second | 114/128 | 0 |
| local-3-final | local-6986484ee84f | weave-rush | first | 86/128 | 0 |
| local-3-final | local-6986484ee84f | weave-rush | second | 94/128 | 0 |
| local-3-final | local-69b86343a744 | starter | first | 103/128 | 0 |
| local-3-final | local-69b86343a744 | starter | second | 110/128 | 0 |
| local-3-final | local-69b86343a744 | weave-rush | first | 81/128 | 0 |
| local-3-final | local-69b86343a744 | weave-rush | second | 89/128 | 0 |
| local-4-screen | original | starter | first | 7/8 | 0 |
| local-4-screen | original | starter | second | 7/8 | 0 |
| local-4-screen | original | weave-rush | first | 4/8 | 0 |
| local-4-screen | original | weave-rush | second | 6/8 | 0 |
| local-4-screen | local-ef3877a4b9d1 | starter | first | 8/8 | 0 |
| local-4-screen | local-ef3877a4b9d1 | starter | second | 7/8 | 0 |
| local-4-screen | local-ef3877a4b9d1 | weave-rush | first | 5/8 | 0 |
| local-4-screen | local-ef3877a4b9d1 | weave-rush | second | 7/8 | 0 |
| local-4-screen | local-2a966e15a84b | starter | first | 8/8 | 0 |
| local-4-screen | local-2a966e15a84b | starter | second | 7/8 | 0 |
| local-4-screen | local-2a966e15a84b | weave-rush | first | 5/8 | 0 |
| local-4-screen | local-2a966e15a84b | weave-rush | second | 7/8 | 0 |
| local-4-screen | local-d60d56b5121e | starter | first | 8/8 | 0 |
| local-4-screen | local-d60d56b5121e | starter | second | 7/8 | 0 |
| local-4-screen | local-d60d56b5121e | weave-rush | first | 5/8 | 0 |
| local-4-screen | local-d60d56b5121e | weave-rush | second | 7/8 | 0 |
| local-4-screen | local-cc17b5a4184e | starter | first | 8/8 | 0 |
| local-4-screen | local-cc17b5a4184e | starter | second | 7/8 | 0 |
| local-4-screen | local-cc17b5a4184e | weave-rush | first | 5/8 | 0 |
| local-4-screen | local-cc17b5a4184e | weave-rush | second | 7/8 | 0 |
| local-4-screen | local-4696a83d1228 | starter | first | 7/8 | 0 |
| local-4-screen | local-4696a83d1228 | starter | second | 8/8 | 0 |
| local-4-screen | local-4696a83d1228 | weave-rush | first | 5/8 | 0 |
| local-4-screen | local-4696a83d1228 | weave-rush | second | 8/8 | 0 |
| local-4-screen | local-c8bb45dbd96b | starter | first | 7/8 | 0 |
| local-4-screen | local-c8bb45dbd96b | starter | second | 8/8 | 0 |
| local-4-screen | local-c8bb45dbd96b | weave-rush | first | 5/8 | 0 |
| local-4-screen | local-c8bb45dbd96b | weave-rush | second | 8/8 | 0 |
| local-4-screen | local-25c08f389eee | starter | first | 8/8 | 0 |
| local-4-screen | local-25c08f389eee | starter | second | 8/8 | 0 |
| local-4-screen | local-25c08f389eee | weave-rush | first | 6/8 | 0 |
| local-4-screen | local-25c08f389eee | weave-rush | second | 6/8 | 0 |
| local-4-screen | local-f9cb473b340c | starter | first | 7/8 | 0 |
| local-4-screen | local-f9cb473b340c | starter | second | 8/8 | 0 |
| local-4-screen | local-f9cb473b340c | weave-rush | first | 5/8 | 0 |
| local-4-screen | local-f9cb473b340c | weave-rush | second | 8/8 | 0 |
| local-4-final | original | starter | first | 99/128 | 0 |
| local-4-final | original | starter | second | 102/128 | 0 |
| local-4-final | original | weave-rush | first | 88/128 | 0 |
| local-4-final | original | weave-rush | second | 92/128 | 0 |
| local-4-final | local-15ae41411775 | starter | first | 109/128 | 0 |
| local-4-final | local-15ae41411775 | starter | second | 105/128 | 0 |
| local-4-final | local-15ae41411775 | weave-rush | first | 88/128 | 0 |
| local-4-final | local-15ae41411775 | weave-rush | second | 96/128 | 0 |
| local-4-final | local-25c08f389eee | starter | first | 109/128 | 0 |
| local-4-final | local-25c08f389eee | starter | second | 112/128 | 0 |
| local-4-final | local-25c08f389eee | weave-rush | first | 86/128 | 0 |
| local-4-final | local-25c08f389eee | weave-rush | second | 97/128 | 0 |
| local-4-final | local-4696a83d1228 | starter | first | 105/128 | 0 |
| local-4-final | local-4696a83d1228 | starter | second | 105/128 | 0 |
| local-4-final | local-4696a83d1228 | weave-rush | first | 84/128 | 0 |
| local-4-final | local-4696a83d1228 | weave-rush | second | 86/128 | 0 |
| local-4-final | local-c8bb45dbd96b | starter | first | 107/128 | 0 |
| local-4-final | local-c8bb45dbd96b | starter | second | 105/128 | 0 |
| local-4-final | local-c8bb45dbd96b | weave-rush | first | 81/128 | 0 |
| local-4-final | local-c8bb45dbd96b | weave-rush | second | 88/128 | 0 |
| local-4-final | local-f9cb473b340c | starter | first | 109/128 | 0 |
| local-4-final | local-f9cb473b340c | starter | second | 102/128 | 0 |
| local-4-final | local-f9cb473b340c | weave-rush | first | 82/128 | 0 |
| local-4-final | local-f9cb473b340c | weave-rush | second | 87/128 | 0 |
| local-4-final | local-033216c87587 | starter | first | 110/128 | 0 |
| local-4-final | local-033216c87587 | starter | second | 105/128 | 0 |
| local-4-final | local-033216c87587 | weave-rush | first | 91/128 | 0 |
| local-4-final | local-033216c87587 | weave-rush | second | 96/128 | 0 |
| local-4-final | local-07b1515204de | starter | first | 109/128 | 0 |
| local-4-final | local-07b1515204de | starter | second | 102/128 | 0 |
| local-4-final | local-07b1515204de | weave-rush | first | 85/128 | 0 |
| local-4-final | local-07b1515204de | weave-rush | second | 88/128 | 0 |
| local-4-final | local-11337c7b39e2 | starter | first | 101/128 | 0 |
| local-4-final | local-11337c7b39e2 | starter | second | 106/128 | 0 |
| local-4-final | local-11337c7b39e2 | weave-rush | first | 89/128 | 0 |
| local-4-final | local-11337c7b39e2 | weave-rush | second | 92/128 | 0 |
| local-4-final | local-2a966e15a84b | starter | first | 108/128 | 0 |
| local-4-final | local-2a966e15a84b | starter | second | 110/128 | 0 |
| local-4-final | local-2a966e15a84b | weave-rush | first | 86/128 | 0 |
| local-4-final | local-2a966e15a84b | weave-rush | second | 93/128 | 0 |

模型身份与对手实际配牌分别记录在 candidate-selection.json；编译配对采用偶/奇种子，不声称牌序完全相同。

本表仅列 12 项代表记录；全部 13536 项保存在 candidate-decisions.json。

| 采样版本 | 候选 | 先手胜/完成 | 后手胜/完成 | 筛选分 | 处理 |
| --- | --- | --- | --- | ---: | --- |
| 832 | tuned-c9fb2e8e9917 | 20/20 | 19/20 | 0.9375 | 本批未选中；不据此认定构筑更差 |
| 814 | tuned-dbef77e4755e | 19/20 | 20/20 | 0.9375 | 本批未选中；不据此认定构筑更差 |
| 799 | tuned-bc7a897ad958 | 19/20 | 20/20 | 0.9375 | 本批未选中；不据此认定构筑更差 |
| 778 | tuned-0de241dfb576 | 19/20 | 19/20 | 0.9375 | 本批未选中；不据此认定构筑更差 |
| 624 | tuned-abacf34869fc | 19/20 | 20/20 | 0.9375 | 本批未选中；不据此认定构筑更差 |
| 619 | tuned-50e7ff745fdb | 19/20 | 20/20 | 0.9375 | 本批未选中；不据此认定构筑更差 |
| 838 | tuned-61d057f5f53f | 18/20 | 20/20 | 0.875 | 本批未选中；不据此认定构筑更差 |
| 820 | tuned-16babd9df9e5 | 19/20 | 19/20 | 0.875 | 本批未选中；不据此认定构筑更差 |
| 819 | tuned-a4eeac21f9a5 | 18/20 | 19/20 | 0.875 | 本批未选中；不据此认定构筑更差 |
| 810 | tuned-9e5191086360 | 19/20 | 19/20 | 0.875 | 本批未选中；不据此认定构筑更差 |
| 808 | tuned-4dc126d8d59c | 18/20 | 19/20 | 0.875 | 本批未选中；不据此认定构筑更差 |
| 782 | tuned-700f716eb6e3 | 19/20 | 19/20 | 0.875 | 本批未选中；不据此认定构筑更差 |

### weave-rush

- 原始筛选胜率：85.7%；候选：91.2%；每种 512 场。
- 选择：candidate；理由：Four-scenario paired local search; one-copy exhaustive, two-copy sampled。
- 全批候选及原始评分：`weave-rush/cards/candidate-history.jsonl`。
- 筛选证据：`weave-rush/candidate-selection.json`；最差阵容胜率：0.87109375；最弱位置胜率：0.8515625。
- 候选适应训练单独完成；本表筛选使用冻结基础策略，最终矩阵另外展示适应后效果。

| 卡牌 | 原始张数 | 候选张数 |
| --- | ---: | ---: |
| B07 判予秋 | 0 | 2 |
| B08 茶花会 | 2 | 0 |
| M02 二次补考 | 0 | 1 |
| M08 开始净空 | 2 | 1 |
| Z05 默示赋命 | 2 | 0 |
| Z07 休息日 | 0 | 2 |

#### 分级候选比较（筛选集，不是独立留出）

完成：True；选择：local-7987ad220ccc；Four-scenario paired local search; one-copy exhaustive, two-copy sampled。


局部搜索换牌过程（得分仅代表同轮共同筛选集）：

| 轮次 | 换出 | 换入 | 父配牌得分 | 选中得分 |
| --- | --- | --- | ---: | ---: |
| 1 | {'M02': 1} | {'M08': 1} | 88.28% | 89.06% |
| 2 | {} | {} | 88.87% | 88.87% |
| 3 | {} | {} | 90.62% | 90.62% |
| 4 | {} | {} | 91.21% | 91.21% |

局部初筛仅展示前 8 及基线；完整候选、父配牌和计数见 JSON。

| 层级 | 候选 | 对手阵容 | 位置 | 胜/完成 | 截断 |
| --- | --- | --- | --- | --- | ---: |
| 0 | original | starter | first | 6/8 | 0 |
| 0 | original | starter | second | 8/8 | 0 |
| 0 | original | weave-rush | first | 8/8 | 0 |
| 0 | original | weave-rush | second | 7/8 | 0 |
| 0 | seed-0 | starter | first | 6/8 | 0 |
| 0 | seed-0 | starter | second | 7/8 | 0 |
| 0 | seed-0 | weave-rush | first | 7/8 | 0 |
| 0 | seed-0 | weave-rush | second | 6/8 | 0 |
| 0 | last | starter | first | 7/8 | 0 |
| 0 | last | starter | second | 8/8 | 0 |
| 0 | last | weave-rush | first | 8/8 | 0 |
| 0 | last | weave-rush | second | 6/8 | 0 |
| 0 | tuned-002d4fbab4e5 | starter | first | 8/8 | 0 |
| 0 | tuned-002d4fbab4e5 | starter | second | 5/8 | 0 |
| 0 | tuned-002d4fbab4e5 | weave-rush | first | 6/8 | 0 |
| 0 | tuned-002d4fbab4e5 | weave-rush | second | 7/8 | 0 |
| 0 | tuned-01bc31b3b6fb | starter | first | 6/8 | 0 |
| 0 | tuned-01bc31b3b6fb | starter | second | 6/8 | 0 |
| 0 | tuned-01bc31b3b6fb | weave-rush | first | 7/8 | 0 |
| 0 | tuned-01bc31b3b6fb | weave-rush | second | 6/8 | 0 |
| 0 | tuned-0316d984f154 | starter | first | 7/8 | 0 |
| 0 | tuned-0316d984f154 | starter | second | 8/8 | 0 |
| 0 | tuned-0316d984f154 | weave-rush | first | 8/8 | 0 |
| 0 | tuned-0316d984f154 | weave-rush | second | 7/8 | 0 |
| 0 | tuned-03fe973deeb9 | starter | first | 7/8 | 0 |
| 0 | tuned-03fe973deeb9 | starter | second | 8/8 | 0 |
| 0 | tuned-03fe973deeb9 | weave-rush | first | 7/8 | 0 |
| 0 | tuned-03fe973deeb9 | weave-rush | second | 8/8 | 0 |
| 0 | tuned-05b7b3c6703e | starter | first | 7/8 | 0 |
| 0 | tuned-05b7b3c6703e | starter | second | 7/8 | 0 |
| 0 | tuned-05b7b3c6703e | weave-rush | first | 6/8 | 0 |
| 0 | tuned-05b7b3c6703e | weave-rush | second | 7/8 | 0 |
| 1 | original | starter | first | 28/32 | 0 |
| 1 | original | starter | second | 28/32 | 0 |
| 1 | original | weave-rush | first | 26/32 | 0 |
| 1 | original | weave-rush | second | 27/32 | 0 |
| 1 | tuned-0316d984f154 | starter | first | 27/32 | 0 |
| 1 | tuned-0316d984f154 | starter | second | 29/32 | 0 |
| 1 | tuned-0316d984f154 | weave-rush | first | 25/32 | 0 |
| 1 | tuned-0316d984f154 | weave-rush | second | 28/32 | 0 |
| 1 | tuned-03fe973deeb9 | starter | first | 29/32 | 0 |
| 1 | tuned-03fe973deeb9 | starter | second | 30/32 | 0 |
| 1 | tuned-03fe973deeb9 | weave-rush | first | 24/32 | 0 |
| 1 | tuned-03fe973deeb9 | weave-rush | second | 26/32 | 0 |
| 1 | last | starter | first | 31/32 | 0 |
| 1 | last | starter | second | 29/32 | 0 |
| 1 | last | weave-rush | first | 24/32 | 0 |
| 1 | last | weave-rush | second | 26/32 | 0 |
| 2 | original | starter | first | 115/128 | 0 |
| 2 | original | starter | second | 112/128 | 0 |
| 2 | original | weave-rush | first | 102/128 | 0 |
| 2 | original | weave-rush | second | 104/128 | 0 |
| 2 | last | starter | first | 119/128 | 0 |
| 2 | last | starter | second | 118/128 | 0 |
| 2 | last | weave-rush | first | 107/128 | 0 |
| 2 | last | weave-rush | second | 110/128 | 0 |
| local-1-screen | original | starter | first | 8/8 | 0 |
| local-1-screen | original | starter | second | 7/8 | 0 |
| local-1-screen | original | weave-rush | first | 7/8 | 0 |
| local-1-screen | original | weave-rush | second | 6/8 | 0 |
| local-1-screen | local-7987ad220ccc | starter | first | 8/8 | 0 |
| local-1-screen | local-7987ad220ccc | starter | second | 8/8 | 0 |
| local-1-screen | local-7987ad220ccc | weave-rush | first | 8/8 | 0 |
| local-1-screen | local-7987ad220ccc | weave-rush | second | 7/8 | 0 |
| local-1-screen | local-f4e9694a1676 | starter | first | 8/8 | 0 |
| local-1-screen | local-f4e9694a1676 | starter | second | 8/8 | 0 |
| local-1-screen | local-f4e9694a1676 | weave-rush | first | 8/8 | 0 |
| local-1-screen | local-f4e9694a1676 | weave-rush | second | 7/8 | 0 |
| local-1-screen | local-fd693c9b33f3 | starter | first | 8/8 | 0 |
| local-1-screen | local-fd693c9b33f3 | starter | second | 8/8 | 0 |
| local-1-screen | local-fd693c9b33f3 | weave-rush | first | 8/8 | 0 |
| local-1-screen | local-fd693c9b33f3 | weave-rush | second | 7/8 | 0 |
| local-1-screen | local-be10fe349549 | starter | first | 8/8 | 0 |
| local-1-screen | local-be10fe349549 | starter | second | 8/8 | 0 |
| local-1-screen | local-be10fe349549 | weave-rush | first | 8/8 | 0 |
| local-1-screen | local-be10fe349549 | weave-rush | second | 7/8 | 0 |
| local-1-screen | local-80995de8970d | starter | first | 8/8 | 0 |
| local-1-screen | local-80995de8970d | starter | second | 8/8 | 0 |
| local-1-screen | local-80995de8970d | weave-rush | first | 8/8 | 0 |
| local-1-screen | local-80995de8970d | weave-rush | second | 7/8 | 0 |
| local-1-screen | local-0fff56f8ad07 | starter | first | 8/8 | 0 |
| local-1-screen | local-0fff56f8ad07 | starter | second | 8/8 | 0 |
| local-1-screen | local-0fff56f8ad07 | weave-rush | first | 8/8 | 0 |
| local-1-screen | local-0fff56f8ad07 | weave-rush | second | 7/8 | 0 |
| local-1-screen | local-ca32f7ec0c5a | starter | first | 8/8 | 0 |
| local-1-screen | local-ca32f7ec0c5a | starter | second | 8/8 | 0 |
| local-1-screen | local-ca32f7ec0c5a | weave-rush | first | 8/8 | 0 |
| local-1-screen | local-ca32f7ec0c5a | weave-rush | second | 7/8 | 0 |
| local-1-screen | local-5480b9070eb7 | starter | first | 8/8 | 0 |
| local-1-screen | local-5480b9070eb7 | starter | second | 8/8 | 0 |
| local-1-screen | local-5480b9070eb7 | weave-rush | first | 8/8 | 0 |
| local-1-screen | local-5480b9070eb7 | weave-rush | second | 8/8 | 0 |
| local-1-final | original | starter | first | 114/128 | 0 |
| local-1-final | original | starter | second | 114/128 | 0 |
| local-1-final | original | weave-rush | first | 101/128 | 0 |
| local-1-final | original | weave-rush | second | 107/128 | 0 |
| local-1-final | tuned-c3defc048fb0 | starter | first | 117/128 | 0 |
| local-1-final | tuned-c3defc048fb0 | starter | second | 116/128 | 0 |
| local-1-final | tuned-c3defc048fb0 | weave-rush | first | 107/128 | 0 |
| local-1-final | tuned-c3defc048fb0 | weave-rush | second | 112/128 | 0 |
| local-1-final | local-5480b9070eb7 | starter | first | 119/128 | 0 |
| local-1-final | local-5480b9070eb7 | starter | second | 120/128 | 0 |
| local-1-final | local-5480b9070eb7 | weave-rush | first | 106/128 | 0 |
| local-1-final | local-5480b9070eb7 | weave-rush | second | 110/128 | 0 |
| local-1-final | local-0fff56f8ad07 | starter | first | 118/128 | 0 |
| local-1-final | local-0fff56f8ad07 | starter | second | 119/128 | 0 |
| local-1-final | local-0fff56f8ad07 | weave-rush | first | 96/128 | 0 |
| local-1-final | local-0fff56f8ad07 | weave-rush | second | 104/128 | 0 |
| local-1-final | local-23554017e781 | starter | first | 116/128 | 0 |
| local-1-final | local-23554017e781 | starter | second | 118/128 | 0 |
| local-1-final | local-23554017e781 | weave-rush | first | 97/128 | 0 |
| local-1-final | local-23554017e781 | weave-rush | second | 106/128 | 0 |
| local-1-final | local-37f620a74923 | starter | first | 117/128 | 0 |
| local-1-final | local-37f620a74923 | starter | second | 111/128 | 0 |
| local-1-final | local-37f620a74923 | weave-rush | first | 96/128 | 0 |
| local-1-final | local-37f620a74923 | weave-rush | second | 103/128 | 0 |
| local-1-final | local-7986ca6d4f39 | starter | first | 115/128 | 0 |
| local-1-final | local-7986ca6d4f39 | starter | second | 122/128 | 0 |
| local-1-final | local-7986ca6d4f39 | weave-rush | first | 108/128 | 0 |
| local-1-final | local-7986ca6d4f39 | weave-rush | second | 109/128 | 0 |
| local-1-final | local-7987ad220ccc | starter | first | 120/128 | 0 |
| local-1-final | local-7987ad220ccc | starter | second | 122/128 | 0 |
| local-1-final | local-7987ad220ccc | weave-rush | first | 106/128 | 0 |
| local-1-final | local-7987ad220ccc | weave-rush | second | 108/128 | 0 |
| local-1-final | local-80995de8970d | starter | first | 120/128 | 0 |
| local-1-final | local-80995de8970d | starter | second | 120/128 | 0 |
| local-1-final | local-80995de8970d | weave-rush | first | 106/128 | 0 |
| local-1-final | local-80995de8970d | weave-rush | second | 103/128 | 0 |
| local-1-final | local-99dc3802517b | starter | first | 116/128 | 0 |
| local-1-final | local-99dc3802517b | starter | second | 121/128 | 0 |
| local-1-final | local-99dc3802517b | weave-rush | first | 104/128 | 0 |
| local-1-final | local-99dc3802517b | weave-rush | second | 109/128 | 0 |
| local-2-screen | original | starter | first | 7/8 | 0 |
| local-2-screen | original | starter | second | 7/8 | 0 |
| local-2-screen | original | weave-rush | first | 7/8 | 0 |
| local-2-screen | original | weave-rush | second | 4/8 | 0 |
| local-2-screen | local-d4d1980d2ba3 | starter | first | 8/8 | 0 |
| local-2-screen | local-d4d1980d2ba3 | starter | second | 8/8 | 0 |
| local-2-screen | local-d4d1980d2ba3 | weave-rush | first | 8/8 | 0 |
| local-2-screen | local-d4d1980d2ba3 | weave-rush | second | 8/8 | 0 |
| local-2-screen | local-6fe6434487f0 | starter | first | 7/8 | 0 |
| local-2-screen | local-6fe6434487f0 | starter | second | 7/8 | 0 |
| local-2-screen | local-6fe6434487f0 | weave-rush | first | 8/8 | 0 |
| local-2-screen | local-6fe6434487f0 | weave-rush | second | 8/8 | 0 |
| local-2-screen | local-7986ca6d4f39 | starter | first | 8/8 | 0 |
| local-2-screen | local-7986ca6d4f39 | starter | second | 8/8 | 0 |
| local-2-screen | local-7986ca6d4f39 | weave-rush | first | 6/8 | 0 |
| local-2-screen | local-7986ca6d4f39 | weave-rush | second | 8/8 | 0 |
| local-2-screen | local-e4eb3574db1b | starter | first | 8/8 | 0 |
| local-2-screen | local-e4eb3574db1b | starter | second | 7/8 | 0 |
| local-2-screen | local-e4eb3574db1b | weave-rush | first | 7/8 | 0 |
| local-2-screen | local-e4eb3574db1b | weave-rush | second | 8/8 | 0 |
| local-2-screen | local-ed95cfa5b934 | starter | first | 7/8 | 0 |
| local-2-screen | local-ed95cfa5b934 | starter | second | 7/8 | 0 |
| local-2-screen | local-ed95cfa5b934 | weave-rush | first | 8/8 | 0 |
| local-2-screen | local-ed95cfa5b934 | weave-rush | second | 8/8 | 0 |
| local-2-screen | local-8be82525c048 | starter | first | 8/8 | 0 |
| local-2-screen | local-8be82525c048 | starter | second | 8/8 | 0 |
| local-2-screen | local-8be82525c048 | weave-rush | first | 7/8 | 0 |
| local-2-screen | local-8be82525c048 | weave-rush | second | 7/8 | 0 |
| local-2-screen | local-b2787975d2f4 | starter | first | 8/8 | 0 |
| local-2-screen | local-b2787975d2f4 | starter | second | 8/8 | 0 |
| local-2-screen | local-b2787975d2f4 | weave-rush | first | 8/8 | 0 |
| local-2-screen | local-b2787975d2f4 | weave-rush | second | 8/8 | 0 |
| local-2-screen | local-a040d72c5987 | starter | first | 8/8 | 0 |
| local-2-screen | local-a040d72c5987 | starter | second | 8/8 | 0 |
| local-2-screen | local-a040d72c5987 | weave-rush | first | 7/8 | 0 |
| local-2-screen | local-a040d72c5987 | weave-rush | second | 7/8 | 0 |
| local-2-final | original | starter | first | 109/128 | 0 |
| local-2-final | original | starter | second | 117/128 | 0 |
| local-2-final | original | weave-rush | first | 107/128 | 0 |
| local-2-final | original | weave-rush | second | 94/128 | 0 |
| local-2-final | local-7987ad220ccc | starter | first | 121/128 | 0 |
| local-2-final | local-7987ad220ccc | starter | second | 121/128 | 0 |
| local-2-final | local-7987ad220ccc | weave-rush | first | 107/128 | 0 |
| local-2-final | local-7987ad220ccc | weave-rush | second | 106/128 | 0 |
| local-2-final | local-b2787975d2f4 | starter | first | 114/128 | 0 |
| local-2-final | local-b2787975d2f4 | starter | second | 120/128 | 0 |
| local-2-final | local-b2787975d2f4 | weave-rush | first | 103/128 | 0 |
| local-2-final | local-b2787975d2f4 | weave-rush | second | 103/128 | 0 |
| local-2-final | local-d4d1980d2ba3 | starter | first | 116/128 | 0 |
| local-2-final | local-d4d1980d2ba3 | starter | second | 120/128 | 0 |
| local-2-final | local-d4d1980d2ba3 | weave-rush | first | 108/128 | 0 |
| local-2-final | local-d4d1980d2ba3 | weave-rush | second | 102/128 | 0 |
| local-2-final | local-6fe6434487f0 | starter | first | 115/128 | 0 |
| local-2-final | local-6fe6434487f0 | starter | second | 120/128 | 0 |
| local-2-final | local-6fe6434487f0 | weave-rush | first | 103/128 | 0 |
| local-2-final | local-6fe6434487f0 | weave-rush | second | 108/128 | 0 |
| local-2-final | local-7986ca6d4f39 | starter | first | 115/128 | 0 |
| local-2-final | local-7986ca6d4f39 | starter | second | 122/128 | 0 |
| local-2-final | local-7986ca6d4f39 | weave-rush | first | 105/128 | 0 |
| local-2-final | local-7986ca6d4f39 | weave-rush | second | 111/128 | 0 |
| local-2-final | local-8be82525c048 | starter | first | 116/128 | 0 |
| local-2-final | local-8be82525c048 | starter | second | 119/128 | 0 |
| local-2-final | local-8be82525c048 | weave-rush | first | 101/128 | 0 |
| local-2-final | local-8be82525c048 | weave-rush | second | 103/128 | 0 |
| local-2-final | local-a040d72c5987 | starter | first | 120/128 | 0 |
| local-2-final | local-a040d72c5987 | starter | second | 124/128 | 0 |
| local-2-final | local-a040d72c5987 | weave-rush | first | 105/128 | 0 |
| local-2-final | local-a040d72c5987 | weave-rush | second | 101/128 | 0 |
| local-2-final | local-c3d56c941926 | starter | first | 115/128 | 0 |
| local-2-final | local-c3d56c941926 | starter | second | 115/128 | 0 |
| local-2-final | local-c3d56c941926 | weave-rush | first | 96/128 | 0 |
| local-2-final | local-c3d56c941926 | weave-rush | second | 100/128 | 0 |
| local-2-final | local-e4eb3574db1b | starter | first | 114/128 | 0 |
| local-2-final | local-e4eb3574db1b | starter | second | 115/128 | 0 |
| local-2-final | local-e4eb3574db1b | weave-rush | first | 106/128 | 0 |
| local-2-final | local-e4eb3574db1b | weave-rush | second | 100/128 | 0 |
| local-3-screen | original | starter | first | 7/8 | 0 |
| local-3-screen | original | starter | second | 8/8 | 0 |
| local-3-screen | original | weave-rush | first | 7/8 | 0 |
| local-3-screen | original | weave-rush | second | 4/8 | 0 |
| local-3-screen | local-d4d1980d2ba3 | starter | first | 7/8 | 0 |
| local-3-screen | local-d4d1980d2ba3 | starter | second | 7/8 | 0 |
| local-3-screen | local-d4d1980d2ba3 | weave-rush | first | 8/8 | 0 |
| local-3-screen | local-d4d1980d2ba3 | weave-rush | second | 7/8 | 0 |
| local-3-screen | local-ca32f7ec0c5a | starter | first | 7/8 | 0 |
| local-3-screen | local-ca32f7ec0c5a | starter | second | 7/8 | 0 |
| local-3-screen | local-ca32f7ec0c5a | weave-rush | first | 8/8 | 0 |
| local-3-screen | local-ca32f7ec0c5a | weave-rush | second | 8/8 | 0 |
| local-3-screen | local-2ccf2c94113f | starter | first | 8/8 | 0 |
| local-3-screen | local-2ccf2c94113f | starter | second | 7/8 | 0 |
| local-3-screen | local-2ccf2c94113f | weave-rush | first | 6/8 | 0 |
| local-3-screen | local-2ccf2c94113f | weave-rush | second | 8/8 | 0 |
| local-3-screen | local-b14875a315da | starter | first | 8/8 | 0 |
| local-3-screen | local-b14875a315da | starter | second | 7/8 | 0 |
| local-3-screen | local-b14875a315da | weave-rush | first | 6/8 | 0 |
| local-3-screen | local-b14875a315da | weave-rush | second | 8/8 | 0 |
| local-3-screen | local-334a703cf1f3 | starter | first | 7/8 | 0 |
| local-3-screen | local-334a703cf1f3 | starter | second | 8/8 | 0 |
| local-3-screen | local-334a703cf1f3 | weave-rush | first | 6/8 | 0 |
| local-3-screen | local-334a703cf1f3 | weave-rush | second | 8/8 | 0 |
| local-3-screen | local-ae739db65f56 | starter | first | 8/8 | 0 |
| local-3-screen | local-ae739db65f56 | starter | second | 7/8 | 0 |
| local-3-screen | local-ae739db65f56 | weave-rush | first | 8/8 | 0 |
| local-3-screen | local-ae739db65f56 | weave-rush | second | 7/8 | 0 |
| local-3-screen | local-74ddf3abcac8 | starter | first | 8/8 | 0 |
| local-3-screen | local-74ddf3abcac8 | starter | second | 7/8 | 0 |
| local-3-screen | local-74ddf3abcac8 | weave-rush | first | 8/8 | 0 |
| local-3-screen | local-74ddf3abcac8 | weave-rush | second | 7/8 | 0 |
| local-3-screen | local-c55a35df7c2b | starter | first | 7/8 | 0 |
| local-3-screen | local-c55a35df7c2b | starter | second | 7/8 | 0 |
| local-3-screen | local-c55a35df7c2b | weave-rush | first | 8/8 | 0 |
| local-3-screen | local-c55a35df7c2b | weave-rush | second | 8/8 | 0 |
| local-3-final | original | starter | first | 114/128 | 0 |
| local-3-final | original | starter | second | 114/128 | 0 |
| local-3-final | original | weave-rush | first | 103/128 | 0 |
| local-3-final | original | weave-rush | second | 103/128 | 0 |
| local-3-final | local-7987ad220ccc | starter | first | 121/128 | 0 |
| local-3-final | local-7987ad220ccc | starter | second | 123/128 | 0 |
| local-3-final | local-7987ad220ccc | weave-rush | first | 111/128 | 0 |
| local-3-final | local-7987ad220ccc | weave-rush | second | 109/128 | 0 |
| local-3-final | local-74ddf3abcac8 | starter | first | 115/128 | 0 |
| local-3-final | local-74ddf3abcac8 | starter | second | 118/128 | 0 |
| local-3-final | local-74ddf3abcac8 | weave-rush | first | 104/128 | 0 |
| local-3-final | local-74ddf3abcac8 | weave-rush | second | 103/128 | 0 |
| local-3-final | local-ae739db65f56 | starter | first | 119/128 | 0 |
| local-3-final | local-ae739db65f56 | starter | second | 116/128 | 0 |
| local-3-final | local-ae739db65f56 | weave-rush | first | 101/128 | 0 |
| local-3-final | local-ae739db65f56 | weave-rush | second | 105/128 | 0 |
| local-3-final | local-c55a35df7c2b | starter | first | 118/128 | 0 |
| local-3-final | local-c55a35df7c2b | starter | second | 120/128 | 0 |
| local-3-final | local-c55a35df7c2b | weave-rush | first | 114/128 | 0 |
| local-3-final | local-c55a35df7c2b | weave-rush | second | 105/128 | 0 |
| local-3-final | local-ca32f7ec0c5a | starter | first | 118/128 | 0 |
| local-3-final | local-ca32f7ec0c5a | starter | second | 119/128 | 0 |
| local-3-final | local-ca32f7ec0c5a | weave-rush | first | 112/128 | 0 |
| local-3-final | local-ca32f7ec0c5a | weave-rush | second | 105/128 | 0 |
| local-3-final | local-2ccf2c94113f | starter | first | 118/128 | 0 |
| local-3-final | local-2ccf2c94113f | starter | second | 122/128 | 0 |
| local-3-final | local-2ccf2c94113f | weave-rush | first | 110/128 | 0 |
| local-3-final | local-2ccf2c94113f | weave-rush | second | 104/128 | 0 |
| local-3-final | local-2e4b3a6208ae | starter | first | 114/128 | 0 |
| local-3-final | local-2e4b3a6208ae | starter | second | 116/128 | 0 |
| local-3-final | local-2e4b3a6208ae | weave-rush | first | 98/128 | 0 |
| local-3-final | local-2e4b3a6208ae | weave-rush | second | 108/128 | 0 |
| local-3-final | local-334a703cf1f3 | starter | first | 114/128 | 0 |
| local-3-final | local-334a703cf1f3 | starter | second | 121/128 | 0 |
| local-3-final | local-334a703cf1f3 | weave-rush | first | 108/128 | 0 |
| local-3-final | local-334a703cf1f3 | weave-rush | second | 104/128 | 0 |
| local-3-final | local-482ee0db912d | starter | first | 116/128 | 0 |
| local-3-final | local-482ee0db912d | starter | second | 117/128 | 0 |
| local-3-final | local-482ee0db912d | weave-rush | first | 102/128 | 0 |
| local-3-final | local-482ee0db912d | weave-rush | second | 110/128 | 0 |
| local-4-screen | original | starter | first | 6/8 | 0 |
| local-4-screen | original | starter | second | 7/8 | 0 |
| local-4-screen | original | weave-rush | first | 7/8 | 0 |
| local-4-screen | original | weave-rush | second | 6/8 | 0 |
| local-4-screen | local-b01cfb985d2c | starter | first | 8/8 | 0 |
| local-4-screen | local-b01cfb985d2c | starter | second | 8/8 | 0 |
| local-4-screen | local-b01cfb985d2c | weave-rush | first | 7/8 | 0 |
| local-4-screen | local-b01cfb985d2c | weave-rush | second | 8/8 | 0 |
| local-4-screen | local-5ef65f2858a3 | starter | first | 8/8 | 0 |
| local-4-screen | local-5ef65f2858a3 | starter | second | 8/8 | 0 |
| local-4-screen | local-5ef65f2858a3 | weave-rush | first | 7/8 | 0 |
| local-4-screen | local-5ef65f2858a3 | weave-rush | second | 8/8 | 0 |
| local-4-screen | local-8863249f17d3 | starter | first | 8/8 | 0 |
| local-4-screen | local-8863249f17d3 | starter | second | 8/8 | 0 |
| local-4-screen | local-8863249f17d3 | weave-rush | first | 8/8 | 0 |
| local-4-screen | local-8863249f17d3 | weave-rush | second | 8/8 | 0 |
| local-4-screen | local-cc88e3523f9d | starter | first | 8/8 | 0 |
| local-4-screen | local-cc88e3523f9d | starter | second | 8/8 | 0 |
| local-4-screen | local-cc88e3523f9d | weave-rush | first | 7/8 | 0 |
| local-4-screen | local-cc88e3523f9d | weave-rush | second | 8/8 | 0 |
| local-4-screen | local-ca27e9aac260 | starter | first | 7/8 | 0 |
| local-4-screen | local-ca27e9aac260 | starter | second | 8/8 | 0 |
| local-4-screen | local-ca27e9aac260 | weave-rush | first | 8/8 | 0 |
| local-4-screen | local-ca27e9aac260 | weave-rush | second | 8/8 | 0 |
| local-4-screen | local-1f5f816b4c57 | starter | first | 8/8 | 0 |
| local-4-screen | local-1f5f816b4c57 | starter | second | 8/8 | 0 |
| local-4-screen | local-1f5f816b4c57 | weave-rush | first | 7/8 | 0 |
| local-4-screen | local-1f5f816b4c57 | weave-rush | second | 8/8 | 0 |
| local-4-screen | local-284d2d3ad1bc | starter | first | 8/8 | 0 |
| local-4-screen | local-284d2d3ad1bc | starter | second | 8/8 | 0 |
| local-4-screen | local-284d2d3ad1bc | weave-rush | first | 7/8 | 0 |
| local-4-screen | local-284d2d3ad1bc | weave-rush | second | 8/8 | 0 |
| local-4-screen | local-e94ba49a0efd | starter | first | 8/8 | 0 |
| local-4-screen | local-e94ba49a0efd | starter | second | 8/8 | 0 |
| local-4-screen | local-e94ba49a0efd | weave-rush | first | 7/8 | 0 |
| local-4-screen | local-e94ba49a0efd | weave-rush | second | 8/8 | 0 |
| local-4-final | original | starter | first | 113/128 | 0 |
| local-4-final | original | starter | second | 114/128 | 0 |
| local-4-final | original | weave-rush | first | 106/128 | 0 |
| local-4-final | original | weave-rush | second | 106/128 | 0 |
| local-4-final | local-7987ad220ccc | starter | first | 122/128 | 0 |
| local-4-final | local-7987ad220ccc | starter | second | 122/128 | 0 |
| local-4-final | local-7987ad220ccc | weave-rush | first | 109/128 | 0 |
| local-4-final | local-7987ad220ccc | weave-rush | second | 114/128 | 0 |
| local-4-final | local-8863249f17d3 | starter | first | 123/128 | 0 |
| local-4-final | local-8863249f17d3 | starter | second | 119/128 | 0 |
| local-4-final | local-8863249f17d3 | weave-rush | first | 109/128 | 0 |
| local-4-final | local-8863249f17d3 | weave-rush | second | 110/128 | 0 |
| local-4-final | local-0aa70ffd117c | starter | first | 118/128 | 0 |
| local-4-final | local-0aa70ffd117c | starter | second | 121/128 | 0 |
| local-4-final | local-0aa70ffd117c | weave-rush | first | 105/128 | 0 |
| local-4-final | local-0aa70ffd117c | weave-rush | second | 108/128 | 0 |
| local-4-final | local-0f5ba4c5501b | starter | first | 115/128 | 0 |
| local-4-final | local-0f5ba4c5501b | starter | second | 122/128 | 0 |
| local-4-final | local-0f5ba4c5501b | weave-rush | first | 105/128 | 0 |
| local-4-final | local-0f5ba4c5501b | weave-rush | second | 105/128 | 0 |
| local-4-final | local-1f5f816b4c57 | starter | first | 121/128 | 0 |
| local-4-final | local-1f5f816b4c57 | starter | second | 116/128 | 0 |
| local-4-final | local-1f5f816b4c57 | weave-rush | first | 105/128 | 0 |
| local-4-final | local-1f5f816b4c57 | weave-rush | second | 106/128 | 0 |
| local-4-final | local-284d2d3ad1bc | starter | first | 116/128 | 0 |
| local-4-final | local-284d2d3ad1bc | starter | second | 118/128 | 0 |
| local-4-final | local-284d2d3ad1bc | weave-rush | first | 100/128 | 0 |
| local-4-final | local-284d2d3ad1bc | weave-rush | second | 110/128 | 0 |
| local-4-final | local-3cb7eaeaf8b6 | starter | first | 114/128 | 0 |
| local-4-final | local-3cb7eaeaf8b6 | starter | second | 119/128 | 0 |
| local-4-final | local-3cb7eaeaf8b6 | weave-rush | first | 99/128 | 0 |
| local-4-final | local-3cb7eaeaf8b6 | weave-rush | second | 111/128 | 0 |
| local-4-final | local-5ef65f2858a3 | starter | first | 118/128 | 0 |
| local-4-final | local-5ef65f2858a3 | starter | second | 117/128 | 0 |
| local-4-final | local-5ef65f2858a3 | weave-rush | first | 99/128 | 0 |
| local-4-final | local-5ef65f2858a3 | weave-rush | second | 108/128 | 0 |
| local-4-final | local-b01cfb985d2c | starter | first | 117/128 | 0 |
| local-4-final | local-b01cfb985d2c | starter | second | 118/128 | 0 |
| local-4-final | local-b01cfb985d2c | weave-rush | first | 102/128 | 0 |
| local-4-final | local-b01cfb985d2c | weave-rush | second | 109/128 | 0 |

模型身份与对手实际配牌分别记录在 candidate-selection.json；编译配对采用偶/奇种子，不声称牌序完全相同。

本表仅列 12 项代表记录；全部 15392 项保存在 candidate-decisions.json。

| 采样版本 | 候选 | 先手胜/完成 | 后手胜/完成 | 筛选分 | 处理 |
| --- | --- | --- | --- | ---: | --- |
| 961 | tuned-c3defc048fb0 | 20/20 | 20/20 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 961 | tuned-20ab126ad067 | 20/20 | 20/20 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 960 | tuned-674b246d9dfc | 19/20 | 20/20 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 959 | tuned-60241bf69836 | 20/20 | 20/20 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 958 | tuned-94b52ea9de6d | 20/20 | 20/20 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 958 | tuned-7998302b2971 | 20/20 | 20/20 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 958 | tuned-436b729a3e52 | 20/20 | 20/20 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 957 | tuned-f8585d3717fc | 20/20 | 20/20 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 956 | tuned-56bb744c79fb | 20/20 | 20/20 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 955 | tuned-e46930db6eee | 20/20 | 20/20 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 947 | tuned-e123e406bdbb | 19/20 | 20/20 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 943 | tuned-f4d37b95df0c | 20/20 | 20/20 | 1.0 | 本批未选中；不据此认定构筑更差 |

## 本轮选定配牌：逐卡与机制摘要

下表仅使用最终同模型、同构筑镜像互搏，统计双方并按实际先后手分组；规则 AI 不参与主逐卡统计，交叉互搏另列于诊断 CSV。每套 64 场时各有 64 个先手和后手观测，仍是 64 场（32 个种子配对），不是 128 场独立对局。未携带或未使用的卡没有对应使用胜率，不能据此声称全卡池已有覆盖。抽到仅计 draw 事件，见到还包括起手、换牌、检索与回收等原卡入手。成功使用只计原卡，复制/衍生来源另列。

| 套牌 | 位置 | 卡牌 | 张数 | 见到局数/总数 | 打出局数 | 打出时胜率 | 平均首次打出回合 | 使用局平均结束回合 |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| starter | first | N02 第二位成员 | 2 | 29/32 | 27 | 48.1% | 6.04 | 19.93 |
| starter | first | N03 要叫大姐头 | 2 | 24/32 | 24 | 41.7% | 6.25 | 19.62 |
| starter | first | N04 不是闯祸精 | 2 | 24/32 | 23 | 39.1% | 10.57 | 20.22 |
| starter | first | N07 心有猛虎 | 1 | 27/32 | 16 | 56.2% | 13.38 | 20.50 |
| starter | first | N08 预备备 | 1 | 19/32 | 9 | 100.0% | 15.89 | 20.56 |
| starter | first | Y02 策略：「一报还一报」 | 2 | 28/32 | 23 | 39.1% | 5.96 | 20.17 |
| starter | first | Y03 策略特性：善良 | 1 | 21/32 | 17 | 29.4% | 9.71 | 20.47 |
| starter | first | Y06 策略特性：宽容 | 2 | 31/32 | 29 | 37.9% | 8.38 | 18.86 |
| starter | first | Y07 想做什么梦？ | 2 | 26/32 | 13 | 46.2% | 13.77 | 23.00 |
| starter | first | Y08 错误的门 | 1 | 16/32 | 8 | 50.0% | 14.50 | 23.25 |
| starter | first | Z03 奇异记叙 | 2 | 28/32 | 23 | 43.5% | 10.65 | 21.09 |
| starter | first | Z04 未决迷数 | 1 | 18/32 | 17 | 47.1% | 6.18 | 20.76 |
| starter | first | Z05 默示赋命 | 2 | 28/32 | 25 | 44.0% | 9.64 | 19.76 |
| starter | first | Z07 休息日 | 1 | 21/32 | 11 | 36.4% | 13.55 | 21.45 |
| starter | first | Z08 倾世之雨 | 2 | 27/32 | 22 | 45.5% | 9.64 | 20.50 |
| starter | first | J01 知晓与制衡 | 1 | 19/32 | 8 | 50.0% | 14.75 | 23.50 |
| starter | first | J02 过往消息皆为利刃 | 2 | 24/32 | 20 | 45.0% | 9.20 | 20.40 |
| starter | first | J03 优势初成 | 1 | 26/32 | 16 | 43.8% | 11.38 | 20.12 |
| starter | first | J04 出手即致命 | 2 | 32/32 | 15 | 20.0% | 11.40 | 19.93 |
| starter | first | J07 使命必达 | 1 | 18/32 | 6 | 33.3% | 18.33 | 24.33 |
| starter | first | J08 现实避难所 | 1 | 20/32 | 3 | 33.3% | 11.67 | 22.67 |
| starter | second | N02 第二位成员 | 2 | 28/32 | 27 | 59.3% | 6.96 | 19.70 |
| starter | second | N03 要叫大姐头 | 2 | 23/32 | 23 | 60.9% | 6.70 | 20.26 |
| starter | second | N04 不是闯祸精 | 2 | 26/32 | 23 | 56.5% | 11.22 | 19.78 |
| starter | second | N07 心有猛虎 | 1 | 25/32 | 15 | 80.0% | 14.80 | 20.60 |
| starter | second | N08 预备备 | 1 | 15/32 | 6 | 83.3% | 15.00 | 20.83 |
| starter | second | Y02 策略：「一报还一报」 | 2 | 29/32 | 24 | 58.3% | 8.08 | 19.79 |
| starter | second | Y03 策略特性：善良 | 1 | 19/32 | 18 | 77.8% | 11.67 | 19.67 |
| starter | second | Y06 策略特性：宽容 | 2 | 28/32 | 27 | 63.0% | 7.85 | 19.81 |
| starter | second | Y07 想做什么梦？ | 2 | 23/32 | 16 | 56.2% | 13.25 | 21.31 |
| starter | second | Y08 错误的门 | 1 | 18/32 | 9 | 44.4% | 13.33 | 23.78 |
| starter | second | Z03 奇异记叙 | 2 | 30/32 | 24 | 50.0% | 10.58 | 19.79 |
| starter | second | Z04 未决迷数 | 1 | 23/32 | 22 | 59.1% | 9.36 | 20.64 |
| starter | second | Z05 默示赋命 | 2 | 27/32 | 23 | 56.5% | 8.96 | 19.96 |
| starter | second | Z07 休息日 | 1 | 23/32 | 12 | 50.0% | 14.00 | 22.08 |
| starter | second | Z08 倾世之雨 | 2 | 28/32 | 25 | 68.0% | 9.84 | 19.36 |
| starter | second | J01 知晓与制衡 | 1 | 17/32 | 6 | 66.7% | 12.67 | 22.50 |
| starter | second | J02 过往消息皆为利刃 | 2 | 22/32 | 14 | 42.9% | 6.43 | 20.36 |
| starter | second | J03 优势初成 | 1 | 22/32 | 14 | 57.1% | 12.71 | 21.50 |
| starter | second | J04 出手即致命 | 2 | 27/32 | 18 | 50.0% | 11.78 | 21.44 |
| starter | second | J07 使命必达 | 1 | 18/32 | 5 | 60.0% | 12.80 | 23.20 |
| starter | second | J08 现实避难所 | 1 | 20/32 | 5 | 100.0% | 14.40 | 21.20 |
| weave-rush | first | M01 特遣行动 | 2 | 28/32 | 27 | 74.1% | 4.33 | 15.96 |
| weave-rush | first | M02 二次补考 | 1 | 17/32 | 16 | 68.8% | 8.00 | 15.50 |
| weave-rush | first | M03 全勤奖金 | 2 | 24/32 | 20 | 55.0% | 10.00 | 17.35 |
| weave-rush | first | M04 午休时间 | 2 | 25/32 | 21 | 71.4% | 8.71 | 16.76 |
| weave-rush | first | M08 开始净空 | 1 | 15/32 | 5 | 40.0% | 11.40 | 16.80 |
| weave-rush | first | B01 唷 队长！ | 2 | 26/32 | 24 | 70.8% | 3.00 | 15.75 |
| weave-rush | first | B02 招魂幡 | 2 | 25/32 | 24 | 75.0% | 6.33 | 16.12 |
| weave-rush | first | B06 于五色为白 | 2 | 25/32 | 17 | 70.6% | 10.88 | 16.65 |
| weave-rush | first | B07 判予秋 | 2 | 25/32 | 22 | 77.3% | 5.82 | 16.36 |
| weave-rush | first | Z03 奇异记叙 | 2 | 26/32 | 21 | 52.4% | 8.52 | 16.52 |
| weave-rush | first | Z04 未决迷数 | 2 | 27/32 | 20 | 65.0% | 9.30 | 17.20 |
| weave-rush | first | Z07 休息日 | 2 | 30/32 | 16 | 81.2% | 9.88 | 17.69 |
| weave-rush | first | Z08 倾世之雨 | 2 | 28/32 | 23 | 69.6% | 7.87 | 15.87 |
| weave-rush | first | Y01 重复囚徒困境 | 2 | 23/32 | 14 | 57.1% | 11.57 | 17.21 |
| weave-rush | first | Y03 策略特性：善良 | 2 | 25/32 | 22 | 68.2% | 6.00 | 16.55 |
| weave-rush | first | Y06 策略特性：宽容 | 2 | 26/32 | 21 | 61.9% | 7.10 | 16.33 |
| weave-rush | first | Y07 想做什么梦？ | 2 | 22/32 | 21 | 61.9% | 6.81 | 15.90 |
| weave-rush | second | M01 特遣行动 | 2 | 28/32 | 27 | 44.4% | 5.41 | 15.89 |
| weave-rush | second | M02 二次补考 | 1 | 16/32 | 16 | 50.0% | 7.25 | 16.06 |
| weave-rush | second | M03 全勤奖金 | 2 | 25/32 | 22 | 31.8% | 9.82 | 16.64 |
| weave-rush | second | M04 午休时间 | 2 | 25/32 | 21 | 47.6% | 8.67 | 15.90 |
| weave-rush | second | M08 开始净空 | 1 | 16/32 | 7 | 14.3% | 11.71 | 18.00 |
| weave-rush | second | B01 唷 队长！ | 2 | 25/32 | 24 | 37.5% | 4.50 | 16.25 |
| weave-rush | second | B02 招魂幡 | 2 | 26/32 | 23 | 39.1% | 6.96 | 16.39 |
| weave-rush | second | B06 于五色为白 | 2 | 23/32 | 17 | 23.5% | 9.29 | 16.59 |
| weave-rush | second | B07 判予秋 | 2 | 25/32 | 21 | 38.1% | 5.90 | 16.24 |
| weave-rush | second | Z03 奇异记叙 | 2 | 25/32 | 18 | 27.8% | 8.00 | 16.56 |
| weave-rush | second | Z04 未决迷数 | 2 | 27/32 | 19 | 36.8% | 9.37 | 16.68 |
| weave-rush | second | Z07 休息日 | 2 | 27/32 | 16 | 31.2% | 8.88 | 17.50 |
| weave-rush | second | Z08 倾世之雨 | 2 | 28/32 | 23 | 43.5% | 7.57 | 15.96 |
| weave-rush | second | Y01 重复囚徒困境 | 2 | 26/32 | 17 | 41.2% | 12.12 | 17.06 |
| weave-rush | second | Y03 策略特性：善良 | 2 | 22/32 | 21 | 38.1% | 5.05 | 16.38 |
| weave-rush | second | Y06 策略特性：宽容 | 2 | 25/32 | 21 | 38.1% | 7.71 | 16.38 |
| weave-rush | second | Y07 想做什么梦？ | 2 | 23/32 | 22 | 45.5% | 6.91 | 16.05 |

### 终结与环合

| 测试 | 位置 | 角色 | 机制 | 次数 | 使用局数 | 使用局胜率 | 可用角色回合 | 首次使用回合 | 同动作对手生命变化/次 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| starter-candidate-mirror | first | nanali | ultimate | 64 | 31 | 41.9% | 71 | 8.612903225806452 | 0.0 |
| starter-candidate-mirror | first | nanali | harmony | 15 | 13 | 61.5% | 18 | 7.923076923076923 | -1.7333333333333334 |
| starter-candidate-mirror | first | iloy | ultimate | 82 | 30 | 43.3% | 93 | 9.2 | 0.0 |
| starter-candidate-mirror | first | iloy | harmony | 10 | 9 | 33.3% | 43 | 16.11111111111111 | -0.2 |
| starter-candidate-mirror | first | zero | ultimate | 74 | 29 | 44.8% | 74 | 8.517241379310345 | 0.0 |
| starter-candidate-mirror | first | zero | harmony | 16 | 12 | 75.0% | 57 | 12.0 | -3.25 |
| starter-candidate-mirror | first | jiuyuan | ultimate | 53 | 28 | 46.4% | 58 | 9.571428571428571 | 0.0 |
| starter-candidate-mirror | first | jiuyuan | harmony | 14 | 10 | 40.0% | 33 | 10.4 | -1.9285714285714286 |
| starter-candidate-mirror | second | nanali | ultimate | 77 | 30 | 63.3% | 90 | 7.533333333333333 | 0.0 |
| starter-candidate-mirror | second | nanali | harmony | 13 | 11 | 63.6% | 24 | 12.545454545454545 | -3.923076923076923 |
| starter-candidate-mirror | second | iloy | ultimate | 82 | 31 | 61.3% | 94 | 8.516129032258064 | 0.0 |
| starter-candidate-mirror | second | iloy | harmony | 7 | 6 | 66.7% | 49 | 10.0 | -0.2857142857142857 |
| starter-candidate-mirror | second | zero | ultimate | 81 | 32 | 59.4% | 81 | 8.25 | 0.0 |
| starter-candidate-mirror | second | zero | harmony | 15 | 14 | 78.6% | 68 | 13.857142857142858 | -3.2666666666666666 |
| starter-candidate-mirror | second | jiuyuan | ultimate | 52 | 31 | 61.3% | 56 | 9.67741935483871 | 0.0 |
| starter-candidate-mirror | second | jiuyuan | harmony | 16 | 14 | 64.3% | 35 | 8.857142857142858 | -3.8125 |
| weave-rush-candidate-mirror | first | bohe | ultimate | 31 | 23 | 82.6% | 37 | 9.695652173913043 | 0.0 |
| weave-rush-candidate-mirror | first | bohe | harmony | 14 | 13 | 92.3% | 22 | 10.076923076923077 | -4.928571428571429 |
| weave-rush-candidate-mirror | first | baicang | ultimate | 45 | 29 | 69.0% | 47 | 7.758620689655173 | 0.0 |
| weave-rush-candidate-mirror | first | baicang | harmony | 2 | 2 | 100.0% | 4 | 10.0 | -8.5 |
| weave-rush-candidate-mirror | first | zero | ultimate | 58 | 29 | 69.0% | 57 | 8.10344827586207 | 0.0 |
| weave-rush-candidate-mirror | first | zero | harmony | 4 | 4 | 100.0% | 17 | 12.5 | -3.75 |
| weave-rush-candidate-mirror | first | iloy | ultimate | 57 | 31 | 61.3% | 78 | 8.935483870967742 | 0.0 |
| weave-rush-candidate-mirror | first | iloy | harmony | 0 | 0 | — | 66 | — | — |
| weave-rush-candidate-mirror | second | bohe | ultimate | 22 | 20 | 40.0% | 29 | 10.0 | 0.0 |
| weave-rush-candidate-mirror | second | bohe | harmony | 11 | 11 | 36.4% | 14 | 11.090909090909092 | -3.909090909090909 |
| weave-rush-candidate-mirror | second | baicang | ultimate | 38 | 29 | 41.4% | 41 | 10.206896551724139 | 0.0 |
| weave-rush-candidate-mirror | second | baicang | harmony | 3 | 3 | 100.0% | 6 | 10.0 | -5.333333333333333 |
| weave-rush-candidate-mirror | second | zero | ultimate | 49 | 28 | 42.9% | 48 | 8.0 | 0.0 |
| weave-rush-candidate-mirror | second | zero | harmony | 0 | 0 | — | 11 | — | — |
| weave-rush-candidate-mirror | second | iloy | ultimate | 49 | 30 | 40.0% | 65 | 8.8 | 0.0 |
| weave-rush-candidate-mirror | second | iloy | harmony | 0 | 0 | — | 55 | — | — |

环合类型、提供者、受益者、终结结束及源效果事件保留在每局 telemetry；机会以存在合法发动动作的角色·全局回合去重，包含普通出击和战斗牌；同回合多次触发另计次数。

## 曲线与统计限制

- `turn-states.csv` 给出各回合仍在进行的样本量及平均生命、行动力、能量、环合值；没有概率预测。
- `selfplay-card-metrics.csv` 是主逐卡数据，仅含最终同模型同构筑镜像，双方按实际先后手统计。`card-metrics.csv` 是全矩阵诊断，规则 AI 与交叉互搏不能混入主统计；非镜像行仅统计学习方。包含抽到/见到/使用的分子分母、来源、平均次数和回合数。
- `training-win-rates.csv` 保留各次更新按对手类型和先后手的胜率序列，不能混入冻结评估。
- `generated-card-metrics.csv` 单列复制与衍生牌的获得、使用和胜负关联，不混入主牌组抽到率。
- `mechanism-metrics.csv` 包含按角色/先后手的机会、次数和同动作生命变化。跨动作/跨回合的净因果收益尚未归因，不将相邻效果全算作某机制贡献。
- `sample-replay.json` 是已记录对局的公开可播放录像；私有种子及隐藏牌库不在其中。
- 全部测试可执行不等于本轮样本量足以判定强度；未使用的卡牌/机制不判弱。
- 本轮不提供可靠预测胜率、长期收敛结论或机制的因果胜率贡献。

## 起手与首回合对照

状态：完整；单元 8/8。

同一组模型、实际构筑及种子，仅改变学习方是否使用学习换牌；对手策略保持一致。详细留牌与首动作见 opening-analysis/opening-report.md。

| 学习方 | 对手阵容 | 起手策略 | 位置 | 胜/完成 |
| --- | --- | --- | --- | --- |
| starter | starter | learned | first | 5/16 |
| starter | starter | learned | second | 6/16 |
| starter | starter | keep_all | first | 10/16 |
| starter | starter | keep_all | second | 2/16 |
| starter | weave-rush | learned | first | 3/16 |
| starter | weave-rush | learned | second | 5/16 |
| starter | weave-rush | keep_all | first | 3/16 |
| starter | weave-rush | keep_all | second | 4/16 |
| weave-rush | starter | learned | first | 14/16 |
| weave-rush | starter | learned | second | 11/16 |
| weave-rush | starter | keep_all | first | 13/16 |
| weave-rush | starter | keep_all | second | 12/16 |
| weave-rush | weave-rush | learned | first | 8/16 |
| weave-rush | weave-rush | learned | second | 13/16 |
| weave-rush | weave-rush | keep_all | first | 5/16 |
| weave-rush | weave-rush | keep_all | second | 8/16 |
