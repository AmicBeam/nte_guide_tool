# 综合训练、最终构筑与模型交付 · 2026-09-16

> 后续规则修订：2026-09-16「诓定解放」改为回合结束立即回费、生命 5 → 6；用户指定高级创生的 Z06 ×2 换为 Z07 ×2。本报告训练与胜率属于修订前规则和构筑，不代表更新后的成绩。当前维护验证见 [Z06 修复记录](duel-v2-z06-fix-20260916.md)。

## 交付结论

**逐卡口径更正：** 主逐卡表已改为互搏。当前 Mac 汇总只含学习方 A，各位置 32 场；Windows 已下线，双方完整统计与新交付包待补，原 ZIP 的旧逐卡主表已过期。模型与构筑无需变更。

- 全部训练阶段共用 **5 小时 44 分 35 秒**，守护进程退出码 0；未超过 6 小时。各阶段按预算结束，未宣称已经收敛。
- 09:00 后只运行冻结评估：核心矩阵 34 个单元、1472 场；候选复评 30 个单元、480 场；最终留出 15 个单元、632 场。共 **79 个单元、2584 场**，全部正常完成，无异常或截断。三类结果不混算为一个强度胜率。
- 创生从整轮搜索短名单中选出 `history-bound`，独立筛选 46/48，原始配牌 43/48；在固定四名角色内更换 12 张配牌。采用第 2886 次更新的基础策略权重，而不是未经比较地取最后一个 checkpoint。
- 覆纹快攻短名单中原始配牌为 47/48，其他两项同分；按预定同分保留原始规则选择原始配牌，采用第 2258 次更新的基础策略。
- 最终模型及真实配牌已成套更新到 `app/modules/card_game/engine/ai/models/`。普通官方预组和玩家保存构筑保持原样；镜像开局使用高级人机实际配牌。
- **未部署 Windows 生产站点。** 私有分支提交/推送后等待用户验收部署。

### 最终留出：对变化规则 AI

| 组合 | 先手 | 后手 | 合计 | 旧网站模型同条件 |
| --- | --- | --- | --- | --- |
| 创生优选配牌 | 30/32，93.8% | 31/32，96.9% | 61/64，95.3% | 54/64，84.4% |
| 覆纹快攻原始配牌 | 31/32，96.9% | 30/32，93.8% | 61/64，95.3% | 60/64，93.8% |

覆纹快攻的差异仅 1 场，不能据此宣称显著提升。创生对固定覆纹快攻规则对手为 8/16，本报告保留该弱项，不只展示总体高胜率。

### 同模型、同构筑镜像

按每局实际先手身份计算，不使用固定玩家 A 的胜率冒充先手胜率。

| 套牌 | 先手方胜率 | 后手方胜率 | 场数 |
| --- | ---: | ---: | ---: |
| 创生最终组合 | 29/64，45.3% | 35/64，54.7% | 64 |
| 覆纹快攻最终组合 | 33/64，51.6% | 31/64，48.4% | 64 |

### 最终交叉互搏

| 套牌 | 先手胜率 | 后手胜率 |
| --- | ---: | ---: |
| 创生 | 10/32，31.3% | 17/32，53.1% |
| 覆纹快攻 | 15/32，46.9% | 22/32，68.8% |

以上来自同一批 64 场；不能把双方展示的行再次相加为 128 场。有限样本不能证明整体平衡。

## 身份、验证与保存位置

- 当前规则身份：`c1abaf1979bf377994a0c019e05e1bb4267098782562937d757c73baadc69d7b`。
- 创生权重 SHA：`2bc124623b2ad34c0ede0c63e4497d171d499f92786d22ccf08c983bfeb7396d`；构筑 SHA：`c618c7c7f25dfa6c7d7c8f01b570b0da3de587f2f2b2b9a92f3ace6a370ec21c`。
- 覆纹快攻权重 SHA：`2020f91830f5c397776965dcbc0b6661e1b8e138c07179e0bf29a92fa8640ba7`；构筑 SHA：`f68d661ce484a83494dc8555d85e74a580ff405adce321bef898c7e132b49206`。
- 各模型清单保留实际训练 `learner_deck`，另用 `serving_build` 与独立证据的构筑哈希绑定实际开局配牌；没有伪造训练构筑。未评估候选仍被拒绝用于网页对局。
- 最终相关回归 **92 项全部通过**，包含两套实际模型完成 API 对局、回放、公开玩家自组，以及模型/构筑绑定与报告打包。
- 本地已通过模型／构筑篡改、缺失字段、角色顺序、缓存重载、旧对局版本保护、镜像一致性、玩家存档不变等回归；真实两套模型能完成 API 对局并保存回放，公开玩家自组开局通过。
- Mac 完整报告与附件：`artifacts/rl-evals/comprehensive-20260916/report/`；精选模型与配牌在其 `final/`。
- 历史 Mac 交付包（逐卡主表已过期，不能作为更正后报告）：`artifacts/rl-evals/comprehensive-20260916/comprehensive-delivery.zip`，24,916,268 字节，33506 个文件校验通过。
- 包 SHA-256：`93b8ff5ca99d2218d66681ec1f54e6761647f38ba58735f0d0ac7f60afc9580f`。
- Windows 完整原始记录：`D:/git/nte_board_game/.rl-comprehensive-20260916/artifacts/comprehensive-six-hour/`。逐局私有日志及其余公开录像留在此处，未打入 Mac 交付包或 Git；公开示例录像随包。
- 仅数字权重、模型清单和整理后的报告进入 Git，训练 checkpoint、候选全记录、CSV 和原始记录不进入 Git。
- 冻结策略按最大分数选择动作，起手保留全部牌。未提供经独立校准的预测胜率；逐回合 CSV 是局势指标，不冒充胜率曲线。机制同动作变化属于关联证据，不等于隔离因果收益。

下文保留完整矩阵、训练过程、候选比较及逐卡／机制摘要。`starter` 为创生，`weave-rush` 为覆纹快攻；`selected` 是最终组合，`old-website` 是对照旧模型。

---

# 双预组综合训练报告

这是本轮有界综合训练与冻结评估报告，单次结果不证明全局最优或游戏平衡。

- 规则身份：`c1abaf1979bf377994a0c019e05e1bb4267098782562937d757c73baadc69d7b`。
- 训练标量与冻结评估分别列出；报告生成不批准上线。
- 预测胜率：未提供经校准的概率模型；附逐回合局势 CSV，原始价值分数不冒充胜率。
- 终结/环合效果附源事件；同动作内变化可能由多个效果共同造成，未声称因果胜率贡献。

## 训练与配牌阶段


### starter 阶段结果

本轮基础模型起点：D:\git\nte_board_game\.rl-public-short-20260915-1aad413\artifacts\play-position-short-20260916-01\latest.pt；hidden=64。训练预算及停止原因见阶段记录。

| 队伍 | 阶段 | 累计更新 | 本轮更新 | 正常终局 | 用时秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| starter | base | 2886 | 2214 | 854064 | 6600.03 |
| starter | cards | 964 | 964 | 123392 | 1798.92 |
| starter | adapted | 3494 | 608 | 264581 | 1800.12 |

### starter 训练内先后手统计（出牌阶段）

| 阶段 | 对手 | 位置 | 胜/负/平 | 胜率 |
| --- | --- | --- | --- | ---: |
| base | rule | first | 82530/2840/0 | 96.7% |
| base | rule | second | 83612/2061/0 | 97.6% |
| base | frozen_models | first | 276937/64019/0 | 81.2% |
| base | frozen_models | second | 293567/48694/0 | 85.8% |
| adapted | rule | first | 25819/451/0 | 98.3% |
| adapted | rule | second | 26344/275/0 | 99.0% |
| adapted | frozen_models | first | 95828/9765/3 | 90.7% |
| adapted | frozen_models | second | 100805/5371/0 | 94.9% |

每次更新的累计分组统计在对应 `metrics.jsonl`；停止时未完成局另见 `position-results.json`。

### weave-rush 阶段结果

本轮基础模型起点：从零初始化；hidden=64。训练预算及停止原因见阶段记录。

| 队伍 | 阶段 | 累计更新 | 本轮更新 | 正常终局 | 用时秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| weave-rush | base | 2258 | 2258 | 1333417 | 6600.11 |
| weave-rush | cards | 1114 | 1114 | 142592 | 1799.24 |
| weave-rush | adapted | 2866 | 608 | 307569 | 1800.13 |

### weave-rush 训练内先后手统计（出牌阶段）

| 阶段 | 对手 | 位置 | 胜/负/平 | 胜率 |
| --- | --- | --- | --- | ---: |
| base | rule | first | 134521/1717/2 | 98.7% |
| base | rule | second | 135576/1074/2 | 99.2% |
| base | frozen_models | first | 510848/19079/31 | 96.4% |
| base | frozen_models | second | 515848/15165/21 | 97.1% |
| adapted | rule | first | 30441/401/0 | 98.7% |
| adapted | rule | second | 30547/203/0 | 99.3% |
| adapted | frozen_models | first | 115845/7309/10 | 94.1% |
| adapted | frozen_models | second | 119242/3600/3 | 97.1% |

每次更新的累计分组统计在对应 `metrics.jsonl`；停止时未完成局另见 `position-results.json`。

## 冻结评估矩阵

| 测试 | 场数 | 学习方先手胜率 | 学习方后手胜率 | 实际先手方胜率 | 平均全局回合 | 异常/截断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| starter-screen-original | 32 | 93.8% | 100.0% | 46.9% | 17.03 | 0/0 |
| starter-screen-candidate | 32 | 87.5% | 93.8% | 46.9% | 16.84 | 0/0 |
| weave-rush-screen-original | 32 | 100.0% | 100.0% | 50.0% | 15.75 | 0/0 |
| weave-rush-screen-candidate | 32 | 87.5% | 87.5% | 50.0% | 15.53 | 0/0 |
| starter-original-rule-starter | 32 | 100.0% | 100.0% | 50.0% | 18.50 | 0/0 |
| starter-original-rule-weave-rush | 32 | 62.5% | 62.5% | 50.0% | 21.50 | 0/0 |
| starter-original-rule-varied | 64 | 81.2% | 90.6% | 45.3% | 19.80 | 0/0 |
| starter-original-model-varied-starter | 64 | 78.1% | 93.8% | 42.2% | 20.44 | 0/0 |
| starter-original-model-varied-weave-rush | 64 | 81.2% | 96.9% | 42.2% | 18.20 | 0/0 |
| starter-original-mirror | 32 | 25.0% | 62.5% | 31.2% | 21.62 | 0/0 |
| weave-rush-original-rule-starter | 32 | 100.0% | 93.8% | 53.1% | 14.53 | 0/0 |
| weave-rush-original-rule-weave-rush | 32 | 62.5% | 68.8% | 46.9% | 17.66 | 0/0 |
| weave-rush-original-rule-varied | 64 | 93.8% | 100.0% | 46.9% | 16.30 | 0/0 |
| weave-rush-original-model-varied-starter | 64 | 96.9% | 100.0% | 48.4% | 15.77 | 0/0 |
| weave-rush-original-model-varied-weave-rush | 64 | 100.0% | 100.0% | 50.0% | 14.84 | 0/0 |
| weave-rush-original-mirror | 32 | 25.0% | 31.2% | 46.9% | 17.38 | 0/0 |
| cross-original | 32 | 12.5% | 31.2% | 40.6% | 17.12 | 0/0 |
| cross-original（覆纹快攻视角，同批对局） | 32 | 68.8% | 87.5% | 40.6% | 17.12 | 同上，勿重复计局 |
| starter-candidate-rule-starter | 32 | 93.8% | 93.8% | 50.0% | 16.62 | 0/0 |
| starter-candidate-rule-weave-rush | 32 | 43.8% | 68.8% | 37.5% | 17.69 | 0/0 |
| starter-candidate-rule-varied | 64 | 90.6% | 100.0% | 45.3% | 17.80 | 0/0 |
| starter-candidate-model-varied-starter | 64 | 96.9% | 100.0% | 48.4% | 18.67 | 0/0 |
| starter-candidate-model-varied-weave-rush | 64 | 96.9% | 93.8% | 51.6% | 16.20 | 0/0 |
| starter-candidate-mirror | 32 | 43.8% | 31.2% | 56.2% | 21.38 | 0/0 |
| weave-rush-candidate-rule-starter | 32 | 87.5% | 93.8% | 46.9% | 16.78 | 0/0 |
| weave-rush-candidate-rule-weave-rush | 32 | 68.8% | 75.0% | 46.9% | 19.44 | 0/0 |
| weave-rush-candidate-rule-varied | 64 | 93.8% | 96.9% | 48.4% | 16.67 | 0/0 |
| weave-rush-candidate-model-varied-starter | 64 | 96.9% | 96.9% | 50.0% | 15.95 | 0/0 |
| weave-rush-candidate-model-varied-weave-rush | 64 | 96.9% | 96.9% | 50.0% | 15.22 | 0/0 |
| weave-rush-candidate-mirror | 32 | 56.2% | 75.0% | 40.6% | 19.41 | 0/0 |
| cross-candidate | 32 | 43.8% | 68.8% | 37.5% | 20.53 | 0/0 |
| cross-candidate（覆纹快攻视角，同批对局） | 32 | 31.2% | 56.2% | 37.5% | 20.53 | 同上，勿重复计局 |
| starter-candidate-old-policy | 32 | 93.8% | 93.8% | 50.0% | 18.12 | 0/0 |
| starter-candidate-new-policy | 32 | 100.0% | 100.0% | 50.0% | 18.31 | 0/0 |
| weave-rush-candidate-old-policy | 32 | 100.0% | 93.8% | 53.1% | 16.91 | 0/0 |
| weave-rush-candidate-new-policy | 32 | 100.0% | 93.8% | 53.1% | 16.78 | 0/0 |
| shortlist-starter-original-rule | 16 | 62.5% | 100.0% | 31.2% | 21.25 | 0/0 |
| shortlist-starter-original-own-base | 16 | 75.0% | 100.0% | 37.5% | 21.38 | 0/0 |
| shortlist-starter-original-other-base | 16 | 100.0% | 100.0% | 50.0% | 19.38 | 0/0 |
| shortlist-starter-last-adapted-rule | 16 | 87.5% | 87.5% | 50.0% | 18.31 | 0/0 |
| shortlist-starter-last-adapted-own-base | 16 | 87.5% | 100.0% | 43.8% | 19.94 | 0/0 |
| shortlist-starter-last-adapted-other-base | 16 | 100.0% | 100.0% | 50.0% | 18.00 | 0/0 |
| shortlist-starter-last-base-rule | 16 | 62.5% | 75.0% | 43.8% | 19.31 | 0/0 |
| shortlist-starter-last-base-own-base | 16 | 75.0% | 87.5% | 43.8% | 18.81 | 0/0 |
| shortlist-starter-last-base-other-base | 16 | 100.0% | 100.0% | 50.0% | 19.25 | 0/0 |
| shortlist-starter-history-bound-rule | 16 | 87.5% | 100.0% | 43.8% | 18.94 | 0/0 |
| shortlist-starter-history-bound-own-base | 16 | 100.0% | 87.5% | 56.2% | 18.56 | 0/0 |
| shortlist-starter-history-bound-other-base | 16 | 100.0% | 100.0% | 50.0% | 14.00 | 0/0 |
| shortlist-starter-recent-rule | 16 | 62.5% | 75.0% | 43.8% | 17.81 | 0/0 |
| shortlist-starter-recent-own-base | 16 | 87.5% | 100.0% | 43.8% | 20.81 | 0/0 |
| shortlist-starter-recent-other-base | 16 | 87.5% | 100.0% | 43.8% | 16.06 | 0/0 |
| shortlist-weave-rush-original-rule | 16 | 100.0% | 100.0% | 50.0% | 16.62 | 0/0 |
| shortlist-weave-rush-original-own-base | 16 | 87.5% | 100.0% | 43.8% | 15.50 | 0/0 |
| shortlist-weave-rush-original-other-base | 16 | 100.0% | 100.0% | 50.0% | 15.88 | 0/0 |
| shortlist-weave-rush-last-adapted-rule | 16 | 100.0% | 87.5% | 56.2% | 17.94 | 0/0 |
| shortlist-weave-rush-last-adapted-own-base | 16 | 87.5% | 100.0% | 43.8% | 17.31 | 0/0 |
| shortlist-weave-rush-last-adapted-other-base | 16 | 100.0% | 87.5% | 56.2% | 17.06 | 0/0 |
| shortlist-weave-rush-last-base-rule | 16 | 100.0% | 100.0% | 50.0% | 17.75 | 0/0 |
| shortlist-weave-rush-last-base-own-base | 16 | 87.5% | 100.0% | 43.8% | 17.19 | 0/0 |
| shortlist-weave-rush-last-base-other-base | 16 | 100.0% | 100.0% | 50.0% | 18.50 | 0/0 |
| shortlist-weave-rush-history-bound-rule | 16 | 87.5% | 75.0% | 56.2% | 18.44 | 0/0 |
| shortlist-weave-rush-history-bound-own-base | 16 | 87.5% | 100.0% | 43.8% | 18.19 | 0/0 |
| shortlist-weave-rush-history-bound-other-base | 16 | 100.0% | 100.0% | 50.0% | 18.25 | 0/0 |
| shortlist-weave-rush-recent-rule | 16 | 100.0% | 100.0% | 50.0% | 15.12 | 0/0 |
| shortlist-weave-rush-recent-own-base | 16 | 100.0% | 87.5% | 56.2% | 15.69 | 0/0 |
| shortlist-weave-rush-recent-other-base | 16 | 100.0% | 100.0% | 50.0% | 16.25 | 0/0 |
| starter-selected-rule-varied | 64 | 93.8% | 96.9% | 48.4% | 17.94 | 0/0 |
| starter-old-website-rule-varied | 64 | 84.4% | 84.4% | 50.0% | 18.11 | 0/0 |
| starter-selected-rule-starter | 16 | 100.0% | 100.0% | 50.0% | 17.00 | 0/0 |
| starter-selected-model-varied-starter | 30 | 86.7% | 100.0% | 43.3% | 17.87 | 0/0 |
| starter-selected-rule-weave-rush | 16 | 50.0% | 50.0% | 50.0% | 18.88 | 0/0 |
| starter-selected-model-varied-weave-rush | 30 | 100.0% | 93.3% | 53.3% | 17.73 | 0/0 |
| starter-selected-mirror | 64 | 40.6% | 50.0% | 45.3% | 20.14 | 0/0 |
| weave-rush-selected-rule-varied | 64 | 96.9% | 93.8% | 51.6% | 15.58 | 0/0 |
| weave-rush-old-website-rule-varied | 64 | 96.9% | 90.6% | 53.1% | 15.00 | 0/0 |
| weave-rush-selected-rule-starter | 16 | 100.0% | 75.0% | 62.5% | 14.50 | 0/0 |
| weave-rush-selected-model-varied-starter | 30 | 93.3% | 93.3% | 50.0% | 17.60 | 0/0 |
| weave-rush-selected-rule-weave-rush | 16 | 50.0% | 75.0% | 37.5% | 17.06 | 0/0 |
| weave-rush-selected-model-varied-weave-rush | 30 | 93.3% | 100.0% | 46.7% | 14.30 | 0/0 |
| weave-rush-selected-mirror | 64 | 53.1% | 50.0% | 51.6% | 18.20 | 0/0 |
| selected-cross | 64 | 31.2% | 53.1% | 39.1% | 17.53 | 0/0 |
| selected-cross（覆纹快攻视角，同批对局） | 64 | 46.9% | 68.8% | 39.1% | 17.53 | 同上，勿重复计局 |

交叉互搏的首行以创生为学习方；第二行换算覆纹快攻视角，来自同批对局。

## 训练末批候选对照（非最终发布选择）

下表只依据独立于最终测试集的筛选样本；小样本用于验证选择流程，不能据此声称最优。候选生成过程是配牌策略采样，不伪造逐张换牌的模型内心理由。

### starter

- 原始筛选胜率：96.9%；候选：90.6%；每种 32 场。
- 选择：original；理由：Candidate did not beat baseline; retain baseline。
- 全批候选及原始评分：`starter/cards/candidate-history.jsonl`。
- 候选适应训练单独完成；本表筛选使用冻结基础策略，最终矩阵另外展示适应后效果。

| 卡牌 | 原始张数 | 候选张数 |
| --- | ---: | ---: |
| J01 知晓与制衡 | 2 | 0 |
| J02 过往消息皆为利刃 | 2 | 1 |
| J03 优势初成 | 0 | 2 |
| J07 使命必达 | 0 | 1 |
| N07 心有猛虎 | 2 | 0 |
| N08 预备备 | 0 | 2 |
| Y02 策略：「一报还一报」 | 2 | 0 |
| Y07 想做什么梦？ | 0 | 2 |
| Z02 探悉天职 | 1 | 2 |
| Z03 奇异记叙 | 0 | 2 |
| Z05 默示赋命 | 2 | 0 |
| Z07 休息日 | 2 | 0 |
| Z08 倾世之雨 | 1 | 2 |

本表仅列 12 项代表记录；全部 15424 项保存在 candidate-decisions.json。

| 采样版本 | 候选 | 先手胜/完成 | 后手胜/完成 | 筛选分 | 处理 |
| --- | --- | --- | --- | ---: | --- |
| 963 | tuned-51100fb712c2 | 4/4 | 4/4 | 1.0 | 选入训练末批配对筛选 |
| 963 | tuned-77876b3bb6c6 | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 963 | tuned-d306b3d554db | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 963 | tuned-14afc69ad6b5 | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 963 | tuned-59edd57664c8 | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 963 | tuned-005494b1a93f | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 963 | tuned-c0392eaa20cf | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 963 | tuned-9373b28daf89 | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 962 | tuned-c36c95f2d2be | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 962 | tuned-1b5cf9bd21b7 | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 962 | tuned-0c69e0952076 | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 962 | tuned-c730f8715f78 | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |

### weave-rush

- 原始筛选胜率：100.0%；候选：87.5%；每种 32 场。
- 选择：original；理由：Candidate did not beat baseline; retain baseline。
- 全批候选及原始评分：`weave-rush/cards/candidate-history.jsonl`。
- 候选适应训练单独完成；本表筛选使用冻结基础策略，最终矩阵另外展示适应后效果。

| 卡牌 | 原始张数 | 候选张数 |
| --- | ---: | ---: |
| B01 唷 队长！ | 2 | 1 |
| B02 招魂幡 | 2 | 1 |
| B03 恶诅不过一开口 | 0 | 2 |
| B06 于五色为白 | 2 | 0 |
| B07 判予秋 | 0 | 2 |
| M02 二次补考 | 0 | 1 |
| M04 午休时间 | 2 | 1 |
| Y02 策略：「一报还一报」 | 0 | 2 |
| Y06 策略特性：宽容 | 2 | 0 |
| Z01 初明凝视 | 0 | 2 |
| Z04 未决迷数 | 2 | 0 |
| Z05 默示赋命 | 2 | 0 |
| Z06 诓定解放 | 0 | 1 |
| Z07 休息日 | 0 | 2 |
| Z08 倾世之雨 | 2 | 1 |

本表仅列 12 项代表记录；全部 17824 项保存在 candidate-decisions.json。

| 采样版本 | 候选 | 先手胜/完成 | 后手胜/完成 | 筛选分 | 处理 |
| --- | --- | --- | --- | ---: | --- |
| 1113 | tuned-5d12c7956ec2 | 4/4 | 4/4 | 1.0 | 选入训练末批配对筛选 |
| 1113 | tuned-a191e1707d54 | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 1113 | tuned-788e98f33906 | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 1113 | tuned-20b736b18ac4 | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 1113 | tuned-11361966a841 | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 1113 | tuned-68260e92fd80 | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 1113 | tuned-67ca3a274e8a | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 1113 | tuned-c4dcdfdeac04 | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 1113 | tuned-25e9e562d892 | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 1113 | tuned-7c48a597e872 | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 1113 | tuned-7829aee667ff | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |
| 1113 | tuned-6cd437c9289d | 4/4 | 4/4 | 1.0 | 本批未选中；不据此认定构筑更差 |

## 最终模型与优选构筑

从完整搜索记录形成有限短名单，再用独立配对筛选；最终留出集不参与选择。最高分同分时保留原始配牌，不声称穷举全局最优。

### starter 最终选择：history-bound

- 数值模型 SHA：`2bc124623b2ad34c0ede0c63e4497d171d499f92786d22ccf08c983bfeb7396d`；来源 `starter/base-export`。
- 搜索记录 15424 条，去重构筑 11961 套，短名单 5 项。

| 短名单候选 | 筛选胜/完成 | 胜率 |
| --- | ---: | ---: |
| original | 43/48 | 89.6% |
| last-adapted | 45/48 | 93.8% |
| last-base | 40/48 | 83.3% |
| history-bound | 46/48 | 95.8% |
| recent | 41/48 | 85.4% |

| 卡牌 | 原始张数 | 最终张数 |
| --- | ---: | ---: |
| J01 知晓与制衡 | 2 | 0 |
| J02 过往消息皆为利刃 | 2 | 1 |
| J03 优势初成 | 0 | 2 |
| J07 使命必达 | 0 | 1 |
| N07 心有猛虎 | 2 | 0 |
| N08 预备备 | 0 | 2 |
| Y02 策略：「一报还一报」 | 2 | 0 |
| Y07 想做什么梦？ | 0 | 2 |
| Z02 探悉天职 | 1 | 0 |
| Z03 奇异记叙 | 0 | 2 |
| Z05 默示赋命 | 2 | 0 |
| Z06 诓定解放 | 0 | 2 |
| Z07 休息日 | 2 | 0 |
| Z08 倾世之雨 | 1 | 2 |

### weave-rush 最终选择：original

- 数值模型 SHA：`2020f91830f5c397776965dcbc0b6661e1b8e138c07179e0bf29a92fa8640ba7`；来源 `weave-rush/base-export`。
- 搜索记录 17824 条，去重构筑 16933 套，短名单 5 项。

| 短名单候选 | 筛选胜/完成 | 胜率 |
| --- | ---: | ---: |
| original | 47/48 | 97.9% |
| last-adapted | 45/48 | 93.8% |
| last-base | 47/48 | 97.9% |
| history-bound | 44/48 | 91.7% |
| recent | 47/48 | 97.9% |

| 卡牌 | 原始张数 | 最终张数 |
| --- | ---: | ---: |
| 配牌保持原始预组 | 32 | 32 |

## 本轮选定配牌：逐卡互搏摘要（学习方 A，待补双方统计）

2026-09-16 更正：原主表误用了变化规则 AI 对局，已撤换。下表只取 Mac 已保存的最终同模型、同构筑镜像汇总，使用学习方 A 的记录，各位置 32 场。它不是双方完整统计；Windows 已下线，双方原始 telemetry 暂不可读，完整重算及新版交付包待 Windows 上线后补齐。原归档 ZIP 保留历史内容，不能用其中旧逐卡主表作为互搏结论。

对应附件：`selfplay-card-metrics-learner-only.csv`。32 个配对种子共 64 场；A 先手和 A 后手各 32 场。成功打出只计原卡，胜率按对局去重，多次使用另计次数。未携带或未打出的卡没有使用胜率，不能声称已覆盖全卡池。使用胜率是关联，不能直接归因为卡牌强弱。

| 套牌 | A 的位置 | 卡牌 | 张数 | 见到局数/总数 | 打出局数 | 打出时胜率 | 平均首次打出回合 | 使用局平均结束回合 |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| starter | first | N08 预备备 | 2 | 23/32 | 15 | 46.7% | 14.20 | 23.13 |
| starter | first | N04 不是闯祸精 | 2 | 23/32 | 20 | 55.0% | 10.00 | 22.30 |
| starter | first | N03 要叫大姐头 | 2 | 24/32 | 23 | 47.8% | 7.87 | 21.57 |
| starter | first | N02 第二位成员 | 2 | 27/32 | 24 | 41.7% | 7.33 | 21.96 |
| starter | first | Y08 错误的门 | 2 | 26/32 | 13 | 30.8% | 15.31 | 21.85 |
| starter | first | Y03 策略特性：善良 | 2 | 30/32 | 30 | 40.0% | 9.20 | 21.00 |
| starter | first | Y06 策略特性：宽容 | 2 | 23/32 | 21 | 52.4% | 8.33 | 21.76 |
| starter | first | Y07 想做什么梦？ | 2 | 23/32 | 22 | 45.5% | 10.00 | 21.68 |
| starter | first | Z04 未决迷数 | 2 | 29/32 | 29 | 44.8% | 8.24 | 21.10 |
| starter | first | Z08 倾世之雨 | 2 | 24/32 | 21 | 52.4% | 11.95 | 22.33 |
| starter | first | Z06 诓定解放 | 2 | 27/32 | 19 | 47.4% | 13.84 | 21.89 |
| starter | first | Z03 奇异记叙 | 2 | 29/32 | 29 | 44.8% | 11.28 | 21.14 |
| starter | first | J08 现实避难所 | 2 | 26/32 | 11 | 63.6% | 14.09 | 23.27 |
| starter | first | J03 优势初成 | 2 | 25/32 | 19 | 57.9% | 13.11 | 21.63 |
| starter | first | J04 出手即致命 | 2 | 25/32 | 19 | 47.4% | 12.47 | 22.84 |
| starter | first | J02 过往消息皆为利刃 | 1 | 16/32 | 2 | 50.0% | 16.00 | 23.50 |
| starter | first | J07 使命必达 | 1 | 19/32 | 10 | 70.0% | 16.40 | 23.70 |
| starter | second | N08 预备备 | 2 | 24/32 | 11 | 45.5% | 13.45 | 21.09 |
| starter | second | N04 不是闯祸精 | 2 | 24/32 | 24 | 54.2% | 10.25 | 20.04 |
| starter | second | N03 要叫大姐头 | 2 | 24/32 | 24 | 62.5% | 7.17 | 19.88 |
| starter | second | N02 第二位成员 | 2 | 25/32 | 23 | 56.5% | 7.22 | 20.17 |
| starter | second | Y08 错误的门 | 2 | 27/32 | 14 | 42.9% | 14.14 | 22.00 |
| starter | second | Y03 策略特性：善良 | 2 | 29/32 | 28 | 50.0% | 9.57 | 19.64 |
| starter | second | Y06 策略特性：宽容 | 2 | 24/32 | 22 | 50.0% | 8.91 | 20.50 |
| starter | second | Y07 想做什么梦？ | 2 | 23/32 | 23 | 60.9% | 10.43 | 20.04 |
| starter | second | Z04 未决迷数 | 2 | 25/32 | 25 | 48.0% | 8.24 | 20.60 |
| starter | second | Z08 倾世之雨 | 2 | 24/32 | 21 | 52.4% | 9.90 | 19.33 |
| starter | second | Z06 诓定解放 | 2 | 26/32 | 21 | 47.6% | 11.81 | 19.38 |
| starter | second | Z03 奇异记叙 | 2 | 30/32 | 26 | 50.0% | 11.31 | 20.12 |
| starter | second | J08 现实避难所 | 2 | 25/32 | 7 | 57.1% | 10.86 | 21.86 |
| starter | second | J03 优势初成 | 2 | 22/32 | 15 | 60.0% | 12.93 | 21.07 |
| starter | second | J04 出手即致命 | 2 | 25/32 | 16 | 37.5% | 12.12 | 19.50 |
| starter | second | J02 过往消息皆为利刃 | 1 | 13/32 | 1 | 0.0% | 20.00 | 33.00 |
| starter | second | J07 使命必达 | 1 | 17/32 | 9 | 77.8% | 14.00 | 21.11 |
| weave-rush | first | M01 特遣行动 | 2 | 24/32 | 24 | 54.2% | 7.42 | 19.25 |
| weave-rush | first | M03 全勤奖金 | 2 | 26/32 | 19 | 52.6% | 11.21 | 19.26 |
| weave-rush | first | M04 午休时间 | 2 | 26/32 | 25 | 52.0% | 9.32 | 19.56 |
| weave-rush | first | M08 开始净空 | 2 | 26/32 | 13 | 76.9% | 13.31 | 20.46 |
| weave-rush | first | B01 唷 队长！ | 2 | 26/32 | 26 | 57.7% | 6.08 | 19.69 |
| weave-rush | first | B02 招魂幡 | 2 | 22/32 | 22 | 63.6% | 9.00 | 20.00 |
| weave-rush | first | B06 于五色为白 | 2 | 25/32 | 25 | 48.0% | 12.12 | 19.32 |
| weave-rush | first | B08 茶话会 | 2 | 24/32 | 10 | 80.0% | 11.80 | 21.00 |
| weave-rush | first | Z03 奇异记叙 | 2 | 23/32 | 17 | 58.8% | 11.00 | 20.06 |
| weave-rush | first | Z04 未决迷数 | 2 | 27/32 | 21 | 66.7% | 11.67 | 20.19 |
| weave-rush | first | Z05 默示赋命 | 2 | 27/32 | 26 | 46.2% | 7.31 | 19.00 |
| weave-rush | first | Z08 倾世之雨 | 2 | 23/32 | 21 | 66.7% | 9.29 | 20.14 |
| weave-rush | first | Y01 重复囚徒困境 | 2 | 28/32 | 20 | 55.0% | 12.00 | 20.35 |
| weave-rush | first | Y03 策略特性：善良 | 2 | 22/32 | 21 | 57.1% | 9.00 | 20.14 |
| weave-rush | first | Y06 策略特性：宽容 | 2 | 25/32 | 22 | 50.0% | 8.00 | 19.45 |
| weave-rush | first | Y07 想做什么梦？ | 2 | 22/32 | 18 | 77.8% | 6.67 | 18.72 |
| weave-rush | second | M01 特遣行动 | 2 | 22/32 | 22 | 50.0% | 6.55 | 17.09 |
| weave-rush | second | M03 全勤奖金 | 2 | 25/32 | 14 | 50.0% | 10.00 | 18.07 |
| weave-rush | second | M04 午休时间 | 2 | 23/32 | 20 | 55.0% | 9.90 | 18.10 |
| weave-rush | second | M08 开始净空 | 2 | 23/32 | 14 | 71.4% | 12.14 | 19.00 |
| weave-rush | second | B01 唷 队长！ | 2 | 25/32 | 24 | 45.8% | 6.83 | 18.29 |
| weave-rush | second | B02 招魂幡 | 2 | 22/32 | 22 | 59.1% | 10.09 | 18.05 |
| weave-rush | second | B06 于五色为白 | 2 | 23/32 | 20 | 45.0% | 10.50 | 18.10 |
| weave-rush | second | B08 茶话会 | 2 | 23/32 | 14 | 50.0% | 8.71 | 17.43 |
| weave-rush | second | Z03 奇异记叙 | 2 | 20/32 | 14 | 50.0% | 10.86 | 19.36 |
| weave-rush | second | Z04 未决迷数 | 2 | 26/32 | 22 | 50.0% | 9.55 | 17.91 |
| weave-rush | second | Z05 默示赋命 | 2 | 24/32 | 24 | 45.8% | 7.42 | 17.71 |
| weave-rush | second | Z08 倾世之雨 | 2 | 24/32 | 20 | 45.0% | 6.90 | 18.10 |
| weave-rush | second | Y01 重复囚徒困境 | 2 | 28/32 | 21 | 28.6% | 11.90 | 18.57 |
| weave-rush | second | Y03 策略特性：善良 | 2 | 24/32 | 21 | 47.6% | 7.71 | 18.71 |
| weave-rush | second | Y06 策略特性：宽容 | 2 | 23/32 | 21 | 42.9% | 8.57 | 18.10 |
| weave-rush | second | Y07 想做什么梦？ | 2 | 21/32 | 20 | 60.0% | 6.70 | 18.00 |

### 终结与环合（同批互搏，学习方 A）

| 测试 | A 的位置 | 角色 | 机制 | 次数 | 使用局数 | 使用局胜率 | 可用角色回合 | 首次使用回合 | 同动作对手生命变化/次 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| starter-selected-mirror | first | nanali | ultimate | 67 | 30 | 43.3% | 71 | 8.87 | 0.00 |
| starter-selected-mirror | first | nanali | harmony | 18 | 14 | 64.3% | 20 | 9.00 | -1.67 |
| starter-selected-mirror | first | iloy | ultimate | 74 | 32 | 40.6% | 86 | 10.81 | 0.00 |
| starter-selected-mirror | first | iloy | harmony | 10 | 10 | 40.0% | 37 | 16.20 | -1.90 |
| starter-selected-mirror | first | zero | ultimate | 72 | 30 | 43.3% | 72 | 9.60 | 0.00 |
| starter-selected-mirror | first | zero | harmony | 16 | 12 | 83.3% | 55 | 16.17 | -4.88 |
| starter-selected-mirror | first | jiuyuan | ultimate | 58 | 27 | 48.1% | 61 | 9.89 | 0.00 |
| starter-selected-mirror | first | jiuyuan | harmony | 8 | 8 | 75.0% | 24 | 17.25 | -4.62 |
| starter-selected-mirror | second | nanali | ultimate | 66 | 31 | 51.6% | 69 | 7.23 | 0.00 |
| starter-selected-mirror | second | nanali | harmony | 17 | 12 | 91.7% | 32 | 13.17 | -4.53 |
| starter-selected-mirror | second | iloy | ultimate | 71 | 31 | 51.6% | 82 | 9.94 | 0.00 |
| starter-selected-mirror | second | iloy | harmony | 4 | 4 | 50.0% | 41 | 13.50 | -0.50 |
| starter-selected-mirror | second | zero | ultimate | 67 | 30 | 53.3% | 67 | 9.13 | 0.00 |
| starter-selected-mirror | second | zero | harmony | 14 | 11 | 81.8% | 62 | 17.09 | -4.14 |
| starter-selected-mirror | second | jiuyuan | ultimate | 48 | 28 | 53.6% | 50 | 10.07 | 0.00 |
| starter-selected-mirror | second | jiuyuan | harmony | 7 | 6 | 66.7% | 32 | 13.67 | -2.57 |
| weave-rush-selected-mirror | first | bohe | ultimate | 43 | 25 | 60.0% | 54 | 10.92 | 0.00 |
| weave-rush-selected-mirror | first | bohe | harmony | 13 | 12 | 75.0% | 30 | 13.83 | -3.62 |
| weave-rush-selected-mirror | first | baicang | ultimate | 48 | 30 | 56.7% | 53 | 9.93 | 0.00 |
| weave-rush-selected-mirror | first | baicang | harmony | 9 | 9 | 66.7% | 12 | 14.56 | -4.44 |
| weave-rush-selected-mirror | first | zero | ultimate | 77 | 31 | 54.8% | 76 | 8.03 | 0.00 |
| weave-rush-selected-mirror | first | zero | harmony | 1 | 1 | 100.0% | 19 | 7.00 | 0.00 |
| weave-rush-selected-mirror | first | iloy | ultimate | 81 | 32 | 53.1% | 96 | 8.31 | 0.00 |
| weave-rush-selected-mirror | first | iloy | harmony | 5 | 4 | 100.0% | 67 | 9.50 | -2.40 |
| weave-rush-selected-mirror | second | bohe | ultimate | 32 | 22 | 63.6% | 39 | 8.91 | 0.00 |
| weave-rush-selected-mirror | second | bohe | harmony | 15 | 12 | 83.3% | 27 | 8.92 | -2.00 |
| weave-rush-selected-mirror | second | baicang | ultimate | 44 | 30 | 53.3% | 48 | 9.93 | 0.00 |
| weave-rush-selected-mirror | second | baicang | harmony | 4 | 4 | 100.0% | 4 | 15.50 | -2.25 |
| weave-rush-selected-mirror | second | zero | ultimate | 66 | 31 | 51.6% | 66 | 8.13 | 0.00 |
| weave-rush-selected-mirror | second | zero | harmony | 6 | 5 | 80.0% | 15 | 8.40 | -3.33 |
| weave-rush-selected-mirror | second | iloy | ultimate | 65 | 31 | 48.4% | 82 | 8.06 | 0.00 |
| weave-rush-selected-mirror | second | iloy | harmony | 11 | 8 | 62.5% | 65 | 10.50 | -1.64 |

同动作生命变化仅为关联证据，不是净因果收益。双方完整统计将使用相同镜像批次，不增加训练或混入规则 AI。

## 曲线与统计限制

- `turn-states.csv` 给出各回合仍在进行的样本量及平均生命、行动力、能量、环合值；没有概率预测。
- `selfplay-card-metrics-learner-only.csv` 是当前更正主表，只有最终镜像学习方 A 的先后手数据；`card-metrics.csv` 保留全部测试诊断，不可将其中规则 AI 记录混入互搏统计。
- `training-win-rates.csv` 保留各次更新按对手类型和先后手的胜率序列，不能混入冻结评估。
- `generated-card-metrics.csv` 单列复制与衍生牌的获得、使用和胜负关联，不混入主牌组抽到率。
- `mechanism-metrics.csv` 包含按角色/先后手的机会、次数和同动作生命变化。跨动作/跨回合的净因果收益尚未归因，不将相邻效果全算作某机制贡献。
- `sample-replay.json` 是已记录对局的公开可播放录像；私有种子及隐藏牌库不在其中。
- 全部测试可执行不等于本轮样本量足以判定强度；未使用的卡牌/机制不判弱。
- 本轮不提供可靠预测胜率、长期收敛结论或机制的因果胜率贡献。
