# 策略学习恢复：残差表示与旧能力保持

本文件说明已实现的研究组件和验收边界。现行规则、正式学习和完整交付仍遵循[训练手册](duel-v2-public-training.md)、[完整评估规范](duel-v2-full-training-report-spec.md)。这些组件没有取代正式训练默认值；实现或小样本记忆通过不能宣称五套已经修复、强度改善或获得网站批准。

## 两个独立问题

1. 网络能否把完整 Gumbel 策略目标拟合成具体动作排序，包括出牌、攻击、结束、牌身份和目标？用固定训练教材、小批记忆、成对根种子隔离的留出分别检验。
2. 迁移和价值学习是否抹掉原有策略？必须比较同一局面所有合法动作分数，不能只看出牌比例或数值权重是否复制。

新的候选表示即使有更好梯度，也可能仅记忆教材，不能泛化。已经查看过的档案留出和确认集仍为已消费数据；不能换一个划分后称为独立强度验收。新模型是否正确判断，需要现行规则的新种子和完整对打。

## 正常幅度的状态条件残差头

`rl/residual_policy.py` 定义 `cross_five_residual_wdl_v2`。继续使用既有可见观察及 `cross_grounded` 候选变换，状态与候选各自经正常幅度的投影、SiLU、两个残差块和 LayerNorm；打分器读取候选、状态及其乘积。残差块末层零初始化，使该块开始时为恒等映射；不把整个候选投影缩小到1%。这不是动作类别与牌身份分开的分层头，也没有出牌加分。

WDL 仍只预测实际完整终局的胜／平／负，价值带观察者是否行动方的上下文。更新直接复用 `outcome_runtime.update` 的 Gumbel 交叉熵和终局 WDL，不新增 PPO 或过程奖励。导出和加载在 `rl/residual_runtime.py`，Torch 与 NumPy 实现相同的 LayerNorm、SiLU、残差及候选掩码；新 schema 绑定结构、候选变换、规则、构筑和数值SHA。旧文件不能直接加载为新结构。

## 精确保留旧能力

`rl/preserved_policy.py` 为创生、覆纹、快攻提供 `cross_five_preserved_residual_wdl_v1`。使用既有 `cross_teacher_distillation` 的具名列适配和来源核验，将旧策略作为冻结缓冲区嵌入，新残差的最后打分层初始为零。因此初始化的全部合法动作 logits 等于旧策略；新的 WDL 主干不与旧策略共享可训练参数，单独更新价值不会改变基线策略。

这与从随机学生重新模仿教师不同：旧能力由冻结分支直接保留，学习只提供局面相关的修正。残差更新仍可能伤害能力，须经独立行为和对局验收后才采纳，失败保留零残差起点。没有针对出牌类别的固定加分。

来源的旧规则SHA保留在 `source_identity`，嵌入的冻结数组逐张量与已验证源数值比较；不得修改源清单或继承源服务批准。旧教师不能表示的行动角色、卡牌、目标或选项明确失败，不伪造旧特征。该分支目前只覆盖三套及旧教师支持的合法动作；真红、浊燃没有旧教师，不能声称被此方法恢复，也不能静默退成三套教师。

## 本机入口与分层验收

以下命令默认只输出计划；`--run` 才执行，输出目录必须新建。网站不导入训练依赖。

```bash
.venv/bin/python scripts/check_duel_v2_residual_learning.py --output artifacts/rl-evals/NEW/offline
.venv/bin/python scripts/check_duel_v2_residual_learning.py --run --memorize --seconds 600 --output artifacts/rl-evals/NEW/capacity
.venv/bin/python scripts/check_duel_v2_residual_learning.py --run --seconds 900 --output artifacts/rl-evals/NEW/offline
.venv/bin/python scripts/check_duel_v2_policy_preservation.py --run --seconds 300 --output artifacts/rl-evals/NEW/preservation
```

- **真实小批容量检查**：每套从允许拟合的训练根确定性抽128行，固定400次更新，只检查是否能拟合；这不是留出结果。`memorization_passed` 不转换为 `offline_fit_passed`。
- **归档留出检查**：原选择集和已经消费的确认种子排除拟合，并为原选择没有覆盖的快攻、浊燃保留成对根种子。所有新策略行完整遍历，末尾小批也参与更新。记录完整目标KL、明确偏好动作命中、具体出牌动作命中、攻击／结束误出牌及梯度。当前预声明的KL和命中阈值是本项目工程检查值，不是论文保证。此检查仍为探索性，不能批准正式长训。
- **精确保留检查**：共享种子账本预约新种子，三套所有无序配对含镜像均交换先手，每个真实决策比较全部合法动作数值。它证明初始化保留能力，不证明训练后超过旧策略。
- **新搜索学习核验**：`rl/recovery_runtime.py` 提供显式研究运行时及有界采集器，调用既有逐决策 `information_search.search`，保存完整 Gumbel `pi`、实际选择、全动作Q诊断、观察者终局WDL和真实动作序列；截断局不返回学习样本。新WDL未通过预检不能进入搜索学习；仅具名的未校准诊断可观察它，不能生成训练许可。该采集器没有正式训练编排、恢复或完整交付批准，不能冒充正式五套流程。

测试命令：

```bash
.venv/bin/python -m unittest tests.test_duel_v2_residual_learning tests.test_duel_v2_outcome_learning tests.test_duel_v2_gumbel_search tests.test_duel_v2_cross_teacher
```

覆盖同候选不同状态分别选择攻击／结束／两张不同牌、梯度、候选置换与填充不变、Torch／NumPy对拍、严格数值身份、恢复、冻结旧策略和价值更新不改变基线。单轮报告、模型和原始记录只能进入 `artifacts/rl-evals/<run-id>/`。

## KataGo 的可借鉴部分

[KataGo 官方方法](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)说明了残差结构与初始化、搜索策略目标、惊讶样本加权和混合搜索预算等方法。本组件借鉴稳定残差表示的方向，不声称复现其围棋网络或训练加速效果。KataGo 的历史 Fixup 初始化与当前方法不同；这里使用 LayerNorm 的小型MLP，是本项目适配。

后续应单独检查价值校准、搜索目标稳定性和每份经验的利用，而不是一次性加入全部技巧。我们的 Gumbel 目标并非 KataGo 的访问计数目标，不能直接套用其目标修剪。Mctx 的 [Q变换实现](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/qtransforms.py)提供可选的局部极差缩放；本项目沿用的缩放会放大微小Q差，是需要新根对照的风险，尚未据此改写正式算法。

## 明牌的边界

明牌可能减少响应和隐蔽世界估值的不确定性，但不能修复策略梯度、表示和排序故障。正式策略看到对方手牌会改变信息契约；本组件没有开放明牌。仅训练教师使用完整状态、学生仍使用合法观察的方案也需独立协议：相同可见局面下，教师可能因不同暗牌给出相反动作，学生只能学习信息集下的统计决策，不能要求它恢复看不到的具体暗牌。

明牌／暗牌应按同规则、同构筑、同计算和优化预算分别对照；首先测学习质量与真实胜负，不能把教师看牌后更强写成普通暗牌学生已经更强。
