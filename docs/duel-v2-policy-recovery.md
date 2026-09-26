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

## Windows离线时的本机研究与预检

新增下列独立研究入口，均不连接Windows或替换网站模型。单轮数字只写本轮artifacts；启动实际采集前继续分配共享账本的新种子和当日带日期09:00硬截止。

| 入口 | 作用与限制 |
| --- | --- |
| `audit_duel_v2_learning_data.py` | 检查同特征候选、最低不可消除KL和重复可见根的标签分歧。相同特征不等于规则等价，不静默合并动作。 |
| `check_duel_v2_numeric_inference.py` | 历史NumPy、独立float64/einsum、Torch三方对拍，记录警告及数值幅度上界，不把警告直接当成真实溢出或擅自改变旧推理。 |
| `check_duel_v2_value_calibration.py` | 归档留出按完整局等权的WDL对数损失、Brier和分桶校准；已经消费的旧留出不批准现行搜索。 |
| `check_duel_v2_search_stability.py` | 现行规则的新根重复Gumbel32，保存完整Q、访问、实际选择和策略目标。区分实际选择、pi首选和表示组。 |
| `check_duel_v2_q_scale.py` | 在已消费诊断根上，同预算重新执行两种完整搜索；旧分支须与共享实现逐项一致，不能只在原树上重算标签后宣称另一算法更强。 |
| `check_duel_v2_full_hand.py` | 默认`--visibility root_only`只让搜索方看到当前对手手牌，对手策略仍用暗牌观察；`both`为双方内部观察均展开。牌库顺序仍采样，不改变真实状态／公开API，不生成训练许可。未按明牌分布训练的数值只能做敏感性诊断，不能证明明牌训练收益。 |
| `check_duel_v2_recovery_value.py` | 三套冻结旧策略的现行规则完整局，只训练新WDL头，逐张量核验策略与主干不变。`--train-replicas 8 --calibrate`使用独立selection种子在预声明温度1／2／4间选择，再用confirmation一次核验；确认数据不回灌。温度烘焙进WDL权重，绑定实际数组SHA。 |
| `check_duel_v2_recovery_chain.py` | 通过匹配数组的WDL预检后，执行三套有界Gumbel32采集、策略行完整遍历、非行动方终局价值更新、数值导出与独立纯网络对打。`--episodes`只复用同源已完成TRAINING局并另分配新确认种子，不重复计算新经验。不代替五套完整流程。 |
| `prepare_duel_v2_offline_readiness.py --prepare` | 生成普通源码／测试／说明的SHA快照和CUDA预检计划，不打包数据库、环境文件、凭证、无关未跟踪文件、外部符号链接或可发布模型，不部署或启动正式训练。 |

### 自然WDL尺度搜索分支

`rl/recovery_search.py` 提供隔离的 `natural_wdl` 分支，复用共享的隐藏世界采样、Node、固定观察者价值、对手私有策略与短终局证明；顺序减半预算不变。其Q变换为 `0.1 × (50 + max_visit) × completed_raw_Q`，不按当前候选的局部极差除法放大；对应Mctx提供的`rescale_values=False`方向。只允许纯WDL范围[-1,1]，没有机制奖或出牌加分。

局部Q差很小时，旧极差缩放可能把估值噪声变成很尖的监督目标；自然尺度会保留其原始幅度。但更稳定、接近先验甚至均匀的目标未必有学习信息。因此必须先保留可用基础策略、预热并核验价值，再检查真实对局。不能仅依据重复目标的JS下降晋级。旧正式算法和运行源码不因该研究分支改变，CUDA及五套强度验收未完成前不切换默认值。

### 恢复、数值与CUDA边界

短链路保存`recovery_restore_check_v1`网络／Adam／采样RNG恢复点。`recovery_checkpoint.py` 使用`weights_only=True`，绑定规则、算法、源数组SHA和实际张量；恢复后的下一次Adam更新与控制副本逐张量一致，测试不改变导出的候选。它是恢复检查，不是自动续跑，不延长旧截止、不继承服务批准。

导出还须在真实策略行上核验Torch／NumPy分数及实际最高分动作一致。`--device cuda`只改变更新设备，采样仍走正式Python规则与NumPy；CUDA不可用必须在创建输出或分配种子前失败，不退回CPU。CPU结果不能代替CUDA数值、梯度、恢复与实际吞吐检查。

新的WDL预检报告绑定本次实际权重。联合更新后的候选权重SHA已经不同，不能继续沿用起点的通过报告当作候选价值已校准；下一次搜索学习必须重新核验。同构筑短测镜像4/8或几场出牌正常都不批准强度，仍需五套、历史对手、独立种子及完整评估规范。

## 正式训练启动状态与剩余接入

离线组件和三套短测完成不等于Windows上线前的全部修复已完成。`train_duel_v2_current_round.py`仍使用旧`cross_runtime`，没有接入恢复网络、自然WDL尺度、匹配权重的价值预检或完整五套新栈。普通CLI、`--worker`入口和直接`learn()`调用均拒绝非smoke正式执行，不能靠修改config或提供吞吐数字绕过。

正式接入之前仍需完成：五套初始策略／价值与源身份协议；包含真红、浊燃的端到端短测；独立首批行为和同条件新旧策略检查；多批重新校准、择优、回退和截止编排；CUDA恢复／梯度／数值及正式预算吞吐验收。旧smoke只诊断旧链路，不是新栈验收。

恢复旧能力、参数变化和标签损失略降不是胜过旧策略的证据。阶段报告必须核对独立对局中的同局面实际动作是否变化、变化是否有真实收益，以及新旧完整对打；若新策略仍逐步选择相同动作，只能报告能力保持或尚无行为进展，不能报告更强模型。不能通过强行增加出牌率或放大Q让动作不同来冒充提升。
