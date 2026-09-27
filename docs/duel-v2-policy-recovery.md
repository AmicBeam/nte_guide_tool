# 策略学习恢复：残差表示与旧能力保持

本文件说明已实现的研究组件和验收边界。现行规则、正式学习和完整交付仍遵循[训练手册](duel-v2-public-training.md)、[完整评估规范](duel-v2-full-training-report-spec.md)。五套入口已提供显式恢复栈并默认选择该栈；实际正式启动仍要求同源码CUDA32预检和容量证据。实现或小样本记忆通过不能宣称强度改善或获得网站批准。

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
- **新搜索学习核验**：`rl/recovery_runtime.py` 提供显式研究运行时及有界采集器，调用既有逐决策 `information_search.search`，保存完整 Gumbel `pi`、实际选择、全动作Q诊断、观察者终局WDL和真实动作序列；截断局不返回学习样本。新WDL未通过预检不能进入搜索学习；仅具名的未校准诊断可观察它，不能生成训练许可。单独采集器不提供完整交付批准；五套编排由 `rl/recovery_round.py` 负责，见文末。

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

局部Q差很小时，旧极差缩放可能把估值噪声变成很尖的监督目标；自然尺度会保留其原始幅度。但更稳定、接近先验甚至均匀的目标未必有学习信息。因此必须先保留可用基础策略、预热并核验价值，再检查真实对局。不能仅依据重复目标的JS下降晋级。旧 `cross` 算法保留；恢复入口明确使用自然尺度，CPU短测不能批准正式CUDA长训或强度。

### 恢复、数值与CUDA边界

短链路保存`recovery_restore_check_v1`网络／Adam／采样RNG恢复点。`recovery_checkpoint.py` 使用`weights_only=True`，绑定规则、算法、源数组SHA和实际张量；恢复后的下一次Adam更新与控制副本逐张量一致，测试不改变导出的候选。它是恢复检查，不是自动续跑，不延长旧截止、不继承服务批准。

导出还须在真实策略行上核验Torch／NumPy分数及实际最高分动作一致。`--device cuda`只改变更新设备，采样仍走正式Python规则与NumPy；CUDA不可用必须在创建输出或分配种子前失败，不退回CPU。CPU结果不能代替CUDA数值、梯度、恢复与实际吞吐检查。

新的WDL预检报告绑定本次实际权重。联合更新后的候选权重SHA已经不同，不能继续沿用起点的通过报告当作候选价值已校准；下一次搜索学习必须重新核验。同构筑短测镜像4/8或几场出牌正常都不批准强度，仍需五套、历史对手、独立种子及完整评估规范。

## 五套恢复栈与启动闸门

`train_duel_v2_current_round.py` 默认 `--runtime recovery`，实际调用 `rl/recovery_round.py`。`--runtime cross` 仅保留历史smoke，旧正式入口继续拒绝。新栈固定创生、覆纹、快攻、真红、浊燃配牌，不换角色、不开明牌、不加出牌奖励、不替换网站模型。

### 五套起点与未知目标

五套恢复栈使用 `rl/recovery_policy.py` 的独立价值主干：策略的 `state_net/cand_net/score` 与价值的 `value_net/wdl` 没有共享可训练参数。真实搜索行仍同时提供策略和终局WDL监督，但价值-only更新和WDL预热不再改变策略。新数值schema绑定独立价值结构SHA；历史共享主干schema保留原数学路径，不能悄悄升级其旧文件。

前三套使用新schema `cross_five_decoupled_partial_preserved_wdl_v1`：旧教师能够表示整个根的合法候选时，初始化全部分数与旧策略一致。只要根内有不能表示的角色／目标／卡牌，**整个根只使用新残差分支**，不把未知身份伪装为旧列，也不混合不同分数尺度。独立行为报告记录 `unsupported_legacy`；这些局面是新学习范围，不能声称保住了不存在的旧能力。旧严格schema仍遇到未知动作就拒绝。

真红与浊燃使用 `cross_five_decoupled_residual_wdl_v1`，显式创建正常幅度残差网络，来源标明 `fresh_residual_no_teacher`；不继续迁入已经发现不出牌的旧瓶颈网络。可通过 `--recovery-source` 续接已验证的五套恢复数值，构筑、规则、schema、实际张量及SHA全部必须一致；优化器与回放重置，不继承质量批准。

### 每批真实执行

1. 初始及每批更新后，先冻结策略，采集五套完整纯网络局，只预热WDL头。预热训练、温度选择和留出使用互不重叠的种子；选择温度1／2／4后另做一次留出。按完整局等权计算Brier／对数损失，报告绑定实际数组SHA。失败不再搜索学习，保留已验证 `best`。
2. 正常批次为20个跨队局与10个镜像局，显式交换先手。每第四批引用冻结 `best` 历史对手，两个座位分别作为当前学习方，共60局；仅收当前方样本。搜索逐决策使用自然WDL尺度Gumbel32。截断和搜索不完整不给终局标签、不进入学习。
3. 全部新策略行先完整通行，再按预声明1／4／8次数有界重复；非行动方真实终局观察另做价值更新。记录逐行使用、三个策略分支的有限非零梯度及数值/最高动作对拍，不以参数变化代替学习。
4. 每批都做独立纯网络初筛与第二块种子复核，对相同冻结对手交换座位和先手，并逐决策对比同局面新旧动作。正式阶段再做同预算搜索初筛和另一块复核。未确认实际收益不晋级，连续两批有效检查无实质改善则停学；不出牌、梯度／覆盖失败或两块均显著退化按P0停止并真实回退到 `best`。
5. 网络、Adam、经验池使用计数、NumPy／Torch／CUDA RNG、轮次、best/latest指针及原截止进入两代恢复点。下一次Adam更新在副本上逐张量核验，不改候选。恢复只能在原学习截止内、同配置同源码下继续；不重新分种子、不延长截止。
6. 停学后冻结所选 `best` 的实际数值，分别执行纯网络与搜索5×5主矩阵。复用正式遥测与动作重放导出公开录像。保留初始权重时明确 `retained_baseline`，不能把旧数组写作全套新模型；主表通过也不代替完整报告规范的变化构筑、换牌及其他专项。

阶段选择是预声明的保守经验规则，不是统计显著性或强度批准。胜率样本、同根相关性和缺项都保留原始记录；选择局和价值留出不进入策略回放，也不填最终确认矩阵。

### 本机与Windows入口

```bash
# 本机五套小预算链路，Gumbel2；不是正式搜索预算验收
.venv/bin/python scripts/train_duel_v2_current_round.py --smoke --device cpu --workers 2 --hidden 32 --seconds 1800 --output artifacts/rl-evals/NEW/smoke --run

# Windows上线后在冻结工作副本执行：五套Gumbel32、实际CUDA梯度/数值/Adam恢复及阶段成本
python scripts/train_duel_v2_current_round.py --preflight --device cuda --workers 6 --output artifacts/rl-evals/NEW/preflight --run

# 中断后仅在原学习截止内、原配置与进程已退出时恢复
python scripts/train_duel_v2_current_round.py --output artifacts/rl-evals/EXISTING --resume --run
```

正式预算预检另用 `recovery_capacity.py`：使用实际完整TRAINING局，并另采集相同对手／先后手分布的新成对种子完整局，作为不入回放的容量留出；两段都采用同预算及探索条件。初始网络副本拟合最多128条训练根，留出最多256条；默认200次更新、学习率0.003。只检验容量与标签泛化，不改变真实训练网络，不把这些训练留出称为独立强度确认。训练KL<0.025、明确偏好动作准确率≥95%，留出KL<0.15、明确偏好动作准确率≥75%，以及三个策略分支的非零有限梯度是预声明工程条件；不是论文保证。没有明确偏好标签时单列样本数，不虚构动作准确率。CUDA正式放行要求该检查通过。

早期按种子排序把整个对手类别留出的检查，只作未见对手外推诊断，不与同分布新种子容量验收混算。`check_duel_v2_recovery_capacity.py` 可复用严格同源的已完成TRAINING局，另分配同对手分布的新留出，不重开已停止训练、不改写旧失败报告、不更新真实模型。新诊断单独解析自己的有界窗口与当日09:00硬截止；精确恢复训练仍沿用原截止，不把过期训练误当作诊断恢复。留出虽用相同探索噪声生成，明确 `training=false`，不能因拥有pi与终局标签而进入策略回放。

预检只生成 `readiness.json`，不批准质量或网站。正式启动必须提供该报告和报告中的 `initial_folder`：同源码、规则、五套数值SHA、CUDA设备、32模拟、进程数、24小时内实测、原始证据SHA及容量均重新核验。CLI、worker与直接learn都检查，不接受手填“通过”布尔值。CPU预检不能代替CUDA。没有CUDA时，在创建输出和分配种子前失败。

正式预算必须排得下预热、真实轮转、所有阶段选择、两张独立矩阵、至少25%容量余量及15分钟保存交付。默认硬截止仍按本轮带日期的北京时间09:00解析；不足则拒绝正式任务，不能沿用历史吞吐或缩减计划冒充完整长训。

验证：`.venv/bin/python -m unittest tests.test_duel_v2_recovery_round tests.test_duel_v2_current_round tests.test_duel_v2_residual_learning`。单轮完成数、预检／失败与剩余条件写入artifacts，不回写成永久强度结论。

恢复旧能力、参数变化和标签损失略降不是胜过旧策略的证据。新策略动作不变时只报告能力保持；动作不同但未获独立对局收益时只报告行为变化。禁止通过固定出牌加分或放大Q制造变化。Windows真实CUDA预检和五套强度验收尚不能由本机测试替代，也不保证下一轮一定提升。
