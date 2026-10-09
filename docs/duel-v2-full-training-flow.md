# 完整训练的轻量 Agentic Flow

本文件维护可复用流程，不保存单轮结果。规则、学习方法、配牌接受及完整报告分别遵循现有设计手册、公开训练手册、配牌协议和完整报告规范。Flow 不提供新的训练CLI，也不代替现有正式引擎或学习执行器。

## 来源与职责

参考 vca-harness-runtime 的 agentic-review-loop：参考版本为04d9b99cd7bbde1511d9224f81021eb7550294e6，具体来源是 skills/l3-omh-playbooks/playbooks/agentic-review-loop.omh_workflow.yaml、stages/vca-omhstage-autonomous-delivery/docs/agentic-main-delivery-workflow.md，以及 references/main-agent-method.md、references/fresh-review.md。沿用一个持续主 agent、内部有界 worker、证据账本、根因修复回路与全新最终审计。没有安装新的OMH Runner、复制OMH私有状态或修改参考仓库。

主 agent持续拥有任务语义、执行和验收。步骤名是训练证据索引，不是独立的OMH Stage或第二个自动路由中心；主 agent根据数据依赖、预算和当前证据决定下一项能力，普通实现或测试失败由主 agent修复。Worker给出结果和证据，不能改变用户范围、延期、降低P0门槛或自行批准模型上线。Fresh Reviewer是独立只读能力，发现回到主 agent处理。

项目入口为 .agents/skills/duel-v2-full-training-flow/SKILL.md。Python API位于 app/modules/card_game/rl/training_flow.py，子 agent交接与报告由 training_flow_agents.py、training_flow_report.py 提供。导入、创建草稿或加载已有账本不启动训练。所有单轮请求、交接、结果、模型、日志与报告写入 artifacts/rl-evals/<run-id>/，不提交Git。

## 开始前的三项确认

通过同一批用户问题确认：

1. 是否包含修改主牌组的换牌候选。可说明只让机制改动角色各自8张参与，其余锁定；起手换牌是另一项策略动作，始终按现行手册处理。
2. 截止时间是否是北京时间09:00。必须给出完整日期和Asia/Shanghai；已过当日09:00可提出下一日09:00，但不能自动接受。确认后冻结绝对截止，失败或重启不重新取得预算。
3. 训练的高级人机范围及每队实际构筑。列出角色及顺序、32张牌、起点模型／账号构筑来源、可变角色范围、暂停或未准备好的队伍。不能仅填写“全部”而不解析本轮实际名单。

没有明确回答的项保持待确认。推荐选项、预选项、超时或旧任务范围都不是本轮批准。已确认的同轮信息通过原消息／答案引用保存，不再重复问；用户改变这三项中的内容时生成新合同并使相关旧证据失效。

## 拆解与完整性

| 证据步骤 | 主职责与输出 |
| --- | --- |
| confirmation | 三项明确回答、用户引用、规范化构筑和请求摘要 |
| freeze | 原模型／构筑／规则／观察／采样源码SHA、独立种子、执行器能力及预算 |
| preflight | 同源真实搜索、CUDA更新、数值／Adam／RNG恢复、WDL和P0闸门、实测吞吐 |
| base_training | 现行信息集Gumbel／终局WDL，完整训练局及每队实际策略更新 |
| candidate_search | 规范化单换／双换、分级筛选、父候选、换入换出及分组分母 |
| baseline_adaptation | 原配牌从共同权重按预定等量优化步数适应 |
| candidate_adaptation | 锁定候选从相同权重、冻结对手及等量步数适应 |
| build_confirmation | 独立种子及预声明统计接受；缺项或证据不足回退 |
| build_audit | 未消费审计种子及原始策略／配牌对照；只有否决权 |
| freeze_selection | initial、selected、latest分开，策略数组差异、模型／配牌与WDL身份 |
| evaluation | N×N纯网络／搜索、真实镜像、新旧交叉、全留起手、变化对手，原始记录与公开录像 |
| analysis | 逐卡与角色、召唤物、显式自动释放／实际使用、终结／环合，分母与去重 |
| report | 中文完整报告、实际牌表与候选决定、未完成项，全部当前身份和引用 |
| delivery | 真实数值模型、恢复点、原始记录、公开录像实际回传与成员SHA |

只有用户明确不含候选，candidate_search至build_audit才记为不适用；其他步骤不能因预算紧张或执行器缺失静默跳过。接受失败而正确回退是执行结果；从未做适应／确认是未完成，两者分开。主 agent在任何昂贵执行前核对全套必要能力已接入，不能用“候选初筛 → 固定配牌训练”冒充包含候选的完整任务。

上述次序展示数据依赖，允许主 agent根据事实安排并行准备、复用有效证据、分段执行和针对性恢复。跨队、先后手、种子、构筑与样本量按本轮计算，不能硬编码历史五队或七队。任何后端不支持本轮角色、动作或规则时应明确阻断，不静默映射到旧契约或PPO。

## 子 Agent 与交接

默认使用已经配置的Grok隔离worker；启动一个步骤时提供本轮确认合同、有效上游摘要、证据路径及SHA、精确文件／输出目录所有权、绝对截止及步骤完成条件。不要传递整段聊天、凭据、账号私有数据或完整私有训练经验。需要这些资料的核验及已授权Windows操作由主 agent的本地／远端工具完成，子 agent接收经过筛选的非敏感汇总。用户显式选择其他agent时通过同一交接接口注入runner，不改变训练协议。

每次有界任务使用独立attempt与token，记录请求摘要和输入证据。结果包含passed／failed／blocked、可独立理解的summary、真实产物路径及SHA和指标。退出0、口头“通过”、文件名ready或只有JSON清单不算训练／交付证据。重复活进程不再启动；加载仅恢复已保存事实。失败结果保留，新尝试不覆盖旧报告或已消费独立种子。

### Python调用与证据字段

主 agent通过既有Python环境导入TrainingFlow，先create草稿、confirm本轮明确用户receipt，再用start／finish登记一个有界任务。confirmation保存source=user与原消息reference；候选开启时mutable_characters也须与草稿一致。无论逐步调用还是run，开始前都要用configure_executors登记完整必要回调；run会自行做此检查。通过bind_stage_handlers(flow, runner=...)可绑定交接回调，但回调存在不表示CUDA或算法能力已通过，主 agent仍须先完成真实执行器检查。默认runner是既有Grok隔离worker，以本attempt输出目录为工作区，普通仓库源码只读，不引入新的训练命令行。指定其他agent时必须显式提供其runner。

核心校验的最小结果字段如下，详细分析仍来自现有实际执行器：

- 基础学习：complete_training_games为正整数或逐队计数，formal_policy_updates覆盖本轮所有学习队且逐项为正。
- 启用候选的freeze：预声明adaptation_target_steps。两种适应的metrics.adaptation保存相同initial_sha256、opponents_sha256、seed_allocation_sha256，以及optimizer_steps；实际逐队步数须等于原声明，不能按各自运行时间充当等量。
- 候选搜索必须保存实际candidate_changes：每条包含key、完整build、removed和added，牌数差异与确认起点一致、四人顺序不变，只改mutable_characters。独立确认与审计保存decision=accept／retain_baseline／insufficient_evidence、reason、正数normal_games及独立seed_allocation_sha256，审计不能复用确认或适应种子身份。
- 评估：planned_cells／completed_cells按本轮N×N计算；matrix_modes必须同时有pure和search，每种模式包含完整cells，逐格保存row、column、planned、completed、first_wins、second_wins、draws、errors、truncated。注册真实matrix逐局JSON／JSONL／gzip记录，核心按实际first及winner复算分母，不能只提供汇总布尔值。
- 模型：冻结选择的models保留每队initial／selected／latest真实身份。交付必须有本轮每队实际NPZ／PT等数值文件、原始记录及公开录像，路径在运行目录中，文件及SHA实际相符。NPZ必须包含完整数值张量，改后缀的JSON或只有清单不接受。
- 子agent结果必须返回当前stage_id、attempt、token和request_sha256；不同attempt的输出不能沿用。失败退出／超时不能靠JSON中的passed收口；同一attempt不得重复派发。
- 逐卡／异能者／召唤／实际使用／自动释放／终结环合等分析专题通过metrics.analysis各项的artifacts绑定实际已登记表格及SHA。complete=true或rows=0不能代替文件；零观测可保留，但须有实际表格与覆盖说明。每次report完成时生成并登记canonical-flow-report.md，展示当前数据、两张实际矩阵、完整配牌、文件索引及缺项，不能只提交note.txt作为完整报告。

步骤pass只表示其已登记证据，未完成完整报告内容或独立审计时workflow_complete仍为false。字段shape验证不替代同源CUDA、实际Gumbel、数值动作对拍、P0、统计接受及网站批准；不能用合成结果启动正式训练。

主 agent用invalidate_evidence记录源码或证据变更的理由，失效相关步骤与下游及审计，保留每次历史attempt和原截止。任何活任务先核验并收敛其进程，再失效，不通过重写state.json或复用旧passed恢复。截止后的report由本地主agent整理已有文件，不启动新的子进程；delivery需主agent按实际已有回传证据登记，未取得资料保持blocked／未完成。

现有能力可按需组合：next_training处理本轮名单，offline_sources冻结源码，local_card_search枚举限定角色候选，seven_round／recovery_round提供搜索学习、更新与恢复能力，build_acceptance提供接受统计，full_cycle_experiment.evaluate_job采集真实冻结对局，new_model_report生成真实N×N及逐卡报告。现有五队CLI及旧full-cycle的能力边界继续保留；它们不会因Flow存在而自动支持任意名单或现行七队完整流程。主 agent先核对所选源码、模型schema、观察及每个步骤能否覆盖用户范围，必要时由有界worker实现并验证所缺连接，不减少名单或换算法凑完成。

主 agent解释结果证明了什么、没有证明什么、使哪些证据失效。普通仓库缺陷和可恢复环境问题在任务内部处理；确实缺少用户语义、外部访问或新授权时才询问具体问题。源码／运行配置变化必须新冻结并核对受影响证据，原截止不变；不热改在途批次。

## 预算、报告与最终审计

开训前为预检、学习、候选公平适应、确认／审计、完整冻结评估、报告和回传分配绝对阶段预算。采用本轮真实吞吐及至少25%余量，并预留回传校验。预算或执行器不足时记录缺项并向用户说明；只有用户改变范围或截止才能变更任务，不通过降低接受阈值或把必需步骤改不适用继续。

报告须覆盖完整报告规范：每队完整32张牌和角色顺序、候选换入换出与决定、真实正式局及更新、initial／selected／latest身份、先手行后手列N×N两种推理表与分子分母、镜像和新旧及起手对照、变化对手、逐卡／角色／召唤物／终结／环合、典型原始记录与公开录像。机会与事件次数分别标明。自动释放不混作实际使用，生成／复制不混作原牌抽到率。缺数据写未完成，不填0%／50%或伪造预测胜率。

所有实际模型、原始记录、录像和报告回传后，使用全新只读reviewer做最终审计，重点找遗漏步骤、假完成、陈旧模型／构筑身份、错误分母、误计使用及只回传清单。审计绑定当前确认合同、模型与证据SHA；修复使依据失效时重新审计。只有当前证据与审计共同覆盖用户任务，主 agent才呈现完成。执行完整性、complete_report、quality_approved和发布授权独立，Flow不自动改网站模型、推送或部署。

audit_basis返回当前已注册产物及步骤状态的摘要。全新reviewer的结果保存为有SHA的JSON，包含verdict=allow／remediate／blocked、reviewer_session、request_sha256、basis_sha256；主 agent用record_final_review明确采纳。引用文件、审计文件或合同改变会使完成状态失效。先冻结并注册待审报告，审计后另写最终呈现文件，不覆盖已注册证据。flow_terminal只表示本轮收口到终态；workflow_complete要求完整步骤、语义报告数据、实际交付及有效allow审计共同满足。合成收口夹具通过始终不提供quality_approved。

到截止后只能停止并保全／读取已有资料，不开展新的学习、适应或独立评估。不完整结果仍应回传，列出真实失败、截止和缺项；不能通过新flow或恢复将原9点延期。完整训练失败不能改写成平台期、收敛或质量通过。

## 验证

使用现有Python测试入口：.venv/bin/python -m unittest tests.test_duel_v2_training_flow tests.test_duel_v2_training_flow_agents tests.test_duel_v2_training_flow_report tests.test_duel_v2_training_flow_integration。确认、范围、时间、候选必需阶段、缺执行器、等量适应、真实原始分母、SHA／路径／attempt、重复执行、当前审计及不完整报告使用合成夹具；合成全链路通过只证明编排，不是CUDA／Gumbel强度或正式训练质量证据。
