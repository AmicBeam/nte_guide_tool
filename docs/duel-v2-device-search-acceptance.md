# GPU 设备内 Gumbel 搜索与 1600 根验收

本协议落实用户要求的完整设备搜索训练和互搏。规则以完整手册为准，学习目标、冻结、停止和报告仍遵循训练手册及完整训练报告规范；本协议不授予模型质量或网站发布批准。

## 执行边界

正式 Python 规则仍是同源对拍参考。设备路径必须在 GPU 保留状态、实体身份、合法候选及顺序、公开观察、隐藏世界采样、逐根随机状态、搜索树、路径和完整策略目标。Gumbel 根选择、内部选择、扩展、固定观察者价值备份、叶子教师和最终选点由设备数据控制。主机可准备冻结源和数值、构造静态批次/计算图、启动整体批次、设置独立守护截止，以及在批次完成后保存和审计结果。

主机逐根生成器、逐步取回动作/观察/Q 再决定下一个规则调用，或 CPU 规则进程池，不符合本协议。缺 CUDA、规则能力、对拍、显存或预算时明确停止，不切换 CPU、截掉合法候选或减小搜索预算继续批准原任务。

## 并发口径

目标是 1600 个独立的真实对局搜索根由 GPU 控制和推进，包含完整 Gumbel 搜索。每根保留原顺序访问依赖，不要求同一根的 32 次模拟无依赖地同时展开。硬件线程块排队、设备工作队列、固定计算图内部的阶段分组均可使用；阶段分歧时某个操作的行数可小于整个活跃 cohort。GPU 自身调度不能替换成主机每 32 根往返规则 RPC。

分别记录总局数、驻留状态数、主机在途根数、设备控制的实际活跃根/任务覆盖、各规则/搜索/推理 dispatch 的行数、完成事件和根身份。`peak_outstanding_roots=1600`、启动参数或最终累计完成 1600 局均不能独自批准容量。设备 driver 的 trace/计数须对应真实独立根，不能用填充行、同一状态的复制样例或仅推理 1600 行替代完整对局搜索。

保持现行 natural-WDL completed-Q、完整合法动作策略目标、最大访问层最终选点、32 次预算、16 个根候选上限、树深和短终局/覆盖教师语义。16 是根搜索候选数，不是合法动作编码容量。单合法动作或公开必胜证明可以按既有协议提前完成，但它们不能代替用于证明 1600 路完整 Gumbel 负载的非平凡根 cohort。

`batched_search/device_search.py` 当前实现具名 `covered_value` 的张量 FSM：`initialize` 保留物理根与模型/观察者绑定，`tick` 用设备掩码推进采样、信息集节点查找、顺序选点、规则步进、三步规则教师、独立辅助 WDL 和完整备份，`unfinished_device` 返回持久 `uint8[B]` 谓词。生产 CUDA 拒绝普通 eager 逐步调用，须接入设备 WHILE graph；其它教师模式明确拒绝。对手教师的原始动作决策仅更新状态与动作租约、完整候选引用和掩码，不复制缺失的观察／网络候选编码；STEP 只消费原始动作。下一次己方 INSPECT 会先完整更新编码，再进入 EVALUATE_NODE，原根信息哈希独立保留。它尚不能替代 serving 的公开证明共用预算层，不能把训练搜索输出标成既有 serving 行为。

统计池 `bind_capture_streams(parent, body)` 必须在捕获前注册同设备的确定 parent/body stream，并由唯一 driver 在线程内使用 `capture_scope()` 构建图；仅此受控捕获允许跨这两条依赖连接的流。普通 eager 操作或任意其它流仍拒绝。`device_info_key.py` 保留 SHA-256 的全部公开字节（观察加合法候选前缀，不含 padding），捕获时调用当前 stream 的 `capture_safe` 内核，不产生 conditional body 禁止的 event/wait 节点。

`gpu_seven_candidates.encode_tokens_device` 只接受已驻留 token，设备上验证完整计数、掩码、ID 与有限数值并返回每行错误；未知行清零而不成为有效候选。静态卡表须在捕获前准备；不创建 UUID 或执行主机 tensor 布尔。除法按正式标量双精度计算后转换 float32，原 host 编码入口另保留。原语图测试与正式规则适配/终局互搏继续分开验收。

`CudaConditionalWhile.capture_loop` 可将独立守护的搜索 WHILE 嵌入已捕获的物理对局 WHILE。每层使用独立实例与 body stream，静态回调只在构图时执行；重放时设备谓词决定搜索和下一次真实行动。嵌套循环需要额外实际 CUDA 验证，接口和 CPU 导入检查不表示完整对局已接通。

`DeviceRootRandomPlan` 在批次准备时，按冻结的 `(job_seed, decision_step, actor)` 有界全集展开与 `root_search_seed` 相同的 NumPy PCG64 状态。搜索热路径只按设备上的实际步数和行动方查表并重置完整 state、increment 与缓存；不能跨决策沿用同一根的连续随机流替代原种子身份。随机表纳入显存预算，超出声明的决策范围返回错误，不截断或取模。捕获流须在构图前显式绑定；主机不按执行中的行动重新播种。随机表和嵌套 WHILE 都只是闭环部件，不批准真实规则、完整局或 1600 容量。

`DeviceRootSeeds.derive` 在 CUDA 上按真实 `(job_seed, decision_step, actor)` 的十进制文本计算 SHA-256，并返回含第 256 位的五个小端 64-bit limb。负游戏种子也按原文本身份哈希；负步数或非法行动方返回逐行错误，不取模或截断。它与 PCG64 状态准备分别对拍，不以哈希通过代替完整随机采样。

`CompactDeviceCandidates` 保留完整原始候选 token 与顺序，按构图时确定的行分片调用既有精确编码、推理和 SHA-256。`scratch_rows` 限制同时展开的行数，**不限制每行合法动作数**；根与当前决策只复制紧凑 token。搜索 FSM 可静态绑定此对象，CPU／受控张量夹具仍保留原 dense 数值参考。`allocated_bytes` 仅计持久 token，模板共享存储另列；Torch 图捕获中间量是否复用及全进程峰值必须实测，不以这一接口保证 1600 显存通过。

`DeviceGameLoop` 在外层设备 WHILE 内，逐轮执行完整内层 Gumbel 后才提交物理动作，先写设备 sink 再推进；输出保留真实赢家、完整搜索数、截断和错误。inactive 物理租约也受保护，错误结果不得回收其他 owner 的租约。这个外围要求实际规则适配器、设备轨迹 sink 和完整 preflight，仍不是具备正式资格的训练 service；受控三步树、接口和完成布尔不提供完整真实局批准。

## 显存与分页接口

持久真实/假想图按实际可达内容存储；不可变常量共享。每根的未知规则字段、共享对象别名、实体和随机状态必须保留。临时 VM 堆、帧和 GC 工作区与根数量解耦，使用设备管理的有界 scratch 租约/分页；设备容量溢出停止，不能截断图。

`batched_duel/device_graph_pool.py` 提供显式容量的页分配原语：

- `reserve(slots, generations, owners, requested_bytes, active)` 创建设备槽租约。
- `grow(handle, requested_bytes, active)` 保留既有页面和地址并增长实际使用范围。
- `release(handle, active)` 回收页面并递增槽世代，旧句柄失效。
- `translate(handle, byte_offsets, active)` 返回设备物理字节偏移与错误位。
- `metrics_device()` 返回实际使用页、活跃槽、逻辑字节和页峰值张量；`allocated_bytes` 是静态已分配存储，不是整个 CUDA 进程峰值。

所有热路径输入/输出为设备张量，禁止 `.cpu/.numpy/.item/.tolist` 或由设备布尔决定 Python 分支。CPU 只通过显式 `allow_cpu_reference` 用作分配器数值 oracle。节点、边、payload 宜分别保留逻辑页空间，避免最大字段范围产生未用地址洞。节点引用仍按正式图 ABI 解析；分页原语不处理图的别名、水合、复制、回溯或 GC，不可据其通过声明完整 VM 已接通。VM 的连续 payload 指针/跨页访问必须另行适配并对拍。

页申请当前使用固定形状排序/prefix 和广播清零，适合先核对正确性；它可能扫描整个全局页数据区。必须实测其工作区、频率和耗时，必要时换成同协议的设备 free-list/局部清零内核；不能只看最终常驻页数忽略临时分配。所有 scratch、搜索池、候选编码、网络中间张量和训练轨迹缓冲都进入实际显存预算。

## 分级验证

### Mctx 批量改造与逐级容量验收

批量改造参考 [Mctx 的设备批量搜索](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/search.py)：每轮对独立根批量选择叶节点、执行状态转移及预测、回传价值，再进入下一轮模拟。本项目继续使用精确正式规则、行动方信息集采样及固定观察者 WDL；不以学习的近似动力学替代规则，不直接继承 Mctx 的质量或性能保证。现有设备队列、紧凑候选和预分配工作区优先复用；已经接入的能力不重复当作新改造。

实现按“批量准备公开决策 → 独立战斗预演任务 → 批量教师选点 → 精确规则推进”拆分。各预演隔离同一公开沙盒的副本，保持原候选顺序、实体、owner/generation、随机语义和稳定同分选择；三个教师步骤仍按状态依赖顺序执行。批量选点只消费正式规则产生的公开预演数值，不能用手写近似伤害替代完整预演。新部件未完成同源原始状态、动作及信息哈希对拍前保持显式未合格，不替换正式执行器。

固定验收顺序为 **8 个真实根的小批完整搜索 → 32 → 128 → 400 → 1600 根**。这些数值是独立设备搜索根数，不是 VM 工作槽、网络 tile、候选数或模拟次数；每根始终使用 **Gumbel32、根候选 cap16、covered_value horizon3** 和本轮冻结的完整合法候选容量及树深。

| 级别 | 独立真实根数 | 晋级所需证据 |
| --- | ---: | --- |
| 小批闭环 | 8 | 全部非平凡根完整完成 32 次模拟；与同源 CPU oracle 核对观察、全部合法候选、教师选点及规则状态；完成同级真实终局与 GPU 原始动作重放 |
| 第一扩容 | 32 | 复用同一已验证执行路径，全部根完整搜索、正常终局与重放通过；取得实际显存及阶段耗时 |
| 第二扩容 | 128 | 同上；额外核对跨队模型、actor/viewer 与根租约隔离，不能以累计小批结果冒充此 cohort |
| 单卡中批 | 400 | 同上；证明完整 workload 的容量与吞吐，不能用静态“显存应够”作为通过 |
| 目标容量 | 1600 | 本协议全部容量证据：真实独立 cohort、完整搜索和正常终局、逐步 GPU 重放／信息哈希、实际显存及延迟／吞吐；随后另验学习与恢复 |

每级先做完整决策检查，再做同级真实终局与重放；两者均通过才进入下一级。起手、正常行动及中途选择根按冻结名单覆盖；单合法动作、已证明即时终局、填充行、合成树及重复初态不能承担非平凡完整 32 次负载证明。原语或单级通过不授予更高级容量、完整训练、模型质量或部署批准。更高级失败时保留已通过较小级别的原证据，不能把较小级别升级成目标达成。

每次启动前固定源码／IR、模型及构筑 SHA、独立根种子、根数、规则工作槽、推理和编码 tile、显存预算、搜索预算、性能目标及 epoch＋monotonic 独立截止。跨级只改变预声明容量参数；规则、编码、采样、教师或搜索实现变化时，从小批同源验证重新开始，不沿用旧批准。越级容量诊断可单独标注，但不能算晋级或绕过本顺序。

整批返回后保存每根 requested/completed simulations、完成决策数、正常终局／失败／截断数、根身份去重、设备控制 trace、CPU 规则调用和热路径 D2H 计数、信息哈希／重放结果及逐文件 SHA。计时分别列编译／准备、公开观察及合法动作、采样与守卫、战斗预演／教师、网络、哈希、搜索管理、物理推进、事后 CPU oracle 和总墙钟；计数缺失为 null，不填零。显存分别列模型、持久图／页池、搜索统计、动作 token、VM scratch、编码／网络临时量、轨迹及全进程峰值，并注明其他 GPU 负载与实际可用预算。

错误、OOM、任何未完成模拟或截止均明确失败并保全结果，不通过降低候选／32 次预算／horizon、提高显存限额或延长同次截止伪造通过。性能未达到预声明目标时，正确性与性能分别报告。批量数组的有限值、尾槽、掩码重放、外来／陈旧租约、重复预演结果、容量整体拒绝和稳定同分选择须有独立检查；CPU oracle 或 CUDA skip 不能填写 CUDA 通过。

设备离线期间只推进源码、文档与明确标注的 CPU 参考检查；不反复 SSH 探测或发起远端作业。用户明确通知设备恢复后，重新核对本任务所属进程及冻结身份，先执行当前新源码的 CUDA 组件和 8 根闭环，再按上述顺序扩容。所有单次记录进入 `artifacts/rl-evals/<run-id>/`。

首阶段组件为 `batched_search/device_teacher_ranking.py` 和 `device_teacher_preview_jobs.py`，均显式不提供 closed-service 或容量资格。前者以完整已知 canonical AST 绑定教师评分，接收公开的四个 float64 预演数值，保留换牌／选择优先级、七队额外动作和原顺序同分选点；动作按它的 `KIND_CODES` 以名称绑定，不能直接套用另一编码器的动作序号。后者只准备和核验 GPU 预演任务，不生成规则结果；回执必须携带任务发出时的 epoch、actor、owner、generation、slot、parent 与原候选 ordinal，迟到、重复、非有限或外来结果拒绝，分组回传需全部完成并通过 `finalize_device` 后才可进入评分。容量不足整体拒绝，不保留部分有效任务。

这些部件尚未接入正式规则教师。后续先将同源 `teacher_prepare/teacher_preview/teacher_finish` 的精确公开沙盒和预演结果解码接入设备 dispatcher，证明全部战斗候选、早期选择分支、实体／RNG／租约与原始状态不变；再将 prepare → preview jobs → ranking → simulate/clean 接入原有三步教师和完整 Gumbel。CPU oracle 显式 opt-in；CUDA 捕获／重放、完整规则对拍与小批闭环须独立执行。验证入口：`.venv/bin/python -m unittest tests.test_duel_v2_device_teacher_ranking tests.test_duel_v2_device_teacher_preview_jobs`。

教师评分的源码 AST 身份使用固定表示：Python 3.13 显式保留空列表，3.12 起新增的空 `type_params` 不进入旧表示；非空类型参数拒绝。原完整 AST SHA 与教师身份保持不变，评分公式、控制流或其他源码变化仍拒绝，不新增不同 Python 版本的宽松白名单。验证需在实际 Windows Python/CUDA 上执行现有评分与预演组件测试；跨版本 CPU 检查不代替 CUDA、正式三步教师或完整搜索验收。

`device_batch_teacher.py` 提供独立、未合格的正式教师桥：冻结 VM 的 `teacher_prepare` 返回原始完整合法动作及公开预演位置，设备解码器保留原次序与别名；预演读取显式使用 `gather_into_device(shared_reads=True)`，允许同一经 owner/generation 核验的原图复制到多个独立 scratch 行；写入与释放仍拒绝重复租约。GPU 按实际任务数分发独立 `teacher_preview`，仅解码四个正式预演标量。CUDA 源码中的中文字段名以等价 UTF-8 八进制字节转义生成，避免 Windows NVRTC 按本地代码页错误解析字符串；字段字节和长度不变。带 epoch、actor 与租约的回执全部核验后，才调用既有批量评分器，返回原始动作图和选点。三步 pipeline 可显式注入该决策工厂；尚未完成同源 CUDA、正式规则及完整搜索证明时，不自动替换默认路径或授予 closed-service 资格。无工作队列跳过预演；容量、未知动作、预演错误、部分结果或守护截止均拒绝，不删合法候选。组件验证入口为 `tests.test_duel_v2_device_batch_teacher`，正式规则与8根闭环另验。

### 完整训练与独立评估

1. 原语：规则、编码、随机、Gumbel 数学、页分配和 CUDA graph 捕获分别对拍。合成 1600 行测试仅证明原语，不批准真实局容量。
2. 完整链路：七队规则根、完整 Gumbel 决策和短完整局，与正式 CPU 同源 oracle 比较。CPU 只在事后重放，不生成 GPU 结果；保持全部候选、动作、逐根随机流及固定观察者叶值。
3. 七队覆盖：49 格互搏可用于初始覆盖和有界计时，但最多 49 根不批准 1600 容量。
4. 1600 容量：在预先声明的有界测试内，真实独立 1600 根设备 cohort、完整搜索预算、正常终局、逐步重放和实际显存/延迟/吞吐同时取得原始证据。阶段分组数、任务覆盖和物理 GPU 调度方式独立报告。
5. 学习：真实完整局产生完整策略目标与终局 WDL 后，执行现有联合更新；记录非零策略梯度/参数变化、价值-only 隔离、全行通行、数值导出、Adam 下一步及 RNG 恢复。采样通过不等于更新通过。
6. 冻结互搏：冻结权重/构筑/规则/随机及搜索身份，以独立种子生成七队先手为行、后手为列的 7×7 胜率表。对角线真实内战；平局进入正常完成分母，失败/截断/未执行另列。同一局只计一次，容量重复压测不增加独立胜率样本。

学习与独立评估预算、样本和带日期截止在新运行前固定；协作或代码修改不恢复已停止的旧任务，不延长原运行截止。完整训练是否含配牌候选及其公平适应仍由本轮合同明确，不以有界链路测试冒充完整训练交付。

## 终局证据审计

正式入口在产生输出、载入权重、分配种子或启动 worker 前调用 `require_closed_device_driver(driver, purpose='training'|'winrate_evaluation')`。它拒绝配置字典、仅标为 `cuda_vm` 的旧 backend、主机逐根调度、合成服务、缺失图规则适配及缺少同源设备搜索/独立预检回执的对象。此检查只判断执行服务和前置资格，不能代替实际原始记录审计或授予容量/质量。

`scripts/evaluate_duel_v2_batched.py` 的静态计划仍可运行；正式 `--run` 在尚未接入合格 closed-device service 时明确退出 2，不创建输出或启动历史 cooperative 实现。该服务只能经具体验证的代码接线，不能通过新增 mode 名、`--device cuda`、1600 配置或 `diag` 标志放行旧路径。原主机执行体只留源复现，不再由正式 CLI 选择。

`batched_search/device_execution_contract.py:audit_device_search` 是批次完成后的纯主机审计入口，要求：

- 具名七队、Gumbel/natural-WDL 和预算身份；真实 CUDA 来源与设备控制边界。
- 显式零 CPU 规则/主机逐根步数/热路径 D2H/设备错误/预算失败计数，以及规则、搜索和重放核验。
- 非平凡真实根完整模拟、完整策略目标和正常终局；重复/外来根拒绝。
- 设备 driver 的实际 cohort 和采样、编码、推理、根/内部选点、规则推进、备份与策略输出 dispatch 证据。
- 实际显存峰值及预算；trace、完整局原始和来源清单实际存在并逐文件 SHA 匹配。

返回的 `evidence_consistent` 只说明这些终局记录结构和引用一致。它不能鉴别伪造 profiler 记录，也不能取代独立 reviewer 对 trace、执行源码和原始文件的检查；始终保留 `independent_cuda_review_required=True` 和 `quality_approved=False`。元数据夹具测试通过不代表 CUDA 执行。

单轮 trace、日志、原始记录、权重、数值 SHA 和胜率表全部放 `artifacts/rl-evals/<run-id>/`，不提交 Git。本协议只保存长期执行和验收口径。

合格设备服务通过 `seven_round.run_closed_device_learning` 接入既有联合学习、择优、检查点及恢复流程。`ClosedDeviceLearningContext` 将完整局收集和冻结评估交给设备服务，拒绝 CPU 规则 `map`、替换收集器及遗漏的任务结果；仅在整批返回后进行 CPU 重放 oracle。训练身份包含此边界源码，边界变化须重新预检。这个桥接接口不创建规则执行器，也不授予实际 CUDA、1600 容量或模型质量证明。

验证入口：`.venv/bin/python -m unittest tests.test_duel_v2_device_execution_contract tests.test_duel_v2_device_graph_pool tests.test_duel_v2_device_statistics tests.test_duel_v2_device_info_key tests.test_duel_v2_device_search tests.test_duel_v2_device_candidates`。没有 CUDA 的主机必须如实记录设备测试跳过；实际设备原语、完整规则和完整对局分开报告。

`validate_duel_v2_gpu_seven.py` 是 14 个开局/正常行动根的有界真实规则对拍，`--seconds` 可显式取 1–3600 秒，但仍受原授权绝对截止及外部进程树守护限制。它按 decision、rule_decision、sample_world、step、commit 原子保存阶段完成、CPU oracle 通过和查询墙钟；后续阶段失败不能抹去或升级此前证据。该脚本使用主机准备对拍请求，不是正式设备内搜索或完整局执行器；其查询时间包含准备/传输，不能当完整 Gumbel 吞吐。物理 commit 也携带同一剩余截止，不获得无限时间。

纯网络冻结评估使用 `device_pure_policy.py`：同一设备规则接口给出完整合法动作后，由设备模型按实际行动者路由并执行 masked argmax。非法输入、陈旧状态租约和非有限分数按行保留错误；搜索预算、访问次数和对局搜索计数全部为0。它不调用世界采样或规则教师，不生成 Gumbel 策略标签；`DeviceTrainingTape`会拒绝零搜索预算。纯网络及搜索模式仍各自需要真实完整局和独立评估，组件通过不授予完整执行服务批准。

规则诊断可显式传入 `validate_duel_v2_gpu_seven.py --vm-max-nodes --vm-max-edges --vm-max-payload-bytes`，仅改变有界查询 scratch，不修改持久状态容量或原默认值；仍须通过14GiB计划检查和原绝对截止。容量规划的估计与设备实测分开保存。扩大 scratch 的14根对拍不能证明生产队列、GPU初始化、堆回收或1600真实根容量。

数字轨迹的策略和价值导出按不同学习对象筛选：`numerical_rows_after_batch`仅输出行动方属于本轮学习范围的完整搜索标签；`value_rows_after_batch`在正常终局后输出每个学习视角的公开观察、实际行动方上下文和纯胜负终局值。冻结对手行动时，学习方的价值行仍须保留。未终局、截断、设备错误或重复终局确认均不得输出学习价值行；对手观察不能错误标为当前行动方。两种导出都发生在整批设备执行结束后，不产生CPU热路径规则调用，也不代替原始动作重放核验。

收集器核对全部物理决策的完整搜索计数时，可用 `numerical_rows_after_batch(include_frozen=True)` 导出冻结行动方的完整策略记录；这些行明确标记 `learning_eligible=False`，终局值仍来自该实际视角。它们用于搜索与重放审计，既有经验导出仍按 `job.train_sides` 过滤，不能把审计导出扩大为新的学习范围。

物理循环在GPU执行正式 `apply_action` 后执行正式 `clean`，与既有七队训练一致；成功后回收原物理状态和已消费的中间状态，失败时保留原状态并回收本行新租约。清理结果不能窃取另一行尚存活的物理中间状态。纯网络物理评估也使用同一清理路径，搜索计数仍为0。

`device_raw_action_tape.py`在原始物理动作图完成压缩后、提交前归档实际使用的node/edge/payload前缀及owner、种子、step、version、actor，GPU有界前缀分配并整体拒绝容量不足的选中队列。非活动补齐行不得覆盖新版本；同一物理根的版本必须递增。GPU `gather_into_device`恢复独立原始动作图供重放，不能用仅chosen ordinal替代。该原语仍不保存不可变开局、不确认物理提交成功或真实终局，也不授予完整对局、1600容量或训练资格；这些须由实际collector和同源预检提供。`allocated_bytes`仅统计其持久Torch缓冲区，不能写成总GPU峰值。

动作图有独立租约：`device_action_leases.py`以零执行错误向量核对action arena的live、owner及generation，回收后将本快照slot标为-1，避免重复释放。搜索覆盖当前决策前保护根动作、传入动作和借用状态；结束／失败清理只回收当前动作，根动作保留到物理消费或放弃后，由同一GameLoop路径回收，纯网络同样适用。复制root/current数字缓冲区不增加底层图引用计数，Adapter不得提前回收仍被这两个快照引用的动作。

分页图的首个合法generation可为0；有效槽位接受非负代号，未分配slot为-1。循环、轨迹和动作回收必须与实际store的live/gen/owner核验一致，不能把toy fixture默认generation1当正式存储契约。独立动作图的成功、失败、冻结／纯策略和首代0测试须单独通过，默认action_slot==state_slot的借用夹具不能掩盖动作泄漏。

完整搜索结果提供原根的 `information_keys`，绑定原观察和完整合法候选前缀的GPU哈希。轻量策略轨迹可保存此哈希和完整pi/访问计数；之后必须以不可变GPU开局和原始动作在GPU重放中恢复x/c并逐行核对哈希，才能交给学习接口。哈希本身不构成重放或终局验收。

`device_initial_archive.py`以设备内有界lane队列将原物理图复制到独立分页arena，保留实体、别名和RNG字段；原物理状态释放或改写不影响归档。复制组件无规则回调和CPU开局重建，只供一个有界对局／重放cohort使用。调用方必须在GPU重放完成前保留这些lease，不能反复重放复制graph覆盖原始副本；最终原始／公开录像和GPU重放仍须实际collector核验。

正式效果刷新与叠加上限使用 `effects[:]` 保留列表身份。设备 VM 的 `vm_collections.py` 对已物化列表／元组右值支持普通及扩展切片写入：先暂存新边，再原子更新原列表头，保留自赋值、元素与容器别名；扩展切片长度不匹配抛出 ValueError，容量不足保留原头和原边并拒绝。未物化迭代器右值仍明确拒绝。实际CUDA回归入口为 `tests.test_duel_v2_vm_slice_assignment`，包括正式 `add_effect` 刷新与最大叠加数分支；组件结果不代替完整搜索。

教师桥可显式启用 `shared_template=True`，要求冻结的 `teacher_prepare_with_template/teacher_preview_with_template`：按正式教师生命周期为同一父状态只匿名化一次，保留不可变模板；每个预演经独立任务包在自己的VM scratch中复制模板，并由正式 `rule_battle_preview` 生成四标量。模板与合法原始动作共享经租约核验的arena，预演包引用与物理原始动作引用分别保留。默认独立准备路径不变；模板版本／事件序号及任务epoch／actor／owner／generation继续核验。复用须重新执行CPU同源、实际CUDA候选隔离／捕获重放、14根正式规则及8根完整搜索，不能沿用未共享模板路径的批准。CPU参考入口为 `tests.test_duel_v2_teacher_prepared_template`，实际组件仍由 `tests.test_duel_v2_device_batch_teacher` 覆盖。

搜索可显式启用 `group_ready_leaves=True`：GPU 先让本次模拟仍处于采样／查找／扩展／对手选择／推进的根继续执行，准备好的叶子等待；全部活跃根到达叶子或回传后，批量执行三步教师，再批量回传。错误与 FINAL／CLEANUP／DONE 根不阻挡等待。每根的种子、选择、状态推进与访问回传顺序保持独立；此模式的 phase_work 只累计实际放行的阶段行，不把等待记成教师工作。分组不能缩减32次预算或混合观察者，CPU混合深度／尾槽结果、随机顺序与实际CUDA捕获对拍需通过，再重验8根真实搜索。

VM可显式启用 `mutable_lookup_cache=True`，只为可变字典／实体的常量字符串键保存32槽查询位置提示。命中须重核 owner、当前偏移／计数、索引范围及当前边的正式键相等；替换值仍读取当前边，缺失键不缓存，碰撞或失效回到原完整扫描。提示不进入状态，每个VM执行重新初始化，GC或字段变化不能沿用未经验证的位置。默认关闭，固定字段、顺序、别名、写入与异常语义不变。实际CUDA回归入口为 `tests.test_duel_v2_vm_mutable_lookup`，包括更新／删后重插／GC与同头字段换位，之后重做正式教师及8根完整搜索；不能依据缓存源码声明性能或容量。

就绪分组继续覆盖对手教师与新节点评估：仍有采样／查找／规则推进的根先执行，随后分组处理对手教师，再处理新节点和叶子，最后回传，保持单根次序。可显式使用 `reuse_interior_observation=True`，只借用已由己方 INSPECT 生成、且slot/generation/owner与当前世界相同的完整观察；重新核验正式metadata的非终局与actor==viewer，辅助价值模型与行动方上下文仍独立计算。其余根、终局及H3后观察不复用。错误或陈旧绑定拒绝；CPU完整搜索、随机顺序及实际CUDA捕获须对照通过，默认仍关闭观察复用。

冻结VM的 `constant_lookup=True` 继续只索引不可变且全部为字符串键的字典；常量引用直接命中后，动态或其它引用的字符串按精确UTF-8值二分查询，保留第一个相等键的位置与缺失结果。混合键、非字符串查询和可变字典回到原正式相等路径，不缓存可变值。验证：`tests.test_duel_v2_vm_constant_lookup` 包含动态字符串、Unicode不同序列、空键、缺失键、属性查找与可变／混合键对照；组件通过后仍需正式教师和完整搜索验收。

设备VM可显式使用 `frozen_class_lookup=True`，按冻结类命名空间的描述符类别顺序和C3继承顺序生成精确静态索引，返回原值引用和类别。常量字符串与动态UTF-8字符串都保留相等和缺失语义；不支持的键仍走原查询。实例字段仍读取当前可变字典，getter、方法绑定与super继续由原对象处理器执行。默认关闭；`tests.test_duel_v2_vm_frozen_attributes` 对照多继承、描述符遮蔽、属性／静态／类方法、动态名称及实例写入，实际CUDA及正式教师／完整搜索分别验收。

原始动作重放的CUDA图捕获保留外部索引、种子和逐根计数张量直到关闭图，避免调用方传入临时张量后设备分配器复用其内存。保留引用不改变输入内容或CPU／GPU边界；`tests.test_duel_v2_device_raw_replay` 以释放调用方引用、分配压力、完整多步／尾槽重放验证，正式原始动作／全合法哈希仍单独验收。


## 无三步叶子延续的独立计时分支

`network_value` 独立计时分支使用 `terminal_horizon=0`，覆盖上文对该分支的固定horizon3要求；历史covered_value冻结任务和回执保持原身份。搜索的叶子直接查询固定观察者WDL，新展开节点复用其已取得的估值并回传，不执行teacher_continue。Gumbel32、cap16、完整合法候选、精确规则和信息集采样不缩减。设备按独立根的就绪阶段批量遍历、扩展、估值和回传，复用现有分组结构；这不表示直接调用Mctx库，也不把通用VM解释执行称作JAX原生规则内核。对手行动策略独立声明，保留公开规则教师时不得宣称所有教师调用均为零。

真红指导在一次根搜索完成之后执行，不进入每个叶子、不把启发式分数当WDL。固定24计划版本复用共享v3排序，使用三个独立信息集世界和明确的公开版本种子；起手留牌不采样。设备复用正式采样转移，并要求转移前后完整原始公开投影（包含合法动作顺序）严格相等：相同投影必然得到相同确定性编码，任何原始差异直接拒绝，不删字段放宽检查。外部截止仍由设备及独立进程守护执行，不按主机时间改变候选计划数。指导可以使用独立的少量VM工作槽，通过 `vm_layout` 声明更大的临时堆，`max_workspace_bytes` 在分配前拒绝超预算；搜索工作槽无需同步扩大。

计时必须分开初始化／图捕获、完整8根Gumbel32、根指导、实际提交和事后重放核验。八个非平凡真实根各自完成32次且零错误才给出有效搜索吞吐；指导须另对两动作组合、换牌双方座位、直接取胜、资源不足、倒地及暗牌置换做CPU／CUDA对拍。仅取得一次决策不代表正常终局、训练或模型质量通过。单轮数据全部保存在artifacts，文档不维护单轮耗时或结果台账。


### 冻结策略对手、指导队列与编码投影

`DeviceGumbelSearch(opponent_mode="policy")` 在模拟中按实际行动方读取其自身公开观察，并调用该方绑定的冻结模型及精确PCG64 categorical；`rule` 默认模式保留。策略对手即使关闭根Gumbel噪声，也必须逐物理决策重置独立随机计划。该模式与 `network_value/h0` 组合时不调用规则对手教师或三步叶子延续。策略对手仍是搜索中的对手近似，不等于对手也执行32次嵌套搜索；物理行动方每次重新执行完整根搜索。

真红根指导可显式启用 `pack_active=True`：只把符合条件的根按原始行顺序排入少量设备工作槽，完成后散回原行。空队列、尾槽、错误及非指导行必须保持原语义。`DeviceGameLoop(search_audit_sink=...)` 在指导前保存原搜索选择、完整策略及访问次数；实际动作归档保留指导后的选择。两个sink独立绑定并对错误拒绝提交，不能用末次基线代替多步审计。两份轨迹逐决策与原始动作、种子、版本、行动方及完整合法哈希对拍，终局只由正式状态确认；预算截止及错误前缀不产生终局标签。

`batched_duel/gpu_policy_projection.py` 从正式公开投影函数派生仅供编码的投影，保留可见性、完整合法动作顺序、数值字段及机械效果，只移除编码未读取的显示说明和重复展示计算，复用已计算的兼容标记。派生点结构检查失败即拒绝；正式引擎、网页投影及模型数组不替换。必须对两方视角的真实多步状态、隐藏信息变换、实体别名和输入纯净性比较完整观察／候选编码字节，并完成实际CUDA对拍；同种子搜索还需逐项核对访问计数、策略与动作，不能仅凭胜负或局部列相等批准。

VM的 `Random.sample(population, k)` 支持已物化列表／元组与整数k，按CPython的pool/set分支及拒绝采样顺序推进精确随机状态；保留重复元素及对象别名。非法k和未支持的参数形式明确拒绝，不降级为有放回采样。验证入口：`tests.test_duel_v2_device_policy_opponent`、`tests.test_duel_v2_policy_projection`、`tests.test_duel_v2_vm_sample`；无CUDA时跳过项不计设备通过。每级并发的完整搜索与自然终局仍须各自验证。


### 显式载荷上界和消费自有工作副本

正式默认临时堆仍保留262144节点、524288边和32MiB载荷下界。显式 `allow_reduced_payload=True` 必须同时给出 `vm_layout`，仅将载荷下界改为不少于压缩输出的声明上界；节点／边下界、整体工作区字节检查和原始输出容量不变。触发堆／压缩／页容量不足时拒绝对应根，不截断数据或候选。每个阶段与角色范围均需重新对拍和实际CUDA测量，不能用布局估算批准更高并发或1600真实根。验证入口为 `tests.test_duel_v2_device_rule_calls`，包括默认约束、显式范围拒绝及较小载荷下的尾槽／错误／租约隔离。

`gpu_owned_transitions` 只供先从不可变租约池复制到独立VM arena的执行器显式冻结。派生 `apply_action` 仅把正式 `clone_state(state)` 换为正式 `hydrate_entities(state)`；同时检查正式clone仍为复制后归一的原结构，保留其余校验、教程特例及完整规则函数。派生模拟入口仅替换内部apply绑定，保留公开缓存过滤和无演出patch上下文。工作副本被消费，错误后不得发布或复用脏堆，下一次调用必须重新装载；持久根不允许直接传入。

该候选须通过真实多步与JSON状态的值、实体归一、内部别名和错误隔离CPU参考；实际CUDA须验证不可变输入字节、完整输出和错误后重新装载。随后用新冻结批次核对完整Gumbel32、原搜索／实际动作双轨迹、完整信息哈希和独立原始动作重放，同种子访问次数、策略及动作逐项比较。组件正确、完整搜索正确和性能更好是三个独立结论；默认网站与既有冻结任务不自动采用。


解释器直接栈分派实验由 `direct_stack_dispatch=True` 显式选择，仅让LOAD_CONST、LOAD_FAST、STORE_FAST和POP_TOP绕过不匹配的字符串／异常／标准库／集合拦截，仍进入原始switch主体；不重写指令，不绕过前置帧／PC／步数／截止检查。默认生成代码必须保持逐字节一致，实际CUDA模块源哈希须与声明选项对应；数值、容器别名、逐指令上限、过去截止及整数溢出的错误状态须与默认解释器相同。组件通过后仍须同种子完整Gumbel32、原始动作和全信息哈希重放、访问／策略／动作对照及独立耗时测量。验证入口为 `tests.test_duel_v2_vm_direct_dispatch`。


观察中的来源席位 `(index+1)/20` 同样属于精确编码契约。连锁来源和复制来源均先在设备上以binary64计算，再单次转换float32；不能用容差替代完整信息哈希，CUDA常量倒数乘法造成的一ULP差异也须修正。测试 `tests.test_duel_v2_gpu_seven` 覆盖全部席位的来源索引及完整观察字节。历史失败轨迹保留原哈希与结论，只能在新冻结批次中重新验证。


## 批量数组迁移与无模拟真红先验

新数组分支的状态／recurrent、共享IR与整数健康数值原语、真红根先验和验证入口见[数组搜索边界](duel-v2-array-search.md)。它覆盖本分支的旧24计划指导约定：`prior_guide=ArrayZhenhongPrior(...)` 在根搜索前调整有界logits，不采样、不试演、不改变WDL，也不在搜索结束后覆盖动作；不得同时配置旧`root_guide`。完整策略与访问计数仍由原Gumbel32产生，冻结配置须绑定指导版本、权重及实际位移。默认及历史冻结分支保持原身份。

数组IR组件或无模拟指导通过不能批准七队数组执行器、实际CUDA、完整搜索、自然终局或训练服务。正式七队规则推进与采样仍使用原VM，直到数组版完整状态、全部机制、合法动作、信息边界及原始重放分别对拍通过。不能以新recurrent接口的存在替代这些迁移与验收。


## 同源原生数组执行分支

`CudaNativeArray` 使用完整预编译控制流替换解释器执行，设备不上传字节码指令流，状态保留完整有界数组图与实体别名；不将此称为全部领域字段SoA化。`NativeArrayRuleCalls/DeviceNativeArrayRules` 保留双信息编码检查、合法动作、原始物理动作和checked租约，固定network-WDL/h0、策略对手及无模拟真红先验。新分支默认冻结时即排除教师入口，不因名字或布尔标记放宽正式设备闸门。

本机同源C++验收须从原生初始化到自然终局逐步比较完整正式状态和编码，再比较同种子采样、假想推进与完整Gumbel32；主机夹具／诊断回传不算闭环GPU执行。实际CUDA仍须用新生成源码和真实设备分别通过规则、完整搜索、重放、自然终局及容量验收；旧解释器回执不继承。跨Python版本只加载已核验源码／协议的冻结程序，不现场改用另一条规则路径。接口与测试见[七队原生数组规则后端](duel-v2-native-array-rules.md)。


## 逐函数原生迁移边界

`batched_duel/native_functions.py` 的 `native_functions=True` 完整模式生成全部冻结函数的原生控制流，默认关闭。调用、属性、闭包、生成器挂起、上下文与异常恢复均须覆盖；任何函数拒绝都使生成失败，不保留旧 PC 分片作为静默回退，不截短函数，不回退 CPU。函数入口／恢复保留 PC 跳转表，内部直线顺接、分支跳转；通用操作沿用编译期已知的原语特化。对象和兼容帧仍在，不能据此声明领域状态已完全去对象化。此选项不与独立 `stack_runs` 实验叠加。

每条逻辑指令仍保留相同步数、截止、错误上下文和位置；操作错误先交回原异常转换与栈展开，已发生的状态修改及对象别名须保持。验证 `tests.test_duel_v2_native_functions` 要求独立编译开关两版，比较全部步数截断、循环／短路分支、容器修改、外层捕获及 CUDA 过去截止。真实规则另对完整合法候选、观察／动作编码字节、输入纯净性和推进后的完整状态进行核验。

完整控制流、局部引用、领域字段及调用路由分别验收；带类型标签的对象、装箱和兼容帧仍保留，不以函数编译数量或源代码结构当作吞吐收益。短测保持独立硬截止，CUDA 组件通过不代替同种子完整 Gumbel32、原始动作重放、正常终局及逐级并发验证；实际收益不足时保留默认关闭。


局部引用／常量字段迁移通过 `native_values=True` 单独选择，默认关闭。`batched_duel/native_values.py` 只提升静态可证明的直线片段，不跨函数、跳转目标、调用、异常或未支持指令；分片模式保留常规PC入口，栈形状不符合前置条件时继续原设备规则；完整函数模式仅保留全部合法的函数入口、调用／挂起之后、异常／分支目标及片段退出恢复点，直线内部不建立无用恢复分支。选段代码预算优先合法动作、伤害／攻击及行动推进，不改变候选范围、顺序、搜索预算或规则公式。

片段中的操作数与局部引用保存在C++局部变量中，内部静态边不反复读取帧／PC元数据；每条原逻辑指令仍核验步数与截止、维护错误上下文，退出或错误时完整恢复活跃栈及已修改局部值。记录常量键按schema直接绑定列偏移，缺失字段、普通字典／序列、Counter、错误和类型边界保持原处理；值装箱、缓存与别名不省略，不将本阶段称为标量去装箱完成。

验证 `tests.test_duel_v2_native_values` 要求独立编译两版，比较全部逻辑步数截断、动态键／不同记录类型同名字段、部分修改／别名、缺失键、未绑定局部值、Counter和CUDA过去截止。真实规则另比较全部候选顺序、观察与候选编码字节、输入纯净性及完整状态推进。静态覆盖和CPU时间均不代替实际CUDA、完整Gumbel32／自然局及并发验收；未取得稳定收益前保持默认关闭。


原生引用片段可包含字段写入、数值／比较／逻辑运算；整数／布尔受保护快路径复用原始数值尾部，保留溢出、装箱、分配数量及容量失败。调用路由保留原闭包／关键字参数绑定、生成器与错误，已知同步字典／列表方法可直接进入原实现；异步迭代与其它特例不能被误认作同步路径。

验证入口为 `tests.test_duel_v2_native_calls`、`tests.test_duel_v2_native_scalar_ops` 和 `tests.test_duel_v2_native_scalar_cuda`。设备数值探针使用完整原堆／库原语，不用假规则或桩替代设备 oracle；既比较新旧逐字节内容，也核对独立期望值／错误与所有行确实执行。它不测完整规则、完整 Gumbel 或并发。大程序优化编译若达到短检截止，明确记录未取得优化构建；无优化的 CPU 正确性构建不能用于宣称性能提升或 CUDA 验收。


显式 `execute_only=True` 仅用于主机控制的CUDA规则诊断：保留相同执行内核、规则库及冻结函数集合，省去诊断未用的投影／导出／解码／压缩内核；这些设备接口主动拒绝。完整规则源身份与实际裁剪后的编译源身份分别记录，不能继承完整闭环或生产模块资格。`tests.test_duel_v2_native_rule_stages` 以正式规则辅助函数核对状态和别名，仍只提供组件证据。

显式 `separate_compilation=True` 仅与完整 `native_functions=True` 配合，默认关闭。运行库、局部值片段和规则函数分属独立编译单元；类型／布局／内联原语保持相同，公共表和非内联函数只有一个定义，所有冻结函数必须参与最终设备链接。使用 `--device-c` 生成可重定位 CUBIN；缓存身份包含各单元源码与完整编译环境，最终链接缓存还包含有序对象身份与链接设置，命中须校验二进制完整性。分单元缓存不能继承另一个规则程序的资格。默认两个独立 NVRTC 程序并行，源码身份和回执互相隔离，错误取消尚未开始的任务，链接保持原顺序；对象编译累计时间与缓存准备墙钟、链接和执行分别记录；跨单元 CPU 对拍、模拟驱动测试与小型 CUDA 链接探针均不代替完整规则和搜索验收。验证入口：`tests.test_duel_v2_native_compile_units`、`tests.test_duel_v2_cuda_compile_units`、`tests.test_duel_v2_cuda_linker`。完整函数模式复用现有直接栈路径的四类操作：局部值读取／写入、常量读取和弹栈直接执行原操作，不再实例化不处理这些操作的通用字符串／异常／标准库／容器处理器；逐步预算、错误和局部值未绑定检查保持。

热点诊断通过 `native_profile.instrument_runtime` 显式改写拆分后的运行库单元；规则／局部值单元和 Context／Heap ABI 保持，非运行库缓存应全部复用。函数时间按实际执行片段累计，操作与嵌套原语时间重叠；直接栈操作和局部融合段不独立计作通用操作。计数按行保存，复位／执行／读回同流串行，不支持并发查询共享计数器。插桩结果只用于定位，收益须由未插桩的新旧版本同状态交替短测、CUDA事件、独立预热与正式重复证明，并逐次核对候选顺序、编码、输入纯净性和完整状态。`indexed_attributes=True` 默认关闭，要求独立完整编译，仅使用现有冻结类 C3／描述符索引改写运行库查找；索引采用运行库专用函数名，只接入运行库独占的非内联函数，共享的原内联定义逐字保持；动态实例字段、getter、绑定、异常和预算不变。源身份保留冻结规则源，优化构建还必须核验单元及 CUBIN 身份。验证入口为 `tests.test_duel_v2_native_profile`、`tests.test_duel_v2_native_runtime_index`。仅 `execute_only=True` 的主机诊断可显式设置 `stack_bytes=65536`，用于共享显存受限时的固定状态检查；正式执行仍至少128 KiB且默认值不变。较小诊断栈通过不提供任意规则路径的栈上限或生产资格，新旧计时须使用相同实际栈配置。

SoA搜索构建器默认复用已保存的非终局己方新节点观察供价值评估，避免同状态再次调用完整合法动作／观察。复用必须绑定相同 arena／slot／generation／owner，核对行动者和终局元数据，不能借给对手、终局或深度截止查询；显式 `reuse_interior_observation=False` 保留对照。模拟预算、cap、信息集、随机消费、模型及终局价值不变。重建图后验证完整预算结果、随机状态、租约回收与有效查询次数；`tests.test_duel_v2_soa_observation_reuse` 的合成搜索及真实规则两状态检查仍不能代替正常终局、完整Gumbel和400并发验收。

原生实验对大型动态调用、参数绑定及SoA写入／复制库设置共享调用边界，仅更改函数调用属性，不修改规则或布局。`NTE_CUDA_COMPILE_THREADS` 只允许显式1–16，默认原旗标不变；并行编译参数进入回执和缓存身份，严格浮点不变。编译、准备与执行耗时分别报告；小组件通过和完整模块编译超时必须同时保留，不得以组件通过替代完整规则／Gumbel／自然局／并发验收。

可组合启用默认关闭的 `direct_runtime_calls`、`direct_type_queries` 与 `indexed_container_methods`，均要求完整函数原生化与独立编译。调用优化只复制原始参数绑定、普通函数及标量内建分支到运行库独占 `call`，类型查询选项只提前处理 `type/isinstance` 和已解析的原生字典 `.get()`；容器方法优化仅对冻结名称引用直接分派，动态名称保留原查询。不得缓存可变效果、绕过类覆盖／描述符或改变逻辑步数、分配及错误时机。共享内联定义和 ABI 必须逐字保持，只有 runtime 单元身份改变；主机对拍覆盖两项与属性索引组合后的状态、原始堆、别名、错误和预算截止，入口 `tests.test_duel_v2_native_runtime_calls`、`tests.test_duel_v2_native_effect_queries`。编译通过或CPU等价不代表GPU提速。

SoA工厂还默认开启 `reuse_root_observation`：只复用本次初始化、相同 arena／slot／generation／owner、actor==viewer 且非终局的完整根观察；独立辅助价值推理及其输入、模型选择和错误必须保持。终局及不同观察者走原路径，陈旧绑定失败关闭；根价值复用核验仍独立读取观察。允许显式关闭，并须重新捕获图。`tests.test_duel_v2_soa_root_observation_reuse` 核对完整32次结果、随机状态、独立价值推理、租约及有效查询减少；规则编译单元不变，不代替真实GPU搜索和400并发整局验收。

完整 SoA 搜索验收脚本可显式传 `--native-complete`，实际启用完整原生函数、局部值和独立编译；`--indexed-attributes` 要求该模式。计划与运行回执记录这些实际选项，默认仍保留原配置。搜索使用完整 CUDA 模块及至少128 KiB栈，不开放 execute-only 或64 KiB诊断配置。验证 `tests.test_duel_v2_soa_device_cli`；选项存在不等于完整搜索或并发验收通过。

无缓存效果快路由 `gpu_effect_queries.query_legacy_effects` 和 `query_all_effects` 从当前正式函数派生：仅 exact flags dict、全部七个旧式效果键缺失且 holder 是普通字典或三种正式实体时提前返回空效果；`query_all_effects` 仅对正式 holder 的 exact 空 list／tuple 效果及 None／exact 空字典 flags 提前返回新空列表；自定义读取、迭代、类覆盖、异常及实时标记修改保留正式顺序，其他输入完整回退，不缓存可变结果。`capture_native_array_program(effect_query_fastpaths=True)` 显式接入冻结替换并记录协议／源码身份，默认关闭；`tests.test_duel_v2_policy_effect_queries` 和 `tests.test_duel_v2_native_array_program` 验证派生、替换及原生结果。生成新规则程序会改变其编译身份，不能沿用旧规则或模型的验收资格。

`indexed_entity_queries=True` 只为有界冻结纯字符串 tuple 建立精确 UTF-8 二分索引，支持运行时动态字符串查询；其他容器／类型／坏引用保持原路，不缓存可变实体或跳过恢复校验。要求独立完整编译，默认关闭，CLI 显式 `--native-complete --indexed-entity-queries` 可选；验证 `tests.test_duel_v2_native_entity_queries` 包括真实角色顺序、实体恢复、别名、容量与步数截止。

完整模块的投影／导出／解码附加源码仅归运行库；未知定义、声明或规则对投影符号的反向引用拒绝拆分。完整及 execute-only 的非运行库对象源码逐字一致。`CudaModule.compilation_unit_specs` 可显式指定完整绑定的非空同签名子集，仅用于对象缓存；最终链接与实际主机绑定保留完整签名，源码／GPU／NVRTC／驱动／编译选项校验不变，不修改并行共享状态。SoA 使用相同规则／时钟对象身份复用缓存；完整模块仍保留全部功能和至少128 KiB栈。验证 `tests.test_duel_v2_native_compile_units`、`tests.test_duel_v2_cuda_compile_units`。

规则工作行显式支持1–64，默认8；稀疏及分页状态工作区同步支持64。容量计划、编译前及 cohort 分配前检查已知 scratch／typed 工作区下界和当前整卡占用；下界通过不表示显存必定足够，捕获后仍核对总预算。完整搜索回执绑定独立种子／owner／slot、有效行覆盖、尾批、错误、固定Gumbel32及临时租约回收；复制状态吞吐和CPU分块对照不授予GPU容量资格。短连续入口可用 `--native-complete --indexed-attributes --rows 65 --rule-work-rows 64 --cohort-only --max-decisions 2 --warmup 0 --repeats 1 --execution-seconds 60 --proof-seconds 30`，同时提供冻结输入与新输出目录。验证 `tests.test_duel_v2_wide_soa_search`、`tests.test_duel_v2_soa_capacity`、`tests.test_duel_v2_sparse_soa_storage`、`tests.test_duel_v2_soa_device_cli`。

设备内层循环被deadline或tick guard打断时，在物理提交前通过 `DeviceGumbelSearch.abort_device` 排空已签发的临时世界、粒子、当前动作及统计租约；根动作继续由物理层统一消费／释放。排空幂等，不推进模拟，不改变已请求／完成计数、信息键、随机流或原错误，不允许将中断结果写为成功。验收脚本先持久化原始执行计数、阶段与捕获后显存分项，再核验租约，避免失败回执丢失。验证 `tests.test_duel_v2_device_action_leases` 的独立动作arena、各阶段截断与重复排空。

`DeviceSoARules`／`DeviceTiledSoARules(require_teacher=True)` 显式准备冻结三步 teacher 和原始规则动作解码；默认仍关闭。精确 int64 选择索引、完整原始动作顺序及独立动作租约不经过策略重排。`staged_teacher=True` 必须同时启用 teacher：一次独立 typed clone，三步分别选择、推进、清理与压实保存，回收中间状态和动作，提前终局停步，亲本租约不变。该适配只提供接线能力，单次规则决策的预览仍受临时及 typed 池容量限制，必须另做完整 GPU 对拍，不能授予训练资格。`soa_training_jobs.adapt_training_jobs` 只绑定真实七队任务、目录／版本／构筑与训练侧，保留起手换牌；不启动规则、推理或学习。验证 `tests.test_duel_v2_soa_teacher`、`tests.test_duel_v2_soa_teacher_pipeline`、`tests.test_duel_v2_soa_training_jobs`。

假想推进可显式启用 `fused_rule_steps=True`（验收 CLI：`--fused-rule-steps`），默认关闭。推进与原 `clean` 之间保留独占 typed 列和有效新根，重置临时堆／视图／克隆缓存后仅保存最终状态；`EventList` 无新槽余量的行仍走原先压实／保存／重新加载路径直接使用第一次推进的新根而不重做动作，并释放中间租约，保留 fresh events 与旧列表别名。分配证明绑定当前正式 clean 与冻结入口，拒绝物理提交，避免原始动作录像重复；不变更规则程序和编译对象身份。验证 `tests.test_duel_v2_soa_fused_steps`；组件通过与完整搜索、自然终局及400并发资格分别验收。

`soa_training_results.verified_training_games` 只在批次结束后校验并适配完整 GPU 标签：真实任务和模型版本、执行搜索回执、完整动作顺序、策略／访问数、双观察者 WDL、自然终局及独立原始动作重放必须一致。缺少 GPU 行或证据不在 CPU 补造，不写文件或授予闭环训练资格。`DeviceTrainingTape.value_rows_after_batch(include_frozen=True)` 显式导出完整双观察者记录，冻结对手仍不可学习；策略审计同时用 `numerical_rows_after_batch(include_frozen=True)`，默认导出行为不变。验证 `tests.test_duel_v2_soa_training_results`、`tests.test_duel_v2_device_training_tape`。


`soa_covered_search.create_soa_covered_search_bundle` 显式绑定已核验且独立的七队辅助价值模型，固定 covered_value、horizon3、rule opponent、32 次模拟及逐模拟世界采样；必须使用 staged teacher，不能覆盖规则／算法身份或沿用过期 scope。组件工厂和训练任务元数据绑定不授予训练资格。

`DeviceTieredSoARules` 为显式组件，默认未接入验收 CLI：宽队列使用较小 scratch，窄队列保留原容量，共用执行器、编译缓存和状态／动作存储。只有原始输入租约有效、静态调用确实返回 VM 容量错误且队列保护无错的行才重试；错误码 1 本身不足以证明容量失败。失败输出只能释放本次新发且归属已核验的租约；物理动作始终走大工作区一次，不重写录像。CPU 测试不证明实际显存或吞吐收益，须另做真实状态、容量压力、完整 Gumbel 与自然终局验收。


运行库单元可显式启用 `runtime_primitive_compare=True` 或 `direct_deepcopy_calls=True`，默认关闭且要求完整函数分单元编译。前者只对经边界核验的有限整数／布尔及字符串相等比较跳过已知拒绝这些类型的库探测；宽整数、浮点、Decimal、自定义对象及未知／坏引用走原处理。后者直接路由至原 deepcopy 实现，保留 memo、别名、参数、分配、步数和错误。两者均不改变 Context／Heap ABI 或非 runtime 编译单元；CPU 堆／错误／步数对拍和构造接线测试不代替真实 GPU、完整搜索及自然终局验收。


价值预热的纯策略采集与 Gumbel 策略标签分开：`DevicePureValueTape` 直接兼容物理游戏 sink，核对零 requested/completed/visits、合法选择、状态／动作租约、所有者、步号、版本及双方数值观察；完整自然终局后记录双观察者 WDL。任一原始行失败、截断或未结束都拒绝整批训练导出；冻结观察者可显式审计，但保留不可学习标记。`soa_pure_jobs` 不要求或伪造首次价值预检证据，完整保留真实 jobs_for／compare_jobs 的身份、牌序及独立 reference bank；不把任务残留 teacher 元数据当作实际搜索，也不授予完整训练资格。


引用片段支持真实共享 cell 的 `LOAD_DEREF`／`STORE_DEREF`，不缓存可变 cell 值；`LOAD_GLOBAL` 每次通过原字典查询，并保留缺失错误及入栈行为。`EXTENDED_ARG`／`NOP` 是零栈效应指令，仍分别消耗原逻辑步数、检查 deadline 并维护错误 PC。所有原调用和不支持指令继续作为片段边界；静态预算不因热点观察主体提优而扩大。须比较共享闭包写后读、跨调用 globals 替换／删除、未绑定错误、真实扩展参数、各步截止、全部临时堆／typed 列及输出别名；CPU 通过不代表新 GPU 编译或吞吐通过。编译公共声明变化须按真实单元 SHA 重建缓存，不能将旧对象冒充新实现。


纯价值校准结果适配须保留原始 `new_game` 的开局事件与缓存，只在重放每个动作后清理。每步必须同时具有实际 GPU 双观察者价值行、原始动作和独立 CUDA 推理回执，核对完整合法动作顺序、编码、信息哈希、模型版本、数值分数与精确 argmax；近似相等但选择不同仍拒绝。无误自然终局、完整连续轨迹及状态／随机数／别名一致后才输出既有学习格式；冻结观察者仅供审计，比较／reference 任务拒绝，缺失数据不从 CPU 补齐。该适配不授予预检、行为比较或闭环训练资格。
