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
