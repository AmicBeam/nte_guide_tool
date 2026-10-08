# 离线批量搜索后端

## 目标与边界

目标是让 1,600 个独立对局通过 GPU 执行完整规则、观察编码与批量搜索，并测量真实终局吞吐与逐决策预算。真实局数、活跃搜索根数、在途假想世界数、节点池容量和操作系统工作进程数是不同指标。

后端位于离线 RL 层。正式 Python 引擎继续作为规则对拍 oracle；网站不导入数组 arena、Torch、CUDA 或新的批量调度器。数组存储能够分配 1,600 个槽位，不表示已经实现 1,600 局规则结算、搜索或 GPU 并发。

已有可运行路线是混合后端：完整正式规则和状态常驻少量 CPU 工作进程，网络与搜索统计按跨根请求批量送到指定设备。`gpu_rule_execution=false`；设备图存储原语和源码导出器都不是完整 GPU 规则解释器。混合后端的 1,600 局在途容量不能当成上述 GPU 规则并发目标已经完成；完整 GPU 链路仍在实现，吞吐与延迟必须由真实完整游戏压测确定。

## 当前基础组件

### CPU 数组 arena

`rl/batched_duel/arena.py` 提供显式容量的连续 int32 真实状态槽位和假想世界工作区。布局宽度由调用者提供，不能沿用旧两队编译后端的行宽冒充完整五队状态。

槽位引用必须绑定 arena、槽位和代次；工作区另绑定所属真实状态。输入和公开快照彼此隔离，释放后的引用及跨 arena 引用明确拒绝。真实槽位仍有工作区时不得释放。容量耗尽明确报错，不能覆盖在途状态。内存统计只说明实际分配的数组，不包括模型、激活、CUDA 分配器或驱动。

此 arena 不理解对局字段、不生成合法动作、不执行规则。独立的完整图 codec 负责保留状态字段；两者不能替代规则执行后端。

### 无损图编码与设备存储

`batched_duel/schema.py` 与 `codec.py` 用连续节点、边、字节载荷、计数和根索引数组保存完整状态图。保留字典顺序、列表/元组类型、三个实体类型、共享对象别名、未知动态字段和任意精度整数 RNG。没有把规则状态藏进 JSON 字符串。编码拒绝未知对象类型、循环、非有限浮点和容量溢出；解码拒绝越界引用、循环、重复字典键、不可达节点与过深嵌套。数组属于调用方，不能因外层 dataclass 冻结而认为数组不可修改。

`device_state.py` 显式把经过 ingress 校验的图复制到指定设备，以设备 `index_select` 批量分叉；返回主机快照是独立的验证操作。该组件只提供驻留和复制，尚不提供 GPU 规则执行或 GPU 观察编码。

### 批量推理与协作搜索

`batched_search/inference.py` 只接受具名观察/候选和模型身份一致的数值模型，显式构造冻结 FP32 网络。按模型分组并按实际 `批量行数 × 最长候选数 × 候选宽度` 控制 padded tensor 预算；切 microbatch 不删除候选。结果绑定请求、模型和权重版本。只求价值的叶子使用独立价值塔，并保留固定观察者是否为行动方的标志。

`traversal.py` 保留逐根 Gumbel 访问依赖、公开终局证明、信息集采样和两种具名 Q 尺度。`oracle_backend.py` 是显式 Python 对拍后端，不是 GPU 实现。`cooperative.py` 仅把显式 backend 协议调用变为挂起请求，在不同根之间分组批量调度；没有 bulk resolver 时默认拒绝，不静默串行回退 Python。队列墙钟必须由调用方在入队前锚定绝对单调时钟截止。

`nodes.py` 在指定 Torch 设备上维护 FP64 先验、累计价值和 int32 访问统计，根/节点引用绑定所属实例、对局和代次。该统计池自身不执行模拟、不读取对手暗牌，也不代表完整搜索已经迁移到 GPU。

### CUDA 编译与随机原语

`batched_duel/cuda_runtime.py` 显式使用 NVRTC 与 CUDA Driver API 编译、加载和执行 kernel，拥有自己的 stream，以事件和张量 stream 记录维护异步依赖。无 CUDA 时失败；不回退 CPU。参数只接受已声明类型的设备张量和有界标量，不接受裸地址/未知 struct；固定禁止 fast-math/FMAD 放宽，不由全局环境变量暗改 JIT 配置。

`random_source.py` 提供 CPython 整数种子展开、MT19937、高位 getrandbits、拒绝采样 randbelow、洗牌及两种既有 RNG 提交顺序的可移植 C/CUDA 原语。随机原语数值对拍不能替代完整 determinization、合法动作或规则对拍。

### CUDA 规则执行组件（完整链路尚未接通）

`vm_image.py` 将经过源码 SHA 验证的 CPython 3.10 程序降低为不可变数字指令、函数、模块、类和常量图；保留跳转、默认参数、闭包及精确数值。冻结程序与数字图分别校验身份。Windows 可读取显式冻结的 3.10 测试程序，不把本机 3.13 字节码静默当成同一协议。

`vm_heap.py` 在 CUDA 修改字典／列表、读取 UTF-8 和数值并深拷贝实体图。只读常量拒绝写入；资源耗尽、整数超范围和非法引用明确失败。设备压缩导出只收集可达对象，保留实体类型、未知字段与共享引用，不放松正式 codec 对垃圾节点或非法 JSON 状态的检查。

`cuda_vm.py` 真正解释已支持的控制流、调用、闭包和生成器；`vm_collections.py` 提供容器操作，`vm_objects.py` 提供冻结模块、C3 类查询、方法／属性及普通对象初始化，`vm_builtins.py` 提供有界内建操作。参数绑定区分位置限定、关键字限定、默认参数和可变参数。缺失指令／原语在入口拒绝，设备遇到未实现类型或语义也明确失败；没有回调 Python 引擎的 CPU 回退。集合暂不保证 CPython 哈希遍历顺序；异常／上下文管理、完整数值／格式化、随机类与完整规则入口仍未齐备。`supports_full_duel_rules=False` 必须保留。

`gpu_candidates.py` 在显式 CUDA 上从紧凑公开 token 构造 254 列候选，包括有效牌与可见原牌身份。显式 CPU 模式只用于数值对拍。主机 token 提取仍是过渡输入适配，不能声称规则和观察已全部在 GPU 生成，也未接入完整 GPU 搜索后端。缺失 CUDA、未知身份、错位批次、非有限数值和容量超限均拒绝，不截断候选凑成功。

`gpu_observation.py` 从紧凑公开 token 在显式 GPU 构造 2,420 列观察，完成角色散射、缺席默认值、资源归一化、牌库存量、金谷、限时增益与可见原牌统计。直接设备 token 入口严格校验类型、形状、身份域、重复角色座位、掩码和有限数值。主机公开 token 提取仍未替换为完整 GPU 投影；不能把这个过渡适配称为 GPU 规则／观察全链路。`tests.test_duel_v2_gpu_observation` 包含五套规则状态和 1,600 行真实 CUDA 编码对拍，行样本不是完整对局。

`vm_numeric.py` 提供有界整数／浮点运算、符号取整、序列原地相加及迭代容器比较。整数与 binary64 的数学相等直接比较精确表示，避免 `int64` 边界的强制类型转换和大整数舍入误判；超出已支持运算域仍明确拒绝。

`vm_reducers.py` 通过 VM 挂起／恢复生成器计算 `all/any/sum/min/max`，保留短路后仍被引用的生成器；`min/max` 的 `key` 回调暂未支持，不会忽略后继续。`vm_frame_gc.py` 从活跃执行帧和待调用输入在设备遍历引用，仅回收已无引用的悬挂生成器执行槽；可达生成器继续保留。受保护生成器的终结逻辑未实现时明确失败，不静默丢弃。

`vm_random.py` 将整数种子的 Random 对象绑定到已对拍的 MT19937 原语，在 GPU 保留每个对象的状态，支持洗牌、选择、取位、整数范围及 53 位浮点抽样。载荷按实际设备地址对齐，奇数行跨度也不产生未对齐访问。无参／非整数种子和完整 `getstate/setstate` 暂不支持，不能宣称任意 Python 随机对象兼容。它仍不能替代完整公开一致性采样流程。以上由 `tests.test_duel_v2_cuda_vm` 的设备执行用例验证，完整 GPU 对局仍单独验收。

上述组件的真实 CUDA 子集对拍、1,600 行标量／编码或实体复制测试都不代表 1,600 个完整 GPU 对局。完整五队合法动作、物理推进、公开一致性采样、观察、搜索预算、录像和行为审计仍须逐项接通并验证。

验证：`.venv/bin/python -m unittest tests.test_duel_v2_vm_image tests.test_duel_v2_vm_heap tests.test_duel_v2_vm_collections tests.test_duel_v2_cuda_vm tests.test_duel_v2_gpu_candidates`。没有 CUDA 时必须记录跳过；Windows 3.13 用显式冻结程序目录 `NTE_VM_TEST_PROGRAM_DIR`，不得以跳过或 source emission 成功替代设备执行。

### 驻留式正式规则与真实记录

`process_backend.py` 使用固定数量的独立工作进程保存完整 Python 状态，每个状态只在导入/明确导出时跨进程传输；后续规则请求携带绑定后端、工作进程、槽位、代次和游戏根的引用。发放时的不可变元数据也参与校验，不能伪造 actor/phase。每个后端承载一轮实验，`max_games` 限制该轮累计唯一游戏准入，不作为可无限补充的流式游戏池。

搜索分支使用正式 `simulate_action` 和公开一致性采样；物理推进使用正式 `apply_action`。假想状态不写公开录像。工作区引用通过弱引用释放队列回收；物理提交按工作进程批量执行，旧版本只能读取，不能再次推进。工作进程禁止导入 Torch/CUDA。超时或规则异常停止整个方法，取消和退出只终止本后端拥有的 Process 对象；不按裸 PID 清理无关任务。

`recording.py` 复用 `GameTelemetry`、既有角色/环合/终结采集和公开回放投影。每个物理动作保存状态 SHA，并逐步重放核验。私有 opening 与逐动作 journal 在运行中持久化，供异常后的证据恢复；截断行和未落盘结果不能当完整对局。完整私有原始与公开旁观录像分别输出。公开手牌只保留隐藏占位及已公开的复制/衍生/揭示牌，不导出隐藏牌库、RNG 或种子。

### 同款 serving 规划与设备统计

`serving.py` 保留当前 recovery serving 的公开一步获胜检查、最多 8 次证明、总计最多 32 次证明/树模拟、16 个根候选、3 步短终局探查和 natural-WDL。每次内部搜索双方使用行动方自己的冻结模型作为代理；真实对手在其物理决策时使用自己的模型。种子按当前网站的 version/side/model SHA 推导，禁止用真实暗牌或实体序号选策略。

预算模式必须明确：`simulations` 固定搜索量并受整轮截止约束；`wall_clock` 要求调用方在入队前提供单调时钟时间戳，共享 3 秒窗口，记录实际完成量。墙钟模式拒绝迟到证明/推理/统计结果；这比旧网站函数在迟到时仍接受已成立证明的行为更严格，不宣称高并发排队下与单人网站逐动作分支完全相同。GPU 运行不可抢占，因此预算不是硬实时延迟保证。

`statistics.py` 把真实遍历的节点创建、选择、备份和最终快照送到 `BatchedNodePool`。每根仍按访问依赖逐次推进，跨根批量计算。设备备份完成后若超过该根截止，恢复该根原来的访问/价值累计，不把半个或迟到模拟计为完成。取消只允许额外读取最终统计快照，不能启动新模拟。节点随根生命周期批量释放，不跨根串用。驻留后端可在调度线程内直接读取已校验的不可变 actor/phase/winner 元数据；只有已经确认清理完成的同代状态才可跳过空转 clean 请求。serving 的首个根推理可绑定同一状态、模型和观察者复用为树根先验／价值，避免重复推理；它不是跨物理决策缓存。固定模拟预算须保持原数值对拍，墙钟下收益由实际完整局测量。

### 完整对局驱动与冻结输入

`evaluator.py` 同时准入指定数量的真实对局，逐轮协作搜索、物理提交、记录和结束。`scripts/evaluate_duel_v2_batched.py` 默认只检查配置；`--run` 才启动，并要求新输出目录、明确预算模式和由共享账本分配的种子基数。1,600 局按 5×5 每格 64 局安排，行是先手模型、列是后手模型，轮换引擎 a/b 座位。同整数种子的两座位局属于一个配对根，不能宣称手牌完全相同。

`offline_sources.snapshot_revision` 从指定 Git 提交的白名单 blob 创建隔离源码，保留工作区并行修改；仅允许额外加入新源码，不能用 extra 静默替换已冻结规则。权重单独按模型清单校验。完整评测 CLI 在运行前核对 `source-manifest.json` 与 `source-revision.json`，不改 serving 绑定让不匹配输入强行通过。

`behavior.py` 独立审计实际物理行为：记录每侧合法出牌决策数及不同回合数、手动出牌、普通出击与终结。正常完整局一侧在至少两个回合有至少两次合法出牌机会，整局却既未出牌、未普通出击也未终结，记为空转；普通出击但不出牌不自动判空转。至少 8 个不同正常局出现该症状时 P0 停止整个方法，保留已完成与截断证据。镜像的两侧仍只计一个失败局；异常／截断不冒充正常局。任何已确认空转都会令 `evaluation_accepted=false`，即使未达到提前停止阈值；CLI 此时退出 1。该闸门不改动作排序、模型、奖励或预算，也不自动把网络回退替换为规则策略。

物理容量完整、行为验收与学习／强度批准分别记录，不能把因对手正常行动而结束的空转局当成模型已学会。验证：`.venv/bin/python -m unittest tests.test_duel_v2_batch_behavior tests.test_duel_v2_batch_recording tests.test_duel_v2_batched_evaluator`。

只有全部实际终局、记录和重放核验完成才标记游戏执行完整；只有实际 1,600 局满足这些条件才标记容量完整。质量、训练全流程和完整训练报告批准独立保留为 false。二维表仅使用正常完整局作分母，未执行/失败/截断单列；Wilson 区间是未作成对聚类修正的参考，不用作模型接受门槛。矩阵不代替新旧、换牌、变化构筑等完整训练报告项目。

`rule_ir/portable_program.py` 是 CPython 3.10 的可审计源程序导出器，保留实际函数/闭包/类/常量/跳转及源码 SHA，显式列出所需原语，未知依赖失败。它不执行设备规则，Windows 的评测链路也不需要导出主机字节码。不要把导出完整或 `unresolved=[]` 当成 CUDA 规则可执行。

### 每根预算账本

`rl/batched_search/budget.py` 的 `SearchLimits` 分别记录决策墙钟、总模拟量、公开证明上限、Gumbel 根候选上限、短终局探查和树深；默认分别为 3 秒、32、8、16、3、10。短终局探查的 3 步不能解释为整个搜索树深度。

`DecisionBudget` 在根进入队列时开始单调时钟计时。公开证明和树模拟共用总额度；预约包含排队工作，取消不返还额度。账本分开统计预约、实际开始、完成、取消和丢弃。

每根只允许一个未结束模拟，保留原搜索的逐次访问更新依赖；并行维度来自不同根。公开证明阶段结束后不能回到证明阶段。工作票据绑定预算实例、对局、根代次和序号；跨根、伪造字段、重复开始或重复完成明确拒绝。

调用者必须在 GPU 事件真实完成后才调用 `complete()`，入队或 kernel 提交不能算完成。取消或到期后的结果返回不采纳，不增加完成计数。该组件由一个调度线程拥有，不提供跨线程原子计数，也不保证 GPU 硬实时截止。

### 搜索身份

预算组件不选择动作，也不决定 Q 尺度。批量规划器必须绑定实际模型、规则、具名观察/候选和搜索版本：当前 recovery 高级策略使用 natural-WDL；旧冻结评估的 legacy completed-Q 仅用于明确标记的复现。不能因为预算相同而混称同一算法。

完整合法动作及顺序保留；16 个 Gumbel 根候选不是合法动作容量。每次模拟继续遵守现有公开一致性采样和种子推导，行动方可见输入不得包含实体编号、对方暗牌身份或 RNG。

## 后续接入顺序

1. 核验驻留规则、物理记录、协作规划及设备统计的组合，包括真实五队、私有选择、取消/迟到和完整重放。
2. 在 Windows 核验冻结数值模型与 CUDA/NumPy 选择、WDL、根访问更新一致性；明确 CPU 规则和 GPU 数值各自的实际成本。
3. 完整检验模型数值身份、固定观察者叶子价值、采样种子与原搜索一致性，拒绝把仅有源码导出/统计 kernel 的原型标为完整 GPU 规则后端。
4. 验证真实取消、异常和记录队列容量；完整回放从真实动作重放核验，假想分支不进入公开录像。
5. 以独立新种子依次测 64、256、512、1,600 个对局，报告完成局/小时、每根实际模拟量、部分结果、回退、p50/p95/p99/max 延迟及 CPU/GPU 峰值。仅分配槽位或启动 kernel 不算对局容量通过。

完整矩阵、训练和模型采用继续遵循训练手册与完整报告规范；单轮资料只进入 artifacts。已由用户停止的任务不因实现新后端而自动恢复。

## 验证入口

```bash
.venv/bin/python -m unittest tests.test_duel_v2_batch_arena tests.test_duel_v2_batch_budget -v
.venv/bin/python -m unittest tests.test_duel_v2_batch_codec tests.test_duel_v2_batch_device_state tests.test_duel_v2_batch_cuda_runtime tests.test_duel_v2_batch_random -v
.venv/bin/python -m unittest tests.test_duel_v2_batch_inference tests.test_duel_v2_batch_nodes tests.test_duel_v2_batch_traversal tests.test_duel_v2_batch_cooperative -v
.venv/bin/python -m unittest tests.test_duel_v2_batch_process_backend tests.test_duel_v2_batch_recording tests.test_duel_v2_batch_serving tests.test_duel_v2_batch_statistics tests.test_duel_v2_batched_evaluator -v
.venv/bin/python scripts/check_duel_v2_batch_inference.py
.venv/bin/python scripts/evaluate_duel_v2_batched.py
```

没有 CUDA 的主机必须明确记录设备用例跳过。`scripts/validate_duel_v2_batch_gpu.py` 默认只输出配置，只有显式 `--run` 才执行合成节点统计验证；其 `game_executions=0`，不能作为真实对局容量或强度证据。上述用例和原语对拍也不能替代五队完整规则、真实端到端 GPU 性能或策略强度验收。


### 校准统计的独立接入

`value_metrics.py`只对已经采集的合法观察进行`value_only`批量推理；保留观察者行动标志、WDL顺序与每局等权，拒绝身份错位／过期／未知设备，不执行规则或改模型。`recovery_round.warm_value`可显式使用CUDA后端，默认NumPy不变；需本轮数值对拍和TF32关闭。GPU批量校准统计不代表完整GPU对局后端已经完成。
