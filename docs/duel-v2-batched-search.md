# 离线批量搜索后端

## 目标与边界

目标是让 1,600 个独立对局通过 GPU 执行完整规则、观察编码与批量搜索，并测量真实终局吞吐与逐决策预算。真实局数、活跃搜索根数、在途假想世界数、节点池容量和操作系统工作进程数是不同指标。

后端位于离线 RL 层。正式 Python 引擎继续作为规则对拍 oracle；网站不导入数组 arena、Torch、CUDA 或新的批量调度器。数组存储能够分配 1,600 个槽位，不表示已经实现 1,600 局规则结算、搜索或 GPU 并发。

当前正式GPU训练和搜索互搏遵循[训练手册的设备内并行硬要求](duel-v2-public-training.md#当前硬要求gpu设备内批量并行禁止cpu规则回退)。下述CPU／混合组件只保留诊断或历史用途，不作为GPU失败后的替代方案；`cuda_vm`规则通过也不能代替完整设备内搜索循环的实际验收。

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

### 正式规则冻结入口与CUDA批量适配

当前正式互搏统一使用七队：创生、覆纹、小吱、真红、浊燃、倾陷、灵可。`evaluate_duel_v2_batched.py`默认七队，`--runtime`仅接受seven；冻结程序CLI也使用七队。旧recovery格式仅保留为历史源码／数据兼容接口，不作为当前互搏的另一套运行方案。`--device`只指定推理／统计设备；规则后端由`--rule-backend resident_cpu|cuda_vm`明确选择，GPU规则失败不回退CPU。

`gpu_schema_contract.py`绑定七队具名3479列观察、1576列候选和当前采样器；独立设备编码及公开图投影覆盖有序顶／底牌、手牌置顶／置底、退出终结、公开计次与抗性。加载冻结程序校验运行时和来源SHA，旧维度不能填充为七队。七队矩阵为49格，先手为行／后手为列，对角线真实内战，仍按实际完成分母统计。

`gpu_rule_capacity.py`提供不加载Torch的容量计划及槽位账本。以真实状态、清理根、采样、嵌套终局探查父引用、推进和返回重叠推导峰值，持久状态与VM执行堆分开。默认每次GPU调用至多32行；1600个已准入搜索根仍保留，不把分批行数称为根并发。导入、参数绑定、推进、提交和导出遵守同一分批界限；源／目标堆使用各自步幅，输出压回持久限额，溢出明确失败。统计池实际候选容量与计划一致。内存报告含临时编码数组、GPU缓冲、搜索池和未建模预留，公式与实测峰值分开；不适配预算时拒绝，不能缩小候选或状态冒充通过。

本地检查示例（不连接设备、不加载权重、不预约种子）：

```bash
.venv/bin/python scripts/evaluate_duel_v2_batched.py --rule-backend cuda_vm --device cuda --games 1600 --search-roots 1600 --gpu-memory-budget-mib 14336
.venv/bin/python scripts/build_duel_v2_gpu_rule_program.py --build --output artifacts/rl-evals/NEW_RUN/frozen-program
.venv/bin/python scripts/validate_duel_v2_gpu_seven.py
```

14336MiB是上述示例的显式可用预算，不是设备实测。正式`--run`还须新输出目录、明确模型目录、冻结程序、独立种子与预算模式；Windows只能读取CPython3.10冻结的程序。设备对拍、完整局／回放以及64→256→512→1600实测仍是后续验收，不因本地计划通过而批准运行性能或模型质量。

设备上线后，`validate_duel_v2_gpu_seven.py --run --program FROZEN_JSON --output NEW_DIR`用于14个七队起手／正常行动根的有界规则、完整候选顺序、观察编码、采样、模拟与物理推进对拍；不读取模型或训练，不代表完整对局或1600容量。默认仅输出配置，没有CUDA明确失败；正式设备执行须另加外层进程时间保护并保留原始失败证据。

`gpu_rule_entries.py` 提供实际正式函数的源码入口：行动方、终局/价值、完整合法动作与公开观察、物理推进、假想推进、清理、观察者一致采样、公开根元数据及一步终局探查。公开视图不按旧座位表截断。采样显式绑定现有恢复运行时，不偷偷改用较旧的默认采样器；该运行时已有的编队与选择范围限制继续生效，目录包含新角色不表示旧恢复编码和采样已经支持它们。

`gpu_rule_program.py` 与 `build_duel_v2_gpu_rule_program.py` 只构建/核验源码程序，不执行设备规则。构建须CPython3.10，Windows3.13只加载显式冻结程序；入口集、程序身份与来源SHA必须一致。来源检查在读取前拒绝越界路径、敏感文件名及指向仓库外的符号链接。默认CLI只检查；`--build --output NEW_DIR`才创建新产物。`declared_capabilities_ready`仅表示声明的指令/原语齐备，`device_ready`和`device_verified`不会因源码生成成功而设为true。`vm_image`的入口依赖扫描同时覆盖函数内部的相对导入；导入的实际被调函数缺指令/原语时，不能把外层入口当作就绪。

`GpuRuleBackend` 对接协作搜索协议，提供有界CUDA图状态槽、代次/归属引用、输入参数合并、批量规则入口调用、分叉/释放、物理提交与明确导出。CPU设备、缺失能力及来源失配被拒绝，不回退到驻留Python规则后端。`gpu_rule_arguments.py`在CUDA合并状态与明确给定的参数字面值；`gpu_rule_projection.py`从GPU公开图生成token并调用原设备观察/候选编码器，运行路径不调用主机token提取或NumPy编码器。既有具名编码表的未知角色/动作仍须显式拒绝，不能填默认值或截断候选。CPU完整原始记录与正式引擎重放属于明确的取证验证边界，不用于生成GPU规则结果。

上述批量适配、上下文与数值原语是研究实现，完整CUDA规则仍需设备逐项对拍与完整对局验证。`supports_full_duel_rules=False`继续保留；源码入口、宿主C++诊断、独立模块用例或低预算动作前缀均不能代替1600场正式预算终局验收。

`vm_numpy.py`与`vm_json.py`接入解释器原生调用、数组索引/写入、迭代、算术和垃圾回收；`vm_stdlib.py`接入回调排序、生成器消费、Counter、组合/排列、枚举与super。只实现明确有界子集，超出类型、维度或容量明确失败。Counter保留零值键；str/int枚举字典键遵循其标量等价关系。`clone_state`仅接受已经规范化且水合不会改动的实体状态，检查身份、别名、阵营与编号计数；拒绝把普通字典deepcopy冒作完整水合。冻结器对`sys.modules[__name__]`只保存当前纯模块的数值/字符串常量契约，不保存宿主进程全部模块；动态模块索引拒绝。

默认严格能力检查仍列出并拒绝`uuid.uuid4`。正式对局验证可显式构造`GpuRuleBackend(..., guard_standalone_entities=True)`，该选项只允许进入包含独立实体兜底分支的入口；一旦实际调用该分支，设备立即报`UNSUPPORTED_TYPE`。它不生成替代UUID、不启用CPU规则回退、不改变完整规则/设备验收标记。来源能力报告继续显示缺失UUID，并与设备执行准入及运行时拒绝列表分开。正式对局使用的局内实体编号也必须继续经过真实设备对拍。

验证入口：`.venv/bin/python -m unittest tests.test_duel_v2_gpu_rule_program tests.test_duel_v2_gpu_rule_imports tests.test_duel_v2_gpu_rule_backend tests.test_duel_v2_vm_image tests.test_duel_v2_vm_stdlib tests.test_duel_v2_vm_numpy tests.test_duel_v2_vm_json tests.test_duel_v2_vm_integrations`；`scripts/check_duel_v2_gpu_rule_backend.py`默认只检查，显式`--run`才进行有界CUDA验证。CPU主机须保留设备跳过记录，单次报告和程序包只进入artifacts。

### 驻留式正式规则与真实记录

`process_backend.py` 使用固定数量的独立工作进程保存完整 Python 状态，每个状态只在导入/明确导出时跨进程传输；后续规则请求携带绑定后端、工作进程、槽位、代次和游戏根的引用。发放时的不可变元数据也参与校验，不能伪造 actor/phase。每个后端承载一轮实验，`max_games` 限制该轮累计唯一游戏准入，不作为可无限补充的流式游戏池。

搜索分支使用正式 `simulate_action` 和公开一致性采样；物理推进使用正式 `apply_action`。假想状态不写公开录像。工作区引用通过弱引用释放队列回收；物理提交按工作进程批量执行，旧版本只能读取，不能再次推进。工作进程禁止导入 Torch/CUDA。超时或规则异常停止整个方法，取消和退出只终止本后端拥有的 Process 对象；不按裸 PID 清理无关任务。

`recording.py` 复用 `GameTelemetry`、既有角色/环合/终结采集和公开回放投影。每个物理动作保存状态 SHA，并逐步重放核验。私有 opening 与逐动作 journal 在运行中持久化，供异常后的证据恢复；截断行和未落盘结果不能当完整对局。完整私有原始与公开旁观录像分别输出。公开手牌只保留隐藏占位及已公开的复制/衍生/揭示牌，不导出隐藏牌库、RNG 或种子。

### 同款 serving 规划与设备统计

`serving.py` 保留当前 recovery serving 的公开一步获胜检查、最多 8 次证明、总计最多 32 次证明/树模拟、16 个根候选、3 步短终局探查和 natural-WDL。每次内部搜索双方使用行动方自己的冻结模型作为代理；真实对手在其物理决策时使用自己的模型。种子按当前网站的 version/side/model SHA 推导，禁止用真实暗牌或实体序号选策略。

预算模式必须明确：`simulations` 固定搜索量并受整轮截止约束；`wall_clock` 要求调用方在入队前提供单调时钟时间戳，共享 3 秒窗口，记录实际完成量。墙钟模式拒绝迟到证明/推理/统计结果；这比旧网站函数在迟到时仍接受已成立证明的行为更严格，不宣称高并发排队下与单人网站逐动作分支完全相同。GPU 运行不可抢占，因此预算不是硬实时延迟保证。

`statistics.py` 把真实遍历的节点创建、选择、备份和最终快照送到 `BatchedNodePool`。每根仍按访问依赖逐次推进，跨根批量计算。设备备份完成后若超过该根截止，恢复该根原来的访问/价值累计，不把半个或迟到模拟计为完成。取消只允许额外读取最终统计快照，不能启动新模拟。节点随根生命周期批量释放，不跨根串用。驻留后端可在调度线程内直接读取已校验的不可变 actor/phase/winner 元数据；只有已经确认清理完成的同代状态才可跳过空转 clean 请求。serving 的首个根推理可绑定同一状态、模型和观察者复用为树根先验／价值，避免重复推理；它不是跨物理决策缓存。固定模拟预算须保持原数值对拍，墙钟下收益由实际完整局测量。

### 完整对局驱动与冻结输入

`evaluator.py` 一次驻留指定数量的真实对局，通过有界滚动窗口准入搜索。每波完成的根立即物理提交，空位补入等待最久的对局，不再等整批慢根完成后统一落地。`--search-roots` 独立设置同时搜索数，默认 `min(games, workers*2)`，这是待实测的保守起点，不是最优吞吐结论；`--games` 是总驻留局数。`scripts/evaluate_duel_v2_batched.py` 默认只检查配置；`--run` 才启动，并要求新输出目录、明确预算模式和由共享账本分配的种子基数。1,600 局按 5×5 每格 64 局安排，行是先手模型、列是后手模型，轮换引擎 a/b 座位。同整数种子的两座位局属于一个配对根，不能宣称手牌完全相同。

滚动窗口采用新的离线协议：`deadline_scope=search_admission`。驻留等待不消耗尚未创建的根预算；准入后规则队列、推理、证明、树搜索均计入同一个3秒窗口，超时不续费。单独报告驻留等待、搜索延迟和真实提交后的端到端延迟p50/p95/p99。它不修改网站从请求开始计时的3秒协议，不能把不含驻留等待的搜索延迟当作用户响应。全局截止仍覆盖等待及全部工作，并保留30秒落盘时间。

`simulations` 模式严格检查完成工作量：公开证明与树模拟合计达到请求额度且树完整，或唯一合法动作／已完成公开必胜证明，才可物理提交。未达预算、未取得策略或搜索不支持的根结束为未完成记录，不用回退动作冒充完整固定预算评估。`wall_clock` 模式保留明确记录的部分搜索和网络回退，二者分别验收。`fixed_budget_complete` 还要求全部对局正常终局并通过重放，局部动作检查通过不代表整局验收通过。

GPU节点统计池按 `33*search_roots` 分配和循环回收，而不是按全部驻留局数分配；保留原32工作额度、完整合法动作容量和所有者代次验证。调度器记录每类规则操作、统计操作、推理的批数、请求数、总墙钟与最大批耗时；驻留后端另报worker累计执行及IPC等待。驻留CPU后端明确支持混合方法时，同一波的编码、采样与步进合并为一次批量RPC，统计后端仍按方法分组；请求身份、每根顺序及截止保持不变。父进程墙钟与各worker累计时间不可直接相加解释为阶段耗时。完整CUDA规则后端仍未完成，本次调度不能冒充GPU规则接入。

`offline_sources.snapshot_revision` 从指定 Git 提交的白名单 blob 创建隔离源码，保留工作区并行修改；仅允许额外加入新源码，不能用 extra 静默替换已冻结规则。权重单独按模型清单校验。完整评测 CLI 在运行前核对 `source-manifest.json` 与 `source-revision.json`，不改 serving 绑定让不匹配输入强行通过。

`behavior.py` 独立审计实际物理行为：记录每侧合法出牌决策数及不同回合数、手动出牌、普通出击与终结。正常完整局一侧在至少两个回合有至少两次合法出牌机会，整局却既未出牌、未普通出击也未终结，记为空转；普通出击但不出牌不自动判空转。至少 8 个不同正常局出现该症状时 P0 停止整个方法，保留已完成与截断证据。镜像的两侧仍只计一个失败局；异常／截断不冒充正常局。任何已确认空转都会令 `evaluation_accepted=false`，即使未达到提前停止阈值；CLI 此时退出 1。该闸门不改动作排序、模型、奖励或预算，也不自动把网络回退替换为规则策略。

物理容量完整、行为验收与学习／强度批准分别记录，不能把因对手正常行动而结束的空转局当成模型已学会。验证：`.venv/bin/python -m unittest tests.test_duel_v2_batch_behavior tests.test_duel_v2_batch_recording tests.test_duel_v2_batched_evaluator`。

只有全部实际终局、记录和重放核验完成才标记游戏执行完整；实际1,600局全部满足时标记 `resident_1600_full_games_complete`。`concurrency_1600_full_games_complete` 额外要求实测峰值同时搜索根为1,600；有界窗口跑完1,600局不冒充1,600搜索同时执行。质量、训练全流程和完整训练报告批准独立保留为 false。二维表仅使用正常完整局作分母，未执行/失败/截断单列；Wilson 区间是未作成对聚类修正的参考，不用作模型接受门槛。矩阵不代替新旧、换牌、变化构筑等完整训练报告项目。

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

### CUDA规则虚拟机的有界执行约束

设备虚拟机加载后明确保证至少64KiB线程调用栈；不降低既有更大上限，驱动不满足时失败。图参数核对实际CUDA声明与注册ABI。生成器文本拼接、列表追加、字典关键字构造／更新和可调用对象的类型方法按设备续接执行；不改为CPU规则。不可变值深拷贝直接保留已校验引用，容器使用64位代次备忘表，避免每个小字段重复清空整个堆；GC及多次复制仍保留共享／循环引用语义。

VM临时列表按几何增长预留边空间，避免方法调用或元组临时分配打断连续尾部后，每次追加都复制整个列表前缀。仅LIST节点的`extra`记录其预留结束边索引；列表长度、顺序和元素别名仍由原有计数及引用决定，移动到分配尾部使旧预留失效。深拷贝目的列表和规范化压缩导出清除该临时记录；容量不足仍明确失败，临近容量上限时保留恰好能放下的单次追加，不截断内容。该优化不提供一般堆GC，也不改变完整搜索验收门槛。验证入口为`tests.test_duel_v2_vm_list_capacity`及既有VM／heap回归；Windows显式加载CPython3.10冻结的`NTE_VM_LIST_CAPACITY_PROGRAM`。

工作区包含多次公开观察和采样的临时对象，最低262144节点、524288边和32MiB payload；备忘表为int64，常量节点按65536上限计入预算，超界拒绝。节点、边和payload错误码优先保留，不能伪装成不支持的属性。该规划仍是上限估算，需实机峰值与完整局验证；内存能分配不等于1600完整搜索对局通过。

七队设备对拍将14例换牌／正常回合根组成同一真实GPU批次，核对动作顺序、公开编码、教师选择、采样、模拟和物理提交。正式200万步上限及期限保留；增大步数的诊断不作为正式回执。未完成14例或完整CUDA32链路时，`device_verified`和1600完整局结论继续为false。

设备文本方法支持`removeprefix/removesuffix`的单次精确UTF-8前后缀删除，空后缀与未匹配输入保持原值，不能用字符集合strip替代。`id`仅提供活跃VM行内已校验对象引用的身份比较，保留别名和独立副本的区别，不返回CPython宿主地址，也不承诺跨行／跨导出身份数值。对应真实CUDA用例位于`tests.test_duel_v2_cuda_vm`，原语通过仍不批准完整规则或设备内搜索。

### 完整设备搜索的接口与验收

长期门槛见[设备搜索验收协议](duel-v2-device-search-acceptance.md)。`device_rule_protocol.py`约定GPU状态、代次／所有者、合法动作图租约、动作类型、教师选择和固定观察者价值接口。根动作和当前假想世界动作不得共享会被覆盖的临时图；错误行停止规则，但仍须按有效租约清理已分配资源。

`device_inference.py`保留冻结模型数值和具名编码，按GPU模型索引批量推理；`device_search.py`使用设备阶段机，`device_statistics.py`、`device_info_key.py`和`device_graph_pool.py`分别管理树统计、完整合法前缀信息集哈希和分页所有权。`gpu_seven_candidates.py`及`gpu_seven_observation.py`的设备入口保留逐行错误掩码，拒绝热路径主机验证、UUID请求标识或动态CPU索引。

`cuda_conditional_graph.py`静态构建WHILE图，设备谓词、迭代数和单调设备时钟控制继续／停止；`CudaModule.launch(capture_safe=True)`只在调用者当前流启动，不产生条件图不允许的事件记录／等待节点。`CudaVm.launch_device`是掩码规则原语，未激活行不调用规则，错误状态留在CUDA，不能用旧主机RPC循环包装后声称闭环已经完成。普通诊断入口仍可在完整批次后显式导出／对拍。

正式搜索的世界种子包括SHA256散列最高位外的第257位和模拟序号进位。VM原生MT种子按Python `abs(integer)`的实际有效uint32字长展开，支持至65个有符号字节；超限明确失败，不能截断64位或把固定40字节填充当成另一条RNG。PCG64搜索噪声是独立随机源，初始状态可在整批准备时上传，热路径按每行合法候选数推进，不能为填充候选消耗随机数。

这些是工程接口和有界原语。源冻结、接口存在、独立CUDA图和1600行原语通过，都不能代替完整正式规则、完整Gumbel遍历、真实终局矩阵、数值／随机一致性、实际并发和显存峰值验收。缺GPU初始化、动作绑定、规则适配器、共享工作区或真实终局时，完整链路结论仍为未完成。

教师策略的`max(range(...), key=(score, -index))`要求range迭代续接和同类型tuple/list的词典序比较，不能以新的评分器替换原教师择优。GPU JSON接受可直接JSON编码动作的`default=str`参数；需要任意对象转换的输入仍明确失败，此限定不等于实现了通用Python回调。

VM字典键视图使用内部tag80保留原字典所有者；len、布尔值、迭代、membership及集合交并差／相等读取活跃字典，不把keys()转成永久快照或任意可序列化状态。视图的GC跟踪其所有者；需要导出时先按原规则转为普通容器。此原语验证不代替完整效果时钟及完整对局对拍。

持久图与规则临时堆分开分配。`CudaVm.prepare_device_workspace/launch_into_device`预分配固定数量的工作槽，执行时就地复用，不为每个持久根fork规则堆；调用结果借用工作槽，下一次调用前必须消费或压缩。`prepare_compaction_workspace/compact_into_device`在设备端保留全部可达字段与别名，允许较小的输出工作槽，超界保留错误；失败调用和未激活行不得覆盖目的图。压缩目的地只能是临时槽，不能覆盖已有存活租约。`DeviceRuleArgumentBinder.bind_into_device`同样使用预分配参数输出，不能直接修改持久局面。

`GraphProjectionKernel.project_tokens_device`输出完整合法动作的原始候选tokens和动作引用，不分配全批密集候选特征。消费方必须在设备端编码分片时完成原有逐行校验，再做推理及完整合法前缀哈希；原始tokens本身不是已经验收的网络输入。不得缩小合法动作上限来节省显存。

每次真实决策须按实际`(job_seed, step, actor)`选取原搜索的PCG64初态，不能把一次初始化的连续流跨决策沿用。`DevicePCG64.reset_device`接受设备选出的完整六字段状态，检查奇数增量及uint32缓存标志，错误行／未激活行保留原状态，不推进随机数；SeedSequence展开仅在批次准备时完成。以上新接口的源码与本地检查不代表新的CUDA实机、完整规则或真实终局验收。

`DeviceRuleResultReader`只从同一冻结VM的实际acting_side／finished输出引用和对应图解码角色、阶段、胜负及版本，保留设备错误；不是另一份CPU元数据判定。`bind_action_into_device`将已按租约核验并压缩的动作图接入临时规则堆，保留未知字段／别名，先核对引用、边界与容量后再写入；调用者仍须检查动作索引、所有者、代次和来源版本。两个接口均不构成完整RuleAdapter或正式学习批准。

`DeviceRuleLanes`静态捕获嵌套WHILE，由GPU游标选择固定数量的临时槽，处理末尾不满一批的根并在设备端写回租约；CPU只构图一次，不读取行谓词或逐根排队。错误根的有效租约仍保留用于清理，规则执行另按错误掩码停止。选择器的processed／peak只统计可执行的槽位选择，不能当成完成规则调用、完整局或1600验收。`CudaVm.load_graph_into_device`只复制实际使用的节点／边／payload到较大VM临时堆，保留引用，不把每个持久根都扩展成大堆；超界或未激活槽暴露空图并保留错误门槛。

紧凑状态存储使用独立节点／边／payload页池，保存各组件的实际代次。读回和释放前统一核对三池的所有权、代次、页表和字节长度；损坏任何组件都不能先释放其他组件。失败搜索的有效租约仍可释放，容量不足回滚本次部分保留，未知字段和别名原样保存；页存储及固定槽调度检查仍不替代正式规则和完整搜索验收。

采样器的`own.sort(key=lambda c: json.dumps(c, sort_keys=True))`依赖list.sort原地稳定排序和局部模块闭包。VM支持key／reverse关键字，排序返回None并保留原列表身份；key回调期间列表暂空，key异常恢复原内容，修改列表时恢复排序内容并抛出ValueError。故障PC是已自增的下一指令地址，诊断须同时核对前一条执行指令，不能因报错地址落在LOAD_CLOSURE就直接改闭包语义。此支持仍待对应同源CUDA及完整采样对拍确认。

标量VM每行有独立堆、帧和RNG，规则入口默认`rule_threads_per_block=1`，让小批独立工作槽分布到更多GPU块；原先把少于128行全部放进一个块会把规则工作局限在单个SM。该启动参数显式保留，1／128几何用例须核对逐行输出和随机一致性。启动几何变更不是实测提速结论，完整采集吞吐仍只能由本轮同源完整终局和计时回执给出。

`LIST_EXTEND`字节码的生成器展开与`list.extend()`的方法返回协议不同：生成器在设备帧中续接，展开结束保留原目标列表栈，不额外压入None；列表／元组星号展开、迭代器和耗尽后复用分别对拍。设备端`type()`只返回已冻结的原始内建／实体类型引用，不把bool擦除成int；未表示类型或动态造类明确失败。`initialize`为独立冻结扩展，调用正式new_game；参数只作数据入站，开局、实体、牌序、RNG和公开board须经过实际CUDA完整对拍，不能用源码捕获代替。

`device_rule_calls.py`将checked图租约通过设备cohort接到正式clean/sample_world/simulate_action/apply_action入口，复用一个大VM工作区，动作先按来源slot／代次／owner核验并压缩。`device_rule_presentation.py`提供同一工作区的正式actor／finished元数据及公开合法决策投影；候选仅持久保存原始tokens，密集编码由消费方分片执行。元数据与决策缓冲每次重置，条件循环期限错误进入输出错误，不允许复用上一批结果冒充新请求。两个组件仍不等于完成的规则适配器：固定观察者价值、教师选择一致性、GPU初始化、动作租约生命周期和完整真实终局须联合验证后才能放行正式服务。

VM列表／元组重复支持序列乘整数及整数乘序列，保留重复元素的别名；列表`*=`保留原容器身份，非正次数得到空序列。重复先核对完整边容量再分配，不截断、不逐项造深副本；生成器星号展开后的`[0.0] * len(CARD_IDS)`纳入同一设备回归。源码支持不代表完整采样预检或完整局已经通过。

冻结器保留已捕获函数在真实模块命名空间中的静态别名；不能只保存函数代码而漏掉作为contract参数调用的`feature_names`绑定，也不动态发现无关模块函数。暂停帧GC对tag75累加器只追踪source/result/key/keys/default-or-sort-owner/current-item引用，op/caller/index/options整数不作heap引用；尚未建帧的mode7累加器作为pending root进入GC。小帧池下的排序回调、星号展开与保留生成器纳入设备压力回归。此GC仅回收暂停帧，不回收堆容量，不能据此宣称initialize节点容量失败已解决。

固定观察者价值计划在CUDA调用正式actor／finished、viewer公共观察和terminal_value，viewer不随当前行动方切换，非终局显式标记value-only上下文。`teacher_continue`为显式冻结的三步covered-value规则续演扩展，复用原教师选择、simulate_action及clean，终局提前结束；它和初始化扩展均须额外冻结与实测，不从历史默认entry集合继承资格。公开投影／图调用组件接线不代表完整Adapter或完整终局已经批准。

`device_teacher_pipeline.py`提供 collector 可显式接线的三步 continuation：先以独立租约在同一设备状态池复制输入，再按实际 actor／finished 元数据调用原教师决策、simulate_action 和 clean，每步将可达输出压缩回分页池，下一次 VM 调用复用临时工作区。三步之间不积累上一调用的临时节点，不改变 horizon、教师评分或正式规则；终局保留已复制状态。独立静态计划不消费外层搜索的调用配额，热路径无 CPU 规则或逐根主机调度。输入租约受保护，中间状态和动作图按 owner／generation 核验后单次释放，重复执行清理旧输出，错误保持在设备向量中。阶段错误可在整个批次返回后读取；此组件没有完整规则验收回执，也不代替完整 Gumbel、1600 真根和终局／重放验收。

流水线回归入口为 `tests.test_duel_v2_device_teacher_pipeline`；Windows 从同源 CPython 3.10 的 `program()` 导出夹具，并通过 `NTE_DEVICE_TEACHER_PIPELINE_PROGRAM` 显式加载。夹具公开决策必须遵守 `(legal_actions, public_view)` 顺序及现行投影协议。受控 CUDA 用例检查三步、终局遮罩、输入不变、首次代次零、部分工作波、重复执行租约回收和 foreign-owner 错误恢复；真实规则另以整个 GPU 批次后的完整状态 oracle 核对。夹具通过不能推广成真实搜索容量或学习批准。

条件图的 capture／body 流使用独立的原生非阻塞 CUDA 流，并由 `torch.cuda.ExternalStream` 接入 PyTorch；不依赖 `torch.cuda.Stream()` 的轮转流池来保证每个长期存在的嵌套计划互不别名。捕获图及张量分配器可能在逻辑计划关闭后仍引用原始流，因此这些句柄保留到有界工作进程退出，由进程 CUDA context 统一回收。计划关闭清理逻辑所有权、图和模块，且幂等；尚未交给 PyTorch 的创建失败句柄单独销毁。所有流在静态准备时创建，设备执行不按根／tick 新建流。`tests.test_duel_v2_cuda_conditional_graph` 覆盖超过池大小的独立句柄、嵌套捕获、重复重放和关闭后的分配器句柄有效性；该回归不替代真实规则搜索验收。

吞吐不能由显存驻留、图捕获或部分模拟的次数推断。`device_performance_gate.py`只接收整批 GPU 执行并同步结束后的实测数据，单独报告准备／编译与稳态执行耗时、完成模拟总数和完整根决策吞吐；部分搜索的完整决策吞吐为零。诊断使用独立 epoch 和 monotonic 硬截止，任一时钟耗尽即超时；准备阶段和执行阶段分别计时，超时应由所属进程控制器停止本任务子树并保全报告，不停止其他进程。默认诊断预算为准备 300 秒、执行 120 秒，预算可配置；这不是训练吞吐验收阈值。

正式 GPU 训练执行器必须在正确性、完整终局／原动作重放和恢复检查之外调用 `require_training_performance`：1600 个独立种子真实根、设备驻留搜索、零 CPU 规则／逐根搜索回退、完整既有策略合法候选、32 模拟／cap16／horizon3 全部完成且无错误或超时，并以同源、不同批次的连续重复测量核对已确定的稳态耗时门槛。门槛尚未确定、测量不足或任何项目失败时明确拒绝进入训练。该工具本身不提供规则执行器资格或用户学习授权；不能用夹具报告绕过 TrainingFlow 的确认与冻结流程。验证入口为 `tests.test_duel_v2_device_performance_gate`。

三步 continuation 可显式选择 `DeviceTeacherPipeline(..., raw_teacher=True)`：内部 `device_teacher_actions.py`只调用正式冻结 `rule_decision`，直接解码完整原动作列表和原选中索引，压缩动作图后接入既有 simulate_action 绑定，不重复生成公开观察或策略候选编码。原动作中的 concede、顺序、未知字段和别名保持；容量按现行 800 个策略动作加一个原始 concede 有界核验，超限或索引非法明确失败。空观察／候选字段仅用于内部步进，不能送入根策略、模型或信息哈希。外层完整策略候选、32 模拟、cap16 和三步 horizon 保持不变。验证入口为 `tests.test_duel_v2_device_teacher_actions` 和流水线 legacy／raw 两种模式；Windows 同源夹具通过 `NTE_DEVICE_TEACHER_ACTIONS_PROGRAM` 加载，真实规则另做整批后的完整状态 oracle。此优化不代表吞吐已经达标。

`vm_compiled_blocks.py`可将冻结 VmImage 中有界选择的直线基本块生成 CUDA helper，并通过 `CudaVm(..., compiled_blocks={...})` 显式接入；默认 `None` 保持原发射源码不变。生成器内联冻结操作数，复用现有 GPU 栈、数值和容器原语，在调用、跳转、返回、生成器、属性 getter、上下文及异常边界分块，不另写游戏规则。helper 在原 execute 的异常处理之后、普通步数递增与取指之前执行；逐指令步数、周期 GPU 截止、局部／栈界限和失败 next-PC 协议保持一致。未选择或不支持的区域继续使用 GPU VM，不能回退 CPU。代码量、函数、基本块、操作数和前缀范围都有显式编译预算，输出标明覆盖范围和身份；这是减少解释分发的第一层编译路径，不等于完成类型化状态／规则全向量化，也不自动证明高吞吐。

验证入口为 `tests.test_duel_v2_vm_compiled_blocks`，Windows 使用同源 CPython 3.10 导出的 `NTE_VM_COMPILED_BLOCKS_PROGRAM`。实际 CUDA 回归比较编译／原 VM 的输出、共享引用、逐指令步数上限、周期截止、溢出及错误函数／PC／步数；真实游戏规则另做整批后的完整状态 oracle。所有提速与 1600 根训练能力仍由分离编译／预热的完整搜索、终局和重放实测确定，不能以生成源码或少量编译块通过代替。

`vm_constant_lookup.py`为冻结镜像中不可变、全为字符串键的字典生成排序索引，通过 `CudaVm(..., constant_lookup=True)` 显式启用，默认关闭保持原堆源码。仅对负常量引用的精确命中返回原插入顺序位置；混合类型键、动态字符串、可变字典、未命中及不满足索引条件的对象继续使用原 GPU `key_equal` 查找。索引不缓存可变状态，也不允许写入冻结字典；重复字符串键按原首次位置处理。镜像验证和原字典类型／引用检查先于快速命中。验证入口 `tests.test_duel_v2_vm_constant_lookup` 比较 CUDA 两种模式的全局变量、类属性、动态字符串、混合数值键及可变字典删除／重插后的输出和状态；Windows 同源 CPython 3.10 夹具通过 `NTE_VM_CONSTANT_LOOKUP_PROGRAM` 加载。真实规则还须做整个 GPU 批次后的完整状态对拍；索引生成、测试通过或内核资源计数均不代替完整搜索吞吐和 1600 根验收。

编译配置须与吞吐证据一起冻结。`NTE_CUDA_FAST_COMPILE=max` 是缩短编译时间的诊断配置，会关闭部分运行时优化；不得据此默认判断正常优化后的稳态吞吐。性能比较应保持规则镜像、根状态、完整搜索预算及严格浮点选项一致，单列编译／预热和设备执行耗时，并记录实际配置。恢复正常优化也须有独立编译截止，不能无限等待编译或把配置变化当成真实并发已通过。

`vm_frame_init.py`通过 `CudaVm(..., compact_frame_init=True)` 显式减少每次 Python 调用帧初始化的写入。保留全部 header 和函数声明的局部变量初始化为 MISSING，在绑定参数前设置有效 cell 数为零；操作数栈由有效长度 `f[2]` 保护，闭包 cell 在构造时逐一写入并由 `f[11]` 保护，未访问的保留槽无需清零。默认关闭保持原初始化源码；不改变函数绑定、冻结指令、步数或错误定位。验证入口 `tests.test_duel_v2_vm_frame_init` 覆盖默认／关键字／变长参数、闭包、有限帧池下的生成器回收和重复调用后未绑定变量的准确错误，Windows 同源夹具使用 `NTE_VM_FRAME_INIT_PROGRAM`。组件正确性仍须配合真实整批状态对拍和完整搜索实测，不能推断 1600 根吞吐已经达标。

换牌采样可以使用单独冻结的内部 sample_world_transition_packet 入口：gpu_sample_transition.py 保留 canonical seven_determinize、原 RNG、代理 playing 状态及原 phase／active_side 恢复，返回 (world, 0.0/1.0) 包。它不是公开 sample_world 替代接口；seven_determinize 内部的信息断言保持原样，只有外层换牌的两次 dense 编码比较移至设备编码器。新入口须重新冻结并核验源码身份，不能从旧镜像继承资格。

device_split_sampler.py 的 DeviceTransitionSampleCalls 只专门化采样的调用与包解码，其余调用沿用既有 checked 图租约流程。原动作录制调用可通过静态多继承组合，仍须验证物理 apply_action 录制钩子。原始包的公共 capture 会明确拒绝；DeviceValidatedSampler 对要求检查的每个根调用已核验的 GPU 公开观察／完整候选编码，并由 device_decision_equality.py 做逐元素精确 float32 比较、合法长度／前缀／编码错误检查，保留 -0 与 +0 相等语义，不使用 epsilon 或仅凭哈希。完整候选按静态行片段编码，CPU 仅用于显式单元 oracle，生产路径拒绝 CPU。错误保持在设备向量中；检查后的动作图按 owner／generation 回收，被拒绝的新世界先回收再清除 active，避免搜索只回收 active 状态时泄漏。原输入状态不释放。

验证入口为 tests.test_duel_v2_device_decision_equality 和 tests.test_duel_v2_gpu_sample_transition；实际 CUDA 还须证明所有换牌检查标志执行、未校验入口被拒绝、观察／候选不一致会失败且临时状态／动作租约回收，以及整个 GPU 批次后的 canonical sample_world／三步教师完整状态对拍。通过后仍要进行完整 32 模拟、cap16、horizon3、真实 1600 根、终局和原动作重放验收；不能用包解码或少量根对拍宣布训练和互搏已完成。

搜索中的规则失败必须保留原错误。DeviceGumbelSearch 在决策返回失败后停止该行的信息键查找，哈希仅处理规则／推理均未失败且决策有效的行。超时产生的空候选不能再派生 KEY_ERROR、伪装成根信息集不一致。tests.test_duel_v2_search_failed_decision 注入根遍历阶段的规则超时，检查 CPU 显式 FSM oracle 与实际 CUDA 都保留 RULE_ERROR、不哈希空候选、不增加 KEY_ERROR；该错误回归不构成搜索吞吐或真实规则容量证据。

vm_call_boundaries.py 可通过 CudaVm(..., device_call_boundaries=True) 把生成的 call、new_frame 和 execute 三个大型设备函数改为显式 CUDA 调用边界。变换仅将其声明／定义的 inline 修饰替换为 __noinline__，跳过注释和字符串，函数体、参数、冻结指令、步数／截止和设备栈预算均保持原样；默认关闭，生成源码不变。该选项用于检验编译内联和寄存器／局部内存开销，不是规则向量化或提速保证。tests.test_duel_v2_vm_call_boundaries 比较实际 CUDA 的嵌套调用、默认／变长参数、闭包、有限帧池的生成器回收和逐指令错误状态，Windows 使用同源 NTE_VM_CALL_BOUNDARIES_PROGRAM 夹具。还须比较相同真实批次、实际编译配置和完整搜索吞吐，不能由修饰符或内核资源计数推断 1600 根验收。

gpu_teacher_preview_parts.py 提供尚待 GPU 批量调度接入的教师拆分入口：teacher_prepare 获取原公开观察、完整原合法顺序和全部战斗预览在公开列表中的索引；teacher_preview 使用原 _without_patches／prepared_preview_sandbox／rule_battle_preview 独立计算每个反事实；teacher_finish 检查完整合法顺序、预览索引顺序与数量，再调用原 VisibleEngineRulePolicy。缺失、重复、重排或越界的预览不能作为完整教师决策接受，mulligan／choose 的原早返回分支保持。compose_teacher_rule_decision 仅供 CPU 单元 oracle，不是生产回退。tests.test_duel_v2_gpu_teacher_preview_parts 验证合成轨迹中的实际战斗预览、完整动作／选中索引和输入不变性；冻结导出仍需单独核对入口能力，uuid.uuid4 只允许既有显式 standalone-entity guard，不能假定可执行。当前没有 GPU 候选预览 fanout 驱动，没有完整搜索或 1600 根资格；未来接入须同时验收队列容量、全部候选完成、租约、真实 CUDA 对拍和吞吐。

`vm_library_boundaries.py` 提供另一个默认关闭的编译选项 `CudaVm(..., library_call_boundaries=True)`，为固定十个大型标准库／对象 helper 的声明及定义设置 `__noinline__`。它与三函数调用边界独立，仍只改变修饰符，保留函数体和错误协议；用于测量设备代码展开量、编译／加载成本和执行耗时，不代表类型化规则向量化，也不能保证提速。`tests.test_duel_v2_vm_library_boundaries_cuda` 使用同源 CPython 3.10 导出的 `NTE_VM_LIBRARY_BOUNDARIES_PROGRAM` 比较实际 CUDA 的排序／最值、签名、生成器回收及步数错误；真实批次另核对采样信息集、三步教师和完整状态。未经这些验证及完整搜索吞吐验收，不放行训练。

实验性采样拆分保留 `seven_sampler.determinize` 的默认完整检查，私有 `_determinize_transition` 只提供同一生成逻辑与原始公开视图，不能直接作为已验收采样器。`gpu_sample_guard_views.py` 的内部包携带生成世界及两组有序公开视图；`device_view_sampler.py` 在接纳世界槽位前由现有 GPU 编码器逐值核对完整 float32 观察／候选。普通行动和选择根保留原来的前置动作顺序；换牌根另外保留转换为 playing 的内部校验及恢复原 phase／active_side 后的外部校验。任何编码、非有限值、数量、掩码或顺序错误均拒绝该世界，未校验包不提供公开 capture 接口。此路径须冻结新 IR、真实批次对拍和故障注入验收；不能继承旧 IR 的并发／性能资格，也不允许主机逐根规则回退。

`device_int32_presentation.py` 在收集器层包装既有公开计划，按原生解析器精度保存候选种类／卡牌／选牌／牌序索引为 int32，数量和租约 ID 保留 int64，完整候选宽度及 float32 特征不变。每次转换先在设备上检查 int32 范围，越界拒绝该行；不是截断候选或近似编码。根快照、当前决策和教师计划使用一致存储类型，既有主规则／投影源码保持不变。`tests.test_duel_v2_device_int32_presentation` 对所有七队开局和换牌视图比较原／压缩存储的完整有序编码、掩码和数量，并检查越界拒绝；实际 1600 根搜索分配／捕获／执行须重新实测，不能由单数组节省量推断总峰值或并发验收。

`device_stable_page_pool.py` 在收集器专用分页分配器中使用固定形状的设备前缀与 scatter，生成与原 `argsort` 完全相同的“空闲页升序、已用页升序”排列，减少排序临时工作区。页面接纳函数除排序步骤外保留原实现；所有权、代次、重复槽位、容量原子拒绝、扩容及释放检查不变。`device_stable_graph_store.py` 只为新建收集器实例的三个分页池绑定该接纳方法，原分页／图存储模块和其他实例不变。验证复用完整分页与图存储生命周期测试，并比较实际 CUDA 排列与捕获重放；仍须实际 1600 根重新测量总峰值和完整 Gumbel 完成数，不能从排序原语通过推断容量或吞吐验收。

`DeviceGumbelSearch(..., queued_row_work=True)` 使用 `device_tensor_row_queue.py` 和 `device_search_row_work.py` 的设备内固定大小行循环，替代策略推理、完整有序信息键及 WDL 评估按根数展开的 Python 捕获片段。每个静态调用点拥有独立队列、错误状态、截止和已预热的原生流；尾行写入额外哨兵槽，空活跃集合跳过循环。模型、float32 编码、SHA 字节流、错误优先级及 WDL 差值计算复用原链路，默认关闭，GPU 数值对拍须通过后才接入真实负载。此选项减少捕获节点重复，不减少真实根、完整候选或搜索模拟数。

`borrow_root_candidates=True` 另为专用根候选缓冲提供显式借用。只有 `DeviceFrozenRules` 能在捕获时证明根是独立的首个决策计划、子节点计划缓冲不同，才允许借用根候选数值；根的状态／动作句柄等其他字段仍复制，当前决策候选仍有独立可写快照。缓冲只在下一次根初始化时更新，不得在旧根结果消费前重复初始化。缺少专用绑定、复用同一缓冲或错误捕获位置均拒绝；默认保留完整深拷贝。两项优化仍需重新验证真实终局、重放、1600 完整搜索和训练恢复，不能由静态绑定或数值原语通过代替。

`device_masked_row_copy.py` 为完整连续候选缓冲提供 GPU 行掩码逐字节更新，直接写入已有目标，避免 `torch.where` 为大字段生成整份临时数组。复制保留 int32／int64／float32／float64／bool 的位模式、NaN 和负零；未选行保持原样。模块在捕获前按设备准备，克隆复用模块；非连续缓冲、非连续掩码或不同偏移的重叠存储保留原 GPU 快照路径，CPU 数值 oracle 不变。需实际 CUDA 全字段、掩码变化与捕获重放验证，并重测真实 1600 负载的峰值及完成数，不能由临时数组消失推断搜索完成。
