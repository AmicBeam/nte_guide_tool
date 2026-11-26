# 批量数组规则与搜索

本文件维护当前改造边界、接口与验证方法；单轮计时、日志和产物保存在 `artifacts/rl-evals/<run-id>/`。本入口属于离线 RL，不改变网站模型、正式游戏规则或既有冻结任务，也不自动启动训练。

## 执行结构

目标链路为：有界数组状态及动作 → 一批规则推进 → 一批公开编码／合法动作 → 策略与固定视角 WDL → 树回传。参考 Mctx 的 recurrent 扩展结构，以 Torch 数组实现；不是直接调用 JAX Mctx，也不把 Python 规则解释器的批量调度称为规则数组化。

当前已实现的组件：

- `rule_ir/lower_array.py` 在准备阶段将共享规则 IR 编译成固定的张量表达式，运行时按批处理混合行动方、活动掩码与整数状态；不逐根解释字节码。输出按行错误，拒绝整数域或输出越界，失败行和非活动行保留输入。
- `batched_search/array_recurrent.py` 的 `ArrayRuleRegistry` 显式登记动作对应的共享 IR 程序，不支持的动作拒绝。`SharedIrTransition` 将该注册表的数组结果接到 `ArrayRecurrentFn`，以固定布局数组保存状态，不分配实体图临时堆。现有共同交战／普通出击切片是组件覆盖，不是完整对局。
- `ArrayRecurrentFn` 一次处理一批推进与评价。策略必须来自各实际行动方自身可见信息，WDL 始终属于显式固定观察者；同一玩家连续行动与换回合均不按树深翻转符号。新到达的终局只给一次真实 WDL reward，并设 discount=0、value=0；已有终局为吸收态，不重复发奖。非终局 reward=0、discount=1。终局的占位动作仅供树运算，不能提交为物理动作。转移或评价错误回退本次状态并保留错误，不生成有效终局。
- `batched_duel/array_health.py` 为减少生命上限提供整数数组数值原语，保留逐次减少与逐次比例四舍五入，不能合并成一次比例缩放。完成状态、生命锁、倒地、事件和胜负由正式调度层处理；原语不单独执行这些规则。

隐藏世界的生成仍遵守可见信息边界。SoA性能分支采用每个物理决策一次采样、全部32次模拟复用的近似；历史多世界分支保持原配置，不共用旧数值回执。完整合法候选、原始动作身份、实际行动方、固定价值视角及完整策略目标继续保留。

### 决策内复用隐藏世界与根编码

`DeviceGumbelSearch(reuse_sampled_world=True, reuse_root_decision=True)` 保留独立于真实根的不可变假想世界租约。32次模拟均从它出发，分支推进创建新状态，暂存世界不因某次分支结束而释放；决策完成或失败后统一回收，下个物理决策重新采样。每路记录实际 `world_samples`，不能把32次访问写成32个独立世界。该近似不预测真实暗牌、不向网络暴露真实暗牌，但会减少隐藏世界覆盖，模型强度仍需独立互搏检验。

根观察与候选只在决策开始时生成，采样通过精确信息检查后复用于各次模拟的根；原始动作快照保留，状态租约显式重绑定到假想世界。内部局面和对手行动仍按其各自可见信息生成观察与动作，不沿用根编码。相同冻结模型的根策略／价值复用沿用已有资格检查，模型或观察不匹配时走原价值路径。

`recovery_search.search(world_sampling='per_decision')` 是显式CPU数值参考：采样一次，分支复制该世界，不改变默认历史 `per_simulation` 分支。两种协议允许输出不同访问数和策略，验收必须让设备与参考采用同一协议。性能报告包含每根实际采样数、完整32预算、分阶段设备计时及批次外正式规则和数值对拍。

## 无额外模拟的真红先验

`batched_search/array_zhenhong_prior.py:ArrayZhenhongPrior` 只读取现有七队公开观察与完整合法候选，支持密集编码及 `CompactDeviceCandidates`。索引、卡牌类型与属性表在捕获前准备；运行时不读取原始对局字典、不调用规则推进、不采样未知世界，也不执行旧24计划指导。

仅在己方有存活、未处于终结的真红且有翳时适用；选择面板及无关队伍不额外加分。评分是启发式机会提示，不保证已经实际触发：

| 项 | 默认 logit 提示 |
| --- | ---: |
| 已有延滞且可入场创生的机会 | 2.0 |
| 建立延滞的入场机会 | 1.0 |
| 尚无兽牙时使用兽牙牌 | 0.75 |
| 有翳战斗牌、存活翳及足够当前资源时，零战斗牌准备 | 1.0 |
| 尚未齐备时的 R01／Y02 过牌准备 | 0.25 |
| 已获得准备提示的动作，按估计付费扣分 | 每点0.1 |
| 起手换掉最后一份兽牙／零战斗牌核心 | 每组−1.0 |

合计位移限制在 [−2, 2]；所有权重必须有限且非负。这些是待验证的启发式参数，不是胜率、奖励或已验证最优值。反击、响应、复制异能和多步收益继续由搜索判断；不把推测的未来抽牌作为已持有资源。倾陷值的累积不等于已经倾陷，准备检查使用当前倾陷状态。

GPU 搜索显式接线：

```python
guide = ArrayZhenhongPrior(device=device)
search = DeviceGumbelSearch(
    rules, inference, auxiliary, statistics, keys,
    rows=rows, max_actions=800, simulations=32, gumbel_candidates=16,
    max_depth=10, teacher_mode='network_value', terminal_horizon=0,
    opponent_mode='policy', random=random, device=device,
    prior_guide=guide,
)
loop = DeviceGameLoop(..., root_guide=None)
```

`prior_guide` 在根的网络 logits 后、Gumbel分配前加入位移；不进入内部节点、对手策略或价值分支，不在搜索后覆盖动作。结果保留 `prior_shifts` 与 `prior_version`，冻结配置另存完整 `ZhenhongPriorWeights`。指导异常拒绝该搜索行；同时配置旧 `root_guide` 会被拒绝，避免两套指导叠加。默认 `prior_guide=None` 保持历史分支；新数组分支必须显式按上述参数启用，不能复用旧冻结回执。

`recovery_search.search(root_prior=guide)` 是同一评分的显式 CPU 参考，`score_numpy` 拒绝 CUDA 对象，不构成 GPU 热路径回传。它只改变根先验，保留原模型 `raw_choice` 与 WDL；零位移须复现原搜索结果。

## 完整领域 SoA 执行路径

`domain_soa_records.py` 声明完整公开规则的封闭类型，`record_soa.py` 编译成逐字段的整数、浮点、索引和存在性列，物理地址为 `[column, row]`。玩家、异能者、实体牌、效果、牌区、操作续步及事件分别有专用记录池；列表、映射和元组只能引用声明的池，不能保存任意对象树或JSON。每个字段区分缺失、null、布尔和零，保存实体类、字段插入顺序和共享引用。

`native_soa_source.py` 将正式规则的完整原生控制流绑定到这些列；`vm_soa_views.py` 的对象视图只有类型／记录引用头，没有游戏字段的通用边表副本。规则读取、修改、洗牌、深复制、初始化和采样直接访问权威列。冻结字段名通过 `vm_soa_constant_fields.py` 直接定位字段。`record_soa_compact.py` 按类型回收不可达记录、重排引用及字符串，CPU参考与CUDA实现使用同一算法。

语言栈、闭包、临时列表、临时标量装箱和公开投影仍有有界工作区；这不等于已经消除了全部临时对象，也不构成性能结论。公开投影可导出到现有数值编码器的临时图接口，导出器拒绝完整Game，规则不能通过该接口退回通用状态图。旧 `native_array_*` 后端保留为独立对照，不是SoA的逐步回退。

`NativeSoACPU` 是显式本机语义参考，接持久SoA状态；`NativeSoAReference` 的完整搜索不逐步重打包Game。`CudaSoARules` 与 `DeviceSoARules` 接CUDA状态池、完整候选、两组精确采样检查、原始物理动作记录及现有设备Gumbel驱动。规则调用顺序复用列、计算栈和回收工作区。搜索固定32／cap16、network-WDL／h0和策略对手，真红仅作数组根先验。

SoA 的 CUDA 编译使用当前 GPU 的 `sm_*` 目标并获取 CUBIN，驱动直接加载机器码；共享 `CudaModule` 的状态回收、投影等辅助内核也默认使用 CUBIN 和 `min` 编译档，显式 `binary_format='ptx'` 保留对照接口。不支持 CUBIN 时明确拒绝，不静默退回巨型 PTX 的驱动即时编译。固定指令分片、共享操作及四个大型语言库入口保留调用边界，操作内部保留内联特化；基础栈指令直接执行既有语义，帧只初始化可寻址的局部变量与头部。执行器要求至少 128 KiB 设备调用栈并核验驱动实际值，默认每个线程块执行一行，避免少量并发共享一个分歧严重的执行束；其他块大小须显式配置并独立验收。报告记录实际 `rule_threads_per_block`；SoA 不编译未使用的通用图执行入口。CPU 语义参考编译相同的函数体。编译配置保存格式、二进制大小、编译及加载耗时，必须与规则执行时间分开报告。`NTE_CUDA_FAST_COMPILE` 与 `NTE_CUDA_JIT_OPT_LEVEL` 仍须显式配置；CUBIN 下后者传给编译器的汇编阶段，快速编译档通常不能与显式汇编等级混用，冲突在编译前拒绝；显式设置汇编等级而未指定快速编译档时使用 `0`。`mid` 未通过运行时值与冻结常量的标量比较验收，CUBIN 路径直接拒绝。`max` 加 `0` 只作为慢速诊断配置，`min` 不设置显式汇编等级。不能把诊断配置的结果当成正式性能数据。采样包的静态路径索引在捕获前上传，捕获中只复用设备索引。

CUBIN 默认缓存到当前源码根的 `artifacts/rl-evals/cuda-cache/`。`NTE_CUDA_BINARY_CACHE_DIR` 可指定独立目录，设为空字符串则禁用。缓存键绑定生成源码、完整编译选项、GPU 架构、NVRTC／驱动接口版本及内核参数契约；命中时重核输入和二进制哈希。失效或损坏的缓存重新编译，目录不可写时使用本次内存编译结果；不加载未经核验的旧机器码。报告分别记录缓存命中、编译、缓存读写和驱动加载耗时及机器码哈希，不能把冷启动算作每步规则时间。设备闭环计时的默认准备预算为 600 秒，实际规则执行与事后重放预算继续独立。

软件对拍、实际CUDA对拍、设备闭环32搜索、自然终局和容量资格分别验收。Windows离线不能授予CUDA、提速、模型胜率或训练吞吐资格；也不能用CPU时间推算GPU完整局时间。

## 早期阶段组件

专用领域SoA另由 `batched_duel/domain_soa_schema.py` 声明字段、类型、存在性与容量；`domain_soa_rules.py`、`domain_soa_effects.py` 和 `domain_soa_random.py` 提供直接列计算的算术、效果匹配与CPython随机语义洗牌。`domain_soa_transition.py` 是预先解析条件下的战斗／时钟切片与有界队列，不包含完整角色回调、卡效、合法动作、观察或信息集采样；它与完整状态编解码的执行契约仍未接通，`schema_bound` 与完整规则资格保持false。不能把这些部件称为七队完整SoA规则或完整设备搜索。

`domain_soa_codec.py` 当前仅支持换牌阶段的受限状态，显式排除展示事件；进入正常行动、出现效果、选择或操作续步时拒绝。目录绑定、整数范围、缺失／null／零、实体类型、共享引用和全部已接收字段均需核验，不能只验证生命列。`validate_duel_v2_domain_soa.py` 使用正式引擎生成输入，仅检查该编解码边界；其正常终局计数仅属于输入生成器，SoA对局计数固定为0，不算新规则执行证据。

本次逐字段布局、批量规则、8路32搜索、自然终局、逐级容量及官方Mctx结果／性能对照，按[1016验收协议](duel-v2-soa-1016-acceptance.md)分别记录。官方对照入口 `scripts/validate_duel_v2_mctx_reference.py` 只在显式运行时加载JAX/Mctx；预录数值树只核搜索语义，不替代真实规则、信息集搜索或GPU吞吐。

早期六人原生训练后端仍不能冒充七队支持。阶段算术组件、旧通用图原生后端与完整领域SoA是不同实现边界；不能共用未经验证的资格标记。容量不足、未知机制或不支持字段须拒绝，不能删除效果、截断候选或悄悄回到CPU规则。

## 验证

```bash
.venv/bin/python -m unittest tests.test_duel_v2_record_soa tests.test_duel_v2_domain_soa_records tests.test_duel_v2_native_soa_cpu tests.test_duel_v2_record_soa_compact tests.test_duel_v2_device_soa_storage tests.test_duel_v2_device_soa_rules
.venv/bin/python scripts/validate_duel_v2_native_soa.py
# 显式执行软件验收；模型及历史教师须已冻结且身份匹配。
.venv/bin/python scripts/validate_duel_v2_native_soa.py --run --program FROZEN_PROGRAM --models MODELS --teachers TEACHERS --games 14 --roots 8 --mechanism-tests --output artifacts/rl-evals/NEW_RUN/software
.venv/bin/python -m unittest tests.test_duel_v2_array_rules tests.test_duel_v2_array_health tests.test_duel_v2_array_recurrent tests.test_duel_v2_array_prior
.venv/bin/python -m unittest tests.test_duel_v2_device_search tests.test_duel_v2_device_games tests.test_duel_v2_device_network_guide tests.test_duel_v2_device_policy_opponent
.venv/bin/python scripts/benchmark_duel_v2_arrays.py --device cpu --rows 8 32 64 --output artifacts/rl-evals/<run-id>/cpu-components
```

Windows 使用已配对隔离环境运行相同测试，并把计时脚本的 `--device` 设为 `cuda`。只有实际 CUDA 捕获／重放和数据对拍通过才记设备通过；无CUDA时的跳过不计通过。计时脚本分开记录真实开局编码上的真红评分与共享交战IR组件，不把组件调用命名为32搜索，不推算完整对局吞吐。

先验需验证两视角换牌、可见组合准备、资源不足、倒地／终结、隐藏信息置换、密集／紧凑候选、错误与非活动尾槽。数组规则需验证混合行动方、输入不变、别名／快照写入语义、越界拒绝及正式规则对拍。完整七队规则、完整Gumbel32、自然终局与模型质量仍各自验收，组件不能自动授予任何一项资格。
