# 离线批量搜索后端

## 目标与边界

目标是让大量独立对局常驻数组存储，按搜索根批量推进，最终验证 1,600 个真实对局的容量。真实局数、活跃搜索根数、在途假想世界数、节点池容量和操作系统工作进程数是不同指标。

后端位于离线 RL 层。正式 Python 引擎继续作为规则对拍 oracle；网站不导入数组 arena、Torch、CUDA 或新的批量调度器。数组存储能够分配 1,600 个槽位，不表示已经实现 1,600 局规则结算、搜索或 GPU 并发。

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

### 每根预算账本

`rl/batched_search/budget.py` 的 `SearchLimits` 分别记录决策墙钟、总模拟量、公开证明上限、Gumbel 根候选上限、短终局探查和树深；默认分别为 3 秒、32、8、16、3、10。短终局探查的 3 步不能解释为整个搜索树深度。

`DecisionBudget` 在根进入队列时开始单调时钟计时。公开证明和树模拟共用总额度；预约包含排队工作，取消不返还额度。账本分开统计预约、实际开始、完成、取消和丢弃。

每根只允许一个未结束模拟，保留原搜索的逐次访问更新依赖；并行维度来自不同根。公开证明阶段结束后不能回到证明阶段。工作票据绑定预算实例、对局、根代次和序号；跨根、伪造字段、重复开始或重复完成明确拒绝。

调用者必须在 GPU 事件真实完成后才调用 `complete()`，入队或 kernel 提交不能算完成。取消或到期后的结果返回不采纳，不增加完成计数。该组件由一个调度线程拥有，不提供跨线程原子计数，也不保证 GPU 硬实时截止。

### 搜索身份

预算组件不选择动作，也不决定 Q 尺度。批量规划器必须绑定实际模型、规则、具名观察/候选和搜索版本：当前 recovery 高级策略使用 natural-WDL；旧冻结评估的 legacy completed-Q 仅用于明确标记的复现。不能因为预算相同而混称同一算法。

完整合法动作及顺序保留；16 个 Gumbel 根候选不是合法动作容量。每次模拟继续遵守现有公开一致性采样和种子推导，行动方可见输入不得包含实体编号、对方暗牌身份或 RNG。

## 后续接入顺序

1. 在现有无损图 codec 基础上补齐五队共享规则执行、合法动作和兼容现有权重的具名编码，先 CPU native 对拍，再 CUDA 对拍。
2. 将真实设备规则 resolver 与协作调度、节点池、工作区、批量模型推理和每根访问更新接通；GPU 结果须绑定根代次并拒绝迟到结果。
3. 完整检验模型数值身份、固定观察者叶子价值、采样种子与原搜索一致性，拒绝把仅有源码导出/统计 kernel 的原型标为规则后端。
4. 验证真实取消、异常和记录队列容量；完整回放从真实动作重放核验，假想分支不进入公开录像。
5. 以独立新种子依次测 64、256、512、1,600 个对局，报告完成局/小时、每根实际模拟量、部分结果、回退、p50/p95/p99/max 延迟及 CPU/GPU 峰值。仅分配槽位或启动 kernel 不算对局容量通过。

完整矩阵、训练和模型采用继续遵循训练手册与完整报告规范；单轮资料只进入 artifacts。已由用户停止的任务不因实现新后端而自动恢复。

## 验证入口

```bash
.venv/bin/python -m unittest tests.test_duel_v2_batch_arena tests.test_duel_v2_batch_budget -v
.venv/bin/python -m unittest tests.test_duel_v2_batch_codec tests.test_duel_v2_batch_device_state tests.test_duel_v2_batch_cuda_runtime tests.test_duel_v2_batch_random -v
.venv/bin/python -m unittest tests.test_duel_v2_batch_inference tests.test_duel_v2_batch_nodes tests.test_duel_v2_batch_traversal tests.test_duel_v2_batch_cooperative -v
```

没有 CUDA 的主机必须明确记录设备用例跳过。`scripts/validate_duel_v2_batch_gpu.py` 默认只输出配置，只有显式 `--run` 才执行合成节点统计验证；其 `game_executions=0`，不能作为真实对局容量或强度证据。上述用例和原语对拍也不能替代五队完整规则、真实端到端 GPU 性能或策略强度验收。
