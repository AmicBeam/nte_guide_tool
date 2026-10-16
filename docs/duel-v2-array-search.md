# 批量数组规则与搜索

本文件维护当前改造边界、接口与验证方法；单轮计时、日志和产物保存在 `artifacts/rl-evals/<run-id>/`。本入口属于离线 RL，不改变网站模型、正式游戏规则或既有冻结任务，也不自动启动训练。

## 执行结构

目标链路为：有界数组状态及动作 → 一批规则推进 → 一批公开编码／合法动作 → 策略与固定视角 WDL → 树回传。参考 Mctx 的 recurrent 扩展结构，以 Torch 数组实现；不是直接调用 JAX Mctx，也不把 Python 规则解释器的批量调度称为规则数组化。

当前已实现的组件：

- `rule_ir/lower_array.py` 在准备阶段将共享规则 IR 编译成固定的张量表达式，运行时按批处理混合行动方、活动掩码与整数状态；不逐根解释字节码。输出按行错误，拒绝整数域或输出越界，失败行和非活动行保留输入。
- `batched_search/array_recurrent.py` 的 `ArrayRuleRegistry` 显式登记动作对应的共享 IR 程序，不支持的动作拒绝。`SharedIrTransition` 将该注册表的数组结果接到 `ArrayRecurrentFn`，以固定布局数组保存状态，不分配实体图临时堆。现有共同交战／普通出击切片是组件覆盖，不是完整对局。
- `ArrayRecurrentFn` 一次处理一批推进与评价。策略必须来自各实际行动方自身可见信息，WDL 始终属于显式固定观察者；同一玩家连续行动与换回合均不按树深翻转符号。新到达的终局只给一次真实 WDL reward，并设 discount=0、value=0；已有终局为吸收态，不重复发奖。非终局 reward=0、discount=1。终局的占位动作仅供树运算，不能提交为物理动作。转移或评价错误回退本次状态并保留错误，不生成有效终局。
- `batched_duel/array_health.py` 为减少生命上限提供整数数组数值原语，保留逐次减少与逐次比例四舍五入，不能合并成一次比例缩放。完成状态、生命锁、倒地、事件和胜负由正式调度层处理；原语不单独执行这些规则。

世界采样仍须按原信息集协议进行。不能把某个采样世界的私有状态缓存成所有世界共同的树状态，也不能以普通完全信息 Mctx 树替代现行信息集搜索。完整合法候选、原始动作身份、实际行动方、固定价值视角及完整策略目标继续保留。

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

## 当前尚未完成的迁移

专用领域SoA另由 `batched_duel/domain_soa_schema.py` 声明字段、类型、存在性与容量；`domain_soa_rules.py`、`domain_soa_effects.py` 和 `domain_soa_random.py` 提供直接列计算的算术、效果匹配与CPython随机语义洗牌。`domain_soa_transition.py` 是预先解析条件下的战斗／时钟切片与有界队列，不包含完整角色回调、卡效、合法动作、观察或信息集采样；它与完整状态编解码的执行契约仍未接通，`schema_bound` 与完整规则资格保持false。不能把这些部件称为七队完整SoA规则或完整设备搜索。

`domain_soa_codec.py` 当前仅支持换牌阶段的受限状态，显式排除展示事件；进入正常行动、出现效果、选择或操作续步时拒绝。目录绑定、整数范围、缺失／null／零、实体类型、共享引用和全部已接收字段均需核验，不能只验证生命列。`validate_duel_v2_domain_soa.py` 使用正式引擎生成输入，仅检查该编解码边界；其正常终局计数仅属于输入生成器，SoA对局计数固定为0，不算新规则执行证据。

本次逐字段布局、批量规则、8路32搜索、自然终局、逐级容量及官方Mctx结果／性能对照，按[1016验收协议](duel-v2-soa-1016-acceptance.md)分别记录。官方对照入口 `scripts/validate_duel_v2_mctx_reference.py` 只在显式运行时加载JAX/Mctx；预录数值树只核搜索语义，不替代真实规则、信息集搜索或GPU吞吐。

这些张量组件之外，七队完整规则现已另接[同源原生数组后端](duel-v2-native-array-rules.md)：冻结规则控制流全部预编译，运行时不读取字节码指令流，初始化、推进、合法动作、公开投影和采样均使用该原生代码。数值编码、先验与树操作继续共享本文件的批量接口。实际CUDA资格与性能仍须单独验证，不能据此宣称搜索主耗时已解决。

早期六人原生训练后端仍不能冒充七队支持。当前七队原生分支保留通用实体／牌区数组图；全部领域字段的专用SoA布局、减少临时对象及设备性能优化仍是后续工作。完整设备搜索、原始动作重放、自然终局和学习服务要按新生成源码分别验收。容量不足、未知机制或不支持字段须拒绝，不能删除效果、截断候选或悄悄回到CPU规则。

## 验证

```bash
.venv/bin/python -m unittest tests.test_duel_v2_array_rules tests.test_duel_v2_array_health tests.test_duel_v2_array_recurrent tests.test_duel_v2_array_prior
.venv/bin/python -m unittest tests.test_duel_v2_device_search tests.test_duel_v2_device_games tests.test_duel_v2_device_network_guide tests.test_duel_v2_device_policy_opponent
.venv/bin/python scripts/benchmark_duel_v2_arrays.py --device cpu --rows 8 32 64 --output artifacts/rl-evals/<run-id>/cpu-components
```

Windows 使用已配对隔离环境运行相同测试，并把计时脚本的 `--device` 设为 `cuda`。只有实际 CUDA 捕获／重放和数据对拍通过才记设备通过；无CUDA时的跳过不计通过。计时脚本分开记录真实开局编码上的真红评分与共享交战IR组件，不把组件调用命名为32搜索，不推算完整对局吞吐。

先验需验证两视角换牌、可见组合准备、资源不足、倒地／终结、隐藏信息置换、密集／紧凑候选、错误与非活动尾槽。数组规则需验证混合行动方、输入不变、别名／快照写入语义、越界拒绝及正式规则对拍。完整七队规则、完整Gumbel32、自然终局与模型质量仍各自验收，组件不能自动授予任何一项资格。
