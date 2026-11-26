# 七队原生数组规则后端

这是离线 RL 的同源原生编译分支。规则函数、卡效、回合和选择继续以正式 Python 引擎为真源；不维护第二套简化卡效，不改变网站或模型权重。该分支的代码、CPU 同源验收和实际 CUDA 验收分开判断。

## 计算结构

`native_array_program.capture_native_array_program()` 冻结当前17名公开角色及其卡牌、初始化、合法动作、公开投影、物理／假想推进和信息集采样。冻结器按函数身份替换数值等价的整数生命上限取整，以及已持有独立工作副本时的推进入口；不修改生产模块的 globals、卡表或引擎函数。

`native_array_codegen` 把冻结程序的全部指令位置编译为分片 C++ 控制流。栈操作直接生成原生语句，复杂操作调用编译期确定的操作实现；调用、闭包、生成器、异常、步数和截止仍保留原语义。运行时不读取指令表，也不回退到字节码解释器。未知操作在准备阶段拒绝，未知控制位置按行失败。

状态仍采用有界 node／edge／payload 数组，保留实体、区域顺序、内部别名和完整状态，不把复杂机制塞进不透明 JSON。**这是完整规则控制流的原生编译与数组执行，不代表已经把全部领域字段改成专用 SoA，或全部规则改成 Torch 表达式。** 观察／候选编码、模型推理、真红先验、树统计及阶段调度继续使用现有批量数值路径。后续可依据实际设备计时继续专门化热点字段和临时存储。

完整领域字段的专用SoA另见[数组搜索执行路径](duel-v2-array-search.md)：`native_soa_*` 复用这里的正式规则冻结与原生控制流，但游戏字段直接存入专用记录列，旧图后端只作独立对照。两个后端的验证与CUDA资格不互相继承。

`CudaNativeArray` 复用已有核验过的数组、租约、参数绑定和压缩 ABI，替换执行代码；设备只保留一个不可使用的指令占位向量，不上传指令流。`NativeArrayCPU` 用同一生成源码加有限的 CUDA 关键字／时钟兼容层编译成本机动态库，支持 Windows 离线时检查真实原生执行。CPU 的显式进出数组、编解码和主机调度是验证边界，不计为GPU驻留吞吐。

本机库的每次构造使用独立文件名，避免 `dlopen` 复用另一份规则的旧机器码；源码摘要与实际编译文件可随运行产物保存。输入引用必须为精确 int32 数组，错误引用和越界容量拒绝；不能通过数值转换截断动作或引用。编译只发生在显式构造执行器时，网站导入不编译、不启动训练。

## 真红与搜索

新冻结入口默认不包含 `teacher_continue`、规则对手教师或旧24计划真红指导。`DeviceNativeArrayRules` 不分配教师站点；错误地请求教师延续／规则对手会拒绝。

```python
executor = CudaNativeArray(image, limits=limits, device='cuda')
calls = NativeArrayRuleCalls(executor, physical_pool, raw_tape=raw, **call_options)
rules = DeviceNativeArrayRules(calls, presentation, action_store=action_pool)
search = create_native_array_search(
    rules, inference, auxiliary, statistics, keys,
    rows=rows, random=random, max_depth=10,
)
loop = DeviceGameLoop(rules, search, sink, seed_provider, root_guide=None, **game_options)
```

工厂固定 Gumbel32、cap16、network-WDL／h0、冻结策略对手及无模拟真红根先验。真红只在搜索前调整有界 logits，完整策略与实际动作由搜索生成；关闭先验须显式声明消融。固定观察者价值、真实终局、原始动作及完整合法候选继续沿用现行契约。

世界采样保留正式原始／生成牌分配、CPython 随机过程及320位种子。采样世界必须通过两个完整、有序的观察／候选编码检查后才发布，不能以原生编译代替信息边界检查。`NativeArrayRuleCalls` 继承双检查采样和原始物理动作记录，`DeviceNativeArrayRules` 绑定相应的设备内检查队列。

旧 `CudaVm` 与已冻结任务保持原解释器身份；不得把它换个模式名冒充原生执行，也不以旧任务的设备回执批准新生成代码。正式训练的闭环设备与模型质量闸门仍保留，CPU通过不自动放行。

## 本机验证

```bash
.venv/bin/python -m unittest tests.test_duel_v2_native_array_codegen tests.test_duel_v2_native_array_cpu tests.test_duel_v2_native_array_program tests.test_duel_v2_native_array_reference tests.test_duel_v2_array_rule_sources
.venv/bin/python scripts/validate_duel_v2_native_arrays.py --output artifacts/rl-evals/<run-id>/rules --games 14 --mechanism-tests
.venv/bin/python scripts/validate_duel_v2_native_search.py --output artifacts/rl-evals/<run-id>/search --models <frozen-model-dir> --teachers <compatibility-teacher-dir> --roots 8
```

完整局验证从原生初始化开始，执行真实开局换牌、轮替、出牌和自然胜负；每次提交后独立比较正式引擎完整状态、合法动作及观察／候选字节。超时或决策上限不产生终局结果。规则夹具在独立测试命名空间中把物理提交交给原生执行，正式引擎只作设置与事后 oracle；主机 mock 回调和超出公开角色范围的夹具明确列为排除。无原生提交的测试不能算原生卡效覆盖。

32搜索验证还比较同种子的完整采样世界、假想推进、固定视角WDL、Gumbel噪声、访问次数、完整策略和选择；不把单步搜索称为完整局，也不把CPU回传接口耗时当作GPU延迟。

实际 CUDA 另用 `NativeArrayCUDAQuery` 做有界整批规则对拍，再用设备内 `DeviceGameLoop` 验证完整32搜索、原始动作／信息哈希重放、自然终局及工作区峰值。CUDA 查询接口的主机编解码不得进入正式搜索热路径。设备离线时只保存待执行配置，不补造设备回执、容量、胜率或提速结论。

单轮证据、源码与机器码、原始记录和报告只保存在 `artifacts/rl-evals/<run-id>/`，不提交Git。模型学习、完整评测和发布继续遵循训练操作手册及相应用户授权。
