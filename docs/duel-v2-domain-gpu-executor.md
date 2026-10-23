# V2 直接领域 GPU 执行器

## 实现边界

`rl/batched_duel/domain_soa_execution.py` 将具名领域状态列和直接 C++ 规则组合为 CUDA 执行器。卡牌、公开角色 hooks、战斗、回合、选择、响应和延后动作由对应 `domain_soa_*_source.py` 与 helpers 实现；不解释规则字节码，不在设备执行失败时调用主机规则。完整物理内核的领域函数与查询函数采用 CUDA 显式非内联调用，避免强制内联造成编译膨胀；物理执行与匿名预览在准备阶段配置至少 65536 字节设备调用栈，捕获后不改变栈限制。内容仅限 schema 明确支持的公开角色与机制；未支持内容、非法状态及容量溢出必须返回错误。

`DomainSoAExecution` 按设备合法动作表选择物理动作，保留卡牌实例、实体身份、选择顺序、暂停出击参数和父操作快照。动作失败时回滚本行动的状态变更。动作边界回收不再引用的物理卡槽和字符串，不删除仍被操作快照、选择或延后动作引用的实体。

持续伤害的保存与重建保留插入顺序，因为逐种结算会影响后续随机流和效果触发。效果保留具名层级、条件、来源、期限、消耗和叠加语义。空队列存在性、显式空值与缺失字段按正式状态契约区分。

## 搜索接线

`rl/batched_search/domain_soa_rule_adapter.py` 实现原设备规则协议，使用具名状态池及独立动作租约；校验状态所有者、代次、版本及动作描述，拒绝过期或跨状态动作。观察为完整 3479 维，候选为完整 1576 维。紧凑候选保存描述，用共享工作区分批编码、推理及哈希，不删减特征或候选。

`DomainSoASampler` 的宽种子接口使用 5 个 unsigned 64-bit limb，保留全部 320 位；旧 63-bit 接口单独保留。默认启用完整观察／候选信息一致性检查，每次模拟重新采样，保护已揭示信息、有效牌面和历史操作引用。搜索适配器按正式 `information_search.sample_world` 处理起手换牌：在行动阶段代理上采样，恢复原阶段与行动方，再核验起手观察与全部候选；底层采样器的行动／选择阶段契约保持独立。

真实教师预览先匿名化隐藏区域，再调用同一直接物理规则，提取正式排序所需的攻击、次数、花伤及反击。预览复用两份私有状态，不按全部候选常驻复制完整世界。

`domain_soa_h3.py` 提供三步真实教师续走和独立辅助价值模型；原规则适配器也向原搜索驱动提供同一教师续走协议。终局按原观察方返回胜／负／平，非终局使用独立辅助价值，不借用主价值头。三步共享父搜索的设备单调截止，叶子不能重置预算，超时返回未完成及错误。

`make_gumbel_search` 固定 32 次模拟、最多 16 个 Gumbel 候选、H3、最大深度 10、`per_simulation`、`mctx_default`。它是显式未验收入口，不自动成为训练后端。

## 本地诊断与设备验收

CPU 编译器 oracle 必须显式传入 `device='cpu', allow_cpu_oracle=True, allow_unqualified=True`。它编译同一份领域源，用于与正式引擎逐字段比较；不构成 GPU、400 并发或训练资格。

可运行以下检查：

```bash
python -m unittest tests.test_duel_v2_domain_soa_execution_semantics tests.test_duel_v2_domain_soa_rule_previews tests.test_duel_v2_domain_soa_h3 tests.test_duel_v2_domain_soa_deferred_payload -v
python -m unittest tests.test_duel_v2_domain_soa_rule_adapter tests.test_duel_v2_domain_soa_search32 tests.test_duel_v2_domain_soa_wide_sampler -v
python -m unittest tests.test_duel_v2_domain_soa_execution_cuda tests.test_duel_v2_domain_soa_search_cuda -v
```

最后一项要求实际 CUDA，覆盖同图从出牌进入选择再继续的物理规则重放，以及逐卡／终结／自然终局、真实 H3 和起手采样对拍；无 CUDA 时必须单独记为跳过。CPU 对拍、源代码编译、构造成功、单次搜索及跳过测试均不能替代设备验收。

完整资格还要求实际 CUDA 编译、图重放、自然终局、400 并发显存及完整局吞吐验证，并保留完整局／决策／模拟／错误／未完成前缀的真实分母。实现的资格标记保持 False，构造需要显式 `allow_unqualified=True`；未通过这些检查时不得接入正式训练。旧训练截止和部署权限不会因实现接线而自动改变。

单轮测试、速度、日志和模型只放在 `artifacts/rl-evals/<run-id>/`；本页只描述实现与验证方式。
