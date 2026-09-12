# 四人轮替 V2 强化学习框架

> 2026-09-12 补充：[Windows 兼容性检查](duel-v2-rl-windows-20260912.md) 已在 RTX 5060 Ti 16GB 上通过全部 20 项测试及 CUDA/MaskablePPO 有限推理检查。以下 2026-09-09 的未安装依赖说明为当时记录；本机环境未改变，Windows 使用独立测试副本。训练仍禁用、没有权重或训练结果。

状态：2026-09-09，框架与确定性接口检查完成，**训练禁用、无模型权重、无对局经验数据**。游戏规则以 [V2.4 手册](everness-item-chain-card-design.md) 为准，接口以 [实现契约](duel-v2-implementation-contract.md) 为准。网站继续使用规则 AI，本框架不改变线上对手。

## 文件与边界

- `app/modules/card_game/rl/backend.py`：仅对接 `engine.duel_v2` 六接口中的开局、行动方、投影、动作和结果；不依赖旧盖卡模型、HTTP 或数据库服务。
- `contracts.py`、`adapter.py`：可注入的纯内存接口与决策适配。策略仅收到观察投影和合法动作，不收到原始 state。
- `encoding.py`：固定尺寸公开局势和候选动作编码。观察 2423 维，单候选 199 维，动作容量 256；容量溢出显式报错，不删除候选。
- `policies.py`：确定性的候选优先级对手，以及已提供模型的 `predict` 接口。`FirstLegalPolicy` 只用于接口调试；引擎首候选可能为认输，不能拿它当强度基线。`RuleFrozenPolicy` 也仅是简单基线。
- `gym_compat.py`：可选 Gymnasium Env 工厂；只有调用 `make_gym_env` 才导入 Gymnasium 和 NumPy。
- `checkpoint.py`：JSON 元数据读写与兼容校验；本轮不保存、读取或反序列化模型权重。
- `scripts/train_duel_v2.py`：配置检查入口，所有训练路径硬禁用。

核心接口依赖现有网站环境即可导入；NumPy/Gymnasium/SB3 仅列在独立 `requirements-rl.txt`，本轮未安装。不要在网站启动链路中创建训练器或环境。模块导入不创建对局、不调用 reset/step/predict/learn。

## 观察与动作

引擎的 `observe(state, side)` 是唯一信息边界。己方手牌、双方角色/资源/弃牌/记录可见；对手手牌通常隐藏，公开复制保留。九原检视对手手牌及其他检视效果，仅在引擎开放的 pending choice 中编码候选。

角色固定按 `nanali/zero/jiuyuan/xun` 排列，卡牌按当前 catalog 顺序编码；元数据校验这些顺序。使用 `shape_id`；保留 last_front、各方 turn_count、疲劳、已用次数、是否曾环合、花伤加成、下次攻击加成、返回标记和负面状态。复制按实际归属浔编码，并保留原作者；复制期限和负面状态期限以对应操作者回合计数计算剩余时间。记录不要求实体卡实例 ID，结算中卡牌单独编码。

手牌最多 10 个槽位，记录 2 个，私有选择 10 个。选择、弃牌和检视都作为独立动作；起手换牌以手牌槽位的多选标记表示。普通出击/觉醒编码角色，出牌/弃一抽一编码可见实体牌，目标编码双方角色或公开牌；choose 同时保留选择下标和候选卡牌特征。

这仍是**部分可观察原型**：不推断隐藏牌，不包含完整历史记忆；只保留最近 8 条事件的类型/方向/数值，不把中文日志输入模型。新增未知事件进入 explicit unknown 槽；未知卡牌/角色/选择类型及未支持的次数键报错。数值归一化有上限，扩展卡池、资源范围、历史需求后须升级编码与测试，不能宣称完整 Markov 状态或已经找到最佳策略。

## 决策、奖励与截断

`DuelAdapter(backend, encoder, learning_side='a', opponent=明确提供的冻结策略)` 支持 a/b。构造不运行环境；显式 `reset(seed=...)` 才开局，并有界推进冻结对手至学习方。无默认对手，不自动启动自我对弈。

`reset` 返回 `Transition`；`decision()` 返回带独立 revision 的局势、动作、特征和 mask。`step(index, revision=...)` 拒绝非法编号、过期 revision 和已结束局面的操作。revision 跨 reset 不重用。返回值与传给策略的数据均隔离复制，策略不能通过修改候选改变结算动作。

胜 +1、负 -1、平局 0，非终局 0，不增加伤害/出牌等塑形奖励。一次学习方 step 可以包含有界的对手动作，奖励从学习方视角计；未实现按内部动作数调整折扣。

`max_game_actions` 默认 1000，`max_opponent_actions` 默认每次推进 100。达到任一保护限制记为 truncated、奖励 0；真正终局优先于限制。info 提供 `truncation_reason` 与 `bootstrap_allowed`。截断保留真实可见候选用于估值，但不能继续 step；真正终局为空 mask。

Gym 工厂返回真正的 `gymnasium.Env`，采用 Dict 观察 `{state, candidates}` 和 Discrete(256)，配合 MaskablePPO 的 `MultiInputPolicy`；`action_masks()` 位于环境内部。不能只把 state 向量喂给策略而丢掉候选特征。

Gym reset 若已经终局/截断或没有可执行动作，明确报错，不把全 False mask 交给策略。截断发生在对手行动窗口时无法可靠地用学习方决策价值估值，Gym step 明确报错，要求提高限制并 reset；仅学习方决策边界截断允许 bootstrap。这是框架保护，不是游戏判负。训练恢复后需使用官方 mask-aware 评估接口，不能用忽略 mask 的随机 `check_env` 代替合法动作验收。

## Checkpoint 与训练开关

元数据包含 schema/rules/design/encoder 版本、policy type、特征维度、候选容量、角色/卡牌/形态排序和 catalog SHA-256。当前仅接受 `weights=null`、`trainable=false`；`load_checkpoint` 仅验证元数据，不加载模型。后续增加权重加载前，必须先完成兼容与可信权重来源校验。catalog 哈希不替代引擎规则版本，规则语义变更时也必须更新版本。

以下命令不会创建模型或对局，也不会采样：

```bash
python3 scripts/train_duel_v2.py --help
python3 scripts/train_duel_v2.py --check
python3 scripts/train_duel_v2.py --check-config /path/to/checkpoint-metadata.json
```

不带检查参数、`--train` 或 `--enable-training` 都以非零状态退出。未来训练需要另行授权与代码改动；当前不存在开启训练的环境变量或隐藏参数。

## 已执行验证与限制

```bash
python3 -m unittest tests.test_duel_v2_rl
```

17 个测试覆盖编码尺寸、复制/期限、10 个选择、字段变化、目标/换牌、事件、记录/结算牌、隐藏数据隔离、候选溢出、非法/过期动作、终局奖励、截断与冻结对手上限、checkpoint、禁用 CLI。其中真实引擎仅执行同一局的出击和认输共 2 个明确动作，其余均使用合成观察或确定性 fake。

未安装训练依赖，因此 Gymnasium/SB3 的实际运行集成未验证；未创建任何模型、未更新权重、未采集 rollout、未运行训练或超参搜索。没有吞吐、收敛、胜率或平衡结论。候选特征矩阵较大，未来开始训练前应测量编码/引擎耗时，再决定是否使用共享候选评分网络压缩输入。

参考：[MaskablePPO 官方说明](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)。
