# V2 公开演出协议（2026-09-12）

日期：2026-09-12。本批由 Grok 实现专业牌桌与演出所需的后端公开协议；浏览器验收仍由主任务进行。本文不是规则真源，不表示已经上线或已经平衡，也不覆盖 [2026-09-09 验收记录](duel-v2-acceptance.md) 或旧盖卡规则。RL 接口、权重和训练状态保持只读。

规则仍以 [完整手册 V2.4](everness-item-chain-card-design.md) 为准。接口冻结见 [实现契约](duel-v2-implementation-contract.md)。

## 本批范围

- 保留原有 6 个纯规则接口和全部公开字段。
- `observe(state, side, after_seq=None)` 增加 `game.presentation`。
- `legal_actions[]` 增加 `interaction`，action 本体不变。
- `GET /api/duel-v2/state?after_seq=` 只过滤演出 events。

## presentation

`{schema_version:1, cursor, oldest_seq, events}`。

- `cursor` 是最新公开事件 seq；`oldest_seq` 是当前窗口最早 seq。
- 事件含 `seq/action_id/action_version/type/side/text`，以及规范 `actor/target`。
- `present` 由后端决定该事件要不要演出。前端必须按 `events` 数组顺序播放，不得把后面的抽牌、结束弹窗或手牌变化提前套到当前画面。
- 环合的触发事件播放艺术字；同次延滞、浊燃、黯星、覆纹、浸染的状态附着事件设置 `present=false`，仍保留日志、顺序和公开 `patch`。失谐等独立触发不被静默。2026-09-21 起新产生的事件遵守此约定，既有存档与回放事件不追改。
- 先后手为 `initiative`，起手换牌为 `mulligan`，对局结束为 `finish`。这三类都进队列；结束弹窗只在 `finish` 演出时打开。
- `resource` 是环合/能量等数值变化，仍按 seq 排队，用小飘字演出，不插队。回合开始抽牌的 seq 在 `turn` 之后。
- `action_id` 至少由对应动作 version 组成；pending 的 `choose` 是下一次 `apply_action`，使用新 version，不和出牌合成一个 ID。
- 已公开卡牌进入 play/gain/record/shape；对手普通抽牌、私有检视候选不会进入对方 presentation。
- `patch` 按角色 id 合并公开变化。从动作前公开 board 按 seq 应用本动作 patch 后，应等于动作后公开 board。
- 同时攻守同 `group_id`，连击不同组。
- 旧快照缺少演出元数据时不补卡名；首次新动作仍生成可回放 patch。起始观察不回放全部历史。

## interaction

每条合法动作单独匹配：`{kind, actor_id, target_ids}`。

- N06 每条候选的 `target_ids` 等于该条 action 的规范 `target_id`。
- 复制 N06 归属浔，排除自己；规范 ID 保留阵营。
- 弧盘指向本人；敌前排战术指向合法前排；弃牌堆回收/手牌选择返回 `choice`。

## 验证

本批自动化覆盖 presentation 单测，包括 N06 逐条目标、复制排除自己、出击/护盾生命混合伤害/弧盘/N06/X08 多击复制，以及去掉演出元数据后的首次新动作回放。浏览器验收、平衡试玩和生产发布不在本批声明范围内。

2026-09-21 环合重复艺术字修复：`tests.test_duel_v2_harmony_presentation` 与区域状态测试共 10 项通过，盈蓄重做与回放回归 40 项通过，JavaScript 演出回归 31 项通过。隔离数据库的真实页面从大厅进入牌桌、拖入角色触发覆纹，DOM 观察计数确认艺术字仅创建 1 次；历史保留触发与状态附着两条记录，刷新保留局面且不重播，浏览器无 error。本次不追改历史回放、不部署。


## 离线假想动作例外

搜索树通过内部`simulate_action`执行假想动作时，仅关闭公开棋盘快照及patch构造；事件基本字段、序号和规则结算保留。这个开关有上下文隔离，不进入对局存档，不改变普通网页动作和评估回放的默认演出。缺patch的假想事件不得作为可播放录像输出；需要录像时仍走普通`apply_action`并核对公开投影。引擎源码指纹照常变化，不为性能优化绕过模型版本校验。

召唤物离场的公开 patch 可含精确的 `{id, removed: true}` 角色删除标记。回放隐私检查仅在 `patch.sides.a/b.characters[]` 位置允许该布尔标记，继续拒绝内部移出游戏牌区 `removed`、其他形状和额外隐藏字段。验证：`tests.test_duel_v2_replay_removal`，以及真实浊燃跨队冻结评估的动作重放与公开导出。

### 2026-09-26 持续伤害与目标提示

同次持续伤害分配先按目标合并，再使用共同的 `group_id` 显示全部伤害数字；期间的触发历史沿用该组，已有显式子伤害分组保留，避免加攻等记录拆开同次分配。主动触发噩梦与回合结束触发共用结算与公开事件。已展开的指定角色／异能者动作把自己的唯一 `target_id` 作为公开交互提示，不扩展到其他候选。变更归属的公开衍生牌使用实际拥有者名称显示效果，例如安魂曲的「家族壮大」；隐藏信息边界不变。


### 2026-10-07 单次伤害上限外框

公开棋盘异能者新增 `damage_limit` 数值或 null；随真实触发、到期和倒地事件的patch同步，回放按该字段还原。当前真红限制显示蓝色盾形外框，与历史 `damage_immune` 完全免疫状态区分。伤害事件的 `amount/before/after` 已由后端按上限和护盾结算，前端不计算伤害上限。
# 2026-10-07 高级人机逐操作呈现补充

高级人机不再将整个回合的所有操作计算完才返回。一次AI请求只提交一个规则动作，牌桌依次应用该动作的公开事件patch并等待其演出完成，然后才发起下一次AI请求。玩家提交结束回合时应立即看到自己的结束与回合开始结果；后续AI出牌、出击、终结、选择分别返回并呈现。单次动作中的不可分割卡效／响应／追加攻击仍完整结算，不在规则中途开放操作。刷新根据权威状态的`room.ai_pending`继续，不从前端预测资源或AI动作。请求版本、幂等编号与事件游标继续防止重复／倒退；思考期间保留上一步已呈现牌桌并显示“对手思考中”。


### 2026-10-08 鬼郎丸受伤弃牌提示

内容回调在黑胶唱片效果实际弃牌后调用 `EffectContext.history(..., present=True)`，生成现有 `effect` 事件的可见提示；默认 `history` 仍为 `present=False`。公开 `actor` 为早雾、`target` 为被弃牌方玩家，正文说明鬼郎丸受伤与实际弃置牌名。该牌已由公开 `discard`／`remove` 事件揭示，提示不附带其他手牌。前端沿用效果 banner，实时与回放各按 seq 播放一次。普通 `discard` 仍静默，空手或未受正数伤害不生成弃牌效果提示。
