# 四人轮替 V2 实现契约

状态：2026-09-09 V2 引擎与网页已接入，功能验收见 [验收记录](duel-v2-acceptance.md)。业务规则唯一真源为 [完整手册](everness-item-chain-card-design.md) V2.4；本文件只冻结并行实现、网页与 RL 的接口，不替换规则。**本轮不得运行强化学习训练或自我对弈训练。**

## 本轮实现分工

- 内容 worker：`app/modules/card_game/content/duel_v2/catalog.json`、`catalog.py`、`__init__.py`，`tests/test_duel_v2_catalog.py`。不要编辑 effects.py。
- 引擎 worker：`app/modules/card_game/engine/duel_v2/`、`app/modules/card_game/content/duel_v2/effects.py`、`tests/test_duel_v2_engine.py`。不改 Web、DAO、数据库、内容 catalog。
- 页面 worker：新增 `templates/card_game/v2_*.html`、`static/js/v2_*.js`、`static/css/v2_*.css`。不修改已有旧版页面和脚本、路由或引擎。
- 主任务：路由、应用服务、账号/房间/构筑/快照适配、版本切换、端到端测试、文档状态与最终验收。
- 协作任务“评估轻量强化学习模型”：`app/modules/card_game/rl/`、`scripts/train_duel_v2.py`、`tests/test_duel_v2_rl*.py`、`docs/duel-v2-rl.md`、`requirements-rl.txt`。不修改以上所有权范围。

所有人共享仓库，不得回退他人改动。Grok worker 只在分配范围内写文件，不提交、推送、部署、访问凭据或继续委派。真实账号、数据库内容与浏览器登录状态由主任务处理，不发送给 Grok。

## 内容契约

角色 ID 固定顺序：`nanali`、`zero`、`jiuyuan`、`xun`。卡牌 ID 固定 `N01..N08`、`Z01..Z08`、`J01..J08`、`X01..X08`，与手册一一对应。

从 `app.modules.card_game.content.duel_v2` 导出：

```python
CHARACTERS: dict[str, dict]
CARDS: dict[str, dict]
STARTER_DECK: dict
get_catalog() -> dict
validate_deck(deck: dict) -> dict  # 不合法抛 ValueError，合法返回规范化新 dict
```

角色字段：`id, name, attribute`（中文光/灵）, `attack, max_hp, avatar, portrait, passive, awakened_passive`。图片复用共享角色头像和立绘；zero 使用鉴定师图片。

卡牌字段：`id, character_id, name, type`（`battle/tactic/form`）, `cost, terminal, description, effect_id, source`。`cost` 一律为 1。弧盘另有 `stat`（`attack` 或 `hp`）与 `stat_value`；战斗牌另有 `shield`（护盾值，可为 0），并可另有 `attack`（无条件攻击加成，至少为 1）。浔的兑现战斗牌另有 `redeem: true`。可选 `instant`、`response`、`derived`。`derived: true` 为对局内衍生牌，展示「不可构筑」，不能进入构筑。effect_id 默认与卡牌 ID 相同。`terminal` 保留为协议字段，现行卡表一律为 false。**08 固定为专属武备**；**07 优先用官方角色 PV 名**，没有角色 PV 时用官方短片或用户指定名。零另有第 6 张「诓定解放」也是武备。source 至少含 Everness 角色 ID；`type` 为 `awaken`（01–06）、`arc`（专属武备，08）、`pv`（官方角色 PV）或 `ability`（短片或指定名），衍生牌为 `derived`。

标准构筑 JSON：`{id, name, character_ids: [4个ID], card_ids: [32个卡牌ID]}`。card_ids 是重复展开列表。validate_deck 严格验证恰好 4 名不同已实现角色、每人 8 张、总数 32、同名最多 2、没有未知或其他角色卡，不能静默截断。返回独立副本。

get_catalog 返回 `{rules_version: "duel_v2", characters: [...], cards: [...], starter_deck: {...}, starter_decks: [...], deck_rules: {size: 32, per_character: 8, max_copies: 2, character_count: 4}}`。内容层默认返回完整卡表；`GET /api/duel-v2/catalog` 按账号过滤：未加入测试白名单的账号只看到公开角色（娜娜莉、零、九原、伊洛伊）及其卡牌，以及不含测试角色的官方预组。账号首次读取 catalog 时，把当前可见的官方预组写入该玩家构筑库，之后按普通 saved_builds 处理，可删除；已完成首次赠送后不再重复发放，但账号升级为测试账号后会补发此前不可见的官方预组。旧的单套构筑会保留，并补齐尚未拥有且当前可见的预组。保存、开局或加入房间时，非测试账号不能使用含测试角色的构筑。

## 纯内存引擎契约

从 `app.modules.card_game.engine.duel_v2` 导出：

```python
new_game(seed: int = 0, decks: dict | None = None,
         first_side: str | None = None, skip_mulligan: bool = False) -> dict
acting_side(state: dict) -> str | None
legal_actions(state: dict, side: str | None = None) -> list[dict]
apply_action(state: dict, side: str, action: dict) -> dict
observe(state: dict, side: str, after_seq: int | None = None) -> dict
choose_action(state: dict, side: str) -> dict  # 非训练的轻量规则/确定性策略
```

- side 固定 `a/b`。decks 为 `{"a": 构筑JSON, "b": 构筑JSON}`；默认两侧基础预组。
- new_game 和 apply_action 返回可 JSON 序列化的状态 dict，其中玩家、异能者、实体卡牌为专用 dict 兼容类实例；apply_action 不修改输入，非法动作抛 ValueError，输入仍不变。每次成功动作增加 `version`。
- 同 seed、构筑和动作序列可重放，随机状态保存在内部 JSON；不使用全局 random 状态，不依赖 HTTP、Flask、DAO、ORM、线程或磁盘。
- phase 为 `mulligan/playing/choice/finished`；`active_side` 是正常回合行动方，`acting_side()` 优先返回待选择的 side；换牌阶段按未完成的一方返回，结束返回 None。
- `winner` 为 null、`a/b` 或 `draw`。`turn` 是全局连续回合号，双方自己回合计数另存以处理倒地和临时复制期限。
- 未结束对局中，任一参与者都可以 concede，不要求当前轮到该方；其余动作仍按当前操作者限制。RL 候选容量初值 256，超出要报错，不在引擎或 adapter 静默截断。
- legal_actions 提供完整合法候选，不用规则 AI 预先剪掉目标或牌；其返回的每个 dict 可以直接传给 apply_action。动作以稳定 JSON 字段编码，无需前端或 RL 推断规则。
- 选择前看不到的牌通过 pending_choice 拆为下一步动作；观察只向选择者显示其合法候选。每个动作都可完整序列化。

动作格式：

```json
{"type":"mulligan","card_ids":["手牌实例ID"]}
{"type":"attack","character_id":"nanali"}
{"type":"play_card","card_id":"手牌实例ID","target_id":"a:zero"}
{"type":"ultimate","character_id":"zero"}
{"type":"choose","choice_id":"候选ID"}
{"type":"end_turn"}
{"type":"concede"}
```

无目标卡不带 target_id。角色目标编码 `side:character_id`；卡牌目标（回收或指定手牌）使用对应卡牌实例 ID。mulligan 的选择为空数组表示保留全部，最多 3 张。已知合法目标可以在 legal_actions 中展开；抽牌后弃牌/检视牌库后的选择必须生成 pending_choice，不能泄露尚未揭示的牌序。

observe 的稳定 JSON：

```text
rules_version, version, phase, active_side, first_side, viewer_side, is_my_turn,
turn, winner,
sides: {
  a/b: {name, hp, shield, ap, normal_attack_available,
        front: 角色ID或null,
        characters: [{id,name,attribute,attack,hp,max_hp,shield,
                      harmony,energy,awakened,ultimate_turns,growth,shape,down_turns,
                      avatar,portrait,passive,awakened_passive}],
        hand: [公开卡牌或{hidden:true}；本人手牌按该方编队角色顺序排列，同角色保持原相对顺序], hand_count, deck_count,
        discard: [公开卡牌], records: [公开记录]}
},
pending_choice: {side,kind,prompt,choices:[{id,label,card?}]} 或 null,
legal_actions: [{action: 原始动作dict, label: 中文, preview: {...},
                 interaction: {kind, actor_id, target_ids}}],
events: [{seq,type,text,side?,source?,target?,amount?}], logs: [中文字符串],
presentation: {schema_version, cursor, oldest_seq, events}
```

公开卡牌字段：`instance_id, card_id, character_id, name, type, cost, terminal, description, copy, expires_turn?`。自己的手牌完整可见；对手只有牌背和数量，已经公开的复制可带公开标记。绝不返回牌库原始列表、随机种子/状态、对手隐藏手牌、对手选择候选或私有日志。records 只记录已经公开的原牌。

preview 至少给出 `ap_cost`，攻击可给 `attack, counter, harmony, will_switch`。预览只能依赖公开局面与操作者本来知道的信息，不模拟抽牌来泄露未来。前端不能自行计算伤害、费用或合法性。UI 可以仅展示引擎提供的保守预览。

## Web 接口（主任务实现）

页面：`/card-game` 大厅、`/build` 构筑、`/codex` 图鉴、`/replays` 回放列表、`/table` 牌桌切换为 V2；`/table?replay=<房间码>` 回放公开局面。旧快照通过 rules_version 判别，旧接口暂保留供旧对局恢复，不让新页调用旧部署接口。

新 API 使用 `/api/duel-v2` 前缀，沿用 Bearer token：

- `GET /catalog` -> 内容 catalog 加 `saved_build`（当前启用的一套）、`saved_builds`（已保存列表）和 `active_build_id`。
- `POST /build`，body 为构筑 JSON -> `{build, saved_build, saved_builds, active_build_id}`。无 id 或 id 为 starter/custom 时新建一套；最多 8 套。
- `POST /build-select`，body `{id}` -> 切换当前启用构筑。
- `POST /build-delete`，body `{id}` -> 删除一套；若删的是当前启用套，改启用剩余第一套。
- `POST /build-reorder`，body `{ids:[构筑id...]}` -> 按传入顺序重排已保存构筑；必须是现有 id 的全排列，不改变当前启用套。
- `POST /start`，body `{mode:"solo"|"pvp", deck?:构筑}` -> 房间/对局投影。solo 立即对局，pvp 创建等待房间。
- `POST /join`，body `{room_code}` -> 房间投影；`POST /ready` body `{is_ready:true}`；`POST /room-start` 正式开局。
- `GET /state` -> `{room: {room_code,mode,status,members,has_replay,replay_code}, game: observe结果或null}`；没有房间返回404。可选 `after_seq` 只过滤 `game.presentation.events`，不改变局势、合法动作或 cursor。
- `POST /action`，body `{room_code, request_id, expected_version, action}` -> 同样的 room/game 信封。
- `POST /leave` -> `{ok:true}`。
- `GET /replays` -> `{replays:[{room_code,mode,status,winner,viewer_side,name_a,name_b,favorited,characters_a,characters_b,event_count,updated_at}], unfavorited_limit, unfavorited_count}`，仅当前玩家参加过的人机/1v1 对局。`characters_*` 为双方开局四人头像摘要。收藏的回放始终列出；未收藏最多保留 30 场，超出的最旧未收藏对当前玩家不可见，若对方也不再保留则删除。
- `POST /replay-star`，body `{room_code, favorited}` -> 与 `GET /replays` 相同的列表信封。
- `POST /replay-import`，body 为按需 `--export-replays` 得到的公开回放 JSON（含 `views`，或单视角 `opening_board`+`events`）-> 写入当前玩家可看的已结束回放，返回与 `GET /replay/<房间码>` 相同信封。不得包含隐藏手牌、牌序或种子。训练过程不自动写入这些文件。
- `GET /replay/<room_code>` -> `{room, replay:{opening_board, events, viewer_side, favorited, ...}}`。`opening_board` 是开局公开局面；`events` 为完整公开演出事件（含 `seq` / `action_id` / `patch`），不截断为 200 条。回放不得包含对手隐藏手牌、牌库顺序、种子、未公开候选或 `legal_actions`。页面入口为 `/table?replay=<房间码>`，用现有牌桌渲染，提供播放/暂停/上一步/下一步。未收藏且已超出 30 场上限的回放返回 404。

room 补充字段：viewer_side、is_host、can_start；members 条目使用 `{side,name,nickname,is_ready,is_host,is_self}`，不返回对方构筑。新版使用独立的构筑/房间/快照表保留旧数据，开局和最终结果同步落盘，普通对局动作采用最新态缓存与异步版本写入。

所有 V2 动作逐次发请求、立即结算。前端需要 room_code 限定对局、稳定 request_id 防重试，带 expected_version 防过期操作；错误返回 `{error:中文}`，过期版本用409。UI 不在收到新状态前清空主要 DOM。

## 页面交互与验收

入口突出固定四人预组，可选人机、创建双人房间、输入房间码加入。构筑页可创建、编辑、删除最多 8 套构筑；每卡 0/1/2 张，显示每人 8 张与总 32 张，保存与恢复由后端严格校验。

牌桌布局明确双方玩家生命、单人战斗区、四人备战/倒地信息、自己的手牌、双方牌库/弃牌堆、行动力、日志。主操作必须支持拖动角色出击、拖动手牌出牌、点击的等价操作、后端候选选择、终结、结束回合和撤退。触屏采用 pointer events 或有可用点击替代，不能只实现桌面 dragstart。

所有角色/卡牌/目标合法性来自 game.legal_actions。拖动使用现有候选预览，不触发写请求；松开到合法区才提交一次，非法区域回弹。请求期间禁重入，动画完成自然更新。观察对手行动时轮询可刷新，但不泄露隐藏信息。

真实页面验收由主任务执行，Grok 页面 worker 仅进行代码层静态检查，不能访问个人浏览器数据。后端/内容单测使用纯内存和临时数据库。RL任务仅搭框架、静态检查与确定性小测试；禁止训练、经验采样或长时间自我对弈。

## RL 只读预审补充

协作任务提出的验收点（不改变上述函数签名）：

- 时间上限截断与真实终局分别处理；截断的可见状态及真实候选不能无说明改为全零，否则会影响未来 value bootstrap。info 区分总动作上限和对手推进上限。
- reset 可能在对手先手推进时直接终局或截断；Gym reset 不能返回这种全空动作状态给模型，应明确拒绝或采取文档定义的有限处理。
- 候选特征必须包含牌的身份、所属者、复制标记、选择目标、换牌组合等可见语义，不能只编码临时候选序号。
- 不为每个候选反复 deepcopy 全部观察和事件列表；本轮不进行吞吐 benchmark，也不承诺训练速度。
- 这是部分可见卡牌环境，不能声称观察是完整 Markov 状态。规则引擎可额外暴露公开的上一前排、疲劳计数、己方回合数、已用被动次数、临时增益等，但不得直接序列化内部 flags 或随机状态。
- terminal 的 mask 可以全空，但未结束的合法选择阶段不能被误判为终局；缺少映射的动作或卡牌必须明确报错。

## 已实现的补充公开字段

- side：`last_front`、`turn_count`、`fatigue`、`used`（稀疏已用次数）、`harmonized`（各角色是否在本局触发过环合）、`extra_flower`、`harmony_available`（当前是否有角色入场会触发环合）。
- character：`shape_id`（卡牌 ID，`shape` 是中文名）、`shape_description`、`harmony_max`、`energy_max`（光/灵/相为 5，暗/魂/咒为 6）、`family`（娜娜莉的家族壮大点数）、`next_attack_bonus`、`return_after_sortie`、`slow`（amount/end 或 null）、`burn`（剩余次数）、`star`（到期己方回合或 null）、`harmony_source`（上一名存活且满环合的出战角色）、`harmony_ready`（入场会触发环合）。到期计数按该角色所属侧的 `turn_count` 解释。
- 临时复制：`original_character_id` 表示原作者，`character_id` 始终是浔；复制的 `expires_turn` 是所属侧的己方回合计数，不是全局 `turn`。`unavailable_reason` 只作为 UI 提示，不由前端据此推算规则。
- `resolving_card`：正在结算的公开牌或 null。pending_choice.kind 实际为 `discard/inspect_top/enemy_hand`；后两者的私有候选仅选择者可见，候选最多受 10 张手牌上限约束。
- events/logs 为最近 200 条公开事件，日志按观察者显示我方/对手；不包含隐蔽抽牌名称或未公开候选。
- 攻击预览先用匿名占位替换双方牌库内容，再只执行已知的战斗操作；保留公开数量、护盾与疲劳规则，不读取或输出未来牌名。

网页同时显示双方个人资源；手机短横屏使用可滚动场地区域与固定手牌栏。卡牌点击打开详情及使用按钮，角色可点击或拖动出击，角色详情单独进入。
## 2026-09-12 公开演出协议

本批由 Grok 实现专业牌桌与演出所需的后端公开协议；浏览器验收仍由主任务进行，本文件不表示已上线或已平衡。不覆盖 2026-09-09 验收记录、旧盖卡规则或 RL 接口/权重/训练状态。

`observe()` 仍兼容两参数调用。新增 `game.presentation`：

```text
schema_version: 1
cursor: 最新公开事件 seq
oldest_seq: 当前窗口最早 seq
events: 投影后的公开演出事件
```

事件最低字段：`seq, action_id, action_version, type, side, text`。`action_id` 至少由对应动作 `version` 组成稳定 ID；一次 `apply_action` 及其后续 `choose` 续步按新 version 分步，不把出牌和 pending 选择合成一个 ID。不同人机动作不能合并。稳定类别包括 play, attack, combat, damage, flower, followup, penetration, move, enter, down, revive, ultimate, shape, growth, resource, draw, gain, remove, discard, turn, finish, record, redeem；其他事件可按 instant 处理。

`actor`/`target` 使用 `a:角色ID` / `b:角色ID` / `a:player` 规范，缺省允许 null。规范 ID 必须保留阵营，不能把裸角色名默认成观察者或当前行动方。`source` 继续保留兼容。play/gain/record/shape 等已公开卡牌带最小投影 `{instance_id,card_id,character_id,name,type,cost,terminal,description,copy,...}`；对手普通抽牌不得带牌名、实例 ID 或私有候选。`private_card`/`private_side` 不得进入对方 presentation，也不得留在对方 `events`/`logs`。

`before`/`after` 可提供当次 HP/shield/resources；同时攻守伤害共享 `group_id`，连击分为不同 group。`patch` 是只含公开变化的展示补丁，与 observe 局势兼容：`{sides:{a:{hp?,shield?,ap?,front?,characters:[{id,...}]},b:{...}},phase?,active_side?,turn?,winner?}`。character 数组按角色 id 合并，其余字段按值覆盖。不要放入合法动作、牌库、种子或隐藏手牌。从动作前公开 board 按 seq 应用本动作 patch 后，应等于动作后公开 board。

起始观察直接用最终快照，不回放旧事件。旧存档缺少演出元数据时兼容回退，不解析中文日志伪造卡牌；旧快照上的首次新动作仍应生成可回放 patch。`GET /state?after_seq=` 只过滤新增演出 events；不传参数仍返回完整最近窗口。

`legal_actions[]` 每条新增 `interaction`，不改变 action 本体：`{kind:attack|cast|target|form|ultimate|choice|other, actor_id, target_ids}`。UI 必须按单条 action 匹配提示。N06 这类已展开的友方目标，每条 hint 的 `target_ids` 只能是该条 action 的规范 `target_id`，不能把全部存活队友塞进每一条。复制 N06 归属浔，排除自己；规范 ID 不得忽略阵营。


## 2026-09-16 对局实体身份

`engine/duel_v2/entities.py` 定义 `PlayerEntity`、`CharacterEntity`、`CardEntity`。类保留 dict 访问和 JSON 序列化，减少现有规则与存储迁移风险；`entity_id` 才是对象身份，不能用 dict 值相等、角色模板 ID 或内存地址判断同一实体。对局内通过独立 `next_entity` 计数器分配 `eN`，与模板无关，保持确定性；跨对局须同时带房间/对局标识。单独构造、尚未入局的实体使用独立 standalone ID，正式开局与生成卡统一使用对局分配器。

- 两名玩家、双方所有异能者、每张牌（含同名原牌、衍生、临时复制）分别持有唯一实体 ID。
- 换区、倒地、复活不新建对象身份；复制牌创建新实体。`deepcopy` 创建新的 Python 对象但保留逻辑实体 ID；记录/事件投影是历史值，不是额外活实体。
- 数据库存储仍为 JSON；`hydrate_entities` 在存档读取与动作副本边界恢复实例、选择区/结算中卡牌别名，并为旧存档按稳定遍历补齐 ID；拒绝重复/冲突 ID。只读观察旧存档时使用副本，不改写原输入。
- `id`、`character_id`、`card_id` 仍标识目录模板或旧接口路由；卡牌 `instance_id` 作为既有唯一卡牌地址保留。阵营内 `front` 等字段暂保留兼容索引，不能把它们当成跨阵营实体身份。
- 公开玩家、异能者和可见牌带 `entity_id`；暗牌仍不公开 ID。事件新增 `actor_entity_id` / `target_entity_id`；现有 `actor` / `target` 的 `a:角色ID` 地址用于旧页面渲染兼容。
- `legal_actions[]` 保留旧 `action`，并提供推荐的新 `entity_action`。支持 `actor_entity_id`（出击/终结）、`card_entity_id`（出牌）、`target_entity_id`（目标）、`choice_entity_id`（选牌）、`card_entity_ids`（换牌）。动作入口先解析实体，再做完整合法性校验；不接受冒用对方角色或冲突的新旧地址。
- native/CUDA 的扁平存储是加速表示，不在 GPU 中创建 Python 类；异能者身份以阵营与槽位组成的实体键比较，不能单用角色槽位排除对方同名对象。正式网站结算始终使用上述类实例。

验证入口：`python3 -m unittest tests.test_duel_v2_entities tests.test_duel_v2_engine tests.test_duel_v2_public_compiled tests.test_duel_v2_api tests.test_duel_v2_tutorial`。


## 2026-09-18 金谷标注与 X选一协议

- 角色内部 `ultimate_used_turn` 记录上次发动终结的全局回合号，实体倒地/复活保留；训练张量和编译行使用 `ch_ultimate_used_turn`。玩家总终结次数仍独立计算。
- 战斗牌可带 `attack_mode: "set"`，表示 attack 设置本次攻击/反击的基值，不是对角色攻击的加成；未提供时保持原加成语义。
- 卡牌模板可带有序 `play_options: [{id,label,description,costs?,cost_label?}]`。`costs` 是所属角色个人资源的额外支付量。主动出牌的合法动作带 `option_id`，与 `card_id` / `card_entity_id`、可选目标一起提交；后端枚举可支付项并验证整个动作。非主动调用默认第一项。UI只按合法动作判定能否选择，不自行扣资源。
- 实体手牌 `jingu_mark: {delta: -1|0|1, source_entity_id}` 锁定金谷变动及来源实体。出牌时复制到操作状态，效果完整完成后兑现；被弃置或返回牌库清除，不带进弃牌堆。响应战斗牌通过操作级延后队列，在实际反击结算后兑现。
- 内部 `private_hand` 事件携带标注更新后的拥有者手牌；投影时只转换为该拥有者的 `patch.sides[side].hand`，其他视角不返回原始字段或隐藏牌身份。公开回放按同一投影与泄露检查处理。`play_options`、`attack_mode`、可见牌的 `jingu_mark` 进入公开卡牌最小字段。
- 拖牌分区的几何切分只决定 `option_id`，对应费用、合法性和预览均由后端提供。半尺寸引用详情从当前账号可见 catalog 按卡名查找，排除自身并去重。


## 离线搜索模拟的演出边界

`engine/duel_v2/simulation.py:simulate_action(state, side, action)`是离线假想动作入口，内部仍调用正式`apply_action`。它保留输入不可变、合法性校验、完整结算、实体／随机／事件序号，只省去公开演出patch和可派生棋盘缓存；不能替代网站持久化动作或公开回放采集。设置按上下文隔离并在异常后恢复。普通六接口默认行为不变，普通`apply_action`仍产生完整演出。

对拍时只有瞬时events和公开棋盘缓存可按约定剔除；其余状态（包括版本、RNG、个人资源、隐藏区和待选择）必须相同。事件还须在去掉patch后保持内容与顺序相同。卡效不得依赖演出patch或棋盘缓存进行规则结算。相关训练操作和源指纹迁移要求见训练手册“搜索内部的轻量演出模式”。


## 2026-10-07 公开单次伤害上限

异能者公开字段新增 `damage_limit: int | null`，无有效限制或倒地时为 null；与完整免疫字段 `damage_immune` 分开。后端通过通用 `damage.limit` 效果查询计算，伤害封顶在护盾之前，战斗溢出使用受限制的伤害。网页只展示后端字段；真实观察与公开棋盘patch携带相同值，历史回放缺少该字段时不推算或补造限制。


### 2026-10-07 残虹加攻计次与训练观察

残虹持续伤害加攻的全局回合计次保存在阵营公开`used`映射，以该角色实体身份索引；换武备和倒地复活不清除此计次。浊燃与统一恢复编码将其转换为双方残虹本回合是否已加攻的布尔特征，不输入实体编号。旧特征与新特征按名称研究迁移，旧规则身份及源数组保留；新规则模型不能继承历史验收或服务批准。
# 2026-10-07 高级人机逐操作补充契约

高级人机的玩家动作与人机动作分次提交。`POST /api/duel-v2/ai-step` 接受 `room_code`、`expected_version`、`request_id`，仅当前advanced房间的人类成员可调用；服务端决定b方动作，客户端不能指定该动作或冒充b方。每次成功请求仅应用一次正式规则动作并保存公开回放。重复请求返回已有结果；版本或房间冲突返回409；不是人机决策时不推进状态。`room.ai_pending` 表示人机是否仍为当前决策方，前端等本次事件演出完成后才请求下一步。普通人机、教程与PVP保持既有调度。
