# 异象对决模块

异象对决是仓库内的网页卡牌桌游模块，页面入口为 `/card-game`。它提供账号登录后的构筑、图鉴、对局数据和单战场牌桌体验。

## 四人轮替

- [四人轮替完整手册](../../../docs/everness-item-chain-card-design.md) 是唯一设计真源。当前完整卡表 13 名角色、104 种卡牌。正式账号图鉴与构筑公开娜娜莉、零、九原、伊洛伊、薄荷、白藏，以及创生预组和覆纹快攻预组；其余角色与含测试角色的官方预组仅测试账号可见。
- 新版设计为每方 4 人、单人战斗区、玩家生命胜负和完整回合轮替；玩家行动力支付费用，所有手牌均为 1 费，每名角色独立保存环合值与能量。角色开局有异能；主动前排战斗成功时全队存活异能者 +1 能量、进攻者额外 +1（合计 +2），环合只给进攻者；反击不获得基础资源，卡牌明确给出的能量仍给 1 点；光/灵/相满槽 5、暗/魂/咒满槽 6。能量满后可发动终结，不消耗行动力，每回合限 1 次，备战区也可使用。终结是持续若干己方回合的状态或一次性效果。自己的回合开始时前排自动返回备战区（倾陷除外）；倾陷角色不能普通出击，但可以被替换下场。入场环合只检测上一名存活出战角色，不要求将其换下。倒地时保留环合与能量，倒地角色不提供环合、不再获得资源。
- 每名异能者八种卡名为本人六个觉醒名、一个专属武备名和一个官方角色 PV 名；08 为专属武备，07 为官方角色 PV。没有角色 PV 时用官方短片或用户指定名。零的 07/08 都是武备。从中选择 8 张带入牌库，同名最多 2 张。武备牌刷新所属异能者的攻击与生命面板。牌面用语见 [卡牌描述术语](../../../docs/duel-v2-card-text.md)。角色 = 异能者 + 玩家；对面称对方。
- 核心职责：娜娜莉自身成长主攻、零自身环合加速、九原强化创生、浔记录兑现、安魂曲噩梦、残虹焚印、早雾负面增伤、灵可同频、真红凝视、哈索尔延滞、伊洛伊治疗加攻与复活、薄荷溢出穿透、白藏自伤叠攻与斩杀。
- 默认主页、构筑、图鉴与牌桌使用现行四人轮替；旧教程和看板仍使用盖卡模式；旧对局可经 `/table?legacy=1` 恢复。旧规则见 [盖卡版归档](../../../docs/archive/duel-simultaneous-deployment-2026-09-08.md)。
- 规则数值仍是试玩参数，功能测试不能证明已经平衡。完整验收目标见手册末节。

## 当前运行模块范围

- 固定四人开局备战，每人从自己的八种卡中携带八张，同名最多两张；总计 32 张。
- 后端负责个人资源、行动力、攻击、换人、创生、终结、复制、抽牌、倒地恢复和胜负。人机使用确定性规则策略，不依赖训练模型。
- 前端展示合法动作与后端预览，支持角色/卡牌拖动、卡牌详情、起手换牌和刷新恢复。出牌拖到中央区域即可；无目标卡立即结算，有目标卡再进入可取消的遮罩选择，合法目标高亮、其余变暗。
- 玩家、登录态、构筑、房间和对局快照写入共享数据库。

新版规则、术语与卡表以完整手册为准；维护旧对局时参照归档。平衡流程与历史报告索引见 [异象对决平衡迭代](../../../docs/duel-balance-iteration.md)。

## 页面入口

- `/card-game`：模块大厅。大厅可进入 V2 新手教学第一关（2v2 面板出击）；完整关卡设计见 [新手教学](../../../docs/duel-v2-tutorial-design.md)。
- `/login`：登录页。
- `/profile`：账号设置。
- `/build`：构筑页，首次进入获赠当前可见的官方预组，之后与普通构筑相同，最多 8 套，可切换、拖动排序、编辑、删除。测试角色与含测试角色的预组仅测试账号可见。
- `/codex`：空间档案；右上角问号打开机制词表，可搜索名称或说明，按 Esc、关闭按钮或点击弹窗外侧关闭。词表模板为 `templates/card_game/v2_codex_glossary.html`，说明与完整手册同步。
- `/replays`：对局回放列表。
- `/analytics`：旧盖卡对局数据，尚未迁移到 V2。
- `/table`：牌桌。
- `/table?replay=<房间码>`：用现有牌桌回放该对局的公开局面。

模块不再提供 `/home` 入口。

## 代码边界

- `app/modules/card_game/content/`：卡牌、异能者、构筑和通用效果定义。
- `app/modules/card_game/engine/`：对局应用服务、流程、规则、AI、投影与状态契约。
- `engine/duel_v2/`：六个纯规则接口及内部状态、战斗、上下文、回合和投影。
- `content/duel_v2/`：卡池、构筑校验、角色包（被动/弧盘/卡效）和效果注册表。引擎发追加时机 `fire` 与决策覆盖 `decide`（如环合种类），不按角色 ID 写死被动。旧盖卡事件总线不用于 V2。代码 `form` 对应玩家可见的「弧盘」。
- `engine/application/v2_*.py` 与 `v2_routes.py`：新版房间、幂等请求、版本校验、同步关键资料与异步快照。
- `app/room_service.py`：旧盖卡房间与开局协作。
- `app/modules/card_game/templates/card_game/`：页面模板。
- `app/modules/card_game/static/`：前端脚本和样式。
- `tests/test_duel_v2_*.py`：新版目录、纯规则、接口及 RL 框架测试。
- `tests/test_solo_room_flow.py`：旧版接口与对局回归。

分层与依赖方向必须遵循仓库根目录 [AGENTS.md](../../../AGENTS.md)。卡牌私有逻辑优先留在内容定义中，不在路由或主流程增加长期硬编码分支。

## 验证

`test_duel_v2_catalog` 是离线目录回归：检查卡牌结构、已明确的卡名和来源元数据，不读取本机临时目录或联网拉取资料。历史 Everness 快照与临时审批卡表不再作为测试输入；官方名称的外部核验按完整手册的来源核对流程执行。

```bash
python3 -m unittest tests.test_module_routes.ModuleRoutesTest.test_card_game_module_page_and_asset
python3 -m unittest tests.test_duel_v2_catalog tests.test_duel_v2_engine tests.test_duel_v2_hooks tests.test_duel_v2_api tests.test_duel_v2_replay tests.test_duel_v2_rl tests.test_duel_v2_rl_train tests.test_duel_v2_gpu_duel tests.test_duel_v2_rule_ir tests.test_duel_v2_table_contract tests.test_duel_v2_tutorial
node --check app/modules/card_game/static/js/v2_api.js
node --check app/modules/card_game/static/js/v2_build.js
node --check app/modules/card_game/static/js/v2_table.js
node --check app/modules/card_game/static/js/v2_table/replay.js
node --check app/modules/card_game/static/js/v2_table/tutorial.js
node --check app/modules/card_game/static/js/v2_home.js
node --check app/modules/card_game/static/js/v2_replays.js
```

涉及真实交互或动画时，还需从 `/card-game` 通过页面进入牌桌完成浏览器验收。本次结果见 [V2 实现验收](../../../docs/duel-v2-acceptance.md)。

## 存档与协议

`/api/duel-v2` 提供 catalog/build/build-select/build-delete/build-reorder/start/join/ready/room-start/state/action/leave，以及回放入口 `GET /replays`、`POST /replay-star`、`POST /replay-import` 与 `GET /replay/<房间码>`。训练对局要看时再 `--export-replays` 转成公开 JSON，可在回放页导入。回放列表用双方开局四人头像表示构筑，可收藏；未收藏最多保留 30 场。`state` 只返回本人视角；对手未公开手牌、牌序和随机状态不返回。完整公开回放日志独立于 `observe()` 的最近 200 条窗口，开局公开 board 加后续带 patch 的公开事件，按 `seq` / `action_id` 步进还原牌桌。完整请求与投影字段见 [实现契约](../../../docs/duel-v2-implementation-contract.md)。 2026-09-12 公开演出字段、action_id 续步、规范实体、interaction 单条匹配和 `GET state?after_seq=` 见 [演出协议](../../../docs/duel-v2-presentation-20260912.md)；大厅导航进入 `/replays` 列表，观看时仍复用现有牌桌。功能测试不能视为已上线或已平衡。

新版使用构筑/房间/成员/快照/回放专表，旧构筑与旧快照不会被转换或覆盖。公开回放日志与对局快照分开保存，离开房间后仍可按房间码观看。普通动作先更新单进程最新态再异步入库，进程异常时可能丢失尚未入库的最后一步；正式结束结果与房间结束状态同步提交。刷新不会重新扣费，网络重试应复用 request_id，并携带 room_code 和 expected_version，旧房间请求不能作用于新房间。当前并发保证以单进程服务为边界，跨进程部署前需要增加集中式状态协调。

[强化学习说明](../../../docs/duel-v2-rl.md)：`python3 scripts/train_duel_v2.py --check` 只检查配置；`--train --output DIR` 使用 GPU 候选打分与多进程采样，默认双方都用当前创生预组，对抗冻结规则启发式，再做固定种子策略评估。`--deck weave-rush` 改为覆纹快攻镜像。1024 局张量环境用 `python3 scripts/train_duel_v2_gpu.py --output DIR --n 1024 --horizon 32`；`--deck weave-rush` 跑快攻镜像，`--matchup cross` 跑创生 vs 快攻，`--resume gpu_policy.pt` 接上一阶段。换少数卡时用 `--resume` 加 `--freeze-generic`（可选 `--kl-coef`），一次训练只打一套牌。训练状态页用 `python3 scripts/rl_monitor.py` 单独开在 5002，大厅无入口。网站不加载训练权重。小样本结果不是平衡证据。
