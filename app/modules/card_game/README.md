# 异象对决模块

异象对决是仓库内的网页卡牌桌游模块，页面入口为 `/card-game`。它提供账号登录后的构筑、图鉴、对局数据和单战场牌桌体验。

## 四人轮替 V2

- [四人轮替版完整手册](../../../docs/everness-item-chain-card-design.md) 是新版唯一设计真源，开头置顶娜娜莉、主角「零」（鉴定师）、九原、浔及其 32 张基础预组。
- 新版设计为每方 4 人、单人战斗区、玩家生命胜负和完整回合轮替；玩家行动力支付费用，每名角色独立保存环合值与能量。角色开局有基础被动，能量满后可主动支付 1 行动力觉醒，无需觉醒卡，觉醒后替换被动。
- 每角色八种卡名固定为本人六个觉醒名＋两个觉醒共鸣名，均已通过 Everness 核实；从中选择 8 张带入牌库，同名最多 2 张。形态牌提供与被动、战斗牌的配合，不再用 PV/EP 补位。
- 核心职责：娜娜莉自身成长主攻、零自身环合加速、九原强化创生、浔记录队友 1 费原牌并通过终结战斗牌兑现爆发与临时复制。浔的兑现独立于自身环合，临时复制仍需付费且不能递归生成。
- 默认主页、构筑、图鉴与牌桌已接入 V2，提供人机和 1v1 房间。旧教程和看板仍使用盖卡模式；旧对局可经 `/table?legacy=1` 恢复。旧规则见 [盖卡版归档](../../../docs/archive/duel-simultaneous-deployment-2026-09-08.md)。
- 规则数值仍是试玩参数，功能测试不能证明已经平衡。完整验收目标见手册末节。

## 当前运行模块范围

- 固定四人开局备战，每人从自己的八种卡中携带八张，同名最多两张；总计 32 张。
- 后端负责个人资源、行动力、攻击、换人、创生、主动觉醒、复制、抽牌、倒地恢复和胜负。人机使用确定性规则策略，不依赖训练模型。
- 前端展示合法动作与后端预览，支持角色/卡牌拖动、卡牌详情、后续选择、起手换牌和刷新恢复。
- 玩家、登录态、构筑、房间和对局快照写入共享数据库。

新版规则、术语与卡表以完整手册为准；维护旧对局时参照归档。平衡流程与历史报告索引见 [异象对决平衡迭代](../../../docs/duel-balance-iteration.md)。

## 页面入口

- `/card-game`：模块主页。
- `/login`：登录页。
- `/profile`：账号设置。
- `/build`：构筑页。
- `/codex`：空间档案。
- `/analytics`：旧盖卡对局数据，尚未迁移到 V2。
- `/table`：牌桌。

模块不再提供 `/home` 入口。

## 代码边界

- `app/modules/card_game/content/`：卡牌、异能者、构筑和通用效果定义。
- `app/modules/card_game/engine/`：对局应用服务、流程、规则、AI、投影与状态契约。
- `engine/duel_v2/`：六个纯规则接口及内部状态、战斗、上下文、回合和投影。
- `content/duel_v2/`：八种/每人卡池、构筑校验和卡牌效果。
- `engine/application/v2_*.py` 与 `v2_routes.py`：新版房间、幂等请求、版本校验、同步关键资料与异步快照。
- `app/room_service.py`：旧盖卡房间与开局协作。
- `app/modules/card_game/templates/card_game/`：页面模板。
- `app/modules/card_game/static/`：前端脚本和样式。
- `tests/test_duel_v2_*.py`：新版目录、纯规则、接口及 RL 框架测试。
- `tests/test_solo_room_flow.py`：旧版接口与对局回归。

分层与依赖方向必须遵循仓库根目录 [AGENTS.md](../../../AGENTS.md)。卡牌私有逻辑优先留在内容定义中，不在路由或主流程增加长期硬编码分支。

## 验证

```bash
python3 -m unittest tests.test_module_routes.ModuleRoutesTest.test_card_game_module_page_and_asset
python3 -m unittest tests.test_duel_v2_catalog tests.test_duel_v2_engine tests.test_duel_v2_api tests.test_duel_v2_rl
node --check app/modules/card_game/static/js/v2_api.js
node --check app/modules/card_game/static/js/v2_build.js
node --check app/modules/card_game/static/js/v2_table.js
```

涉及真实交互或动画时，还需从 `/card-game` 通过页面进入牌桌完成浏览器验收。本次结果见 [V2 实现验收](../../../docs/duel-v2-acceptance.md)。

## 存档与协议

`/api/duel-v2` 提供 catalog/build/start/join/ready/room-start/state/action/leave。`state` 只返回本人视角；对手未公开手牌、牌序和随机状态不返回。完整请求与投影字段见 [实现契约](../../../docs/duel-v2-implementation-contract.md)。 2026-09-12 公开演出字段、action_id 续步、规范实体、interaction 单条匹配和 `GET state?after_seq=` 见 [演出协议](../../../docs/duel-v2-presentation-20260912.md)；本批由 Grok 实现，浏览器验收仍由主任务进行，不能视为已上线或已平衡。

新版使用四张专表，旧构筑与旧快照不会被转换或覆盖。普通动作先更新单进程最新态再异步入库，进程异常时可能丢失尚未入库的最后一步；正式结束结果与房间结束状态同步提交。刷新不会重新扣费，网络重试应复用 request_id，并携带 room_code 和 expected_version，旧房间请求不能作用于新房间。当前并发保证以单进程服务为边界，跨进程部署前需要增加集中式状态协调。

[强化学习框架说明](../../../docs/duel-v2-rl.md) 与 `python3 scripts/train_duel_v2.py --check` 可查看配置。强化学习仅提供与 V2 六接口对接的独立框架；不自动训练，不用旧版动作空间。部署网站无需安装训练依赖。
