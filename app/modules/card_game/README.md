# 异能对决模块

异能对决是仓库内的网页卡牌桌游，页面入口为 `/card-game`。登录后可构筑、查阅图鉴、进入单战场牌桌，并查看公开回放。默认主页、构筑、图鉴和牌桌使用四人轮替 V2；旧盖卡页面通过 `?legacy=1` 访问。

本文件只说明模块范围、页面入口、代码边界和验证命令。规则、数值和角色职责以完整手册为准，不在这里另写一份。

2026-09-15 起模块名称改为「异能对决」。Logo 使用青金色字标与 `ESPER DUEL` 副标，移除异象道具卡片及徽章；旧名称的构筑文本仍可导入。

## 文档入口

| 文档 | 用途 |
| --- | --- |
| [四人轮替完整手册](../../../docs/everness-item-chain-card-design.md) | 现行规则、卡表、预组和试玩参数的唯一设计真源 |
| [卡牌描述术语](../../../docs/duel-v2-card-text.md) | 牌面、被动和图鉴用语；不新增规则 |
| [新手教学](../../../docs/duel-v2-tutorial-design.md) | V2 教学关卡、引导 UI 与验收 |
| [实现契约](../../../docs/duel-v2-implementation-contract.md) | 引擎、网页与训练共用的接口字段 |
| [演出协议](../../../docs/duel-v2-presentation-20260912.md) | 公开演出字段、`action_id` 续步和增量 `state` |
| [实现验收](../../../docs/duel-v2-acceptance.md) | 功能验收记录，不是平衡证据 |
| [平衡迭代](../../../docs/duel-balance-iteration.md) | 测试流程与报告索引 |
| [强化学习](../../../docs/duel-v2-rl.md) | 训练框架、旧镜像入口与观察边界 |
| [公开卡池训练](../../../docs/duel-v2-public-training.md) | 现行完整公开卡池训练、分析与录像入口 |
| [盖卡版归档](../../../docs/archive/duel-simultaneous-deployment-2026-09-08.md) | 仅用于维护尚未迁移的旧对局 |

分层与依赖方向见仓库根目录 [AGENTS.md](../../../AGENTS.md)。

## 模块范围

- 完整卡表 13 名角色、104 种卡牌。正式账号公开娜娜莉、零、九原、伊洛伊、薄荷、白藏，以及创生预组和覆纹快攻预组；其余角色与含测试角色的官方预组仅测试账号可见。
- 每方 4 人、单人战斗区、玩家生命胜负、完整回合轮替。玩家行动力支付费用，手牌一律 1 费；每名角色独立保存环合值与能量。能量满后可发动终结，不消耗行动力，每回合限 1 次，备战区也可使用。
- 每个回合开始通常抽 1 张，包括先手首回合。起手 5 张；先手首回合 1 点行动力，后手开局 5 点护盾。正式 Python 引擎、张量训练与编译训练后端同步执行该抽牌规则。
- 后端负责资源、出击、换人、创生、终结、复制、抽牌、倒地恢复和胜负。普通人机使用确定性规则策略；高级人机使用冻结模型在 CPU 上选行动，规则仍由正式 Python 引擎结算。
- 前端只渲染公开投影、合法动作和后端预览，不在浏览器里计算费用、伤害或胜负。
- 玩家、登录态、构筑、房间和对局快照写入共享数据库。V2 使用独立构筑/房间/成员/快照/回放表，不覆盖旧盖卡数据。
- 规则数值仍是试玩参数。功能测试、训练短跑和页面验收都不能证明已经平衡。
- 旧教程和 `/analytics` 数据看板仍使用盖卡模式，尚未迁移。

核心职责只作索引，细则见手册：娜娜莉自身成长主攻、零自身环合加速、九原强化创生、浔记录兑现、安魂曲噩梦、残虹焚印、早雾负面增伤、灵可同频、真红凝视、哈索尔延滞、伊洛伊治疗加攻与复活、薄荷溢出穿透、白藏自伤叠攻与斩杀。

## 页面入口

- `/card-game`：模块大厅。可进入 V2 新手教学第一关（2v2 面板出击）。
- `/login`：登录页。
- `/profile`：账号设置。
- `/build`：构筑页。首次进入获赠当前可见的官方预组，之后与普通构筑相同，最多 8 套。手机端按 CSS 视口适配，编队置于卡表之前；横屏四列、窄屏两列。测试角色与含测试角色的预组仅测试账号可见。
- `/codex`：空间档案。右上角问号打开机制词表，可搜索名称或说明。
- `/replays`：对局回放列表。双方开局四人头像表示构筑，可收藏；未收藏最多保留 30 场。
- `/analytics`：旧盖卡对局数据，尚未迁移到 V2。
- `/table`：牌桌。
- `/table?replay=<房间码>`：用现有牌桌回放该对局的公开局面。
- `/table?legacy=1`、`/build?legacy=1`、`/codex?legacy=1`：旧盖卡页面。

模块不再提供 `/home` 入口。

## 牌桌约定

这些是当前网页实现口径，不替代手册规则。

### 操作

- 菜单提供「拖动选择目标」开关，默认开启并保存在本机。按下或点击卡牌时不显示拖动热区；按住移动超过 8 像素后才显示拖动卡影与合法目标。有目标牌只高亮目标，拖到目标松开出牌，非法落点回弹，不显示取消按钮。无目标卡（含固定作用于所属角色的装备，以及自动选取攻击目标的战斗牌）可拖到整个战斗大区，包含双方备战区、前排和其间空白。关闭后，所有牌先拖到同一战斗大区松开，有目标卡再进入可取消的点选界面。
- 手牌瞬发虚线边框仅在我方本回合瞬发机会尚未使用时显示；机会用完后恢复普通可用牌边框，响应标记保留。
- 可出牌的整张手牌（含描述区域）优先响应拖动。普通手牌与起手换牌均不提供描述内滚动，超长内容截断，完整说明在大图中阅读。

### 信息与标记

- 待生效效果按生命周期显示：倒地会清除的效果标记放在我方异能者卡片下方、对方卡片上方；不依附角色存活的效果放在战斗区侧边。标记显示名称和触发时机，悬停说明具体效果及存活条件。「错误的门」已触发的追加创生不因伊洛伊倒地而取消，放在战斗区左侧；结算时原环合角色仍须存活。环合状态继续放在右侧。实时局面与公开回放共用后端标记投影。
- 战斗区状态显示在牌桌战斗区右侧，附带倒计时或「本回合」。浊燃与黯星在区域所属方回合开始结算，无前排时伤害玩家；开始效果与倒计时处理后才退回前排，再通常抽牌。
- 事件局面同步携带环合来源、可入场角色和战斗区高亮标记；倒地、离场、资源变化后随对应动画结算立即更新。环合来源角色倒地时，后端清除其最近出战记录；复活与自然恢复不恢复旧环合机会，个人环合值和能量仍保留。
- 异能、武备和延后效果的触发结果写入历史与公开回放。内容回调用 `EffectContext.history()` 记录已实际发生的公开效果，不能在合法性或预览检查中记录。
- 入弃牌堆事件不显示弹出提示、历史提示或回放步骤文案；事件与状态更新保留，弃牌堆数量和回放局面照常更新。

### 演出

- 普通抽牌与生成牌动画落到所属角色手牌组的末尾；该角色暂无手牌时按编队顺序插入。对手未公开抽牌的放大与入手动画均显示牌背；已公开的牌按公开牌面演出。
- 起手换牌沿用换牌区内的旧牌切出、新牌切入；换入牌不播放普通抽牌的放大飞入动画。
- 获得环合值或能量时只让卡片上变化的资源数字跳动，不播放资源飘字或字幕。
- 交战阶段只播放出击与反击动作，不提前显示伤害数值字幕。援护技免疫反击时，攻击者在原反击扣血时机飘出金色「无敌」；不把其他零伤害当作无敌。
- 衍生牌与临时复制成功加入手牌时，用公开 `gain.card` 先在视野中央显现牌面，再缩小飞入获得方手牌。
- 触发环合时，对战区中央播放对应名称的艺术字；素材保存在模块 `static/images/harmony/`。
- 演出提示统一使用 `_notice(title, detail, position)`。回合、通用通知及角色状态提示放上方，出牌、抽牌等过程说明放下方。

### 字号与布局

牌桌菜单提供全局字号五档：小（80%）、偏小（90%）、标准（100%，默认）、偏大（110%）、大（120%）；另提供「布局大小」0.8×、1×、1.5×、2× 四档。字号乘以布局比例，两个选项独立保存。回放共用此设置。

### 手机

- 起手换牌顶部提供「收起换牌 / 展开换牌」。手机换牌卡显示效果摘要，完整说明在大图中阅读。
- 大厅、构筑、图鉴按 CSS 视口排版；手机横竖屏允许纵向滚动。图鉴角色导航横向滚动，卡片横屏四列、窄屏两列。
- PC 终结按钮为立绘顶部的扁按钮，位于护盾、攻击与生命数值上方。手机终结按钮放在角色卡右侧竖排，待生效标记在左侧仅竖排两个字；环合与能量优先同一行，必要时缩写为「环」「能」。
- 手机出牌演出牌面宽度至多 220px、高度至多屏高的 65%（上限 300px）。
- 粗指针且宽度不超过 1200 CSS px 时，基础系数为 0.8，再乘布局档位与字号档位。横屏正常对战，竖屏只提示「请横屏游玩」。

## 构筑纯文本分享

构筑页「导出」复制或下载 UTF-8 `.txt`；只导出通过现行校验的当前草稿。文本无空行，`#` 后为注释，包含格式版本、名称、按编队顺序排列的角色 ID、卡牌 ID 与数量。名称由引号保护，名称内的 `#` 是正文：

```text
# 异能对决构筑
异能对决构筑 1
名称 '创生预组'
角色 nanali # 娜娜莉
卡牌 N01 2 # 帮派初建成
```

「导入」粘贴完整文本，后端校验四名角色、每人八张、同名最多两张及当前账号的角色可见权限。成功后创建本页新草稿，点击保存才新增构筑，仍受八套上限约束。文本不携带账号或已保存构筑 ID，不覆盖已有构筑。

`POST /api/duel-v2/build-export` 接受构筑对象、返回 `text`；`POST /api/duel-v2/build-import` 接受 `text`、返回未保存的 `build`。编解码位于 `engine/application/v2_build_text.py`。

## 高级人机

大厅选择「高级人机」，再选择对手使用的「创生预组」或「覆纹快攻」。我方构筑可选择「与对手相同的预组」或「当前保存的构筑（仅公开角色）」；保存的个人构筑不会被覆盖。自定义构筑需要对应模型已声明支持。当前随 app 保存的模型只支持镜像预组，不能当作已完成自由构筑训练。房间模式为 `advanced`。起手换牌时模型保留全部起手牌。对手名及回放显示「高级人机」。

服务端仅依赖可选的 NumPy CPU 推理：`pip install -r requirements-ai.txt`。默认读取 `app/modules/card_game/engine/ai/models/{starter,weave-rush}.{npz,json}`，可用 `NTE_DUEL_AI_MODEL_DIR` 指定目录。数字权重不允许 pickle；加载时校验 SHA-256、卡池/座位顺序、观察 schema、规则哈希和参数形状。缺依赖、缺文件或版本不匹配会在创建房间前明确拒绝，不降级冒充高级人机。

先手首回合抽牌调整后，随 app 发布的两套旧权重继续使用，不重训、不改变观察或动作接口。模型清单的 `rule_identity` 保留训练时规则，`runtime_rule_hash` 精确限定已核验可运行的新规则。该兼容性不代表旧策略在新规则下最优，历史胜率不能直接沿用。

`engine/ai/advanced_model.py` 负责纯 CPU 编码与选择；`rl/gpu_duel/resident_schema.py` 共享训练观察字段。`scripts/export_duel_v2_ai.py` 把训练环境中的 `best.pt` 导出为数值权重；`scripts/verify_duel_v2_ai_export.py --directory DIR` 在无 torch 的服务环境验证输出。Git 的 `engine/ai/models/` 只保留两套当前策略，合计约 695 KB，随 `scripts/deploy_windows_app.sh --deploy` 发布。完整 checkpoint、优化器、对照样本和原始日志留在 `artifacts/`，不进入 app 或 Git。网站高级人机只加载数值导出文件，不加载训练 checkpoint，也不启动训练。

完整公开卡池训练入口见 [公开卡池训练](../../../docs/duel-v2-public-training.md)。导出模型须匹配完整评估后才允许网页使用；当前随 app 保存的权重仍仅开放镜像。

## 对局白热化

新建 V2 对局在总第 6 回合起，每回合开始为当前玩家最低能量存活异能者 +1 能量（同值随机），并有每回合 2 次终结机会；第 13 回合双方能量上限 -1；第 20 回合双方环合上限降为 1。上限变化会同步截断双方已有资源。战斗区中间左侧显示下一阶段倒计时。旧快照和前 11 关教学不追加。第 12 关「白热化」显式启用，从后手总第 4 回合开始，到第 6 回合完成补能与两次终结。完整口径见手册「对局白热化」。

规则快攻镜像对照：`python3 scripts/evaluate_duel_v2_escalation.py --pairs 64 --workers 4 --output /private/tmp/nte-escalation-eval.json`。

## 代码边界

- `content/duel_v2/`：现行卡池、构筑校验、角色包（被动/弧盘/卡效）和效果注册表。引擎通过 `fire` 追加时机、通过 `decide` 覆盖决策，不按角色 ID 写死被动。代码里的 `form` 对应玩家可见的「弧盘」。
- `content/` 其余目录：旧盖卡卡牌、异能者、构筑和通用效果。
- `engine/duel_v2/`：纯 JSON 规则引擎，含状态、战斗、效果上下文、回合、投影、教学和白热化。
- `engine/application/v2_*.py` 与 `v2_routes.py`：新版房间、幂等请求、版本校验、同步关键资料与异步快照。
- `engine/ai/`：普通人机规则策略，以及高级人机的 CPU 推理与随 app 发布的模型。
- `rl/`：训练框架、编译规则、公开卡池观察和离线分析。导入该包不得创建对局或启动训练。
- `engine/` 其余目录：旧盖卡应用服务、流程、规则、投影与房间协作。
- `templates/card_game/` 与 `static/`：页面模板、前端脚本和样式。`v2_*` 为现行页面；无前缀文件服务旧盖卡。
- `tests/test_duel_v2_*.py`：新版目录、纯规则、接口、教学、演出与训练测试。
- `tests/test_solo_room_flow.py`：旧版接口与对局回归。

卡牌私有逻辑优先留在内容定义中，不在路由或主流程增加长期硬编码分支。

## 存档与协议

`/api/duel-v2` 提供 catalog、构筑保存/导入导出/删除/选择/排序、开局、加入、准备、房主开局、state、action、离开，以及回放列表、收藏、导入和按房间码读取。`state` 只返回本人视角；对手未公开手牌、牌序和随机状态不返回。完整公开回放日志独立于 `observe()` 的最近 200 条窗口，开局公开 board 加后续带 patch 的公开事件，按 `seq` / `action_id` 步进还原牌桌。

普通动作先更新单进程最新态再异步入库，进程异常时可能丢失尚未入库的最后一步；正式结束结果与房间结束状态同步提交。刷新不会重新扣费。网络重试应复用 `request_id`，并携带 `room_code` 和 `expected_version`。当前并发保证以单进程服务为边界。

训练对局要看时再 `--export-replays` 转成公开 JSON，可在回放页导入。完整请求与投影字段见实现契约和演出协议。

## 训练

固定角色配牌学习入口为 `scripts/train_duel_v2_cards.py`：用户指定四名公开角色，模型只调整每人 8 张配牌（同名最多 2 张）。冻结出牌策略执行终局评估，独立配牌网络学习；不自动更换角色或网站预组。先用 `--check --team starter` 检查配置，CUDA 训练与候选分析命令见公开卡池训练文档。

网站运行时不加载训练权重，也不在导入 `rl/` 时自动训练。现行完整公开卡池入口是 `scripts/train_duel_v2_public.py`，分析与录像走 `scripts/analyze_duel_v2_policy.py`。旧镜像、SB3 和 `train_duel_v2_session.py` 只保留给既有实验。训练状态页用 `python3 scripts/rl_monitor.py` 单独开在 5002，大厅无入口。小样本结果不是平衡证据。命令、配置和日志字段见强化学习说明与公开卡池训练文档。

## 验证

按改动范围选择对应检查，不要把旧盖卡测试或训练短跑当成新版平衡证据。

网页与规则改动：

```bash
python3 -m unittest tests.test_module_routes.ModuleRoutesTest.test_card_game_module_page_and_asset
python3 -m unittest tests.test_duel_v2_zone_status tests.test_duel_v2_catalog tests.test_duel_v2_engine tests.test_duel_v2_hooks tests.test_duel_v2_api tests.test_duel_v2_replay tests.test_duel_v2_table_contract tests.test_duel_v2_tutorial tests.test_duel_v2_opening_draw tests.test_duel_v2_escalation tests.test_duel_v2_trigger_history
node --check app/modules/card_game/static/js/v2_api.js
node --check app/modules/card_game/static/js/v2_build.js
node --check app/modules/card_game/static/js/v2_table.js
node --check app/modules/card_game/static/js/v2_table/replay.js
node --check app/modules/card_game/static/js/v2_table/tutorial.js
node --check app/modules/card_game/static/js/v2_home.js
node --check app/modules/card_game/static/js/v2_replays.js
```

高级人机：

```bash
python3 -m unittest tests.test_duel_v2_advanced_ai tests.test_duel_v2_api tests.test_duel_v2_replay
```

训练框架或编译规则：

```bash
python3 -m unittest tests.test_duel_v2_rl tests.test_duel_v2_rl_train tests.test_duel_v2_gpu_duel tests.test_duel_v2_rule_ir
```

完整 V2 测试：

```bash
python3 -m unittest discover -s tests -p 'test_duel_v2_*.py'
```

涉及真实交互或动画时，还需从 `/card-game` 通过页面进入牌桌，完成出击、换人、出牌、终结、结束回合、刷新恢复和查看日志。
