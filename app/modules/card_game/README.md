# 异能对决模块

异能对决是仓库内的网页卡牌桌游，页面入口为 `/card-game`。登录后可构筑、查阅图鉴、进入单战场牌桌，并查看公开回放。默认主页、构筑、图鉴和牌桌使用四人轮替 V2；旧盖卡页面通过 `?legacy=1` 访问。

本文件只说明模块范围、页面入口、代码边界和验证命令。训练先读[当前执行链路](../../../docs/duel-v2-public-training.md#当前训练链路与报告落点)和[完整评估规范](../../../docs/duel-v2-full-training-report-spec.md)；单轮报告存放在被Git忽略的 `artifacts/rl-evals/`，下文历史报告链接仅指向本机资料。规则、数值和角色职责以完整手册为准，不在这里另写一份。

恢复训练可显式使用 `--value-full-tower --value-replicas 12 --value-check-replicas 3` 扩大独立价值分支校准；轮数／温度只在选择集决定，不改变策略或放松留出条件。不同协议需新CUDA预检，见[价值校准协议](../../../docs/duel-v2-policy-recovery.md#独立价值分支的扩大校准)。验证入口：`.venv/bin/python -m unittest tests.test_duel_v2_recovery_round tests.test_duel_v2_current_round`。

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
| [离线批量搜索](../../../docs/duel-v2-batched-search.md) | 无损图编码、设备存储、批量推理、协作搜索及五队 GPU 规则接入边界 |
| [训练结束标准与首轮结果](../../../docs/duel-v2-training-stop-criteria.md) | 互搏训练停止口径及 2026-09-15 历史双预组结果 |
| [公开卡池训练](../../../docs/duel-v2-public-training.md) | 现行完整公开卡池训练、分析与录像入口 |
| [完整训练与报告要求](../../../docs/duel-v2-full-training-report-spec.md) | 双预组测试矩阵、配牌调整、先后手胜率、逐卡与机制分析的交付规范 |
| [全链路短测验收](../../../artifacts/rl-evals/reports/duel-v2-full-chain-smoke-20260916.md) | 双预组训练/配牌/互搏/报告生成及 Mac 交付的实测记录 |
| [纯胜负奖励与 N01 链路验证](../../../artifacts/rl-evals/reports/duel-v2-outcome-chain-20260916.md) | 旧策略热启动、CUDA 换牌、纯胜负奖励、新卡效与 Mac 报告交付实测 |
| [盖卡版归档](../../../docs/archive/duel-simultaneous-deployment-2026-09-08.md) | 仅用于维护尚未迁移的旧对局 |

分层与依赖方向见仓库根目录 [AGENTS.md](../../../AGENTS.md)。

## 模块范围

- 完整卡表 19 名角色、152 种可构筑卡牌。正式账号公开娜娜莉、零、九原、伊洛伊、薄荷、白藏、小吱、海月、真红、翳、安魂曲、残虹、早雾、阿德勒，以及创生、覆纹、小吱、真红、浊燃五套预组。随机简单人机从这五套公开预组中抽取。
- 每方 4 人、单人战斗区、玩家生命胜负、完整回合轮替。玩家行动力支付费用，手牌按牌面支付行动力，浔包含零费战术；每名角色独立保存环合值与能量。能量满后可发动终结，不消耗行动力，每回合限 1 次，备战区也可使用。
- 每个回合开始通常抽 1 张，包括先手首回合。起手 5 张；先手首回合 1 点行动力，后手开局 5 点护盾。正式 Python 引擎、张量训练与编译训练后端同步执行该抽牌规则。
- 2026-09-19 回合边界纠正：玩家护盾在己方回合开始效果前清零（含后手开局护盾）；家族壮大从整局第 5 回合起具有瞬发。正式规则与训练源码同步，专项验证见 [回合边界修正](../../../docs/duel-v2-turn-boundaries-20260919.md)。
- 薄荷 M07 在对方回合将面板攻击力变为 0，战斗牌本次加攻仍另加；详见 [M07 光环修正](../../../docs/duel-v2-m07-aura-20260919.md)。
- 后端负责资源、出击、换人、创生、终结、复制、抽牌、倒地恢复和胜负。普通人机使用确定性规则策略；高级人机使用冻结模型和有界信息集搜索在 CPU 上逐操作选行动，规则仍由正式 Python 引擎结算。
- 前端只渲染公开投影、合法动作和后端预览，不在浏览器里计算费用、伤害或胜负。
- 玩家、登录态、构筑、房间和对局快照写入共享数据库。V2 使用独立构筑/房间/成员/快照/回放表，不覆盖旧盖卡数据。
- N01「帮派初建成」按 2026-09-16 用户口径改为回复娜娜莉至当前生命上限，移除检索和瞬发；正式规则与训练后端同步。
- 规则数值仍是试玩参数。功能测试、训练短跑和页面验收都不能证明已经平衡。
- 旧教程和 `/analytics` 数据看板仍使用盖卡模式，尚未迁移。

核心职责只作索引，细则见手册：娜娜莉自身成长主攻、零自身环合加速、九原强化创生、浔时计与荒时支援、安魂曲噩梦、残虹持续伤害、早雾鬼郎丸、阿德勒护盾、灵可同频、真红盈蓄追加攻击、翳兽牙影刺、哈索尔延滞、伊洛伊治疗加攻与复活、薄荷溢出穿透、白藏自伤叠攻与斩杀。

## 页面入口

- `/card-game`：模块大厅。可进入 V2 新手教学（共 13 关，第一关为 2v2 面板出击，末关为远程）。
- `/login`：登录页。
- `/profile`：账号设置。
- `/build`：构筑页。首次进入获赠当前可见的官方预组，之后与普通构筑相同，最多 8 套。手机端按 CSS 视口适配，编队置于卡表之前；横屏四列、窄屏两列。测试角色与含测试角色的预组仅测试账号可见。
- `/codex`：空间档案。右上角问号打开机制词表，可搜索名称或说明。环合词条统一按效果、持续时间、补充说明排列，己方／对方以施加方为视角；盈蓄与失谐单列触发条件。其他词条按发动、次数、限制等含义分段，操作提示独立展示；抽牌、白热化终结次数、倒地响应例外与入场环合的回合条件按现行规则说明。
- `/replays`：对局回放列表。双方开局四人头像表示构筑，可收藏；未收藏最多保留 30 场。
- `/analytics`：旧盖卡对局数据，尚未迁移到 V2。
- `/table`：牌桌。
- `/table?replay=<房间码>`：用现有牌桌回放该对局的公开局面。
- `/table?legacy=1`、`/build?legacy=1`、`/codex?legacy=1`：旧盖卡页面。

模块不再提供 `/home` 入口。

## 牌桌约定

这些是当前网页实现口径，不替代手册规则。

### 操作

- 回放顶部「更换视角」在当前步切换双方上下位置并暂停；角色、资源、手牌展示及历史/步骤的我方与对手称谓同步切换。只使用当前回放已保存的信息，未记录的暗牌仍显示牌背；数据还原继续沿用原始记录视角，避免换视角改变手牌身份与计数。
- 回放换牌面板随当前公开局面的阶段显示，自动播放进入第一回合时立即收起；暂停、单步与跳转使用相同阶段判断。
- 回放顶部控制栏以回合下拉框替代时间显示：可选择开局或已有回合，立即暂停并还原该回合起点的公开局面；播放和单步前进时同步当前回合。
- 回放控制栏占满顶部生命面板与右侧状态区之间的可用宽度；步骤摘要悬停显示完整内容与步数，场中央不再重复显示步骤。白热化进度放在中央区域左半部，避免伸入左侧异能者编队。

- 菜单提供「拖动选择目标」开关，默认开启并保存在本机。按下或点击卡牌时不显示拖动热区；按住移动超过 8 像素后才显示拖动卡影与合法目标。有目标牌只高亮目标，拖到目标松开出牌，非法落点回弹，不显示取消按钮。无目标卡（含固定作用于所属角色的装备，以及自动选取攻击目标的战斗牌）可拖到整个战斗大区，包含双方备战区、前排和其间空白。关闭后，所有牌先拖到同一战斗大区松开，有目标卡再进入可取消的点选界面。
- 手牌瞬发虚线边框仅在我方本回合瞬发机会尚未使用时显示；虚线与内光保留卡牌类型原色，选中时也不改写边框颜色。机会用完后恢复普通可用牌边框，响应标记保留。
- 可出牌的整张手牌（含描述区域）优先响应拖动。普通手牌与起手换牌均不提供描述内滚动，超长内容截断，完整说明在大图中阅读。

### 信息与标记

- 牌桌战斗区域的角色卡不显示势力标签，势力信息保留在角色详情中。
- 待生效效果按生命周期显示：倒地会清除的效果标记放在我方异能者卡片下方、对方卡片上方；不依附角色存活的效果放在战斗区侧边。角色标记仅显示绿色向上箭头增益图标或红色向下箭头减益图标（每个效果一个）；悬停显示名称、具体效果、触发时机及存活条件，查看角色详情时展开完整说明。手机同样沿用我方下方、对方上方的位置。战斗区侧边标记继续显示名称和触发时机。「错误的门」已触发的追加创生不因伊洛伊倒地而取消，放在战斗区左侧；结算时原环合角色仍须存活。环合状态继续放在右侧。实时局面与公开回放共用后端标记投影。
- 战斗区状态显示在牌桌战斗区右侧，附带倒计时或「本回合」。浊燃与黯星在区域所属方回合开始结算，无前排时伤害玩家；开始效果与倒计时处理后才退回前排，再通常抽牌。
- 2026-09-20：前排退下瞬间存活且环合已满才留下空战斗区环合机会；备战区之后补满不追补。自动与主动返回均适用。白热化降上限先于回合开始退下；海月先获得本次战斗资源再返回。规则、预览与回放共用后端判定。
- 事件局面同步携带环合来源、可入场角色和战斗区高亮标记；倒地、离场、资源变化后随对应动画结算立即更新。环合来源角色倒地时，后端清除其最近出战记录；复活与自然恢复不恢复旧环合机会，个人环合值和能量仍保留。
- 异能、武备和延后效果的触发结果写入历史与公开回放。内容回调用 `EffectContext.history()` 记录已实际发生的公开效果，不能在合法性或预览检查中记录。
- 入弃牌堆事件不显示弹出提示、历史提示或回放步骤文案；事件与状态更新保留，弃牌堆数量和回放局面照常更新。

倒地结算在每个伤害阶段完成：攻守伤害 → 伤害后且倒地前的回血（白藏「招魂幡」）→ 完整倒地连锁 → 胜负。倒地触发的追加伤害不能遗留 0 血、倒计时 0 的角色至下一操作；正式 Python 与 native/CUDA 共用生成规则均覆盖该顺序。

### 武备时装（2026-09-20）

双方前排与备战区的角色立绘随当前武备切换，角色详情同步；手牌、构筑和图鉴仍使用默认图。08 优先居家服，07 按用户选择；缺图恢复默认，倒地卸装恢复默认，回放按当时的公开武备 ID 展示。公开与测试角色共用逻辑，不改变权限或规则。默认头像与立绘后续优先从 NTEData、Nanoka 核对，保留原有缩放与头部对齐；既有默认图未因来源约定改变而批量替换。映射和完整来源见 [角色图片资源说明](../../../docs/character-image-assets.md)。图片选择留在前端展示层，不修改卡池与模型规则哈希。

回归：`python3 -m unittest tests.test_duel_v2_costume_art tests.test_duel_v2_card_art`。

### 演出

- 普通抽牌与生成牌动画落到所属角色手牌组的末尾；该角色暂无手牌时按编队顺序插入。对手未公开抽牌的放大与入手动画均显示牌背；已公开的牌按公开牌面演出。
- 起手换牌沿用换牌区内的旧牌切出、新牌切入；换入牌不播放普通抽牌的放大飞入动画。
- 获得环合值或能量时只让卡片上变化的资源数字跳动，不播放资源飘字或字幕。
- 交战阶段只播放出击与反击动作，不提前显示伤害数值字幕。援护技免疫反击时，攻击者在原反击扣血时机飘出金色「无敌」；不把其他零伤害当作无敌。
- 衍生牌与临时复制成功加入手牌时，用公开 `gain.card` 先在视野中央显现牌面，再缩小飞入获得方手牌。
- 触发环合时，对战区中央播放一次对应名称的艺术字；随后“获得延滞／浊燃／黯星／覆纹／浸染”的状态附着事件只更新局面与历史，不再重复播放。独立触发的失谐仍正常演出；素材保存在模块 `static/images/harmony/`。专项验证：`python -m unittest tests.test_duel_v2_harmony_presentation`。
- 演出提示统一使用 `_notice(title, detail, position)`。回合、通用通知及角色状态提示放上方，出牌、抽牌等过程说明放下方。

### 字号与布局

战斗区中线显示一条半透明水平分界线，不拦截操作。备战区左右对称向中央收近：四人外侧槽位距边缘 15%、内侧 28%；双人教学使用左右各 28%，保持双方上下间距。
手机四人编队的外侧弧形偏移按卡片高度限制在各自半场，中线两侧各留至少 6px。备战角色高于中央透明容器，教程框选不改变原有命中层级，保证收近后仍可拖拽与点击终结。

底部手牌区不再显示手牌数量与常驻换牌提示行，牌列使用完整可用高度；起手换牌说明仍显示在专用换牌面板内。
小吱终结的金谷标记绘制在牌桌独立的半透明悬浮层，跟随对应手牌移动；不增加手牌区高度或内边距，不改变手牌高度，也不拦截点击与拖动。横向滚动、布局缩放和标记清除时同步更新。
双方手牌数量与牌库、弃牌数量一起显示在牌桌侧边，使用后端公开状态同步更新，回放沿用同一显示；对手只展示数量，不公开暗牌内容。

牌桌菜单提供全局字号五档：小（80%）、偏小（90%）、标准（100%，默认）、偏大（110%）、大（120%）；另提供「布局大小」0.8×、1×、1.5×、2× 四档。字号乘以布局比例，两个选项独立保存。回放共用此设置。

前排角色卡保持 3:4 比例，在槽位宽度与高度允许的范围内居中放大；窗口变高不再纵向拉伸。矮屏隐藏前排标题后，卡片仍占用可伸缩行。极矮窗口搭配过大的布局档位可能没有足够战斗区空间，应降低布局大小。

### 手机

- 横屏默认隐藏双方手牌区域，保留玩家生命面板，底部不再为手牌预留高度。点击牌桌空白处展开／收起我方悬浮手牌；展开会覆盖我方异能者，不挤压战斗区。拖动期间保持展开，合法松手后收起以便选目标，非法松手保留手牌。角色、其他按钮、牌堆与选择面板上的点击不切换手牌。起手换牌仍使用独立换牌面板；教学需要操作手牌时自动展开。
- 手机大图保留原侧边位置、宽度和左侧半尺寸关联卡，仅将高度延伸到屏幕上下各留 8px。主卡长内容可滚动，56×56px 关闭按钮固定在大图右上角。
- 大图和关联卡的攻击、生命／护盾固定在各自立绘底部两侧：攻击靠左，生命／护盾靠右，随立绘滚动，不与右上角关闭按钮争用空间。
- 手机悬浮手牌缩为至多 32% 屏高（上限 148px），卡宽至多 13vw。卡牌上的左右滑动交给浏览器滚动，明确向上移动才进入出牌拖拽；一次横滑不会途中转成出牌。单点继续查看大图，拖动期间手牌保持展开。
- 手机固定松手后二次选择：目标牌先拖入战斗大区，再点击目标；多选效果牌在松手后选择效果。菜单开关在手机上禁用，电脑端偏好保留。无目标牌合法松手后直接结算。
- 手机顶部信息栏也不保留整条背景和布局占位：对手生命、回合与菜单悬浮在战斗区上方，中央空白可直接操作牌桌；对手牌堆计数下移避开生命面板。
- 手机点击我方「手牌数」也可展开／收起手牌，按钮同步标记展开状态。悬浮手牌在右侧留出牌堆按钮空间，展开后仍能点击手牌数收起；拖动期间与二次选择期间不切换。
- V2 牌桌（含回放）在粗指针设备首次点击时尝试全屏并锁定横屏，每次页面加载至多一次；竖屏提示提供「尝试自动横屏」按钮。返回大厅不触发尝试。接口缺失或请求被拒绝时保留手动旋转提示，不阻塞对局操作；退出全屏或离开页面释放已取得的方向锁。大厅导航会切换文档，因此不能依赖大厅点击授权，需在牌桌内首次点击时尝试。
- 起手换牌顶部提供「收起换牌 / 展开换牌」。手机换牌卡显示效果摘要，完整说明在大图中阅读。
- 大厅、构筑、图鉴按 CSS 视口排版；手机横竖屏允许纵向滚动。图鉴角色导航横向滚动，卡片横屏四列、窄屏两列。
- PC 终结按钮为立绘顶部的扁按钮，位于护盾、攻击与生命数值上方。手机终结按钮放在角色卡右侧竖排，待生效效果以增益／减益小图标显示在我方角色下方、对方角色上方；环合与能量优先同一行，必要时缩写为「环」「能」。
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

根节点末端选点的离线消融见[根选点诊断协议](../../../docs/duel-v2-root-recommendation.md)。`scripts/check_duel_v2_root_selection.py` 从显式旧评估复制冻结源码／模型，以新种子比较原最大访问层、完整访问轮次与收缩价值推荐；默认网站动作与训练目标不变。验证：`.venv/bin/python -m unittest tests.test_duel_v2_root_recommendation tests.test_duel_v2_gumbel_search tests.test_duel_v2_learning_audits tests.test_duel_v2_serving_search`。

2026-10-07 起，高级人机采用逐操作请求：玩家动作立即返回自己的结果；后端不再在同一请求内计算整个人机回合。牌桌完成当前操作的事件演出后，调用 `POST /api/duel-v2/ai-step` 请求下一次人机操作，每次只推进一个正式引擎动作。响应 `room.ai_pending` 指示是否还有人机操作；刷新也能继续。该接口沿用房间、版本与请求编号校验，重复请求不重复执行，旧版本与旧房间请求拒绝。单个动作内部的卡效、响应和追加攻击仍按正式规则完整结算，不能拆开规则原子操作。

每个操作的搜索预算为至多 **3 秒、32 次模拟**，先到先停；公开即时胜利检查与后续树搜索共用预算。网站数值权重不变：支持的固定十人、跨队与公开联赛运行时复用信息集 Gumbel 搜索，以自己的可见观察和采样隐藏世界规划，不读取对手实际暗牌。模型同时作为未知玩家打法的观察策略代理，不代表预测到了玩家策略。可证明且不依赖隐藏响应、抽牌或随机性的即时胜利优先；其余执行搜索选择，超时仍使用已完成模拟的结果。无有效搜索或旧契约／不支持的选择阶段保留原合法网络动作。网页仍仅依赖NumPy，不加载Torch、checkpoint，不训练。

搜索接入不自动授予旧权重新规则兼容或质量批准；不匹配的模型仍在建房前拒绝。验证：`python -m unittest tests.test_duel_v2_ai_step tests.test_duel_v2_serving_search`；前端操作／演出顺序与重试：`node --test tests/js/duel_v2_ai_steps.test.cjs`。模型规则身份不匹配时保持拒绝，不修改模型清单绕过。

大厅选择「高级人机」，对手策略按实际高级人机阵容顺序取角色首字，当前显示「娜伊零九」「薄白零伊」「吱零薄九」；娜白零伊已按用户要求移除。三套保留策略原先已按用户明确要求启用 2026-09-21 五套固定配牌训练中的最新候选；真红队未接入。我方默认使用当前保存的构筑（仅公开角色），大厅不再提供我方构筑来源选项；保存的个人构筑不会被覆盖。仓库保留历史六人 `resident_public_v3` 模型，但其旧验证不适用于当前八人规则；新版 `official_public_v5` 默认要求验收通过；用户明确启用候选时，另存绑定模型、构筑与规则哈希的 `serving_authorization`，不改写 `validation.approved=false` 的质量结论。历史部署状态见当时交付报告。高级人机实际配牌由清单中的 `serving_build` 决定；旧客户端显式请求同构筑时仍复制这份实际配牌，不修改普通官方预组或玩家存档。房间模式为 `advanced`。起手策略按模型版本区分：旧 v2 模型保留全部起手牌，当前 v3 模型使用已训练的换牌评分；旧权重不会自动获得新能力。对手名及回放显示「高级人机」。

服务端仅依赖可选的 NumPy CPU 推理：`pip install -r requirements-ai.txt`。默认读取 `app/modules/card_game/engine/ai/models/{starter,weave-rush}.{npz,json}`，可用 `NTE_DUEL_AI_MODEL_DIR` 指定目录。数字权重不允许 pickle；加载时校验 SHA-256、卡池/座位顺序、观察 schema、规则哈希和参数形状。缺依赖、缺文件或版本不匹配会在创建房间前明确拒绝，不降级冒充高级人机。

2026-09-16 按用户指定将创生高级人机的 Z06 ×2 替换为 Z07「休息日」×2；这是人工换牌，不是重新训练或再次搜索的最优结论。「诓定解放」同步修复为回合结束立即回费，武备生命 5 → 6。

当前模型使用完整规则哈希及模型/构筑独立评估证据。`rule_identity` 保留训练规则身份；规则修复后以 `runtime_rule_hash` 绑定实际执行规则，并在新规则、实际构筑下重新取得独立评估证据才允许加载。历史验证保存在 `previous_validation`，旧胜率不能直接沿用为当前结果。

`engine/ai/advanced_model.py` 负责纯 CPU 编码与选择；`rl/public_schema.py` 和 `public_observation.py` 定义当前观察接口。`scripts/export_duel_v2_public.py` 导出数字模型，并可用 `--approve-report REPORT --serving-build BUILD` 验证模型与实际配牌的绑定。Git 的 `engine/ai/models/` 只保留两套当前数字策略及 JSON 清单；部署仍需用户验收授权。完整 checkpoint、优化器、对照样本和原始日志留在 `artifacts/`，不进入 app 或 Git。网站高级人机只加载数值导出文件，不加载训练 checkpoint，也不启动训练。

新训练默认学习起手换牌，并重点面对预组固定角色下的变化配牌；开局对照入口为 `scripts/analyze_duel_v2_opening.py`。综合新模型报告必须比较学习换牌与全留，具体运行与兼容边界见公开卡池训练文档第 10 节。

完整公开卡池训练入口见 [公开卡池训练](../../../docs/duel-v2-public-training.md)。导出模型须匹配完整评估后才允许网页使用；本轮模型、优选配牌及验收结果见 [综合训练交付报告](../../../artifacts/rl-evals/reports/duel-v2-comprehensive-release-20260916.md)。模型或构筑版本变化后，旧高级对局需要重新开局。

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

完整链路短测：`scripts/smoke_duel_v2_full.py`；报告打包：`scripts/package_duel_v2_report.py`。两套预组均支持固定角色候选配牌继续训练，并输出训练内先后手计数及冻结评估报告；运行和 Mac 交付约定见 [公开卡池训练](../../../docs/duel-v2-public-training.md#7-双预组全链路短测与报告交付)。

Windows 已完成一次有界 CUDA 出牌/配牌短训练，启动所需 MSVC 环境与可选快速编译档见 [短训练记录](../../../artifacts/rl-evals/reports/duel-v2-windows-short-training-20260915.md)。默认编译配置保持不变。

对局策略训练在原奖励基础上增加每局累计回报上限 10；失败仍扣 10，过程分保留，计数跨 rollout 并逐局重置。奖励口径和验证命令见 [公开卡池训练](../../../docs/duel-v2-public-training.md#对局奖励累计上限2026-09-15)。

候选训练以创生、覆纹快攻的固定四人阵容为重点，在阵容内变化配牌，保留少量任意公开队伍。综合流程先准备两套基础模型，再执行分级候选比较（每阵容 16／64／256 场）及适应训练；配置预览用 `scripts/smoke_duel_v2_full.py --check --profile comprehensive --seconds 21600`，不启动训练。细节见公开卡池训练文档第 9 节。

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
python3 -m unittest tests.test_duel_v2_zero_refund tests.test_duel_v2_serving_build tests.test_duel_v2_advanced_ai tests.test_duel_v2_api tests.test_duel_v2_replay
```

训练框架或编译规则：

```bash
python3 -m unittest tests.test_duel_v2_candidate_opponents tests.test_duel_v2_candidate_selection tests.test_duel_v2_card_tuning tests.test_duel_v2_card_tuning_cli tests.test_duel_v2_full_chain
python3 -m unittest tests.test_duel_v2_rl tests.test_duel_v2_rl_train tests.test_duel_v2_gpu_duel tests.test_duel_v2_rule_ir
```

完整 V2 测试：

```bash
python3 -m unittest discover -s tests -p 'test_duel_v2_*.py'
```

涉及真实交互或动画时，还需从 `/card-game` 通过页面进入牌桌，完成出击、换人、出牌、终结、结束回合、刷新恢复和查看日志。


### 2026-09-16 伊洛伊「想做什么梦？」倒地时点调整

牌面：己方其他异能者倒地后，造成 2 点伤害。前排优先。

这是用户指定的桌游规则调整。伊洛伊自身倒地不触发；同一伤害阶段的倒地角色先全部返回备战区、完成倒地清理并卸下武备，再通知存活异能者「己方异能者倒地后」。伊洛伊须存活且仍装备该武备。双方前排同时倒地时，存活的伊洛伊因此造成的伤害按空前排打到对方玩家，不受 A/B 座位顺序影响。仅对方异能者倒地不触发。


### 对局实体身份

V2 玩家、异能者和每张实体卡牌均为独立类实例，使用 `entity_id` 区分身份；同名角色和同名牌不能按模板 ID 判为同一个对象。JSON 存档读回时恢复实体，旧字段保留作接口兼容。牌桌优先提交后端给出的实体动作，旧服务响应才回退旧动作。动作的实体寻址、暗牌边界与迁移细则见 [实现契约](../../../docs/duel-v2-implementation-contract.md#2026-09-16-对局实体身份)。白藏「亦或是祝祷」只排除施法实体，终结只斩杀对方异能者。

### 局部配牌优化

综合训练在整套配牌学习之外追加 4 轮单张／双张替换搜索；单独入口为 `scripts/refine_duel_v2_cards.py`。冻结出牌模型、固定四名角色，按四场景平均胜率筛选，输出替换过程而不自动部署。验证：`python -m unittest tests.test_duel_v2_local_card_search tests.test_duel_v2_candidate_selection`（后者部分用例需要 PyTorch）。详见公开卡池训练说明第 11 节。

本轮追加训练、4 轮局部搜索与独立复核结果见 [2026-09-17 局部配牌报告](../../../artifacts/rl-evals/reports/duel-v2-local-cards-20260917.md)。

2026-09-17 深度训练的两套 v3 模型与最终配牌已写入项目高级人机，包含学习换牌和纯胜负奖励训练；详见 [最终交付报告](../../../artifacts/rl-evals/reports/duel-v2-deep-release-20260917.md)。生产部署等待用户验收。

固定覆纹/N06 枚举训练：`scripts/train_duel_v2_fixed_n06.py`；入口参数和候选比较范围见 `docs/duel-v2-public-training.md`。固定对手重置与枚举契约可用带 PyTorch 的隔离环境运行 `python -m unittest tests.test_duel_v2_fixed_opponent`，网站环境不新增训练依赖。

录像手牌计数以引擎事件 patch 为准，导出不再按抽牌/出牌/弃牌事件重复加减。中间态的隐藏占位同步至权威数量；私有换牌无法确定存留身份时隐藏不确定牌面，避免展示旧牌。已导出的旧文件需由原动作记录重新生成；回归命令：`python -m unittest tests.test_duel_v2_replay_hand_counts tests.test_duel_v2_replay`。

策略质量诊断入口：`scripts/diagnose_duel_v2_strategy.py`。N06 专项流程现包含共同基础、等更新候选、固定验证选权重、合法中盘采样和综合巩固；不修改纯胜负奖励。隔离 PyTorch 环境验证：`python -m unittest tests.test_duel_v2_strategy_quality tests.test_duel_v2_fixed_opponent tests.test_duel_v2_public_training`。运行说明与训练/验证/留出种子边界见 `docs/duel-v2-public-training.md`。


2026-09-18 新增仅测试账号可见的小吱与海月及 16 张卡。内容包位于 `content/duel_v2/characters/xiaozhi.py`、`haiyue.py`；通用出击支付、备战区攻击、回合结束响应由纯规则引擎扩展。金谷在攻击上方以黄色金币显示，支持负数及公开回放。验证：`python3 -m unittest tests.test_duel_v2_xiaozhi_haiyue`。公开六人权限、预组与训练编码不扩展；具体边界和来源见完整手册新增节。

本轮规则、网页与高级人机兼容验证见 [小吱与海月接入验收](../../../docs/duel-v2-test-characters-20260918.md)。

### 2026-09-18 本地卡效更新

本地服务同步 Y07 治疗重做、N04 创生加成、N05 指定对方异能者追击、Y04 瞬发及薄荷终结文案。保留其他测试角色的本地修改。新规则下旧高级人机模型不绕过身份校验，待完整训练验收后成套更新。规则回归：`python -m unittest tests.test_duel_v2_september18`。

2026-09-18：高级人机大厅仅选择对手预组，我方固定使用当前保存构筑。`/api/duel-v2/start` 省略 `ai_player_deck` 时默认 `saved`；显式 `mirror` 仅保留旧客户端兼容。验证默认行为：`tests.test_duel_v2_advanced_ai.AdvancedRoomTest.test_advanced_uses_saved_public_deck_not_bot_preset`。

2026-09-18 默认预组同步本地测试账号配牌：创生、覆纹快攻、盈蓄；新增仅测试账号可见的快攻预组（小吱、零、薄荷、海月）。简单人机从公开默认预组随机选择配牌，账号已有构筑与高级人机的独立模型配牌不覆盖。

用户补充：简单人机每次新建对局从公开默认预组等概率随机选择；当前为创生和覆纹快攻。即使操作者是测试账号，也不选择含测试角色的预组。所有实际牌表仍从同一默认预组目录读取。

### 新角色手牌立绘自动绑定

前端资源表自动接收 catalog、对局公开状态及回放开局中的角色 `id/name/attribute/avatar/portrait`。手牌、换牌、公开对方手牌和悬浮大图均按卡牌 `character_id` 查询所属角色立绘；新增角色只需在内容目录提供有效头像、立绘路径，不再要求手工补充 JavaScript 角色表。旧静态映射仅用于兼容缺少资源字段的历史数据；不缓存角色生命等局内状态，隐藏手牌仍为牌背。

回归：`python -m unittest tests.test_duel_v2_card_art`，覆盖未预登记的新角色、直接打开牌桌、回放、局部更新、旧资源覆盖与隐藏手牌。

2026-09-18 名称更正：公开薄荷、白藏、零、伊洛伊队统一显示为“覆纹预组”，内部ID仍为 weave-rush；原 sync-curse 残虹测试队显示为“覆纹预组（测试）”以避免重名。仅更改名称，不调整卡牌、个人存档或训练中的冻结构筑；历史报告保留原名称。


2026-09-18 金谷标注：小吱倒地保留金谷，战斗牌使用 `attack_mode=set` 设置本次攻/反击；终结将实体手牌标注保存在 `jingu_mark`，牌顶外侧显示，结算后兑现。`private_hand` 事件只投影给手牌拥有者，回放沿用同一隐私边界。所有角色使用 `ultimate_used_turn` 限制同回合重复终结，倒地/复活不重置；训练行同步为 `ch_ultimate_used_turn`。回归入口：`tests.test_duel_v2_jingu_marks`。

X选一通过卡牌 `play_options` 和动作 `option_id` 传递；拖拽战斗大区从左到右选择，资源不足区禁用。大图左侧半尺寸展示「」中引用的可见卡牌，启动牌桌时加载卡池用于解析引用。验证包含 `tests.test_duel_v2_jingu_marks`、`tests.test_duel_v2_card_art`。

### 2026-09-18 浔重做

浔改为 3/6、时计／荒时终结与七张战术、一张武备；详情见完整手册第 7 节。`engine/duel_v2/temporal.py` 管理至多两份己方回合起点快照，`summons.py` 管理前排召唤物。快照和牌库检视不进入公开回放；回溯保留网络版本和移除账本。前端支持终结二次选择、时计、势力、生命遮罩、顺序牌库和回溯黑场。

`rl/backend.py` 过滤并拒绝标记为 `training_excluded` 的回溯牌，普通人机同样不选择此牌；本次不训练。回归入口：`python3 -m unittest tests.test_duel_v2_xun_rework`。旧浔对局应重新开局。

### 2026-09-18 埃德嘉

新增仅测试账号可见的埃德嘉与 E01–E08。角色包 `content/duel_v2/characters/edgar.py` 管理真理之匙、治疗、武备与卡效。通用引擎增加跨回合盈蓄事实、无牌可抽胜负替换、治疗来源与实际回复回调、手牌多选调度、可暂停恢复的通常抽牌替换。牌桌支持真理之匙显示和至多三张的调度确认。

验证：`python3 -m unittest tests.test_duel_v2_edgar`，并回归浔、回放、实体、抽牌、区域状态与现有角色。公开六人、预组及高级人机训练范围不变。

埃德嘉本轮验证与交付边界见 [接入验收](../../../docs/duel-v2-edgar-20260918.md)。


### 2026-09-18 快攻预组开放与第三高级策略

用户授权公开小吱、海月及 quick-rush「快攻预组」（小吱、零、薄荷、海月）。当前公开八人、三套预组：创生、覆纹、快攻。普通账号可查看、保存和使用快攻；简单人机继续从公开预组等概率随机，因此同步包含快攻。未改动八张卡的效果、既有个人构筑或其余测试角色权限。

高级人机登记第三策略 quick-rush，大厅标为“待训练”；服务端明确拒绝其开局，直到八人观察编码、两名角色的训练后端、独立策略训练及验收全部完成。不能用旧六人模型或普通规则策略冒充高级人机。正在 Windows 运行的两套策略90分钟分支实验保持冻结；三套联赛属于后续实验，不中途改写对照组。


### 八人三策略训练

公开快攻的正式 Python 训练／NumPy 推理使用独立 `official_public_v5` 契约，旧六人 CUDA 合约保留。三套共用90分钟预算、真实终局分支学习、阶段择优与独立验收，详见 [三策略流程](../../../artifacts/rl-evals/reports/duel-v2-three-strategy-training-20260918.md)。验证：`python -m unittest tests.test_duel_v2_league`。八人模型未验收前仍拒绝高级人机开局，普通人机可使用快攻。

逐操作对照入口 `scripts/analyze_duel_v2_league_decision.py`：以指定快照和冻结八人模型作有界终局采样，分别展示动作概率与模拟胜率区间；只用于离线复盘，不把校准未知的价值头直接展示为胜率。


### 2026-09-18 攻击预览隐藏信息修复

攻击预览仅按可见信息计算：对方未公开手牌使用匿名占位，保留公开数量，不预测隐藏响应；已公开手牌仍可参与响应预览。未知牌库与对方私有金谷标注不参与预测。普通规则人机使用同一预览，因此同时消除其通过预览间接获取隐藏响应牌的路径。实际出牌与响应结算不变。验证：`tests.test_duel_v2_preview_privacy`。

已启动的Windows八人训练保持冻结源码，神经模型输入使用无预览投影，不受上述路径影响；最终70种组合覆盖在评估恢复阶段改用先匿名化对方未公开手牌的规则策略，未沿用泄漏路径。后续启用模型仍需匹配修复后的规则身份及独立验收，不改写旧训练身份。


### 下一轮：错误优先分支学习

训练入口默认使用 `terminal_tiebreak_v3`：普通PPO每批仅一步，累计3步有效分支才允许1步普通优化；对模拟明显更好但策略仍选错的局面进行有界优先回放，按证据解除错误旧策略约束。相同粒子胜负下以弱排序鼓励更早获胜、更晚失败，并保守认证一步斩杀；终局奖励保持不变。见 [实现与验证](../../../docs/duel-v2-prioritized-branch-learning.md)。本次仅提交代码和单元验证，未重训或替换服务权重。

30 分钟三策略短测使用相同入口 `--seconds 1800`：六个训练阶段各至多 190 秒，预留 600 秒验收及 60 秒启动/收尾；总绝对截止和进程树监督仍生效，不自动发布。小于 600 秒仅允许 `--smoke`。


### 独立搜索纠错小实验

`scripts/test_duel_v2_search_pilot.py --starter SOURCE.pt --weave SOURCE.pt --output NEW_DIR --seconds 600` 在本机 CPU 执行最多 10 分钟的小实验，不替换联赛默认入口或服务模型。`rl/league_search.py` 提供全根动作的两步公开信息斩杀证明、8 粒子筛选后独立 32 粒子复核及全合法动作蒸馏；`rl/search_pilot.py` 生成分离的合成训练/选择/确认/留出局面。牌组数量保持合法，但这些局面并不声称由自然对局产生。验证：`python -m unittest tests.test_duel_v2_league_search tests.test_duel_v2_league_learning tests.test_duel_v2_league`。Torch 只在训练和梯度测试时加载。


### 完整对局搜索增强自博弈短训

`scripts/train_duel_v2_information_search.py --starter SOURCE.pt --weave SOURCE.pt --output NEW_DIR --seconds 600 --cycles 2 --simulations 24 --workers 4` 是独立本机 CPU 短训入口。双方每个实际决策（包括双方开局调度）执行信息集搜索，整局结束后联合拟合搜索访问分布和本方终局胜负，不使用斩杀题或 PPO。`rl/information_search.py` 实现搜索与样本契约，`rl/information_experiment.py` 执行两轮自博弈、回放训练及独立对照。搜索内部对手用其可见信息策略；这是近似信息集搜索，非完整 ReBeL。当前实际根支持公开三预组的调度和普通行动；未来含私有检视的根需先扩展信念抽样，不允许偷看或悄悄降级。模拟中的选择按行动方可见信息处理。验证：`python -m unittest tests.test_duel_v2_information_search tests.test_duel_v2_league tests.test_duel_v2_league_learning tests.test_duel_v2_league_search`。默认联赛入口和线上纯网络策略未替换。


### 冻结模型搜索预算对照

`scripts/evaluate_duel_v2_search_budgets.py --learner MODEL_DIR --opponent FROZEN_OPPONENT_DIR --output NEW_DIR --seconds 1200 --seeds-per-pair 2 --repeats 3 --workers 4` 比较纯网络、24、128、256 档。只读取并复制三套数值模型，不加载优化器或更新权重。固定局面来自新种子自然完整对局、既有战术回归及可选 `--user-snapshot`；整局按同一模型、对手、开局种子和先后手配对。记录动作、实际模拟次数、耗时、随机种子稳定性与独立斩杀审计；审计只记分不改选动作。输出前后校验冻结模型文件哈希，截断不得算输/平或稳定通过。时延包含编码和推理/搜索，不含模型加载、规则提交与审计；四进程并发数据不是线上单请求 SLA。验证：`python -m unittest tests.test_duel_v2_search_budgets tests.test_duel_v2_information_search`。


预算对照使用牌数一致的战术夹具 v2：仅对人工构造的旧夹具补齐每角色 8 张、同名至多 2 张的牌库记录，保留测试手牌和角色/玩家场面，缺失的已生成家族牌记入弃牌；保留前后数量说明。自然对局和用户录像不做此修复。所有固定局面在排队前必须通过信念抽样预检，历史夹具的失败记录不改写。耗时主要比较同一批自然局面，避免不同夹具或补测负载混淆。


### 明确日期截止的长期搜索训练

`scripts/train_duel_v2_until.py` 使用带时区的 `--train-until`、`--evaluate-until`、`--hard-deadline`，分别限制学习、计算验收与整棵进程树。初始与固定参考模型由 `--initial` / `--reference` 指定数值目录；支持 CUDA 更新、CPU 多进程自博弈、每轮恢复点、定期独立检查点选择和限时最终报告。Windows 监督进程脱离 SSH 作业，运行期间阻止自动系统休眠，退出恢复；可恢复故障最多自动重试一次，沿用原截止。详见 [2026-09-19 中午验收计划](../../../artifacts/rl-evals/reports/duel-v2-noon-plan-20260919.md)。验证：`python -m unittest tests.test_duel_v2_information_longrun`。此入口不会自动上线模型。


已停止的长期任务可用 `scripts/finalize_duel_v2_until.py --source-root FROZEN_SOURCE --output EXISTING_RUN` 从已校验的完整检查点继续**只评估**。调用前须确认原工作进程树和监督器已退出并保存原guard；该入口拒绝仍运行或重复启动，固定已完成轮次、保留原验收与硬截止，完成后验证训练计数没有改变。它不修改冻结规则源码，不用于重新开始学习。


### 2026-09-19 中午训练模型接入

按用户“更新项目的模型吧”明确要求，三套高级人机使用第 28 轮 `latest` 数值权重（含快攻），采用纯网络 CPU 推理与已学习换牌，不在网页执行 256 次搜索。此前“待训练”和“未启用”描述是历史状态。配牌取模型内的 `build`；个人构筑与普通预组不变。质量验收整体未提升，`validation.approved` 仍为 false；仅本次指定权重通过独立 `serving_authorization` 启用，所有身份与数值校验仍强制执行。旧版本高级房间需要重新开局。详情与回退方式见 [模型更新记录](../../../artifacts/rl-evals/reports/duel-v2-noon-model-update-20260919.md)。

验证：`python -m unittest tests.test_duel_v2_league tests.test_duel_v2_league_service tests.test_duel_v2_advanced_ai.CpuEncoderTest tests.test_duel_v2_advanced_ai.AdvancedRoomTest.test_real_models_finish_all_presets_and_save_replays tests.test_duel_v2_advanced_ai.AdvancedRoomTest.test_published_models_support_public_saved_builds`，覆盖未授权拒绝、授权绑定、质量状态保留、三套实际模型 API 完整对局及回放。


### 小吱加强后快攻适应

`train_duel_v2_quick_adaptation.py`在显式新规则热启动后仅更新快攻，创生与覆纹冻结；采用原绝对截止监督，保留初始/最佳/最后及详细角色贡献事件。规则迁移不继承旧批准，结果不自动上线。预算、配对与归因边界见 [快攻适应计划](../../../artifacts/rl-evals/reports/duel-v2-quick-adaptation-20260919.md)。验证：`python -m unittest tests.test_duel_v2_quick_adaptation`（需独立Torch环境）。

### 搜索待选择局面

`information_search`支持在检视、弃牌和已查看对方手牌等待选择根局面继续搜索，保留选择方可见候选，未知信息仍按信念采样重建。可用`failure_dir`在搜索失败时记录私有复现局面，仅留离线产物。修复及续训证据见[选择搜索修复](../../../docs/duel-v2-choice-search-fix-20260920.md)。验证：`python -m unittest tests.test_duel_v2_information_search tests.test_duel_v2_information_longrun`；真实变化配牌回归独立于优化训练统计。


### 2026-09-20 哈尼娅

新增仅测试账号可见的哈尼娅及 V01–V08。角色包 `content/duel_v2/characters/haniya.py` 管理主角光环、终结、五张战术和三张武备；通用规则增加本局黯星计数、出击结算前团队回调、攻击命中玩家后回调及指定角色本回合普通出击免行动力。蓝色音符在金谷同位置显示，公开回放同步资源；免费出击效果具有本回合标记。规则与来源见完整手册哈尼娅节。

验证：`python3 -m unittest tests.test_duel_v2_haniya tests.test_duel_v2_catalog tests.test_duel_v2_xiaozhi_haiyue tests.test_duel_v2_jingu_marks tests.test_duel_v2_preview_privacy tests.test_duel_v2_zone_status`。正式 Python 引擎接入，不扩展独立训练执行器及公开高级人机角色范围。

哈尼娅本轮自动化、真实页面与交付边界见 [接入验收](../../../docs/duel-v2-haniya-20260920.md)。

### 2026-09-20 小吱预组

默认 quick-rush 更名为「小吱预组」，阵容为小吱、零、薄荷、九原；九原采用创生预组现有 8 张配牌。新赠送预组与简单人机随目录更新，已保存的个人构筑及高级人机模型配牌不回写。具体配牌见完整手册。验证：`python3 -m unittest tests.test_duel_v2_catalog tests.test_duel_v2_api`。

2026-09-20：J01「知晓与制衡」新增瞬发，检视效果不变。按用户后续确认，高级人机阵容与普通赠送预组独立，以模型清单的合法构筑及构筑哈希为准；小吱普通预组换九原不改变高级人机的海月队。三套现有模型允许兼容当前规则，`runtime_compatibility` 记录用户认可的运行规则、原训练规则、权重及构筑绑定；不修改权重、配牌或历史质量验收。结构、编码、数值和构筑完整性仍校验。验证：`python3 -m unittest tests.test_duel_v2_league tests.test_duel_v2_advanced_ai`。

### 三套配牌的确认与交付

后续完整训练使用`python scripts/train_duel_v2_full_cycle.py --output NEW_DIR`检查计划，显式`--run`才执行；默认不启动训练。新流程规范化构筑、保留独立种子账本、等量适应原/新配牌，并在确认、审计与战术门槛失败时返回原配牌和原权重。流程完成与质量批准分别记录。旧双预组入口只用于历史复现，旧夜间脚本不作为新协议交付入口。详见[配牌确认协议](../../../docs/duel-v2-build-acceptance.md)，验证`python -m unittest tests.test_duel_v2_build_acceptance`。2026-09-20配牌已按用户要求全部废弃，数值检查点保留研究但禁止默认复用。

### 2026-09-20 中速高级人机登记

新增 `midrange`「中速」，固定顺序为零、伊洛伊、娜娜莉、白藏。大厅显示待训练状态及阵容；服务端在创建房间前明确拒绝未训练策略，不使用普通规则策略或其他模型代替。这里只登记阵容，不指定最终配牌，也不改变普通预组与既有三套模型的规则身份。

用户随后要求“先不用安排”，本次未创建定时任务、未启动训练。现有完整训练协议仍为三策略；四策略初始化、配牌搜索、对手矩阵与独立验收待下一轮训练准备时接入，登记不代表可直接沿用旧入口训练。

验证：新增中速拒绝开局、未知策略拒绝及缺模型无房间副作用三项专项测试通过；真实浏览器确认阵容提示与开局拒绝。`.venv/bin/python -m unittest tests.test_duel_v2_advanced_ai` 共 20 项，13 通过、7 项因既有编码未就绪或模型规则身份不匹配失败；恢复本次修改前的 `advanced_model.py` 对照复跑，同七项仍失败（新中速测试因基线无登记另失败），未将全套回归标为通过。JavaScript 语法检查通过。

### 2026-09-20 高级人机名称

高级人机名称改为实际阵容按顺序取角色首字：starter「娜伊零九」、weave-rush「薄白零伊」、quick-rush「吱零薄海」、midrange「零伊娜白（待训练）」。快攻高级人机仍使用海月，因此取“海”，不随普通小吱预组的九原改变。内部策略 ID、权重、配牌、普通预组名称和训练状态不变。

用户后续指定小吱简称取“吱”；零伊娜白排第一但暂时隐藏并禁用，默认选择首个可见策略娜伊零九。训练状态不变。


### 2026-09-20 延滞与盈蓄角色重做

真红 R01–R08 重做并增加终结衍生 RF01；翳作为测试角色新增 I01–I08。`engine/duel_v2/deferred.py` 负责原操作完成后的可序列化追加动作，`state.effective_card` 负责手牌临时视为。角色内容继续留在各自角色包。能量资格统一用于资源获取与白热化候选；倒地响应、自然到期和离场结束分开处理。延滞在施加方回合开始递减，首回合减攻窗口独立于剩余时间。

网页与回放显示翳的黄色兽牙倒计时、威慑凝视武备行和转换手牌；完整来源及边界见规则手册。本轮不训练、发布或重新授权历史高级人机。旧真红对局需要重新开局。

专项验证：`.venv/bin/python -m unittest tests.test_duel_v2_surplus_rework`；同时回归目录、通用卡牌执行、区域状态、实体、响应、预览隐私、回放和 API。验收记录见 [延滞与盈蓄重做验收](../../../docs/duel-v2-surplus-rework-20260920.md)。


### 2026-09-20 真红预组与指定普通人机

新增 `zhenhong`「真红预组」（真红、零、伊洛伊、翳），仅测试账号可见，旧盈蓄预组移除，由真红预组替代。大厅普通人机可选择账号可见的对手预组；`/api/duel-v2/start` 在 `mode=solo` 时接受 `ai_deck`，省略或 `random` 仍随机公开预组，其他值按可见预组 ID 校验。所选套牌由规则策略执行，刷新保留对手名字，切换对手须先离开当前房间。

验证：`.venv/bin/python -m unittest tests.test_duel_v2_catalog tests.test_duel_v2_api`，覆盖预组完整性、自动赠送、普通账号拒绝测试人机、指定开局及恢复。既有高级人机待训练文案断言不属于本次改动。

验收结果：目录、API 与回放共 52 项中 51 项通过，仅既有高级人机待训练提示断言失败；新增预组权限、指定对手、同房恢复及换对手冲突检查通过。规则人机镜像在 39 个动作、总第 10 回合以玩家生命胜负正常结束（seed 73，不作为平衡证据）。真实浏览器在临时数据库房间 B256B6 中确认对手名「真红预组」及真红／零／伊洛伊／翳阵容，无 console error。

旧盈蓄预组迁移由 `engine/application/v2_preset_migrations.py` 处理，只迁移有赠送记录且仍与原官方牌表完全一致的构筑；保留改过的个人构筑与进行中对局。验证：`tests.test_duel_v2_preset_migrations`。


### 2026-09-20 人机随机与镜像选择

大厅普通人机和高级人机的对手套牌均默认选择「随机」，仍可指定具体预组。普通随机延续公开预组等概率抽取，测试账号也不随机到测试预组；指定预组按账号可见权限校验。高级随机从已登记、非待训练且通过现有模型加载校验的策略中等概率抽取，保存实际策略 ID 与版本；无可用模型时明确拒绝，不降级为普通人机。

普通人机新增「镜像对局」：复制玩家当前出战构筑的四人顺序与全部 32 张配牌，双方独立洗牌与实体化。服务端在建房前按卡池 `training_excluded` 字段检查；含任意标记牌则列出牌名并要求调整构筑，不自动删牌、补牌或修改已保存构筑。测试角色仍按玩家权限使用，没有该标记牌的合法测试构筑可以镜像。

随机与镜像仅在新建对局时解析；同模式重复请求（省略对手或选择随机）恢复已有对局，不重新随机、不重新复制玩家后来修改的构筑。指定另一套牌须先离开当前对局。刷新与回放保留镜像对手名。

验证：`python -m unittest tests.test_duel_v2_opponent_selection`；覆盖随机候选池、默认与显式随机、镜像精确配牌、通用禁用标记、测试角色、刷新恢复、换对手冲突及无模型时无房间副作用。

验收结果：7 项专项测试全部通过，JavaScript 语法检查通过。扩展接口／普通预组／高级模型回归为 64 项、55 通过、9 项失败；用修改前 `v2_service.py` 对照运行原有 59 项，复现同 9 项失败（旧公开预组与待训练断言、旧编码能力及现有模型加载失败），无新增失败。真实页面确认两种模式默认随机，普通镜像能开局、完成换牌并刷新恢复；高级随机在现有模型均加载失败时显示无可用对手，不创建房间。开发服务继续监听 `0.0.0.0:5001`，本机与当前局域网 `10.87.24.100:5001` 均返回 200。


### 2026-09-20 九原部分重做

枚约附着从 J08 移至九原异能，J05 改为瞬发直接伤害玩家，J06 按本局创生命中对方玩家的段数连续造成伤害。`state.py` 保存包含护盾吸收的公开次数，`combat.py` 记录本回合创生，角色包管理卡效与动态描述。J06 局内末尾显示「（X为N）」，计数随快照、公开投影及回放 patch 保存；新牌效请重新开局。J02/J03 与预组未修改。

专项：`python3 -m unittest tests.test_duel_v2_jiuyuan_rework`。本轮仅验证正式 Python 引擎与页面，不训练、不更新模型兼容授权或旧独立编译执行器。

验收：14 项九原专项、74 项相关回归及 4 项既有九原测试通过（共 92 项）。独立临时数据库的真实页面从大厅进入房间，确认两段创生使 X 从 0 到 2、刷新保留、J05 免费造成 3 点、J06 造成两段各 1 点，历史携带「（X为2）」且浏览器无 error。常驻服务已重启载入新卡效，本机与当前局域网 `192.168.10.5:5001` 均返回 200。上述是功能验收，不是平衡结论。

### 五套固定配牌策略训练（2026-09-20）

`scripts/train_duel_v2_fixed_lineups.py` 默认检查计划，显式 `--run` 启动 90 分钟有界离线训练。训练创生、覆纹、小吱／零／薄荷／九原、真红／零／伊洛伊／翳、娜娜莉／白藏／零／伊洛伊；旧三套高级模型冻结陪练。`rl/fixed_lineup.py` 为真红、翳和九原新计数提供独立十人观察契约；`fixed_lineup_training.py` 在正式规则引擎执行完整终局 PPO 与独立评估。固定 32 张配牌，不搜索换牌，不自动启用模型。来源、时间预算、验证及局限见 [五套训练说明](../../../artifacts/rl-evals/reports/duel-v2-five-fixed-training-20260920.md)。专项：`python -m unittest tests.test_duel_v2_fixed_lineups`。

### 2026-09-21 四套模型更新

按用户“除了真红队，先更新进仓库模型吧”，更新 starter、weave-rush、quick-rush 并新增 midrange。四套使用 `fixed_ten_v1` 数值权重，`engine/ai/fixed_model.py` 单独校验启用授权与规则／构筑／权重哈希；训练模块和观察编码未改，不需要重写原训练身份。保留 `validation.approved=false`，启用来自用户明确授权，不等于全量质量门槛通过。网页仍只允许公开八人的个人构筑。

小吱高级人机为小吱／零／薄荷／九原；中速为娜娜莉／白藏／零／伊洛伊，显示「娜白零伊」且参与随机对手选择。使用已学习起手换牌，NumPy CPU 推理不加载 Torch。旧高级对局应重新开局；已保存个人构筑与普通预组不改。权重和评估来源见 [训练与接入记录](../../../artifacts/rl-evals/reports/duel-v2-five-fixed-training-20260920.md)。验证：`python -m unittest tests.test_duel_v2_fixed_serving tests.test_duel_v2_fixed_lineups tests.test_duel_v2_opponent_selection`。

### 2026-09-21 翳加强与零行动力边框

I01／I02 兽牙改为 4 个己方回合，I03 瞬发且为 6 个己方回合，I04 改为不消耗行动力且为 4 个己方回合。通用费用结算避免零费用牌因被赋予瞬发而占用瞬发机会。公开牌面新增 `action_point_free`，由后端判断不依赖瞬发机会的零行动力费用；前端可用手牌绘制卡类原色的低密度虚线，并同步回放牌面。无关牌效与训练权重不改。验证：`python -m unittest tests.test_duel_v2_surplus_rework tests.test_duel_v2_xun_rework tests.test_duel_v2_jiuyuan_rework`、`node --test tests/js/duel_v2_instant_border.test.cjs`。

本轮上述 61 项规则测试与 2 项 JS 边框测试通过；独立浏览器验收页使用真实后端手牌投影和现有渲染函数，确认普通瞬发／零行动力两种密度及瞬发耗尽后的区别。16 项旧演出回归有 3 失败、1 错误，用修改前 flow／presentation 重跑复现同项；不将该组声称为全通过。本轮改变全局规则指纹，已有高级模型按严格身份校验暂拒绝加载，未未经确认重绑模型。

2026-09-21 群犬吠形纠正：I08 仅在对方战斗区有延滞时使本牌获得瞬发，无需已装备；不再给翳其他手牌授予瞬发。专项覆盖未装备时零行动力打出、无延滞／瞬发已用时付费、装备后其他牌不获得瞬发。验证：`python -m unittest tests.test_duel_v2_surplus_rework`。

### 2026-09-21 翳异能加强（已回退）

用户后续要求已回退为：攻击延长己方现存延滞 +1 个己方回合，至多 2 个己方回合；已有更长延滞不缩短。下述加强与验收仅保留为历史记录。现行验证仍使用下述三组测试。

翳攻击使己方施加的现存延滞延长 2 个己方回合，上限 3 个己方回合；不缩短原本超过 3 回合的状态，不重开首回合减攻窗口。角色包、图鉴与角色详情异能文案同步，历史记录实际延长后的回合数。数值与来源边界见规则手册。

验证：`python -m unittest tests.test_duel_v2_surplus_rework tests.test_duel_v2_zone_status tests.test_duel_v2_harmony_presentation`，覆盖 1／2／3／4 回合、归属、无状态、己方倒计时、兽牙暂停、存档与公开回放。

本轮 46 项相关测试通过。隔离数据库的真实牌桌从大厅进入后，确认翳详情显示“延长 2、至多 3”，拖入翳出击使延滞倒计时从 1 变为 3，历史记录延长结果且浏览器无 error。仅功能验收，不代表平衡结论；未部署。

### 2026-09-21 梦的边缘前排限制移除

R07「梦的边缘」在己方触发盈蓄时造成全体对方异能者 2 点伤害，真红在备战区也能触发；存活、装备与同次盈蓄只触发一次的要求保留。验证：`python -m unittest tests.test_duel_v2_r07 tests.test_duel_v2_surplus_rework.SurplusReworkTest.test_surplus_weapon_triggers_in_front_and_on_bench`，覆盖前排／备战、九原重复、倒地、未装备和公开回放局面。


### 2026-09-21 护盾与创生计数修复

正式 Python 引擎改为累加新获得的护盾；九原 J06 只累计实际扣除对方玩家生命的创生段数。图鉴机制词表与规则手册同步，既有到期、返回和倒地清盾规则不变。旧存档已有计数不追溯重算。独立旧训练执行器与模型权重不在本次修改范围，不作为新规则验收依据。

验证：`.venv/bin/python -m unittest tests.test_duel_v2_shield_stacking tests.test_duel_v2_jiuyuan_rework tests.test_duel_v2_xiaozhi_haiyue tests.test_duel_v2_turn_boundaries`。

本次相关回归 76 项通过，浏览器已验证图鉴词表展示与搜索。扩大回归共 158 项，另发现既有倒地连锁用例失败（在内存恢复本次两项旧规则后仍复现），以及 hooks 角色注册表包含教学角色的断言失败；未将其计为本次通过。

### 2026-09-21 真红加强与免伤外框

真红盈蓄异能保留额外出击，新增至下个己方回合开始的免伤；R02／R03／R04 在盈蓄条件下获得瞬发并抽对应类型牌，RF01 基础抽 1 张。终结新增移入战斗区；通用入场环合仅在入场者自己的回合触发，含响应与移动效果。后端统一保存免伤到期时点和公开 `damage_immune`；前排、备战区与回放按该字段绘制金色盾形外框，不在战斗区外显期限文字。详细边界见完整手册。

验证：`python -m unittest tests.test_duel_v2_zhenhong_buff tests.test_duel_v2_surplus_rework`；前端：`node --test tests/js/duel_v2_immunity_frame.test.cjs tests/js/duel_v2_instant_border.test.cjs`。本轮不训练、不调整模型权重或启用授权；独立旧编译执行器未同步此轮规则。验收见 [真红加强记录](../../../docs/duel-v2-zhenhong-buff-20260921.md)。新牌效请重新开局。

### 2026-09-21 翳终结兽牙累加

终结直接将兽牙剩余时间 +1，没有时从 0 变为 1；后半段附着物增伤与两回合期限不变。专项验证：`python -m unittest tests.test_duel_v2_yi_ultimate`。规则来源为用户桌游口径，不更新模型权重或授权。

### 2026-09-21 娜娜莉追击目标修复

战斗追击使用该次攻击已锁定的目标，原异能者倒地或离开战斗区后跳过，不再重新按前排优先转打玩家。N03、终结和 N08 强化的战斗追击共用此规则；N05 按牌面指定目标不变。图鉴追击词条和规则文档同步。

验证：`tests.test_duel_v2_followup_target` 覆盖击倒、离场、换前排、护盾吸收、原玩家目标与未攻击；连同既有娜娜莉、九原、护盾、预览、触发历史和回放回归，共 50 项通过。浏览器已确认图鉴新词条。未修改独立训练执行器或模型授权。


### 2026-09-21 真红视为牌外观

后端公开卡牌可携带仅用于展示的 `hand_face`，保留原牌名称、描述、卡框、面板和排序；顶层仍为实际结算牌，费用、合法动作与真红立绘沿用它。隐藏手牌不带原牌外观。实时手牌、入手演出与拥有者回放复用此字段，真红大图继续由描述中的卡名生成关联小图。

验证：`python -m unittest tests.test_duel_v2_hand_face tests.test_duel_v2_surplus_rework tests.test_duel_v2_card_art`，`node --test tests/js/duel_v2_hand_face.test.cjs`。

本次 57 项 Python 回归与 34 项前端测试通过；包含主动换下真红只结束终结、不触发到期额外攻击的回归。浏览器使用正式牌桌模板与渲染器的隔离局面，已确认原牌外观、真红背景，以及角色大图旁「升腾之赤」的 +4/+4 关联小图。RF01 费用 1，保留抽 1 张牌，移除盈蓄额外攻击加成；参数来自用户桌游规则。


### 2026-09-21 一报还一报抽牌隐私

Y02 采用私有抽牌事件，使用者看具体牌面，对手与公开回放只看牌背和被选异能者名称。`draw_owned_card` 的其他调用保持原行为。验证：`python -m unittest tests.test_duel_v2_iloy_draw_privacy` 两项通过，覆盖双方座位、公开事件隐私和无匹配牌；回放与当前手牌展示回归 8 项、动画 31 项通过。

### 2026-09-21 起手逐槽换牌

`v2_table/sync.js` 将本机确认后的起手换牌与对应私有抽牌事件组成逐槽演出；`helpers.js` 提供逐张切出、原位替换、切入与取消检查。全部完成后才应用显示快照并移入排序后的手牌，后续回合事件继续按原顺序播放。尺寸保留小数，避免手机缩放导致槽位漂移。

验证：`node --test tests/js/duel_v2_table_state.test.cjs tests/js/duel_v2_animation.test.cjs`，45 项通过。隔离数据库的真实大厅／牌桌在 1280×800 桌面与 944×427 触屏横屏确认三张不相邻起手牌：每次只播放一张，五个槽位始终固定，最后统一移入手牌。未修改规则与服务端抽取结果。

### 真红单队快速训练（2026-09-21）

固定配牌训练入口新增 `--learner-build`、`--learner-source`，仅更新真红策略，支持已验证旧十人权重向新增公开免伤观察列迁移。新增 `--hard-deadline` 保留已有总截止；评估仍用冻结对手和隔离种子。具体用户固定牌表、迁移及预算边界见 [真红快速训练](../../../artifacts/rl-evals/reports/duel-v2-zhenhong-quick-20260921.md)。验证：`python -m unittest tests.test_duel_v2_fixed_lineups tests.test_duel_v2_surplus_rework`。


### 2026-09-21 真红异能额外攻击环合

盈蓄异能的额外攻击通过操作级标记跳过基础战斗环合值，保留个人能量和全队分享；终结自然到期、普通出击与战斗牌不受影响。入场环合及明确效果给予的环合照常。数值口径来自用户桌游调整，沿用已核实角色主题，不是原作数值。验证：`python -m unittest tests.test_duel_v2_zhenhong_harmony tests.test_duel_v2_surplus_rework tests.test_duel_v2_zhenhong_buff`。


### 2026-09-21 真红盈蓄临时加攻

盈蓄的攻击 +1 与免伤同在下个己方回合开始、开始效果之前到期；重复盈蓄仍叠加攻击，到期只移除该来源累计值，其他加攻保留。`atk_buff_expiring` 单独保存数值与到期己方回合，随存档与公开策略状态保留；倒地统一清除。角色文案和回放同步，旧存档已混入的历史加攻不追溯扣除。验证：`tests.test_duel_v2_zhenhong_buff`、`tests.test_duel_v2_surplus_rework`，共 52 项通过。

### 2026-09-21 回放保留起手牌面修复

回放重建手牌时显式处理 `mulligan.card_ids`，只移除实际换走的牌，保留其余起手身份。网页按同样方式推进手牌；旧回放起手阶段的隐藏占位仅在全部保留牌身份已知、数量完全匹配时用已知牌恢复，不猜测未知牌，不公开对手暗牌。跳转与连续播放共用相同补丁处理。

验证：`python -m unittest tests.test_duel_v2_mulligan_replay tests.test_duel_v2_replay`，7 项通过，含双方视角更换 0～3 张的逐事件检查；`node --test tests/js/duel_v2_table_state.test.cjs tests/js/duel_v2_animation.test.cjs`，47 项通过。隔离测试服务的旧回放 A59AC8 经真实页面跳到第 1 回合、返回开局再连续播放，均为 5 张明牌、0 张暗牌，无脚本错误。

### 真红被动奖励（2026-09-21）

固定配牌 PPO 对真红学习方新增默认每次实际盈蓄被动触发 +0.2，`--zhenhong-passive-reward` 可调。计数位于角色回调，倒地时队伍盈蓄不计、九原重复段不重复计，逐决策回报只包含未来触发；其他策略、冻结对手和独立胜率口径不变。详见 [奖励说明](../../../docs/duel-v2-zhenhong-passive-reward.md)。68项相关测试通过，本次没有启动新训练。

### 五套完整配牌训练（2026-09-21）

固定配牌入口新增显式 `--full-cycle --warm-models`，通过 `rl/five_full_cycle.py` 执行五套基础学习、4轮有界配牌搜索、原／新配牌等量适应、独立确认与审计回退。真红保留被动次数奖励；免伤期间仅记录实际队友倒地风险与白送战术用例，不检查空前排、不添加额外惩罚。详见 [09:00截止完整计划](../../../artifacts/rl-evals/reports/duel-v2-five-full-night-20260921.md)。验证：`python -m unittest tests.test_duel_v2_five_full_cycle tests.test_duel_v2_build_acceptance tests.test_duel_v2_passive_reward tests.test_duel_v2_fixed_lineups`。导入或计划检查不启动训练。

### 2026-09-21 真红与翳公开

两名角色加入公开卡池，真红预组纳入普通账号图鉴、构筑、赠送及普通人机随机／指定对手。其他测试角色仍隐藏。夜间候选模型接入暂被训练机离线阻断，未替换数值权重或将旧模型重新标为候选。已保留本机接入准备补丁，待精确回传五套候选数值后再校验。

开放验证：目录／翳规则／人机选择54项通过，API全套38项中37项通过，仅既有高级人机“待训练”旧文案断言失败（实际为模型暂不可用）；新增普通账号真红构筑保存和开局通过。开发服务已恢复到0.0.0.0:5001，本机及当前局域网 `10.87.24.100:5001` 均返回200。夜间模型同步的精确来源与SHA记录在本机 `artifacts/rl-evals/five-full-night-20260921/pending-model-update.json`。

2026-09-21 评测范围纠正：上述新五套 `--full-cycle` 实际为纯网络PPO及固定旧对手配牌搜索，不等同于此前逐步256档搜索学习，也未覆盖完整报告规范中的镜像、直接互搏、换牌对照、泛化与逐卡机制记录。详见既有夜间报告顶部纠正说明；不能以其 `workflow_complete` 标记声称全规范完成。


### 2026-09-21 移除测试预组

官方预组目录移除 requiem「浊燃预组」与 sync-curse「覆纹预组（测试）」。普通人机指定对手和新赠送构筑统一只有创生、覆纹、小吱、真红四套，包括测试账号；已经保存的个人构筑、已有对局、回放和测试角色权限保留。角色机制与权限测试改用显式测试构筑，不依赖已移除的官方入口。

目录、普通人机选择、API 与受影响角色机制／权限验证共 59 项，58 通过；高级人机待训练提示旧断言失败，内存恢复原六套目录后仍复现同项。新增覆盖测试账号目录、新赠送仅四套，以及两个旧人机 ID 均拒绝且不建房。真实浏览器使用隔离测试账号确认大厅只有随机、镜像与四套公开预组。


### 2026-09-21 统一效果与光环

`engine/duel_v2/modifiers.py` 管理可序列化效果、条件光环、叠加、来源清理与纯属性查询；`attack_modifiers.py` 固定本次攻击的计算层；`effect_lifecycle.py` 统一绝对回合到期与自然到期扳机，真红终结已接入。持续攻击增减益、下次攻击、攻击条件光环、真红限时免伤和能量许可已迁移，费用与伤害数值也可使用统一效果查询。旧字段只通过兼容入口读取和派生，不再作为新效果的第二份可变真源。详见 [统一效果与属性计算](../../../docs/duel-v2-modifier-system.md)。

验证：`python -m unittest tests.test_duel_v2_modifiers tests.test_duel_v2_surplus_rework tests.test_duel_v2_zhenhong_buff tests.test_duel_v2_haniya tests.test_duel_v2_xiaozhi_haiyue tests.test_duel_v2_m07_aura tests.test_duel_v2_effect_markers`。本次不训练、不更新模型或重新授权旧编译执行器。


### 2026-09-21 战斗准备层与本回合减抗

伊洛伊与哈尼娅的战斗准备加成统一在战斗牌覆写之前。小吱 Q02 改用 `damage.resistance`、属性条件、伤害消费与本回合结束期限；玩家与异能者目标、预览、公开策略字段和标记共用该实例。旧减抗表在当前回合结束清除。

验证：`python -m unittest tests.test_duel_v2_resistance tests.test_duel_v2_xiaozhi_haiyue tests.test_duel_v2_jingu_marks tests.test_duel_v2_haniya tests.test_duel_v2_modifiers tests.test_duel_v2_effect_markers`。


### 2026-09-21 武备与永久成长分层

`engine/duel_v2/equipment.py` 管理基础面板、永久成长、装备来源和明确的装备回满操作。武备／角色条件仍由内容定义；`shape_events` 自动绑定装备来源，`weapon_hooks` 提供按当前装备派发的事件入口，零的三张回合结束武备已迁移。JSON 旧基础字段兼容读取，公开状态与回放包含分层来源，原数值字段仍提供派生值。详见 [实现说明](../../../docs/duel-v2-equipment-layers.md)。

验证：`python -m unittest tests.test_duel_v2_equipment tests.test_duel_v2_modifiers tests.test_duel_v2_surplus_rework tests.test_duel_v2_zhenhong_buff tests.test_duel_v2_xiaozhi_haiyue tests.test_duel_v2_haniya tests.test_duel_v2_m07_aura`。


### 2026-09-21 远程与教学

新增「远程」关键字加粗与图鉴机制词条。海月终结改为「本回合，海月可以在备战区远程出击。」；沿用既有通用备战出击能力：不入场、不换下前排、不受反击。普通费用、次数、前排优先和战斗资源不变。

教学追加第十三关 `tutorial_l14_ranged`，使用独立教学海月演示 1 生命远程出击与保留薄荷前排。旧十二关毕业账号可继续进入新关，原进度保留。关卡与教学角色定义在 `content/duel_v2/tutorial/`，无需新增路由或改动进度表。

验证：`python -m unittest tests.test_duel_v2_tutorial tests.test_duel_v2_xiaozhi_haiyue`，API 的旧毕业继续新关用例，以及 `node --test tests/js/duel_v2_ranged.test.cjs`。

本次 66 项规则／教学回归、4 项教学 API 用例及 1 项前端渲染测试通过。隔离测试数据库的真实页面从大厅进入第十三关，完成战斗牌、终结、远程拖拽与通关；确认海月保留 1 生命、薄荷仍在前排，图鉴可搜索远程且终结关键字加粗，浏览器无 error。

### 现行学习与旧五套入口

五套恢复入口：`train_duel_v2_current_round.py` 默认 `--runtime recovery`，由 `rl/recovery_round.py` 编排五套网络起点、逐批价值校准、完整搜索标签更新、独立两块选择种子、无行为收益停学／P0回退、两代恢复与纯网络／搜索主矩阵。策略与终局价值使用独立主干，价值-only更新不会移动策略。前三套旧教师不能表示的整个根显式使用新残差，真红／浊燃没有教师。`--smoke --device cpu --run`只作本机短测；正式CLI、worker、learn都要求同源码真实CUDA32预检和容量证据，旧cross正式入口仍拒绝。协议见 `docs/duel-v2-policy-recovery.md`；验证：`.venv/bin/python -m unittest tests.test_duel_v2_recovery_round tests.test_duel_v2_current_round`。实现与CPU短测不代表新策略已胜过旧模型，也不自动更新服务模型。

恢复采样由 `recovery_sampler.py` 区分原版A03、临时A03、C02／Q05复制和NF01，不读取对手未公开时效标记，不把已过期牌放回隐藏手牌。`check_duel_v2_rollout_coverage.py` 在替代配牌完整局中核验采样信息集、规则掩码与快速模拟对拍。教师模式通过 `--teacher-mode` 明确选择；固定预组 `network` 教师不获正式覆盖许可。`covered_terminal/covered_expert` 真实终局续局；`covered_value --covered-value-source DIR` 使用经90训练／30选择／30留出校准的独立专家价值代理，准备和复核分别用 `prepare_duel_v2_covered_value.py`、`verify_duel_v2_covered_value.py`。规则策略只接收观察，优化选择预览而不改变动作；这些入口不更改奖励或网站模型。

搜索规则教师的战斗预览只省去演出patch与公开棋盘缓存生成，继续使用同一隐私过滤和规则计算。`tests.test_duel_v2_covered_rollout` 对拍五套真实轨迹的预览与规则动作，并检查异常后的上下文恢复；辅助价值须按新算法SHA完整重验，不能沿用失效scope。单根加速不代替Windows完整局吞吐。

择优按队确认并复制候选／原模型数组，组合后再用新种子校准价值和复核搜索，全部完成才更新best。测试：`.venv/bin/python -m unittest tests.test_duel_v2_recovery_sampler tests.test_duel_v2_covered_rollout tests.test_duel_v2_covered_value tests.test_duel_v2_recovery_round`。教师身份、辅助值SHA及额外阶段成本进入预检／恢复协议。

搜索教师收益诊断：`scripts/check_duel_v2_search_teacher.py --source-run TRAINING_RUN --output NEW_OUTPUT --run` 在已消费训练根上比较16/32/64预算和重复搜索，并用另行预约的隐藏世界、配对冻结可见策略续局检验首选动作。默认每套1根、每档2次、8个续局世界、600秒；结果与私有分支轨迹写入output，不更新模型。未结束的配对不得计分，条件诊断不代替完整强度确认。验证：`.venv/bin/python -m unittest tests.test_duel_v2_teacher_validation`，协议见恢复文档。

扩大局面覆盖可用 `--roots-per-key 3 --seconds 1200`，按队交错运行。结果同时保存访问Q与一步固定观察者价值／条件真实终局误差，直接终局值和未完成配对不进入网络误差。冻结续局与搜索自博弈有分布差异，诊断不能代替匹配分布的价值校准或直接更新模型。

`scripts/verify_duel_v2_teacher_run.py --run-dir RUN --output NEW_JSON` 可在已有私有诊断上核验输入轨迹、分支合法性、固定观察者价值与终局，并审计采样牌池是否超出冻结策略的来源配牌。它不分配新种子或修改模型；未公开牌表仍不进入采样条件，来源牌池匹配不代表模型已具备能力。

完整本机预检还提供训练数据表示审计、三方数值对拍、WDL按局校准、同根搜索标签稳定性／自然尺度对照，以及仅搜索方手牌明牌的离线敏感性检查。`check_duel_v2_recovery_value.py` 用独立训练／温度选择／确认种子预热WDL；`check_duel_v2_recovery_chain.py` 要求报告绑定实际起点SHA，采集完整Gumbel32局、更新全部策略行及非行动方价值行，并核验导出与Adam／RNG恢复。`--device cuda`在无CUDA时明确拒绝；`prepare_duel_v2_offline_readiness.py --prepare`只生成源码与预检计划，不连接Windows、不部署、不自动长训。测试：`.venv/bin/python -m unittest tests.test_duel_v2_learning_audits tests.test_duel_v2_residual_learning tests.test_duel_v2_outcome_learning tests.test_duel_v2_gumbel_search tests.test_duel_v2_cross_teacher tests.test_duel_v2_information_search tests.test_duel_v2_training_recovery`。规则与网站模型不随独立研究检查改变；五套恢复入口的明确版本另按上文协议。

Windows 离线时的策略恢复研究见 `docs/duel-v2-policy-recovery.md`：`rl/residual_policy.py`／`residual_runtime.py` 提供独立版本的正常幅度残差策略；`rl/preserved_policy.py` 将创生、覆纹、快攻旧策略冻结并学习零初始化修正，真红和浊燃没有对应旧教师。`rl/recovery_runtime.py` 是五套恢复编排与独立研究共用的显式采集器，继续使用信息集搜索；单独调用不提供训练批准。`check_duel_v2_residual_learning.py` 默认输出计划，`--run` 才做CPU拟合，`--memorize` 只作真实小批容量检查；`check_duel_v2_policy_preservation.py` 以新种子完整对局检查初始化全部动作分数精确保留。不能把小批记忆或旧能力保持称为正式五套已修复；研究入口不自动批准网站；五套恢复默认入口与正式启动闸门见上文。验证：`.venv/bin/python -m unittest tests.test_duel_v2_residual_learning tests.test_duel_v2_outcome_learning tests.test_duel_v2_gumbel_search tests.test_duel_v2_cross_teacher`。

现行学习使用类别动作头、胜／平／负价值和 Gumbel 搜索，名单是创生、覆纹、快攻、真红，入口和奖励默认见 `docs/duel-v2-public-training.md` 的「现行学习方法」。五套交叉博弈入口是 `scripts/train_duel_v2_current_round.py`，统一 `cross_five_grounded_wdl_v1`，含浊燃与其他四套的真实对打；每个配对显式双先手，纯网络／搜索分别按25格评估。该入口固定配牌，不代替配牌搜索、择优和完整报告全项验收。新权重若比仓库已有同契约策略大幅变差，或纯网络不再出牌，按训练手册的 P0 停止并排查训练方式，不能加长同一配置。`scripts/train_duel_v2_full_cycle.py --ten` 仍编排旧的 `fixed_ten_v1` 标量五套，默认含已移除的娜白；它只复现旧实验，不能当作现行训练启动。`rl/new_model_report.py` 从冻结权重的直接互搏生成先手×后手矩阵及逐卡、角色和机制资料；报告与原始记录仅写入 `artifacts/rl-evals/`，不自动更新网站模型。

验证：`python -m unittest tests.test_duel_v2_ten_search tests.test_duel_v2_information_search tests.test_duel_v2_build_acceptance tests.test_duel_v2_outcome_learning tests.test_duel_v2_grounded_candidates`。正式运行前另完成 Windows CUDA 有界链路预检，不能把本机单元测试当作完整训练质量证据。旧五套长训若被明确要求复现，仍在启动前用 `--throughput-report` 核验同规则、同搜索预算的容量，并逐张量确认权重是否更新。

### 2026-09-22 移除娜白高级人机

按用户要求移除 midrange「娜白零伊」的大厅选项、随机高级人机池及项目服务模型文件。显式请求该策略在建房前拒绝；个人构筑、普通预组、历史对局与离线研究资料保留。

现行训练默认关闭真红前置奖和被动次数奖。`rl/setup_reward.py` 仍保留首次“延滞＋兽牙”0.1、64 步退场，以及独立的实际被动计数；只在训练手册允许的显式非零配置下使用。参数见 `docs/duel-v2-zhenhong-passive-reward.md`，不改变网站卡效或胜负。

真红前置奖励的小时级开关对照使用 `scripts/train_duel_v2_setup_trial.py`，复用共享搜索学习并按完整配对对局同步更新两组，独立比较初始／control／setup的胜率与机制触发。验证：`python -m unittest tests.test_duel_v2_setup_trial tests.test_duel_v2_setup_reward`；数值产物仅进入artifacts。

### 纯胜负经验学习试验

`rl/outcome_runtime.py`维护独立WDL数值模型，`rl/episode_replay.py`维护按局经验与信息集重分析，`rl/outcome_trial.py`编排有界对照。入口为`scripts/train_duel_v2_outcome_trial.py`；范围及预算见[训练手册](../../../docs/duel-v2-public-training.md#纯胜负经验学习的有界试验)。验证：`python -m unittest tests.test_duel_v2_outcome_learning tests.test_duel_v2_information_search tests.test_duel_v2_ten_search`。不改变网站模型或已有冻结任务。

`--profile selfplay_budget`及`rl/selfplay_trial.py`提供固定陪练／镜像双方学习／混合搜索预算三组对照，每组独立采样槽和统一截止。专项验证：`python -m unittest tests.test_duel_v2_selfplay_trial tests.test_duel_v2_information_search tests.test_duel_v2_outcome_learning`。

`rl/trajectory_checkpoint.py`保存私有搜索轨迹；selfplay采样按决策边界分段续接，保持原数值模型、RNG与截止。`scripts/check_duel_v2_learning_pipeline.py`提供搜索等价／性能、连续恢复、根动作对照及隔离拟合诊断；不自动启动长训。回归：`python -m unittest tests.test_duel_v2_trajectory_resume`。

离线搜索树使用`engine/duel_v2/simulation.py`省去演出patch生成，真实对局与回放仍走普通动作入口。对拍与计时入口`scripts/check_duel_v2_simulation.py`；专项回归`python -m unittest tests.test_duel_v2_simulation`。旧数值模型仅可显式迁移到未批准研究副本，不能据性能对拍继承网站批准。

`scripts/check_duel_v2_simulation_game.py`对同种子完整搜索采样计时，预算固定32，分别记录普通／轻量路径；只采样、不更新参数，不把保存动作的执行速度当作训练吞吐。

低预算搜索由`rl/search_policy.py`维护Gumbel访问计划和改进策略目标。现行新任务默认 32 次、根候选上限 16；这组搜索默认值不把旧 `--ten` 标量入口变成现行训练。固定真红可用`--profile gumbel_low_budget`对比16/32。`scripts/check_duel_v2_gumbel_learning.py`只做有界重分析、隔离拟合与新种子评估；原排查录像不进入梯度。验证：`python -m unittest tests.test_duel_v2_gumbel_search tests.test_duel_v2_information_search tests.test_duel_v2_training_recovery`。

动作表示与真红专项验收：`train_duel_v2_outcome_trial.py --profile grounded_actions`比较继承／重置旧动作头与类别角色状态动作头；`audit_duel_v2_zhenhong.py`只读核验原始评估中的准备、启动、收割与实际胜者。流程见训练手册。验证：`python -m unittest tests.test_duel_v2_grounded_candidates tests.test_duel_v2_tactical_audit tests.test_duel_v2_outcome_learning`。


### 五套先后手日程与统一研究编码

`rl/cross_schedule.py` 独立定义配对先手和矩阵格子，不从种子奇偶推断。`cross_lineup.py`、`cross_grounded.py`、`cross_runtime.py`承载统一观察、类别动作及WDL迁移，旧十人与浊燃schema保持独立。构筑沿用源模型32张，网站模型不自动更新。运行预算、迁移重置列和报告边界见训练手册。

验证：`python -m unittest tests.test_duel_v2_current_round tests.test_duel_v2_gumbel_search tests.test_duel_v2_murk_encoding`；正式启动前在CUDA运行完整短测并用正式32搜索档测容量。

正式32次搜索预检触发过仿效娜娜莉出击时的追击判定：仿效者没有追击武备也没有装备武备，旧逻辑把两个缺省值 `None` 误判相等并以 `None` 查牌表。当前仅在追击武备ID确实存在且已装备时追加武备追击，普通娜娜莉追击不变。回归：`tests.test_duel_v2_followup_target`。

Windows 后台启动使用 `scripts/launch_duel_v2_cross_detached.py`，只在 CUDA 短测及正式搜索吞吐测量通过后调用。它将带绝对09:00截止的训练守护进程脱离 SSH Job Object，保存 PID、源码 SHA 和独立 stdout/stderr；训练器自身负责全进程树硬截止。启动后仍须核对真实对局与参数更新，单有 PID 不算开始成功。

2026-09-25：浊燃队四名角色（安魂曲、残虹、早雾、阿德勒）及 `murk` 预组改为普通账号可见，普通人机随机池随公开预组扩为五套；旧段落的“测试账号限定”是历史状态。原五套交叉训练里仓库缺少的真红、浊燃权重留存在 `engine/ai/models/research/five-cross-20260924/`。用户另行授权它们作为**实验性高级人机**显式选择：服务只加载绑定 SHA 与兼容指纹的这两份研究权重，界面与对局标明实验性，随机高级人机也会提示可能抽到实验策略；未获强度验收，不代表正式推荐。原三套服务模型不变。

### 2026-09-26 浊燃队规则与牌面核对

现行四人卡表与持续伤害时点见规则手册「浊燃队现行卡表」。牌面补齐瞬发、可重复使用、不消耗行动力和倒地可用，负攻击修正在卡框显示。主动触发噩梦与回合结束使用同一结算入口；幻境的实际伤害叠层和同次分配演出共组。指定异能者／角色的合法目标与公开交互提示同步；阿德勒的全体角色护盾包含玩家。A04 达到 10 层仍将茄汁金属乐加入手牌，按用户追加确认不自动释放。

验证：`python -m unittest tests.test_duel_v2_continuous_cast tests.test_duel_v2_catalog tests.test_duel_v2_zone_status tests.test_duel_v2_followup_target`；`node --test tests/js/duel_v2_ranged.test.cjs tests/js/duel_v2_animation.test.cjs`。不以规则单测作为平衡或模型质量证据。

复制生成的家族壮大按实际拥有者展示；安魂曲 A07 的回合开始效果使用独立武备钩子，复制异能不遮掉自己的武备效果。相关覆盖也在 `tests.test_duel_v2_continuous_cast`。模拟对拍单独运行 `python -m unittest tests.test_duel_v2_simulation`；既有实体测试与模拟测试混跑的补丁隔离断言在修改前同样失败，不作为本次新增问题。


### 2026-10-07 真红伤害上限与翳抽牌

真红盈蓄保护改为每个伤害段至多 1 点、护盾前封顶，沿用下个己方回合开始到期；公开 `damage_limit` 与实时／回放蓝色盾形外框同步。I03 生成 4 个己方回合兽牙并抽 1 张，I04 生成 2 个并抽 1 张，分别保留瞬发／真正零费。数据来自用户桌游调整，细则见完整手册。

验证：`.venv/bin/python -m unittest tests.test_duel_v2_zhenhong_buff tests.test_duel_v2_surplus_rework tests.test_duel_v2_simulation`；`node --test tests/js/duel_v2_immunity_frame.test.cjs`。覆盖伤害段、增伤／抗性／护盾／穿透、零伤害、援护免反击、到期与倒地、JSON与回放、抽牌隐私／空库及费用。新规则请重新开局，旧训练身份和回放记录不回写。

恢复训练扩大预检：`scripts/train_duel_v2_current_round.py --preflight --preflight-replicas 3`在当前规则下生成训练／留出各90个完整Gumbel32对局，容量拟合完整遍历策略行；复制寿命与真红伤害上限使用合法可见输入。正式入口的批次数、闸门根数和停学时间比例可在启动前显式声明，仍须通过同源码CUDA预检、实测容量及原有退化闸门。详见`docs/duel-v2-policy-recovery.md`。验证增加`tests.test_duel_v2_recovery_observation`，覆盖上限／无上限、复制寿命、实体改名不改变编码及对方暗牌不能进入候选上下文。

专家价值代理可用`prepare_duel_v2_covered_value.py --train-replicas 12 --eval-replicas 3`扩大到360／90／90局；仅在选择集确定轮数与温度1／2／4，新的scope端点和留出原始SHA必须匹配，留出失败不会授予搜索训练许可。

恢复观察还读取真红终结期间自己已可见的`hand_face`原牌库存，候选保留替代牌的原牌模板身份，以区分兽牙结算后不同的弃牌结果；对应原牌语义和对方隐藏信息隔离见`tests.test_duel_v2_recovery_observation`。

冷启动的真红／浊燃可使用`scripts/prepare_duel_v2_rule_foundation.py`进行可见规则AI的神经初始化，仅消费核验的当前规则专家TRAINING轨迹，内部成对种子验证选择4／8／16轮；三套旧模型数值不变，输出没有正式搜索或服务批准。随后仍须新Gumbel32预检。统一候选变换在自动战斗中提供公开对方前排上下文，显式目标不变；验证增加`tests.test_duel_v2_rule_foundation`。


### 2026-10-07 残虹异能限次纠正（历史）

> 每回合加攻已被下文初始环合调整移除；本节保留当时实现与验证记录。

残虹异能的「每回合 1 次」限制持续伤害施加带来的攻击 +1；浊燃重复触发可在同回合叠至 3 层。规则真源、角色异能与图鉴词条同步。正式引擎与复用它的模拟路径共用角色实体计次，换武备、倒地复活及 JSON 恢复不重置。旧冻结诊断及模型仍保留原身份，不能作为新规则的平衡证据。

验证：`.venv/bin/python -m unittest tests.test_duel_v2_continuous_cast tests.test_duel_v2_catalog tests.test_duel_v2_zone_status`；模拟路径单独运行 `.venv/bin/python -m unittest tests.test_duel_v2_simulation`。

本轮上述规则 42 项与模拟 3 项通过，真实图鉴确认异能限次与浊燃词条；`tests.test_duel_v2_hooks` 的角色包清单断言仍因额外教学角色失败，修改前同样复现。开发服务监听 `0.0.0.0:5001`，本机与当前局域网 `192.168.10.3:5001` 均返回 200。


单队适应复用恢复栈辅助函数的显式`keys`／`learning_keys`，只收学习侧完整局经验并更新指定队伍；其余队在准备阶段价值校准后冻结。单队有界编排仍须同源码CUDA、价值和全网络留出预检，辅助函数不提供网站批准。历史残虹加攻计次列保留在浊燃与统一恢复观察中以兼容既有模型结构；现行规则不再产生该计次。验证：`tests.test_duel_v2_murk_encoding`及`tests.test_duel_v2_recovery_round`的单队调度、拒绝冻结侧样本和独立闸门用例。


### 2026-10-07 阿德勒终结与机制说明卡

阿德勒终结改为施加2个己方回合诛恶护持，并即时移除全部对方角色现有护盾。规则和边界见完整手册，旧即时伤害与返能被替换。三名角色的私有持续伤害说明通过角色目录的`mechanisms`字段展示，末尾说明卡不进入`CARDS`或构筑；图鉴与对局关联小卡共用同一数据，蚀心／鸩火别名去重。机制词典只保留通用持续伤害定义。

验证：`.venv/bin/python -m unittest tests.test_duel_v2_continuous_cast tests.test_duel_v2_catalog tests.test_duel_v2_zone_status`；`.venv/bin/python -m unittest tests.test_duel_v2_simulation`；`node --test tests/js/duel_v2_mechanisms.test.cjs tests/js/duel_v2_ranged.test.cjs tests/js/duel_v2_hand_face.test.cjs`。规则修改不覆盖此前冻结评估或自动重绑模型。

本轮验收：48 项 Python 与 7 项 JavaScript 测试通过；真实图鉴末尾说明、私有词条移出和牌桌关联小卡（蚀心／鸩火去重）均验证，前端 error 日志为空。开发服务仍监听 `0.0.0.0:5001`，本机和当前局域网 `192.168.10.3:5001` 均返回 200。

### 高级人机模型与搜索（2026-10-07）

五套策略统一使用 `engine/ai/models/recovery-20261007/` 中的冻结 NumPy 模型与原始训练清单；浊燃使用训练择优的 best，其余四队使用同批冻结模型。旧根目录的三套模型保留为嵌入旧策略的校验来源，不再作为默认对手。`recovery_model.py` 校验源清单、权重、配牌 SHA、具名特征、张量与当前规则授权；单独的 `serving.json` 保存用户明确启用的绑定。原始质量结论、训练规则身份及未批准标记保持原样；当前阿德勒终结与真红基础 1/6、武备 2/6／1/8 晚于训练修改，启用不代表通过新规则强度验收。

每个 `/api/duel-v2/ai-step` 使用当前正式引擎与公开信息采样，执行 Gumbel 信息集搜索（最多 32 次模拟、16 个根候选、3 步短终局检查、3 秒搜索预算），采用新模型独立终局价值塔及 natural-WDL Q 尺度。立即获胜检查共享预算；有已完成模拟时采用搜索结果，预算耗尽且无完整模拟或不支持的私有选择根才回退合法网络动作。起手换牌也由新网络选择。对手模拟使用同一冻结策略作为代理，不读取真实隐藏手牌。网站只运行 NumPy 推理，不加载 PT、Torch 或训练任务。旧对局的权重／构筑版本不匹配时要求重新开局。

回归：`.venv/bin/python -m unittest tests.test_duel_v2_recovery_serving tests.test_duel_v2_serving_search tests.test_duel_v2_ai_step tests.test_duel_v2_experimental_advanced`。

### 离线批量搜索基础

批量评测分别记录物理容量和行为验收；有合法牌可出却整局既不出牌、不普通出击也不终结的空转会拒绝验收，8 个不同正常局达到 P0 停止门槛。闸门只审计真实动作，不强迫出牌或修改模型／奖励。验证：`.venv/bin/python -m unittest tests.test_duel_v2_batch_behavior tests.test_duel_v2_batch_recording tests.test_duel_v2_batched_evaluator`。

`rl/batched_duel` 提供显式容量的 CPU arena、无损实体图 codec、设备数组分叉、真实 NVRTC/Driver kernel 加载器和兼容 CPython 的随机原语；`rl/batched_search` 提供根预算账本、模型身份绑定的 FP32 批量推理、可挂起的 Gumbel 遍历、跨根协作调度和设备搜索统计池。公开证明与树搜索共用预算，排队计入截止，取消或到期后拒绝迟到结果。组件显式构造才工作，导入不启动对局、推理或 CUDA。

离线入口 `scripts/evaluate_duel_v2_batched.py` 将完整正式规则驻留在有界 CPU 进程池，跨对局合并网络推理与 GPU 搜索统计；不是 1,600 个系统线程，也不是 GPU 规则解释器。运行要求冻结源码、模型、独立种子和显式预算模式；`wall_clock` 每决策从入队起计 3 秒，公开证明和树搜索共用最多 32 次预算，迟到结果不计完成模拟。实际动作、私有原始记录和公开回放分别保存，异常及未完成局不得算正常终局。设备规则执行尚未接通；存储、合成根和单次推理测试不代表 1,600 真实局容量验收。服务和训练入口不自动切换。验证、产物及容量门槛见[离线批量搜索](../../../docs/duel-v2-batched-search.md)。

### 2026-10-07 真红基础面板 1/6

按用户桌游试玩参数，真红基础攻击／生命上限改为 1/6。新开局、目录和公开投影从同一内容定义读取；武备继续使用各自独立面板，历史对局与冻结评估不回写。现有实验性高级人机仅核验具名观察／候选、数值结构和合法动作后更新独立运行兼容绑定，原始训练清单、权重 SHA 与质量标记保留；不启动训练或强度评估。

回归：`.venv/bin/python -m unittest tests.test_duel_v2_zhenhong_buff tests.test_duel_v2_surplus_rework tests.test_duel_v2_zhenhong_harmony tests.test_duel_v2_catalog`；模拟对拍单独运行 `tests.test_duel_v2_simulation`。

### 2026-10-07 真红武备与盈蓄启动方向

按用户指定，R07「梦的边缘」为 2/6，R08「穿过胭红蜃景」为 1/8，效果保留现行定义。对创生的后续平衡重点是减少凑齐盈蓄的前置成本与环合断链，具体目标和机制边界见规则手册；不将能量、终结调整作为本轮主要方向，也不自动采纳新增卡效。实验性高级人机核验新数值的观察／候选契约后单独记录运行兼容身份，原始训练清单与权重字节保留。验证沿用真红／翳规则、装备分层、模拟对拍与 recovery serving 接入检查。


### 2026-10-07 残虹初始环合

按用户桌游规则，残虹移除每回合施加持续伤害加攻，改为初始具有 2 点环合值。角色包通过已有开局钩子赋予一次，目录、图鉴与规则手册同步；回合推进、倒地复活、换武备和 JSON 恢复不补发，复制异能不追溯开局。新规则请重新开局；旧模型、训练观察结构及兼容授权不改。

验证：`.venv/bin/python -m unittest tests.test_duel_v2_continuous_cast tests.test_duel_v2_catalog tests.test_duel_v2_zone_status tests.test_duel_v2_engine tests.test_duel_v2_replay`；模拟路径单独运行 `.venv/bin/python -m unittest tests.test_duel_v2_simulation`。

本轮目录、持续伤害、区域状态与回放 54 项通过，模拟对拍 3 项通过；扩大引擎回归的 6 项失败在恢复旧残虹实现后同样复现。真实大厅进入浊燃普通人机牌桌，确认换牌前残虹为环合 2/2，确认换牌并刷新后保留，历史只记录一次开局获得。图鉴异能已显示新文案；开发服务继续监听 `0.0.0.0:5001`，本机与当前局域网 `192.168.10.3:5001` 均返回 200。

### 2026-10-07 真红开局 1 点环合

真红内容钩子 `on_game_start` 在初始化时通过通用资源接口授予 1 点个人环合，并记录公开开局效果；牌面同步说明。JSON 恢复、倒地复活、换人及终结不重发该钩子；最近出战者、存活与环合支付规则保持现行约束，R02 额外环合候选未实施。实验性高级模型沿用原数值，核验编码兼容后单独记录当前运行身份。

回归：`.venv/bin/python -m unittest tests.test_duel_v2_zhenhong_initial tests.test_duel_v2_zhenhong_buff tests.test_duel_v2_surplus_rework tests.test_duel_v2_zhenhong_harmony`；模拟对拍单独运行 `tests.test_duel_v2_simulation`。

### 2026-10-07 真红三张战斗牌分类抽牌

R02／R03／R04 移除盈蓄条件与瞬发，始终在出击前抽对应类型牌；攻击修正 -1／0／+1、护盾 2／1／0，费用均为 1。分类抽牌与牌面／预览同步，终结视为 RF01 的路径仍独立。用户桌游数值不代表强度验收。回归沿用 `tests.test_duel_v2_zhenhong_buff`、`tests.test_duel_v2_surplus_rework`、`tests.test_duel_v2_zhenhong_initial`，模拟对拍单独运行 `tests.test_duel_v2_simulation`。


### 卡效测试维护

规则测试以完整手册的现行卡效为准。Y07 仅在装备及伊洛伊终结延后回能时治疗，不因倒地追加伤害；倒地顺序使用真实攻守伤害和 B02 倒地前治疗构造。N03 的原目标被攻击击倒后跳过追击；N06 是无目标自复活，Y06 才选择己方倒地异能者。J01 承载检视选择，J03 不再用于旧检视夹具；X05 返回前排，不再记录、兑现或生成复制。新卡效没有产生旧记录事件的断言须随规则删除或改为验证不产生该事件。

目录检查按每角色八种可构筑牌验收，注册表区分正式卡池与独立教学包，不能用历史角色总数约束现行目录。环合夹具须在退下前充满，浔的倾陷在下个己方回合结束时清除。高级人机缺模型测试显式模拟加载失败，不依赖某套策略长期未训练。残虹持续伤害不再产生加攻计次，现行观察的旧计次列应为 0；历史计次列的实体改名兼容测试用明确的旧快照字段构造。旧快照和冻结编码的兼容性测试仍独立保留；旧 native 编译执行器的完整性校验失败不作为现行卡效已验收，也不能通过删断言或忽略异常掩盖。

专项入口：`.venv/bin/python -m unittest tests.test_duel_v2_engine tests.test_duel_v2_iloy_knockdown tests.test_duel_v2_september18 tests.test_duel_v2_followup_target tests.test_duel_v2_presentation tests.test_duel_v2_action_presentation_boundary tests.test_duel_v2_edgar tests.test_duel_v2_hooks tests.test_duel_v2_xiaozhi_haiyue tests.test_duel_v2_xun_rework tests.test_duel_v2_murk_encoding tests.test_duel_v2_api`。

补充检查：`.venv/bin/python -m unittest tests.test_duel_v2_table_contract tests.test_duel_v2_rl tests.test_duel_v2_rl_eval tests.test_duel_v2_fixed_lineups tests.test_duel_v2_tactical_audit`。旧编码／旧免伤报告夹具显式标记历史兼容。覆盖评估构筑仅携带每人八种可构筑牌，衍生牌只进入全目录使用与效果统计，不进入开局牌组；细则见[卡效覆盖评估](../../../docs/duel-v2-card-eval.md)。

### 2026-10-07 用户指定真红队翳配牌

普通真红预组与高级人机服务配牌统一为 I01／I03／I04／I08 各 2 张。`recovery_model` 支持在独立 `serving.json` 的对应模型项中指定 `serving_build`、`serving_build_sha256` 与非空 `serving_build_request`；原来源清单与配牌哈希仍严格核验，服务配牌须合法且保持原四人顺序。原数值权重与训练清单不改写，旧局的构筑身份校验仍要求重新开局。回归见 `tests.test_duel_v2_recovery_serving` 与 `tests.test_duel_v2_experimental_advanced`。


### 2026-10-07 真红独行与机制卡展示

盈蓄异能触发「独行」并回复至当前生命上限；独行不强制视为援护技，正常移入战斗区并按实际入场环合判定；额外攻击允许对方正常反击，即使该次入场触发环合；R06 从现有独行触发计数（`surplus_passive_triggers`）读取加成，终结自然到期改为对所有对方角色造成 3 点伤害。提前结束不触发，完整规则见手册。图鉴机制卡接在普通卡后共用网格，牌桌关联小卡共用普通卡立绘区和缩放；仅展示资料，不进卡池。新规则请重新开局，旧模型与冻结记录保留原身份。

验证：`.venv/bin/python -m unittest tests.test_duel_v2_solitary tests.test_duel_v2_zhenhong_buff tests.test_duel_v2_zhenhong_initial tests.test_duel_v2_zhenhong_harmony tests.test_duel_v2_surplus_rework tests.test_duel_v2_catalog`；模拟对拍单独运行 `tests.test_duel_v2_simulation`；`node --test tests/js/duel_v2_mechanisms.test.cjs tests/js/duel_v2_hand_face.test.cjs`。

### 真红展开辅助规则策略（离线实验）

`engine/ai/zhenhong_guide.py` 提供用户指定的兽牙→延滞→光系环合→创生盈蓄指导，使用3个信息集采样世界与至多24个两动作计划，保留直接取胜及保命例外，不更新模型或训练奖励。当前默认网站不开启；接口、人工排序、耗时与采用边界见 [辅助策略协议](../../../docs/duel-v2-zhenhong-guide.md)。验证：`.venv/bin/python -m unittest tests.test_duel_v2_zhenhong_guide`。

### 真红辅助策略起手留牌（v2，离线）

指导现在包含起手换牌：保留一张兽牙来源与一张零的战斗牌，优先 I04／I03；核心缺失时保留 R01 或 Y02 作为补牌备选。只减少模型拟换出的牌，不读取对手暗牌或牌序，不改普通规则和模型权重。旧 v1 录像与统计保持原记录，当前网站默认仍不开启辅助。细则与验证见 [辅助策略协议](../../../docs/duel-v2-zhenhong-guide.md)。

### 真红辅助完整准备组合（v3，离线）

优先完整检查零战斗牌→翳战斗牌：零蓄满环合，翳入场同时获得兽牙并施加延滞。为该两步与已就绪后半段预留检查预算；条件不足才退回单铺兽牙。留牌相应优先保留零／翳战斗牌组合；没有零战斗牌时仍优先免费兽牙来源。三世界合法性、行动力、付款人和两人存活由正式引擎验证，当前默认网站不开启。


### 离线真红指导训练

`rl/guided_recovery.py` 为显式的真红v3辅助教师：保留原Gumbel标签，改变实走时生成声明的混合策略标签，价值仍取真实终局WDL；默认和网站不启用。接入恢复采样及冻结评估，指导版本进入算法指纹。单次配置／产物均放 `artifacts/rl-evals/<run-id>/`，操作和验收见训练手册。验证：`.venv/bin/python -m unittest tests.test_duel_v2_guided_recovery tests.test_duel_v2_zhenhong_guide tests.test_duel_v2_recovery_round`。

### CUDA 规则执行组件的边界

`rl/batched_duel` 新增冻结程序数字图、CUDA 图堆／解释器、容器、模块／方法查询和候选编码组件。设备对拍覆盖的是已实现子集；完整规则、随机采样、GPU 观察和搜索尚未全部接通，`supports_full_duel_rules=False`。旧 CPU 进程池的 1,600 局在途容量不能当成真正 GPU 规则并发已完成。接口、验证与剩余门槛见[离线批量搜索](../../../docs/duel-v2-batched-search.md)。

`rl/batched_duel/gpu_observation.py` 提供 2,420 列观察的真实 CUDA 编码及直接设备 token 接口，与 254 列候选分别逐列对拍。公开 token 的主机提取仍是过渡适配，完整 GPU 投影和规则搜索未完成。验证：`.venv/bin/python -m unittest tests.test_duel_v2_gpu_observation`；1,600 行编码样本不计为 1,600 个完整对局。


### 短在线指导诊断

显式 `guided_online_short_v1` 仅用于用户授权的最长600秒真红在线短测；复用恢复栈完整搜索／终局联合更新，每批新教材通行1遍，并做CUDA／价值／恢复／独立P0检查。原容量失败保持失败，短测不能替代默认完整训练回执。协议、冻结及回传见训练手册；单轮编排和结果放在 `artifacts/rl-evals/<run-id>/`。

`rl/batched_duel/gpu_observation.py` 提供 2,420 列观察的真实 CUDA 编码及直接设备 token 接口，与 254 列候选分别逐列对拍。公开 token 的主机提取仍是过渡适配，完整 GPU 投影和规则搜索未完成。验证：`.venv/bin/python -m unittest tests.test_duel_v2_gpu_observation`；1,600 行编码样本不计为 1,600 个完整对局。


### 2026-10-08 限伤与穿透

穿透按前排限伤前的攻击伤害、适用护盾和伤害前生命计算溢出；真红自身仍按单段限伤结算。无护盾时 8 攻薄荷对 5 血真红造成 1 点生命伤害、3 点玩家穿透伤害。预览与回放共用正式引擎，图鉴词条同步。验证：`.venv/bin/python -m unittest tests.test_duel_v2_zhenhong_buff tests.test_duel_v2_solitary tests.test_duel_v2_engine tests.test_duel_v2_shield_stacking`，模拟对拍单独运行 `tests.test_duel_v2_simulation`。


### 2026-10-08 卡牌文案与机制展示整理

N05 新增瞬发，I06 改为一次主动选择对方存活异能者并集中施加累计减攻；响应牌明确自动使用，N02 不再复述通用反击规则。C07 自动使用仍为零行动力、手动使用仍为 1 费。S07 条件句、噩梦与诛恶护持机制说明按手册整理。枚约接入统一 debuff 标记；关联机制小卡去掉立绘并保持正常字号。鬼郎丸使用本地透明主体 WebP，来源与提示词见[图片说明](../../../docs/character-image-assets.md)。

验证：`.venv/bin/python -m unittest tests.test_duel_v2_card_polish tests.test_duel_v2_effect_markers tests.test_duel_v2_catalog tests.test_duel_v2_surplus_rework tests.test_duel_v2_continuous_cast tests.test_duel_v2_september18 tests.test_duel_v2_engine tests.test_duel_v2_hooks`；模拟对拍单独运行 `tests.test_duel_v2_simulation`；`node --test tests/js/duel_v2_mechanisms.test.cjs tests/js/duel_v2_hand_face.test.cjs`。


### 固定构筑在线训练与批量价值统计

`fixed_build_online_guided_v1`用于用户明确指定截止的完整固定构筑在线任务，保留起手留牌与真红指导，构筑换卡可按用户要求跳过；不把短测ready或旧容量失败冒作默认完整训练批准。`rl/batched_search/value_metrics.py`复用已验证批量推理计算完整局等权WDL统计，`recovery_round.warm_value`可显式选择`value_metric_backend=cuda_batched`，默认NumPy不变。此组件不执行GPU规则，未知后端和CUDA隐式回退拒绝。验证：`.venv/bin/python -m unittest tests.test_duel_v2_batched_value_metrics tests.test_duel_v2_batch_inference tests.test_duel_v2_recovery_round`。


### 2026-10-08 黑胶鬼郎丸弃牌反馈

弥散性的朦白所召唤鬼郎丸的黑胶效果实际弃牌后，播放包含被弃牌名称的效果提示，双方与公开回放一致；普通进入弃牌堆仍静默。公共效果历史可显式设置 `present=True`，默认行为保持静默。验证：`.venv/bin/python -m unittest tests.test_duel_v2_continuous_cast tests.test_duel_v2_trigger_history tests.test_duel_v2_presentation`；`node --test tests/js/duel_v2_animation.test.cjs tests/js/duel_v2_table_state.test.cjs`。真实牌桌以祈愿性的依归使黑胶鬼郎丸受伤，确认弃牌提示及数量同步。


### 2026-10-08 图鉴未公开标记

图鉴角色导航及角色详情标题旁，对目录 `access_level=test` 的角色显示「未公开」标记。公开角色不显示，缺少字段的旧目录不误标；角色可见范围仍由后端目录过滤决定。

### 2026-10-08 恢复五套实验性高级人机

用户要求恢复对战后，五套 `recovery-20261007` 服务模型重新核验现行具名观察／动作、数值结构和构筑，再更新独立 `serving.json` 的运行规则绑定与兼容历史。原训练清单、权重 SHA、原构筑身份和真红队翳的指定服务配牌保留；不采用失败预检候选、不训练，也不升级质量批准。现有 3 秒／最多 32 次模拟的信息集搜索继续使用。高级人机缓存同时绑定现行规则身份，运行身份变化时必须重新校验，不能复用旧的已加载对象。

恢复验收覆盖普通非测试账号对五套高级人机的真实 HTTP 开局、各队真实有界搜索及搜索所选动作、绑定／权重篡改与热缓存规则漂移拒绝；网页仅限公开角色构筑，仍标为实验性对手。验证：`.venv/bin/python -m unittest tests.test_duel_v2_recovery_serving tests.test_duel_v2_experimental_advanced tests.test_duel_v2_serving_search`。这证明可对战和搜索接通，不代表新规则强度验收。


### 2026-10-08 高级人机预组显示名

高级人机选项恢复设计名称：创生预组、覆纹预组、小吱预组、真红预组、浊燃预组；保留实验性标记。大厅构筑摘要跟随选项名称，不再使用阵容首字简称。内部策略 ID、实际模型配牌及已有房间身份不因展示名变化而改变。

### 真红高级人机单独切换训练来源

用户可明确选择已回传的真红短训 `latest` 候选作为实验性网页对手。`serving.json` 的模型项可通过成套 `source_rule_hash`、`source_request`、`source_report` 记录独立训练来源；原五队顶层来源与其他四队不变。加载仍要求当前运行规则、原始模型/清单/构筑SHA、具名观察/动作、数值结构和未批准质量状态全部一致；来源字段缺失或与原清单不符则拒绝，不能重写训练清单冒充当前规则训练。

真红当前使用2026-10-08短训练未批准候选，保留 I01/I03/I04/I08 各2张的服务配牌，继续现有信息集搜索，不在网页运行真红v3指导；更新权重后旧真红房间须重新开局。此为用户指定试用，不是确认强度提升，也不采用五队失败预检的模型。验证沿用恢复服务、实验性对手和搜索测试；原候选、旧服务模型及检查记录留在 `artifacts/rl-evals/`。
