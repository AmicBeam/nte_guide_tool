# 攻略数据库

公共资料可匿名阅读，提供异环角色、弧盘、料理、轨外之境、全站深境及攻略文章；攻略撰写和个人练度同步须登录本站，养成计算仍可匿名使用。角色工具页通过排轴公开接口展示相关轴，并链接到排轴模块的配装收益比较。工具主页可进入 `/guide`；不依赖其他模块的领域服务，不使用异能对决的原创卡牌数值。当前快照包含 25 个角色条目（男女主分别计数）、51 个弧盘、15 个环线，以及全站深境200层（每50层一组）；包含数据集中已提供的条目，不以列表顺序推断实装或赛期开放状态。

## 网页和机器读取

- `/guide?kind=characters|arcs|foods|challenges|endless&q=关键词`：服务端渲染索引，搜索技能正文、弧盘效果和敌人名；角色支持 `element=魂&rarity=5&arc_type=4`（弧盘类型ID）。属性、星级、弧盘类型使用三组图标单选按钮，各组提供“全部”，首次页面包含全部25个角色卡片与去重的文本检索字段，后续图标筛选与搜索只在前端显示／隐藏卡片，不发起查询请求、不跳转页面；保留滚动位置并同步地址栏。直接打开带筛选参数的URL仍由服务端渲染相同初始结果。无JavaScript时可点击搜索提交。
- `/guide/<分类>/<id>`：完整详情，正文无需 JavaScript；角色技能、属性成长与弧盘精炼提供等级滑块，切换当前档位数值；详情采用左侧悬浮目录，技能有二级目录并保留“普通攻击：”“天赋一：”等类型前缀，当前章节随滚动高亮，窄屏通过左下角目录按钮展开；挑战详情可加 `?stage=12` 只看某关。
- `/guide/<分类>/<id>.md`：相同内容的 Markdown，可加 `section=abilities`、`section=effect` 或 `stage=12` 限定返回量。
- `/api/guide/<分类>`：分页摘要，支持 `q`、`element`、`rarity`、`limit`（1–100）、`offset`，返回 `total`、`next_offset`、`items`；摘要含三种详情链接。
- `/api/guide/search?q=关键词`：跨分类全文检索，分页参数相同。
- `/api/guide/<分类>/<id>`：结构化详情，返回 `schema_version`、`kind`、`data`；支持 `section` 与 `stage`。
- `/api/guide/schema`：接口约定、数值语义、字段标签。
- `/llms.txt` 与 `/guide/llms.txt`：LLM 发现入口。角色、弧盘与轨外之境详情统一通过页底“LLM 阅读入口”发现 Markdown、JSON 等读取方式，详情顶部不再显示独立格式入口。
- 不存在的分类、ID、字段或关卡返回 404；无效分页和星级返回 400；错误 JSON 为 `error.code` 和 `error.message`。

角色保留属性成长、完整技能、空幕格位与驱动特化、角色突破与技能升级材料、觉醒、共鸣和机制词条；弧盘保留成长、五档精炼效果、专属效果与突破材料；挑战保留赛期增益、目标、半场、波次、敌人数值与抗性。角色另提供好感度等级、文字奖励与解锁、满好感属性、喜好礼物与礼物商店标价；料理提供按功能分组的食品、效果和说明。不纳入剧情、语音、物品库、商店、地图等独立资料功能。

数值数组保留原始顺序：属性成长按等级、技能参数按技能等级、弧盘效果按精炼档位。说明中的 `{0}` 等对应同条目 `values` 内的 `id_value`；不猜测或提前替换。百分比保留字段表示，敌人抗性是小数比例。缺失值是 `null`，网页显示“未提供”。国服轮换环线按既定日历补齐日期，日期结束日为下一期切换日，统一北京时间05:00切换；仍缺日期的未知赛期不推断开放状态。空幕 `slots` 的 -1 表示不可用格、0 表示可装配格，网页按原矩阵显示布局。材料按目标等级/突破阶段列出单次消耗，不是累计消耗；`items` 附中文材料名和数量，`items_id` 与 `amount` 保留原始配对。角色 `materials` 包含 `breakthrough` 和按技能分组的 `skills`，弧盘 `materials` 为突破阶段列表。角色 `experience` 提供1–80级配置对应的 `required_exp`（经验需求），配置等级表示升入的目标等级，等级1的配置不计入从1级开始的升级消耗。网页不再单列等级经验章节；JSON、Markdown、CLI 支持 `section=experience` 读取完整经验表。角色工具区提供带角色与推荐卡带参数的空幕计算入口。数据查询不自动生成配队强度结论。

## CLI

下载 `/guide/cli.py` 为 `nte-guide.py`，只需 Python 3.10+ 标准库；不会导入 Flask、初始化数据库或要求登录。

```bash
python nte-guide.py --base-url http://127.0.0.1:5001 search 黑羽
python nte-guide.py get characters 1042 --section abilities --format markdown
python nte-guide.py get characters 1042 --section equip_slots
python nte-guide.py get characters 1042 --section materials --format markdown
python nte-guide.py list arcs --limit 10 --offset 10
python nte-guide.py list challenges
python nte-guide.py get challenges Abyss_14 --stage 12
python nte-guide.py schema
```

默认访问 `http://127.0.0.1:5001`，可设置 `NTE_GUIDE_URL` 或在子命令前传 `--base-url`。生产环境使用实际网站根 URL。输出默认是 JSON；`get --format markdown` 输出纯文本。HTTP、网络和解析失败向 stderr 输出错误并返回退出码 1，参数错误返回 2。

## 更新与代码边界

`static/guide.js` 仅按等级索引读取已有数值并更新表格，不推算伤害或成长公式；网页初始等级由服务端渲染，前端只展示当前等级，完整数组通过 JSON/Markdown/CLI 读取。空幕使用一句完整的驱动特化说明和格位图，不再重复列出名称、数值、百分比属性卡片；养成计算中的材料按目标等级分组，以带稀有度底色、WebP图标、数量与名称的卡片展示；角色与弧盘顶部材料概览复用卡片并隐藏数量。

`service.py` 只读取本地 `data/catalog.json`、好感度快照和图片／角色元数据绑定，提供检索、摘要、章节选择和 Markdown；`__init__.py` 仅做路由适配和参数校验；模板、样式均由本模块拥有；`cli.py` 是可独立下载运行的 HTTP 客户端。

```bash
python scripts/refresh_guide_data.py
# 系统证书信任由 curl 管理的环境：
python scripts/refresh_guide_data.py --transport curl
```

刷新使用 Nanoka 完整条目及 NTEData 技能参数/伤害表；默认通过已登录的 `gh` 只读获取技能与等级经验表，也可传 `--ntedata-dir /path/to/tables` 使用本地 `DT_GameplayAbilityTipsData.json`、`DT_SkillDamageData.json` 与 `DT_CharacterUpgradeDataTable.json`。所有联网仅发生在显式刷新阶段。

刷新先完整读取索引与每个详情，校验 ID、技能、效果和关卡数量，清理游戏富文本和非展示资源引用；全部成功后原子替换数据文件，失败保留原数据。不关闭 TLS 校验，不记录来源台账。服务进程读取后缓存数据，刷新后重启服务生效；请求和启动不执行同步。

## 验证

```bash
.venv/bin/python -m unittest tests.test_guide tests.test_module_routes -v
```

浏览器匿名验证主页入口、四类索引、搜索/筛选、空结果、角色和弧盘详情、技能等级滑块及属性/精炼滑块的首尾数值、空幕格位及空幕计算跳转、材料概览、完整经验数据接口、挑战单关、Markdown 与 CLI 下载。完成后确认本机与局域网 5001 端口均可访问。

## 本地角色图片

共享资源位于 `app/static/images/characters/`，按 `avatar/`、`portrait/`、`fullbody/` 分类；`index.json` 只维护角色 ID 与本地图片绑定。当前25个角色条目各有256×256头像、沿用既有画布的默认立绘、2160×2560高清立绘，共75张WebP。复用原有合适图片，补齐缺失文件，不覆盖其他模块原有引用。角色详情顶部只显示右侧高清立绘，不显示左侧头像、查看高清立绘按钮或Markdown／JSON按钮；立绘使用完整展示高度，不预留说明按钮区域；索引卡片显示头像。图片全部由本站静态路径提供，不依赖外部图床或另一个项目目录。`assets` 字段同时出现在角色 JSON/Markdown 与 CLI 详情中。高清资源保留原生构图，部分原图触及底边，不保证完整到脚。

## 招式分组与命中明细

角色详情顶部展示属性与材料概览，正文按技能、觉醒、共鸣、机制词条、空幕、好感度等章节展示，机制词条紧随共鸣；左侧目录使用相同顺序。角色技能按 `groups` 展示具体招式，每组保留描述、倍率和其他参数；可明确按名称对应的参数进入对应招式，多招式共用规则集中展示。无法唯一对应描述阶段的伤害参数按其自身展示名称独立成组，不猜测挂到某一段。每个技能的等级滑块同步更新所有招式的倍率与明细，页面不再额外提供全部等级展开表。

导入器通过技能参数中的伤害效果引用连接单次命中条目。`param_rows[].hit_details` 包含攻击/防御/生命倍率、`harmony_gain`（基础环合）、`energy_gain`（基础终结回能）、`stagger`（基础倾陷）和展示式可明确识别的重复次数。数值为单次命中基础值，不含面板效率、被动、觉醒、环合机制或其他状态修正；不按段数平均分配，不把环合强度/倾陷强度当成资源量。重复次数不是自动计算出的实际总命中次数。未提供的数据使用空值，不用零补齐；没有伤害效果引用的纯机制参数不伪造命中明细。

## 索引与弧盘展示

角色索引以属性图标显示属性，以四星A／五星S图标显示稀有度，保留无障碍名称。弧盘索引按固态、液态、气态、等离子、聚合分组，图片从本站 `app/static/images/arcs/` 读取WebP。显式同步图片命令：`python scripts/sync_guide_arc_images.py`。弧盘详情先展示属性成长，再展示效果；精炼滑块直接替换效果正文中的数值，所有数字使用绿色，不另列效果参数表。结构化详情仍保留各精炼档位数组。

## 角色弧盘适配与定位

角色页顶部用属性图标与稀有度图标（四星A、五星S）替代对应文字，稀有度图标放在 `app/static/images/rarities/`；弧盘适配类型及其图标与属性、稀有度放在同一行，角色定位标签另行展示并显示对应图标。`data/character_metadata.json` 保存按角色ID绑定的 `arc_type` 和 `char_tags`；图标使用本站 `app/static/images/arc-types/` 与 `character-tags/` 下的WebP。JSON、Markdown和CLI详情同时提供适配类型及标签图标路径。显式同步命令为 `python scripts/sync_guide_character_metadata.py`；同步按角色配置中的类型枚举与弧盘类型表连接，不根据专武名称推断角色适配。网站启动和请求只读取本地数据。

## 角色养成计算器

养成结果以紧凑材料卡显示，桌面图标90px、窄屏76px；甲硬币在前，经验书按经验值从高到低紧随其后，再列其他材料。网页不显示所需经验和溢出经验汇总，内部经验书配给、费用计算及API经验字段保留。

角色详情顶部提供工具入口，进入 `/guide/characters/<id>/cultivation`。输入当前与目标角色等级及技能／天赋／都市技能基础等级；突破阶段按等级自动判断，仅在20、30等临界等级选择是否已突破。技能以“普通攻击”“变轨攻击”等类型名显示。服务端汇总角色经验、经验书、甲硬币、突破材料、技能材料和都市技能使用的方斯等资源。0级表示尚未解锁的天赋或都市技能；普通战斗技能从1级起。觉醒／共鸣带来的额外技能等级不参与材料计算，觉醒、共鸣与弧盘混频不在此计算器范围内；弧盘等级与突破可选一并计算。

突破阶段与等级上限按角色配置判断；材料只累计当前之后至目标的阶段，技能须满足对应突破条件。角色经验按升入目标等级的需求逐级相加，不计当前等级内已累计经验；每段到达突破上限后分别用经验书补齐，按高档优先、低档补齐组成使用方案，显示溢出经验。甲硬币合计包含这份方案的经验书使用费用，不假设溢出退费。不扣除库存，不自动折算合成，也不估算副本次数或体力。

材料展示绑定存于 `data/material_metadata.json`，WebP复用 `app/static/images/item/` 或放在 `app/static/images/materials/`；`presentation.py` 只读取图标与稀有度配色，不改变材料数量。显式同步：`python scripts/sync_guide_material_images.py`。页面初始结果由服务端渲染，无需脚本加载材料。

`cultivation.py` 只对本模块离线数据计算；GET表单由服务端计算，`static/cultivation.js` 维护等级与突破选项、已同步练度预填和可选弧盘表单，材料计算仍在服务端完成。`POST /api/guide/characters/<id>/cultivation` 接收 JSON 对象，字段为 `current_level`、`target_level`、`current_ascended`、`target_ascended`（临界等级是否已突破，布尔值或0／1） 和 `current_skill_<index>`／`target_skill_<index>`；非临界等级忽略突破选项，旧突破次数字段与等级内经验字段不参与计算。返回状态包含自动推定的突破次数。技能索引按角色 `materials.skills` 顺序，缺省按1级未突破至80级满突破、所有技能满级计算。返回 `data.state`、`experience`、`book_experience`、`overflow_experience`、`level_up_gold`、`total_gold`、`items` 和 `breakdown`。无效练度返回400，未知角色返回404。

刷新还读取角色表、物品表及突破表的经验书和等级上限；`--ntedata-dir` 目录需同时包含 `DT_Character.json`、`DT_ItemConfig.json` 和 `DT_CharacterBreakthroughDataTable.json`。本地快照只保留实际参与计算的字段，不在请求时联网。

验证覆盖全部25角色、满养成资源汇总、经验书使用费、忽略等级内经验、自动推定突破阶段、临界等级原地突破、当前等于目标、无效等级／突破／技能及网页表单。浏览器检查输入、计算结果、错误提示和空幕跳转。

## 好感度

角色详情的“好感度”章节展示每级配置的好感值上限、文字奖励与解锁、满好感属性加成、喜好及专属礼物、每份赠送好感值和普通次数限制标记。没有角色好感度配置的条目显示未提供，不套用其他角色数据。这里不记录玩家当前好感度。 浔、达芙蒂尔、九原分别通过明确的角色编号映射1052→1029、1054→1005、1055→1014读取好感度等级、喜好与专属礼物；两个零条目仍无好感度配置。

礼物表采用“礼物、好感值、单价、总花费”四列，礼物名称以tooltip展示购买渠道，支持悬停、聚焦和点按；不新增渠道列。仅取方斯最低商店标价计算单价与总花费，按总花费由小到大排列；没有方斯售价的礼物排在最后，单价与总花费显示“—”，不展示购买份数；影券不展示其他货币的详细价格。JSON 保留全部商店与商品配置。无价格不按免费计算。 每个角色内按好感档位比较最低单价，所有并列最低且满足以下门槛的礼物单价数字用绿色提示推荐：100好感低于450方斯、200好感低于4000方斯、400好感小于等于12000方斯。其他档位与无方斯售价的礼物不标色。

`affinity.py` 在读取本地快照后计算 `total_affinity`（所有等级好感值上限之和）、`required_gifts`（总好感值除以每份赠送值，向上取整）与 `costs`（仅方斯的 `unit_price` 和 `total_cost`）。总花费为从零到配置满好感、只赠送该礼物的理论花费，不计主角赠礼加成、折扣、聊天／任务好感或库存和购买限制。缺少等级配置时不推算总花费。

网页、JSON、Markdown 与 CLI 共用 `data/affinity.json`，支持 `section=affinity`，CLI 示例：`python nte-guide.py get characters 1042 --section affinity`。全文搜索可按礼物名检索角色。

显式刷新：`python scripts/sync_guide_affinity.py`；支持 `--ntedata-dir` 读取本地同名角色好感度、礼物、商品、商店、物品、货币、食物、客户端掉落组、掉落序列与好感属性修正表。文字奖励按等级奖励引用逐层解析，保留物品数量与解锁说明；满好感属性通过角色的 `MapModifyData` 读取实际修正，不按角色定位推断。缺少关联配置时刷新失败并保留旧快照。刷新验证全部条目后原子替换快照；请求和启动不联网。常规技能与材料刷新不覆盖此独立快照。

验证：`.venv/bin/python -m unittest tests.test_guide_affinity tests.test_guide tests.test_module_routes -v`；浏览器核对目录跳转、四列表格、总价与向上取整、仅方斯与升序排列、好感度等级及缺失配置提示。

## 轨外之境赛期日历

国服本初环线为2026-04-23至2026-05-29；未明环线从2026-05-29起进入14日轮换。澄明环线 `Abyss_12` 为2026-10-16至2026-10-30，飞芒环线 `Abyss_13` 为2026-10-30至2026-11-13，`Abyss_14` 目前显示“未命名环线”，按规律排期为2026-11-13至2026-11-27。开始日期05:00生效，结束日期04:59:59结束。未来按规律排期可能调整，不表示该环线已经开放。

`scripts/guide_challenge_schedule.py` 统一维护这14期日历，在显式刷新时覆盖不适用于国服的旧日期，并只处理已约定的赛期ID；常驻节理环线不填起止日期，未知新增赛期不自动延伸。可单独运行 `python scripts/guide_challenge_schedule.py` 更新本地日历；常规 `refresh_guide_data.py` 自动应用相同规则。网页、JSON、Markdown、CLI使用相同日期。

日历验证：`.venv/bin/python -m unittest tests.test_guide_challenge_schedule tests.test_guide -v`，覆盖首期特例、14日连续性、未知赛期保留空值、国服日期与末期名称在各格式中的一致性。

环线详情的共鸣为单个对象，独立显示 `resonance.name` 与 `resonance.description`；角色共鸣仍按列表展示。测试覆盖全部15个环线的共鸣名称与完整效果正文。

## 角色相关轴与配装收益

角色详情在技能前提供养成计算器、排轴与配装、空幕计算三张工具卡片，材料章节保留轻量养成入口。“排轴与配装”进入独立的 `/guide/characters/<角色ID>/builds` 页面；角色ID在URL中固定，公开轴、我的轴、收藏和队友筛选始终包含该角色。匿名用户可以浏览包含该角色的公开轴，按一位队友筛选、按最新／DPS／点赞／收藏排序和加载更多；登录后可切换自己的轴或收藏。卡片显示队伍、觉醒数、保存版本、备注、轴长与保存时的DPS，可原地打开排轴模拟器的原生时间轴预览，再用该轴比较配装。无匹配结果、读取失败和未登录个人范围均有独立提示。

`static/related_axes.js` 只通过 `/api/shaft/catalog`、`/api/shaft/market`、`/api/shaft/me/axes`、`/api/shaft/me/favorites` 和既有轴详情接口组合展示；不导入排轴领域服务，不运行伤害公式。角色按目录真实名称匹配，两个零条目1046／1051明确对应排轴的主角；不按属性或列表位置猜测。权限仍由排轴接口校验。

页面不提供独立的配装收益按钮；找到轴后，通过轴卡片或其预览中的“用此轴比较配装”进入 `/guide/characters/<攻略角色ID>/axes/<轴ID>/compare`。比较固定该轴与当前角色，不提供另选轴、浏览器草稿或示例轴入口；比较页面路由由攻略模块持有，计算仍复用排轴浏览器引擎；候选弧盘按S级1精、A级／B级5精处理，参见其README；此页不保存玩家练度、修改原轴或写入浏览器排轴草稿。问答尚未接入。

验证：`.venv/bin/python -m unittest tests.test_guide tests.test_guide_builds tests.test_module_routes tests.test_shaft_comparison -v`。浏览器检查有轴／空结果角色、队友筛选、个人范围、原地预览及带轴跳转试算；两个零条目的映射均须核对。

详情目录固定在视口左侧，不占正文布局宽度；正文沿用1240px居中容器和对称留白。视口不超过1680px时，目录通过左下角按钮展开为浮层；展开与收起都不移动或压缩正文。

## 环线怪物展示

环线详情顶部显示共鸣名称与完整效果；关卡按编号倒序排列，按半场和波次展示怪物贴图、等级、数量、弱点；仅可吞噬怪物在卡片右上角显示鬼郎丸圆形图标（悬停或无障碍名称为“可被鬼郎丸吞噬”），不显示吞噬性质文字。单体生命值直接显示在怪物概览中，使用千分位数字；攻防、倾陷、抗性与出场条件默认折叠，可逐项展开；未标注的吞噬性质不显示图标，不按体型推断。`data/monster_metadata.json` 按怪物ID绑定本站WebP路径和吞噬性质，不按中文名称推断图片；图片由本模块 `static/monsters/` 持有，网站运行不依赖其他仓库或外部图床。JSON、Markdown与CLI同步提供 `image` 与 `swallowable` 字段。

验证：`.venv/bin/python -m unittest tests.test_guide_challenges tests.test_guide_challenge_schedule tests.test_guide tests.test_module_routes -v`；浏览器检查共鸣、关卡筛选、贴图、吞噬标记、默认折叠与展开，以及窄屏布局。

技能招式说明与技能／被动正文统一使用16px字号，倍率与命中数据表保持独立字号。

轨外之境目录在每次请求时按系统当前时间判断限时环线：北京时间开始日05:00（含）至结束日05:00（不含）高亮卡片边缘。常驻、日期缺失、尚未开始和已结束的环线不高亮；判断不缓存，刷新页面即可按新时间更新。验证包含05:00切换边界和不同时区的等价时刻。

攻略目录顶部直接展示分类导航、搜索与筛选，不显示宣传大标题和重复数量区；分类导航保留各类条目数，首屏优先展示资料列表。

弧盘目录复用本地筛选：星级（三星B、四星A、五星S）与弧盘类型各组单选图标，首次加载全部51张卡片，筛选和名称／效果搜索不发查询请求、不重置滚动。空类型分组自动隐藏，分组数量随结果更新。弧盘目录、详情与筛选均使用同一套B／A／S图标；`arc_type` 参数在弧盘目录表示弧盘类型ID。


## 玩家练度与站内同步

养成页保留匿名手填计算；“我的练度”仅在登录本站后可用。短信验证直接由网站后端对接塔吉多关联的老虎用户中心，无需独立登录服务。登录后选择游戏账号并显式同步，`GuideGameProfile` 按本站玩家与游戏账号隔离，同步成功后同步事务写入角色列表；失败或空结果不覆盖旧资料。页面可切换已保存账号、填入当前练度、查看已装备资料和删除该账号的本站数据。

“查看已同步技能与装备”显示保存的技能展示等级与已装备弧盘的混频数；弧盘混频只读取 `equipment.fork.slev` 的1–5档，不使用角色级字段代替，缺失或越界时显示未提供。技能名称按忽略大小写的完整本地技能ID匹配，未匹配时保留接口名称或ID；没有等级时明确提示未提供，计算预填仍保留基础等级核对。

数据库只保存当前角色等级、技能展示等级、觉醒／混频、好感经验、面板属性和已装备弧盘、空幕卡带及其属性，不保存手机号、验证码、访问／刷新凭据，也不收集背包或额外账号资料。数据当前仅供养成预填；不接入配装推荐。同步时间为实际快照更新时间，不表示游戏内实时状态。

技能按忽略大小写的完整技能ID与本地材料项匹配，不按列表顺序、名称或类型猜测；未匹配、重复或超出基础等级范围时恢复初始值并提示手填，不将展示等级钳为上限。接口不能确认角色临界等级是否突破，保留“是否已突破”选项。接口技能等级可能包含额外加成，使用同步数据后须核对基础等级与突破状态再计算。目标练度仍由用户选择。每项技能以两行滑块分别设置当前与目标等级，数值随拖动或键盘操作实时显示；目标低于当前时自动提高到当前等级，同步预填同时更新滑块与数值。

`game_provider.py` 只处理有界 HTTPS 协议、短信认证和角色读取，拒绝重定向，不记录请求体、凭据或上游响应；`game_sync.py` 管理10分钟进程内临时验证会话、随机会话标识、账号归属、验证码尝试及发送频率限制。成功即移除会话；不保存长期登录态，每次重新同步游戏数据需短信验证。当前部署使用单进程多线程；切换多进程时需先把临时会话和限流迁移到共享的带TTL存储。普通网页、网站启动及养成计算均不自动访问上游。

`user_data.py` 持有模块专表、字段白名单与纯预填适配；`user_routes.py` 仅做登录校验和调用服务。`GET /api/guide/me/profiles` 读取自己的资料，`DELETE /api/guide/me/profiles/<role_id>` 删除自己的资料，`GET /api/guide/me/profiles/<role_id>/characters/<id>/prefill` 返回可填写值和核对提示。`POST /api/guide/me/sync/sms` 接收 `mobile`，`POST /api/guide/me/sync/verify` 接收 `session_id` 与 `code`；多账号时返回 `select_role`，再以 `session_id` 与 `role_id` 调用 `POST /api/guide/me/sync/select-role`。以上接口均要求本站 Bearer 登录并返回 `Cache-Control: no-store`，不向浏览器返回游戏凭据。

网站在发送短信前检查加密依赖（含原生组件加载）；缺失时返回503，不发送验证码。私有资料接口的意外错误返回带错误编号的JSON，日志只记录编号、接口、异常类型与代码位置，不记录异常原文或请求资料。页面遇到代理HTML错误页或失效登录时显示可读提示。

网站依赖增加 `pycryptodome`，更新时须安装 `requirements.txt`；运行账户需要能通过 HTTPS 访问游戏服务，并保留正常证书验证。上游协议／风控可能变化，错误时保留旧数据并显示重试提示；首次真实短信和账号字段仍需在实际账号上验收，不以合成接口测试代替。

验证：`.venv/bin/python -m unittest tests.test_guide_user_data tests.test_guide tests.test_module_routes -v`，覆盖匿名计算与私有接口隔离、多用户／多账号归属、重连数据库、白名单保存、旧快照保护、短信频率与尝试次数、技能缺失／越界、临界突破及协议加密。浏览器核对同步弹窗、账号选择、预填确认、计算结果和390px窄屏布局。

角色与弧盘目录卡片不显示顶部编号行和右下角跳转箭头；图片居中，属性与稀有度图标竖排在右上角，名称、别名和底部编号居中对齐；底部编号在同排卡片中对齐，整张卡片仍可点击进入详情。

攻略页导航固定在视口顶部，滚动时保持可见并为正文与章节锚点预留空间。左侧 NTE TOOLS 返回工具站，其后为详情／工具页的返回按钮：详情页返回所属分类目录，养成与排轴配装页返回对应角色资料。右侧“首页”进入数据库目录，“登录”在本站登录有效时显示为“账户”并进入账号设置；窄屏保持单行。登录入口保留当前页面地址；未登录或登录已失效时点击“同步游戏数据”先进入本站登录页，成功后返回原页面。

轨外之境目录与列表接口按现有环线顺序倒序展示，较新的轮换环线在前，常驻节理环线在最后；筛选和分页沿用同一顺序。

角色详情将属性成长与顶部生命、攻击、防御合并为同一属性区，默认满级，可用等级滑块同时切换所有面板属性，百分比标记保持显示；下方不再重复属性表。下一行只显示角色突破与技能升级的独特材料，各材料系列按实际品质保留最高档，去重后展示图标和名称，不显示数量，也不列货币、经验书及通用都市技能材料。角色详情不再提供“升级材料”“等级经验”章节；完整材料、经验及属性数组仍可通过JSON／Markdown／CLI读取，养成计算器保留原消耗计算。

轨外之境详情不展示奖励章节或“关卡编号”提交表单。关卡使用与弧盘目录相同的单选筛选按钮，倒序排列；默认仅显示第12关，不含第12关时显示最高编号关卡。切换只在页面内显示／隐藏已加载的关卡，不请求、不跳转、不滚动，并同步地址栏 `stage` 参数；带 `stage` 的详情链接保留全部切换按钮并选中对应关卡。JSON与Markdown仍保留奖励和原有单关读取契约，不受网页隐藏规则影响。

弧盘目录在每个类型分组内按稀有度从高到低展示（S、A、B），同级保持原有顺序；切换筛选后沿用该顺序。

怪物按原始配置条目分别展示，不合并不同ID；名称右侧以小字显示怪物ID，窄屏时可自然换行。生命值始终表示单只怪物的生命，不乘数量；原始快照、JSON、Markdown保留独立配置条目。

角色与弧盘详情共用顶部属性／升级材料概览：角色属性区不显示异能名；弧盘属性默认满级，使用同一等级滑块切换，百分比属性保留标记。弧盘下一行只展示去重后的最高品质突破材料图标与名称，不显示数量、低档材料或货币；正文与目录不再另列属性和升级材料章节，属性入口指向顶部概览。原始属性与材料数据仍通过JSON、Markdown和CLI保留。

角色详情首行采用两栏，左侧显示角色身份、定位与立绘，右侧显示等级滑块、属性数值和最高品质材料；左栏立绘贴右侧撑满展示高度，图标与文字在立绘上层，允许遮挡立绘。视口不超过900px时改为上下排列。弧盘页沿用原有纵向属性概览布局。

## 技能颜色与名词提示

角色技能、觉醒、共鸣和机制词条按原描述标记显示属性颜色、强调与斜体；明确带名词ID的文字支持鼠标悬停、键盘聚焦和手机点按查看注释，Esc或点击外部关闭。提示限制在视口内，长说明可滚动。未绑定的文字保持纯文本，不按名称猜测链接。

`data/character_rich_text.json` 保存25名角色的结构化描述片段、词条说明和技能样式颜色；运行时只渲染允许的文本／颜色／词条字段，未知标签不作为HTML执行，现有纯文本JSON、Markdown和倍率不变。显式刷新：`python scripts/sync_guide_rich_text.py`，在更新攻略正文后同步运行；可用 `--characters-json` 读取按角色ID映射的本地原始条目、`--style-table` 读取本地技能富文本样式表。同步核对全部现有招式正文与名词引用后原子替换快照；不匹配时保留旧快照，请求和启动不联网。

验证：`.venv/bin/python -m unittest tests.test_guide_rich_text tests.test_guide -v`；浏览器检查属性色、长词条提示、键盘与触屏操作、视口边缘及技能等级切换。

角色详情的技能等级默认10级；上限低于10级时取其最高等级。服务端初始倍率、命中明细与等级标记保持一致，后续拖动仍按实际等级数组更新。

相关轴预览通过同源嵌入 `/shaft/rotation?preview=<轴ID>&embedded=1` 复用排轴原生轨道、时间刻度、滚轮缩放、显示全部、DPS／轴长和伤害占比。嵌入模式只加载目录与有权访问的轴，不初始化编辑器、不读取或写入排轴草稿，不显示保存按钮；关闭预览会卸载嵌入页。原有公开／私有轴权限继续由排轴接口校验。

角色觉醒卡片展示对应图标与名称，不显示“觉醒＋序号”；共鸣展示图标及“共鸣 Lv.3”格式的等级标签。图标由模块 `static/passive-icons/` 持有，`data/passive_icons.json` 按角色、章节与名称绑定；显式刷新使用 `python scripts/sync_guide_passive_icons.py`，请求和启动不联网。 明音凛的六个觉醒图标当前为空，缺失时只显示名称，不生成破图或替用其他角色图标。

怪物详细数据中的抗性按配置与数值标色：明确标记的强化抗性数值使用橙色，并提供“强化抗性”提示；其余数值并非全相同时，最低抗性（含并列最低）使用弱点绿色。缺失值不标色；没有强化标记且全部相同时不标色。数值仍按原有小数比例显示。

安魂曲觉醒A的参数按普通攻击当前等级下“噩梦”伤害倍率乘以4/3显示，默认跟随10级，展示最多保留两位小数。倍率提示标题显示“噩梦伤害倍率”，不展示计算式；等级联动与实际倍率保持原有计算。`description_parameters.py` 提供显式技能ID绑定与逐级展示数组；网页保留原有颜色和名词注释，对应技能滑块同时更新觉醒正文与提示中的等级／倍率。娜娜莉觉醒C跟随普通攻击，灵可觉醒D跟随极轨终结，逐级倍率由本地 `data/awakening_damage.json` 提供。三处倍率均可悬停、聚焦或点按查看依据技能、当前技能等级、当前倍率和1级倍率。显式更新追加伤害数组：`python scripts/sync_guide_awakening_damage.py --damage-table /path/to/DT_SkillDamageData.json`。JSON／Markdown保留说明模板及 `parameters`，不修改技能原始倍率数组。未建立明确参数绑定的占位符不猜测补值。

角色详情空幕特化说明使用16px正文，格位说明使用14px；空幕计算入口位于顶部工具区，不插入格位正文。跳转使用 `/kongmu?character=<正式角色ID>&cartridge=<推荐卡带ID>`，预选对应角色和推荐卡带。`data/kongmu_links.json` 保存离线映射；显式刷新 `python scripts/sync_guide_kongmu_links.py`，支持 `--plan-table` 与 `--equipment-table` 读取本地表，请求和启动不联网。未知角色不生成跳转，缺推荐配置时只携带角色，不推断推荐。

怪物头像覆盖当前44个配置ID，每个ID读取其明确配置的图片并保存到模块本地；同名的普通、Snow、Rain等变体使用各自绑定，不进行同名兜底。缺少ID绑定时显示暂无贴图，不借用同名怪物。显式同步命令：`python scripts/sync_guide_monster_images.py`，逐ID核对原始图标字段，拒绝缺失或冲突绑定；所有图片验证成功后才替换绑定文件，不改动关卡数值。

角色技能标题左侧显示对应的本地技能图标，覆盖普通攻击、变轨技能、极轨终结、援护、被动、特性和都市技能；桌面56px、窄屏48px，等级滑块保持独立。`data/skill_icons.json` 按角色ID与技能ID绑定，资源位于模块 `static/skill-icons/`；无有效绑定时仅显示原标题。显式刷新 `python scripts/sync_guide_skill_icons.py`，支持 `--tips-table`、`--city-table` 读取本地技能表，使用 `cwebp` 转码；网站启动和页面请求不联网取图。 明音凛“随心拨弦”暂无可用独立图片，该项保留文字标题。

比较页固定来源角色，不再提供试算角色或统一精炼选择框。每行弧盘显示本次采用的精炼档位，原轴100%基线保留原轴实际精炼。个别排轴条目缺少稀有度时，仅在比较页内按本地攻略弧盘的同名明确配置补齐；无法确定时停止试算，不从攻击值猜测品质。

## 全站深境

`/guide?kind=endless` 按1–50、51–100、101–150、151–200层分为四组；每组详情复用环线怪物卡片，按实际层号升序切换，默认显示组内首层。切换只显示／隐藏已加载楼层并同步 `stage`，分享链接和刷新保留选择；组间通过顶部层数范围跳转。敌人保留独立属性配置ID、单只生命、等级、数量、攻击、防御、倾陷上限与抗性。全部58个敌人模板按深境专表绑定名称、弱点及强化抗性；缺少图标或吞噬标记时保留未提供状态，不借用同名敌人的属性；基础属性不含局内增益。各层限时120秒，展示原始计分目标，不推断开放状态。

网页、搜索、JSON、Markdown与CLI共用独立本地快照 `data/endless.json`。JSON列表 `/api/guide/endless`，详情 `/api/guide/endless/EndlessTrial_51_100?stage=100`；Markdown `/guide/endless/EndlessTrial_51_100.md?stage=100`；CLI `python nte-guide.py get endless EndlessTrial_51_100 --stage 100`。层号越组或不存在返回404。

显式刷新：`python scripts/sync_guide_endless.py --ntedata-dir /path/to/NTEData`。读取解包目录的 `DataTable/CloneSystem/DT_EndlessTrialMonsterInfo.json`、`DataTable/PackData/DT_MonsterPackData.json`、`DataAssets/DataAssetSet/EndlessTrial/DT_EndlessTrialFloor_1.json`、`DT_EndlessTrialMonsterConfig.json` 及 `Text/PlayerSkillDes/EndlessTrail_Des.json`。名称与抗性标记来自深境怪物表，实体模板路径的功能绑定 `data/endless_identities.json` 提供本地图标和已知吞噬标记；可用 `--identities` 指定更新后的绑定。核验200层齐全、所有属性引用、固定等级与零成长项后原子替换快照；非零成长参数拒绝导入，不猜公式。常规角色／环线刷新不覆盖该独立快照；请求和启动不联网。刷新后重启服务生效。

验证：`.venv/bin/python -m unittest tests.test_guide_endless tests.test_guide tests.test_guide_challenges tests.test_module_routes -v`；浏览器核对四组入口、组首／末层、层数切换、详情展开和旧环线。

深境逐层机制包含推荐属性、强化抗性属性、节点起始／关底标记、备选角色数和角色栏位数。每10层关底显示当层配置的5组正向效果与反向效果，逐组保留配对关系，并显示默认正向效果数量及关联楼层的数量加成。敌人属性表使用固定属性包中的抗性，不额外按强化抗性标记猜测叠加值；关底可选的全属性抗性、生命、防御等修正不默认计入敌人数值。JSON与Markdown保留完整规则和每个效果的作用对象；不根据配置字段推断未明确提供的选择持续时间或生效时机。

安魂曲“恶魔的礼物”在详细命中数据中展示46种习得技巧、52项伤害效果；命中项包含变身名称和攻击说明，攻击倍率跟随变轨技能等级切换。多伤害效果分别列出，不合计为一次攻击；未确认重复次数时保留未知。`data/lacrimosa_forms.json` 维护变身技能与专用伤害效果的显式绑定，常规离线刷新从伤害表重建倍率；缺失或归属不符时中止刷新，不使用怪物自身伤害。网页、JSON、Markdown和CLI共用这些明细。


## CLI 个人资料与机器人配装图

独立CLI新增 `me [--role-id <游戏账号ID>] [--character <角色ID>]`，通过 `NTE_GUIDE_TOKEN` 环境变量携带本站登录凭据，读取 `/api/guide/me/profiles` 后只在当前用户结果内筛选。返回 `{"data":[profile,...]}`，包括同步时间、角色当前练度与已装备弧盘／空幕卡带；没有有效凭据则失败，不通过仅提供账号ID的方式读取他人资料。私有请求拒绝重定向，错误不回显上游响应或凭据，公共 `list/search/get/schema` 不携带 token。

机器人可信运行时按当前消息事件的发送者QQ号读取本人已保存配装；只读 token 桥接和一图流绘图实现在 `qqbot_fairy` 仓库的 `src/plugins/`，随机器人代码发布，本站不再保留这两个副本。本站继续维护独立 CLI 与按网站用户隔离的鉴权 API，部署边界见 `plugins/README.md`。图片仅展示已同步快照，不推断基础技能等级或评分；不自动同步游戏，公共资料的离线刷新机制保持不变。

验证：`.venv/bin/python -m unittest tests.test_guide_bot_tools tests.test_nte_account_db tests.test_guide_user_data tests.test_guide -v`，使用合成资料检查 CLI 认证、所有权、私有错误脱敏、注册兼容和旧公共命令；只读桥接与 PNG 渲染测试在机器人仓库执行。

同步错误回归：`node --test tests/test_guide_private_response.cjs`，覆盖HTML错误页、失效登录、JSON错误及删除成功响应。

## 账号练度页面

`/guide/account` 为独立账号练度页，入口位于目录分类行“全站深境”右侧。登录后复用 `/api/guide/me/profiles` 读取本人已保存账号，可切换账号，点击角色打开配装详情弹窗，展示等级、觉醒、技能、面板与已装备弧盘／空幕卡带明细，并跳转角色资料或养成计算。角色名、头像及技能名复用本地攻略数据；未知角色保留已同步名称和ID，不猜测映射。角色摘要只显示等级和觉醒阶数；不将含义未核实的角色 `slev` 显示为共鸣或激活觉醒。页面显示快照同步时间，只有主动点击同步并完成短信验证才拉取游戏数据；同步弹窗和调用逻辑与养成页共用，删除仍使用现有本人资料接口。

验证：`.venv/bin/python -m unittest tests.test_guide_account tests.test_guide_user_data tests.test_guide tests.test_module_routes -q`；`node --test tests/test_guide_private_response.cjs`。浏览器检查分类栏入口、账号切换、弹窗详情、未知角色、同步后选择账号、删除空状态、未登录回跳及390px窄屏。

相关轴预览按视口分配弹窗高度，标题、比较入口和底部统计保留空间；嵌入时间轴默认显示全部，四条轨道随可用高度分配，初次打开不产生嵌套滚动条。主动放大后保留时间轴横向滚动。

账号练度弹窗上方采用角色立绘与面板两栏，技能图标和同步等级位于其下；下方展示弧盘及核心／驱动块卡片，主属性与附加属性分别排列，不计算评分或评级。立绘、技能和弧盘图片只按本地明确ID绑定复用，缺失数据保留未提供；支持关闭按钮、Esc、点击遮罩关闭，手机两列展示空幕明细。

角色弹窗主要面板固定优先展示“通用伤害增强”和本地角色属性对应的“属性异能伤害增强”，与其余基础属性合计展示至多8项；被移出的属性仍保留在“查看其余属性”中，不重复展示，不从装备另行汇总或补算数值。

角色与弧盘目录使用紧凑卡片：视口宽度1100px起每行6张，800–1099px每行4张，600–799px每行3张，更窄时每行2张；图片、图标与内边距同步缩小，轨外之境卡片列数保持原样。

目录卡片样式地址带文件版本参数，样式更新后浏览器请求新版本。

练度弹窗的弧盘图片按忽略大小写的完整ID匹配；ID不匹配时，仅按本地唯一名称（兼容外层书名引号）定位图片。`data/equipment_images.json` 为核心卡带和12种驱动形状绑定本模块 `static/equipment/` 图片，驱动模板及品质变体按明确ID映射形状，同型号多形状时不按名称猜测。图片只在本站加载；未知条目保留文字，图片加载失败隐藏破图。验证：`node --test tests/test_guide_equipment_images.cjs`，浏览器核对实际图片加载、不同驱动形状及手机排版。


## 推荐弧盘与联合养成

角色详情顶部的弧盘类型可点击进入该角色首个推荐弧盘。`data/arc_cultivation.json` 按角色保存单个推荐弧盘，按弧盘保存等级上限与经验配置，并保存经验染剂的经验值与使用费用；推荐严格取配置列表首项，不按专属归属或个人装备推断。缺少推荐时类型标签保持普通文字。

养成页在角色当前／目标练度下方提供“一并计算弧盘（可选）”，默认关闭。开启后可选择一张该角色适配类型的弧盘，填写当前／目标等级及临界等级突破状态。匿名默认预选首个推荐弧盘；有已同步数据时优先填入已装备弧盘与等级，可确认的弧盘突破阶数同时填入。选换其他弧盘从1级开始，目标为上限。升级按每个突破区间分别使用染剂，计算使用费用与突破材料，不计等级内已有经验、库存、合成、混频或消耗其他弧盘。角色与弧盘的同种材料及甲硬币合并，分项消耗保留独立弧盘明细。

进入养成页且本站登录有效时，自动选择包含当前角色的已保存游戏账号并填入练度；没有记录时继续手填。切换游戏账号会填入该账号保存值，保留“使用已同步练度”按钮以重新应用。用户开始手动输入后，迟到的自动预填不覆盖输入；计算结果页与带 `calculate=1` 的表单保留提交值，不重复自动填入。接口不明确的技能或角色突破状态继续显示提示，不要求勾选核对复选框，不自动同步游戏。

养成GET表单和POST接口增加 `include_arc`（0／1）、`arc_id`、`arc_current_level`、`arc_target_level`、`arc_current_ascended`、`arc_target_ascended`；未启用时忽略弧盘输入。启用时返回 `data.arc` 的独立消耗与状态，`data.items`、`total_gold`、`breakdown` 为合并结果，原角色经验字段保持角色口径。弧盘ID、类型、等级或突破无效时返回400。

显式刷新：`python scripts/sync_guide_arc_cultivation.py`；可用 `--tables-dir` 指定包含 `DT_CultivationGuideData.json`、`DT_ForkItemData.json`、`DT_ForkUpgradeData.json`、`DT_ForkBreakthroughData.json` 和 `DT_ItemConfig.json` 的本地目录。核对全部弧盘材料、费用、经验与等级上限后原子替换；随后运行 `scripts/sync_guide_material_images.py` 补齐染剂展示。请求和启动不联网。

验证：`.venv/bin/python -m unittest tests.test_guide_arc_cultivation tests.test_guide_user_data tests.test_guide tests.test_guide_account tests.test_module_routes -q`；`node --test tests/test_guide_private_response.cjs tests/test_guide_equipment_images.cjs`。覆盖全部弧盘、首次升级、原地突破、无消耗、材料合计、无效类型／等级、推荐链接和同步预填；浏览器核对自动填入、手填保留、可选表单、换选弧盘与窄屏。

“相关排轴与配装收益”标题右侧仅为有合轴第一页的角色显示“合轴收益表”按钮，进入 `/overlap?character=<合轴角色ID>` 并预选对应角色。`data/overlap_links.json` 保存攻略角色ID与已发布合轴角色ID的明确绑定，当前覆盖安魂曲、残虹、黑羽、明音凛；其他角色不显示。链接不进入双角色分页，不导入合轴领域服务；合轴第一页角色范围变化时同步更新绑定并运行 `tests.test_guide_builds`。

轨外之境列表卡片不显示左上序号、关数和跳转图标；常驻／轮换标签位于标题下一行，日期或长期开放状态位于右下角，环线ID保留在左下角。整张卡片仍可点击，当前赛期边缘高亮保留。

账号练度弹窗在“觉醒 X”气泡右侧展示同步 `awakenEffect` 对应图标，悬停、键盘聚焦或点按可查看名称。仅当效果值与本角色本地觉醒名称完全相同时匹配图标；其余显示未识别项数并保留标识提示，不按编号或觉醒阶数猜测选中项。空列表不展示图标，也不表示未激活；列表是否代表已激活尚未确认。

相关排轴卡片显示赞、踩、收藏计数；仅本站登录有效时显示操作按钮，再次点击已选操作即可取消，赞与踩按排轴接口互斥。每次写操作先校验登录态，且不传匿名收藏标识；同一条轴提交时禁用三个按钮，成功后按服务端结果更新状态和计数，失败保留原值并显示提示。自己的公开轴不能收藏，私有轴不显示评价收藏操作；在收藏范围取消收藏后移出当前列表。复用 `/api/shaft/axes/<id>/like|dislike|favorite` 的POST／DELETE，不修改排轴规则、存储或原广场匿名收藏行为。

评价收藏回归：`node --test tests/test_guide_related_reactions.cjs`，覆盖登录限制、互斥与取消、自己的轴、收藏范围移除、写失败和重复点击；浏览器以临时合成账号通过真实排轴接口验收。

相关排轴预览弹窗底部提供“打开排轴模拟器”，跳转 `/shaft/rotation?preview=<当前轴ID>` 打开完整模拟器及该轴预览；“用此轴比较配装”位于底部左侧，“打开排轴模拟器”靠右下角，窄屏自动换行并保持靠右。读取成功后才显示两个入口，读取中或失败时隐藏，继续使用原轴访问权限。

## 登录后浮动攻略助手

所有 `/guide` 下HTML页面共用右下角助手入口，包括目录、角色／弧盘／关卡、养成、账号、角色排轴与比较页。匿名或失效登录不展示；接口仍独立鉴权。桌面为非模态浮窗，页面保持可点击、可滚动；650px及以下展开为覆盖页面的移动面板，收起后恢复网页操作。可输入文字、选择或粘贴PNG／JPEG／WebP图片，单条最多2张；浏览器压缩后由后端校验格式、签名和大小，再以真实多模态内容发送模型。

会话只保存在浏览器IndexedDB，按本站玩家隔离，包含文字、附图和回答引用；不建立服务器聊天表，不写入日志。页面切换和刷新可恢复，支持历史选择、新对话和删除。新消息距上一条消息超过6小时开启新会话，旧会话仍在历史列表；后端也独立执行6小时边界。完整历史保留在浏览器，模型每次最多读取最近32条／60000字及6张用户附图，超出窗口的旧图片明确标记为未包含。存储不可用或空间不足会显示提示，不冒充已保存。

每次请求另带当前页面标题和经过过滤的页面地址，用于理解“这个角色”等指代；仅作为本轮临时系统上下文，不加入浏览器历史或用户消息。页面切换后使用新页面信息；未知查询参数不会发给模型。

配置在根目录 `.env`：`NTE_GUIDE_CHAT_BASE_URL`、`NTE_GUIDE_CHAT_API_KEY`、`NTE_GUIDE_CHAT_MODEL`，可选`NTE_GUIDE_CHAT_TIMEOUT`（默认90秒）、`NTE_GUIDE_CHAT_MAX_TOKENS`（默认4096）及受信任CA文件`NTE_GUIDE_CHAT_CA_BUNDLE`。接口兼容`/chat/completions`、函数工具调用和`image_url`多模态消息；需选择同时具备工具调用和图片理解能力的模型。凭据只在后端使用，接口、模板和工具参数不返回密钥；保留TLS验证，不复制其他项目关闭证书验证的行为。

`GET /api/guide/chat/config` 和 `POST /api/guide/chat` 均要求本站登录并返回`no-store`。POST采用NDJSON依次返回`status`、`answer`或`error`；不记录消息、附图或上游错误正文。每账号同时1个回答、进程最多2个，模型查询轮数和总用时有界；停止按钮终止浏览器接收，后端在连接关闭或超时后释放执行位。生产服务须使用多线程，反向代理需允许流式响应；不依赖浏览器连接qqbot进程。

工具覆盖CLI的目录、搜索、详情、schema、本人已同步配装，以及在线轴、养成计算、TapTap攻略搜索和帖子读取。只允许注册的结构化查询，无shell、任意路径、短信同步、发布或写入工具。本人资料始终绑定登录者；在线轴沿用排轴现有公开／私有／受限角色权限。`app`组合入口注入排轴只读适配器，攻略模块不导入排轴领域服务。工具与帖子内容按不可信资料处理，回答区使用安全DOM渲染并提供原文引用。

TapTap工具使用公开只读搜索与帖子接口，不读取用户登录Cookie；正文与搜索摘要分开，帖子附图按需作为图片交给模型，每次读取最多2张、单轮最多4张。读取失败明确返回失败，不把搜索链接或摘要冒充已阅读全文。接口可能调整；异常时保留原文入口。

## CLI 在线动作轴

`axes [--character <攻略ID／排轴ID／完整中文名>] [--scope public|mine|favorites] [--query <名称或备注>] [--sort new|dps|likes|favorites] [--page 1] [--limit 10]`查询在线轴；角色可重复至多4次，表示同时包含。默认公开范围，每页1–20条。`axis <ID> [--section all|summary|axis|result]`读取有权查看的队伍、配装、真实动作步骤和保存结果。返回预览链接与详情链接；保存DPS不代表当前引擎重算或实战结果。

HTTP入口为`GET /api/guide/axes`（重复`character`参数）和`GET /api/guide/axes/<ID>`。私有与收藏范围须通过`NTE_GUIDE_TOKEN`携带本人本站登录态；提供token时拒绝重定向且错误不回显私有正文。原有公共`list/search/get/schema`仍不携带token。网页助手通过同一查询服务执行，避免模型等待服务进程对自身发起HTTP请求导致线程耗尽。

验证：`.venv/bin/python -m unittest tests.test_guide_chat tests.test_guide_bot_tools tests.test_guide_builds tests.test_module_routes -q`；`node --test tests/test_guide_chat_state.cjs`。还需真实模型验证工具调用和图片识别，真实TapTap验证搜索、正文和一图流；浏览器验证登录显示、桌面非模态操作、390px手机、跨页恢复、6小时边界、账号切换、图片上传与错误状态。

验收状态：单元与模块回归通过；真实模型已验证本站资料查询、当前页面切换、TapTap搜索与图片识别。桌面1280×900和手机390×844已核对浮窗／全屏布局、弹窗外操作、收起恢复滚动、刷新恢复图文历史及匿名隐藏入口。模型上下文按单条12000字裁剪并保留有界回答引用，完整本地记录不变；外部附图在同批工具结果全部返回后追加。

## 攻略助手消息额度

每个本站账号在滚动60分钟内最多发送10条消息，文字和附图消息均按一次发送计数，模型内部工具查询不额外计数。通过输入校验且取得并发执行位后即计数，模型失败或用户停止仍计入；未登录、未配置、无效输入和并发繁忙请求不计入。达到上限返回HTTP 429、`chat_rate_limited`、`Retry-After`秒数，并在现有对话状态区提示预计等待分钟数。额度按账号跨页面／对话共用。

`chat_quota.py`通过独立`GuideChatQuota`表和IMMEDIATE事务原子保留发送额度，只保存账号关联与至多10个请求时间，不保存消息或附图；服务启动自动建表，重启及多进程共用同一数据库时额度保持有效。过期时间在下次成功保留时清理。验证：`.venv/bin/python -m unittest tests.test_guide_chat -q`，覆盖第10／11条、滚动边界、账号隔离、重开数据库、并发请求及拒绝请求不扣额度。

## 攻略助手流式回答与站内链接

模型请求启用SSE流式输出，后端增量拼接文字与工具参数，并向浏览器发送`answer_delta`正文片段；工具调用轮次用`answer_reset`清除临时文字，最终`answer`确认完整正文与引用。状态查询继续发送`status`，异常发送`error`。浏览器接收片段后即显示；停止或传输失败保留已收到内容并标记未完成，刷新仍可恢复，后续模型上下文带中断说明。上游连接随取消关闭；总用时、正文长度、响应体积及工具轮数继续受限，不转发上游错误正文。

模型指令要求使用`[中文说明](/站内路径)`，不添加本机地址或猜测域名。`chat_links.py`为已核实的攻略条目生成HTML链接，`wiki_search`与`wiki_get`直接返回`links.detail`；角色另含养成、排轴配装，以及有明确绑定时的空幕与合轴入口。关卡详情保留已选层号。在线轴使用工具提供的`preview_url`，不将JSON接口当网页链接，不混用攻略角色ID与排轴角色ID。回答正文及引用继续通过安全DOM渲染链接，禁止模型HTML注入。

验证：`.venv/bin/python -m unittest tests.test_guide_chat_stream tests.test_guide_chat tests.test_guide_bot_tools tests.test_guide tests.test_module_routes -q`；`node --test tests/test_guide_chat_state.cjs`。浏览器使用隔离合成账号与延迟模型验证首段先于完成显示、站内链接跳转、停止后部分回答保留及刷新恢复。真实模型服务需支持SSE和工具调用，反向代理需关闭响应缓冲。

## 轨外之境公开赛期范围

网页目录、分类计数、JSON列表与全文搜索、HTML／JSON／Markdown详情、攻略助手及CLI统一只返回常驻环线和截至当前已开始的轮换赛期。轮换赛期在配置开始日北京时间05:00（含）开放，历史赛期继续可读；未来赛期不进入搜索或分页分母，直接请求返回404，CLI的JSON／Markdown读取同样失败。没有有效开始日期的非常驻赛期不公开，不按编号推断开放。

`service.public_records`在每次查询时过滤，离线完整快照继续保留未来配置；日期切换无需刷新数据或重启服务。CLI共用服务端判定，不依赖客户端时钟。验证：`.venv/bin/python -m unittest tests.test_guide_challenge_visibility tests.test_guide_challenge_schedule tests.test_guide_current_challenge tests.test_guide_challenges tests.test_guide tests.test_module_routes -q`，覆盖05:00边界、跨时区、缓存后时间推进、分类计数、直接链接、搜索及CLI。

## 按需读取与助手查询体积

CLI可用`get characters 1042 --section overview`读取角色机制、技能ID和可读章节；`get characters 1042 --ability GA_BlackBird_Melee --level 10`只取指定技能的10级参数。`--ability`对应`ability_id`查询参数，仅适用于角色技能；`--level`为1–13，JSON和Markdown均支持。原有`--section`与`--stage`继续有效。CLI角色读取默认技能10级、不含逐命中与资源明细；完整等级和明细须显式开启，公共API未传参数时仍返回完整数据。

`section=overview`不含角色全部倍率、成长和材料表；关卡概览只列关卡目录，敌人数值用`stage`读取。指定技能时，招式分组中重复的参数行通过`parameter_keys`引用该技能顶层`param_rows`，不同内容的行仍独立保留。显式指定等级后，`values`、`attack_values`、`defense_values`、`hp_values`等级数组变为`{level,max_level,value}`；不足目标档时取已有最高档并标明实际档位，不插值、不推算。不指定等级保留原数组。概览中觉醒等绑定参数默认取10级，显式`level`可覆盖；技能精确参数仍需指定技能读取。

助手`wiki_get`默认读取概览；角色技能章节未给`ability_id`时也只返回机制与技能目录，精确参数用`ability_id`和`level`，其余数据按章节／关卡读取。已有准确ID时直接读取，避免重复搜索和整份资料往返。工具执行后显示“查询已完成，正在整理回答…”，区分本地查询与后续模型处理。公共API的默认输出不跟随助手缩略；CLI按下节规定采用10级技能与可选命中明细。

验证：`.venv/bin/python -m unittest tests.test_guide_narrow_reads tests.test_guide tests.test_guide_chat tests.test_guide_bot_tools tests.test_guide_challenge_visibility tests.test_module_routes -q`。覆盖全角色概览体积、原始数据不被修改、指定技能及等级精确值、分组引用、无效参数、Markdown与CLI参数传递及阶段提示。

## CLI默认技能等级与明细

CLI的`get characters <ID>`、`--section abilities`及`--ability <技能ID>`默认只返回10级技能参数，不返回`hit_details`及回能、环合值、倾陷等资源参数行。技能机制正文、持续时间和强化效果等常规参数仍保留，角色属性成长、经验、养成材料以及弧盘／关卡数据不受影响。`--level 8`指定其他技能等级，`--all-levels`保留全部技能等级数组，两者互斥；不足目标等级时取已有最高档并标明实际等级。`--hit-details`显式包含逐命中明细和资源参数；与`--all-levels`组合恢复全部等级及明细。JSON和Markdown使用相同规则，`--section overview`继续只看机制概览。

CLI通过`skill_level=10&hit_details=0`请求服务端裁剪，省去无用传输。`--all-levels`省略`skill_level`，`--hit-details`改为`hit_details=1`；公共API未传这两个参数时保留原始完整输出。`skill_level`只影响技能及关联觉醒／共鸣／词条参数，不将角色面板改为10级；显式`level`参数仍按原有通用档位接口处理。参数行的key不唯一时不转换为分组引用，避免不同招式数值混淆。

示例：`python nte-guide.py get characters 1042 --section abilities`；`python nte-guide.py get characters 1042 --ability GA_BlackBird_Melee --hit-details`；`python nte-guide.py get characters 1042 --section abilities --all-levels --hit-details`。

验证：`.venv/bin/python -m unittest tests.test_guide_cli_skill_defaults tests.test_guide_narrow_reads tests.test_guide tests.test_guide_bot_tools tests.test_guide_chat tests.test_guide_challenge_visibility tests.test_module_routes -q`，覆盖全角色等级数值、明细开关、分组引用、默认成长资料不变、Markdown、API兼容及互斥参数。更新服务端后须重新下载CLI才能使用新默认值与开关。

## 攻略助手配装判断口径

网页助手的实际系统提示词包含异环配装基础：满级橙色卡带主属性、驱动块固定主属性与副词条类别／每格基数、卡带和全部驱动块副词条的单项合计上限，以及基础暴击率和暴击伤害。副词条上限不视为角色最终属性上限；弧盘、卡带主词条、套装与角色被动等合法增益另算。

常规成型配装默认空幕特性已拿满；用户明确未完成时才按实际调整。好感度、家具、噗咔特训的小额常驻增益须核对具体值，并区分快照面板已包含项，避免错误扣除或重复叠加。双暴建议以适用伤害的战斗最终属性评估1:2附近与100%有效暴击率上限，计入弧盘、卡带、角色与队伍效果时核对触发条件、覆盖率和伤害范围，不仅凭静态面板判断。

验证：`.venv/bin/python -m unittest tests.test_guide_chat tests.test_guide_chat_stream tests.test_guide_narrow_reads -q`，并通过合成模型请求核对配装知识确实进入系统消息；不以提示词同步代表模型建议或伤害结论已经完成实战验证。

轨外之境目录将起止日期作为一个不可换行的行内布局，统一行高并禁止收缩；卡片底部ID与日期按基线对齐。卡片保留16px内边距，700px及以下使用单列，为完整日期范围留出空间；常驻环线仍显示“长期开放”。验证时同时检查320、390、700、701、1024和1280px宽度下的日期同排、卡片边界及页面横向溢出。

## 角色预告模式

目录额外提供 Exe（暗属性S级）和曳尔沐（光属性S级）的预告卡片，左上角显示“预告”，属性、星级和搜索筛选沿用现有目录。预告详情只显示完整宣传图及已确认的所需升级材料；材料为空时省略材料区，即使记录中有其他字段也不显示技能、面板、羁遇、定位或工具入口。

`data/character_previews.json` 独立维护预告，`previews.py` 在公共查询中与完整角色合并，不写入正式计算目录；同ID已有完整资料时优先使用完整条目。曳尔沐使用1090，Exe暂用`preview-exe`，取得正式角色编号时再修改，不以NPC编号猜测。`preview_image`绑定模块本地WebP横幅，`assets.avatar`绑定目录头像。可选`materials`沿用角色材料结构（`breakthrough[].items`和`skills[].levels[].items`）；顶部按现有规则去重展示最高品质材料及名称，不显示数量。当前两人的升级材料尚未配置。

HTML、列表、搜索、JSON、Markdown和CLI支持预告条目，`status=preview`表示资料状态；预告概览仅返回身份、状态、图片及材料，不补正式角色字段。养成、排轴、比较入口对预告返回404，助手不生成这些工具链接。普通离线刷新不覆盖预告文件；图片随应用提交，不依赖用户本地下载目录或外部图床。

验证：`.venv/bin/python -m unittest tests.test_guide_previews tests.test_guide tests.test_guide_narrow_reads tests.test_module_routes -q`。浏览器检查目录角标与属性／星级筛选、两张横幅完整显示、无材料时仅显示图片、有材料时显示材料卡，以及桌面／手机无横向溢出。

目录与账号练度页的分类栏取消外层容器叠加的顶部留白和分类栏额外上边距，保留固定导航及安全区域所需占位，导航下方约15px进入分类栏。该调整仅作用于直接包含分类栏的攻略页，角色详情等其他页面的正文间距保持原有布局。浏览器验收覆盖桌面、390px和320px窄屏，分类栏不得被固定导航遮挡或产生整页横向溢出。


## 攻略与撰写

`/guide?kind=beginner` 提供攻略分类，`/guide/beginner/getting-started` 展示完整新手入坑图文。正文包含前言、游戏玩法、货币与消费、版本活动、抽卡规划及结语，保留作者的版本语境与个人观点。一级、二级标题复用详情页悬浮目录、滚动高亮和窄屏目录按钮，不另建目录脚本。正文仅渲染结构化段落、标题和图片，不执行文章HTML。攻略正文统一18px、1.9倍行高，一级／二级章节标题为26px／22px，目录为15px／14px，手机端保持相同可读字号。

`data/beginner.json` 为独立本地内容，65张正文配图及表情由模块 `static/beginner/getting-started/` 持有；图片保留原始分辨率，延迟加载并可打开原图。普通资料刷新不覆盖；启动和请求不联网。网页、全文搜索、JSON、Markdown和CLI共用完整内容。JSON：`/api/guide/beginner/getting-started`；Markdown：`/guide/beginner/getting-started.md`。支持 `section=intro|gameplay|currencies|events|wishes|closing`，分节返回保留作者链接。旧 `progression` 参数映射到 `gameplay`。图片链接均为本站根路径，与CLI的站点地址组合即可读取原图。

CLI：`python nte-guide.py beginner`，`python nte-guide.py beginner --format markdown`，`python nte-guide.py beginner --section events`；也支持 `list beginner` 与 `get beginner getting-started`。更新后需重新下载CLI。

验证：`.venv/bin/python -m unittest tests.test_guide_beginner tests.test_guide tests.test_module_routes -q`；核对正文完整性、本地图片可解码和尺寸、JSON／Markdown／CLI全量及分节一致；浏览器检查分类入口、两级目录跳转、长文图片及窄屏布局。

通用攻略另提供 `/guide/beginner/combat-mechanics`「战斗机制深入攻略」，通过B站官方网页播放器内嵌视频，默认关闭自动播放、支持全屏并提供原视频入口。播放器随容器保持16:9；仅浏览器加载外部播放器，后端不请求视频服务。JSON及Markdown提供视频BV号和观看链接，CLI可用 `get beginner combat-mechanics --format markdown` 读取。

通用攻略列表按本地内容数组顺序展示：新手入坑图文、`/guide/beginner/damage-mechanics`「异环数据机制导论」、战斗机制视频。数据机制条目完整展开正文及图表，保留NGA原帖和附表入口；网页、列表API及CLI使用相同顺序。

数据机制导论按 `chapter_titles` 配置顺序渲染前言及五个PART；JSON、Markdown和CLI支持按 `regular_damage`、`harmony`、`stagger`、`other_mechanics`、`appendices` 分节读取。正文与17张本地图表全部展开，不保留折叠按钮、缺失占位或更新日志区。图表位于模块 `static/beginner/damage-mechanics/`，延迟加载、保持原始分辨率并可点击查看原图；延迟加载不隐藏章节正文。正文、公式、附表链接与图片路径在网页和机器读取接口中保持一致。

数据机制导论的文字颜色由段落内 `runs[].color` 按蓝、红、紫三种语义标记；模板只接受固定颜色类，深色背景使用可读的浅色调，保留原文着色范围。相同词语在其他未标色位置不自动着色，公式与普通段落共用安全文本渲染；JSON提供颜色标记，Markdown仍输出原正文。

### 自建攻略与草稿

攻略页分“通用攻略”和“自建攻略”，原 `kind=beginner` 与既有详情／CLI链接兼容。搜索栏右侧“我要撰写”进入 `/guide/write`；登录后编辑Markdown，支持标题、加粗、列表、链接、图片、表格、实时预览与B站视频插入。通用及自建攻略卡片均使用无图片卡片样式，标题没有额外顶部空行，不显示内部英文ID。

草稿每次输入立即按本站账号和草稿ID保留本机备份，停止输入约800ms后通过鉴权API同步入库；失败时保留内容并提示重试，恢复网络后继续同步。刷新可恢复未同步备份；本机与云端版本冲突时明确选择版本，禁止无提示覆盖。草稿列表支持继续编辑和新建；只有点击“发布攻略”才生成公开快照，再次自动保存不改动已发布正文。作者可从公开页“编辑攻略”继续修改。账号切换后停止旧会话的编辑与保存。

模块本地 `articles.py` 持有 `GuideArticle` 专表与同步写事务，应用启动注册建表；不使用对局异步队列。草稿和公开快照分别保存标题、Markdown及挂载位置，版本号防止多标签页旧写覆盖，冲突返回409。私有API仅以本站Bearer身份确定作者，并返回 `Cache-Control: no-store`；不能读取、修改或发布其他账号草稿。公开列表与详情只读已发布快照，不暴露草稿或账号凭据。

挂载可选角色、弧盘、已开放轨外之境或全站深境的一个详情页，也可不挂载；发布前校验目标可见性。用户撰写的文章始终归入“自建攻略”，不进入“通用攻略”；未挂载时仅展示在自建攻略分区，挂载后额外展示在目标页“相关攻略”。角色页位于养成／排轴／空幕工具组件下方，轨外之境位于共鸣效果上方，弧盘与全站深境位于顶部资料下方。无挂载文章不展示空模块；所有详情页移除重复的数值顺序／JSON说明句。

Markdown由服务端Mistune解析并转义原始HTML，链接和图片只允许安全地址。视频使用固定B站播放器和经过校验的BV号，默认不自动播放；编辑器负责生成 `bilibili` 围栏代码块，不接受用户任意iframe。标题生成稳定章节锚点，发布页复用现有目录能力。网站更新须安装 `requirements.txt` 中的Mistune依赖；本机已安装。

接口：`GET /api/guide/article-targets`；`GET /api/guide/me/articles`；`GET|PUT /api/guide/me/articles/<uuid>`；`POST /api/guide/me/articles/<uuid>/publish`；`POST /api/guide/articles/preview`。保存字段为 `title`、`markdown`、`target_kind`、`target_id`、`revision`；发布需提交当前 `revision`。标题最长160字，正文最长150000字。

公开读取：`GET /api/guide/articles`、`GET /api/guide/articles/<uuid>`、`/guide/articles/<uuid>`、`/guide/articles/<uuid>.md`；全文搜索及攻略分类列表包含已发布自建攻略。CLI支持 `list articles`、`get articles <uuid> --format markdown`。

验证：`.venv/bin/python -m unittest tests.test_guide_articles tests.test_guide_beginner tests.test_guide tests.test_module_routes tests.test_guide_challenges tests.test_guide_previews -q`；`node --test tests/test_guide_article_editor.cjs`。覆盖账号隔离、重连数据库、过期版本、发布快照隔离、Markdown注入与视频、四类挂载和位置、分类搜索、离线备份恢复、保存途中继续输入及冲突版本选择；浏览器用隔离合成账号验收撰写、自动保存、刷新恢复、预览、发布和手机排版。

## 料理

`/guide?kind=foods` 提供料理卡片，按攻击力、暴击率、暴击伤害、防御力、即时恢复、持续恢复、复活、降低体力消耗、加快体力回复九种具体加成分组；分组内展示本地图片、名称与效果，加成／恢复量高亮而持续时间不高亮；卡片底部标注最低方斯标价，缺价显示未提供。每组按加成数值从大到小排列，同值按方斯标价从小到大，缺价排在同值末尾；生命恢复、复活恢复按22,000生命上限将百分比与固定量相加排序，持续恢复计入完整持续时间内的各次恢复；页面保留原始效果数值。料理图片在卡片内水平居中。筛选栏默认勾选“隐藏剧情料理”，按明确的剧情／羁遇获取来源识别，不把缺价或缺来源直接当作剧情料理；取消后显示全部。`hide_story=0`保留取消状态，API默认仍返回全部，可用`hide_story=1`过滤。加成单选与名称／效果搜索均在浏览器过滤，隐藏空分组、更新数量并同步 `bonus` 与 `q` 参数，保留无JavaScript的查询提交。分类直接来自配置，不按名称猜测。当前142种料理不表示所有条目均已开放或可购买。

料理卡片点击后在本页打开详情弹窗，展示完整效果、物品说明、获取途径与方斯商店标价，支持关闭按钮、Esc与点击遮罩，关闭后恢复焦点与页面滚动。`food=<ID>`可分享并恢复弹窗；旧`/guide/foods/<ID>`网页地址跳转到目录弹窗，不再提供独立料理详情页。JSON、Markdown、CLI以及攻略助手共用 `foods` 分类，原机器读取详情接口继续保留。`/api/guide/foods?bonus=crit_rate`按具体加成筛选，原`function=Revive`等宽分类API参数继续支持。料理只提供资料，不自动计入角色面板或排轴计算。

显式刷新：`python scripts/sync_guide_foods.py`，支持 `--ntedata-dir` 读取本地 `DT_FoodItemData.json`、`DT_ItemEffectTypeData.json`、商品／商店及物品来源关联／明细表。加成从明确的效果正文匹配，未知类型拒绝导入；方斯价格仅取方斯商品最低标价，不混算其他货币或限时折扣。获取途径从实际来源关联读取，缺失不猜地点。图片保存于模块 `static/foods/`；数据、分类与图片全部通过校验后原子替换 `data/foods.json`，请求和启动不联网。

验证：`.venv/bin/python -m unittest tests.test_guide_foods tests.test_guide_affinity tests.test_guide tests.test_module_routes tests.test_guide_account tests.test_guide_bot_tools tests.test_guide_chat -q`；浏览器检查料理分组／搜索／详情和窄屏、等级奖励、不同角色满好感属性、购买渠道tooltip的悬停／键盘／点按及原推荐单价。

## 攻略数据库顶部标题与游戏版本

站点定位为攻略数据库，按角色养成、配装、战斗机制与关卡攻略需要收录相关数据，不以收录全部游戏资料为目标。所有攻略模块页面页脚共用这一范围说明，页面描述使用相同定位。

所有攻略模块页面共用顶部居中标题“攻略数据库”与游戏数据版本，独立于左侧工具站／返回入口和右侧首页／账号按钮。`data/site_metadata.json` 维护运行时展示的站点标题、`game_version`；游戏版本显示在标题下方，不展示逻辑版本或相关悬停提示。版本不是接口 `schema_version`，不按当前日期或角色名单推断；更新采用的数据版本时同步维护游戏版本字段。网页请求不联网查询版本。手机端压缩左侧品牌文字与按钮间距，保留中间标题及版本，320px起不遮挡两侧导航。验证：`.venv/bin/python -m unittest tests.test_guide_header tests.test_guide -q`；浏览器核对320、390和桌面宽度下的居中、间距及横向溢出。


## 常驻加成

`/guide?kind=permanent`提供“异象家具”和“噗咔特训”两个条目。家具收录无名刃、休憩的拳王／待战的拳王、空铠甲的战斗属性，旧滑板、纸飞机、双鱼旗的探索耐力，以及织梦结／织梦网的本性像素上限；每项展示激活条件、满级加成及可展开的逐级加成／升级要求表。七件家具各显示本地WebP贴图，位于模块`static/furniture/`，与数据刷新同步获取；JSON及Markdown中的`image`为本站图片路径。家具表的材料为当前级升至下一级的单次要求，满级无下一档；本性像素使用消耗进度要求，不作为提交材料。

特训展示1–20级的噗卡计划等级门槛、升入本级的单次方斯、累计方斯及本级七项总加成。20级累计11,000,000方斯，满级攻击力＋28、防御力＋30、生命值＋420、暴击率＋2.5%、暴击伤害＋5%、环合强度＋12、倾陷强度＋12；各级总加成不能跨级累加。家具和特训资料不自动写入个人面板或其他模块。

本地数据为`data/permanent_bonuses.json`。网页、搜索、JSON、Markdown、CLI及助手查询共用`permanent`分类；详情ID为`furniture`和`puka-training`，可按`section=entries|levels|notes`读取实际存在的章节。CLI例：`python nte-guide.py get permanent puka-training --format markdown`。

显式刷新：`python scripts/sync_guide_permanent_bonuses.py`，或`--ntedata-dir /path/to/NTEData`读取本地解包目录／同名数据表。逐级属性引用、加法操作、材料名称和等级连续性通过校验后原子替换快照；未知机制拒绝导入。请求与启动不联网。

验证：`.venv/bin/python -m unittest tests.test_guide_permanent tests.test_guide tests.test_guide_chat tests.test_guide_bot_tools tests.test_module_routes -q`；浏览器检查分类、搜索、家具升级展开、特训方斯及总加成，和窄屏表格内部横向滚动。


## 角色一图流攻略制作

完整角色页的工具区提供“制作角色一图流攻略”，进入 `/guide/characters/<id>/infographic`；须登录后使用，预告角色不提供入口。页面复用本地角色头像／立绘、适配弧盘与核心卡带图片，初始模板沿用现有推荐绑定，不自动生成面板目标、评分、队伍或攻略结论。页面数据由 `infographic.py` 提供，不引用其他模块的领域服务。

`static/infographic_state.js` 管理字段校验、配置封装与导入，`infographic.js` 管理表单、按角色隔离的浏览器草稿、撤销及助手编辑桥接，`infographic_canvas.js` 在浏览器内完成预览和 PNG 生成。支持横排／竖排、杂志／均衡／紧凑三种布局、三种配色、角色图片与缩放、标题／署名／版本／页脚；弧盘与卡带、参考面板、雷达、技能优先级、两组配队、操作循环、养成与抽取建议七类模块可单独开关及排序，配装保留两个可选方案。推荐文字、评分和目标由作者填写。

草稿只存在当前浏览器，不自动发布或同步到账号；自动保存失败时提示导出配置。模板导入限制256KB，校验角色ID、配置版本、字段长度与可选素材ID，不接受任意图片地址或HTML。导入／重置／助手修改均可撤销。高清 PNG 提供2倍和3倍选项，预览与导出使用同一渲染器，文字自动换行并增高画布；导出尺寸受8192像素单边及3000万像素总量限制，防止浏览器分配过大。图片复制需要安全上下文与浏览器剪贴板支持，失败时可下载PNG。

仅该网页登录后的右下角攻略助手提供 `update_infographic`：请求临时附带当前角色、画布配置和修订号，模型可在用户本轮明确要求时准备顶层参数替换；服务端校验，完整回答后返回页面事件，由浏览器应用并自动预览。无编辑上下文的普通聊天不提供此工具；CLI、公共读取工具和服务器攻略数据不增加写入能力。用户在回答期间手动编辑时，修订号不一致则拒绝覆盖；失败／中断的回答不应用修改。

验证：`.venv/bin/python -m unittest tests.test_guide_infographic tests.test_guide_chat tests.test_guide_chat_stream tests.test_guide_builds tests.test_guide_previews -q`；`node --test tests/test_guide_infographic.cjs tests/test_guide_chat_state.cjs`。浏览器检查角色入口、横竖排与三种布局、模块开关和排序、草稿刷新恢复、PNG实际下载、配置导入导出、手机宽度及合成模型编辑／撤销／并发防覆盖。


一图流右侧提供画板交互层 `infographic_board.js`：七类内容板块可拖动顶部手柄自由放置，移动预览与画布同比例；选中后方向键按10画布像素移动，Alt＋方向键按1像素微调，支持撤销和“恢复自动排版”。已移动板块绘制于自动板块上方。横竖方向或布局方式切换时清空自由坐标，重新自动排版，避免旧坐标跨尺寸错位；模块开关不丢弃其位置。

手填标题、副标题、配装／配队说明、参考面板、操作建议、署名、版本与页脚可在画板上双击打开原位编辑框，保存或Ctrl／Cmd＋Enter提交，Esc取消；自动生成的角色、装备名称与模块标题不允许文本覆盖。雷达的六个维度名称可在侧栏或画板上编辑，每个非空且最多12字，评分与名称按同一索引对应。`radar_labels` 与 `positions` 随草稿、配置导入导出、撤销及本页助手修改保留；导入旧配置时补默认名称和空坐标。

角色属性与稀有度复用本站 `element_icon`／`rarity_icon` 贴图，预览及PNG均显示属性图标与A／S稀有度标识。画板交互框、拖动手柄与编辑框只用于编辑，不进入PNG。渲染器返回同一次排版的文字命中区及板块边界，避免DOM交互层与导出位置各算一遍。验证还须覆盖双击编辑联动、自由拖动与键盘微调、刷新／撤销／自动排版恢复、自定义雷达名称、旧配置迁移、横竖版及导出图标。


配队每套保留4个格子，每格可选择首选角色，并持续添加／移除候选，不限制为两三名；候选范围为本站完整角色目录，同格首选与候选不得重复，其他格子独立选择。侧栏每格独立展开编辑，切换首选时从本格候选中移除同一角色。候选可折叠为小头像网格收纳在同格，或逐行展开；画板点击“候选 N”可切换。未选首选的格子仍保留位置，候选不会挤到前面的格子。PNG保留所有已选择候选及当前折叠方式。

`teams[].members` 保持4个首选ID；`candidates` 为4组候选ID数组，`folded` 为4个布尔值。两者随草稿、配置、撤销与仅本页助手修改保留，旧草稿导入时补空候选及默认折叠。每格最多容纳完整角色目录中的不同角色，不接受未知ID或同格重复；选择器过滤已选候选。验证覆盖超过两三人的添加、全角色候选校验、同格折叠／展开、移除撤销、旧草稿恢复，以及候选头像与名称的完整PNG渲染。


## 一图流绘图JS与自动控件

一图流制作须登录。网页仅返回登录检查外壳，`GET /api/guide/characters/<id>/infographic` 经Bearer鉴权返回公共素材与默认JS，响应`Cache-Control: no-store`；未登录不能取得编辑器初始化数据。登录状态变化时停止使用编辑器。JS、参数和配装等配置按本站账号与角色在当前浏览器隔离保存；旧未归属账号的草稿只迁移给首个打开它的登录账号，不回写服务器。

默认绘图JS来自 `static/infographic_canvas.js`，当前模板的 `config.script` 保存完整可编辑JS，`script_params` 保存声明参数值。JS顶部必须包含一个 `/* @editor {"builtin":[...],"controls":[...]} */` JSON声明：builtin选择现有控件组，controls自动生成新增控件。每项包含key／label／type／default，类型支持text、number、color、boolean、select；number可声明min／max／step，select须声明options。新参数通过 `config.script_params.key` 读取，声明与参数值统一校验。识别仅解析注释JSON，不执行代码。完整协议由 `infographic_script.py` 同步提供给本页助手；JS端由 `infographic_script.js` 校验。

脚本安装 `window.GuideInfographicCanvas.render(canvas, config, data)`，其中canvas为OffscreenCanvas、window为Worker兼容别名；可自由实现Canvas2D绘图逻辑，不限于默认板块。使用 `await __igLoadImage(path)` 获取本页白名单本站素材的ImageBitmap。可返回scene声明文字命中区及自定义板块边界，主界面只接受指向现有字符串参数的安全字段路径与有界坐标。脚本没有页面DOM、浏览器登录存储或网络访问。

`infographic_runtime.js` 在 `sandbox="allow-scripts"` 的独立来源iframe中启动Worker；沙箱CSP禁止网络、外部脚本、表单与嵌套页面，用户代码只在Worker执行。通过校验后的位图与交互元数据返回主页面，JS不接触本站token。每次运行限8秒，单边最多16000像素、合计最多2400万像素；失败或超时保留上一张成功画布。新JS经实际试绘成功后才应用，期间发生手动修改则拒绝覆盖。

统一使用“下载模板”保存含JS、参数及角色配置的完整JSON，不再单独提供JS下载按钮；导入仅接受完整JSON模板。JS编辑面板可直接应用预览，更新进入本机草稿与撤销历史。预览和PNG导出使用同一脚本沙箱。外围控件随声明自动更新，未声明的控件组隐藏。

本页Bot的 `update_infographic` 支持patch.script完整替换或script_edits唯一文本替换，并支持script_params。Bot接收完整当前代码、参数和声明协议；改动不进入网站源码、CLI或公共工具。返回后浏览器验证脚本再应用，错误会在编辑页明确展示。

验证在既有一图流、聊天和前端测试之外，覆盖鉴权引导数据与no-store、JS声明和参数类型、模板往返及旧草稿迁移、唯一代码替换、自动控件、新绘图逻辑、DOM／存储／网络隔离、死循环限时、成功画布保留和合成Bot修改链路。


攻略分类 `/guide?kind=beginner` 的搜索栏旁同时提供“我要撰写”和“制作一图流”。一图流入口进入 `/guide/infographic`，以本地完整角色头像列表选择角色，再跳转到对应 `/guide/characters/<id>/infographic`；预告角色不列入制作选项。角色选择页只展示公共角色身份，具体制作沿用登录校验与浏览器模板保存。手机端两个创作入口在搜索栏下方并排展示。验证覆盖攻略入口、角色选择链接与预告排除，以及桌面和窄屏按钮布局。
