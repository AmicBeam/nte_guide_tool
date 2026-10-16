# 攻略数据库

独立匿名只读模块，提供异环角色、弧盘、轨外之境三类战斗数据。工具主页可进入 `/guide`；不依赖其他模块的领域服务，不使用异能对决的原创卡牌数值。当前快照包含 25 个角色条目（男女主分别计数）、51 个弧盘、15 个环线；包含数据集中已提供的条目，不以列表顺序推断实装或赛期开放状态。

## 网页和机器读取

- `/guide?kind=characters|arcs|challenges&q=关键词`：服务端渲染索引，搜索技能正文、弧盘效果和敌人名；角色支持 `element=魂&rarity=5`。
- `/guide/<分类>/<id>`：完整详情，正文无需 JavaScript；角色技能、属性成长与弧盘精炼提供等级滑块，切换当前档位数值；详情采用左侧悬浮目录，技能有二级目录并保留“普通攻击：”“天赋一：”等类型前缀，当前章节随滚动高亮，窄屏通过左下角目录按钮展开；挑战详情可加 `?stage=12` 只看某关。
- `/guide/<分类>/<id>.md`：相同内容的 Markdown，可加 `section=abilities`、`section=effect` 或 `stage=12` 限定返回量。
- `/api/guide/<分类>`：分页摘要，支持 `q`、`element`、`rarity`、`limit`（1–100）、`offset`，返回 `total`、`next_offset`、`items`；摘要含三种详情链接。
- `/api/guide/search?q=关键词`：跨分类全文检索，分页参数相同。
- `/api/guide/<分类>/<id>`：结构化详情，返回 `schema_version`、`kind`、`data`；支持 `section` 与 `stage`。
- `/api/guide/schema`：接口约定、数值语义、字段标签。
- `/llms.txt` 与 `/guide/llms.txt`：LLM 发现入口。
- 不存在的分类、ID、字段或关卡返回 404；无效分页和星级返回 400；错误 JSON 为 `error.code` 和 `error.message`。

角色保留属性成长、完整技能、空幕格位与驱动特化、角色突破与技能升级材料、觉醒、共鸣和机制词条；弧盘保留成长、五档精炼效果、专属效果与突破材料；挑战保留赛期增益、目标、半场、波次、敌人数值与抗性。不纳入剧情、语音、物品库、商店、地图等独立资料功能。

数值数组保留原始顺序：属性成长按等级、技能参数按技能等级、弧盘效果按精炼档位。说明中的 `{0}` 等对应同条目 `values` 内的 `id_value`；不猜测或提前替换。百分比保留字段表示，敌人抗性是小数比例。缺失值是 `null`，网页显示“未提供”。赛期日期缺失不代表当前开放。空幕 `slots` 的 -1 表示不可用格、0 表示可装配格，网页按原矩阵显示布局。材料按目标等级/突破阶段列出单次消耗，不是累计消耗；`items` 附中文材料名和数量，`items_id` 与 `amount` 保留原始配对。角色 `materials` 包含 `breakthrough` 和按技能分组的 `skills`，弧盘 `materials` 为突破阶段列表。角色 `experience` 提供1–80级配置对应的 `required_exp`（经验需求），配置等级表示升入的目标等级，等级1的配置不计入从1级开始的升级消耗。网页在“等级经验”章节展开完整表；JSON、Markdown、CLI 支持 `section=experience`。空幕章节提供 `/kongmu` 空幕计算入口。数据查询不自动生成配队强度结论。

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

`static/guide.js` 仅按等级索引读取已有数值并更新表格，不推算伤害或成长公式；网页初始等级由服务端渲染，前端只展示当前等级，完整数组通过 JSON/Markdown/CLI 读取。空幕使用一句完整的驱动特化说明和格位图，不再重复列出名称、数值、百分比属性卡片；材料使用按目标等级分组的表格。

`service.py` 只读取 `data/catalog.json`，提供检索、摘要、章节选择和 Markdown；`__init__.py` 仅做路由适配和参数校验；模板、样式均由本模块拥有；`cli.py` 是可独立下载运行的 HTTP 客户端。

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

浏览器匿名验证主页入口、三类索引、搜索/筛选、空结果、角色和弧盘详情、技能等级滑块及属性/精炼滑块的首尾数值、空幕格位及空幕计算跳转、材料表格、1–80级经验表、挑战单关、Markdown 与 CLI 下载。完成后确认本机与局域网 5001 端口均可访问。

## 本地角色图片

共享资源位于 `app/static/images/characters/`，按 `avatar/`、`portrait/`、`fullbody/` 分类；`index.json` 只维护角色 ID 与本地图片绑定。当前25个角色条目各有256×256头像、沿用既有画布的默认立绘、2160×2560高清立绘，共75张WebP。复用原有合适图片，补齐缺失文件，不覆盖其他模块原有引用。详情页显示头像与高清立绘，不再单列头像／默认立绘下载区；索引卡片显示头像。图片全部由本站静态路径提供，不依赖外部图床或另一个项目目录。`assets` 字段同时出现在角色 JSON/Markdown 与 CLI 详情中。高清资源保留原生构图，部分原图触及底边，不保证完整到脚。

## 招式分组与命中明细

角色详情章节按技能、觉醒、共鸣、空幕、升级材料的顺序展示。角色技能按 `groups` 展示具体招式，每组保留描述、倍率和其他参数；可明确按名称对应的参数进入对应招式，多招式共用规则集中展示。无法唯一对应描述阶段的伤害参数按其自身展示名称独立成组，不猜测挂到某一段。每个技能的等级滑块同步更新所有招式的倍率与明细，页面不再额外提供全部等级展开表。

导入器通过技能参数中的伤害效果引用连接单次命中条目。`param_rows[].hit_details` 包含攻击/防御/生命倍率、`harmony_gain`（基础环合）、`energy_gain`（基础终结回能）、`stagger`（基础倾陷）和展示式可明确识别的重复次数。数值为单次命中基础值，不含面板效率、被动、觉醒、环合机制或其他状态修正；不按段数平均分配，不把环合强度/倾陷强度当成资源量。重复次数不是自动计算出的实际总命中次数。未提供的数据使用空值，不用零补齐；没有伤害效果引用的纯机制参数不伪造命中明细。

## 索引与弧盘展示

角色索引以属性图标显示属性，保留无障碍名称。弧盘索引按固态、液态、气态、等离子、聚合分组，图片从本站 `app/static/images/arcs/` 读取WebP。显式同步图片命令：`python scripts/sync_guide_arc_images.py`。弧盘详情先展示属性成长，再展示效果；精炼滑块直接替换效果正文中的数值，所有数字使用绿色，不另列效果参数表。结构化详情仍保留各精炼档位数组。

## 角色弧盘适配与定位

角色页顶部用属性图标与稀有度图标（四星A、五星S）替代对应文字，稀有度图标放在 `app/static/images/rarities/`；同时显示弧盘适配类型及其图标，并为每个角色定位标签显示对应图标。`data/character_metadata.json` 保存按角色ID绑定的 `arc_type` 和 `char_tags`；图标使用本站 `app/static/images/arc-types/` 与 `character-tags/` 下的WebP。JSON、Markdown和CLI详情同时提供适配类型及标签图标路径。显式同步命令为 `python scripts/sync_guide_character_metadata.py`；同步按角色配置中的类型枚举与弧盘类型表连接，不根据专武名称推断角色适配。网站启动和请求只读取本地数据。

## 角色养成计算器

角色详情顶部与升级材料章节提供醒目的大按钮，进入 `/guide/characters/<id>/cultivation`。输入当前与目标角色等级、已突破次数、技能／天赋／都市技能基础等级，以及当前等级已积累经验；服务端汇总角色经验、经验书、甲硬币、突破材料、技能材料和都市技能使用的方斯等资源。0级表示尚未解锁的天赋或都市技能；普通战斗技能从1级起。觉醒／共鸣带来的额外技能等级不参与材料计算，觉醒、共鸣和弧盘养成不在此计算器范围内。

突破次数与等级上限按角色配置校验；材料只累计当前之后至目标的阶段，技能须满足对应突破条件。角色经验按升入目标等级的需求逐级相加，扣除当前等级已有经验；每段到达突破上限后分别用经验书补齐，按高档优先、低档补齐组成使用方案，显示溢出经验。甲硬币合计包含这份方案的经验书使用费用，不假设溢出退费。不扣除库存，不自动折算合成，也不估算副本次数或体力。

`cultivation.py` 只对本模块离线数据计算；GET表单无需JavaScript。`POST /api/guide/characters/<id>/cultivation` 接收 JSON 对象，字段为 `current_level`、`target_level`、`current_breakthrough`、`target_breakthrough`、`current_exp` 和 `current_skill_<index>`／`target_skill_<index>`；技能索引按角色 `materials.skills` 顺序，缺省按1级未突破至80级满突破、所有技能满级计算。返回 `data.state`、`experience`、`book_experience`、`overflow_experience`、`level_up_gold`、`total_gold`、`items` 和 `breakdown`。无效练度返回400，未知角色返回404。

刷新还读取角色表、物品表及突破表的经验书和等级上限；`--ntedata-dir` 目录需同时包含 `DT_Character.json`、`DT_ItemConfig.json` 和 `DT_CharacterBreakthroughDataTable.json`。本地快照只保留实际参与计算的字段，不在请求时联网。

验证覆盖全部25角色、满养成资源汇总、经验书使用费、已有经验、原地突破、当前等于目标、无效等级／突破／技能及网页表单。浏览器检查输入、计算结果、错误提示和空幕跳转。
