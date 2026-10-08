# 角色头像与立绘资源

## 来源与适用范围

自 2026-09-24 起，新增或更新角色头像、默认立绘时**优先查看最新 [NTEData](https://github.com/mc-ctrl/NTEData) 原始资源，再以 Nanoka 当前角色数据核对角色 ID、图片字段和资源路径**；NTEData 缺图时使用 Nanoka 资产 CDN。Everness 不再作为新素材的取图源。已有默认图不因来源调整批量替换，更新时仍须保持角色缩放、头部对齐和牌面构图。

2026-09-20 的头像及默认立绘回退、20 张武备时装导入属于历史操作，记录保留在下文。角色机制、属性、技能和数值的核实按根 `AGENTS.md` 执行；本文件只约定图片资源。道具、弧盘图标不在本说明范围内。

## NTEData / Nanoka 素材获取

1. 用 Nanoka 当前角色索引按角色名确认 ID，再查看角色记录中的 `icon_gacha`、`icon` 和默认服装（如编号 `_0`）字段。`icon_gacha` 是数据字段，不是 NTEData 文件名；必须按返回的资源路径到 NTEData 当前文件树核对，不能由角色名猜路径。
2. 默认立绘优先使用与 `icon_gacha` 对应的 `UI/UI/Gacha/` 或 `UI/UI/Appearance/Fashion/1024/` 原图。头像先检查 `UI/UI_Icon/AvatarImage/256/` 的默认造型图及其透明蒙版；256×256 画布也可能只含圆形头像。没有合适的方形图时，从已核实的默认立绘确定性裁切。不要把时装造型或带徽标道具图当作默认头像。逐张打开确认角色、服装和构图。
3. 使用已授权的只读 GitHub 访问取得 NTEData 最新提交及对应文件，记录提交 SHA、原始路径和 SHA256。若原图不可用，再按 Nanoka 返回的路径下载 `https://static.nanoka.cc/assets/nte/<资源路径>.webp`；完整 URL 不重复拼接路径或扩展名。
4. 下载到临时目录，核对尺寸、透明通道、人物占比、头部位置和现有卡面构图后再放入共享目录。默认立绘沿用现有的 `1024×1024` 展示标准；`2160/` 全身图的画布与人物比例不同，不能直接覆盖。仅尺寸相同也不代表视平线对齐。
5. PNG 等原图在本地转换为 WebP，记录参数与输出校验值。头像若已有独立方形图，直接使用该图；若只有圆形蒙版，从立绘裁切，记录裁切坐标和缩放方法，不放大圆形蒙版填补缺失像素。

## 本地资源与 WebP 交付

- 跨模块头像放在 `app/static/images/characters/avatar/`，立绘放在 `app/static/images/characters/portrait/`。复用现有角色命名与路径，不在各模块重复存放同一图片。
- 已有合适的 WebP 可直接复用；PNG 等其他格式在本地转码，服务器静态图片默认交付 WebP。当前共享目录已大量使用 `.webp`，但仍有 PNG 等历史资源，不能假定全部完成转换。
- 在本地导入阶段完成转码，并提交最终 WebP；无需服务器运行时下载私有仓库或临时转码。现有发布脚本只打包已提交的 `app/`，所以图片与对应路径引用必须一起提交。
- 保留透明通道、宽高比与角色完整构图；不要直接把扩展名从 `.png` 改成 `.webp`。需要缩放或裁切时按使用场景明确记录，不能只因源图较大就裁掉人物。
- 可用 Pillow 或 `cwebp`。质量参数先按实际体积和视觉检查选择；下面的 `quality=90, method=6` 仅为起点，不是经过统一验收的固定标准。小头像边缘或细节不佳时可使用无损 WebP。

Pillow 转码示例（需在具备 Pillow 和 WebP 编码支持的工具环境执行）：

```python
from PIL import Image

with Image.open('/tmp/nte-character-source.png') as source:
    source.convert('RGBA').save(
        '/tmp/nte-character-output.webp',
        format='WEBP', quality=90, method=6,
    )
```

确认输出可解码、尺寸和透明度正确后，才放入共享目录并更新引用。网页应读取本站静态资源，不直接依赖私有 GitHub 图片 URL。

## 实际替换时的验收

1. 对照原图检查角色、造型、透明背景、边缘、清晰度、裁切和文件体积。
2. 搜索旧路径在各模块配置、模板和脚本中的引用，保持头像与立绘语义一致；不要用小头像替代大幅立绘。
3. 在受影响的真实页面检查加载成功、无破图、无拉伸，并检查桌面与移动端表现。删除旧文件前确认已无引用。
4. 按根 `AGENTS.md` 完成适用验证、局域网访问检查和本次改动的 Git 提交；发布另按仓库发布流程执行。

## 旧版头像与立绘恢复记录

头像恢复基准为 `a40b55ac4` 的父提交 `0c9ebd3f2`，按 Git 原始字节恢复，包括残虹、灵可的方形 PNG，以及伊洛伊、真红的空幕头像。撤销本次新增的四个头像 WebP 别名，前端恢复原路径；图片缓存版本同步更新。默认立绘保持原有 `1024×1024` 构图、人物缩放和头部对齐。

下面仅记录 2026-09-20 恢复后的实际本地路径；历史文件未逐张保存原始下载 URL，不将推测 URL 写成已核实来源。表内路径相对仓库根目录；`单位.webp`、当时的黑羽文字占位不属于该轮角色原图替换。新图片按本文开头的 NTEData / Nanoka 流程获取。

| 角色 | 旧版头像路径 | 旧版立绘路径 |
| --- | --- | --- |
| 早雾 | `app/static/images/characters/avatar/早雾.webp` | `app/static/images/characters/portrait/早雾.webp` |
| 安魂曲 | `app/static/images/characters/avatar/安魂曲.webp` | `app/static/images/characters/portrait/安魂曲.webp` |
| 翳 | `app/static/images/characters/avatar/翳.webp` | `app/static/images/characters/portrait/翳.webp` |
| 娜娜莉 | `app/static/images/characters/avatar/娜娜莉.webp` | `app/static/images/characters/portrait/娜娜莉.webp` |
| 薄荷 | `app/static/images/characters/avatar/薄荷.webp` | `app/static/images/characters/portrait/薄荷.webp` |
| 哈尼娅 | `app/static/images/characters/avatar/哈尼娅.webp` | `app/static/images/characters/portrait/哈尼娅.webp` |
| 埃德嘉 | `app/static/images/characters/avatar/埃德嘉.webp` | `app/static/images/characters/portrait/埃德嘉.webp` |
| 白藏 | `app/static/images/characters/avatar/白藏.webp` | `app/static/images/characters/portrait/白藏.webp` |
| 哈索尔 | `app/static/images/characters/avatar/哈索尔.webp` | `app/static/images/characters/portrait/哈索尔.webp` |
| 阿德勒 | `app/static/images/characters/avatar/阿德勒.webp` | `app/static/images/characters/portrait/阿德勒.webp` |
| 残虹 | `app/static/images/characters/avatar/残虹.png` | `app/static/images/characters/portrait/残虹.webp` |
| 法帝娅 | `app/static/images/characters/avatar/法帝娅.webp` | `app/static/images/characters/portrait/法帝娅.webp` |
| 男主 | `app/static/images/characters/avatar/男主.webp` | 无独立文件（保留原有引用） |
| 鉴定师 | `app/static/images/characters/avatar/鉴定师.webp` | `app/static/images/characters/portrait/鉴定师.webp` |
| 浔 | `app/static/images/characters/avatar/浔.webp` | `app/static/images/characters/portrait/浔.webp` |
| 达芙蒂尔 | `app/static/images/characters/avatar/达芙蒂尔.webp` | `app/static/images/characters/portrait/达芙蒂尔.webp` |
| 九原 | `app/static/images/characters/avatar/九原.webp` | `app/static/images/characters/portrait/九原.webp` |
| 海月 | `app/static/images/characters/avatar/海月.webp` | `app/static/images/characters/portrait/海月.webp` |
| 灵可 | `app/static/images/characters/avatar/灵可.png` | `app/static/images/characters/portrait/灵可.webp` |
| 小吱 | `app/static/images/characters/avatar/小吱.webp` | `app/static/images/characters/portrait/小吱.webp` |
| 伊洛伊 | `app/modules/kongmu/static/images/characters/player_yiluoyi_256.webp` | `app/static/images/characters/portrait/伊洛伊.webp` |
| 卡厄斯 | `app/static/images/characters/avatar/卡厄斯.webp` | `app/static/images/characters/portrait/卡厄斯.webp` |
| 真红 | `app/modules/kongmu/static/images/characters/player_zhenhong_256.webp` | `app/static/images/characters/portrait/真红.webp` |

## 保留的武备时装

该轮保留已确认的 20 张 NTEData 时装，固定上游提交 `6887b8ced1812db9671d3d4800a0f15d6322a251`。图片使用 `DT_AppearanceData.SmallPortraitImg` 对应的 `1024×1024` 展示图，转换为 WebP，质量 90、method 6，保留尺寸与透明通道，不额外缩放或裁切。逐图来源、校验值和武备映射见 [时装资源清单](character-image-assets-manifest.json)。该轮未批量替换头像或默认立绘。

### 桌游映射

换装只影响 V2 牌桌双方前排、备战区及角色详情；手牌、构筑、图鉴使用默认图。按当前实体的公开 `shape_id` 选图，不按角色 ID 缓存装备状态；普通返回保留装备外观，换武备立即切换，倒地卸装恢复默认，实时局面与回放一致。未配置、上游缺图或时装加载失败回退默认图；旧回放没有武备 ID 时不猜测。此为展示约定，不增加或修改任何武备效果，也不修改规则哈希或高级人机模型。

| 角色 | 07 武备 | 08 武备 |
| --- | --- | --- |
| 娜娜莉 | 学园之星（用户指定校园装） | 绵绵飞虎 |
| 零 | 猎人不上班 | 默认（无居家服） |
| 九原 | 默认（额外时装已用于 08） | 自律时间 |
| 浔 | 不适用：07 为战术 | 闲满庭 |
| 安魂曲 | 鎏金交响诗（用户指定） | 新月摇篮曲 |
| 残虹 | 赤葵 | 月下闲庭 |
| 早雾 | 默认（仅一套额外睡衣，留给 08） | 南瓜魔法（节日睡衣） |
| 灵可 | 暮春物语（用户指定校园装） | 待机状态 |
| 真红 | 地海高校生 | 迷宫屋主 |
| 伊洛伊 | 织梦者 | 星月夜 |
| 薄荷 | 幽火照影 | 悠闲假日 |
| 小吱 | 青竹猗猗 | 绒绒春日 |
| 白藏、哈索尔、海月、埃德嘉、哈尼娅 | 默认（本次上游无额外时装图） | 默认（本次上游无额外时装图） |

08 优先使用上游 `nighty` 居家服；早雾仅有节日睡衣，按睡衣候选采用。07 按本次用户选择配置；其他武备编号仍默认。测试角色适用同一逻辑，不改变账号可见权限。维护入口为 `app/modules/card_game/static/js/v2_api.js` 的 `EQUIPMENT_PORTRAITS`，图片位于共享 `images/characters/costume/`，切勿为了换图修改 `catalog.json` 或规则引擎。

验证：`python3 -m unittest tests.test_duel_v2_costume_art tests.test_duel_v2_card_art tests.test_duel_v2_table_contract tests.test_duel_v2_presentation tests.test_duel_v2_replay`。资源完整性检查不依赖 Pillow；真实解码、透明度及页面视觉检查在导入时执行。

### 首次换装接入的历史验证（头像随后已回退）

- 新增换装/资源完整性、已有角色牌面、公开回放、catalog 与空幕页面资源测试：19 项通过。43 张 WebP 全部解码通过，尺寸与透明通道和最终选定源图一致；默认立绘与任务前 Git 文件逐字节一致。
- 真实大厅进入人机房间 `0B0D1E`，实际打出 N07、将娜娜莉出击到前排并刷新，确认校园装保留；第 5 回合娜娜莉倒地后恢复默认图。结束测试局并在真实回放跳转第 4 回合，确认校园装正确加载。浏览器无 console error；桌面及 844×390 窄横屏检查无图片拉伸。窄横屏检查仅为视口布局，不代表触屏输入验收。
- 扩展的旧 table-contract 有 3 项既有字符串断言失败，已用 HEAD 原文件对照复现；presentation 有 3 项失败、1 项错误，涉及旧 N06 目标、growth、交互提示和伤害阶段预期。相关引擎、卡池与这些测试文件均未改动，不把这些旧失败记为本次通过。

### 头像回退验证

已逐文件校验头像及兼容路径与 `0c9ebd3f2` 的 Git 内容一致，默认立绘未变；20 张时装文件保持不变。换装和卡面资源回归使用 `tests.test_duel_v2_costume_art` 与 `tests.test_duel_v2_card_art`。

## 黑羽附件核对与误用撤回（2026-09-20）

用户提供 `UI.7z`，要求为排轴与空幕补默认头像。初次误用了 `UI/UI_Icon/AvatarImage/256/player_heiyu2_256.png`；经用户指出、再与 `Fashion_1042_2_Avatar.png` 和 `YH_lihui_1024_Heiyu3.png` 对照，确认是同一时装造型，不能当作默认头像。

该包的独立黑羽头像候选只有上述时装版本（200/256 及带框版本）；默认造型见于活动宣传图、细长培养页图及 `YH_lihui_1042_01`–`04` 道具图，后者带圆框和爱心徽标，不直接作为角色头像。此次未找到干净的独立默认头像。

当时撤回了排轴与空幕的时装头像引用，改用 `app/static/images/characters/avatar/黑羽.svg` 文字占位。该附件未包含干净的默认造型头像；不能仅凭文件名包含角色名判定造型。2026-09-24 在最新 NTEData 中取得正确图片后，该占位已被下述资源替换。

## 黑羽默认造型补图（2026-09-24）

核对 [NTEData](https://github.com/mc-ctrl/NTEData) 提交 `43dd106003160cbb7566f120d6579288091b9609` 与 Nanoka NTE `1.4.6` 角色 `1042`：Nanoka 的 `icon_gacha` 指向 `YH_lihui_fashionshop_heiyu`，`icon` 指向 `player_heiyu_256`；NTEData 当前文件树均有对应原图。角色默认服装为 `Fashion_1042_0`「黑之「魔女」」，不同于旧附件中误取的 `player_heiyu2_256` 时装。实际打开 `player_heiyu_256` 后发现其 256×256 画布只有圆形人物蒙版，因此方形头像由默认立绘裁切。

| 用途 | NTEData 原始路径 | 原图 SHA256 | 本地 WebP 路径 | 转换 |
| --- | --- | --- | --- | --- |
| 默认立绘 | `UI/UI/Appearance/Fashion/1024/YH_lihui_fashionshop_heiyu.png` | `58f9a601b1b7eeed4793719c6746cde76bca3576e4cb121a356a255f7fa9a5bc` | `app/static/images/characters/portrait/黑羽.webp` | `cwebp -q 90 -m 6`，1024×1024 |
| 方形头像 | 与立绘同源；另核对 `UI/UI_Icon/AvatarImage/256/player_heiyu_256.png` 的造型 | 立绘原图同上；圆形蒙版头像 SHA256 为 `b04067b046c84d65efeea69d7e58101d19ed1dc441a3cbc6d541d33a048ea394` | `app/static/images/characters/avatar/黑羽.webp` | 立绘裁切 `crop=340:340:365:190`，Lanczos 缩至256×256，`cwebp -lossless -m 6` |

输出 WebP SHA256 分别为 `744e8a7831ce71c1dac19218930a86ada96cca31d354f6d96181defa0cd0707d`、`9a19b51d97c3364abff03f86a31547fc84124ecf9c666c52732c48376a6e477c`。裁切保留人物面部和肩部，使用方形画布，无圆形蒙版；排轴使用立绘和头像，空幕使用共享头像。空幕角色对所有用户开放，排轴权限独立。

## 明音凛默认造型补图（2026-09-24）

核对 NTEData 同一提交 `43dd106003160cbb7566f120d6579288091b9609` 与 Nanoka NTE `1.4.6` 角色 `1057`：`icon_gacha` 对应 `YH_lihui_fashionshop_mingyin`，默认服装为 `Fashion_1057_0`「律动节拍」。`player_mingyin_256` 虽为 256×256 画布，但人物受圆形蒙版裁掉四角；共享方形头像从默认立绘原图裁切。

| 用途 | NTEData 原始路径 | 原图 SHA256 | 本地 WebP 路径 | 转换 |
| --- | --- | --- | --- | --- |
| 默认立绘 | `UI/UI/Appearance/Fashion/1024/YH_lihui_fashionshop_mingyin.png` | `4cd11b1eabb42f1aa5c7c359fc54be15e4143767e1e1b3ae0dfcacd012063a3b` | `app/static/images/characters/portrait/明音凛.webp` | `cwebp -q 90 -m 6`，1024×1024 |
| 方形头像 | 与立绘同源；另核对 `UI/UI_Icon/AvatarImage/256/player_mingyin_256.png` 的造型 | 立绘原图同上；圆形蒙版头像 SHA256 为 `295d886dbebc847cb86ffdfeaffd5bbcceaef8fd14849a080f5b77da4db324d2` | `app/static/images/characters/avatar/明音凛.webp` | 立绘裁切 `crop=400:400:290:105`，Lanczos 缩至256×256，`cwebp -lossless -m 6` |

输出 WebP SHA256 分别为 `0f6923e70c1001190fc860a8058b0d545bdd940cc0bd151a8c66b27a5eebf776`、`a5bcd73ab2be180213c2cea6273210a7dfdd764ae7636fd2137bc73593a4cc72`。排轴展示头像与立绘，空幕规划器仅使用共享头像。空幕对所有用户开放，排轴权限独立。


## 鬼郎丸透明主体（2026-10-08）

用户明确指定 [NTEData 档案图](https://github.com/mc-ctrl/NTEData/blob/d7fe705451f189cec8447d876e6c31db7ed3d9df/UI/UI/VisionExplore/BigArchive/UI_YH_Archive_FileImages_50.png)。上游提交为 `d7fe705451f189cec8447d876e6c31db7ed3d9df`，原图 411×412 RGBA，SHA256 `40d0ac4c9ed28e12909399b38a4ec8e890625aa74683a9b65ea34e0a791e375b`。这是召唤物指定档案图，不是早雾或其他异能者的默认服装；不推断 Nanoka 角色字段。

通过内置 image_gen 执行 `background-extraction`，保留黑色犬形主体、红耳、红色长舌、缠绕部分与相连杆部，移除纸张、印刷条带、文字、白框及外框，输出真实透明 alpha。交付 `app/static/images/characters/portrait/鬼郎丸.webp`，由生成的 1254×1254 RGBA 结果等比缩至 1024×1024，Lanczos，无损 WebP／method=6；不在服务器运行时抠图。输出 SHA256 `d0e3b8a65c9c9f61f77ff011de7dfedf18b267c151d52e24d9aa8616319a2ad1`。鬼郎丸的召唤投影及回放使用该本地路径；其他角色图片不受影响。

使用的内置工具提示词：

> Use case: background-extraction. Edit target: supplied NTEData UI_YH_Archive_FileImages_50.png. Asset: transparent character cutout for the 鬼郎丸 card portrait in a game. Extract ONLY the existing black dog-like creature with red ears, red long tongue, red curled wrapping/tail and the long attached dark staff/pole extending to bottom left. Preserve the original creature identity, pose, proportions, facial features, colors, flat illustrated rendering and all visible subject details exactly. Remove the cream paper, cyan and gray printed bands, all text, white card border and black outside frame completely. Preserve clean anti-aliased original subject edges with actual transparent alpha everywhere else. Keep the entire visible subject and attached staff within the canvas, centered with modest transparent margins, without adding or inventing missing details. Do not redraw or restyle, no text, no shadow, no new backdrop.


## 灵可默认立绘更新（2026-10-08）

用户指定将保留的旧默认立绘更新为 NTEData 当前修订图，继续使用 WebP。核对 NTEData 最新提交 `d7fe705451f189cec8447d876e6c31db7ed3d9df` 与 Nanoka NTE `1.4.6` 角色 `1072`：`icon_gacha` 对应 `YH_lihui_fashionshop_lingke`，默认服装为 `Fashion_1072_0`「电波未达」。本次采用 [NTEData 默认造型原图](https://github.com/mc-ctrl/NTEData/blob/d7fe705451f189cec8447d876e6c31db7ed3d9df/UI/UI/Appearance/Fashion/1024/YH_lihui_fashionshop_lingke.png)，不是编号 `lingke1`–`lingke3` 的额外时装。

原图路径 `UI/UI/Appearance/Fashion/1024/YH_lihui_fashionshop_lingke.png`，1024×1024 RGBA、905212 字节，SHA256 `3abce6cd5da5393a5a9e63dc4aeeeebdee86e70df9bcefa317019596339cabc5`。使用 `cwebp -q 90 -m 6` 转换，不额外缩放或裁切，保留透明通道与上游构图；输出覆盖 `app/static/images/characters/portrait/灵可.webp`，1024×1024、192990 字节，SHA256 `a42c25c5d56d240c81576c78d1f5e2efe29f47f629d6fb8627530f8b328697b3`。

与旧图对照，姿势、角色占比和头部位置接近，采用上游修订后的脸部、发丝及服装细节。头像、武备时装和角色可见权限保持原设置；V2 图片缓存版本同步更新。上文旧版恢复记录保留当时状态，灵可默认立绘的当前来源以本节为准。
