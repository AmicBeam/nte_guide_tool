# 角色头像与立绘资源

## 来源与适用范围

按 2026-09-20 用户最新口径，角色头像与默认立绘恢复沿用 **Everness / Nanoka 素材流程**，不再要求全仓库统一改用 NTEData。新增角色优先复用已保存的 Everness 素材；更新图片必须保持现有角色缩放、头部对齐和牌面构图。

按用户最新要求，头像已全部恢复到导入前文件，默认立绘也已恢复；仅保留已选武备时装及换装功能。头像与默认立绘沿用旧版 Everness / Nanoka 素材流程，时装仍如实记录 NTEData 来源。资源清单现在只登记保留的 20 张时装。

角色机制、属性、技能和数值仍按根 `AGENTS.md` 使用 Nanoka / Everness 核实；图片来源约定不改变业务数据真源。道具、弧盘图标等不在本说明范围内。

## Everness 素材获取

1. 使用 `everness-info` 技能读取 `everness.info` 角色资料，按角色名核对角色 ID，再查看响应中的图片字段。不要根据编号、拼音或旧缓存猜测角色映射。
2. 头像检查 `icon`、`icon_small`、`iconBig`；默认立绘检查 `iconGacha` 等实际返回的资源，时装可查看 `portraitImg`。字段名只是定位线索，必须打开图片确认用途、角色和造型，不能把任意大图直接当作牌面立绘。
3. 已返回完整 URL 时使用该 URL。对于 Nanoka 的相对资产路径，可参考仓库 `scripts/refresh_kongmu_data.py` 的 `asset_url()`：`https://static.nanoka.cc/assets/nte/<资源路径>.webp`。该格式来自现有导入代码；同步前应核实实际响应和下载结果，不对完整 URL 重复拼接路径或扩展名。
4. 下载到临时目录，核对尺寸、透明通道、人物占比、头部位置和旧版构图后再放入共享目录。默认立绘沿用现有的 `1024×1024` 展示标准，不能用不同构图的全身图直接覆盖；仅尺寸相同也不代表人物已经对齐。
5. 记录角色 ID、查询日期、图片字段、源 URL、本地路径与处理参数。已有 WebP 优先直接保存，不为统一流程重复有损转码。

## NTEData 补充素材与追溯

私有仓库 [mc-ctrl/NTEData](https://github.com/mc-ctrl/NTEData) 仅作为本次已选武备时装等补充素材的来源，不是默认头像、立绘的强制来源。使用用户已登录的 `gh` 身份读取，记录上游提交 SHA、路径和文件校验值；凭证不进入文档或前端。

上游 `UI/UI_Icon/AvatarImage/` 为头像目录；时装展示图在 `UI/UI/Appearance/Fashion/1024/`，应通过 `DT_AppearanceData.SmallPortraitImg` 核对。`2160/` 全身大图的画布和人物比例不同，不可直接替换旧版立绘。全仓库递归树可能被截断，需按子树查询；返回 Git LFS 指针时应先获取真实图片。

## 本地资源与 WebP 交付

- 跨模块头像放在 `app/static/images/characters/avatar/`，立绘放在 `app/static/images/characters/portrait/`。复用现有角色命名与路径，不在各模块重复存放同一图片。
- Everness 已提供的 WebP 直接复用；只有 PNG 等其他格式需要时才在本地转码，服务器静态图片默认交付 WebP。当前共享目录已大量使用 `.webp`，但仍有 PNG 等历史资源，不能假定全部完成转换。
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

旧版素材来源沿用 Everness / Nanoka，获取入口与 CDN 路径规则见上文。下面记录恢复后的实际本地来源路径；历史文件未逐张保存原始下载 URL，不将推测 URL 写成已核实来源。表内路径相对仓库根目录；`单位.webp`、黑羽文字占位不属于此次角色原图替换。

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

保留已确认的 20 张 NTEData 时装，固定上游提交 `6887b8ced1812db9671d3d4800a0f15d6322a251`。图片使用 `DT_AppearanceData.SmallPortraitImg` 对应的 `1024×1024` 展示图，转换为 WebP，质量 90、method 6，保留尺寸与透明通道，不额外缩放或裁切。逐图来源、校验值和武备映射见 [时装资源清单](character-image-assets-manifest.json)。不再从 NTEData 批量替换头像或默认立绘。

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

已撤回排轴与空幕的时装头像引用，恢复 `app/static/images/characters/avatar/黑羽.svg` 文字占位，移除误导入的 `黑羽.webp`。角色数值、权限及桌游已确认的武备时装均不变。后续须获取并核实默认造型资源，不能仅凭文件名包含角色名判定为默认头像。
