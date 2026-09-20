# 角色头像与立绘资源

## 来源与适用范围

2026-09-20 用户确定：本仓库各模块的角色头像、立绘统一使用私有仓库 [mc-ctrl/NTEData](https://github.com/mc-ctrl/NTEData) 作为图片来源。新增和替换资源遵循本说明。首批实际同步范围、逐文件来源与转码参数见下方记录及资源清单；清单以外的历史图片不声明已核实。

角色机制、属性、技能和数值仍按根 `AGENTS.md` 使用 Nanoka / Everness 核实；图片来源约定不改变业务数据真源。道具、弧盘图标等不在本说明范围内。

## 已核实的上游目录

以下路径在 2026-09-20 使用用户 GitHub 身份读取 `main` 文件树时确认存在；路径均相对于 NTEData 仓库根目录，注意连续两层 `UI/UI` 不是笔误。

| 用途 | 路径 | 示例文件 |
| --- | --- | --- |
| 头像 | `UI/UI_Icon/AvatarImage/` | `200/player_haiyue.png`、`200/player_yiluoyi.png`；另有 `100/` 等版本 |
| 抽卡角色立绘 | `UI/UI/Gacha/` | `YH_lihui_character_nanali.png` |
| 角色与服装立绘 | `UI/UI/Appearance/Fashion/1024/` | `YH_lihui_character_jiuyuan.png`、`YH_lihui_character_xun.png` |
| 时装大图 | `UI/UI/Appearance/Fashion/2160/` | `YH_UI_shizhuang_big_nanali.png` 及多套时装变体 |

目录数字和文件名只用于定位，实际像素尺寸、透明通道、裁切和角色身份必须打开图片核实。不要仅凭 `player_003` 等编号或拼音猜测角色映射；同一角色的默认造型与时装也要区分。

## 认证读取与版本追溯

- 使用用户本机已登录的 `gh` 身份访问。未认证网页返回 404 不代表仓库被删除；先用 `gh api repos/mc-ctrl/NTEData` 检查访问能力。不要把 token 写入文档、下载链接、前端或提交。
- 每次同步先确定上游提交 SHA，再按该提交读取目录和图片，避免同步过程中分支变动。
- 全仓库递归树较大，GitHub 可能返回 `truncated: true`。应从根树找到 `UI` 的 SHA，再读取该子树；仍截断则继续拆分，不能用截断结果断言资源不存在。
- 下载选定图片到临时目录；若取得的是 Git LFS 指针，须通过已认证的 LFS 下载真实内容，不能把指针当作 PNG。
- 每次实际导入应在对应变更说明或资源清单中记录：角色名、用途、上游提交 SHA、上游路径、本地路径，以及转码参数、裁切或缩放信息。存量来源未核实的图片不要补写成已验证来源。

读取示例（将占位值替换为本次核实的值）：

```bash
gh api repos/mc-ctrl/NTEData/commits/main --jq .sha
gh api 'repos/mc-ctrl/NTEData/git/trees/<commit-sha>'
gh api 'repos/mc-ctrl/NTEData/git/trees/<ui-tree-sha>?recursive=1'
gh api -H 'Accept: application/vnd.github.raw+json' \
  'repos/mc-ctrl/NTEData/contents/<image-path>?ref=<commit-sha>' \
  > /tmp/nte-character-source.png
```

## 本地资源与 WebP 交付

- 跨模块头像放在 `app/static/images/characters/avatar/`，立绘放在 `app/static/images/characters/portrait/`。复用现有角色命名与路径，不在各模块重复存放同一图片。
- 上游 PNG 作为转码输入；服务器使用的静态图片默认交付 WebP。当前共享目录已大量使用 `.webp`，但仍有 PNG 等历史资源，不能假定全部完成转换。
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

## 2026-09-20 首批同步与桌游武备换装

上游固定提交：`6887b8ced1812db9671d3d4800a0f15d6322a251`。从 `DT_Character`、`DT_DefaultAppearanceData` 和 `DT_AppearanceData` 核对角色 ID、默认装、头像和立绘路径，不靠文件名猜测。最终保留 23 张默认头像，以及 20 张武备时装图；逐文件 SHA、尺寸、上游路径、本地路径和兼容副本见 [资源清单](character-image-assets-manifest.json)。

头像包含桌游全部 17 人，以及共享目录已有的其他角色、男主。`鉴定师` 对应女主 1051，`男主` 对应 1046。`单位.webp` 与黑羽文字占位不冒充角色原图，保持原样。灵可、残虹的旧 PNG 路径，以及伊洛伊、真红的空幕头像路径保留同源兼容副本；桌游展示统一转向共享 WebP。旧缓存路径在桌游追加图片版本参数，避免 immutable 缓存继续显示旧图。

默认立绘批量替换已按用户反馈完全回退：原来统一的人物缩放与头部对齐必须保留。不能以 `2160` 全身图直接替换既有 `1024×1024` 展示图。武备时装改用上游 `SmallPortraitImg` 指向的 `1024×1024` 展示图，经逐图与牌桌核对后接入。WebP 保留原图尺寸与透明通道，不额外缩放或裁切；质量 90、method 6。上游 PNG 与未采用时装仅留在任务临时目录，不随网站发布。其他模块已缓存旧同名图片的客户端可能需要强制刷新；这次没有批量修改各模块的业务数据或历史存档。

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

### 本次验证边界

- 新增换装/资源完整性、已有角色牌面、公开回放、catalog 与空幕页面资源测试：19 项通过。43 张 WebP 全部解码通过，尺寸与透明通道和最终选定源图一致；默认立绘与任务前 Git 文件逐字节一致。
- 真实大厅进入人机房间 `0B0D1E`，实际打出 N07、将娜娜莉出击到前排并刷新，确认校园装保留；第 5 回合娜娜莉倒地后恢复默认图。结束测试局并在真实回放跳转第 4 回合，确认校园装正确加载。浏览器无 console error；桌面及 844×390 窄横屏检查无图片拉伸。窄横屏检查仅为视口布局，不代表触屏输入验收。
- 扩展的旧 table-contract 有 3 项既有字符串断言失败，已用 HEAD 原文件对照复现；presentation 有 3 项失败、1 项错误，涉及旧 N06 目标、growth、交互提示和伤害阶段预期。相关引擎、卡池与这些测试文件均未改动，不把这些旧失败记为本次通过。
