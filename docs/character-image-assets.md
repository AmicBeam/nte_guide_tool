# 角色头像与立绘资源

## 来源与适用范围

2026-09-20 用户确定：本仓库各模块的角色头像、立绘统一使用私有仓库 [mc-ctrl/NTEData](https://github.com/mc-ctrl/NTEData) 作为图片来源。新增和替换资源遵循本说明；这次约定仅记录来源与流程，不表示存量图片已逐一迁移或核实来源。

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
