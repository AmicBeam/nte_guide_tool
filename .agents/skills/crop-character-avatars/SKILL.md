---
name: crop-character-avatars
description: 裁剪、重裁或校准本仓库角色头像，尤其是视平线偏低、人脸大小不一致、头顶留白过多或原始圆形头像底边的问题。维护曳尔沐头像时复用已确认裁剪参数；不用于替换角色立绘或修改角色数值。
---

# 角色头像裁剪与视平线校准

先读根 `AGENTS.md`、`app/static/AGENTS.md` 和 `docs/character-image-assets.md`；涉及模块绑定时按根文档读取对应模块 README。仅重裁头像时，使用仓库已核实的默认立绘，不因裁剪任务刷新角色数值或开放权限。

## 曳尔沐当前参数

以下参数仅适用于当前共享的1024×1024默认立绘；更换源图构图或尺寸后须重新校准，不能直接套用到其他角色。

| 项目 | 当前值 |
| --- | --- |
| 角色 | 曳尔沐，攻略／空幕ID `1090`，排轴ID `char_elyms` |
| 输入 | `app/static/images/characters/portrait/曳尔沐.webp` |
| 裁剪框 | `(left, top, right, bottom) = (340, 100, 780, 540)`，右／下边界不包含 |
| 裁剪区域 | 440×440；不得先缩放或拉伸立绘再套坐标 |
| 输出 | 256×256 RGBA；Lanczos缩放，无损WebP，`exact=True`保留透明像素 |
| 当前主头像 | `app/static/images/characters/avatar/曳尔沐-square-v2.webp` |

在仓库根目录复用已有脚本：

```bash
.venv/bin/python scripts/crop_character_avatar.py \
  app/static/images/characters/portrait/曳尔沐.webp \
  app/static/images/characters/avatar/曳尔沐-square-v2.webp \
  --box 340 100 780 540
```

素材处理与像素校验依赖见 `requirements-assets.txt`。

## 校准原则

- 以早雾、薄荷等现有默认头像为参考，按同尺寸并排观察眼睛高度、脸部大小与头顶留白；不要只比较透明边界或是否装下全部头发。
- 曳尔沐保持440×440取景尺寸，以保持人物大小；上下移动裁剪框来调整视平线。裁剪框向下移动，输出中人物及眼睛向上移动。
- 当前构图优先对齐眼睛高度，不为完整保留顶端呆毛而把脸压低。调整后同时检查面部、下巴与肩部边缘，避免退回原始圆形底边头像。
- 参数变化时生成旧／新并排图，并与参考角色同尺寸查看；最后检查真实角色目录的小头像显示。

## 绑定、兼容与缓存

共享角色图使用长效 immutable 缓存。头像内容变化时采用新的主文件名并更新当前引用，避免浏览器继续使用旧图；表中的 `v2` 是当前绑定，新改图时检查是否需要递增。

检查 `app/static/images/characters/index.json`、排轴 `static/data/characters.json` 和空幕 `service.py` 中的头像绑定。曳尔沐现有兼容图片为共享头像目录下的 `曳尔沐.webp`、`曳尔沐-square.webp`，以及 `app/modules/guide/static/character-previews/elyms-avatar.webp`；需要保留这些地址时同步裁剪结果，不复制原始圆形头像。

修改排轴角色配置后，按模块约定重新生成合轴静态结果并验证。图片调整不改变曳尔沐在排轴／空幕的测试账号权限。

## 验收与交付

复用 `tests.test_guide_elyms_complete` 中的确定性裁剪像素检查；参数或绑定变化时同步其预期值，以及受影响的头像绑定断言。检查256×256尺寸、RGBA透明度、解码成功与兼容图片一致。

运行受影响的攻略、排轴、空幕测试；涉及合轴派生结果时执行对应回归。同步维护 `docs/character-image-assets.md` 与攻略 README 中的当前裁剪参数，避免技能、文档和测试各自保存不同取景框。

交付前展示新头像或旧／新对照图，检查本机与局域网页面可访问，并按根 `AGENTS.md` 提交／推送本次相关文件。保存的是当前有效参数与验证方式，不新增上游来源台账或单轮操作记录。
