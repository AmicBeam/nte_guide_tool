# 空幕计算模块

空幕计算是独立网页工具，页面入口为 `/kongmu`，用于选择角色与卡带并生成可用的空幕搭配方案。

## 模块范围

- 页面展示角色、卡带和计算结果。
- 浏览器只提交选择并渲染结果，方案计算由后端应用服务完成。
- 该模块不依赖异象对决的房间、对局快照或规则结算。
- 空幕规划器的全部角色对匿名、普通、受邀和测试账号统一开放，角色目录与方案计算不读取排轴角色权限。
- 主角目录头像使用共享本地角色资源，通过长效 immutable 静态缓存加载，不依赖 Nanoka 远程回退。
- 残虹头像使用共享资源 `app/static/images/characters/avatar/残虹.png`，空幕与排轴统一引用该文件。
- 残虹的Ⅲ型特化、“每个Ⅲ型驱动增加16%暴击伤害”和20格空幕拓扑已按 Nanoka NTE 1.3.4 正式角色 ID `1036` 的角色详情接入；规划器保留排轴稳定角色 ID，但角色已公开。
- 灵可使用 Nanoka 角色 ID `1072`，为灵属性5星角色；空幕为Ⅲ型特化，每个Ⅲ型驱动增加8%暴击率。头像使用共享资源 `app/static/images/characters/avatar/灵可.png`。

## 代码边界

- `app/modules/kongmu/service.py`：catalog 与布局方案计算。
- `app/modules/kongmu/templates/kongmu/index.html`：页面模板。
- `app/modules/kongmu/static/`：页面脚本、样式与模块静态数据。
- `app/routes.py` 中 `/kongmu`、`/api/kongmu/catalog` 和 `/api/kongmu/plan`：页面与 API 入口。

新增算法应留在模块服务中，路由只做输入解析、并发保护和错误转换。

## 验证

```bash
python3 -m unittest tests.test_module_routes.ModuleRoutesTest.test_kongmu_module_page_catalog_and_asset
python3 -m unittest tests.test_lingke_modules.LingkeModulesTest
```

涉及交互调整时，还需通过 `/kongmu` 检查角色选择、卡带选择、方案生成和返回工具主页。

## 黑羽（2026-09-16 接入）

黑羽使用稳定ID `char_heiyu`，魂属性；在空幕中对所有用户开放，排轴权限独立。现有数值来源为用户提供的[腾讯文档《黑羽资料汇总》](https://docs.qq.com/aio/DVVRZempSSmZ4THVj?p=MluEyftYtfADwlqnzgRzww)；Nanoka NTE `1.4.6` 已有角色 `1042` 的资源记录，本次只用来核对图片，不改写数值来源。Ⅲ型特化，每个Ⅲ型驱动提供8%暴击率。5×5拓扑按文档原始方向录入：

```text
11110
11110
11011
01111
01111
```

1表示可用格，合计20格；0表示空洞。头像 `images/characters/avatar/黑羽.webp` 由最新 NTEData 默认立绘裁成方形；同仓库的独立256图只有圆形人物蒙版。旧附件 `UI.7z` 的 `player_heiyu2_256.png` 属于时装，仍不使用；新图的上游提交、路径与裁切参数见[角色图片资源说明](../../../docs/character-image-assets.md#黑羽默认造型补图2026-09-24)。验证见 `tests.test_shaft_heiyu` 和 `tests.test_shaft_publication`。

## 明音凛（2026-09-24 接入）

明音凛沿用排轴稳定 ID `char_akane`，在空幕中对所有用户开放。空幕拓扑与被动来自 Nanoka NTE `1.4.6` 角色 `1057`：7×7 画布中间四行各五格，共20格；Ⅲ型特化，每个Ⅲ型驱动使攻击力增加10%。本次只为独立空幕规划器接入该配置，不改排轴既有的用户估算面板和机制。头像使用最新 NTEData 默认立绘裁成的共享方形图，来源和裁切参数见[角色图片资源说明](../../../docs/character-image-assets.md#明音凛默认造型补图2026-09-24)。验证：`tests.test_shaft_publication.ShaftPublicationTest.test_akane_kongmu_is_public_with_source_backed_grid`。
