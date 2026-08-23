# 空幕计算模块

空幕计算是独立网页工具，页面入口为 `/kongmu`，用于选择角色与卡带并生成可用的空幕搭配方案。

## 模块范围

- 页面展示角色、卡带和计算结果。
- 浏览器只提交选择并渲染结果，方案计算由后端应用服务完成。
- 该模块不依赖异象对决的房间、对局快照或规则结算。
- 空幕规划器公开展示并允许计算半开放角色；残虹无需登录或排轴受邀/测试权限即可使用。该公开范围只属于空幕模块，不改变排轴模块的角色权限。
- 角色目录会随登录账号权限变化，因此目录接口使用私有不缓存响应，并按 `Authorization` 区分变体，避免匿名目录覆盖测试账号目录。
- 主角目录头像使用共享本地角色资源，通过长效 immutable 静态缓存加载，不依赖 Nanoka 远程回退。
- 残虹头像使用共享资源 `app/static/images/characters/avatar/残虹.png`，空幕与排轴统一引用该文件。
- 残虹的Ⅲ型特化、“每个Ⅲ型驱动增加16%暴击伤害”和20格空幕拓扑已按 Nanoka NTE 1.3.4 正式角色 ID `1036` 的角色详情接入；规划器仍使用半开放角色 ID，以保持受邀/测试权限边界。

## 代码边界

- `app/modules/kongmu/service.py`：catalog 与布局方案计算。
- `app/modules/kongmu/templates/kongmu/index.html`：页面模板。
- `app/modules/kongmu/static/`：页面脚本、样式与模块静态数据。
- `app/routes.py` 中 `/kongmu`、`/api/kongmu/catalog` 和 `/api/kongmu/plan`：页面与 API 入口。

新增算法应留在模块服务中，路由只做输入解析、并发保护和错误转换。

## 验证

```bash
python3 -m unittest tests.test_module_routes.ModuleRoutesTest.test_kongmu_module_page_catalog_and_asset
python3 -m unittest tests.test_module_routes.ModuleRoutesTest.test_kongmu_half_open_character_is_public
```

涉及交互调整时，还需通过 `/kongmu` 检查角色选择、卡带选择、方案生成和返回工具主页。
