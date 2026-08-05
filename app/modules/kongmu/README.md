# 空幕计算模块

空幕计算是独立网页工具，页面入口为 `/kongmu`，用于选择角色与卡带并生成可用的空幕搭配方案。

## 模块范围

- 页面展示角色、卡带和计算结果。
- 浏览器只提交选择并渲染结果，方案计算由后端应用服务完成。
- 该模块不依赖异象对决的房间、对局快照或规则结算。
- 残红属于测试角色。服务端仅在当前登录账号的 `Player.shaft_test_whitelisted` 为真时把残红加入目录并允许计算；普通账号和未登录访客均不可见，直接调用计算接口也按角色不存在处理。浏览器会在空幕请求中携带已有登录 token，但可见性判定只在服务端完成。
- 残红的Ⅲ型特化与“每个Ⅲ型驱动增加16%暴击伤害”来自2026-08-04用户提供的腾讯文档。文档未提供空幕格子拓扑，当前暂时复用真红的同类型格子拓扑，并在角色数据中标明来源边界。

## 代码边界

- `app/modules/kongmu/service.py`：catalog 与布局方案计算。
- `app/modules/kongmu/templates/kongmu/index.html`：页面模板。
- `app/modules/kongmu/static/`：页面脚本、样式与模块静态数据。
- `app/routes.py` 中 `/kongmu`、`/api/kongmu/catalog` 和 `/api/kongmu/plan`：页面与 API 入口。

新增算法应留在模块服务中，路由只做输入解析、并发保护和错误转换。

## 验证

```bash
python3 -m unittest tests.test_module_routes.ModuleRoutesTest.test_kongmu_module_page_catalog_and_asset
python3 -m unittest tests.test_module_routes.ModuleRoutesTest.test_kongmu_test_character_requires_test_permission
```

涉及交互调整时，还需通过 `/kongmu` 检查角色选择、卡带选择、方案生成和返回工具主页。
