# 预配队模块

预配队是即将下线的独立队伍搭配预览工具，页面入口为 `/preteam`，用于浏览主 C、候选队友和属性组合。其能力将由排轴模块取代；下线前保留现有页面和计算功能。

## 模块范围

- 展示主 C 与队友候选、头像和属性信息。
- 在页面内完成配队组合与结果预览。
- 工具主页已移除预配队入口；旧链接 `/preteam` 仍可访问，页面继续展示“即将下线，并由排轴模块取代”的迁移说明。
- 不参与异象对决构筑、对局规则或排轴伤害计算。

## 代码边界

- `app/modules/preteam/catalog.py`：主 C 与队友候选数据。
- `app/modules/preteam/templates/preteam/index.html`：页面结构与当前交互实现。
- `app/routes.py`：`/preteam` 页面适配入口。
- `app/modules/preteam/static/`：模块专用静态资源。
- `app/static/images/characters/`：与其他模块共享的角色图片。

后续拆分候选数据或交互脚本时，仍应保持在预配队模块范围内，避免写入异象对决内容定义。

## 验证

```bash
python3 -m unittest tests.test_module_routes.ModuleRoutesTest.test_preteam_module_page_and_asset
```

涉及交互调整时，还需通过 `/preteam` 检查主 C 切换、队友选择、属性展示、响应式布局和返回工具主页。
