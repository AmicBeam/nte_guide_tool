---
name: shaft-content-maintenance
description: Maintain NTE shaft-module characters, actions, arcs, cartridges, buffs, and their source-backed tests. Use when adding or revising shaft content data or its catalog ordering; do not use for account permissions or unrelated card-game content.
---

# 排轴内容维护

开始修改前完整读取 `app/modules/shaft/README.md`，并按根 `AGENTS.md` 的任务范围继续读取排轴动作、Buff 或架构文档。数值与机制来源边界遵循仓库现有规则，不用记忆补全。

## 弧盘目录顺序

弧盘选择器按角色适应类型过滤 `app/modules/shaft/static/data/arcs.json`，并保留文件中的相对顺序。

- 新增弧盘时，将它插入同适应类型所有现有弧盘之前，使本次新增弧盘成为该类型下拉框第一项；不要仅追加到文件末尾。
- 同一次增加多个弧盘时，按用户要求排列；未指定时按最新录入者优先。不要重排其他适应类型的既有条目。
- 在排轴前端或目录测试中断言该弧盘是对应适应类型的第一项。以后新增同类型弧盘时，同步把断言更新为更新的首项。
- 弧盘本体、精炼数据、Buff 配置和来源备注仍需保持 ID 与覆盖范围一致；顺序调整不能改变已有 ID 或配装兼容性。

## 验证

至少运行受影响的排轴目录测试；涉及计算或 Buff 时运行完整排轴测试：

```bash
python3 -m unittest tests.test_shaft_frontend_layout tests.test_shaft_arc_compatibility
python3 -m unittest discover -s tests -p 'test_shaft_*.py'
```
