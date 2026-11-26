---
name: nte-wiki-cli
description: 使用 NTE Tools Wiki CLI 查询异环的角色、技能、弧盘、材料、料理、副本、轨外之境、全站深境、攻略和已保存排轴，也可按当前登录账号读取已同步的角色配装。适用于需要从 Wiki 核对资料和数值的任务。
---

# 异环 Wiki CLI

用 Python 3.10+ 运行本技能包内与本文件同目录的 `nte-guide.py`。独立下载的 CLI 也可使用。先运行 `python3 nte-guide.py --help`，再按需查看子命令帮助。CLI 通过 HTTP 访问 Wiki，可在任意目录运行，命令中使用实际脚本路径。

## 连接 Wiki

- 优先使用用户指定的 NTE Tools 站点。用 `NTE_GUIDE_URL` 或子命令前的 `--base-url` 指定站点根 URL。两者均未设置时，CLI 使用本地开发站 `http://127.0.0.1:5001`。
- 公共查询可匿名执行。读取本人同步资料、本人排轴或收藏时，沿用可信运行环境已注入的 `NTE_GUIDE_TOKEN`。只向凭据对应的站点发送请求，避免在对话、命令参数或日志中暴露凭据。若缺少凭据，只继续可匿名执行的查询。

## 查找和读取

以下命令假定当前目录是解压后的技能目录，站点已按上节配置。

```bash
python3 nte-guide.py schema
python3 nte-guide.py search 黑羽 --limit 10
python3 nte-guide.py list arcs --limit 10
python3 nte-guide.py get characters 1042 --section overview
python3 nte-guide.py get characters 1042 --ability GA_BlackBird_Melee --level 10 --hit-details
python3 nte-guide.py get characters 1042 --section materials --format markdown
python3 nte-guide.py list challenges
python3 nte-guide.py beginner --format markdown
```

有精确 ID 时直接读取。只有名称时先搜索，从结果中核对分类和 ID；结果同名时先核对目标。分组名、技能 ID、关卡和详情链接以站点返回为准。读取指定关卡时，可在详情命令上用 `--stage`，副本可再传 `--encounter`。

角色技能默认读取 10 级，可用 `--level` 指定等级、`--all-levels` 读取完整等级数组、`--hit-details` 展开逐命中及资源明细。引用数值时注明等级、单位或精炼档位；缺失值保留为未知，必要时用 `schema` 确认单位和字段含义。

## 已同步配装与已保存排轴

```bash
python3 nte-guide.py me --character 1042
python3 nte-guide.py axes --character 黑羽 --sort dps
python3 nte-guide.py axes --scope mine
python3 nte-guide.py axis 123 --section axis
```

用实际轴 ID 替换示例中的 `123`。需要动作步骤用 `--section axis`，需要保存时的计算结果用 `--section result`，需要完整记录用 `--section all`。说明 DPS 属于保存快照。基础 Wiki CLI 的排轴命令提供只读访问，改轴与重算使用排轴模拟器。
