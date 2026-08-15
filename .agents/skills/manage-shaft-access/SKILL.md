---
name: manage-shaft-access
description: Manage NTE shaft-module account permissions and character release levels. Use when adding or removing invited users or test users, changing a character between public / half-open / test-only access, checking the permission naming convention, or generating local and Windows production-server commands for these operations.
---

# 排轴权限管理

统一使用三级权限，不新增同义字段或临时白名单：

| 等级 | 角色 `access_level` | 用户字段 | 能力 |
| --- | --- | --- | --- |
| 正式 | `public` | 无 | 查看公开角色 |
| 受邀 | `invited` | `Player.shaft_invited` | 查看公开和半开放角色 |
| 测试 | `test` | `Player.shaft_test_whitelisted` | 查看全部角色 |

测试权限高于受邀权限。账号同时拥有两个字段时按测试用户处理。

## 命名格式

- 中文用户名称固定为“正式用户”“受邀用户”“测试用户”。
- 中文角色状态固定为“公开角色”“半开放角色”“测试角色”。
- 数据库角色等级固定为 `public`、`invited`、`test`，不要使用 `half-open`、`preview`、`beta` 等别名。
- 用户字段固定为 `shaft_invited` 和 `shaft_test_whitelisted`。
- CLI 动词固定为：
  - 受邀用户：`invite`、`uninvite`、`list-invited`。
  - 测试用户：`add`、`remove`、`list`。这些旧动词为兼容现有测试白名单脚本而保留。
  - 查询任意账号：`status <玩家账号>`，输出 `public`、`invited` 或 `test`。

## 账号授权

优先运行仓库脚本：

```bash
python3 scripts/manage_shaft_test_whitelist.py invite <玩家账号>
python3 scripts/manage_shaft_test_whitelist.py uninvite <玩家账号>
python3 scripts/manage_shaft_test_whitelist.py add <玩家账号>
python3 scripts/manage_shaft_test_whitelist.py remove <玩家账号>
python3 scripts/manage_shaft_test_whitelist.py status <玩家账号>
```

用户明确说“服务器上的，本地不用”时，不运行本地脚本。先部署包含迁移的代码，再通过部署脚本使用的同一 SSH 配置操作生产库：

```bash
scripts/deploy_windows_app.sh --deploy --allow-dirty-app

source scripts/deploy_windows_app.local.env && \
ssh "$NTE_DEPLOY_HOST" powershell.exe -NoProfile -Command \
"Set-Location -LiteralPath '$NTE_DEPLOY_PROJECT'; python -c 'from app.db import init_db; from app.models import Player; init_db([Player]); p=Player.get_or_none(Player.player_uid == str(<玩家账号>)); assert p is not None; p.shaft_invited=True; p.save(only=[Player.shaft_invited]); print(p.player_uid,p.shaft_invited)'"
```

设置测试用户时，把 Python 代码最后一段替换为：

```python
p.shaft_test_whitelisted=True; p.save(only=[Player.shaft_test_whitelisted]); print(p.player_uid,p.shaft_test_whitelisted)
```

取消权限时把对应值设为 `False`。不要在代码中硬编码具体玩家账号，也不要把生产账号迁移伪装成开发环境默认数据。

## 角色开放级别

角色状态由 `ShaftCharacterPublication` 保存：

- 公开角色：`is_published=True` 且 `access_level='public'`。
- 半开放角色：`is_published=False` 且 `access_level='invited'`。
- 测试角色：`is_published=False` 且 `access_level='test'`。

新增默认角色时，在 `app/modules/shaft/service.py` 使用以下集合命名：

```python
DEFAULT_UNPUBLISHED_CHARACTERS = {...}  # 所有非公开角色
HALF_OPEN_CHARACTERS = {...}            # 其中 access_level=invited 的角色
RELEASED_CHARACTERS = {...}             # 强制迁移为 public 的角色
```

同步修改网站模型与迁移、`plugins/nte_account_db.py` 的独立表映射、排轴目录/保存/发布/市场过滤、共享角色目录（当前包括空幕）、模块 README 和权限测试。机器人 `公开角色 <角色名>` 只负责把角色提升为完全公开。

## 验证

1. 验证正式用户不能选择受限角色。
2. 验证受邀用户能选择 `invited`，但不能选择 `test`。
3. 验证测试用户能选择两者。
4. 同口径验证保存、发布、市场、收藏和详情接口。
5. 运行：

```bash
python3 -m unittest discover -s tests -p 'test_shaft_*.py'
python3 -m unittest tests.test_module_routes.ModuleRoutesTest.test_kongmu_test_character_requires_test_permission
```

生产账号变更后必须回读 `status` 或打印字段值。报告本地与服务器分别是否修改，不把“代码已支持”表述成“生产账号已授权”。
