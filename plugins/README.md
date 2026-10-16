# NTE 机器人桥接

本目录由 NTE Tools 维护，机器人宿主负责加载、消息事件和工具编排。不导入 `app/` Python 模块，不读取其他业务数据库。

- `nte_account.py`、`nte_account_db.py`：原有注册、重置密码、角色公开管理；保持现有共享数据库表名和行为；玩家表镜像包含网站的两个权限列，新注册默认关闭，已有权限不改写，并兼容旧插件表缺列。
- 账号只读桥接和配装 PNG 绘图已迁至 `qqbot_fairy` 仓库的 `src/plugins/nte_guide_access.py` 与 `src/plugins/nte_build_card.py`，随机器人发布。本目录不再保留副本；相关实现和测试由机器人仓库维护。
- 公共 wiki CLI 真源为 `app/modules/guide/cli.py`，它本身只依赖标准库，可作为独立文件复制到机器人运行目录；无需导入 Flask。

## 可信运行时契约

QQ 身份必须来自当前消息事件（例如 OneBot `event.user_id`），不可由模型工具参数指定。agent 工具不得暴露 `qq_id`、`token`、数据库路径、脚本路径、输出路径或任意命令参数；游戏账号 ID 仅用于在当前发送者自己的快照中选择。桥接返回的 token 只在运行时内存中使用，通过子进程 `NTE_GUIDE_TOKEN` 环境变量或 HTTP Authorization 头传入，不进入模型、命令行参数、聊天、异常文本或日志。不能把这个只读桥接作为公网的“凭 QQ 号查询 token”接口。

实际数据仍通过本站鉴权 API 读取，由网站再次校验 token 对应用户。有效 token 缺失时提示先登录本站；没有已同步角色时提示在养成页同步，不自动发起游戏短信验证。客户端不得因为 token 过期而自行生成新 token。

## 部署文件与配置

机器人宿主配置并调用以下路径：

- `NTE_GUIDE_CLI_PATH` → 独立部署的 `app/modules/guide/cli.py`。
- 账号桥接和绘图由机器人内置插件按固定位置调用，无需配置 `NTE_GUIDE_ACCESS_PATH` 或 `NTE_BUILD_CARD_PATH`。
- `NTE_GUIDE_PYTHON` → 安装可选依赖的 Python 解释器。
- `NTE_GUIDE_URL` → wiki 根地址；私有请求只向管理员配置的本站地址发送。
- `NTE_DATABASE_PATH` → 网站账号数据库，必须与网站一致，不能指向 bot 的问答／记忆库。
- `NTE_GUIDE_FONT` → 可选中文字体文件；否则依次检测 macOS 黑体／宋体、Windows 微软雅黑／黑体、Linux Noto CJK／文泉驿。

图片依赖在机器人仓库安装；注册插件依赖沿用宿主的 NoneBot、OneBot、Peewee、python-dotenv 环境。不要仅因接入新工具而删除或复制替换既有账号／问答数据库。

## 调用示例

公共查询不带 token：

```bash
python nte-guide.py search 黑羽
python nte-guide.py get characters 1042 --section abilities
python nte-guide.py list arcs --limit 20
```

可信运行时注入 `NTE_GUIDE_TOKEN` 后，读取当前用户自己的快照：

```bash
python nte-guide.py me
python nte-guide.py me --role-id GAME_ROLE_ID --character 1042
```

输出为 `{"data":[profile,...]}`；`profile` 包含 `role_id`、`role_name`、`updated_at`、`characters`。角色保留已保存的 `level`、`awaken_level`、`resonance_level`、`properties`、`skills`、`equipment.fork` 与 `equipment.suit.core/pie`。`me` 不支持指定其他本站用户。认证请求拒绝重定向；认证错误不回显服务响应。

```bash
python nte_build_card.py --input - --character 1042 --role-id GAME_ROLE_ID --output build.png
```

stdin 传上述 CLI JSON 或单个 `profile`，也可将 `--input` 设为可信运行时的临时 JSON 文件。stdout 只返回 `image_path`、`width`、`height`；诊断写 stderr，失败退出码为1。可通过 `--reference` 指定离线的公共角色详情 JSON；否则仅从本站补充角色头像与技能名称，网络失败仍可使用已保存字段渲染。Windows/Linux/macOS 使用同一份脚本。

机器人必须自行建立私有临时目录，固定脚本与输出路径，读出 PNG 字节后清理 JSON/PNG，并通过适配器的图片消息发送；不要让模型拼 CQ 图片码或指定任意本地文件。图片只用于响应当前发送者自己的配装请求，不做主动推送。

验证：`.venv/bin/python -m unittest tests.test_guide_bot_tools tests.test_nte_account_db tests.test_guide_user_data tests.test_guide -v`。网站侧使用合成资料验证 CLI 认证、重定向、私有错误脱敏与注册兼容；只读桥接、PNG 和 QQ 消息收发测试由机器人仓库负责。
