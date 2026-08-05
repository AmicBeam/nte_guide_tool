# NTE Tools

本仓库是异环主题网页工具集合，而不是单一桌游项目。仓库使用同一个 Flask 应用承载多个相互独立的功能模块，并共享账号、数据库、基础页面框架和通用静态资源。

访问 `/` 可进入仓库级工具主页，再选择具体模块。各模块的产品目标、入口、代码边界和验证方式由各自的 README 维护；根 README 只说明仓库整体结构和公共约定。

工具主页同时提供项目 GitHub、静默之光和异环攻略组的外部入口，并展示网站备案号。

## 模块

| 模块 | 页面入口 | 说明文档 | 用途 |
| --- | --- | --- | --- |
| 异象对决 | `/card-game` | [模块 README](app/modules/card_game/README.md) | 网页卡牌牌桌、构筑、图鉴与对局数据 |
| 空幕计算 | `/kongmu` | [模块 README](app/modules/kongmu/README.md) | 角色空幕与卡带搭配计算 |
| 预配队 | `/preteam` | [模块 README](app/modules/preteam/README.md) | 即将下线；由排轴模块取代，当前保留主 C、队友和属性搭配预览 |
| 排轴计算 | `/shaft` | [模块 README](app/modules/shaft/README.md) | 配装、动作轴、伤害计算与方案广场 |

新增或修改模块时，应优先更新对应模块 README；只有模块索引、共享设施或全仓库约定发生变化时才更新本文件。

## 仓库结构

- `app/`：Flask 应用与共享账号、数据库、路由适配等基础设施。
- `app/modules/`：桌游、空幕、预配队和排轴四个业务模块；每个模块自行维护 README、模板和静态资源。
- `app/templates/`：仓库入口与共享基础模板。
- `app/static/`：通用样式、脚本和跨模块图片资源。
- `app/modules/card_game/content/`、`app/modules/card_game/engine/`：异象对决的内容、规则与对局服务。
- `app/modules/shaft/domain/`：排轴模块的领域计算代码。
- `plugins/`：机器人账号桥接参考实现，与网站仅通过共享数据库交互。
- `docs/`：跨模块设计资料和专项报告。
- `scripts/`：数据导入、本地联调和维护脚本。
- `tests/`：后端、页面与前端静态检查。

详细分层、依赖方向、持久化策略和前端约束见 [AGENTS.md](AGENTS.md)。

## 公共运行方式

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

默认打开 `http://127.0.0.1:5001/`。

当前环境不运行机器人时，可执行下面的命令创建本地联调账号：

```bash
python3 scripts/seed_mock_account.py
```

- 玩家号：`10001`
- 密码：`654321`

## Windows Server 应用目录发布

仓库提供仅更新服务器 `app/` 目录的推送式发布脚本。发布源固定为当前 Git `HEAD`
中已提交的 `app/`；如果 `app/` 存在已暂存、未暂存或未跟踪改动，检查、打包和发布
都会拒绝执行，避免把本地半成品同步到服务器。脚本不要求提交已经推送或合入
`main`，发布版本由当前分支的 `HEAD` 决定。脚本不会上传或覆盖项目根目录的
`.env`、数据库、日志、虚拟环境和其他目录。

首次使用时复制本机配置模板并填写真实 SSH 地址和 Windows 项目路径；本机配置已被
Git 忽略，不会把服务器信息提交到仓库：

```bash
cp scripts/deploy_windows_app.env.example scripts/deploy_windows_app.local.env
```

只检查本机准备情况，不连接服务器：

```bash
scripts/deploy_windows_app.sh --check
```

只在 `dist/deploy/` 生成当前 `HEAD` 的 `app/` 发布包，不连接服务器：

```bash
scripts/deploy_windows_app.sh --prepare
```

确认发布时必须显式执行：

```bash
scripts/deploy_windows_app.sh --deploy
```

远端替换前会把旧 `app/` 移到 Windows 临时目录中作为备份。当前服务器通过
`start_waitress.bat` 启动 Waitress，因此脚本默认停止监听 `8000` 端口的旧进程，
替换 `app/` 后通过 Windows WMI 创建脱离 SSH 会话的独立进程来运行该批处理，并等待
端口连续稳定监听。批处理的标准输入会重定向到 `NUL`，避免其末尾的 `pause` 在进程
退出后遗留隐藏窗口；后台输出追加写入 `logs/waitress-deploy.log`。Caddy 配置不在
`app/` 中，发布时无需重启 Caddy。

如果以后改成 Windows 服务，可设置 `NTE_DEPLOY_SERVICE`，让发布成功后改为重启服务：

```bash
NTE_DEPLOY_SERVICE=NteBoardGame scripts/deploy_windows_app.sh --deploy
```

服务器地址、项目路径可分别通过 `NTE_DEPLOY_HOST` 和
`NTE_DEPLOY_PROJECT` 覆盖。Waitress 批处理和监听端口可分别通过
`NTE_DEPLOY_RESTART_BATCH` 和 `NTE_DEPLOY_LISTEN_PORT` 覆盖；把
`NTE_DEPLOY_RESTART_BATCH` 设置为空字符串可禁用进程重启。

## 共享工程约定

- 路由只负责参数校验、鉴权和调用服务，业务计算留在对应模块的服务或领域层。
- 各模块可以共享账号、数据库和页面基础设施，但不能把某个模块的领域规则写入另一个模块。
- `plugins/` 不能直接引用 `app/`，机器人和网站只通过共享数据库表交互。
- 账号、构筑、房间等关键资料同步入库；异象对决的对局快照使用进程内最新态缓存与异步入库。
- 模块入口应使用能表达模块含义的路由，不使用 `/home` 这类仓库范围不明确的地址。

## 验证

先运行与改动模块对应 README 中列出的检查。公共冒烟检查可运行：

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
