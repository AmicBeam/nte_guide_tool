# NTE Tools

本仓库是异环主题网页工具站，不是单一桌游项目。同一个 Flask 应用承载多个独立模块，并共享账号、数据库、基础页面和通用静态资源。

访问 `/` 进入工具主页，再选择模块。合轴收益表仅通过 `/overlap` 直达链接访问，主页不展示其入口。各模块的产品范围、页面入口、代码边界和验证命令由各自 README 维护；本文件只说明仓库入口、模块索引和公共运行方式。

主页同时提供 [GitHub](https://github.com/AmicBeam/nte_board_game)、静默之光和异环攻略组入口，并展示备案号。

## 模块

| 模块 | 入口 | 说明 | 用途 |
| --- | --- | --- | --- |
| 异能对决 | `/card-game` | [模块 README](app/modules/card_game/README.md) | 现行四人轮替牌桌、构筑、图鉴、回放、教学与人机 |
| 空幕计算 | `/kongmu` | [模块 README](app/modules/kongmu/README.md) | 角色空幕与卡带搭配计算 |
| 排轴计算 | `/shaft` | [模块 README](app/modules/shaft/README.md) | 配装、动作轴、伤害计算、方案广场与比较计算引擎 |
| 攻略数据库 | `/guide` | [模块 README](app/modules/guide/README.md) | 角色、弧盘、料理、轨外之境、全站深境、角色相关轴与配装收益比较、通用／自建攻略及自动保存的Markdown撰写；匿名阅读与登录创作，支持JSON、Markdown与CLI |
| 合轴收益表 | `/overlap` | [模块 README](app/modules/overlap/README.md) | 静态收益表；双角色分页在浏览器匹配可循环合轴方案 |

异能对决的现行规则见 [完整手册](docs/everness-item-chain-card-design.md)。默认页面已是四人轮替；旧盖卡入口与数据看板尚未迁移，见模块 README。

新增或修改模块时，先更新对应模块 README；只有模块索引、共享设施或全仓库运行方式变化时才改本文件。

## 仓库结构

- `app/`：Flask 应用与共享账号、数据库、鉴权和路由适配。
- `app/modules/`：五个业务模块；各自维护 README、模板和静态资源。
- `app/modules/card_game/content/duel_v2/`、`engine/duel_v2/`：现行异能对决卡池与纯规则引擎。
- `app/modules/card_game/rl/`：训练框架；导入或启动网站时不会自动训练。
- `app/modules/shaft/domain/`：排轴领域计算。
- `app/templates/`、`app/static/`：仓库入口、共享基础模板和跨模块静态资源。
- `plugins/`：机器人账号桥接；与网站只通过共享数据库交互，不能引用 `app/`。
- `artifacts/rl-evals/`：本地训练／评估报告、原始数据和权重，已被Git忽略；不随仓库推送。
- `docs/`：跨模块长期设计资料与训练手册，不存放单次训练报告。异能对决规则以完整手册为准，排轴机制以对应验收文档为准。
- `scripts/`：本地联调、数据导入、训练入口和 Windows 发布脚本。
- `tests/`：后端、页面与前端静态检查。

分层、依赖方向和持久化策略见 [AGENTS.md](AGENTS.md)。共享 SQLite 写事务使用 `BEGIN IMMEDIATE`，避免异步对局存档与登录等读后写竞争。

## 本地运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

开发服务监听 `0.0.0.0:5001`，本机打开 `http://127.0.0.1:5001/`，局域网用当前机器 IP 的同一端口。Windows 可双击 `start.bat`。

不跑机器人时，创建本地联调账号：

```bash
python3 scripts/seed_mock_account.py
```

- 玩家号：`10001`
- 密码：`654321`
- 该账号带测试白名单，并写入当前可见的官方预组。

可选依赖不进入默认网站环境：

- 异能对决「高级人机」：`pip install -r requirements-ai.txt`，仅 CPU 推理。两套镜像模型随 `app/` 发布，路径见 [模块说明](app/modules/card_game/README.md#高级人机)。
- 训练与评估：`pip install -r requirements-rl.txt`。网站启动不会加载训练权重或开始训练；命令见 [公开卡池训练](docs/duel-v2-public-training.md)。

## 安卓安装包

异能对决提供复用现有网站的轻量安卓外壳。构建与安装说明见 [安卓项目](android/README.md)。安装后需要联网，网站服务与数据继续复用。

## Windows 发布

`scripts/deploy_windows_app.sh` 只把当前 Git `HEAD` 里的 `app/` 推到服务器，不上传 `.env`、数据库、日志或虚拟环境。`app/` 有未提交改动时默认拒绝；确认只发 `HEAD` 时可加 `--allow-dirty-app`。不要求该提交已推送或合入 `main`。

```bash
cp scripts/deploy_windows_app.env.example scripts/deploy_windows_app.local.env
scripts/deploy_windows_app.sh --check
scripts/deploy_windows_app.sh --prepare
scripts/deploy_windows_app.sh --deploy
```

`--deploy` 自动读取服务器项目根目录的 `.nte-deploy-app.json`，按逐文件 SHA-256 比较当前提交：只上传新增／修改文件，并携带完整目标清单与删除清单。改名按删除旧路径＋新增新路径处理，纯删除、大小写改名、文件与目录互换、换分支和回退旧提交均支持；比较不依赖 Git 历史或本机旧 ZIP。首次使用新版脚本没有服务器基准时自动全量，也可通过 `--deploy --full` 强制全量恢复。

一次部署共用同一条 SSH 连接，密码认证通常只需输入一次；读取基准、上传和激活复用该连接。首次连接和握手超时为 15 秒，握手失败时保留 SSH 错误并在上传前退出。连接丢失时后续步骤直接报错，不再重复询问密码。脚本结束（包括失败）会关闭连接并清理本机私有控制套接字，不保存密码。辅助脚本上传和 PowerShell 模块进度输出已静默，应用包上传进度及错误仍显示。

服务器在项目根目录的 `.nte-deploy/<release-id>/` 中组装并核验完整新 `app/`，删除清单中的文件不会进入新目录；旧 `app/` 保留为该目录下的 `backup-app/`。所有增量基准文件（包括将修改／删除的文件）都须与服务器清单一致，手工改动或缺文件会拒绝增量，须检查后使用 `--full`。服务器部署锁与基准身份校验阻止并发包交错应用；重启或状态提交失败时回滚，成功后才原子更新基准。切换前写入 `.nte-deploy-app.pending.json` 恢复记录：进程异常退出或回滚暂时失败后，下一次读取基准／部署会先在锁内恢复旧目录和原进程状态；若新基准已经提交，则保留新目录并清理记录。恢复失败会保留记录与备份并明确报错，不继续叠加新的增量。无文件变化时不替换目录、不重启进程。账号、数据库、日志等项目根目录资料不进入部署范围；`app/` 内缓存或未跟踪文件不带入新目录。

`--prepare` 默认只在 `dist/deploy/` 生成全量 ZIP，不连接服务器；可用 `--prepare --base-state /path/to/saved-state.json` 离线生成增量包。清单基于 Git 导出的已提交字节（保留 `export-ignore` 等属性），不读取未提交的 `app/` 文件。本机需 Python 3.10+，无需第三方库。`--deploy` 成功后自动删除本机和服务器上的本次 ZIP，失败时保留包；`--prepare` 的包保留供检查。服务器旧版本备份仍保留，清理历史 ZIP 不会影响下一次增量比较。

默认停止 8000 端口上的 Waitress，再以后台方式运行 `start_waitress.bat`（本机回环 `127.0.0.1:8000`，由 Caddy 反代）；Windows 服务模式会先停服务再切换目录。Caddy 配置不在 `app/` 中，发布时不重启。

可用环境变量覆盖：`NTE_DEPLOY_HOST`、`NTE_DEPLOY_PROJECT`、`NTE_DEPLOY_RESTART_BATCH`、`NTE_DEPLOY_LISTEN_PORT`。若改成 Windows 服务，设置 `NTE_DEPLOY_SERVICE` 后改为重启服务。

部署专项验证：`python3 -m unittest tests.test_windows_app_deploy -v`。有 `pwsh`／Windows `powershell.exe` 时同时执行真实文件激活、纯删除、无变化、损坏包、基准漂移、并发锁、重启失败回滚及关键切换点异常退出后的恢复；也可用 `NTE_TEST_PWSH=/path/to/pwsh` 指定运行时。macOS 测试仅替换 Windows 输入路径规范化，实际校验、目录切换和状态提交代码保持一致；Windows 服务与 Waitress/WMI 重启仍需 Windows 环境验证。

## 共享约定

- 新增或更新角色头像、默认立绘时从最新 NTEData 获取，缺图时保留现有资源或明确缺失；保持原有缩放与头部对齐。具体来源核对与 WebP 交付流程见 [角色图片资源说明](docs/character-image-assets.md)。
- 路由只做参数校验、鉴权和调用服务；业务计算留在对应模块。
- 模块可共享账号、数据库和页面基础设施，不能把一个模块的领域规则写入另一个模块。
- 账号、构筑、房间等关键资料同步入库；异能对决对局快照使用进程内最新态加异步入库。
- 模块入口使用能表达模块含义的路径。

## 验证

先跑对应模块 README 中的检查。仓库级冒烟：

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

攻略数据库的个人练度同步直接集成于网站后端；养成计算仍可匿名使用，读取和保存个人游戏资料须先登录本站。更新部署需同步安装 `requirements.txt` 中的 `pycryptodome`；数据库专表由网站启动时创建，详情见攻略模块 README。

机器人账号只读桥接与配装绘图由 `qqbot_fairy/src/plugins/` 维护，本仓库不再保留这两个副本。本站保留注册插件、公开 Wiki CLI 与本人资料鉴权 API，部署边界见 [机器人桥接说明](plugins/README.md)。CLI 公共查询无需登录，`me` 通过运行时注入的 `NTE_GUIDE_TOKEN` 读取本人已保存配装。

Wiki 登录用户可使用右下角浮动攻略助手，支持图片、本站资料／在线轴查询和TapTap攻略搜索。部署时在`.env`配置`NTE_GUIDE_CHAT_BASE_URL`、`NTE_GUIDE_CHAT_API_KEY`、`NTE_GUIDE_CHAT_MODEL`，模型需支持视觉和函数工具调用；可选受信任CA路径`NTE_GUIDE_CHAT_CA_BUNDLE`。会话只保存在浏览器，超过6小时间隔另开上下文；页面标题与地址临时附带，不进入历史。详细边界与验证见攻略模块README。
