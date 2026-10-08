# NTE Tools

本仓库是异环主题网页工具站，不是单一桌游项目。同一个 Flask 应用承载多个独立模块，并共享账号、数据库、基础页面和通用静态资源。

访问 `/` 进入工具主页，再选择模块。各模块的产品范围、页面入口、代码边界和验证命令由各自 README 维护；本文件只说明仓库入口、模块索引和公共运行方式。

主页同时提供 [GitHub](https://github.com/AmicBeam/nte_board_game)、静默之光和异环攻略组入口，并展示备案号。

## 模块

| 模块 | 入口 | 说明 | 用途 |
| --- | --- | --- | --- |
| 异能对决 | `/card-game` | [模块 README](app/modules/card_game/README.md) | 现行四人轮替牌桌、构筑、图鉴、回放、教学与人机 |
| 空幕计算 | `/kongmu` | [模块 README](app/modules/kongmu/README.md) | 角色空幕与卡带搭配计算 |
| 排轴计算 | `/shaft` | [模块 README](app/modules/shaft/README.md) | 配装、动作轴、伤害计算与方案广场 |
| 预配队 | `/preteam`（主页入口已移除） | [模块 README](app/modules/preteam/README.md) | 即将下线，由排轴取代；旧链接仍可访问 |

异能对决的现行规则见 [完整手册](docs/everness-item-chain-card-design.md)。默认页面已是四人轮替；旧盖卡入口与数据看板尚未迁移，见模块 README。

新增或修改模块时，先更新对应模块 README；只有模块索引、共享设施或全仓库运行方式变化时才改本文件。

## 仓库结构

- `app/`：Flask 应用与共享账号、数据库、鉴权和路由适配。
- `app/modules/`：四个业务模块；各自维护 README、模板和静态资源。
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

## Windows 发布

`scripts/deploy_windows_app.sh` 只把当前 Git `HEAD` 里的 `app/` 推到服务器，不上传 `.env`、数据库、日志或虚拟环境。`app/` 有未提交改动时默认拒绝；确认只发 `HEAD` 时可加 `--allow-dirty-app`。不要求该提交已推送或合入 `main`。

```bash
cp scripts/deploy_windows_app.env.example scripts/deploy_windows_app.local.env
scripts/deploy_windows_app.sh --check
scripts/deploy_windows_app.sh --prepare
scripts/deploy_windows_app.sh --deploy
```

`--prepare` 只在 `dist/deploy/` 打 ZIP。`--deploy` 先备份远端 `app/`，再用 `robocopy` 替换；失败则回滚。默认停止 8000 端口上的 Waitress，再以后台方式运行 `start_waitress.bat`（本机回环 `127.0.0.1:8000`，由 Caddy 反代）。Caddy 配置不在 `app/` 中，发布时不重启。

可用环境变量覆盖：`NTE_DEPLOY_HOST`、`NTE_DEPLOY_PROJECT`、`NTE_DEPLOY_RESTART_BATCH`、`NTE_DEPLOY_LISTEN_PORT`。若改成 Windows 服务，设置 `NTE_DEPLOY_SERVICE` 后改为重启服务。

## 共享约定

- 新增或更新角色头像、默认立绘时优先从最新 NTEData 获取，缺图时使用 Nanoka；保持原有缩放与头部对齐。具体来源核对与 WebP 交付流程见 [角色图片资源说明](docs/character-image-assets.md)。
- 数值模型清单按原始字节校验；Git 属性禁止模型 JSON 自动换行转换，保留 Windows 导出文件及其 SHA。Markdown 的双空格换行按文档语法保留。
- 路由只做参数校验、鉴权和调用服务；业务计算留在对应模块。
- 模块可共享账号、数据库和页面基础设施，不能把一个模块的领域规则写入另一个模块。
- 账号、构筑、房间等关键资料同步入库；异能对决对局快照使用进程内最新态加异步入库。
- 模块入口使用能表达模块含义的路径。

## 验证

先跑对应模块 README 中的检查。仓库级冒烟：

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
