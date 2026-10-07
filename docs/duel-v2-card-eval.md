# V2 卡牌效果自对弈覆盖评估

版本：现行 V2 卡池契约 · 更新：2026-10-07

这是**随机合法动作自对弈覆盖基线**（stochastic legal-action selfplay coverage baseline），**不是**权重训练、RL 对战强度或平衡证明。`model=0`，`training_updates=0`。没有 torch 依赖。

本评估只消费冻结的不可变 V2 接口 `new_game` / `acting_side` / `observe` / `apply_action`。策略只看公开观察里的 `legal_actions[].action`，不读取隐藏牌序或内部状态。监控在 `apply_action` 之后离线检查 `state['events']`，不把隐藏信息交给策略。

## 做什么

- 半局基础预组，半局 coverage 构筑：每角色 8 种可构筑牌各 1 张，共 32 张，经 `catalog.validate_deck`；衍生牌不进入开局构筑。
- 先后手 `a/b` 轮换；每局独立确定性 seed。
- 加权选择偏向 `play_card` 以收集效果，其次 `attack` / `awaken` / `cycle`；只要存在非认输动作，就不会随机认输。包含起手换牌与 pending `choose`。
- 不修改规则、卡数据或数值。

## 硬限制（先到先停，中局也截断）

| 项 | 上限 |
| --- | ---: |
| 原始 `apply_action` 次数 | 10000 |
| 已开局数 | 128 |
| 墙钟秒 | 1800 |
| 单局原始动作 | 500 |

CLI 不允许突破上述上限。输出目录必须是新目录，不覆盖已有实验。

## 产物

- `config.json`：预算、PID、`model=0`、`training_updates=0`
- `status.json`：第一次成功 `apply` 后立即原子写入且 `raw_steps > 0`；之后周期性更新
- `progress.jsonl`：第 1 步、每 10 步、以及出牌/效果相关步，fsync
- `summary.json`：正常结束、异常、预算耗尽都会写
- 异常时还有 `traceback.txt`；超时有 `timeouts` 计数

计数：胜/平/截断、座位与先后手、卡牌使用、复制使用、公开事件次数、按卡归因的效果事件。**出牌使用 ≠ 效果被观察到**。当前 `CARD_ID_ORDER` 中所有卡牌（含衍生牌）的初始计数均为 0；这与单套构筑的 32 张牌分别统计；没有可靠卡上下文的效果不归因；从未观察到效果的卡明确标为 `NOT_COVERED`。这是代理覆盖，不声称测过每条机制。不保存牌库内部顺序或密钥。

## 命令

检查（不 `new_game`、不步进）：

```bash
python3 scripts/evaluate_duel_v2_cards.py --check
```

Windows 冻结 venv（由主任务启动，本实现不连 Windows）：

```powershell
& .venv\Scripts\python.exe -X utf8 scripts/evaluate_duel_v2_cards.py --check
```

前台监督运行（监督进程管墙钟，worker 内 watchdog 管卡住的 `apply`）：

```bash
python3 scripts/evaluate_duel_v2_cards.py --run --output /tmp/duel-v2-card-eval-new
```

Windows 无人值守分离启动（`--launch` 拉起同一 `sys.executable` 的监督进程并打印 PID；SSH 断开后监督进程继续；监督再拉 worker，等待 `max_seconds<=1800` 后终止子进程。worker 不再派生子进程）：

```powershell
& .venv\Scripts\python.exe -X utf8 scripts/evaluate_duel_v2_cards.py --launch --output D:\eval\duel-v2-card-eval-new
```

可选预算（均不可超过硬上限）：

```bash
python3 scripts/evaluate_duel_v2_cards.py --launch --output /tmp/duel-v2-card-eval-new \
  --max-raw-steps 10000 --max-games 128 --max-seconds 1800
```

若存在 `source-manifest.json`，启动前按 SHA-256 校验列出的文件。

`--supervise` / `--worker` 为内部模式。不要在启动后继续无限监控；外部只需确认 `status.json` 的 `raw_steps > 0` 与 `model=0`。

## 证据边界

功能覆盖不等于平衡。当前目录中未观察到效果的卡牌是 `NOT_COVERED`，不是“效果不存在”。复制牌与原牌分开计数。pending 选择的后续效果只有在公开 `card` 字段或同一出牌上下文可可靠连接时才归因。

组牌与统计回归：`.venv/bin/python -m unittest tests.test_duel_v2_rl_eval`。假引擎验证预算、两种构筑轮换、复制的历史统计及公开事件归因；单元测试不启动真实自对弈或模型训练。
