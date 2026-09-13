# 四人轮替 V2 强化学习框架

> 2026-09-12：训练用 GPU 候选打分网络 + 可选 Windows 多进程采样。网站对手仍是规则 AI，不加载训练权重。卡效覆盖评估由独立任务维护。

状态：2026-09-12，可运行短程训练与固定种子策略评估。游戏规则以 [V2.4 手册](everness-item-chain-card-design.md) 为准，接口以 [实现契约](duel-v2-implementation-contract.md) 为准。小样本结果只用于发现问题，不是平衡结论。

## 文件与边界

- `app/modules/card_game/rl/backend.py`：仅对接 `engine.duel_v2` 六接口中的开局、行动方、投影、动作和结果；不依赖旧盖卡模型、HTTP 或数据库服务。
- `contracts.py`、`adapter.py`：可注入的纯内存接口与决策适配。策略仅收到观察投影和合法动作，不收到原始 state。
- `encoding.py`：固定尺寸公开局势和候选动作编码。观察与单候选维度随 catalog 变化，当前约 2468 / 203，动作容量 256；容量溢出显式报错，不删除候选。
- `policies.py`：`VisibleEngineRulePolicy` 用公开 `legal_actions` 复现引擎规则 AI；`RandomMaskedPolicy` 是未训练基线；`MaskablePolicy` 对已加载模型做 mask 推理。
- `gym_compat.py`：可选 Gymnasium Env 工厂；只有调用 `make_gym_env` 才导入 Gymnasium 和 NumPy。
- `checkpoint.py`：`--check` 仍只读写无权重元数据；训练产物另存 `model.zip` + `checkpoint.json`（累计步数、更新次数、版本）。
- `networks.py`：`CandidateScoringPolicy` 在 GPU 上对每个候选打分，不把 256×199 展平进 MLP。打分拆成通用分（动作类型、卡类、费用、目标座位）和卡身份分（catalog ID embedding）。
- `transfer.py`：单套牌解析、同角色换卡、新卡 embedding 从同类旧卡初始化；换牌时冻住通用分，可选对共享动作做 KL。一次训练只使用一套牌，不混合多构筑。
- `scripts/train_duel_v2.py`：`--check` 不创建对局；`--train --output DIR` 默认 8 环境 spawn、候选打分策略。学习方（训练和评估）不提供认输动作，网站玩家仍可认输。评估对手仍是公开规则 AI。

核心接口依赖现有网站环境即可导入；NumPy/Gymnasium/SB3 仅列在独立 `requirements-rl.txt`。不要在网站启动链路中创建训练器或环境。导入 `app.modules.card_game.rl` 不创建对局、不调用 reset/step/predict/learn。

## 观察与动作

引擎的 `observe(state, side)` 是唯一信息边界。己方手牌、双方角色/资源/弃牌/记录可见；对手手牌通常隐藏，公开复制保留。九原检视对手手牌及其他检视效果，仅在引擎开放的 pending choice 中编码候选。

角色座位跟随当前官方创生预组（现为 `nanali/iloy/zero/jiuyuan`），卡牌按这些角色在 catalog 中的顺序编码；元数据校验这些顺序。使用 `shape_id`；保留 last_front、各方 turn_count、疲劳、已用次数（含瞬发/同频及角色 once 键）、是否曾环合、花伤加成、下次攻击加成、返回标记和负面状态。未知 `used` 键会报错，需升级编码器，不能静默丢掉。复制按公开归属座位编码，并保留原作者；复制期限和负面状态期限以对应操作者回合计数计算剩余时间。记录不要求实体卡实例 ID，结算中卡牌单独编码。

手牌最多 10 个槽位，记录 2 个，私有选择 10 个。选择、弃牌和检视都作为独立动作；起手换牌以手牌槽位的多选标记表示。普通出击/终结编码角色，出牌编码可见实体牌，目标编码双方角色或公开牌；choose 同时保留选择下标和候选卡牌特征。对局不再提供弃一抽一。

这仍是**部分可观察原型**：不推断隐藏牌，不包含完整历史记忆；只保留最近 8 条事件的类型/方向/数值，不把中文日志输入模型。新增未知事件进入 explicit unknown 槽；未知卡牌/角色/选择类型及未支持的次数键报错。能量上限按属性为 5 或 6，编码按 6 归一化；扩展卡池、资源范围、历史需求后须升级编码与测试，不能宣称完整 Markov 状态或已经找到最佳策略。

## 决策、奖励与截断

`DuelAdapter(backend, encoder, learning_side='a', opponent=明确提供的冻结策略)` 支持 a/b。构造不运行环境；显式 `reset(seed=...)` 才开局，并有界推进冻结对手至学习方。无默认对手，不自动启动自我对弈。

`reset` 返回 `Transition`；`decision()` 返回带独立 revision 的局势、动作、特征和 mask。`step(index, revision=...)` 拒绝非法编号、过期 revision 和已结束局面的操作。revision 跨 reset 不重用。返回值与传给策略的数据均隔离复制，策略不能通过修改候选改变结算动作。

终局是胜 +10、负 -10、平局 0。学习方非终局 step 另加少量公开事件塑形：己方环合 +0.05、己方觉醒 +0.08、对对手造成的生命损失每点 +0.02、对手角色倒地 +0.10、对手进入倾陷 +0.02，单步塑形上限 0.5。终局那一步只记 ±10 / 0，不把中间分叠成 -9.x。对局得分分开记 `terminal` 与 `shaping`（训练写入 `episodes.jsonl`）。最大中间事件相对获胜约 100 倍。只计「进入倾陷」，不计倾陷值 +1 或倾陷结束。对手环合/觉醒、自己受伤、己方倒地和己方倾陷不计分。截断不当输。

`max_game_actions` 默认 1000，`max_opponent_actions` 默认每次推进 100。达到任一保护限制记为 truncated、奖励 0；真正终局优先于限制。info 提供 `truncation_reason` 与 `bootstrap_allowed`。截断保留真实可见候选用于估值，但不能继续 step；真正终局为空 mask。

Gym 工厂返回真正的 `gymnasium.Env`，采用 Dict 观察 `{state, candidates}` 和 Discrete(256)，配合 MaskablePPO 的 `MultiInputPolicy`；`action_masks()` 位于环境内部。不能只把 state 向量喂给策略而丢掉候选特征。

Gym reset 若已经终局或没有可执行动作，明确报错，不把全 False mask 交给策略。对手窗口或预算截断返回 `truncated=True`、奖励 0，不记为输局，也不中断 Gym 循环；`bootstrap_allowed` 仍标明能否用学习方决策价值估值。训练与评估必须走 mask-aware 接口，不能用忽略 mask 的随机 `check_env` 代替合法动作验收。

## Checkpoint 与训练开关

`--check` 元数据仍是 `weights=null`、`trainable=false`。训练目录另存 `model.zip` 与带 `num_timesteps` / `n_updates` 的 `checkpoint.json`，加载前校验 catalog/encoder/规则版本。网站不读取这些权重。catalog 哈希不替代引擎规则版本。

```bash
python3 scripts/train_duel_v2.py --help
python3 scripts/train_duel_v2.py --check
python3 scripts/train_duel_v2.py --check-config /path/to/checkpoint-metadata.json
python3 scripts/train_duel_v2.py --train --output artifacts/duel-v2-rl-run
python3 scripts/train_duel_v2.py --train --output artifacts/duel-v2-rl-adapt \
  --resume artifacts/duel-v2-rl-run --swap N03:N05 --freeze-generic --kl-coef 0.1
python3 scripts/rl_monitor.py
```

训练状态页是独立服务，默认 `http://127.0.0.1:5002/`，不挂大厅导航。只读 `artifacts/` 下带 `status.json` 或 `train.log` 的目录。Windows 上多环境采样用同进程线程，避免 spawn 管道在大观察上断开；每 10 次更新写入 `model.zip`。超过几分钟的训练用 `python scripts/rl_train_detached.py -- --train --output DIR ...` 拉起，避免被外壳作业对象杀掉。

`--check` 不创建对局或模型。`--train` 需要新的 `--output` 目录；总预算默认 10000 原始引擎动作 / 128 开局 / 1800 秒，训练先为评估预留额度。`n_updates=0`、随机自对弈或仅 `predict` 成功都不算训练完成。训练和评估不自动写网页录像；要看某次评估对局时，对输出目录执行 `--export-replays DIR`，按 `eval_games.jsonl` 里的种子和动作重放成公开 `replays/*.json`。在 `/replays` 导入该 JSON 后，用 `/table?replay=<房间码>` 播放；不含隐藏手牌、牌序或种子。

默认训练牌组是当前 catalog 的官方创生预组；学习方和冻结规则 AI 对手都用这一套。编码器座位与卡表跟随创生预组四人（含伊洛伊的 Y 卡），浊燃等其他人的官方预组仍不能直接编码。换少数卡用 `--swap OLD:NEW`（必须同角色），或传入同四人座位的 `--deck` JSON。创生预组卡表变化会改变 `character_ids` / `card_ids` / catalog 哈希，旧权重不能加载。

换牌适应走顺序微调，不在一次训练里混合多套牌：

- `--resume` 加载上一套权重。
- `--freeze-generic` 冻住局势编码和通用动作分，只更新卡身份 embedding。
- 新卡 embedding 用「同角色且同卡类」的旧卡均值初始化，找不到再退到同卡类。
- `--kl-coef` 把上一套冻住策略当教师，只在出击/结束回合/终结/换牌，或两套都有的出牌上加 KL；环境里仍然只有当前这一套牌。

这保住的是基础操作，不保证上一套牌的胜率。旧 `CandidateScoringPolicy` 权重与 `generic_plus_card_v1` 不兼容。

## 已执行验证与限制

```bash
python3 -m unittest tests.test_duel_v2_rl tests.test_duel_v2_rl_train
```

框架测试覆盖编码、隔离、截断、`--check`、通用/卡身份特征拆分，以及编码器四人预组换卡。训练测试覆盖预算计数、对手窗口截断不为输、可选的 MaskablePPO 一次更新、以及冻住通用分后卡身份参数仍可训练。没有吞吐或平衡结论。网站加载模型仍属下一阶段。

参考：[MaskablePPO 官方说明](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)。
