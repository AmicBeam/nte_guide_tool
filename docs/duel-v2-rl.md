# 四人轮替 V2 强化学习框架

> **训练前先读：**[训练手册的当前链路与报告落点](duel-v2-public-training.md#当前训练链路与报告落点)及[完整训练、测试与报告要求](duel-v2-full-training-report-spec.md)。本页镜像、SB3、六人编译和早期PPO说明是历史技术背景，不覆盖已确认的信息集搜索学习链路；单轮报告和原始产物统一放在 `artifacts/rl-evals/`，不提交Git。

> 当前网站已接入两套冻结模型，见文末 2026-09-15 章节。下方 2026-09-12 的单预组训练说明保留其适用范围；不能把旧镜像训练成绩当作公开角色自由构筑的验证结果。

状态：2026-09-12，可运行短程训练与固定种子策略评估。游戏规则以 [V2.4 手册](everness-item-chain-card-design.md) 为准，接口以 [实现契约](duel-v2-implementation-contract.md) 为准。小样本结果只用于发现问题，不是平衡结论。

## 文件与边界

离线批量搜索基础位于 `rl/batched_duel` 和 `rl/batched_search`，提供 CPU arena、无损实体图 codec、设备存储与随机原语、每根预算、批量推理、协作 Gumbel 遍历和设备统计池，验证及后续五队 native/CUDA 对拍见[批量搜索后端](duel-v2-batched-search.md)。完整正式规则通过有界 CPU 驻留进程池执行，离线评测入口 `scripts/evaluate_duel_v2_batched.py` 合并跨局网络推理和 GPU 搜索统计，并保存实际动作、原始记录及公开回放。`wall_clock` 从每根入队起严格截断 3 秒，共享最多 32 次证明／树模拟预算并记录实际完成次数；迟到结果丢弃。五队规则、合法动作、观察编码和采样的设备执行尚未接通，不自动接入服务或训练。合成根或重复推理行不算真实对局容量验收。

独立恢复研究组件：`residual_policy`／`residual_runtime` 管理正常幅度残差网络与数值版本；`preserved_policy` 保留三套旧策略，`recovery_runtime` 管理显式研究采集，`recovery_search` 管理自然WDL尺度对照，`recovery_checkpoint` 检查网络／Adam／RNG恢复；`policy_equivalence`、`search_stability`、`archive_diagnostics` 只做表示／目标／数值审计，`full_hand_diagnostic` 只对合成离线状态提供具名手牌权限，`offline_sources` 生成排除敏感文件的源码快照。它们不自动接入网站或正式长训；具体入口及CPU／CUDA验收见[策略学习恢复](duel-v2-policy-recovery.md)。

- `app/modules/card_game/rl/backend.py`：仅对接 `engine.duel_v2` 六接口中的开局、行动方、投影、动作和结果；不依赖旧盖卡模型、HTTP 或数据库服务。
- `contracts.py`、`adapter.py`：可注入的纯内存接口与决策适配。策略仅收到观察投影和合法动作，不收到原始 state。
- `encoding.py`：固定尺寸公开局势和候选动作编码。观察与单候选维度随 catalog 变化，当前约 3416 / 279（创生+覆纹快攻座位），动作容量 256；容量溢出显式报错，不删除候选。
- `policies.py`：`VisibleEngineRulePolicy` 用公开 `legal_actions` 复现引擎规则 AI；`RandomMaskedPolicy` 是未训练基线；`MaskablePolicy` 对已加载模型做 mask 推理。
- `gym_compat.py`：可选 Gymnasium Env 工厂；只有调用 `make_gym_env` 才导入 Gymnasium 和 NumPy。
- `checkpoint.py`：`--check` 仍只读写无权重元数据；训练产物另存 `model.zip` + `checkpoint.json`（累计步数、更新次数、版本）。
- `networks.py`：`CandidateScoringPolicy` 在 GPU 上对每个候选打分，不把 256×199 展平进 MLP。打分拆成通用分（动作类型、卡类、费用、目标座位）和卡身份分（catalog ID embedding）。
- `transfer.py`：单套牌解析、同角色换卡、新卡 embedding 从同类旧卡初始化；换牌时冻住通用分，可选对共享动作做 KL。一次训练只使用一套牌（创生镜像或覆纹快攻镜像），不混合多构筑。
- `rule_ir/`：共享规则 IR。战斗切片是攻击/护盾/同时伤害/倒地/胜负，同一程序生成 CPU 张量内核和 NVRTC CUDA。创生预组卡效、被动挂钩和衍生牌是第二份 IR（`starter.py`），编译成整数表再由 `gpu_duel` 张量核执行，不再按卡 ID 往规则里加长期分支。生成 CUDA 与手写实验核、Python `combat.attack` 对拍；4096 行吞吐对照要求不低于手写核约 80%。手写核只作对照，不是业务真源。网站 Python 引擎仍是 oracle。
- `gpu_duel/`：定长对局环境，座位与网站编码器相同（创生+覆纹快攻并集）。`--deck starter|weave-rush` 镜像，`--matchup cross` 为创生 vs 覆纹快攻。开局可 Python 装箱或 GPU 内洗牌重置；学习方动作后由规则对手推进到下一次学习方决策。默认观察走网站 `V2Encoder` 布局。`scripts/train_duel_v2_gpu.py --n 1024 --horizon 32` 同步 PPO（规则张量默认在 CPU 上批量执行，网络在 CUDA）。`--resume gpu_policy.pt` 可接上一阶段。`--compact` 才用短观察。
- `scripts/train_duel_v2.py`：`--check` 不创建对局；`--train --output DIR` 默认按 CPU 核数开进程采样（上限 32），GPU 候选打分。学习方（训练和评估）不提供认输动作，网站玩家仍可认输。评估对手仍是公开规则 AI。

核心接口依赖现有网站环境即可导入；NumPy/Gymnasium/SB3 仅列在独立 `requirements-rl.txt`。不要在网站启动链路中创建训练器或环境。导入 `app.modules.card_game.rl` 不创建对局、不调用 reset/step/predict/learn。

## 观察与动作

引擎的 `observe(state, side)` 是唯一信息边界。己方手牌、双方角色/资源/弃牌/记录可见；对手手牌通常隐藏，公开复制保留。九原检视对手手牌及其他检视效果，仅在引擎开放的 pending choice 中编码候选。

角色座位是创生预组与覆纹快攻的并集（`nanali/iloy/zero/jiuyuan/bohe/baicang`），卡牌按这些角色在 catalog 中的顺序编码；元数据校验这些顺序。每方对局仍是 4 人，缺席座位填空。使用 `shape_id`；保留 last_front、各方 turn_count、疲劳、已用次数（含瞬发/同频及角色 once 键）、是否曾环合、花伤加成、下次攻击加成、返回标记和负面状态。未知 `used` 键会报错，需升级编码器，不能静默丢掉。复制按公开归属座位编码，并保留原作者；复制期限和负面状态期限以对应操作者回合计数计算剩余时间。记录不要求实体卡实例 ID，结算中卡牌单独编码。

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
python3 scripts/train_duel_v2.py --train --output artifacts/duel-v2-rl-rush --deck weave-rush
python3 scripts/train_duel_v2.py --train --output artifacts/duel-v2-rl-adapt \
  --resume artifacts/duel-v2-rl-run --swap N03:N05 --freeze-generic --kl-coef 0.1
python3 scripts/train_duel_v2_gpu.py --output artifacts/duel-v2-gpu-genesis --n 1024 --horizon 32 --updates 8
python3 scripts/train_duel_v2_gpu.py --output artifacts/duel-v2-gpu-rush --deck weave-rush --n 1024 --horizon 32 --updates 8 --resume artifacts/duel-v2-gpu-genesis/gpu_policy.pt
python3 scripts/train_duel_v2_gpu.py --output artifacts/duel-v2-gpu-cross --matchup cross --n 1024 --horizon 32 --updates 8 --resume artifacts/duel-v2-gpu-rush/gpu_policy.pt
python3 scripts/compare_duel_v2_trainers.py --output artifacts/duel-v2-trainer-compare
python3 scripts/rl_monitor.py
```

训练状态页是独立服务，默认 `http://127.0.0.1:5002/`，不挂大厅导航。只读 `artifacts/` 下带 `status.json` 或 `train.log` 的目录。多环境采样默认按 CPU 核数开进程（上限 32，`--n-envs` 最大 128）；观察矩阵走共享内存，避免 Windows 管道被大观察撑断。牌局仍在 CPU 进程里结算，GPU 只做候选打分。每 10 次更新写入 `model.zip`。超过几分钟的训练用 `python scripts/rl_train_detached.py -- --train --output DIR ...` 拉起，避免被外壳作业对象杀掉。

`--check` 不创建对局或模型。`--train` 需要新的 `--output` 目录；总预算默认 10000 原始引擎动作 / 128 开局 / 1800 秒，训练先为评估预留额度。`n_updates=0`、随机自对弈或仅 `predict` 成功都不算训练完成。训练和评估不自动写网页录像；要看某次评估对局时，对输出目录执行 `--export-replays DIR`，按 `eval_games.jsonl` 里的种子和动作重放成公开 `replays/*.json`。在 `/replays` 导入该 JSON 后，用 `/table?replay=<房间码>` 播放；不含隐藏手牌、牌序或种子。

默认训练牌组是当前 catalog 的官方创生预组；学习方和冻结规则 AI 对手都用这一套。`--deck weave-rush` 改为覆纹快攻镜像。浊燃、盈蓄、覆纹预组仍不能训练。创生 vs 快攻走 GPU 张量环境 `--matchup cross`。换少数卡用 `--swap OLD:NEW`（必须同角色且都在编码器卡表内）。卡表或座位变化会改变 `character_ids` / `card_ids` / catalog 哈希，旧权重不能加载。

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

## 2026-09-15：常驻 CUDA 双预组训练

`python scripts/train_duel_v2_session.py --output DIR --n 4096 --horizon 16 --seconds 10800 --deadline 2026-09-15T09:00:00+08:00 --detach` 是本次有界训练入口。日期为本次用户指定窗口，未来运行必须显式设置新截止时间；已过期时拒绝启动。Windows 子进程使用 breakaway 标志离开 SSH Job Object。`--resume-root DIR` 分别加载各预组的 best/latest 模型和优化器，校验观察、卡池及规则源码版本；两种预组不共享权重。

- `compiled_backend='cuda'` 使用连续 int32 状态，GpuState 各字段是同一 CUDA 存储的视图；共享 C 源同时生成 native/CUDA 内核。step/reset 不逐局 pack/unpack，也不落回 Python/旧张量规则。仅允许精确的创生、覆纹快攻镜像构筑；cross、自选卡和其他预组仍未获此路径验收。
- `resident_obs.py` 以张量运算编码双方公开角色、个人资源/标记、玩家资源、自己完整手牌和公开弃牌计数；不编码对手暗牌、牌库顺序和 RNG。schema 为 `resident_public_v1`，没有历史记忆，不兼容网站 V2Encoder 权重。不要把默认 `encoder_observe()` 的逐局 JSON 路径当作高并发入口。
- 学习方座位与先手独立分配；两模型轮流更新，对手是固定批量规则策略。独立验收使用 Python 正式引擎和 `VisibleEngineRulePolicy`，不能将批量训练对手的终局胜率当验收胜率。
- 默认每 300 秒对每模型做 64 对种子配对（128 局），同一种子互换先后手。每套牌独立要求先后手等权总体胜率 ≥50%，平局不算胜，分别报告先手/后手和 Wilson 95% 区间；任何截断或不完整配对均不能达标。
- 默认平台期为连续 6 次完整评估没有超过 0.01 的胜率改善，且至少 1800 秒没有显著改善。只有最佳完整评估已达到 50% 才能平台期停止。两套均达标并平台期、整体 3 小时或绝对截止时间取最早；时间到但未达标如实记录。有限样本的经验胜率门槛不等于证明真实胜率下界 ≥50%。
- `metrics.jsonl` 保存 reward、loss、value loss、KL、熵、clip fraction、explained variance、优化器步数和吞吐；训练期间不收集出牌分析日志。每 10 次更新及退出时保存 latest；独立评估改善时保存 best。
- 优化结束后各自加载最佳有效模型，另跑有限日志评估：默认 128 对/256 局、每套最多 600 秒，不更新参数。此额外评估可在训练截止后完成。`final-play-log.jsonl` 保存元数据、game_start、successful_play、game_end，按 game_id 连接卡牌、出牌方、先后手、种子、模型版本及结果；同方同局同卡可去重，未出现的卡无样本。只存原始数据，不自动输出每卡胜率分析。
- `run_status.json` 的 phase、stop_reason、exit_code 与 `final-eval.json` 用于确认退出结果；训练完成与胜率达标分别记录。日志文件完整性在短训验收。

验证：`python scripts/verify_duel_v2_compiled.py --backend cuda --deck starter --trajectories 48`、对应 `--deck weave-rush --trajectories 96`，以及 `python -m unittest tests.test_duel_v2_compiled tests.test_duel_v2_resident tests.test_duel_v2_gpu_transitions`。完整结果与性能见 [常驻编译引擎验收](../artifacts/rl-evals/reports/duel-v2-compiled-migration.md)。


## 2026-09-15：冻结模型接入高级人机

训练成功后，通过 `scripts/export_duel_v2_ai.py --root TRAIN_RUN --output EXPORT_DIR` 导出两个最佳模型；该脚本需要训练环境的 torch/NumPy，但不更新参数。服务端只读取 `.npz` 和 JSON 版本清单，依赖 `requirements-ai.txt`，不需要 torch、CUDA 或本地 C 编译器。

大厅 `advanced` 模式提供两套精确镜像预组。模型只给合法动作打分，实际出牌和胜负继续通过正式 Python 规则接口。训练用的 `resident_public_v1` 与 CPU 编码使用同一字段列表，并有导出数值/动作对照验证；对手暗牌和牌库顺序不进入输入。训练 checkpoint 与最终出牌分析日志仍作为训练产物保存，网页不加载这些日志。

本次 CPU 推理导出文件约 345 / 344 KB，按用户要求，当前用于人机的四个文件已放入 `app/modules/card_game/engine/ai/models/` 并随 app 部署；该目录只保留两套当前策略及版本校验文件。后续先向临时产物目录导出和验证，再仅覆盖这四个文件，不把原始 checkpoint、优化器、对照样本或日志放进 app。完整训练产物另已从 Windows 备份至 `artifacts/windows-training-backup-20260915/paired-long-20260915`，51 个文件逐个 SHA-256 校验通过，包含 best/latest、优化器、评估结果及最终出牌日志；未执行每卡胜率分析。

## 2026-09-15 第一阶段：固定 AI 预组与公开玩家构筑

以下记录第一阶段的能力与当时缺口；后续完整共享编译规则、观察、导出、报告和录像接口已进入上方独立指南。旧权重没有因此自动获得自由构筑资格。

用户确认的下一阶段范围：AI 固定使用创生或覆纹快攻，玩家可以使用六名公开角色内的合法自定义构筑。测试账号不扩大此模式的角色范围。训练学习方固定预组，对手构筑需要覆盖不同角色组合及候补卡；不要求 AI 学习自行组牌。

### 覆盖边界

- 通用 `V2Encoder` 的卡表包含六名公开角色、48 张候选卡和衍生牌 `NF01`。卡牌身份可编码不等于每种状态都已验证，也不等于模型已经学习对应策略。
- 高速规则 IR 当前缺少 `B03/B04/B05`。现有常驻 CUDA 的验收范围仍是两个精确镜像预组；不能把自定义构筑直接交给该路径，或静默忽略缺失卡效。
- 公开构筑框架优先通过正式 Python 引擎结算，并保留可替换的对手策略接口。训练与网页模型导出使用各自的观察和网络契约，不能直接互换不同 schema 的权重。
- 当前随 app 保存的两套权重未经过公开自由构筑训练。本轮不修改它们的能力声明，不启动正式长训，也不报告自由构筑胜率。

### 本轮框架接口

- `rl/capability.py` 校验公开构筑、按种子生成玩家构筑，并定义模型能力声明。`training_matchup` 保持学习方的预组不变，变化的是对手构筑；默认未指定对手时仍使用镜像。
- `train_duel_v2.py --sample-public --deck starter`（或 `weave-rush`）按局采样公开对手构筑；`--opponent-deck` 可指定独立的固定对手构筑，与 `--sample-public` 互斥。学习方网络仍使用现有 GPU 候选打分器，规则由 Python 进程执行。未启动训练时可用 `--check` 检查原有元数据。
- `evaluate_strategies` 支持独立 `opponent` 策略和 `opponent_deck`；`sample_opponent=True` 变化对手构筑。评估用同一种子与构筑互换学习方座位，固定 a 先手，保证先后手配对；日志保存双方构筑和策略类型。公开回放按实际双方构筑恢复，不能把未知模型对手默认为规则 AI 重放。
- 高级人机开局参数 `ai_deck` 选择 AI 预组，`ai_player_deck` 为 `mirror`（默认，兼容旧入口）或 `saved`。`saved` 从账号读取当前构筑，客户端传入的 `deck` 不覆盖它。创建房间前校验公开范围及模型能力，不修改保存的构筑。
- **网页导出尚未支持公开自定义构筑模型。** 当前 `resident_public_v1` 编码与导出器只接受镜像能力；即使给旧权重补上自由构筑声明也会被拒绝。Python 通用训练器的权重使用另一种网络与编码，不能直接交给现有网页导出器。后续需补齐观察状态及对应 CPU 导出，然后再用经过评估的新模型开放此能力。

本轮按用户要求只完成框架和本机验证；正式训练留待夜间 Windows GPU 环境，不自动启动或预约训练。可用的 Python 规则采样入口示例（本轮未执行）：

```bash
python scripts/train_duel_v2.py --train --sample-public --deck starter \
  --n-envs 8 --max-seconds 7200 --output artifacts/public-starter-run
```

这条命令在 CUDA 可用时使用 GPU 训练网络，规则采样仍在 CPU；不等于全流程常驻 CUDA，也不能续接现有 `best.pt`。现有常驻 CUDA 脚本继续只用于两套镜像预组。

验证：`python -m unittest tests.test_duel_v2_advanced_ai tests.test_duel_v2_rl tests.test_duel_v2_rl_train tests.test_duel_v2_public_training tests.test_duel_v2_api tests.test_duel_v2_replay`。本机 93 项中 83 项通过、10 项因可选 Gymnasium/SB3 依赖缺失跳过；包含真实 NumPy 模型对局。100 个固定采样种子覆盖 15 种四人组合与 48 张候选卡的构筑出现，**不代表全部卡效时机覆盖或训练强度达标**。

### 后续开放验收

1. 对合法公开角色组合和全部候选卡进行观察、动作、状态与对局完成检查；遇到不支持的状态应明确失败。
2. 学习方固定预组，对手构筑采样可复现；训练与评估分开记录构筑和种子，并覆盖先后手。截断、异常和未完成对局不能当作胜局。
3. 模型对手仅接收可见观察与合法动作；后续可加入冻结历史策略池，不让策略读取对方暗牌或牌库顺序。
4. 经独立评估并导出兼容模型后，才赋予对应的自由构筑能力。旧模型继续支持其已声明范围，历史镜像胜率不升级为新范围的成绩。

## 研究用类别动作头与终局价值

`outcome_runtime.py`研究入口支持`fixed_ten_wdl_v1`和`fixed_ten_grounded_wdl_v1`。后者由`grounded_candidates.py`把固定十人原始候选中的身份列转为类别，并拼接行动角色与目标的公开状态；原始观察接口、旧模型数值推理不变。模型清单绑定变换SHA，导出后需核验NumPy／Torch一致以及恢复后预测一致。它不自动获得服务资格。三组动作表示消融、镜像样本归属及真红验收以[训练操作手册](duel-v2-public-training.md#动作表示对照与真红行为验收)为准。


## 五套恢复编排与独立价值主干

现行五套入口 `scripts/train_duel_v2_current_round.py --runtime recovery` 使用正式Python规则、信息集Gumbel32和终局WDL。`recovery_policy.py` 为策略与价值提供独立可训练主干，`recovery_runtime.py` 对版本化数值加载、导出和价值预检做严格绑定；旧共享主干schema不被静默改写。

`recovery_round.py` 负责五套起点、每批全量策略行通行、非行动方价值行、三段独立闸门、能力保持／确认对照、两代恢复与原截止监督。`recovery_capacity.py` 仅在隔离副本检验真实训练教材容量和按成对种子留出，不把拟合成绩写作强度。`--resume --run` 仅恢复原输出与配置的有效轮次边界，原进程须退出、原学习截止未过；不重新授权时长。

纯网络与带搜索的冻结评估通过 `full_cycle_experiment.evaluate_job` 的显式recovery分支复用遥测、状态摘要重放和公开隐私过滤；恢复数值没有服务批准。启动、阶段、CPU／CUDA和完整报告边界以[训练手册](duel-v2-public-training.md)及[恢复协议](duel-v2-policy-recovery.md)为准。验证：`tests.test_duel_v2_recovery_round`，其中价值-only在已有Adam动量时也不得改变策略数组或动作分数。


### 单队恢复辅助与残虹限次观察

`recovery_round`的调度、批次更新和独立选择支持显式学习队子集，`recovery_capacity.fit(keys=...)`按同一子集检验完整搜索教材和独立留出。默认仍是五队；单队编排必须完成训练手册规定的同源码预检，不借此绕过正式启动保护。`murk_lineup`和`cross_lineup`新增双方残虹本回合持续伤害加攻已用状态；它来自公开信息，实体重新编号不改变编码。数值迁移需按具名列映射首层并将新增列置零。
