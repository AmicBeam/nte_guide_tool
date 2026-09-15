# 公开卡池训练、按需分析与录像

## 范围与状态

本轮实现范围是 **6 名公开角色、48 张候选卡及衍生牌「家族壮大」**，不包含测试角色。

- 学习方固定使用创生或覆纹快攻；对手构筑覆盖公开角色与候补卡。
- 规则使用共享 C 程序生成 native/CUDA 后端，支持每行独立构筑和实际编队顺序。先手首回合抽牌、白热化 6/13/20 阶段进入编译规则。
- `resident_public_v2` 同时用于训练与 NumPy 推理，包含白热化、剩余终结次数、动态资源上限、明牌信息与候补卡状态。候选动作额外区分同名牌中已明牌的那一张。
- 历史对手池有固定容量，使用冻结模型快照及规则策略。快照仅在 rollout 之间替换，单个 rollout 内不更新对手参数。
- 训练只写指标、状态和 checkpoint；不会自动保存逐操作日志、生成报告或导出录像。
- **本轮没有正式训练，也没有连接 Windows。** 本机验证 shared native 规则、观察、权重导出与网页录像；真实 CUDA 执行必须在 Windows 上通过预检。新的训练入口会自动执行该预检，通过后才创建优化器。

旧张量实验入口、SB3 入口和 `train_duel_v2_session.py` 保留用于既有实验；本页的完整流程使用下列新入口。随网站保存的两套旧模型仍为镜像能力，本轮不把它们标成已完成公开卡池训练。

## 1. 夜间开始前检查

在已有 PyTorch CUDA 环境、项目根目录运行：

```bash
python scripts/train_duel_v2_public.py --check
python scripts/verify_duel_v2_public.py --backend cuda --cases 32
```

第一条只读取 schema 与规则身份；第二条运行有界规则/编码对拍和少量冻结模型行动，不创建优化器。没有 CUDA 时明确失败，不退回 CPU 长训。

## 2. 有界训练

两套预组建议依次运行，避免同时占用 GPU 显存。每次选择新的输出目录：

```bash
python scripts/train_duel_v2_public.py --train --deck starter  --n 4096 --horizon 16 --seconds 7200  --output artifacts/public-starter-run --detach
```

另一套将 `--deck` 改成 `weave-rush`，并更换输出目录。Windows 后台进程使用 breakaway 标志脱离 SSH Job Object；启动返回只表示进程已创建，实际状态看输出目录的 `status.json`、`console.log` 和 `preflight.json`。

- `--warm-start 旧版best.pt`：迁移匹配预组的 `resident_public_v1` 权重，保留已有输入列，新输入列为零；优化器和历史池重新开始。需要使用与旧模型相同的 `--hidden`。
- `--resume 新版latest.pt`：恢复相同 schema、预组和规则身份的模型、优化器与历史池，重新开始采样对局。与 `--warm-start` 互斥。
- `--history` 控制历史池容量；`--snapshot-updates` 控制快照更新间隔。
- `--max-updates` 可用于有界验证，默认为 0，正式运行由时间限制停止。

`metrics.jsonl` 保存损失、熵、KL、完成局数等标量；`latest.pt` 保存模型、优化器、历史池和 RNG 状态。停止时保存 checkpoint，但不自动分析。训练完成不等于胜率达标。

## 3. 指定模型，额外跑分析

先将选定的新 checkpoint 导出为仅含数字的 NumPy 模型：

```bash
python scripts/export_duel_v2_public.py  --checkpoint artifacts/public-starter-run/latest.pt  --output artifacts/public-starter-export
```

导出会验证 PyTorch 与 NumPy 分数及选动作一致性。导出结果先标为待评估，可用于离线分析，不能直接作为网页高级人机。

需要查看时才额外跑一轮：

```bash
python scripts/analyze_duel_v2_policy.py  --model artifacts/public-starter-export/starter.npz --deck starter  --pairs 32 --max-actions 400 --max-seconds 600  --output artifacts/public-starter-report
```

- 每对对局使用相同构筑和种子交换先后手；公开对手编队按全部 15 种四人组合分层轮换，卡牌按种子采样。
- `--opponent-deck` 可指定公开构筑 JSON；`--opponent-model` 可指定第二份数值模型。默认使用公开规则策略作为独立对手。
- 默认生成随机种子并只保存在原始记录中；`--seed` 可显式指定可复现种子。
- `report.md` / `report.json` 包含模型身份、先后手结果、对手编队、双方实际出牌（含响应、衍生牌）、使用与胜负关联，以及完整对局索引。异常计入未完成对局，不能当成胜局。
- `raw/` 保存原始种子、构筑和完整操作；`public/` 保存可分享录像。分析不更新模型参数。

固定预组只会使用自己携带的卡；未使用某张牌不代表该牌弱。构筑出现、实际出牌和卡效时机覆盖是不同指标，报告不据此宣称卡牌已平衡。

## 4. 从报告选一局录像

复制报告中的 `game_id`：

```bash
python scripts/analyze_duel_v2_policy.py  --source artifacts/public-starter-report --export-game starter-00-first  --replay-output artifacts/selected-game.json
```

这是读取已记录的公开事件，不加载模型、不重新模拟，因此不会因重新采样或规则变化而变成另一局。随后在网站 `/replays` 点击「导入训练录像」，选择 JSON 并播放。

公开导出隐藏双方未公开手牌、私有抽牌和检视候选；已公开打出的牌、明牌与公开衍生牌仍可展示。不包含 RNG、种子或隐藏牌库。原有参与者视角录像不自动冒充这种公开文件；旧分析文件若缺少公开视角标记，需要重新生成分析记录。

## 5. 网页模型验收

需要接入网站时，再使用匹配报告验证导出文件：

```bash
python scripts/export_duel_v2_public.py  --model artifacts/public-starter-export/starter.npz  --approve-report artifacts/public-starter-report/report.json
```

默认要求：模型已有训练更新；报告匹配同一权重 SHA、schema、规则和预组；至少 32 对完整对局，覆盖全部 15 种公开对手编队，无异常和截断；经验胜率至少 50%。这是发布门槛，不是对真实胜率下界或平衡的证明。

通过后只更新该导出目录的能力声明。不会自动覆盖网站模型、重启服务器或部署；按既有模型发布流程替换固定文件。旧模型的训练身份保留，新规则的兼容记录不改写旧训练成绩。

## 验证边界

`tests.test_duel_v2_public_kernel_matrix` 逐张执行 48 张候选卡、选择续步、武备后续效果和白热化边界，并比较正式 Python 引擎与 native 编译结果。相同最低能量的随机分支验证“仅一名最低者获得能量”和编译后端可复现性；Python 与编译后端采用不同 RNG，不保证相同数值种子选中同一人。分析录像直接保存实际公开事件，不依赖跨后端重放种子。

`verify_duel_v2_public.py` 检查 NumPy 与张量观察/候选对齐，以及目标后端与 native 的逐行结果；附带少量公开构筑、冻结对手和设备内 reset 检查。它不代替正式训练强度评估。

### 本机验收记录（2026-09-15）

- 合并回归 171 项：159 通过、12 跳过，无失败。覆盖公开卡池矩阵、compiled/resident/IR、观察与导出、离线报告、模型能力校验、API、回放、白热化和基础 RL 契约。跳过项包含本机没有 CUDA、未安装 Gymnasium/SB3 的检查及旧抽样用例。
- 本机 native 预检完成 32 个局面；逐卡矩阵同时核对 GPU 布局上的张量观察与正式公开投影的 NumPy 观察。
- 隔离数据库额外跑 2 场规则策略对局，选出一场 28 回合、602 条原始事件的录像。通过真实页面导入，验证公开起手牌背、衍生牌展示、播放/暂停、逐步推进、刷新后重新加载，无浏览器错误。刷新会从头播放，不声称保留播放位置。
- 本轮未执行 Windows CUDA 实机测试、正式训练或模型部署。现有两份旧权重未重训，仅更新编译布局兼容记录，能力仍为镜像预组。
