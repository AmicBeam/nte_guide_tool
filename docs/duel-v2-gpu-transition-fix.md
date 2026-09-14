# 张量训练终局与屏蔽修复（2026-09-14）

本次只修复训练转移契约，不改变卡牌或网页规则；并未实现完整 GPU 规则结算。

- `step_index` 根据原始动作索引检查合法性；负数是批量 no-op，越界索引也不执行。用于安全读取候选的 clamp 不改变合法性判断。
- `GpuDuelEnv.step_learner` 推进到学习方决策或终局后返回，不自动重开。调用方先读取终局、奖励与最终状态，再显式 `reset_finished`。
- `collect_rollout` 记录当前局奖励、done 和 winner 的独立副本后重置。GAE 在 done 处切断，不使用新一局价值为上一局终局 bootstrap。普通采样段末尾仍使用下一决策价值。
- 采样阶段关闭梯度；非学习方窗口、终局未重置或空合法动作不伪装成第 0 个合法动作，不写入虚假样本。
- PPO 检查 loss/梯度有限；训练报告增加本批终局胜负平计数、实际优化器步数与权重变化量。`done_frac` 从 rollout 统计，不再读取已重开的环境。

本次修复前产生的权重和 reward_mean 保留为历史产物，不能因训练完成即证明学会正确策略。短训从新模型开始。

验证命令：

```bash
python -m unittest tests.test_duel_v2_gpu_transitions tests.test_duel_v2_gpu_duel tests.test_duel_v2_rule_ir -v
python scripts/train_duel_v2_gpu.py --output artifacts/duel-v2-transition-fix-smoke-20260914 --n 32 --horizon 32 --updates 2 --minibatch 128 --epochs 2 --compact --skip-eval
```

短训共 2048 个学习方步，只验收转移边界和参数更新，不证明策略强度、平衡或完整引擎一致性。规则张量仍在 CPU，网络在 CUDA；compact 权重不等同于网站 encoder 权重。真实运行结果另记产物报告。

## Windows 实测结果

2026-09-14，RTX 5060 Ti：新增 4 项转移回归、现有 GPU 环境与共享 IR 19 项测试全部通过（共 23 项，包含 CUDA 测试）。Mac 默认 Python 未安装 torch，新测试在 Mac 跳过，未将其算作通过。

短训产物位于 Windows `D:/git/nte_board_game/artifacts/duel-v2-transition-fix-smoke-20260914`，没有加载旧权重。进程退出码 0，含启动共 147.20 秒。

| 更新 | 学习方步 | 本批终局 | 胜 / 负 / 平 | 优化器步数 | 权重变化 L1 | 最后 loss |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 1024 | 6 | 2 / 4 / 0 | 16 | 257.93 | 2.2532 |
| 2 | 1024 | 25 | 4 / 21 / 0 | 16 | 207.74 | 4.5800 |

合计 2048 个学习方步、32 次优化器更新、31 次终局。显存峰值 236.8 MiB。上述终局为训练过程统计，不是独立评估胜率；小并行数短测也不代表 1024 局吞吐。损失与梯度检查均通过。

新权重为产物目录内 `gpu_policy.pt`；原始日志 `console.log`、训练指标 `gpu_train.json`、进程结果 `run_status.json` 和加载验证 `verification.json` 均留在 Windows，不加入 Git。
