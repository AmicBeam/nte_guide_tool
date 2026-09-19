# 2026-09-19 本地高级人机模型更新

用户在收到中午验收结论后明确要求“更新项目的模型吧”。因此本次启用第 28 轮最新候选 `latest`，不是仍与初始一致的阶段最佳。三套均为 official_public_v5、纯网络 CPU 推理，支持公开八人构筑；没有启用网页 256 次搜索，没有继续训练或恢复定时任务。

| 策略 | 数值权重 SHA256 |
| --- | --- |
| 创生 starter | `7207c7de951f1dbbdee0ed68bb1898b0724210d736b3f42373411c77b1f3fc5c` |
| 覆纹 weave-rush | `a65fc02c9bc734a9fb23151abfff0499afc6b8defe61aba5e275e46b06c0c8d5` |
| 快攻 quick-rush | `d7627b43f9d5932c87dd11008e560f584ac5daee092eb6dd2a7ab2017f9fa88f` |

来源：noon-20260919-r4，第 28 轮，336 局完整学习、414 次优化。交付压缩包 SHA256：`f535234d81be39205e2a355ef798d4e3fd84f01dd5f3106ebf2b85644dad0a08`；规则身份：`42e8840fe013df1d8b9e2b38a9e538e2fbcab0c674ca13979967c911ece9f085`。安装前按当前规则与构筑独立加载验证，没有修改模型的规则哈希。

## 授权与质量的区别

[中午验收报告](duel-v2-noon-results-20260919.md) 保持原始结论：纯网络胜局 24→22/48，搜索胜局 36→32/48，整体未提升。原第 3 回合最新纯网络选择 N02，动作偏好约 90.17%，该值不是胜率。用户此次启用不意味着质量门槛通过。

模型清单保留 `validation.approved=false`、`quality_gate_passed=false`；`complete=true` 仅表示该次计算验收完整结束。单独的 `serving_authorization` 记录用户原话、日期、来源、变体及精确模型/构筑/规则哈希。加载器只允许原本验收通过的权重，或具有完整且身份一致的明确候选授权的权重用于服务。未授权导出仍拒绝；规则、特征、有限数值、张量形状与公开角色范围检查保持不变。

## 本地接入与回退

服务模型位于 `app/modules/card_game/engine/ai/models/`，新增快攻 json/npz，替换创生和覆纹。它们是明确选择的服务数值资产；原始训练断点、回放、优化器和批量产物不提交。大厅快攻选项解除禁用。更新模型版本后，旧高级房间按原有保护要求重新开局，不混用两套权重。

原服务双模型备份到本地 `artifacts/model-backups/pre-noon-20260919/`（不提交）。需要回退本次接入时可回退本次提交，或成套恢复对应版本的代码与清单；旧六人模型是否能用于当前规则必须重新校验，不自动声称可用。实验初始/最佳权重仍保存在独立验收产物中。

本次仅更新本地项目与本地服务，不推送或部署远程生产。

## 验证

- 网站隔离数据库测试：18 项，17 通过、1 项 Torch 梯度测试按依赖跳过；三套真实模型均完成 API 开局、行动、终局及公开回放保存。
- 隔离 Torch 环境重新运行 `tests.test_duel_v2_league`：8 项全通过，补齐上述跳过项。
- 未授权模型拒绝、授权内容缺失/模型/构筑/规则绑定失效拒绝、质量批准状态不被覆盖、当前规则不匹配拒绝均通过。
- 扩大运行旧高级人机与 serving_build 测试时，8 项旧六人夹具报 `encoder is not ready for public asymmetric serving`。直接调用未修改的 `capability_ready_manifest` 可复现；未改变旧六人能力判断来放宽其范围。两项依赖旧默认双模型目录/加载器的断言已按新版三策略更新并通过。不将本次专项验证称为全库通过。
- 本地常驻服务重启为 PID 70079，监听 `0.0.0.0:5001`。`127.0.0.1:5001/card-game`、`192.168.10.5:5001/card-game`（当前 Wi-Fi）与 `10.4.13.217:5001/card-game` 均返回 200。
- 真实浏览器大厅切换高级人机、选择快攻成功，显示「对手：快攻预组」，没有控制台错误。保留既有进行中房间，完整对局在临时数据库中验证。
- `git diff --check` 通过。

专项命令：

```sh
python -m unittest tests.test_duel_v2_league tests.test_duel_v2_league_service tests.test_duel_v2_advanced_ai.CpuEncoderTest tests.test_duel_v2_advanced_ai.AdvancedRoomTest.test_real_models_finish_all_presets_and_save_replays tests.test_duel_v2_advanced_ai.AdvancedRoomTest.test_published_models_support_public_saved_builds
```

