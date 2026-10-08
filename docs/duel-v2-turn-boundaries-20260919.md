# 2026-09-19 回合边界修正

用户明确要求两项规则纠正：玩家护盾在己方回合开始清除；家族壮大从整局第 5 回合起具有瞬发。属于用户确定的桌游规则，不是原作数值或平衡结论。

- 玩家护盾在回合提示之后、任何回合开始效果之前清零。后手开局 5 盾仍能抵挡先手首回合伤害，后手首回合开始清除剩余值；对方回合开始不清除本方护盾。
- NF01 使用总回合 `turn >= 5`，不再使用每方 `turn_count >= 5`。先手第三次行动（总第 5 回合）与后手第三次行动（总第 6 回合）均可享受瞬发；仍然只有每个己方回合首次瞬发免费。
- 正式 Python 引擎、张量训练源码、native/CUDA 共用生成源码、卡表与图鉴词条同步。历史冻结训练不修改，不重跑训练。

## 模型兼容

延续此前用户明确启用的三套中午候选，数值权重 SHA 和构筑保持原值。为应用这两项规则修正，清单保留原 `training_rule_hash` 与历史 `validation`，更新执行规则绑定并在 `serving_authorization.runtime_amendment` 记录本次请求。仍然 `validation.approved=false`；本次只验证规则边界和模型运行兼容，不证明策略强度或质量门槛通过。

三套真实权重在临时数据库中完成 API 开局、完整行动、终局与公开回放保存，公开个人构筑接入也通过。旧训练胜率不作为新规则证据。

## 验证

专项运行 25 项，24 通过，1 项 Torch 梯度验证因当前环境缺依赖跳过：

```sh
.venv/bin/python -m unittest tests.test_duel_v2_opening_draw tests.test_duel_v2_turn_boundaries tests.test_duel_v2_catalog tests.test_duel_v2_engine.DuelV2EngineTest.test_nanali_family_becomes_instant_from_global_turn_five tests.test_duel_v2_league tests.test_duel_v2_advanced_ai.AdvancedRoomTest.test_real_models_finish_all_presets_and_save_replays tests.test_duel_v2_advanced_ai.AdvancedRoomTest.test_published_models_support_public_saved_builds
```

覆盖玩家护盾先吸收伤害、残盾到期、仅清除当前方、先于浊燃伤害清除，以及 NF01 第 4/5/6 回合的合法性、公开瞬发标记与首次免费后仍需付费。生成内核使用隔离夹具验证玩家盾与瞬发边界；仅在该单测中隔离既有伊洛伊内容/IR 完整性缺口，不放宽生产编译入口。

张量回归补充批次掩码、双方局部清盾及 NF01 合法性/支付测试；本机无 Torch，未执行 CUDA 验证。

扩大引擎测试发现 4 个既有失败测试（含子用例）：伊洛伊旧倒地反伤及其白藏/薄荷连锁断言。以 HEAD 修改前 flow 在内存中重跑，同样失败；未把本次专项通过表述为全库通过。完整编译后端另有既有 `iloy.on_ally_down content handler missing` 门槛失败。

本地服务已重启并监听 `0.0.0.0:5001`；`127.0.0.1`、当前 Wi-Fi `192.168.10.5` 及 `10.4.13.217` 的 `/card-game` 均返回 200。三套服务模型严格加载通过，仍明确标记质量未批准。
