# 2026-09-18 小吱与海月接入验收

新增仅测试账号可见的小吱、海月及各 8 张牌。完整规则、卡名与用户确认边界见 [现行手册](everness-item-chain-card-design.md#2026-09-18-测试角色小吱与海月)。官方资料通过 Everness GraphQL 实时核对；桌游面板、资源和效果来自本次用户方案。本记录是功能与兼容验证，不是平衡结论。

## 功能验证

专项 `tests.test_duel_v2_xiaozhi_haiyue`：24 项通过，覆盖权限与图片、金谷负数/回合增长/伤害归属/倒地、终结借还与刷新、随机状态 JSON 恢复、战斗链及死亡中断、光抗性一次性消费、响应费用、洗牌抽牌、盈蓄门槛、武备出击、海月不入前排与预览、指定备战目标、瞬发条件重置、黯星和攻击成长、公开投影与回放字段。

相关回归共 199 项通过，1 项因环境条件跳过。命令：

```bash
.venv/bin/python -m unittest tests.test_duel_v2_xiaozhi_haiyue tests.test_duel_v2_catalog tests.test_duel_v2_engine tests.test_duel_v2_zone_status tests.test_duel_v2_api tests.test_duel_v2_replay tests.test_duel_v2_trigger_history tests.test_duel_v2_entities tests.test_duel_v2_effect_markers tests.test_duel_v2_tutorial tests.test_duel_v2_opening_draw tests.test_duel_v2_escalation tests.test_duel_v2_zero_refund tests.test_duel_v2_serving_build
```

`v2_api.js`、`v2_table/render.js` 通过 `node --check`。

## 浏览器验收

独立本地测试账号通过登录页登录、构筑页导入并保存合法四人 32 张构筑，再从大厅进入普通人机牌桌（验收房间 `86D1E3`）。实际 UI 完成换牌、小吱拖拽出击、结束回合、海月拖拽出击、零战斗牌出牌、查看历史、刷新恢复及打开公开回放。

- 小吱黄色金币图标在攻击上方，开局 0、己方回合开始变为 1；打中玩家开局护盾不增加金谷。
- 小吱倒地后金谷归零，环合与能量仍保留；刷新结果一致。
- 海月主动攻击后以剩余生命返回备战区，前排为空，正常获得主动战斗资源；历史明确记录返回。
- 公开回放重现小吱 0→1 金谷，沿用同一图标。普通账号图鉴仍仅显示六名公开角色。
- 终结、负数和长链的边界由正式引擎专项测试验证，本次 UI 对局未覆盖所有 16 张牌。

## 已有测试问题

对修改前 `HEAD` 的 `app/`、`tests/` 使用 `git archive` 复制到临时目录，运行同一组旧演出/页面契约测试，复现以下 8 个问题：

- `test_duel_v2_presentation`：N06 旧目标选择、旧卡互动类型、旧成长事件、旧护盾受伤场景，共 3 个失败和 1 个异常。
- `test_duel_v2_table_contract`：旧 CSS 字面量和已删除的 `presentReplayMulligan` 断言，3 个失败。
- `test_duel_v2_hooks`：教学角色已注册后仍要求注册表与普通角色集合完全相同，1 个失败。

这些基线问题未在本次扩展范围内改写；不能声称全仓测试全绿。

## 高级人机兼容

新增测试角色不进入高级人机的公开六人编码。由于规则身份包含全部引擎与内容文件，需要在最终代码下重新验证原有冻结模型，再绑定新的运行时规则哈希。保留模型权重、原配牌、训练规则身份和旧验证记录，不训练新权重，也不扩大模型角色范围。


最终规则身份：`564acf493f28cbf1f302667fd5ee3d40770181f85366269b9a2e48fe0c0fdbd5`。

| 模型 | 完整对局 | 对手角色组合 | 结果 |
| --- | ---: | ---: | --- |
| 创生 | 64（32 对先后手互换） | 全部 15 种公开四人组合 | 61 胜；现有发布验收函数通过 |
| 覆纹快攻 | 64（32 对先后手互换） | 全部 15 种公开四人组合 | 63 胜；现有发布验收函数通过 |

两套均完成实际学习换牌、白热化与先手抽牌，没有超时截断或错误对局。对手为现有公开规则策略；该结果仅用作旧模型兼容与绑定证据，不是新增角色强度、对人胜率或平衡结论。报告及逐局公开录像留在 `artifacts/duel-v2-test-characters-20260918/{starter,weave-rush}-report/`，不提交训练/录像产物。清单的 `previous_validation` 保留更新前证据，数字权重不变。

基线复现版本：`00666d286`。命名参考：[Everness GraphQL](https://everness.info/api/graphql)、[异环官方小吱 EP：Afterglow](https://www.bilibili.com/video/BV1hwQwBnEN4/)。

高级人机与绑定回归：`OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest tests.test_duel_v2_advanced_ai tests.test_duel_v2_serving_build`，24 项通过，包括真实模型开局、完整对局与回放存储。

开发服务使用项目 `.venv`，监听 `0.0.0.0:5001`。本机 `http://127.0.0.1:5001/card-game` 与当前局域网 `http://192.168.10.5:5001/card-game` 均返回 HTTP 200；仓库历史地址 `10.4.28.184` 已不属于当前网络，访问超时。

## 同日后续：武备效果互换

用户要求小吱两张武备效果互换，海月「银河暂留」与「往日赋格」效果互换。用户随后补充面板也互换：往日赋格 5/4，银河暂留 3/4，小吱两张仍为 1/5；原卡名、编号和费用保留。专项、卡池与引擎回归共 109 项通过。上文浏览器对局和规则身份为互换前记录；后续规则身份与模型兼容结果另列于下。

互换后的两套模型均完成 64 场评估，但同一工作区的另一项卡牌规则任务在评估期间修改了公共引擎/卡池，导致报告规则哈希与实时工作区不一致；正式绑定校验拒绝更新。本次未改写模型清单或伪造兼容证据，需待并行规则修改稳定后统一重新评估。109 项回归记录对应并行公共卡牌修改前；提交前再次运行新增角色专项确认互换行为。
