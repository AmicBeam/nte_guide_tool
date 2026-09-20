# 夜间训练机制与录像附录 · 2026-09-20

仅统计最终镜像和交叉的冻结对局；机制机会按“每局、每方、每角色、每个总回合至少存在一次合法动作”去重，不按每帧累加。环合机会包括普通与战斗牌入场；实际触发次数可因同回合额外出击超过唯一机会数，所以使用比例用“实际使用的机会回合/可用机会回合”，不以总触发次数除机会。

胜负关联不证明机制因果收益。直接效果只按有明确actor/target及生命before/after的事件统计；同一次操作同时有终结、环合、攻击时，不把整次生命变化重复归因给每个机制。

## 创生镜像机制

| 位置 | 角色 | 机制 | 可用机会 | 用到的机会 | 未用机会 | 触发次数 | 使用局胜/局 | 首次使用总回合均值 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 先手 | 娜娜莉 | 终结 | 93 | 82 | 11 | 82 | 17/31 | 7.26 |
| 先手 | 娜娜莉 | 环合 | 21 | 18 | 3 | 18 | 9/13 | 8.08 |
| 后手 | 娜娜莉 | 终结 | 78 | 72 | 6 | 72 | 15/31 | 7.68 |
| 后手 | 娜娜莉 | 环合 | 17 | 12 | 5 | 12 | 9/10 | 10.60 |
| 先手 | 伊洛伊 | 终结 | 89 | 87 | 2 | 87 | 17/31 | 9.00 |
| 先手 | 伊洛伊 | 环合 | 44 | 5 | 39 | 5 | 1/5 | 18.20 |
| 后手 | 伊洛伊 | 终结 | 79 | 78 | 1 | 78 | 15/30 | 8.53 |
| 后手 | 伊洛伊 | 环合 | 49 | 6 | 43 | 6 | 3/5 | 11.20 |
| 先手 | 零 | 终结 | 80 | 75 | 5 | 75 | 17/32 | 9.75 |
| 先手 | 零 | 环合 | 84 | 13 | 71 | 13 | 9/12 | 13.33 |
| 后手 | 零 | 终结 | 72 | 68 | 4 | 68 | 15/29 | 8.83 |
| 后手 | 零 | 环合 | 65 | 14 | 51 | 14 | 6/11 | 11.64 |
| 先手 | 九原 | 终结 | 70 | 50 | 20 | 50 | 17/31 | 11.06 |
| 先手 | 九原 | 环合 | 31 | 13 | 18 | 13 | 6/11 | 9.36 |
| 后手 | 九原 | 终结 | 66 | 52 | 14 | 52 | 15/30 | 9.73 |
| 后手 | 九原 | 环合 | 40 | 16 | 24 | 16 | 6/13 | 9.38 |

明确归属角色的直接事件收益（分别统计先后手32个侧观测，护盾吸收不计入生命伤害；不声称终结/环合的因果贡献）：

| 位置 | 角色 | 玩家实际生命伤害 | 异能者实际生命伤害 | 对方护盾吸收 | 实际治疗 |
| --- | --- | ---: | ---: | ---: | ---: |
| 先手 | 娜娜莉 | 550 | 959 | 122 | 25 |
| 先手 | 伊洛伊 | 13 | 41 | 12 | 4 |
| 先手 | 零 | 77 | 231 | 37 | 9 |
| 先手 | 九原 | 112 | 311 | 23 | 6 |
| 后手 | 娜娜莉 | 466 | 812 | 100 | 14 |
| 后手 | 零 | 145 | 200 | 32 | 8 |
| 后手 | 九原 | 136 | 394 | 41 | 0 |
| 后手 | 伊洛伊 | 6 | 83 | 10 | 5 |

## 覆纹镜像机制

| 位置 | 角色 | 机制 | 可用机会 | 用到的机会 | 未用机会 | 触发次数 | 使用局胜/局 | 首次使用总回合均值 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 先手 | 薄荷 | 终结 | 57 | 48 | 9 | 48 | 11/30 | 10.27 |
| 先手 | 薄荷 | 环合 | 33 | 24 | 9 | 24 | 9/17 | 10.88 |
| 后手 | 薄荷 | 终结 | 67 | 57 | 10 | 57 | 20/28 | 7.71 |
| 后手 | 薄荷 | 环合 | 29 | 19 | 10 | 19 | 11/13 | 10.15 |
| 先手 | 白藏 | 终结 | 51 | 47 | 4 | 47 | 11/30 | 10.67 |
| 先手 | 白藏 | 环合 | 13 | 9 | 4 | 9 | 5/8 | 14.50 |
| 后手 | 白藏 | 终结 | 49 | 45 | 4 | 45 | 20/28 | 10.14 |
| 后手 | 白藏 | 环合 | 16 | 15 | 1 | 15 | 14/14 | 12.14 |
| 先手 | 零 | 终结 | 73 | 72 | 1 | 72 | 11/30 | 7.60 |
| 先手 | 零 | 环合 | 27 | 3 | 24 | 3 | 2/3 | 9.67 |
| 后手 | 零 | 终结 | 77 | 76 | 1 | 76 | 21/30 | 7.80 |
| 后手 | 零 | 环合 | 37 | 5 | 32 | 5 | 4/5 | 8.00 |
| 先手 | 伊洛伊 | 终结 | 98 | 85 | 13 | 85 | 11/31 | 7.84 |
| 先手 | 伊洛伊 | 环合 | 61 | 0 | 61 | 0 | 0/0 | — |
| 后手 | 伊洛伊 | 终结 | 110 | 80 | 30 | 80 | 21/31 | 8.06 |
| 后手 | 伊洛伊 | 环合 | 58 | 0 | 58 | 0 | 0/0 | — |

明确归属角色的直接事件收益（分别统计先后手32个侧观测，护盾吸收不计入生命伤害；不声称终结/环合的因果贡献）：

| 位置 | 角色 | 玩家实际生命伤害 | 异能者实际生命伤害 | 对方护盾吸收 | 实际治疗 |
| --- | --- | ---: | ---: | ---: | ---: |
| 先手 | 白藏 | 157 | 628 | 94 | 103 |
| 先手 | 薄荷 | 369 | 468 | 21 | 17 |
| 先手 | 伊洛伊 | 12 | 43 | 14 | 21 |
| 先手 | 零 | 145 | 303 | 25 | 27 |
| 后手 | 白藏 | 160 | 697 | 38 | 114 |
| 后手 | 伊洛伊 | 7 | 19 | 6 | 14 |
| 后手 | 薄荷 | 423 | 524 | 25 | 22 |
| 后手 | 零 | 150 | 335 | 29 | 41 |

## 快攻镜像机制

| 位置 | 角色 | 机制 | 可用机会 | 用到的机会 | 未用机会 | 触发次数 | 使用局胜/局 | 首次使用总回合均值 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 先手 | 小吱 | 终结 | 41 | 41 | 0 | 41 | 9/28 | 6.64 |
| 先手 | 小吱 | 环合 | 1 | 1 | 0 | 1 | 0/1 | 11.00 |
| 后手 | 小吱 | 终结 | 49 | 46 | 3 | 46 | 22/30 | 6.40 |
| 后手 | 小吱 | 环合 | 2 | 2 | 0 | 2 | 2/2 | 12.00 |
| 先手 | 零 | 终结 | 53 | 39 | 14 | 39 | 9/30 | 8.00 |
| 先手 | 零 | 环合 | 5 | 3 | 2 | 3 | 1/3 | 10.33 |
| 后手 | 零 | 终结 | 55 | 36 | 19 | 36 | 20/28 | 7.36 |
| 后手 | 零 | 环合 | 3 | 0 | 3 | 0 | 0/0 | — |
| 先手 | 薄荷 | 终结 | 59 | 31 | 28 | 31 | 9/24 | 8.08 |
| 先手 | 薄荷 | 环合 | 12 | 4 | 8 | 4 | 2/4 | 10.50 |
| 后手 | 薄荷 | 终结 | 45 | 33 | 12 | 33 | 18/25 | 7.84 |
| 后手 | 薄荷 | 环合 | 12 | 2 | 10 | 2 | 2/2 | 9.00 |
| 先手 | 海月 | 终结 | 46 | 45 | 1 | 45 | 9/29 | 5.41 |
| 先手 | 海月 | 环合 | 0 | 0 | 0 | 0 | 0/0 | — |
| 后手 | 海月 | 终结 | 43 | 43 | 0 | 43 | 21/29 | 4.34 |
| 后手 | 海月 | 环合 | 0 | 0 | 0 | 0 | 0/0 | — |

明确归属角色的直接事件收益（分别统计先后手32个侧观测，护盾吸收不计入生命伤害；不声称终结/环合的因果贡献）：

| 位置 | 角色 | 玩家实际生命伤害 | 异能者实际生命伤害 | 对方护盾吸收 | 实际治疗 |
| --- | --- | ---: | ---: | ---: | ---: |
| 先手 | 小吱 | 133 | 128 | 9 | 0 |
| 先手 | 海月 | 306 | 159 | 128 | 0 |
| 先手 | 薄荷 | 161 | 169 | 24 | 0 |
| 先手 | 零 | 93 | 150 | 18 | 0 |
| 后手 | 小吱 | 161 | 136 | 14 | 0 |
| 后手 | 海月 | 399 | 129 | 37 | 0 |
| 后手 | 零 | 123 | 168 | 17 | 0 |
| 后手 | 薄荷 | 206 | 157 | 14 | 0 |

## 环合提供者与受益者

| 对阵 | 类型 | 提供者 | 受益者 | 次数 |
| --- | --- | --- | --- | ---: |
| cross/starter/quick-rush | 创生 | a:jiuyuan | a:zero | 6 |
| cross/starter/quick-rush | 创生 | a:nanali | a:zero | 11 |
| cross/starter/quick-rush | 创生 | a:zero | a:iloy | 2 |
| cross/starter/quick-rush | 创生 | a:zero | a:jiuyuan | 9 |
| cross/starter/quick-rush | 创生 | a:zero | a:nanali | 14 |
| cross/starter/quick-rush | 创生 | b:bohe | b:xiaozhi | 3 |
| cross/starter/quick-rush | 创生 | b:bohe | b:zero | 3 |
| cross/starter/quick-rush | 创生 | b:xiaozhi | b:bohe | 1 |
| cross/starter/quick-rush | 创生 | b:zero | b:bohe | 4 |
| cross/starter/weave-rush | 创生 | a:nanali | a:zero | 8 |
| cross/starter/weave-rush | 创生 | a:zero | a:iloy | 6 |
| cross/starter/weave-rush | 创生 | a:zero | a:jiuyuan | 16 |
| cross/starter/weave-rush | 创生 | a:zero | a:nanali | 11 |
| cross/starter/weave-rush | 创生 | b:bohe | b:zero | 9 |
| cross/starter/weave-rush | 创生 | b:zero | b:bohe | 10 |
| cross/starter/weave-rush | 覆纹 | b:baicang | b:bohe | 4 |
| cross/starter/weave-rush | 覆纹 | b:bohe | b:baicang | 5 |
| cross/weave-rush/quick-rush | 创生 | a:bohe | a:zero | 9 |
| cross/weave-rush/quick-rush | 创生 | a:zero | a:bohe | 11 |
| cross/weave-rush/quick-rush | 创生 | b:bohe | b:xiaozhi | 1 |
| cross/weave-rush/quick-rush | 创生 | b:bohe | b:zero | 1 |
| cross/weave-rush/quick-rush | 创生 | b:xiaozhi | b:bohe | 1 |
| cross/weave-rush/quick-rush | 创生 | b:zero | b:bohe | 6 |
| cross/weave-rush/quick-rush | 覆纹 | a:baicang | a:bohe | 10 |
| cross/weave-rush/quick-rush | 覆纹 | a:bohe | a:baicang | 8 |
| mirror/quick-rush | 创生 | a:bohe | a:zero | 1 |
| mirror/quick-rush | 创生 | a:xiaozhi | a:bohe | 2 |
| mirror/quick-rush | 创生 | b:bohe | b:xiaozhi | 3 |
| mirror/quick-rush | 创生 | b:bohe | b:zero | 2 |
| mirror/quick-rush | 创生 | b:xiaozhi | b:bohe | 1 |
| mirror/quick-rush | 创生 | b:zero | b:bohe | 3 |
| mirror/starter | 创生 | a:jiuyuan | a:zero | 1 |
| mirror/starter | 创生 | a:nanali | a:zero | 16 |
| mirror/starter | 创生 | a:zero | a:iloy | 6 |
| mirror/starter | 创生 | a:zero | a:jiuyuan | 17 |
| mirror/starter | 创生 | a:zero | a:nanali | 18 |
| mirror/starter | 创生 | b:nanali | b:zero | 10 |
| mirror/starter | 创生 | b:zero | b:iloy | 5 |
| mirror/starter | 创生 | b:zero | b:jiuyuan | 12 |
| mirror/starter | 创生 | b:zero | b:nanali | 12 |
| mirror/weave-rush | 创生 | a:bohe | a:zero | 2 |
| mirror/weave-rush | 创生 | a:zero | a:bohe | 13 |
| mirror/weave-rush | 创生 | b:bohe | b:zero | 6 |
| mirror/weave-rush | 创生 | b:zero | b:bohe | 11 |
| mirror/weave-rush | 覆纹 | a:baicang | a:bohe | 11 |
| mirror/weave-rush | 覆纹 | a:bohe | a:baicang | 13 |
| mirror/weave-rush | 覆纹 | a:iloy | a:baicang | 1 |
| mirror/weave-rush | 覆纹 | b:baicang | b:bohe | 8 |
| mirror/weave-rush | 覆纹 | b:bohe | b:baicang | 9 |
| mirror/weave-rush | 覆纹 | b:iloy | b:baicang | 1 |

`analysis.json → harmony_details`保留每次真实事件的提供者、受益者、类型、回合及同操作后续事件；同操作总生命差不能直接当作环合伤害。`mechanism_rows/effect_rows`中的交叉数据按对手单列。终结开始、到期、倒地取消和持续效果的原始事件均在逐局raw文件中；本表不把难以唯一归因的持续增益换算为胜率贡献。

## 48局已保存公开回放索引

每个镜像或交叉单元保存前4个种子的两种先手，共8局。文件均由原评估事件导出，没有重新抽样。此处不公开种子或暗牌；私有记录仅在本地分析产物。

| 对局ID | 对阵 | 实际先手 | 胜者 | 总回合 | 公开回放文件 |
| --- | --- | --- | --- | ---: | --- |
| g00320 | mirror/starter | 创生（a） | 创生（b） | 20 | [g00320.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g00320.json.gz) |
| g00321 | mirror/starter | 创生（b） | 创生（b） | 11 | [g00321.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g00321.json.gz) |
| g00322 | mirror/starter | 创生（a） | 创生（b） | 18 | [g00322.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g00322.json.gz) |
| g00323 | mirror/starter | 创生（b） | 创生（b） | 23 | [g00323.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g00323.json.gz) |
| g00324 | mirror/starter | 创生（a） | 创生（b） | 22 | [g00324.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g00324.json.gz) |
| g00325 | mirror/starter | 创生（b） | 创生（a） | 18 | [g00325.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g00325.json.gz) |
| g00326 | mirror/starter | 创生（a） | 创生（a） | 13 | [g00326.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g00326.json.gz) |
| g00327 | mirror/starter | 创生（b） | 创生（b） | 17 | [g00327.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g00327.json.gz) |
| g00672 | mirror/weave-rush | 覆纹（a） | 覆纹（b） | 22 | [g00672.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g00672.json.gz) |
| g00673 | mirror/weave-rush | 覆纹（b） | 覆纹（a） | 20 | [g00673.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g00673.json.gz) |
| g00674 | mirror/weave-rush | 覆纹（a） | 覆纹（a） | 15 | [g00674.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g00674.json.gz) |
| g00675 | mirror/weave-rush | 覆纹（b） | 覆纹（b） | 13 | [g00675.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g00675.json.gz) |
| g00676 | mirror/weave-rush | 覆纹（a） | 覆纹（a） | 13 | [g00676.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g00676.json.gz) |
| g00677 | mirror/weave-rush | 覆纹（b） | 覆纹（a） | 20 | [g00677.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g00677.json.gz) |
| g00678 | mirror/weave-rush | 覆纹（a） | 覆纹（b） | 26 | [g00678.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g00678.json.gz) |
| g00679 | mirror/weave-rush | 覆纹（b） | 覆纹（b） | 13 | [g00679.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g00679.json.gz) |
| g01024 | mirror/quick-rush | 快攻（a） | 快攻（b） | 12 | [g01024.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01024.json.gz) |
| g01025 | mirror/quick-rush | 快攻（b） | 快攻（a） | 14 | [g01025.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01025.json.gz) |
| g01026 | mirror/quick-rush | 快攻（a） | 快攻（b） | 12 | [g01026.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01026.json.gz) |
| g01027 | mirror/quick-rush | 快攻（b） | 快攻（b） | 13 | [g01027.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01027.json.gz) |
| g01028 | mirror/quick-rush | 快攻（a） | 快攻（b） | 12 | [g01028.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01028.json.gz) |
| g01029 | mirror/quick-rush | 快攻（b） | 快攻（a） | 14 | [g01029.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01029.json.gz) |
| g01030 | mirror/quick-rush | 快攻（a） | 快攻（b） | 10 | [g01030.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01030.json.gz) |
| g01031 | mirror/quick-rush | 快攻（b） | 快攻（a） | 10 | [g01031.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01031.json.gz) |
| g01056 | cross/starter/weave-rush | 创生（a） | 创生（a） | 25 | [g01056.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01056.json.gz) |
| g01057 | cross/starter/weave-rush | 覆纹（b） | 覆纹（b） | 32 | [g01057.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01057.json.gz) |
| g01058 | cross/starter/weave-rush | 创生（a） | 创生（a） | 17 | [g01058.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01058.json.gz) |
| g01059 | cross/starter/weave-rush | 覆纹（b） | 创生（a） | 18 | [g01059.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01059.json.gz) |
| g01060 | cross/starter/weave-rush | 创生（a） | 创生（a） | 15 | [g01060.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01060.json.gz) |
| g01061 | cross/starter/weave-rush | 覆纹（b） | 创生（a） | 14 | [g01061.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01061.json.gz) |
| g01062 | cross/starter/weave-rush | 创生（a） | 创生（a） | 11 | [g01062.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01062.json.gz) |
| g01063 | cross/starter/weave-rush | 覆纹（b） | 覆纹（b） | 33 | [g01063.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01063.json.gz) |
| g01088 | cross/starter/quick-rush | 创生（a） | 创生（a） | 13 | [g01088.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01088.json.gz) |
| g01089 | cross/starter/quick-rush | 快攻（b） | 创生（a） | 12 | [g01089.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01089.json.gz) |
| g01090 | cross/starter/quick-rush | 创生（a） | 创生（a） | 13 | [g01090.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01090.json.gz) |
| g01091 | cross/starter/quick-rush | 快攻（b） | 创生（a） | 20 | [g01091.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01091.json.gz) |
| g01092 | cross/starter/quick-rush | 创生（a） | 快攻（b） | 10 | [g01092.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01092.json.gz) |
| g01093 | cross/starter/quick-rush | 快攻（b） | 创生（a） | 14 | [g01093.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01093.json.gz) |
| g01094 | cross/starter/quick-rush | 创生（a） | 创生（a） | 19 | [g01094.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01094.json.gz) |
| g01095 | cross/starter/quick-rush | 快攻（b） | 快攻（b） | 19 | [g01095.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01095.json.gz) |
| g01120 | cross/weave-rush/quick-rush | 覆纹（a） | 快攻（b） | 14 | [g01120.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01120.json.gz) |
| g01121 | cross/weave-rush/quick-rush | 快攻（b） | 覆纹（a） | 12 | [g01121.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01121.json.gz) |
| g01122 | cross/weave-rush/quick-rush | 覆纹（a） | 覆纹（a） | 17 | [g01122.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01122.json.gz) |
| g01123 | cross/weave-rush/quick-rush | 快攻（b） | 覆纹（a） | 16 | [g01123.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01123.json.gz) |
| g01124 | cross/weave-rush/quick-rush | 覆纹（a） | 覆纹（a） | 13 | [g01124.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01124.json.gz) |
| g01125 | cross/weave-rush/quick-rush | 快攻（b） | 覆纹（a） | 12 | [g01125.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01125.json.gz) |
| g01126 | cross/weave-rush/quick-rush | 覆纹（a） | 覆纹（a） | 13 | [g01126.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01126.json.gz) |
| g01127 | cross/weave-rush/quick-rush | 快攻（b） | 覆纹（a） | 16 | [g01127.json.gz](/Users/bytedance/projects/nte_board_game/artifacts/rl-evals/night-20260920/completed-v3/verified/evaluation/replays/g01127.json.gz) |
