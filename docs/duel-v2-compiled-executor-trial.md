# 共享规则双后端执行器试点（2026-09-14）

## 结论

共享规则生成 CPU 本机库和 CUDA 内核，并接入现有训练流程的结构已跑通；本次没有测出实质性端到端提速，不启用为默认训练器。

首版编译条件保守，1494 次普通出击中只有20次进入编译路径（1.34%）；响应、环合、装备/终结、角色倒地及特殊效果仍回退到原实现。代码没有通过删除这些机制来获得速度，也不能据此称“完整GPU规则引擎已实现”。

## 实现

- `rl/rule_ir/ast.py`：Program支持可选字段布局与side alias；校验代码生成标识符、字段重复和alias目标。原19字段战斗程序继续兼容。
- `rl/rule_ir/ordinary_sortie.py`：普通出击常见路径的唯一试验IR，复用现有战斗表达式，加入普通入场、旧前排护盾与last_front、个人环合和全队能量。输入归一化为攻击方视角，每次仅运行一步。
- `rl/rule_ir/lower_cpu.py`：张量解释器可执行相同布局，用于生成代码对照。
- `rl/rule_ir/lower_cuda.py`：从同一IR生成CUDA计算代码。
- `rl/rule_ir/lower_native.py`：沿用同一标量计算代码，生成CPU逐行循环；Windows用已有MSVC编译DLL，Mac/Linux使用clang++或g++。调用者显式提供编译目录，导入时不编译。CPU库支持显式close，防止Windows临时DLL无法清理。
- `rl/gpu_duel/compiled_trial.py`：显式、进程内、限作用域的分派器。试点保留原训练器接口，仅临时替换sortie调用，退出后恢复。不可与同一进程内其他并发任务同时启用，不是生产全局路由。

Python仍负责支持范围判断、数据打包和结果写回，也复用原来的attack_power计算。费用与普通出击次数由原来的上层流程扣除。此试点不是“所有普通出击计算已经编译”。CPU生成代码也是单线程原生循环，不是完整多核CPU执行器。

### 支持与回退

仅接管普通出击（战斗牌/响应调用直接回退）、原四名角色参与、没有特殊状态、没有潜在响应牌、没有入场环合且本次不会使角色倒地的局面。支持玩家生命归零，终局不积累战斗资源。

支持检查在修改状态前完成。未覆盖的行仍调用原实现；已经完成编译的行用mask屏蔽。由于原执行器依然按整批张量执行，少数行被编译并不足以跳过整批Python调度/张量算子成本。CUDA试点使用CPU环境，因此包含CPU→GPU包搬运→CUDA计算→CPU写回；未实现显存常驻全对局。

## 正确性与环境

Windows RTX5060Ti；已有Visual Studio 2022 C++工具链及PyTorch NVRTC。没有安装新依赖或修改旧权重。

28项Windows相关测试全部通过，含新增完整张量状态对照（CPU本机与CUDA两后端均执行）、原IR及CUDA、CPU快速路径、终局回归。新测试覆盖正常入场/反击、玩家致死与终局资源抑制，以及响应、装备、环合、角色死亡回退；CPU/CUDA均与同一IR的张量解释结果一致。Mac未安装torch，执行了纯IR布局校验和语法检查，编译运行测试在Mac跳过。

这里的新增完整状态对照以当前张量规则实现为参照；现有测试另含若干Python网站引擎oracle检查。不能把有限测试称为完整卡池和所有轨迹都已与网站规则一致。

```bash
python -m unittest tests.test_duel_v2_compiled_trial tests.test_duel_v2_gpu_cpu_fastpath tests.test_duel_v2_gpu_transitions tests.test_duel_v2_gpu_duel tests.test_duel_v2_rule_ir -v
```

## 三路径真实短训练

1024局×8学习方步×2轮，compact观察，hidden256、minibatch1024、epochs2，CPU算子20线程，torch种子20260914。每条路径均16384学习方步、32次优化器更新。模型在内存中真实更新，但不写回此前的checkpoint。

| 路径 | 两轮总耗时 | 学习方步/秒 | 编译普通出击 |
| --- | ---: | ---: | ---: |
| 当前已优化张量执行器 | 30.007秒 | 546.00 | — |
| 共享IR→CPU本机代码 | 32.304秒 | 507.18 | 20 / 1494 |
| 共享IR→CUDA代码 | 29.779秒 | 550.18 | 20 / 1494 |

CUDA约0.8%的差异在单次短测下不能视为可靠提升；CPU这次更慢，也不能把全部差额都归因于分派（GPU更新时间存在波动）。当前证据只支持“结构和等价性试点通过，性能收益未证实”。

三路径每轮结束时完整GpuState张量和模型参数的SHA-256一致：

- `b92c08db4b28c6c1f991019acdc4d66088bb02541e05ef9340ec9ebef060cf52`
- `dbf38ce78949ae4c0be3f064b17a6c11273ebdd3b5f5d424fa6f9d00e46c0578`

结果包含全部回退、支持检查、打包及CPU/GPU搬运；排除启动编译和checkpoint。采用相同分阶段同步计时，插桩会改变异步重叠。只覆盖对局前16个学习方决策，不能代表长训。

Windows产物：`D:/git/nte_board_game/artifacts/duel-v2-compiled-trial-{baseline,native,cuda}-20260914/`，内含profile.json；native/cuda另有equivalence.json。CPU库及生成源在native产物目录的compiler-cache，不纳入Git。

## 后续决策

保留当前已优化执行器为默认。若继续投入，应优先把高频战斗牌、响应与环合纳入共享IR，提高实际覆盖率，并将支持/回退分组与状态布局设计成可高效批量执行；只扩大当前无特殊效果切片不足以解决瓶颈。最终需要整个规则、合法动作、观察和reset驻留GPU，才能减少每步Python调度和数据往返。下一阶段仍需完整差分与同条件端到端验证，不承诺十倍。
