# Windows CUDA 短训练记录 · 2026-09-15

## 结果

用户授权在 `192.168.10.4` 做短训练。使用 RTX 5060 Ti 16GB、驱动 610.62、PyTorch 2.14.0+cu130；复用 `D:/git/nte_board_game/.rl-validation-20260912/.venv/Scripts/python.exe`。

| 检查 | 结果 |
| --- | --- |
| CUDA/native 与公开观察预检 | 16 个局面通过；含先手首回合抽牌及白热化相关观察、设备内公开构筑 reset |
| 出牌训练 | 256 环境，horizon 16，hidden 64；3 次更新、共 24 个优化器步骤，约 15.076 秒，完成 136 场对局 |
| 配牌训练 | 固定娜娜莉、伊洛伊、零、九原；8 个候选、每个 2 场，16 场全部结束，0 截断；1 次更新，约 5.674 秒 |
| checkpoint 加载 | 配牌 checkpoint 重新加载成功，导出 2 个合法候选；角色及顺序保持固定 |
| 数值模型导出 | 8 个 Torch/NumPy 一致性局面通过，最大分数误差 2.682209e-7，仍为待评估模型 |
| 本机代码回归 | 60 项通过，涵盖编译后端、公开卡池矩阵、设备别名、编译选项和高级人机 |

出牌训练上限为 3 次更新或 90 秒，配牌为 1 次更新或 120 秒；均按更新次数提前结束。上述时间是训练循环内时间，不含同步、编译、预检和导出。GPU 同时存在其他负载，且使用下述短测编译档，不能据此推算独占 GPU 或默认编译档吞吐。

训练没有写逐动作日志、分析报告或录像。出牌 `metrics.jsonl` 991 字节，配牌 174 字节；分别保存约 923 KB、20 KB 的 checkpoint，配牌另存当前批候选与标量分数。数值模型未批准上线，未覆盖网站模型。

## 发现与修复

1. SSH 会话未加载 Visual Studio 头文件路径，native 对照编译报 `stdint.h: no include path set`。仅在测试进程加载已有 `VsDevCmd.bat -arch=x64 -host_arch=x64`，没有重装编译器或修改系统环境。
2. 默认 `cuModuleLoadData` 即时编译长时间未返回；定时 Python 栈确认等待在模块装载阶段，尚未开始优化器更新。单独降低驱动 JIT 优化仍等待。启用 `NTE_CUDA_FAST_COMPILE=max` 与 `NTE_CUDA_JIT_OPT_LEVEL=0` 后通过。前者使用 [NVIDIA NVRTC 的快速编译选项](https://docs.nvidia.com/cuda/archive/12.9.0/nvrtc/index.html)，后者使用 [CUDA 模块装载选项](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)。默认配置没有改变；默认高优化档的长等待尚未解决。
3. 公开构筑 reset 将 `torch.device('cuda')` 与实际张量设备 `cuda:0` 直接比较，导致错误拒绝。改为比较各张量的具体设备；CPU 设备别名回归与 CUDA 实机 reset 均通过。

短测编译档只改变编译优化选择，不修改抽牌、费用、角色或卡牌规则。后续强度和长期稳定性仍需独立训练及评估，本次少量更新不能证明模型已经掌握全部公开卡牌。

## 文件与身份

远端隔离目录：`D:/git/nte_board_game/.rl-public-short-20260915-1aad413`。原 Windows checkout 保留不动。同步 293 个白名单源码/测试/文档，初始 SHA-256 清单为 `source-manifest.json`；实机修复两文件后，以 `short-test-source-overrides.json` 记录覆盖后的哈希。

- 规则身份：`c1abaf1979bf377994a0c019e05e1bb4267098782562937d757c73baadc69d7b`。
- `compile_engine.py`：`9c6a02e12bf17672fcaf692c5ef5c3ae7dedd106962d88efc7cf0504c75ac3cb`。
- `compiled_backend.py`：`e38237e5be0552e2b01ad19e432ecb0d27ea466deef0846dd3827f3b2878bc1a`。
- 出牌 checkpoint：`artifacts/play-short/latest.pt`，SHA-256 `7e19c25efc9cd4a9f5101dd3ee02235f5928f52d77ee1f4fd9631d92c10c7373`。
- 配牌 checkpoint：`artifacts/cards-short/latest.pt`，SHA-256 `defa32b198b4aed95c85a9b3dc892bd879fdd4bfc16e2b0dfaf969a85fb67143`。
- 配牌候选：`artifacts/cards-short/candidate.json`；重新加载后候选在 `artifacts/cards-proposals/`。
- 数值模型：`artifacts/play-export/starter.npz`，SHA-256 `17b1adca2529e92103fdecd611b5bd9f1c4689547bdbdfa059d3f7f678a3c6bf`，仍禁止直接上线。

两段短训练已退出；仅为诊断启动的旧预检进程也已结束。没有启动夜间长期训练。
