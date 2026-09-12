# V2 RL Windows 兼容性检查 · 2026-09-12

结论：Windows 原生 Python 环境中的 RL 接口、Gymnasium 与 CUDA 推理检查通过。**没有训练、优化器更新、经验收集或权重保存；不代表收敛、对战强度或平衡验收。**

## 测试位置与源码

用户指定仓库为 `D:/git/nte_board_game`。其 checkout 尚为旧盖卡版本（提交 `801954b`），没有 V2/RL 文件；因此在仓库下新建 `.rl-validation-20260912` 隔离副本，并在副本内建立 `.venv`。原 checkout 源码及原有 `app.zip` 未修改。

隔离副本从 Mac 当前工作区复制 37 个明确列出的 V2/RL 源码、直接依赖、测试和说明文件。没有复制 `.env`、数据库、账号或凭据。测试前逐文件校验 `source-manifest.json` 中的 SHA-256。源包 SHA-256：`3bcc8e594a8bc0522b7bb4b40dcde29b316c14e2bc77139e08d095b2e0877770`。

通过已有 SSH 密钥连接用户指定 Windows。安装依赖仅使用 Windows 现有本机代理，未修改全局 Python、驱动、系统代理或网站服务。

## 实测环境

| 项目 | 结果 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti，16311 MiB |
| 驱动 | 610.62 |
| GPU compute capability | 12.0 |
| CPU / 内存 | 用户提供 Ultra 265K / 48GB，本轮未做性能测试 |
| Python | 3.13.5，Windows x64 |
| PyTorch | 2.14.0+cu130，官方 CUDA 13.0 wheel |
| NumPy | 2.5.3 |
| Gymnasium | 1.2.3 |
| Stable Baselines3 / sb3-contrib | 2.9.0 / 2.9.0 |

## 结果

- `python -m unittest tests.test_duel_v2_rl -v`：20 项通过，0 跳过，耗时 0.641 秒。
- 真实引擎单测只执行出击、认输共两个明确动作；其余单测使用确定性 fake 或合成公开投影。
- CUDA 上执行固定 16×16 矩阵运算并与 CPU 结果比较，通过。
- 真实 `gymnasium.Env` 的 Dict 观察、候选矩阵、mask、终局 reset 拒绝和截断边界测试通过。
- 创建一次未训练的 MaskablePPO `MultiInputPolicy`，actor/critic 各两层 64 单元，分别调用直接 masked predict 和 `MaskablePolicy` 接口；两次都选择合法候选。
- 模型实际 device 为 `cuda`，参数量 6,856,129；本进程 PyTorch 峰值已分配显存 63,453,184 字节（约 60.5 MiB）。这是初始化与推理检查，**不是训练显存峰值，也不包含全部驱动占用**。
- 推理检查环境仅初始化，不执行 step；`num_timesteps=0`、`_n_updates=0`，未调用 learn、rollout 或优化器，未写权重。

复用时在 PowerShell 执行：

```powershell
Set-Location D:/git/nte_board_game/.rl-validation-20260912
& .venv/Scripts/python.exe -X utf8 scripts/train_duel_v2.py --check
& .venv/Scripts/python.exe -X utf8 -m unittest tests.test_duel_v2_rl -v
```

隔离目录中的 `windows_smoke.py` 为本轮有限 CUDA/推理诊断脚本。测试环境保留便于后续使用；原仓库尚未更新到 V2，不能直接在原目录调用新训练脚本。正式训练仍需另行实现训练循环、权重保存/加载、评估和 Windows 多进程测试。

本次没有测量并行对局吞吐、CPU/GPU 训练速度、长时间稳定性或收敛效果；这些不能由接口测试与单次推理推断。
