---
name: shaft-axis-workflow
description: 使用 NTE Tools 排轴模拟器 CLI 读取在线轴和分享轴，编辑动作 JSON，用网站同版 JavaScript 引擎复算与自检，并按用户要求创建或更新私有轴。
---

# 异环排轴模拟器 CLI

用 Python 3.10+ 运行包内的 nte-guide.py。编辑示例和复算需要本机 Node.js，建议使用 Node.js 22。可在仓库外运行。

## 登录与站点

每个 shaft 操作都需要先登录目标站点。由运行环境提供 NTE_GUIDE_TOKEN，并通过 NTE_GUIDE_URL 或子命令前的 --base-url 选定同一站点。生产站使用可信的 HTTPS 地址。CLI 会先验证登录，再使用当前账号权限。凭据留在可信运行环境，避免在对话、命令参数和日志中显示。分享链接也只交给对应的站点。

## 读取、编辑与自检

先运行 python3 nte-guide.py --version 和 python3 nte-guide.py shaft --help。CLI 版本标识客户端，实际复算规则由目标站点提供。以下示例在解压目录运行，123 代表实际轴 ID。

~~~bash
python3 nte-guide.py shaft axes --scope mine
python3 nte-guide.py shaft catalog > catalog.json
python3 nte-guide.py shaft axis 123 > original.json
python3 nte-guide.py shaft shared '/shaft/rotation?share=<实际分享码>' > shared.json
node examples/move-step.js original.json '<实际步骤ID>' 55 > edited.json
python3 nte-guide.py shaft simulate edited.json > calculated.json
~~~

1. 从 axis.steps 取真实步骤 ID，从 catalog 核对 action_id 和队伍槽位。编辑时保留 team、enemy、options 和无关步骤。更新已有轴还要保留外层 id、updated_at、is_owner 和 visibility。
2. start_tick 是编辑器坐标，单位 0.1 秒。启用时停或遮罩时，用 result.details 里的 start_tick、end_tick 和 q_instant_release_anchor_step_id 核对实际出手时间和遮罩归属。visual_start_tick 用于核对显示位置。
3. simulate 从目标站点获取本次部署的 shaft_engine.js、shaft_self_check.js 和账号目录，再通过本机 Node 运行 simulateAxis 和 inspectAxis。检查 self_check、result.details 中的 warnings 和 result.summary。需要按动作统计残像等后台伤害时，按 result.periodic_damage_events 的 source_step_id 对应结果中的 step_id。
4. 再改动作或配置后重新复算。保存使用 calculated.json，轴内容变化后旧结果会被 CLI 拒绝。

## 保存

仅在用户要求写入时保存。另存使用 create，修改原私有轴使用 update 并明确指定 ID。收到版本冲突时，重新读取服务器最新轴并重新应用修改和复算。

~~~bash
python3 nte-guide.py shaft create calculated.json --title '三 E 循环' > created.json
python3 nte-guide.py shaft update 123 calculated.json > saved.json
~~~

保存成功后 CLI 会回读目标轴。汇报轴 ID、自检和用户要求核对的指标。分享轴可用作另存起点，更新原轴需要本人拥有该私有轴。若收到已写入但回读失败的提示，先读回目标轴，避免重复创建。
