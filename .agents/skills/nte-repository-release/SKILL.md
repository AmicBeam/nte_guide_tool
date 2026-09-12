---
name: nte-repository-release
description: Manage this repository's public/private Git relationship, isolated private development and deployment branches, and selective releases through public pull requests. Use when checking remotes, pushing private work, separating deployable code, or merging approved changes into the public repository; do not use for ordinary local commits.
---

# NTE 仓库与公开合码

## 固定仓库关系

- `origin` 是公共开源仓库：`https://github.com/AmicBeam/nte_guide_tool.git`，默认分支为 `main`。
- `private` 是私有开发仓库：`https://github.com/AmicBeam/nte_guide_tool_private.git`，默认分支为 `main`。
- 提交身份固定为 `AmicBeam <1425257712@qq.com>`。
- 私有仓库承担未公开开发和隔离分支；公共仓库只接收用户明确批准公开的提交。

开始操作前读取 `git status -sb`、`git remote -v`、`git branch -vv`，并核对提交身份。若远端名称、URL 或身份不符合以上约定，停止写操作并向用户确认，不静默改写远端。

## 当前分支职责

- `codex/private-deploy`：私有部署分支。部署不应包含尚未获准上线的隔离功能。
- `codex/containment-roguelike`：桌游隔离开发分支；其中桌游重构不得因为部署其他模块而进入部署包或公共仓库。

分支职责可能随任务调整，因此每次都以本地和两个远端的实际提交哈希复核，不能只看同名分支。同一分支名在 `origin` 与 `private` 上可能指向不同提交。

## 默认推送规则

- 用户只说“提交”或“推送”，且没有明确要求公开时，推送到 `private`，不得推送到 `origin`。
- 使用显式远端和目标分支，例如 `git push private HEAD:<branch>`；涉及公开发布时更不能依赖当前 upstream 猜测目标。
- 禁止对公共仓库执行 `git push --all`、`git push --mirror`、向 `main` 强推，或把整个私有开发分支直接推到公共仓库。
- 推送后用远端分支哈希核对结果，并确认公共对应分支没有意外移动。

## 部署分支

`scripts/deploy_windows_app.sh` 从当前 Git `HEAD` 中已提交的 `app/` 打包，不根据远端选择内容，也不会包含未提交的 `app/` 改动。

部署前：

1. 切到用户指定的部署分支；未另行指定时使用 `codex/private-deploy`。
2. 确认该分支不含被隔离的功能提交，并确保 `app/` 工作区干净。
3. 运行 `scripts/deploy_windows_app.sh --check` 和 `--prepare`。
4. 只有用户要求实际部署时才运行 `scripts/deploy_windows_app.sh --deploy`。

不要把 `--allow-dirty-app` 当作日常选项。它会忽略未提交的 `app/` 改动，容易让用户误判实际部署内容。

## 向公共仓库合码

私有仓库中的分支不能直接作为公共仓库 PR 的源分支。公开 PR 的 head 分支必须先存在于公共仓库，因此采用“从公共 `main` 建发布分支，只挑选获准提交”的流程：

1. 先确认用户明确批准公开的功能、文件和提交范围；“推送到私有仓库”不等于授权公开。
2. 获取两个远端最新状态，检查私有功能分支相对 `origin/main` 的提交与完整差异。
3. 从最新 `origin/main` 新建 `codex/public-release-<topic>`，不要从包含其他私有工作的分支直接推送。
4. 只 `cherry-pick` 范围清晰且获准公开的提交。若一个提交混有私有内容，不得整笔挑选；应在私有侧先拆分，或在公开发布分支上重做获准部分。
5. 检查相对 `origin/main` 的提交列表和完整 diff，确认没有私有配置、凭据、内部文档、未获准功能或无关提交；运行受影响模块测试和 `git diff --check`。
6. 经用户确认公开后，显式推送发布分支：`git push -u origin HEAD:codex/public-release-<topic>`。
7. 创建目标为公共仓库 `main` 的 PR，例如：

   ```bash
   gh pr create \
     --repo AmicBeam/nte_guide_tool \
     --base main \
     --head AmicBeam:codex/public-release-<topic>
   ```

8. 报告 PR 链接和公开 diff 摘要。除非用户明确要求并且检查通过，不代替用户合并 PR。

公共 PR 合并后，可按任务需要把最新 `origin/main` 合入私有长期分支并推送到 `private`；不通过改写历史或强推来同步。

## 公开前停止条件

遇到以下任一情况，不推送公共分支，也不创建 PR：

- 用户没有明确授权公开；
- 获准范围与提交边界不一致；
- diff 中存在凭据、私有配置、隔离功能或来源不明文件；
- 测试失败，或者无法说明公开分支相对 `origin/main` 的完整差异。
