"""Chinese complete-training flow report. No training, provider, or website writes."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path

from app.modules.card_game.rl.training_flow_agents import (
    REPO_ROOT, as_epoch, candidate_stages, run_root_of, snapshot_of, stage_order,
)

MISSING = '未完成'
DEFAULT_REPORT_NAME = 'flow-report.md'
EVIDENCE_TOPICS = ('card', 'esper', 'summon', 'use', 'release', 'mechanism')
TOPIC_LABELS = {
    'card': '逐卡',
    'esper': '异能者',
    'summon': '召唤',
    'use': '使用/打出',
    'release': '自动释放（与实际使用分开）',
    'mechanism': '机制',
}
DELIVERY_TOPICS = ('raw', 'public_replay', 'model')
DELIVERY_LABELS = {
    'raw': '原始记录',
    'public_replay': '公开录像',
    'model': '数值模型',
}
EVAL_MODES = (('pure', '纯网络'), ('search', '带搜索'))
EVAL_CONTRASTS = (
    ('baseline', '基线对照'),
    ('new_old', '新旧模型对照'),
    ('mulligan', '起手换牌对照'),
    ('varied_opponents', '变化对手'),
)


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _inside_root(path, root):
    real_root = os.path.realpath(root)
    real_path = os.path.realpath(path)
    return real_path == real_root or real_path.startswith(real_root + os.sep)


def _last_result(info):
    info = _mapping(info)
    if info.get("invalidated"):
        return {}
    attempts = info.get("attempts")
    if isinstance(attempts, list) and attempts:
        last = attempts[-1] if isinstance(attempts[-1], dict) else {}
        return last.get("result") or last
    return info.get("result") or {}


def collect_metrics(snapshot):
    merged = {}
    if isinstance(snapshot.get("metrics"), dict):
        merged.update(snapshot["metrics"])
    skip = ("returncode", "timed_out", "launched", "exit_zero_is_not_success", "stderr_preview")
    for name in stage_order():
        metrics = _mapping(stage_state(snapshot, name).get("metrics"))
        for key, value in metrics.items():
            if key in skip:
                continue
            if name == "delivery" and key == "models":
                delivery = _mapping(merged.get("delivery"))
                delivery["models"] = value
                merged["delivery"] = delivery
                continue
            merged[key] = value
    selection = _mapping(stage_state(snapshot, "freeze_selection").get("metrics"))
    if "models" in selection:
        merged["models"] = selection["models"]
    delivery_metrics = _mapping(stage_state(snapshot, "delivery").get("metrics"))
    if delivery_metrics:
        merged["delivery"] = {**_mapping(merged.get("delivery")), **delivery_metrics}
    registered = stage_state(snapshot, 'delivery')['artifacts']
    if registered:
        delivery = merged.setdefault('delivery', {})
        numeric_files = [item for item in registered if Path(item['path']).suffix in ('.npz', '.pt', '.pth', '.ckpt')]
        if numeric_files:
            delivery.setdefault('model_files', numeric_files)
        delivery.setdefault('raw_records', [item for item in registered if 'raw' in item['path']])
        delivery.setdefault('public_replays', [item for item in registered if 'replay' in item['path']])
    if snapshot.get('final_review'):
        merged['final_audit'] = snapshot['final_review']
    return merged


def stage_state(snapshot, stage_id):
    info = _mapping(_mapping(snapshot.get("stages")).get(stage_id))
    result = _last_result(info)
    attempts = info.get("attempts") if isinstance(info.get("attempts"), list) else []
    return {
        "status": info.get("status") or result.get("status") or "pending",
        "skip_reason": info.get("skip_reason") or info.get("reason"),
        "verdict": result.get("verdict"),
        "summary": result.get("summary"),
        "attempts": attempts,
        "artifacts": result.get("artifacts") or info.get("artifacts") or [],
        "metrics": result.get("metrics") or info.get("metrics") or {},
    }


def present(value):
    return value not in (None, "", [], {}, ())


def display(value):
    return value if present(value) else MISSING


def coverage(planned, completed):
    if planned is None and completed is None:
        return MISSING
    if planned is None:
        return f"{MISSING}（完成 {completed}，计划缺失）"
    if completed is None:
        return f"{MISSING}（计划 {planned}，完成缺失）"
    if completed < planned or planned <= 0:
        return f"{MISSING}（{completed}/{planned}）"
    return f"{completed}/{planned}"


def winrate(wins, completed):
    if not present(completed) or completed in (0, "0") or wins is None:
        return MISSING
    try:
        total = float(completed)
        hit = float(wins)
    except (TypeError, ValueError):
        return MISSING
    if total <= 0 or not 0 <= hit <= total:
        return MISSING
    return f"{hit / total:.1%}（{int(hit)}/{int(total)}）"


def _count_table(build):
    cards = list(_mapping(build).get("card_ids") or ())
    counts = Counter(cards)
    return len(cards), dict(counts)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _nested(metrics, *keys):
    current = metrics
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _identity_map(metrics):
    models = metrics.get("models") or metrics.get("model_identities") or {}
    if not isinstance(models, dict):
        return {}
    out = {}
    for key, value in models.items():
        if isinstance(value, dict):
            out[str(key)] = {
                "initial": value.get("initial") or value.get("initial_sha256"),
                "selected": value.get("selected") or value.get("best") or value.get("selected_sha256"),
                "latest": value.get("latest") or value.get("latest_sha256"),
            }
        else:
            out[str(key)] = {"initial": MISSING, "selected": MISSING, "latest": display(value)}
    return out


def _mode_block(metrics, mode):
    evaluation = _mapping(metrics.get("evaluation") or metrics.get("eval"))
    modes = _mapping(evaluation.get("modes") or evaluation.get("matrices"))
    block = modes.get(mode) or _mapping(metrics.get('matrix_modes')).get(mode) or evaluation.get(mode) or {}
    return _mapping(block)


def _planned_completed(block):
    def _number(*keys):
        for key in keys:
            if key in block and block[key] is not None:
                return block[key]
        return None
    planned = _number("planned_cells", "planned", "planned_games")
    completed = _number("completed_cells", "completed", "completed_games")
    matrix = block.get("matrix") or block.get("cells")
    if planned is None and isinstance(matrix, dict):
        planned = len(matrix)
        completed = sum(1 for cell in matrix.values() if _mapping(cell).get("completed"))
    if planned is None and isinstance(matrix, list):
        planned = len(matrix)
        completed = sum(1 for cell in matrix if _mapping(cell).get("completed"))
    return planned, completed


def _topic_status(metrics, topic):
    analysis = _mapping(metrics.get("analysis") or metrics.get("detail") or metrics)
    block = analysis.get(topic) or analysis.get(topic + "s") or metrics.get(topic)
    if block is None:
        return MISSING
    if isinstance(block, dict):
        if block.get("status") in ("missing", "incomplete", MISSING):
            return MISSING
        if present(block.get("rows") or block.get("games") or block.get("completed")):
            planned, completed = _planned_completed(block)
            if planned is not None:
                return coverage(planned, completed)
            return "已记录"
        if block.get("complete") is True:
            return "已记录"
        if block.get("complete") is False:
            return MISSING
        if present(block):
            return "已记录"
        return MISSING
    if block in (False, 0, "0"):
        return MISSING
    return display(block)


def _delivery_status(metrics, topic):
    delivery = _mapping(metrics.get("delivery"))
    aliases = {
        "raw": ("raw", "raw_records", "evaluation_raw"),
        "public_replay": ("public_replay", "public_replays", "replays"),
        "model": ("numerical_models", "model_files", "models", "model"),
    }
    block = None
    for key in aliases[topic]:
        if key in delivery:
            block = delivery[key]
            break
    if block is None and topic != "model":
        for key in aliases[topic]:
            if key in metrics:
                block = metrics[key]
                break
    if block is None:
        return MISSING
    if isinstance(block, list):
        files = [item for item in block if isinstance(item, dict) and item.get("path") and item.get("sha256")]
        if not files:
            return MISSING
        if topic == "model" and not any(str(item.get("path", "")).endswith((".npz", ".pt", ".pth", ".bin")) for item in files):
            return f"{MISSING}（仅有清单，无数值文件）"
        return f"{len(files)} 个哈希文件"
    if isinstance(block, dict):
        artifacts = block.get("artifacts") or block.get("files") or block.get("paths")
        if isinstance(artifacts, list) and artifacts:
            return _delivery_status({"delivery": {topic: artifacts}}, topic)
        if block.get("json_only") or block.get("metadata_only"):
            return f"{MISSING}（仅元数据）"
        if present(block.get("sha256")) and present(block.get("path")):
            suffix = str(block.get("path"))
            if topic == "model" and not suffix.endswith((".npz", ".pt", ".pth", ".bin")):
                return f"{MISSING}（仅有清单，无数值文件）"
            return "1 个哈希文件"
        return MISSING
    return MISSING


def _gumbel_block(metrics):
    learning = _mapping(metrics.get("learning") or metrics.get("search") or metrics)
    algorithm = learning.get("algorithm") or metrics.get("search_algorithm") or learning.get("search_algorithm")
    simulations = learning.get("simulations") or metrics.get("simulations")
    candidates = learning.get("gumbel_candidates") or metrics.get("gumbel_candidates")
    reward = learning.get("reward") or metrics.get("reward")
    wdl = learning.get("terminal_wdl")
    if wdl is None:
        wdl = metrics.get("terminal_wdl")
    reward_text = reward
    if reward is True:
        reward_text = "terminal_wdl"
    return {
        "algorithm": display(algorithm),
        "simulations": display(simulations),
        "gumbel_candidates": display(candidates),
        "reward": display(reward_text),
        "terminal_wdl": display(wdl if wdl is not None else (reward_text if reward_text == "terminal_wdl" else None)),
    }


def _updates(metrics, roster):
    games = metrics.get("complete_training_games")
    updates = metrics.get("formal_policy_updates")
    names = list(roster or ())
    if isinstance(games, dict) or isinstance(updates, dict):
        keys = list(dict.fromkeys(names + list(_mapping(games)) + list(_mapping(updates))))
        rows = []
        for key in keys:
            game_value = games.get(key) if isinstance(games, dict) else games
            update_value = updates.get(key) if isinstance(updates, dict) else updates
            if game_value in (None, 0, "0"):
                game_text = MISSING if game_value in (None, "") else f"{MISSING}（{game_value}）"
            else:
                game_text = display(game_value)
            if update_value in (None, 0, "0"):
                update_text = MISSING if update_value in (None, "") else f"{MISSING}（{update_value}）"
            else:
                update_text = display(update_value)
            rows.append((key, game_text, update_text))
        return rows
    if names:
        game_text = MISSING if games in (None, 0, "0") else display(games)
        update_text = MISSING if updates in (None, 0, "0") else display(updates)
        return [(",".join(names), game_text, update_text)]
    return [("roster", MISSING, MISSING)]


def _candidate_section(snapshot, metrics):
    request = _mapping(snapshot.get("request"))
    include = request.get("include_candidates")
    confirmation = _mapping(snapshot.get("confirmation"))
    if confirmation.get("include_candidates") is not None:
        include = confirmation.get("include_candidates")
    skipped = []
    for name in candidate_stages():
        state = stage_state(snapshot, name)
        if state["status"] == "skipped" or state["skip_reason"]:
            skipped.append((name, state["skip_reason"] or "user include_candidates=false"))
    changes = (
        metrics.get("candidate_changes")
        or _nested(metrics, "candidates", "changes")
        or _nested(metrics, "candidate_search", "changes")
        or []
    )
    decision = (
        metrics.get("acceptance_decision")
        or _nested(metrics, "candidates", "decision")
        or _nested(metrics, "build_confirmation", "decision")
        or _nested(metrics, "build_audit", "decision")
    )
    return include, skipped, changes if isinstance(changes, list) else [], decision


def _audit_block(snapshot, metrics):
    request_sha = snapshot.get("request_sha256")
    audit = _mapping(metrics.get("final_audit") or metrics.get("audit") or _nested(metrics, "build_audit") or {})
    current = audit.get("request_sha256") or audit.get("basis_request_sha256")
    candidate = audit.get("candidate_sha256") or audit.get("candidate_identity")
    artifacts = audit.get("artifact_sha256") or audit.get("artifacts")
    match = None
    if present(request_sha) and present(current):
        match = request_sha == current
    return {
        "request_sha256": display(request_sha),
        "audit_request_sha256": display(current),
        "candidate_identity": display(candidate),
        "artifact_hashes": display(artifacts if not isinstance(artifacts, list) else _json(artifacts)),
        "matches_current_basis": MISSING if match is None else ("一致" if match else "不一致"),
    }


def missing_report_evidence(flow):
    """Required semantic data, separate from successful step execution."""
    snapshot = snapshot_of(flow)
    request = snapshot['request']
    roster = request['roster']
    metrics = collect_metrics(snapshot)
    missing = []
    learning = _gumbel_block(metrics)
    if any(learning[key] == MISSING for key in ('algorithm', 'simulations', 'gumbel_candidates', 'terminal_wdl')):
        missing.append('搜索/WDL方法身份')
    identities = _identity_map(metrics)
    for key in roster:
        if any(not identities.get(key, {}).get(variant) or identities[key][variant] == MISSING for variant in ('initial', 'selected', 'latest')):
            missing.append(key + ' initial/selected/latest身份')
    expected = {(row, col) for row in roster for col in roster}
    for mode, _label in EVAL_MODES:
        block = _mode_block(metrics, mode)
        cells = block.get('cells') or block.get('matrix') or []
        if isinstance(cells, dict):
            cells = list(cells.values())
        found = set()
        valid = isinstance(cells, list)
        for cell in cells if valid else []:
            row = _mapping(cell)
            pair = (row.get('row'), row.get('column'))
            if pair in found or pair not in expected:
                valid = False
            found.add(pair)
            counts = [row.get(key) for key in ('planned', 'completed', 'first_wins', 'second_wins', 'draws')]
            if any(type(value) is not int or value < 0 for value in counts):
                valid = False
                continue
            if counts[0] <= 0 or counts[0] != counts[1] or sum(counts[2:]) != counts[1] or row.get('errors', 0) or row.get('truncated', 0):
                valid = False
        if not valid or found != expected:
            missing.append(mode + '实际NxN胜负分母')
    evaluation = _mapping(metrics.get('evaluation') or metrics.get('eval'))
    for key, label in EVAL_CONTRASTS:
        if not present(evaluation.get(key) or metrics.get(key)):
            missing.append(label)
    registered = {item['path']: item['sha256'] for name in stage_order()
                  for item in stage_state(snapshot, name)['artifacts']}
    analysis = _mapping(metrics.get('analysis') or metrics.get('detail'))
    for topic in EVIDENCE_TOPICS:
        block = _mapping(analysis.get(topic))
        tables = block.get('artifacts') or []
        proven = bool(tables) and all(isinstance(item, dict) and registered.get(item.get('path')) == item.get('sha256') for item in tables)
        if type(block.get('games')) is not int or block['games'] <= 0:
            proven = False
        if _topic_status(metrics, topic) == MISSING or not proven:
            missing.append(TOPIC_LABELS[topic])
    if request['include_candidates']:
        _include, _skipped, changes, decision = _candidate_section(snapshot, metrics)
        if not changes or not decision:
            missing.append('候选变化及接受/保留理由')
    return missing


def _matrix_lines(metrics, mode, roster):
    block = _mode_block(metrics, mode)
    values = block.get('cells') or block.get('matrix') or []
    if isinstance(values, dict):
        values = list(values.values())
    cells = {(x.get('row'), x.get('column')): x for x in values if isinstance(x, dict)}
    lines = ['| 先手＼后手 | ' + ' | '.join(roster) + ' |', '| --- | ' + ' | '.join('---' for _ in roster) + ' |']
    for row in roster:
        rendered = []
        for col in roster:
            cell = cells.get((row, col)) or {}
            total = cell.get('completed')
            planned = cell.get('planned')
            if type(total) is not int or total <= 0 or type(planned) is not int or total != planned or cell.get('errors') or cell.get('truncated'):
                rendered.append(MISSING)
            else:
                rendered.append(winrate(cell.get('first_wins'), total))
        lines.append('| ' + row + ' | ' + ' | '.join(rendered) + ' |')
    return lines


def render_flow_report(flow):
    snapshot = deepcopy(snapshot_of(flow))
    request = _mapping(snapshot.get("request"))
    metrics = collect_metrics(snapshot)
    roster = list(request.get("roster") or ())
    deadline = request.get("hard_deadline") or snapshot.get("hard_deadline")
    epoch = snapshot.get("hard_deadline_epoch") or as_epoch(deadline)
    include, skipped, changes, decision = _candidate_section(snapshot, metrics)
    learning = _gumbel_block(metrics)
    identities = _identity_map(metrics)
    audit = _audit_block(snapshot, metrics)
    workflow = snapshot.get("workflow_complete")
    quality = snapshot.get("quality_approved")
    complete_report = snapshot.get("complete_report")
    directory = getattr(flow, "directory", None) or snapshot.get("directory")
    confirmation = _mapping(snapshot.get("confirmation"))
    lines = [
        "# 完整训练流程报告",
        "",
        "本报告只陈述流程账本中的证据。workflow_complete 不表示质量批准，也不批准网站模型。合成夹具与流程完成度不能证明 CUDA/Gumbel 训练质量。缺失项记为未完成，不以 0% 或 0 次冒充结果。",
        "",
        "## 范围与截止",
        "",
        f"- 运行目录：{display(directory)}",
        f"- 请求摘要：{display(snapshot.get('request_sha256'))}",
        f"- 硬截止：{display(deadline)}（epoch {display(epoch)}）",
        f"- 名单：{display(', '.join(roster) if roster else None)}",
        f"- 包含候选配牌：{display(include)}（指 32 张构筑变更，不是起手换牌）",
        f"- 确认来源：{display(confirmation.get('source'))} / 引用 {display(confirmation.get('reference'))}",
        f"- workflow_complete：{display(workflow)}",
        f"- quality_approved：{display(quality)}",
        f"- complete_report：{display(complete_report)}",
        "",
        "## 实际构筑张数表",
        "",
    ]
    builds = _mapping(request.get("builds"))
    if not builds:
        lines += [MISSING, ""]
    else:
        lines += ["| 套牌 | 角色 | 实际张数 | 卡牌计数 |", "| --- | --- | ---: | --- |"]
        for key in roster or list(builds):
            build = _mapping(builds.get(key))
            total, counts = _count_table(build)
            roles = display(",".join(build.get("character_ids") or []))
            count_text = _json(counts) if counts else MISSING
            total_text = str(total) if total else MISSING
            lines.append(f"| {key} | {roles} | {total_text} | {count_text} |")
        lines.append("")
    lines += ["## 候选变更与接受决定", ""]
    if include is False:
        reason = skipped[0][1] if skipped else "用户确认不包含候选配牌"
        lines += [f"候选阶段按用户确认跳过：{reason}。未执行搜索不得记为通过。", ""]
        if skipped:
            lines += ["| 阶段 | 跳过原因 |", "| --- | --- |"]
            lines += [f"| {name} | {item} |" for name, item in skipped]
            lines.append("")
    elif include is True:
        if skipped and not changes:
            lines += [f"{MISSING}：候选已授权，但阶段被跳过或缺少变更记录。", ""]
        if changes:
            lines += ["| 套牌 | 换出 | 换入 | 相对父版本 | 决定 |", "| --- | --- | --- | --- | --- |"]
            for item in changes:
                row = _mapping(item)
                lines.append(
                    "| {build} | {removed} | {added} | {parent} | {decision} |".format(
                        build=display(row.get("build") or row.get("preset") or row.get("key")),
                        removed=display(row.get("removed") or row.get("out")),
                        added=display(row.get("added") or row.get("in_cards") or row.get("added_cards")),
                        parent=display(row.get("parent") or row.get("parent_id")),
                        decision=display(row.get("decision") or decision),
                    )
                )
            lines.append("")
        else:
            lines += [f"候选变更记录：{MISSING}", ""]
        lines.append(f"接受决定：{display(decision)}")
        lines.append("")
    else:
        lines += [f"是否包含候选配牌：{MISSING}", ""]
    lines += [
        "## Gumbel / 终局 WDL / 更新身份",
        "",
        f"- 搜索算法：{learning['algorithm']}",
        f"- Gumbel 模拟次数：{learning['simulations']}",
        f"- 根候选上限：{learning['gumbel_candidates']}",
        f"- 奖励：{learning['reward']}",
        f"- 终局 WDL：{learning['terminal_wdl']}",
        "",
        "| 套牌 | 完整训练局数 | 正式策略更新 |",
        "| --- | --- | --- |",
    ]
    for key, games, updates in _updates(metrics, roster):
        lines.append(f"| {key} | {games} | {updates} |")
    lines += ["", "## 初始 / 选定 / 最新模型身份", ""]
    if not identities:
        lines += [MISSING, ""]
    else:
        lines += ["| 套牌 | initial | selected | latest |", "| --- | --- | --- | --- |"]
        for key in roster or list(identities):
            row = identities.get(key) or {}
            lines.append(
                "| {key} | {initial} | {selected} | {latest} |".format(
                    key=key,
                    initial=display(row.get("initial")),
                    selected=display(row.get("selected")),
                    latest=display(row.get("latest")),
                )
            )
        lines.append("selected 不是自动的 latest；未核验数组变化时不得称为新模型。")
        lines.append("")
    lines += ["## N×N 评估分母", "", "| 模式 | 计划格数 | 完成格数 | 覆盖 |", "| --- | ---: | ---: | --- |"]
    saw_mode = False
    for mode, label in EVAL_MODES:
        block = _mode_block(metrics, mode)
        planned, completed = _planned_completed(block)
        if planned is not None or completed is not None:
            saw_mode = True
        lines.append(f"| {label} | {display(planned)} | {display(completed)} | {coverage(planned, completed)} |")
    if not saw_mode:
        planned, completed = metrics.get("planned_cells"), metrics.get("completed_cells")
        lines.append(f"| 未分纯网络/搜索 | {display(planned)} | {display(completed)} | {coverage(planned, completed)} |")
    lines += ["", "行＝先手、列＝后手，格内为先手胜率。缺格为未完成，不得填 0% 或 50%。", ""]
    for mode, label in EVAL_MODES:
        lines += ['### ' + label + '先手胜率', ''] + _matrix_lines(metrics, mode, roster) + ['']
    lines += ["| 对照 | 状态 |", "| --- | --- |"]
    evaluation = _mapping(metrics.get("evaluation") or metrics.get("eval"))
    for key, label in EVAL_CONTRASTS:
        block = evaluation.get(key) or metrics.get(key)
        if block is None:
            lines.append(f"| {label} | {MISSING} |")
            continue
        planned, completed = _planned_completed(_mapping(block))
        if planned is not None or completed is not None:
            lines.append(f"| {label} | {coverage(planned, completed)} |")
        else:
            lines.append(f"| {label} | {'已记录' if present(block) else MISSING} |")
    lines += ["", "## 逐卡 / 异能者 / 召唤 / 使用 / 释放 / 机制", "", "| 专题 | 状态 |", "| --- | --- |"]
    for topic in EVIDENCE_TOPICS:
        lines.append(f"| {TOPIC_LABELS[topic]} | {_topic_status(metrics, topic)} |")
    lines += ["", "## 原始记录 / 公开录像 / 模型交付", "", "| 交付 | 状态 |", "| --- | --- |"]
    for topic in DELIVERY_TOPICS:
        lines.append(f"| {DELIVERY_LABELS[topic]} | {_delivery_status(metrics, topic)} |")
    lines += ['', '### 已登记数据与报告', '', '| 步骤 | 产物 | SHA-256 |', '| --- | --- | --- |']
    for name in stage_order():
        for item in stage_state(snapshot, name)['artifacts']:
            path = Path(str(directory or '')) / item['path']
            label = Path(item['path']).name.replace('[', '\\[').replace(']', '\\]')
            lines.append(f"| {name} | [{label}](<{path}>) | {item['sha256']} |")
    lines += [
        "",
        "JSON 清单不能替代 .npz/.pt 数值文件，统计表不能替代原始评估记录，公开录像必须过滤隐藏信息。",
        "",
        "## 阶段尝试与缺失证据",
        "",
        "| 阶段 | 状态 | 尝试次数 | 结论 | 缺失 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    missing_any = False
    for name in stage_order():
        state = stage_state(snapshot, name)
        attempt_n = len(state["attempts"]) if state["attempts"] else (1 if state["status"] not in ("pending", None) else 0)
        skipped_stage = state["status"] == "skipped" or (include is False and name in candidate_stages())
        if skipped_stage:
            missing = "按用户确认跳过" if include is False else display(state["skip_reason"])
            verdict = "skipped"
        else:
            needed = state["status"] not in ("passed",) or not present(state["verdict"])
            if name == "confirmation":
                needed = confirmation.get("source") != "user"
            missing = "" if not needed else MISSING
            if missing == MISSING:
                missing_any = True
            verdict = display(state["verdict"] or state["status"])
        lines.append(f"| {name} | {display(state['status'])} | {attempt_n} | {verdict} | {missing or '无'} |")
        if state["summary"]:
            lines.append(f"|  | 摘要 |  | {state['summary']} |  |")
    gaps = missing_report_evidence(flow)
    if gaps:
        lines += ['报告语义缺项：' + '、'.join(gaps), '']
    lines += [
        "",
        "## 流程完整性与质量批准",
        "",
        f"- 流程账本 workflow_complete={display(workflow)}，质量批准 quality_approved={display(quality)}，完整报告 complete_report={display(complete_report)}。",
        "- 三者相互独立。子 worker 不得把质量批准或模型晋升写入结论。",
        "- Fresh Reviewer 只能提供只读证据，不能标记完整训练已接受、不能晋升模型、不能改范围。",
    ]
    if quality and missing_any:
        lines.append("- 记录的 quality_approved 与缺失证据矛盾，不能视为质量批准。")
    if workflow and missing_any:
        lines.append(f"- workflow_complete 不能填补 {MISSING} 的训练或评估格。")
    lines += [
        "",
        "## 最终审计依据",
        "",
        f"- 当前请求摘要：{audit['request_sha256']}",
        f"- 审计绑定请求摘要：{audit['audit_request_sha256']}",
        f"- 候选身份：{audit['candidate_identity']}",
        f"- 产物哈希：{audit['artifact_hashes']}",
        f"- 与当前依据是否一致：{audit['matches_current_basis']}",
        "",
        "本报告不回写旧冻结运行，也不修改网站模型。",
        "",
    ]
    return "\n".join(lines)


def write_flow_report(flow, path=None):
    snapshot = snapshot_of(flow)
    root = run_root_of(flow, snapshot).resolve()
    if root == REPO_ROOT.resolve():
        raise ValueError("refusing to write report at repository root")
    dest = root / DEFAULT_REPORT_NAME if path is None else Path(path)
    if not dest.is_absolute():
        dest = root / dest
    if dest.is_symlink() or dest.parent.is_symlink():
        raise ValueError("symlink report path rejected")
    dest = dest.resolve()
    if not _inside_root(dest, root):
        raise ValueError("report path outside run root")
    for stage in snapshot.get('stages', {}).values():
        for attempt in stage.get('attempts', []):
            for artifact in (attempt.get('result') or {}).get('artifacts', []):
                if (root / artifact['path']).resolve() == dest:
                    raise ValueError('registered evidence is immutable; write a new report path')
    text = render_flow_report(flow)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(dest)
    digest = sha256(text.encode("utf-8")).hexdigest()
    relative = os.path.relpath(dest, root).replace(chr(92), "/")
    return {"path": relative, "sha256": digest}
