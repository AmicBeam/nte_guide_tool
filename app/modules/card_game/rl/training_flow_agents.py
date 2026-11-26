"""Stage-agent dispatch for the complete-training flow. No provider call on import."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

SHANGHAI = timezone(timedelta(hours=8))
CONTRACT_STAGE_ORDER = (
    'confirmation', 'freeze', 'preflight', 'base_training', 'candidate_search',
    'baseline_adaptation', 'candidate_adaptation', 'build_confirmation',
    'build_audit', 'freeze_selection', 'evaluation', 'analysis', 'report',
    'delivery',
)
CONTRACT_CANDIDATE_STAGES = (
    'candidate_search', 'baseline_adaptation', 'candidate_adaptation',
    'build_confirmation', 'build_audit',
)
VALID_VERDICTS = ('passed', 'failed', 'blocked')
REPO_ROOT = Path(__file__).resolve().parents[4]
WORKER_RELATIVE = Path('skills/grok-delegation/scripts/run-grok-worker.sh')
SENSITIVE_PARTS = (
    'password', 'secret', 'token', 'cookie', 'credential', 'authorization',
    'private_key', 'ssh', 'api_key', 'access_key', '.env', 'cookiejar',
)
PRIVATE_EXPERIENCE_PARTS = (
    'experience', 'replay_buffer', 'trajectory', 'hidden_hand', 'private_history',
    'optimizer_state', 'rng_state', 'deck_order', 'seed_sequence', 'private_replay',
)
STAGE_EVIDENCE = {
    'freeze': 'Frozen source/rule/schema identities and SHA list inside output_directory. No credentials.',
    'preflight': 'CPU/CUDA consistency, public observation, legal actions. Failure must not skip ahead.',
    'base_training': 'Gumbel + terminal WDL only. Positive complete_training_games and formal_policy_updates for every roster. Record initial/latest identities.',
    'candidate_search': 'Actual 32-card count tables and candidate vs parent/original changes. Candidate means build change, not opening mulligan.',
    'baseline_adaptation': 'Fair original-build adaptation: positive formal_policy_updates, same seeds/opponents as the candidate branch.',
    'candidate_adaptation': 'Fair candidate adaptation with the same budget/seeds/opponents. Zero updates on either branch is not success.',
    'build_confirmation': 'Independent holdout decision: accept, retain_baseline, or insufficient evidence. Not the final quality approval.',
    'build_audit': 'Audit hashes for current candidate, request digest, and artifacts. Stale hashes are not current evidence.',
    'freeze_selection': 'initial/selected/latest identities. selected is never automatically latest.',
    'evaluation': 'NxN pure-network and search matrices with planned/completed cell denominators; baseline, new-old, mulligan, varied opponents; raw evidence hashes. Missing cells are 未完成, not 0%.',
    'analysis': 'Card/esper/summon/use/release/mechanism tables with denominators. Missing is 未完成.',
    'report': 'Chinese report covering required sections. Do not set complete_report or quality_approved.',
    'delivery': 'Actual numerical models + raw records + public replays inside the run root. Never JSON-only. Do not rewrite website models.',
}


def stage_order():
    try:
        from app.modules.card_game.rl.training_flow import STAGE_ORDER
        return tuple(STAGE_ORDER)
    except Exception:
        return CONTRACT_STAGE_ORDER


def candidate_stages():
    try:
        from app.modules.card_game.rl.training_flow import CANDIDATE_STAGES
        return tuple(CANDIDATE_STAGES)
    except Exception:
        return CONTRACT_CANDIDATE_STAGES


def grok_worker_script():
    home = Path(os.environ.get('CODEX_HOME') or (Path.home() / '.codex'))
    return home / WORKER_RELATIVE


def grok_worker_argv(workdir, prompt):
    return [str(grok_worker_script()), str(workdir), str(prompt)]


def snapshot_of(flow):
    if hasattr(flow, 'snapshot') and callable(flow.snapshot):
        return flow.snapshot()
    if isinstance(flow, dict):
        return flow
    raise TypeError('flow must provide snapshot()')


def run_root_of(flow, snapshot=None):
    for name in ('directory', 'root', 'path', '_root'):
        value = getattr(flow, name, None)
        if value:
            return Path(value)
    data = snapshot if snapshot is not None else snapshot_of(flow)
    for key in ('directory', 'root', 'path', 'run_directory'):
        value = data.get(key) if isinstance(data, dict) else None
        if value:
            return Path(value)
    raise ValueError('run root missing')


def as_epoch(value):
    if value is None or value == '':
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError('datetime must have an explicit timezone')
        return value.timestamp()
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('timestamp must have an explicit timezone')
    return parsed.timestamp()


def remaining_timeout(assignment, snapshot, now, timeout=None):
    limits = []
    if timeout is not None:
        limits.append(float(timeout))
    for raw in (
        (assignment or {}).get('deadline'),
        (snapshot or {}).get('hard_deadline_epoch'),
        (snapshot or {}).get('hard_deadline'),
        ((snapshot or {}).get('request') or {}).get('hard_deadline'),
    ):
        epoch = as_epoch(raw)
        if epoch is not None:
            limits.append(epoch - float(now))
    if not limits:
        return None
    return max(0.0, min(limits))


def _sensitive(key):
    text = str(key).lower()
    return any(part in text for part in SENSITIVE_PARTS + PRIVATE_EXPERIENCE_PARTS)


def public_metrics(value, *, depth=0):
    if depth > 6:
        return 'unexpanded'
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if _sensitive(key):
                continue
            if key in ('quality_approved', 'workflow_complete', 'complete_report'):
                continue
            out[str(key)] = public_metrics(item, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        if len(value) > 32:
            return {'count': len(value), 'preview': [public_metrics(x, depth=depth + 1) for x in value[:8]]}
        return [public_metrics(item, depth=depth + 1) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 4000:
            return value[:400] + '...'
        return value
    return str(value)


def _build_scope(snapshot):
    request = snapshot.get('request') or {}
    builds = {}
    for key, build in (request.get('builds') or {}).items():
        cards = list((build or {}).get('card_ids') or ())
        builds[str(key)] = {
            'character_ids': list((build or {}).get('character_ids') or ()),
            'card_count': len(cards),
            'card_counts': dict(Counter(cards)),
            'card_ids': cards,
        }
    confirmation = snapshot.get('confirmation') or {}
    return {
        'request_sha256': snapshot.get('request_sha256'),
        'include_candidates': request.get('include_candidates'),
        'hard_deadline': request.get('hard_deadline') or snapshot.get('hard_deadline'),
        'hard_deadline_epoch': snapshot.get('hard_deadline_epoch') or as_epoch(snapshot.get('hard_deadline')),
        'roster': list(request.get('roster') or ()),
        'builds': builds,
        'mutable_characters': request.get('mutable_characters'),
        'default_agent': request.get('default_agent') or 'grok',
        'confirmation': {
            'source': confirmation.get('source'),
            'reference': confirmation.get('reference'),
            'include_candidates': confirmation.get('include_candidates'),
        },
    }


def _stage_blob(snapshot, stage_id):
    return ((snapshot.get('stages') or {}).get(stage_id) or {})


def _last_result(stage_info):
    if stage_info.get('invalidated'):
        return {}
    attempts = stage_info.get('attempts')
    if isinstance(attempts, list) and attempts:
        last = attempts[-1] if isinstance(attempts[-1], dict) else {}
        return last.get('result') or last
    return stage_info.get('result') or {}


def upstream_evidence(snapshot, stage_id):
    order = list(stage_order())
    index = order.index(stage_id) if stage_id in order else len(order)
    rows = []
    for name in order[:index]:
        info = _stage_blob(snapshot, name)
        result = _last_result(info) if isinstance(info, dict) else {}
        artifacts = []
        for item in (result.get('artifacts') or info.get('artifacts') or ()):
            if isinstance(item, dict) and item.get('path') and item.get('sha256'):
                artifacts.append({'path': item['path'], 'sha256': item['sha256']})
        rows.append({
            'stage_id': name,
            'status': info.get('status') or result.get('status') or 'pending',
            'skip_reason': info.get('skip_reason') or info.get('reason'),
            'verdict': result.get('verdict'),
            'summary': result.get('summary'),
            'artifacts': artifacts,
            'metrics': public_metrics(result.get('metrics') or info.get('metrics') or {}),
        })
    return rows


def _candidate_policy(snapshot):
    request = snapshot.get('request') or {}
    include = request.get('include_candidates')
    confirmation = snapshot.get('confirmation') or {}
    if confirmation.get('include_candidates') is not None:
        include = confirmation.get('include_candidates')
    skipped = {}
    for name in candidate_stages():
        info = _stage_blob(snapshot, name)
        reason = info.get('skip_reason') or info.get('reason')
        if (info.get('status') == 'skipped') or reason:
            skipped[name] = reason or 'skipped'
    return {'include_candidates': include, 'skipped': skipped}


def build_stage_prompt(flow, assignment, now=None):
    snapshot = snapshot_of(flow)
    assignment = dict(assignment or {})
    stage_id = assignment.get('stage_id')
    if not stage_id:
        raise ValueError('assignment.stage_id required')
    now = time.time() if now is None else as_epoch(now) or time.time()
    remaining = remaining_timeout(assignment, snapshot, now)
    try:
        root = str(run_root_of(flow, snapshot))
    except ValueError:
        root = None
    payload = {
        'role': 'training-flow evidence worker',
        'agent': assignment.get('agent') or 'grok',
        'stage_id': stage_id,
        'attempt': assignment.get('attempt'),
        'token': assignment.get('token'),
        'request_sha256': assignment.get('request_sha256') or snapshot.get('request_sha256'),
        'deadline': assignment.get('deadline'),
        'remaining_seconds': remaining,
        'run_root': root,
        'read_only_repository': str(REPO_ROOT),
        'output_directory': assignment.get('output_directory'),
        'immutable_scope': _build_scope(snapshot),
        'candidate_policy': _candidate_policy(snapshot),
        'upstream_evidence': upstream_evidence(snapshot, stage_id),
        'required_evidence': STAGE_EVIDENCE.get(stage_id, 'Produce hash-verified artifacts for this evidence unit.'),
        'learning_stack': 'Keep Gumbel search and terminal WDL. Do not substitute PPO or another algorithm.',
        'structured_result': {
            'stage_id': stage_id,
            'attempt': assignment.get('attempt'),
            'token': assignment.get('token'),
            'request_sha256': assignment.get('request_sha256'),
            'verdict': 'passed|failed|blocked',
            'summary': 'nonempty string',
            'artifacts': [{'path': 'relative to run root', 'sha256': 'hex'}],
            'metrics': {'object': 'counts, identities, planned/completed denominators'},
            'external_action_request': 'optional; emit and stop instead of reading credentials',
        },
    }
    rules = [
        'You are an explicit-only worker for one training-flow evidence unit, not an independent OMH stage owner.',
        'STAGE_ORDER names required evidence units. The persistent Main Agent owns decisions, remediation, and final acceptance.',
        'Edit only assignment.output_directory inside the run root. Preserve concurrent changes.',
        'Bind output ownership, original deadline, request_sha256, and upstream artifact hashes in every claim.',
        'Never access or transmit secrets, .env, auth files, private keys, cookies, databases, private accounts, private histories, private training experience, model binaries outside output ownership, or repository-external symlinks.',
        'Never commit, push, publish, deploy, send messages, change authentication, delete material data, use danger-full-access, spawn further agents, or change frozen scope.',
        'Never read SSH credentials or private keys. If a bounded external action is required, emit external_action_request and stop; the primary Agent supplies the handler.',
        'Stay within worker permission boundaries. Do not mark workflow_complete, quality_approved, or complete_report. Do not promote a model or rewrite website models.',
        'Worker results only supply evidence. Exit code 0 is not success. Return one JSON object with verdict/summary/artifacts/metrics.',
        'After the original hard cutoff, only collect existing files and report incomplete evidence. Never start new learning.',
        'Missing evidence is 未完成, not 0 or fabricated quality.',
    ]
    return (
        'Complete-training flow stage assignment.\n'
        + '\n'.join(f'- {line}' for line in rules)
        + '\n\nASSIGNMENT_JSON=\n'
        + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + '\n'
    )


def extract_json_object(text):
    if not text:
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != '{':
            continue
        try:
            obj, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and not text[end:].strip() and obj.get('verdict') in VALID_VERDICTS and str(obj.get('summary') or '').strip():
            return obj
    return None


def _normalize_artifacts(items):
    rows = []
    for item in items or ():
        if not isinstance(item, dict):
            continue
        path = item.get('path')
        digest = item.get('sha256')
        if path and digest:
            rows.append({'path': str(path).replace(chr(92), '/'), 'sha256': str(digest)})
    return rows


def normalize_attempt_result(payload, *, returncode=None, timed_out=False, launched=True):
    data = dict(payload or {})
    verdict = data.get('verdict')
    summary = str(data.get('summary') or '').strip()
    if verdict not in VALID_VERDICTS or not summary:
        verdict = 'blocked' if timed_out or not launched else 'failed'
        if not summary:
            if timed_out:
                summary = 'child timed out before a structured result'
            elif not launched:
                summary = 'original deadline elapsed; child not launched'
            elif returncode == 0:
                summary = 'structured result missing verdict or nonempty summary; exit 0 is not success'
            else:
                summary = 'child produced no structured result'
    if timed_out or not launched:
        verdict = 'blocked'
        summary = 'child timed out or was not launched; no successful settlement'
    elif returncode not in (None, 0):
        verdict = 'failed'
        summary = 'child exited unsuccessfully; ' + summary
    metrics = public_metrics(data.get('metrics') or {})
    if not isinstance(metrics, dict):
        metrics = {'value': metrics}
    metrics['returncode'] = returncode
    metrics['timed_out'] = bool(timed_out)
    metrics['launched'] = bool(launched)
    metrics['exit_zero_is_not_success'] = True
    for banned in ('quality_approved', 'workflow_complete', 'complete_report'):
        if banned in data:
            metrics['child_overreach_' + banned] = data.get(banned)
    result = {
        'verdict': verdict,
        'summary': summary,
        'artifacts': _normalize_artifacts(data.get('artifacts')),
        'metrics': metrics,
    }
    for key in ('stage_id', 'attempt', 'token', 'request_sha256'):
        if key in data:
            result[key] = data[key]
    if data.get('external_action_request') is not None:
        result['external_action_request'] = data.get('external_action_request')
    return result


def parse_structured_result(stdout, stderr='', returncode=0, timed_out=False, launched=True):
    payload = extract_json_object(stdout if isinstance(stdout, str) else str(stdout or ''))
    if payload is None:
        payload = {'metrics': {'stderr_preview': str(stderr or '')[:500]}}
        if timed_out:
            payload['verdict'] = 'blocked'
            payload['summary'] = 'child timed out before a structured result'
        elif returncode == 0:
            payload['summary'] = 'no structured JSON result; exit 0 is not success'
        else:
            payload['summary'] = f'child exited {returncode} without structured JSON'
    elif stderr:
        metrics = dict(payload.get('metrics') or {})
        metrics.setdefault('stderr_preview', str(stderr)[:500])
        payload['metrics'] = metrics
    return normalize_attempt_result(payload, returncode=returncode, timed_out=timed_out, launched=launched)


def run_isolated_command(argv, *, timeout, cwd):
    if not isinstance(argv, (list, tuple)) or not argv:
        raise ValueError('argv-safe runner requires a nonempty argument list')
    command = [str(part) for part in argv]
    proc = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=None if timeout is None else float(timeout))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError, PermissionError, OSError):
            proc.kill()
        stdout, stderr = proc.communicate()
    decode = lambda blob: blob.decode('utf-8', 'replace') if isinstance(blob, bytes) else (blob or '')
    return {
        'returncode': proc.returncode,
        'stdout': decode(stdout),
        'stderr': decode(stderr),
        'timed_out': timed_out,
    }


def isolated_grok_runner(prompt, *, workdir, timeout, assignment=None, **_ignored):
    return run_isolated_command(grok_worker_argv(workdir, prompt), timeout=timeout, cwd=workdir)


def _coerce_runner_output(raw):
    if isinstance(raw, dict) and raw.get('verdict') in VALID_VERDICTS and 'stdout' not in raw:
        return normalize_attempt_result(raw, returncode=raw.get('returncode'), timed_out=bool(raw.get('timed_out')))
    if isinstance(raw, dict):
        return parse_structured_result(
            raw.get('stdout') or '',
            raw.get('stderr') or '',
            raw.get('returncode', 0),
            timed_out=bool(raw.get('timed_out')),
            launched=raw.get('launched', True),
        )
    if hasattr(raw, 'returncode'):
        return parse_structured_result(
            getattr(raw, 'stdout', '') or '',
            getattr(raw, 'stderr', '') or '',
            getattr(raw, 'returncode', 1),
        )
    raise TypeError('runner must return structured result or process output')


def bind_stage_handlers(flow, runner=None, timeout=None, external_handler=None):
    owned = flow

    def handler(*args, **kwargs):
        if len(args) == 1:
            target, assignment = owned, args[0]
        elif len(args) == 2:
            target, assignment = args
        else:
            raise TypeError('stage handler expects assignment or (flow, assignment)')
        return run_stage_agent(
            target, assignment, runner=runner, timeout=kwargs.get('timeout', timeout),
            now=kwargs.get('now'), external_handler=external_handler,
        )

    return {stage: handler for stage in stage_order() if stage != 'confirmation'}


def run_stage_agent(flow, assignment, runner=None, timeout=None, now=None, external_handler=None):
    snapshot = snapshot_of(flow)
    assignment = dict(assignment or {})
    now_epoch = time.time() if now is None else as_epoch(now) or time.time()
    remaining = remaining_timeout(assignment, snapshot, now_epoch, timeout)
    prompt = build_stage_prompt(flow, assignment, now=now_epoch)
    if remaining is not None and remaining <= 0:
        if assignment.get('stage_id') == 'report' and hasattr(flow, 'audit_basis'):
            from .training_flow_report import write_flow_report
            artifact = write_flow_report(flow, Path(assignment['output_directory']) / 'incomplete-report.md')
            return {'verdict': 'passed', 'summary': '截止后仅由主agent本地整理已有证据，未启动子进程或学习。',
                    'artifacts': [artifact], 'metrics': {'post_cutoff_collection': True, 'launched': False},
                    **{key: assignment[key] for key in ('stage_id', 'attempt', 'token', 'request_sha256')}}
        return normalize_attempt_result(
            {'verdict': 'blocked', 'summary': 'original deadline elapsed; child not launched'},
            launched=False, timed_out=False,
        )
    transport = isolated_grok_runner if runner is None else runner
    if runner is None and assignment.get('agent', 'grok') != 'grok':
        raise ValueError('explicit non-Grok agent requires its configured runner')
    if hasattr(flow, 'audit_basis'):
        current = snapshot['stages'][assignment['stage_id']]
        attempt = current['attempts'][-1] if current['attempts'] else {}
        if current['status'] != 'running' or any(assignment.get(key) != attempt.get(key) for key in ('token', 'attempt', 'request_sha256')):
            raise ValueError('stale or inactive agent assignment')
        output = Path(assignment['output_directory'])
        expected_output = Path(flow.directory) / attempt['output_directory']
        if output.is_symlink() or output.resolve() != expected_output.resolve():
            raise ValueError('assignment output ownership changed')
        claim = output / 'agent-dispatch.json'
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
        try:
            fd = os.open(claim, flags, 0o600)
        except FileExistsError as exc:
            raise ValueError('agent attempt already dispatched; settle and start a new attempt') from exc
        with os.fdopen(fd, 'w') as f:
            json.dump({key: assignment[key] for key in ('stage_id', 'attempt', 'token', 'request_sha256')}, f)
    raw = transport(prompt, workdir=str(Path(assignment['output_directory']).resolve()), timeout=remaining, assignment=assignment)
    result = _coerce_runner_output(raw)
    if hasattr(flow, 'audit_basis'):
        if result['verdict'] == 'passed' and any(result.get(key) != assignment.get(key) for key in ('stage_id', 'attempt', 'token', 'request_sha256')):
            return normalize_attempt_result({'verdict': 'failed', 'summary': 'worker result is missing or mismatches current attempt provenance'})
    request = result.get('external_action_request')
    if request is not None:
        if external_handler is None:
            result['metrics']['external_action_pending_primary'] = True
            if result['verdict'] == 'passed':
                result['verdict'] = 'blocked'
                result['summary'] = 'external action requested; child cannot read credentials'
        else:
            handled = external_handler(deepcopy(request))
            result['metrics']['external_action_result'] = public_metrics(handled)
    return result
