"""Durable lightweight complete-training flow. Stdlib only; never trains."""
from contextlib import contextmanager
import ast
from collections import Counter
from datetime import datetime, timezone
try:
    import fcntl
except ImportError:  # Windows training hosts
    fcntl = None
    import msvcrt
from hashlib import sha256
import json
import gzip
import inspect
import os
import re
import struct
from pathlib import Path
import threading
import zipfile
from uuid import uuid4

SCHEMA = 'duel_v2_complete_training_flow_v1'
STAGE_ORDER = (
    'confirmation', 'freeze', 'preflight', 'base_training', 'candidate_search',
    'baseline_adaptation', 'candidate_adaptation', 'build_confirmation',
    'build_audit', 'freeze_selection', 'evaluation', 'analysis', 'report',
    'delivery',
)
CANDIDATE_STAGES = (
    'candidate_search', 'baseline_adaptation', 'candidate_adaptation',
    'build_confirmation', 'build_audit',
)
_POST_CUTOFF_STAGES = ('report', 'delivery')
_TERMINAL = ('passed', 'failed', 'blocked', 'skipped')
_WEIGHT_SUFFIXES = ('.pt', '.pth', '.npz', '.ckpt')
_PROCESS_LOCKS = {}
_PROCESS_GUARD = threading.Lock()


def confirmation_questions():
    return (
        '本轮是否包含主牌组换牌候选（include_candidates），即改动每队32张配牌，而非起手换牌？请明确选择包含或不包含。',
        '本轮截止日期和时间是否采用北京时间09:00（Asia/Shanghai）？请确认带日期、时区的ISO时刻；已过当日09:00可提出次日09:00，不能自动批准。',
        '本轮训练哪些高级人机（roster）及实际32张构筑（builds）？请确认每队四名角色及顺序、起点来源；包含候选时明确mutable_characters，其余角色锁定。',
    )


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'))


def _digest(value):
    return sha256(_canonical_json(value).encode('utf-8')).hexdigest()


def _require_aware(moment, label):
    if moment is None:
        return datetime.now(timezone.utc)
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        raise ValueError(label + ' must be a timezone-aware datetime')
    return moment


def _parse_deadline(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('hard_deadline must be ISO-8601 with timezone')
    try:
        moment = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('hard_deadline must be ISO-8601 with timezone') from exc
    if moment.tzinfo is None:
        raise ValueError('hard_deadline must include a timezone')
    return moment


def _iso(moment):
    return moment.isoformat()


def _strings(value, label, *, count=None, distinct=False):
    if not isinstance(value, (list, tuple)):
        raise ValueError(label + ' must be a list')
    if count is None and not value:
        raise ValueError(label + ' must be a nonempty list')
    items = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(label + ' entries must be nonempty strings')
        items.append(item)
    if count is not None and len(items) != count:
        raise ValueError(label + ' must contain exactly %s entries' % count)
    if distinct and len(set(items)) != len(items):
        raise ValueError(label + ' must be distinct')
    return list(items)


def _normalize_build(build, label):
    if not isinstance(build, dict):
        raise ValueError(label + ' must be a mapping')
    return {
        'character_ids': _strings(build.get('character_ids'), label + '.character_ids', count=4, distinct=True),
        'card_ids': _strings(build.get('card_ids'), label + '.card_ids', count=32),
    }


def _normalize_request(request):
    if not isinstance(request, dict):
        raise ValueError('request must be a mapping')
    if 'include_candidates' not in request or not isinstance(request['include_candidates'], bool):
        raise ValueError('include_candidates must be an explicit bool')
    include_candidates = request['include_candidates']
    deadline = _parse_deadline(request.get('hard_deadline'))
    roster = _strings(request.get('roster'), 'roster', distinct=True)
    builds = request.get('builds')
    if not isinstance(builds, dict) or set(builds) != set(roster):
        raise ValueError('builds must map each roster id')
    normalized_builds = {key: _normalize_build(builds[key], 'builds.' + key) for key in roster}
    agent = request.get('default_agent', 'grok')
    if not isinstance(agent, str) or not agent.strip():
        raise ValueError('default_agent must be a nonempty explicit agent name')
    mutable = request.get('mutable_characters')
    if include_candidates:
        if not isinstance(mutable, dict) or set(mutable) != set(roster):
            raise ValueError('mutable_characters is required for every roster when candidates are enabled')
        normalized_mutable = {}
        for key in roster:
            roles = _strings(mutable[key], 'mutable_characters.' + key, distinct=True) if mutable[key] else []
            allowed = set(normalized_builds[key]['character_ids'])
            if any(role not in allowed for role in roles):
                raise ValueError('mutable_characters must be a subset of ' + key + ' character_ids')
            normalized_mutable[key] = roles
        if not any(normalized_mutable.values()):
            raise ValueError('candidate inclusion requires at least one confirmed mutable character')
    else:
        if mutable not in (None, {}):
            if not isinstance(mutable, dict):
                raise ValueError('mutable_characters must be a mapping')
            extra = set(mutable) - set(roster)
            if extra:
                raise ValueError('mutable_characters contains unknown roster ids')
        normalized_mutable = {}
    body = {
        'include_candidates': include_candidates,
        'hard_deadline': _iso(deadline),
        'roster': roster,
        'builds': normalized_builds,
        'mutable_characters': normalized_mutable,
        'default_agent': agent,
    }
    return body, deadline


def _stage_record(status='pending'):
    return {'status': status, 'attempts': [], 'skip_reason': None}


def _empty_stages():
    return {stage: _stage_record() for stage in STAGE_ORDER}


class TrainingFlow:
    def __init__(self, root, state):
        self._root = Path(root)
        self._state = state

    @property
    def directory(self):
        return self._root

    @classmethod
    def create(cls, directory, request, now=None):
        now = _require_aware(now, 'now')
        root = Path(directory)
        if root.exists() and (root.is_symlink() or not root.is_dir()):
            raise ValueError('run directory must be a real directory')
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve()
        if (root / 'state.json').exists():
            raise ValueError('training flow already exists; use load()')
        body, deadline = _normalize_request(request)
        state = {
            'schema': SCHEMA,
            'request': body,
            'request_sha256': _digest(body),
            'confirmation': None,
            'stages': _empty_stages(),
            'hard_deadline': body['hard_deadline'],
            'hard_deadline_epoch': deadline.timestamp(),
            'workflow_complete': False,
            'quality_approved': False,
            'complete_report': False,
            'final_review': None,
            'execution_plan': {},
            'created_at': _iso(now),
            'updated_at': _iso(now),
        }
        flow = cls(root, state)
        with flow._exclusive():
            if (root / 'state.json').exists():
                raise ValueError('training flow already exists; use load()')
            flow._persist()
        return flow

    @classmethod
    def load(cls, directory):
        root = Path(directory)
        if root.is_symlink():
            raise ValueError('run directory must not be a symlink')
        root = root.resolve()
        path = root / 'state.json'
        if path.is_symlink() or not path.is_file():
            raise ValueError('training flow state is missing')
        state = json.loads(path.read_text(encoding='utf-8'))
        if state.get('schema') != SCHEMA:
            raise ValueError('unsupported training flow schema')
        flow = cls(root, state)
        flow._validate_identity()
        return flow

    def snapshot(self):
        self._refresh_flags()
        data = json.loads(_canonical_json(self._state))
        data['directory'] = str(self._root)
        return data

    def next_stage(self):
        if self._state['confirmation'] is None:
            return 'confirmation'
        if self._running_stage() is not None:
            return None
        halted = self._learning_halted()
        for stage in self._required_stages():
            status = self._status(stage)
            if status in ('passed', 'skipped'):
                continue
            if stage in _POST_CUTOFF_STAGES:
                if status not in ('failed', 'blocked'):
                    return stage
                continue
            if halted:
                continue
            if status in ('failed', 'blocked'):
                halted = True
                continue
            return stage
        return None

    def confirm(self, confirmation, now=None):
        now = _require_aware(now, 'now')
        with self._exclusive():
            self._reload()
            if self._state['confirmation'] is not None:
                raise ValueError('flow is already confirmed')
            receipt = self._validated_confirmation(confirmation)
            receipt['confirmed_at'] = _iso(now)
            self._state['confirmation'] = receipt
            self._stage('confirmation')['status'] = 'passed'
            if not self._request()['include_candidates']:
                reason = 'user excluded candidate 32-card build search; opening mulligan is not candidate inclusion'
                for stage in CANDIDATE_STAGES:
                    record = self._stage(stage)
                    record['status'] = 'skipped'
                    record['skip_reason'] = reason
            self._touch(now)
            self._persist()
            return self.snapshot()

    def start(self, stage_id, now=None):
        now = _require_aware(now, 'now')
        with self._exclusive():
            self._reload()
            self._assert_startable(stage_id, now)
            attempt = len(self._stage(stage_id)['attempts']) + 1
            token = uuid4().hex
            output = Path('stages') / stage_id / ('attempt-%s' % attempt)
            absolute = self._root / output
            absolute.mkdir(parents=True, exist_ok=True)
            record = {
                'attempt': attempt,
                'token': token,
                'started_at': _iso(now),
                'finished_at': None,
                'deadline': self._state['hard_deadline'],
                'agent': self._request()['default_agent'],
                'output_directory': output.as_posix(),
                'request_sha256': self._state['request_sha256'],
                'result': None,
            }
            self._stage(stage_id)['attempts'].append(record)
            self._stage(stage_id)['status'] = 'running'
            self._stage(stage_id)['invalidated'] = False
            self._touch(now)
            self._persist()
            return {
                'stage_id': stage_id,
                'attempt': attempt,
                'token': token,
                'deadline': self._state['hard_deadline'],
                'request_sha256': self._state['request_sha256'],
                'agent': self._request()['default_agent'],
                'output_directory': str(absolute),
            }

    def finish(self, stage_id, token, result, now=None):
        now = _require_aware(now, 'now')
        with self._exclusive():
            self._reload()
            self._finish_locked(stage_id, token, result, now, force=False)
            self._persist()
            return self.snapshot()

    def run(self, handlers, now=None):
        clock = now if callable(now) else lambda: _require_aware(now, 'now')
        if not isinstance(handlers, dict):
            raise ValueError('handlers must be a stage mapping')
        with self._exclusive():
            self._reload()
            missing = [stage for stage in self._possible_stages() if stage not in handlers]
            if missing:
                raise ValueError('missing required stage handlers before launch: ' + ', '.join(missing))
            if any(not callable(handlers[stage]) for stage in self._possible_stages()):
                raise ValueError('every required stage handler must be callable')
        self.configure_executors(handlers)
        snapshots = []
        while True:
            current = _require_aware(clock(), 'now')
            stage = self._runnable_stage(current)
            if stage is None:
                break
            assignment = self.start(stage, now=current)
            try:
                payload = handlers[stage](assignment)
            except Exception as exc:
                payload = {
                    'verdict': 'failed',
                    'summary': '%s: %s' % (type(exc).__name__, exc),
                    'artifacts': [],
                    'metrics': {},
                }
            try:
                snapshots.append(self.finish(stage, assignment['token'], payload, now=_require_aware(clock(), 'now')))
            except ValueError as exc:
                snapshots.append(self._force_fail(stage, assignment['token'], str(exc), _require_aware(clock(), 'now')))
        return snapshots[-1] if snapshots else self.snapshot()

    def configure_executors(self, handlers):
        """Main must register the entire required capability set before any step."""
        if not isinstance(handlers, dict):
            raise ValueError('handlers must be a stage mapping')
        required = self._required_stages()
        missing = [name for name in required if not callable(handlers.get(name))]
        if missing:
            raise ValueError('missing required executors before launch: ' + ', '.join(missing))
        plan = {}
        for name in required:
            handler = handlers[name]
            identity = getattr(handler, '__module__', '') + '.' + getattr(handler, '__qualname__', type(handler).__qualname__)
            try:
                source = inspect.getsource(handler)
            except (OSError, TypeError):
                source = identity
            plan[name] = {'identity': identity, 'source_sha256': sha256(source.encode()).hexdigest()}
        with self._exclusive():
            self._reload()
            self._state['execution_plan'] = plan
            self._persist()
        return self.snapshot()

    def invalidate_evidence(self, stage_id, reason, now=None):
        """Main-agent remediation, retaining attempts and original cutoff."""
        if stage_id not in STAGE_ORDER or stage_id == 'confirmation' or not isinstance(reason, str) or not reason.strip():
            raise ValueError('invalidation requires an evidence step and explicit reason')
        now = _require_aware(now, 'now')
        with self._exclusive():
            self._reload()
            if self._running_stage() is not None:
                raise ValueError('settle active work before invalidating evidence')
            affected = []
            for name in STAGE_ORDER[STAGE_ORDER.index(stage_id):]:
                record = self._stage(name)
                if record['status'] == 'skipped':
                    continue
                record['status'] = 'pending'
                record['invalidated'] = True
                affected.append(name)
            self._state.setdefault('decisions', []).append({'kind': 'invalidate', 'reason': reason,
                                                          'steps': affected, 'at': _iso(now)})
            self._state['final_review'] = None
            self._refresh_flags()
            self._touch(now)
            self._persist()
        return self.snapshot()

    def _request(self):
        return self._state['request']

    def _stage(self, stage_id):
        return self._state['stages'][stage_id]

    def _status(self, stage_id):
        return self._stage(stage_id)['status']

    def _required_stages(self):
        stages = [stage for stage in STAGE_ORDER if stage != 'confirmation']
        if not self._request()['include_candidates']:
            stages = [stage for stage in stages if stage not in CANDIDATE_STAGES]
        return stages

    def _learning_halted(self):
        for stage in self._required_stages():
            if stage in _POST_CUTOFF_STAGES:
                continue
            if self._status(stage) in ('failed', 'blocked'):
                return True
        return False

    def _running_stage(self):
        for stage in STAGE_ORDER:
            if self._status(stage) == 'running':
                return stage
        return None

    def _possible_stages(self):
        planned = []
        halted = self._learning_halted()
        for stage in self._required_stages():
            status = self._status(stage)
            if status in ('passed', 'skipped'):
                continue
            if status in ('failed', 'blocked') and stage not in _POST_CUTOFF_STAGES:
                halted = True
                continue
            if halted and stage not in _POST_CUTOFF_STAGES:
                continue
            if status not in ('failed', 'blocked'):
                planned.append(stage)
        return planned

    def _runnable_stage(self, now):
        stage = self.next_stage()
        if stage in (None, 'confirmation'):
            return None
        if stage not in _POST_CUTOFF_STAGES and self._past_deadline(now):
            for later in _POST_CUTOFF_STAGES:
                if self._status(later) not in _TERMINAL and self._status(later) != 'running':
                    return later
            return None
        return stage

    def _past_deadline(self, now):
        return now.timestamp() >= self._state['hard_deadline_epoch']

    def _validated_confirmation(self, confirmation):
        if not isinstance(confirmation, dict):
            raise ValueError('confirmation must be an explicit user receipt')
        if confirmation.get('source') != 'user':
            raise ValueError("confirmation source must be 'user'")
        reference = confirmation.get('reference')
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError('confirmation reference must be a nonempty message/answer identifier')
        for key in ('include_candidates', 'hard_deadline', 'roster', 'builds'):
            if key not in confirmation:
                raise ValueError('confirmation is missing ' + key + '; omission is not approval')
        stored_mutable = self._request()['mutable_characters']
        if self._request()['include_candidates'] and 'mutable_characters' not in confirmation:
            raise ValueError('confirmation is missing mutable_characters')
        if 'mutable_characters' in confirmation and confirmation['mutable_characters'] != stored_mutable:
            raise ValueError('confirmation mutable_characters does not match the frozen request')
        body, deadline = _normalize_request({
            'include_candidates': confirmation['include_candidates'],
            'hard_deadline': confirmation['hard_deadline'],
            'roster': confirmation['roster'],
            'builds': confirmation['builds'],
            'mutable_characters': stored_mutable or None,
            'default_agent': self._request()['default_agent'],
        })
        request = self._request()
        if body['include_candidates'] != request['include_candidates']:
            raise ValueError('confirmation include_candidates does not match the frozen request')
        if abs(deadline.timestamp() - self._state['hard_deadline_epoch']) > 1e-6:
            raise ValueError('confirmation hard_deadline does not match the original cutoff')
        if body['roster'] != request['roster'] or body['builds'] != request['builds']:
            raise ValueError('confirmation roster/builds do not match the frozen request')
        return {
            'source': 'user',
            'reference': reference.strip(),
            'include_candidates': body['include_candidates'],
            'hard_deadline': request['hard_deadline'],
            'roster': list(request['roster']),
            'builds': json.loads(_canonical_json(request['builds'])),
            'mutable_characters': json.loads(_canonical_json(stored_mutable)),
        }

    def _assert_startable(self, stage_id, now):
        if stage_id not in STAGE_ORDER:
            raise ValueError('unknown stage: ' + str(stage_id))
        if stage_id == 'confirmation':
            raise ValueError('confirmation is completed only by confirm()')
        if self._state['confirmation'] is None:
            raise ValueError('explicit user confirmation is required before stages start')
        if stage_id not in _POST_CUTOFF_STAGES and set(self._state.get('execution_plan', {})) != set(self._required_stages()):
            raise ValueError('register all required executors before starting any work')
        if self._digest_now() != self._state['request_sha256']:
            raise ValueError('frozen request digest changed')
        running = self._running_stage()
        if running is not None:
            raise ValueError('stage %s already has an active claim' % running)
        status = self._status(stage_id)
        if status == 'skipped':
            raise ValueError(stage_id + ' was skipped and cannot be claimed')
        if status == 'passed':
            raise ValueError(stage_id + ' already passed')
        if status == 'running':
            raise ValueError(stage_id + ' already has an active claim')
        if stage_id in CANDIDATE_STAGES and not self._request()['include_candidates']:
            raise ValueError('candidate stages are disabled for this flow')
        past = self._past_deadline(now)
        if past and stage_id not in _POST_CUTOFF_STAGES:
            raise ValueError('original hard deadline has passed; only report/delivery may collect existing files')
        if not self._prerequisites_met(stage_id, past=past):
            raise ValueError(stage_id + ' prerequisites are not satisfied')
        if stage_id not in _POST_CUTOFF_STAGES:
            for name in STAGE_ORDER:
                if self._status(name) == 'passed':
                    for attempt in self._stage(name)['attempts'][-1:]:
                        for item in (attempt.get('result') or {}).get('artifacts', []):
                            self._verify_artifact(item)

    def _prerequisites_met(self, stage_id, *, past):
        if stage_id in _POST_CUTOFF_STAGES:
            if stage_id == 'report':
                return self._learning_halted() or past or self._learning_ready_for_close()
            return self._status('report') in ('passed', 'failed', 'blocked')
        predecessors = []
        for stage in self._required_stages():
            if stage == stage_id:
                break
            if stage in _POST_CUTOFF_STAGES:
                continue
            predecessors.append(stage)
        return all(self._status(stage) == 'passed' for stage in predecessors)

    def _learning_ready_for_close(self):
        return all(
            self._status(stage) in _TERMINAL
            for stage in self._required_stages()
            if stage not in _POST_CUTOFF_STAGES
        )

    def _digest_now(self):
        return _digest(self._request())

    def _finish_locked(self, stage_id, token, result, now, *, force):
        if stage_id not in STAGE_ORDER:
            raise ValueError('unknown stage: ' + str(stage_id))
        record = self._active_attempt(stage_id, token)
        payload = self._validated_result(stage_id, result, force=force)
        if payload['verdict'] == 'passed' and stage_id not in _POST_CUTOFF_STAGES and self._past_deadline(now):
            payload['verdict'] = 'failed'
            payload['summary'] = 'original hard deadline elapsed before stage settlement: ' + payload['summary']
        record['finished_at'] = _iso(now)
        record['result'] = payload
        self._stage(stage_id)['status'] = payload['verdict']
        self._refresh_flags()
        self._touch(now)

    def _active_attempt(self, stage_id, token):
        if self._status(stage_id) != 'running':
            raise ValueError(stage_id + ' has no active claim')
        attempts = self._stage(stage_id)['attempts']
        if not attempts:
            raise ValueError(stage_id + ' has no active claim')
        record = attempts[-1]
        if record.get('token') != token or record.get('result') is not None:
            raise ValueError('result token does not match the active claim')
        if record.get('request_sha256') != self._state['request_sha256']:
            raise ValueError('stale result does not match the frozen request digest')
        return record

    def _validated_result(self, stage_id, result, *, force):
        if not isinstance(result, dict):
            raise ValueError('result must be a mapping')
        verdict = result.get('verdict')
        if verdict not in ('passed', 'failed', 'blocked'):
            raise ValueError('result verdict must be passed, failed, or blocked')
        summary = result.get('summary')
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError('result summary must be nonempty')
        artifacts = result.get('artifacts') or []
        metrics = result.get('metrics')
        if metrics is None:
            metrics = {}
        if not isinstance(metrics, dict):
            raise ValueError('result metrics must be a mapping')
        if result.get('request_sha256') not in (None, self._state['request_sha256']):
            raise ValueError('stale result does not match the frozen request digest')
        checked = []
        if not force:
            if verdict == 'passed':
                if not isinstance(artifacts, list) or not artifacts:
                    raise ValueError('passed requires hash-verified artifacts')
                checked = [self._verify_artifact(item) for item in artifacts]
                self._validate_pass_evidence(stage_id, checked, metrics)
            elif artifacts:
                if not isinstance(artifacts, list):
                    raise ValueError('result artifacts must be a list')
                checked = [self._verify_artifact(item) for item in artifacts]
        elif artifacts:
            checked = list(artifacts)
        if stage_id == 'report' and verdict == 'passed' and not force:
            from .training_flow_report import write_flow_report
            snapshot = json.loads(_canonical_json(self._state))
            snapshot['directory'] = str(self._root)
            attempt = snapshot['stages']['report']['attempts'][-1]
            attempt['result'] = {'verdict': verdict, 'summary': summary, 'artifacts': checked, 'metrics': metrics}
            report = write_flow_report(snapshot, self._root / attempt['output_directory'] / 'canonical-flow-report.md')
            checked.append(self._verify_artifact(report))
        return {
            'verdict': verdict,
            'summary': summary.strip(),
            'artifacts': checked,
            'metrics': json.loads(_canonical_json(metrics)),
        }

    def _verify_artifact(self, item):
        if not isinstance(item, dict) or 'path' not in item or 'sha256' not in item:
            raise ValueError('artifact must contain path and sha256')
        digest = item['sha256']
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError('artifact sha256 is invalid')
        path = self._inside_root(item['path'])
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != digest.lower():
            raise ValueError('artifact sha256 does not match ' + path.name)
        rel = path.relative_to(self._root).as_posix()
        return {'path': rel, 'sha256': actual}

    def _inside_root(self, raw):
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError('artifact path must be a string inside the run root')
        root = self._root.resolve()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        if '..' in Path(raw).parts:
            raise ValueError('artifact path traversal is not allowed')
        if candidate.is_symlink():
            raise ValueError('symlink artifacts are rejected')
        for parent in candidate.parents:
            if parent == root:
                break
            if root in parent.parents or parent == root.parent:
                if parent.is_symlink():
                    raise ValueError('symlink artifacts are rejected')
        if not candidate.exists() or not candidate.is_file():
            raise ValueError('artifact is missing: ' + raw)
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError('artifact path is outside the run root') from exc
        if resolved.is_symlink() or candidate.is_symlink():
            raise ValueError('symlink artifacts are rejected')
        return resolved

    def _validate_pass_evidence(self, stage_id, artifacts, metrics):
        if stage_id == 'freeze' and self._request()['include_candidates']:
            steps = metrics.get('adaptation_target_steps')
            if type(steps) is not int or steps <= 0:
                raise ValueError('freeze must declare adaptation_target_steps before learning')
        elif stage_id == 'base_training':
            games = metrics.get('complete_training_games')
            if isinstance(games, dict):
                self._require_roster_counts(metrics, 'complete_training_games')
            elif type(games) is not int or games <= 0:
                raise ValueError('base_training requires positive complete_training_games')
            self._require_roster_counts(metrics, 'formal_policy_updates')
        elif stage_id == 'candidate_search':
            changes = metrics.get('candidate_changes')
            if not isinstance(changes, list) or (not changes and any(self._request()['mutable_characters'].values())):
                raise ValueError('candidate search requires actual build changes and screening evidence')
            from ..content.duel_v2 import CARDS, validate_deck
            for change in changes:
                key = change.get('key') if isinstance(change, dict) else None
                if key not in self._request()['roster']:
                    raise ValueError('candidate outside confirmed roster')
                original = self._request()['builds'][key]
                candidate = validate_deck(change.get('build'))
                if candidate['character_ids'] != original['character_ids']:
                    raise ValueError('candidate reordered confirmed characters')
                old = Counter(original['card_ids'])
                new = Counter(candidate['card_ids'])
                removed, added = old - new, new - old
                if not removed or dict(removed) != change.get('removed') or dict(added) != change.get('added'):
                    raise ValueError('candidate delta does not match actual card counts')
                mutable = set(self._request()['mutable_characters'][key])
                if any(CARDS[card]['character_id'] not in mutable for card in set(removed) | set(added)):
                    raise ValueError('candidate changed a locked character')
        elif stage_id in ('build_confirmation', 'build_audit'):
            decision = metrics.get('decision')
            if decision not in ('accept', 'retain_baseline', 'insufficient_evidence') or not metrics.get('reason'):
                raise ValueError('build confirmation/audit requires decision and reason')
            if type(metrics.get('normal_games')) is not int or metrics['normal_games'] <= 0:
                raise ValueError('build confirmation/audit requires independent completed games')
            seed_hash = metrics.get('seed_allocation_sha256')
            if not isinstance(seed_hash, str) or len(seed_hash) != 64:
                raise ValueError('independent seed identity is missing')
            predecessors = ('candidate_search', 'baseline_adaptation', 'candidate_adaptation') + (('build_confirmation',) if stage_id == 'build_audit' else ())
            for name in predecessors:
                proof = self._stage(name)['attempts'][-1]['result']['metrics']
                if seed_hash in (proof.get('seed_allocation_sha256'), (proof.get('adaptation') or {}).get('seed_allocation_sha256')):
                    raise ValueError('independent seed identity was already consumed upstream')
        elif stage_id in ('baseline_adaptation', 'candidate_adaptation'):
            proof = metrics.get('adaptation')
            if not isinstance(proof, dict):
                raise ValueError('adaptation requires actual equal-budget evidence')
            freeze = self._stage('freeze')['attempts'][-1]['result']['metrics']
            target = freeze['adaptation_target_steps']
            for key in ('initial_sha256', 'opponents_sha256', 'seed_allocation_sha256'):
                if not proof.get(key):
                    raise ValueError('adaptation missing common ' + key)
            steps = proof.get('optimizer_steps')
            if not isinstance(steps, dict) or any(type(steps.get(key)) is not int or steps[key] != target for key, roles in self._request()['mutable_characters'].items() if roles):
                raise ValueError('adaptation did not complete predeclared equal optimizer steps')
            if stage_id == 'candidate_adaptation':
                baseline = self._stage('baseline_adaptation')['attempts'][-1]['result']['metrics']['adaptation']
                if any(proof[key] != baseline[key] for key in ('initial_sha256', 'opponents_sha256', 'seed_allocation_sha256', 'optimizer_steps')):
                    raise ValueError('candidate/original adaptation conditions are unequal')
        elif stage_id == 'evaluation':
            planned = metrics.get('planned_cells')
            completed = metrics.get('completed_cells')
            if isinstance(planned, bool) or isinstance(completed, bool):
                raise ValueError('evaluation denominators must be real cell counts, not booleans')
            if not isinstance(planned, int) or not isinstance(completed, int):
                raise ValueError('evaluation requires planned_cells and completed_cells')
            expected = len(self._request()['roster']) ** 2
            if planned != expected or completed < 0 or completed > planned:
                raise ValueError('evaluation cell denominators are invalid')
            if completed != planned:
                raise ValueError('evaluation passed requires every planned cell completed')
            if not any(_is_raw_record(item['path']) for item in artifacts):
                raise ValueError('evaluation passed requires raw evidence files')
            modes = metrics.get('matrix_modes')
            if not isinstance(modes, dict) or set(modes) != {'pure', 'search'}:
                raise ValueError('evaluation requires both pure and search matrix_modes')
            for mode, data in modes.items():
                if not isinstance(data, dict) or data.get('planned_cells') != expected or data.get('completed_cells') != expected:
                    raise ValueError('evaluation requires complete NxN cells for ' + mode)
                cells = data.get('cells')
                if not isinstance(cells, list) or len(cells) != expected:
                    raise ValueError('evaluation requires actual cell win/loss denominators')
                pairs = set()
                for cell in cells:
                    if not isinstance(cell, dict):
                        raise ValueError('invalid evaluation cell')
                    pair = (cell.get('row'), cell.get('column'))
                    if pair in pairs or any(key not in self._request()['roster'] for key in pair):
                        raise ValueError('duplicate or out-of-scope matrix cell')
                    pairs.add(pair)
                    counts = [cell.get(key) for key in ('planned', 'completed', 'first_wins', 'second_wins', 'draws')]
                    if any(type(count) is not int or count < 0 for count in counts) or not counts[0] or counts[0] != counts[1] or sum(counts[2:]) != counts[1] or cell.get('errors', 0) or cell.get('truncated', 0):
                        raise ValueError('incomplete or inconsistent matrix cell denominator')
            self._verify_matrix_raw(artifacts, modes)
        elif stage_id == 'delivery':
            paths = [item['path'] for item in artifacts]
            numerical = [path for path in paths if Path(path).suffix in _WEIGHT_SUFFIXES]
            if not numerical:
                raise ValueError('delivery passed requires actual numerical model files, never JSON-only')
            for path in numerical:
                _validate_numerical_file(self._inside_root(path))
            for key in self._request()['roster']:
                if not any(Path(path).stem == key or key in Path(path).parts for path in numerical):
                    raise ValueError('delivery missing actual numerical model for ' + key)
            if not any(_is_raw_record(path) for path in paths):
                raise ValueError('delivery passed requires raw records')
            if not any(_is_public_replay(path) for path in paths):
                raise ValueError('delivery passed requires public replays')
            required = [stage for stage in self._required_stages() if stage not in _POST_CUTOFF_STAGES]
            if any(self._status(stage) != 'passed' and self._status(stage) != 'skipped' for stage in required):
                raise ValueError('delivery cannot pass while required evidence stages are incomplete')

    def _verify_matrix_raw(self, artifacts, modes):
        """Recount terminal records; a summary claim is not a matrix denominator."""
        tallies = Counter()
        seen = set()
        for item in artifacts:
            if not _is_raw_record(item['path']):
                continue
            path = self._inside_root(item['path'])
            opener = gzip.open if path.suffix == '.gz' else open
            try:
                with opener(path, 'rt', encoding='utf8') as f:
                    content = f.read()
                data = [json.loads(line) for line in content.splitlines() if line.strip()] if '.jsonl' in path.suffixes else json.loads(content)
            except (ValueError, OSError) as exc:
                raise ValueError('raw evaluation records are unreadable') from exc
            records = data if isinstance(data, list) else [data]
            for game in records:
                if not isinstance(game, dict):
                    raise ValueError('raw matrix record must be a mapping')
                if isinstance(game.get('job'), dict):
                    game = {**game['job'], **game}
                if game.get('stage') not in (None, 'matrix'):
                    continue
                mode = 'pure' if game.get('mode') in ('pure', 'argmax') else game.get('mode')
                if mode not in modes:
                    continue
                if not game.get('id') or game['id'] in seen or game.get('complete') is not True or game.get('error') or game.get('winner') not in ('a', 'b', 'draw') or game.get('first') not in ('a', 'b'):
                    raise ValueError('duplicate or nonterminal raw matrix game')
                seen.add(game['id'])
                policies = game.get('policies') or {}
                pair = (game.get('left') or (policies.get('a') or [None, None])[-1],
                        game.get('right') or (policies.get('b') or [None, None])[-1])
                if game['first'] == 'b':
                    pair = pair[::-1]
                if any(key not in self._request()['roster'] for key in pair):
                    raise ValueError('raw matrix game outside roster')
                kind = 'draws' if game['winner'] == 'draw' else 'first_wins' if game['winner'] == game['first'] else 'second_wins'
                tallies[(mode, *pair, 'completed')] += 1
                tallies[(mode, *pair, kind)] += 1
        for mode, data in modes.items():
            for cell in data['cells']:
                for kind in ('completed', 'first_wins', 'second_wins', 'draws'):
                    if tallies[(mode, cell['row'], cell['column'], kind)] != cell[kind]:
                        raise ValueError('raw records disagree with matrix denominator')

    def _require_roster_counts(self, metrics, key):
        values = metrics.get(key)
        if not isinstance(values, dict):
            raise ValueError('base_training requires %s for every roster' % key)
        for roster_id in self._request()['roster']:
            count = values.get(roster_id)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError('base_training %s must be a positive count for %s' % (key, roster_id))

    def _refresh_flags(self):
        self._validate_identity()
        required = self._required_stages()
        learning = [stage for stage in required if stage not in _POST_CUTOFF_STAGES]
        learning_ok = all(self._status(stage) in ('passed', 'skipped') for stage in learning)
        report_passed = self._status('report') == 'passed'
        delivery_terminal = self._status('delivery') in ('passed', 'failed', 'blocked')
        review = self._state.get('final_review') or {}
        basis = self.audit_basis()
        review_ok = bool(basis is not None and review.get('verdict') == 'allow' and review.get('basis_sha256') == basis)
        if review_ok:
            try:
                self._verify_artifact(review['artifact'])
            except (ValueError, KeyError):
                review_ok = False
        from .training_flow_report import missing_report_evidence
        semantic_complete = not missing_report_evidence(self._state)
        self._state['complete_report'] = bool(learning_ok and report_passed and review_ok and semantic_complete)
        self._state['workflow_complete'] = bool(self._state['complete_report'] and self._status('delivery') == 'passed')
        self._state['flow_terminal'] = bool(delivery_terminal and self._running_stage() is None)
        self._state['quality_approved'] = False

    def _force_fail(self, stage_id, token, summary, now):
        with self._exclusive():
            self._reload()
            self._finish_locked(
                stage_id, token,
                {'verdict': 'failed', 'summary': summary, 'artifacts': [], 'metrics': {}},
                now, force=True,
            )
            self._persist()
            return self.snapshot()

    def _touch(self, now):
        self._state['updated_at'] = _iso(now)

    def _state_path(self):
        return self._root / 'state.json'

    def _lock_path(self):
        return self._root / 'state.lock'

    def _reload(self):
        path = self._state_path()
        if path.is_symlink():
            raise ValueError('symlink flow state is rejected')
        if path.is_file():
            self._state = json.loads(path.read_text(encoding='utf-8'))
            self._validate_identity()

    def _validate_identity(self):
        state = self._state
        if state.get('schema') != SCHEMA or _digest(state.get('request')) != state.get('request_sha256'):
            raise ValueError('frozen request digest changed')
        deadline = _parse_deadline(state['request']['hard_deadline'])
        if state.get('hard_deadline') != state['request']['hard_deadline'] or state.get('hard_deadline_epoch') != deadline.timestamp():
            raise ValueError('original deadline identity changed')
        if set(state.get('stages', {})) != set(STAGE_ORDER):
            raise ValueError('required training evidence stages changed')
        for name in STAGE_ORDER:
            if state['stages'][name]['status'] == 'skipped' and (name not in CANDIDATE_STAGES or state['request']['include_candidates']):
                raise ValueError('required stage cannot be skipped')

    def audit_basis(self):
        evidence = []
        for name in STAGE_ORDER:
            stage = self._stage(name)
            for attempt in stage['attempts'][-1:]:
                for item in (attempt.get('result') or {}).get('artifacts', []):
                    try:
                        checked = self._verify_artifact(item)
                    except ValueError:
                        return None
                    evidence.append((name, attempt['attempt'], checked['path'], checked['sha256']))
        return _digest({'request_sha256': self._state['request_sha256'], 'evidence': evidence,
                        'statuses': {name: self._status(name) for name in STAGE_ORDER}})

    def record_final_review(self, artifact, now=None):
        """Main-agent adoption of fresh review after delivery; never child routing."""
        now = _require_aware(now, 'now')
        with self._exclusive():
            self._reload()
            checked = self._verify_artifact(artifact)
            review = json.loads(self._inside_root(checked['path']).read_text(encoding='utf-8'))
            if review.get('verdict') not in ('allow', 'remediate', 'blocked') or not review.get('reviewer_session'):
                raise ValueError('fresh review requires a distinct reviewer session and verdict')
            basis = self.audit_basis()
            if basis is None or review.get('basis_sha256') != basis or review.get('request_sha256') != self._state['request_sha256']:
                raise ValueError('fresh review basis is stale')
            self._state['final_review'] = {**review, 'artifact': checked, 'adopted_at': _iso(now)}
            self._refresh_flags()
            self._touch(now)
            self._persist()
            return self.snapshot()

    def _persist(self):
        path = self._state_path()
        temporary = path.with_suffix('.json.tmp')
        payload = _canonical_json(self._state)
        temporary.write_text(payload, encoding='utf-8')
        os.replace(temporary, path)

    @contextmanager
    def _exclusive(self):
        key = str(self._root.resolve())
        with _PROCESS_GUARD:
            lock = _PROCESS_LOCKS.setdefault(key, threading.Lock())
        with lock:
            self._root.mkdir(parents=True, exist_ok=True)
            flags = os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
            fd = os.open(self._lock_path(), flags, 0o600)
            locked = False
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                else:
                    if not os.fstat(fd).st_size:
                        os.write(fd, b'\0')
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                locked = True
                yield
            finally:
                if locked:
                    if fcntl is not None:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    else:
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                os.close(fd)


def _validate_numerical_file(path):
    """A renamed JSON/text file cannot masquerade as actual numerical weights."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if path.suffix == '.npz':
                arrays = [name for name in names if name.endswith('.npy')]
                if not arrays:
                    raise ValueError('numerical NPZ arrays are missing')
                for name in arrays:
                    data = archive.read(name)
                    if len(data) < 10 or not data.startswith(b'\x93NUMPY'):
                        raise ValueError('numerical NPZ arrays are invalid')
                    major = data[6]
                    if major not in (1, 2, 3):
                        raise ValueError('unsupported numeric array format')
                    size = 2 if major == 1 else 4
                    length = int.from_bytes(data[8:8 + size], 'little')
                    offset = 8 + size
                    if length > 65536 or offset + length > len(data):
                        raise ValueError('invalid numeric array header')
                    header = ast.literal_eval(data[offset:offset + length].decode('utf8' if major == 3 else 'latin1').strip())
                    dtype = header.get('descr') if isinstance(header, dict) else None
                    shape = header.get('shape') if isinstance(header, dict) else None
                    match = re.fullmatch(r'[<>=|]([biufc])(\d+)', dtype or '')
                    if match is None or not isinstance(shape, tuple) or any(type(n) is not int or n < 0 for n in shape):
                        raise ValueError('array must contain numeric tensors, not object/metadata')
                    count = 1
                    for n in shape:
                        count *= n
                    if count * int(match.group(2)) != len(data) - offset - length:
                        raise ValueError('numeric tensor data size mismatch')
            elif not any(name.endswith('/data.pkl') for name in names) or not any('/data/' in name for name in names):
                raise ValueError('numerical checkpoint tensor storage is missing')
    except (OSError, zipfile.BadZipFile, SyntaxError, UnicodeError, TypeError) as exc:
        raise ValueError('delivery numerical file is not an actual numeric archive') from exc


def _is_raw_record(path):
    name = Path(path).name.lower()
    parts = {part.lower() for part in Path(path).parts}
    return 'raw' in parts or 'raw' in name or name.endswith('.json.gz')


def _is_public_replay(path):
    name = Path(path).name.lower()
    parts = {part.lower() for part in Path(path).parts}
    return 'replay' in name or 'replays' in parts
