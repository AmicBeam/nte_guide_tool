"""Offline public-17 / seven-key SoA conformance-input collector.

Natural official-preset trajectories plus the native-array physical-action
fixture router produce snapshots for a later SoA implementation. This is not
a SoA rule engine, trainer, GPU benchmark, or simplified duel.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

from app.modules.card_game.content.duel_v2 import PUBLIC_CHARACTER_IDS, STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import acting_side, apply_action, legal_actions, new_game
from app.modules.card_game.engine.duel_v2.entities import (
    CardEntity,
    CharacterEntity,
    PlayerEntity,
    clone_state,
)
from app.modules.card_game.rl.batched_duel.domain_soa_schema import (
    COVERAGE,
    GAME_KEYS,
    PUBLIC_CHARACTER_ORDER,
)
from app.modules.card_game.rl.batched_duel.native_array_conformance import (
    MODULES as MECHANISM_MODULES,
    run_rule_conformance,
)


supports_full_duel_rules = False
PROTOCOL = 'domain_soa_conformance_inputs_v1'
KEYS = tuple(COVERAGE['seven_official_presets'])
PRESENTATION_KEYS = ('events', '_public_board')
ENTITY_CLASSES = (PlayerEntity, CharacterEntity, CardEntity)
REQUIRED_PUBLIC17_MECHANISMS = (
    'pending_choice',
    'operation',
    'deferred_actions',
    'summons',
    'flags',
    'effects',
    'front_debuff',
    'alias_relations',
)
REPO_ROOT = Path(__file__).resolve().parents[5]
EVALS_ROOT = (REPO_ROOT / 'artifacts' / 'rl-evals').resolve()


def coverage():
    return {
        'supports_full_duel_rules': False,
        'complete_soa': False,
        'full_public17': False,
        'soa_rules_executed': False,
        'collector': PROTOCOL,
        'roster': 'public17',
        'public_character_ids': list(PUBLIC_CHARACTER_ORDER),
        'seven_official_presets': KEYS,
        'mechanism_modules': list(MECHANISM_MODULES),
        'required_public17_mechanisms': list(REQUIRED_PUBLIC17_MECHANISMS),
        'presentation_keys_dropped': list(PRESENTATION_KEYS),
    }


def official_seven_decks():
    decks = {item['id']: item for item in STARTER_DECKS}
    missing = [key for key in KEYS if key not in decks]
    if missing:
        raise RuntimeError('Official seven-key presets missing from STARTER_DECKS: ' + ','.join(missing))
    extra = [key for key in decks if key not in KEYS]
    if extra:
        raise RuntimeError('STARTER_DECKS has non-seven-key ids; refusing invented extras: ' + ','.join(extra))
    for key in KEYS:
        deck = decks[key]
        cards = list(deck.get('card_ids') or ())
        characters = list(deck.get('character_ids') or ())
        if len(cards) != 32 or len(characters) != 4 or len(set(characters)) != 4:
            raise RuntimeError(f'Official preset {key} is not a 4x32 public deck')
        if set(characters) - set(PUBLIC_CHARACTER_IDS):
            raise RuntimeError(f'Official preset {key} is not public17')
    return {key: deepcopy(decks[key]) for key in KEYS}


def validate_output_dir(path):
    output = Path(path).expanduser().resolve()
    try:
        output.relative_to(EVALS_ROOT)
    except ValueError as exc:
        raise ValueError('Output must be a new directory under artifacts/rl-evals') from exc
    if output == EVALS_ROOT:
        raise ValueError('Refusing to write on artifacts/rl-evals itself')
    if output.exists():
        raise FileExistsError(f'Output must be a new directory: {output}')
    return output


def validate_modules(modules):
    allowed = set(MECHANISM_MODULES)
    chosen = tuple(MECHANISM_MODULES if modules is None else modules)
    extra = [name for name in chosen if name not in allowed]
    if extra:
        raise ValueError('Refusing non-native-conformance modules as public17 evidence: ' + ','.join(extra))
    return chosen


def empty_inventory():
    return {
        'types': Counter(),
        'classes': Counter(),
        'fields': {},
        'shapes': {},
        'mechanisms': {name: {'count': 0, 'details': Counter()} for name in (
            *REQUIRED_PUBLIC17_MECHANISMS,
            'choice_phase',
            'mulligan_phase',
            'playing_phase',
            'finished_phase',
            'response_lock',
            'down_responses',
            'turn_start_pending',
            'deferred_card_marks',
        )},
        'public_characters': set(),
        'summoned_ids': set(),
        'action_types': Counter(),
        'choice_kinds': Counter(),
        'flag_keys': Counter(),
        'effect_keys': Counter(),
        'effect_ids': Counter(),
        'front_debuff_keys': Counter(),
        'dot_kinds': Counter(),
        'operation_keys': Counter(),
        'deferred_kinds': Counter(),
        'alias_group_count': 0,
        'alias_max_group': 0,
        'states_seen': 0,
        'unknown_root_keys': Counter(),
    }


def _pattern(path):
    out = []
    for part in path:
        if isinstance(part, int):
            out.append('[]')
        elif part in ('a', 'b') and out and out[-1] == 'sides':
            out.append('{side}')
        else:
            out.append(str(part))
    text = ''
    for part in out:
        if part == '[]':
            text += '[]'
        else:
            text += ('.' if text else '') + part
    return text


def _field_entry(inventory, path, type_name):
    key = _pattern(path)
    row = inventory['fields'].setdefault(key, {'count': 0, 'types': Counter()})
    row['count'] += 1
    row['types'][type_name] += 1


def _shape_entry(inventory, path, length):
    key = _pattern(path)
    row = inventory['shapes'].setdefault(key, {'count': 0, 'min': length, 'max': length})
    row['count'] += 1
    row['min'] = min(row['min'], length)
    row['max'] = max(row['max'], length)


def _mark(inventory, name, detail=None):
    row = inventory['mechanisms'].setdefault(name, {'count': 0, 'details': Counter()})
    row['count'] += 1
    if detail is not None:
        row['details'][str(detail)] += 1


def alias_groups(state):
    groups, seen = {}, {}

    def walk(value, path):
        if not isinstance(value, (dict, list, tuple)):
            return
        ident = id(value)
        if ident in seen:
            groups.setdefault(ident, [seen[ident]]).append(_path_text(path))
            return
        seen[ident] = _path_text(path)
        if isinstance(value, dict):
            for key, nested in value.items():
                if not path and key in PRESENTATION_KEYS:
                    continue
                walk(nested, path + (key,))
        else:
            for index, nested in enumerate(value):
                walk(nested, path + (index,))

    walk(state, ())
    return [paths for paths in groups.values() if len(paths) > 1]


def _path_text(path):
    text = ''
    for part in path:
        if isinstance(part, int):
            text += f'[{part}]'
        else:
            text += ('.' if text else '') + str(part)
    return text


def entity_class_map(state):
    found = {}

    def walk(value, path):
        if isinstance(value, ENTITY_CLASSES):
            found[_path_text(path)] = type(value).__name__
        if isinstance(value, dict):
            for key, nested in value.items():
                if not path and key in PRESENTATION_KEYS:
                    continue
                walk(nested, path + (key,))
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                walk(nested, path + (index,))

    walk(state, ())
    return found


def authoritative_state(state):
    return {key: value for key, value in state.items() if key not in PRESENTATION_KEYS}


def jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        out = {}
        for key, nested in value.items():
            out[key if type(key) is str else str(key)] = jsonable(nested)
        return out
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    raise TypeError(f'Unsupported snapshot type {type(value).__name__}')


def snapshot_bundle(state):
    live = authoritative_state(state)
    aliases = alias_groups(live)
    payload = jsonable(live)
    rng = live.get('rng')
    return {
        'rng': rng,
        'rng_str': str(rng),
        'phase': live.get('phase'),
        'entity_classes': entity_class_map(live),
        'aliases': aliases,
        'digest': _digest(payload),
        'state': payload,
    }


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def observe_action(inventory, action):
    if not isinstance(action, dict):
        inventory['action_types'][type(action).__name__] += 1
        return
    inventory['action_types'][str(action.get('type'))] += 1
    _field_entry(inventory, ('action',), type(action).__name__)
    for key, nested in action.items():
        _field_entry(inventory, ('action', key), type(nested).__name__)


def observe_state(inventory, state, origin='state'):
    inventory['states_seen'] += 1
    live = authoritative_state(state)
    _walk_inventory(inventory, live, ())
    phase = live.get('phase')
    if phase in ('mulligan', 'playing', 'choice', 'finished'):
        _mark(inventory, phase + '_phase', origin)
    pending = live.get('pending_choice')
    if isinstance(pending, dict) and pending:
        _mark(inventory, 'pending_choice', pending.get('kind'))
        if pending.get('kind') is not None:
            inventory['choice_kinds'][str(pending.get('kind'))] += 1
    operation = live.get('operation')
    if isinstance(operation, dict) and operation:
        _mark(inventory, 'operation', origin)
        for key in operation:
            inventory['operation_keys'][str(key)] += 1
    deferred = live.get('deferred_actions')
    if isinstance(deferred, list) and deferred:
        _mark(inventory, 'deferred_actions', len(deferred))
        for item in deferred:
            if isinstance(item, dict):
                inventory['deferred_kinds'][str(item.get('kind'))] += 1
    if live.get('_response_lock'):
        _mark(inventory, 'response_lock', origin)
    if live.get('_pending_down_responses'):
        _mark(inventory, 'down_responses', origin)
    if live.get('turn_start_pending'):
        _mark(inventory, 'turn_start_pending', origin)
    if live.get('_deferred_card_marks'):
        _mark(inventory, 'deferred_card_marks', origin)
    aliases = alias_groups(live)
    if aliases:
        _mark(inventory, 'alias_relations', len(aliases))
        inventory['alias_group_count'] += len(aliases)
        inventory['alias_max_group'] = max(inventory['alias_max_group'], max(len(group) for group in aliases))
    for key in live:
        if key not in GAME_KEYS:
            inventory['unknown_root_keys'][str(key)] += 1
    for side, team in (live.get('sides') or {}).items():
        if not isinstance(team, dict):
            continue
        flags = team.get('flags')
        if flags:
            _mark(inventory, 'flags', f'player:{side}')
            for key in flags:
                inventory['flag_keys'][str(key)] += 1
        effects = team.get('effects')
        if effects:
            _mark(inventory, 'effects', f'player:{side}')
            _observe_effects(inventory, effects)
        debuff = team.get('front_debuff')
        if debuff:
            _mark(inventory, 'front_debuff', side)
            for key, nested in debuff.items():
                inventory['front_debuff_keys'][str(key)] += 1
                if key == 'dots' and isinstance(nested, dict):
                    for kind in nested:
                        inventory['dot_kinds'][str(kind)] += 1
        for cid, hero in (team.get('characters') or {}).items():
            if not isinstance(hero, dict):
                continue
            if hero.get('summoned'):
                _mark(inventory, 'summons', cid)
                inventory['summoned_ids'].add(str(cid))
            else:
                inventory['public_characters'].add(str(hero.get('id') or cid))
            if hero.get('flags'):
                _mark(inventory, 'flags', f'character:{cid}')
                for key in hero['flags']:
                    inventory['flag_keys'][str(key)] += 1
            if hero.get('effects'):
                _mark(inventory, 'effects', f'character:{cid}')
                _observe_effects(inventory, hero['effects'])


def _observe_effects(inventory, effects):
    for effect in effects:
        if not isinstance(effect, dict):
            continue
        for key in effect:
            inventory['effect_keys'][str(key)] += 1
        if effect.get('id') is not None:
            inventory['effect_ids'][str(effect['id'])] += 1


def _walk_inventory(inventory, value, path):
    type_name = type(value).__name__
    inventory['types'][type_name] += 1
    if isinstance(value, ENTITY_CLASSES):
        inventory['classes'][type_name] += 1
    if path:
        _field_entry(inventory, path, type_name)
    if isinstance(value, dict):
        _shape_entry(inventory, path, len(value))
        for key, nested in value.items():
            if not path and key in PRESENTATION_KEYS:
                continue
            _walk_inventory(inventory, nested, path + (key,))
    elif isinstance(value, (list, tuple)):
        _shape_entry(inventory, path, len(value))
        for index, nested in enumerate(value):
            _walk_inventory(inventory, nested, path + (index,))


def freeze_inventory(inventory):
    def freeze(value):
        if isinstance(value, Counter):
            return dict(value)
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, dict):
            return {str(key): freeze(nested) for key, nested in value.items()}
        return value
    return freeze(inventory)


def missing_coverage(inventory, *, keys_seen, mechanism_modules, mechanism_commits):
    missing = []
    if set(keys_seen) != set(KEYS):
        absent = [key for key in KEYS if key not in set(keys_seen)]
        missing.append('seven_keys:' + ','.join(absent))
    seen_chars = set(inventory['public_characters'])
    absent_chars = [cid for cid in PUBLIC_CHARACTER_ORDER if cid not in seen_chars]
    if absent_chars:
        missing.append('public17:' + ','.join(absent_chars))
    if tuple(mechanism_modules) != MECHANISM_MODULES:
        missing.append('mechanism_modules_not_native_conformance_list')
    if mechanism_commits <= 0:
        missing.append('mechanism_physical_commits')
    for name in REQUIRED_PUBLIC17_MECHANISMS:
        if inventory['mechanisms'].get(name, {}).get('count', 0) <= 0:
            missing.append(name)
    return missing


def plan_report(*, games=14, max_decisions=1000, seconds=600, seed=10160000):
    return {
        'mode': 'plan',
        'protocol': PROTOCOL,
        'games': games,
        'max_decisions': max_decisions,
        'seconds': seconds,
        'seed': seed,
        'keys': list(KEYS),
        'public_character_ids': list(PUBLIC_CHARACTER_ORDER),
        'mechanism_modules': list(MECHANISM_MODULES),
        'required_public17_mechanisms': list(REQUIRED_PUBLIC17_MECHANISMS),
        'scope': 'formal-engine natural 7-key trajectories plus native-array mechanism fixture routing',
        'soa_rules_executed': False,
        'gpu_qualified': False,
        'training': False,
        'output_created': False,
        'coverage': coverage(),
    }


def collect_natural_games(output, inventory, *, games, max_decisions, seconds, seed, begun, budget):
    decks = official_seven_decks()
    directory = Path(output) / 'natural'
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    keys_seen = []
    for game in range(games):
        first = 'a' if game % 2 == 0 else 'b'
        pair = (game // 2) % len(KEYS)
        left, right = KEYS[pair], KEYS[(pair + 1) % len(KEYS)]
        game_seed = seed + game
        chooser = random.Random(game_seed ^ 0x1016)
        steps = []
        record = dict(game=game, a=left, b=right, seed=game_seed, first_side=first,
                      natural_terminal=False, decisions=0, stop=None)
        initial = None
        try:
            budget()
            state = new_game(seed=game_seed, decks={'a': decks[left], 'b': decks[right]}, first_side=first)
            for key in (left, right):
                if key not in keys_seen:
                    keys_seen.append(key)
            observe_state(inventory, state, 'natural_initial')
            initial = snapshot_bundle(state)
            for decision in range(max_decisions + 1):
                budget()
                if state['phase'] == 'finished':
                    record.update(natural_terminal=True, winner=state.get('winner'),
                                  turn=state.get('turn'), reason=state.get('reason'), stop='natural_terminal')
                    break
                if decision >= max_decisions:
                    record['stop'] = 'decision_limit'
                    break
                actor = acting_side(state)
                actions = legal_actions(clone_state(state), actor)
                playable = [item for item in actions if item.get('type') != 'concede']
                if not playable:
                    raise RuntimeError('Nonterminal formal state has no diagnostic action')
                action = chooser.choice(playable)
                observe_action(inventory, action)
                pre = snapshot_bundle(state)
                observe_state(inventory, state, 'natural_pre')
                state = apply_action(state, actor, action)
                observe_state(inventory, state, 'natural_post')
                post = snapshot_bundle(state)
                steps.append({
                    'decision': decision,
                    'side': actor,
                    'action': jsonable(action),
                    'legal_actions': jsonable(actions),
                    'legal_count': len(actions),
                    'rng_pre': pre['rng'],
                    'rng_post': post['rng'],
                    'pre': pre,
                    'post': post,
                })
                record['decisions'] += 1
        except TimeoutError:
            record['stop'] = 'deadline'
            if initial is None:
                break
        path = directory / f'game-{game:03d}.json.gz'
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            json.dump({
                'game': record,
                'initial': initial,
                'steps': steps,
            }, stream, ensure_ascii=False)
        record['path'] = str(path.relative_to(output))
        records.append(record)
        if record['stop'] == 'deadline' or time.monotonic() - begun >= seconds:
            break
    return records, keys_seen


class FormalRecordingRuntime:
    """Formal apply_action stand-in that records native-router physical commits."""

    def __init__(self, inventory, traces):
        self.inventory = inventory
        self.traces = traces

    def call(self, entry, rows):
        if entry != 'checked_apply_action':
            raise AssertionError(entry)
        packets = []
        for state, side, action in rows:
            packets.append(self._checked_apply(state, side, action))
        return packets

    def _checked_apply(self, state, side, action):
        probe = clone_state(state)
        legal_error = None
        try:
            actions = legal_actions(probe, side)
        except Exception as exc:
            actions = []
            legal_error = f'{type(exc).__name__}: {exc}'
        observe_action(self.inventory, action)
        observe_state(self.inventory, state, 'mechanism_pre')
        pre = snapshot_bundle(state)
        try:
            after = apply_action(state, side, action)
        except (ValueError, TypeError) as exc:
            self.traces.append({
                'accepted': False,
                'side': side,
                'action': jsonable(action),
                'legal_actions': jsonable(actions),
                'legal_count': len(actions),
                'legal_error': legal_error,
                'error': type(exc).__name__,
                'error_args': list(exc.args),
                'pre': pre,
            })
            return (False, type(exc).__name__, tuple(exc.args))
        observe_state(self.inventory, after, 'mechanism_post')
        post = snapshot_bundle(after)
        self.traces.append({
            'accepted': True,
            'side': side,
            'action': jsonable(action),
            'legal_actions': jsonable(actions),
            'legal_count': len(actions),
            'legal_error': legal_error,
            'rng_pre': pre['rng'],
            'rng_post': post['rng'],
            'pre': pre,
            'post': post,
        })
        return (True, after)


def collect_mechanism_cases(output, inventory, *, modules, budget):
    modules = validate_modules(modules)
    directory = Path(output) / 'mechanisms'
    traces = []
    if not modules:
        directory.mkdir(parents=True, exist_ok=True)
        report = dict(scope='no mechanism modules requested', passed=True, tests=0, skipped=0,
                      failures=0, errors=0, native_commits=0, native_rejections=0,
                      tests_with_native_commits=0, excluded=[], rows=[], gpu_qualified=False)
        (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                               encoding='utf-8')
        return report, traces
    runtime = FormalRecordingRuntime(inventory, traces)
    report = run_rule_conformance(runtime, directory, modules=modules, budget=budget)
    commits = json.loads((directory / 'commits.json').read_text(encoding='utf-8'))
    rejections = json.loads((directory / 'rejections.json').read_text(encoding='utf-8'))
    accepted = [row for row in traces if row['accepted']]
    rejected = [row for row in traces if not row['accepted']]
    if len(accepted) != len(commits) or len(rejected) != len(rejections):
        report = dict(report, trace_commit_mismatch=True,
                      trace_accepted=len(accepted), commit_rows=len(commits),
                      trace_rejected=len(rejected), rejection_rows=len(rejections))
    for row, commit in zip(accepted, commits):
        row['test'] = commit.get('test')
    for row, item in zip(rejected, rejections):
        row['test'] = item.get('test')
    with gzip.open(directory / 'traces.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for row in traces:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return report, traces


def execute(output, *, games=14, max_decisions=1000, seconds=600, seed=10160000,
            modules=None, require_coverage=True):
    if not 1 <= games <= 140 or not 1 <= max_decisions <= 10000 or not 1 <= seconds <= 3600:
        raise ValueError('Bounded games/decisions/time required')
    output = validate_output_dir(output)
    modules = validate_modules(modules)
    output.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    inventory = empty_inventory()
    failures = []

    def budget():
        if time.monotonic() - begun >= seconds:
            raise TimeoutError('SoA case collector budget exhausted')

    config = dict(protocol=PROTOCOL, games=games, max_decisions=max_decisions, seconds=seconds,
                  seed=seed, keys=list(KEYS), mechanism_modules=list(modules),
                  require_coverage=require_coverage, coverage=coverage(),
                  soa_rules_executed=False, training=False, gpu_qualified=False)
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n',
                                        encoding='utf-8')
    natural, keys_seen = [], []
    mechanisms = dict(passed=True, tests=0, skipped=0, failures=0, errors=0, native_commits=0,
                      native_rejections=0, excluded=[], rows=[])
    traces = []
    try:
        natural, keys_seen = collect_natural_games(
            output, inventory, games=games, max_decisions=max_decisions,
            seconds=seconds, seed=seed, begun=begun, budget=budget)
    except Exception as exc:
        failures.append(dict(stage='natural', type=type(exc).__name__, error=str(exc)))
    try:
        if time.monotonic() - begun < seconds:
            mechanisms, traces = collect_mechanism_cases(
                output, inventory, modules=modules, budget=budget)
        else:
            failures.append(dict(stage='mechanisms', type='TimeoutError',
                                 error='Budget exhausted before mechanism routing'))
    except Exception as extra:
        failures.append(dict(stage='mechanisms', type=type(extra).__name__, error=str(extra)))
    frozen = freeze_inventory(inventory)
    missing = missing_coverage(
        inventory,
        keys_seen=keys_seen,
        mechanism_modules=modules,
        mechanism_commits=int(mechanisms.get('native_commits') or 0),
    )
    if not require_coverage:
        missing = []
    natural_complete = (len(natural) == games and all(row.get('natural_terminal') for row in natural)
                        and not any(row.get('stop') != 'natural_terminal' for row in natural))
    passed = (not failures and not missing and natural_complete
              and bool(mechanisms.get('passed', False)))
    report = dict(
        protocol=PROTOCOL,
        passed=passed,
        actual_backend='formal_cpu_input_generator_and_native_array_conformance_router',
        formal_engine_is_input_generator=True,
        soa_rules_executed=False,
        soa_state_complete=False,
        batched_rules_complete=False,
        gpu_qualified=False,
        training=False,
        presentation_keys_explicitly_excluded=list(PRESENTATION_KEYS),
        planned_games=games,
        attempted_games=len(natural),
        natural_input_games=sum(1 for row in natural if row.get('natural_terminal')),
        physical_decisions=sum(row.get('decisions', 0) for row in natural),
        mechanism_tests=mechanisms.get('tests', 0),
        mechanism_skipped=mechanisms.get('skipped', 0),
        mechanism_failures=mechanisms.get('failures', 0),
        mechanism_errors=mechanisms.get('errors', 0),
        mechanism_commits=mechanisms.get('native_commits', 0),
        mechanism_rejections=mechanisms.get('native_rejections', 0),
        mechanism_excluded=len(mechanisms.get('excluded') or []),
        required_public17_mechanisms=list(REQUIRED_PUBLIC17_MECHANISMS),
        missing_coverage=missing,
        seconds=time.monotonic() - begun,
        games=natural,
        mechanisms=mechanisms,
        failures=failures,
        coverage=coverage(),
    )
    (output / 'inventory.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + '\n',
                                           encoding='utf-8')
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                        encoding='utf-8')
    return report
