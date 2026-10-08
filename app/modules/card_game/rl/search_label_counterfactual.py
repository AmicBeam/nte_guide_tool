"""Bounded, evaluation-only paired roots for five-preset search-label checks."""
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import time

from .build_acceptance import SEED_PHASES, SEED_WIDTH

ROSTER = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')
INITIATIVES = ('a', 'b')
PAIR_ORDER = (
    ('starter', 'weave-rush'), ('quick-rush', 'zhenhong'),
    ('starter', 'murk'), ('weave-rush', 'quick-rush'),
    ('zhenhong', 'murk'), ('starter', 'quick-rush'),
    ('weave-rush', 'zhenhong'), ('quick-rush', 'murk'),
    ('starter', 'zhenhong'), ('weave-rush', 'murk'),
)
SEARCH = dict(algorithm='gumbel', simulations=32, gumbel_candidates=16, terminal_horizon=3, noise=False)
ARCHIVED_SEED_PHASES = SEED_PHASES
DEFAULT_LEDGER = Path(__file__).resolve().parents[4] / 'artifacts' / 'rl-seed-ledger.json'
PAIRED_ROOT_RUNTIME = True
PAIRED_ROOT_GAP = 'Paired roots provide bounded continuations, not full-game strength or training labels.'
MAX_WALL_SECONDS = 3600
CONTINUATION_ACTIONS = 8
WORLD_SEED_STRIDE = 104729


def current_rule_identity():
    from .cross_lineup import identity
    return identity()


def planned_roots(seeds):
    """All ten cross-pairs with both actual initiative directions."""
    base = _require_int(seeds.get('foundation'), 'foundation')
    planned = []
    if set(PAIR_ORDER) != set(combinations(ROSTER, 2)):
        raise ValueError('Cross-pair order does not cover ten unique pairs')
    for index, (left, right) in enumerate(PAIR_ORDER):
        for first in INITIATIVES:
            planned.append(dict(left=left, right=right, first=first,
                                seed=base + index, matchup='cross'))
    if [row['first'] for row in planned] != ['a', 'b'] * 10:
        raise ValueError('Both initiatives are required for every preset')
    if len(planned) != 20:
        raise ValueError('Ten cross-pairs require twenty initiative jobs')
    return planned


def load_seed_ledger(path=None):
    path = Path(path or DEFAULT_LEDGER)
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('runs'), list):
        raise ValueError('Invalid seed ledger')
    return data


def archived_seed_ranges(ledger):
    ranges = []
    for run in ledger.get('runs') or []:
        start, end = run.get('start'), run.get('end')
        run_id = run.get('run_id')
        if type(start) is int and type(end) is int and end > start:
            ranges.append(dict(run_id=run_id, start=start, end=end))
            continue
        seeds = run.get('seeds') or {}
        values = [seeds[phase] for phase in ARCHIVED_SEED_PHASES if type(seeds.get(phase)) is int]
        if values:
            ranges.append(dict(run_id=run_id, start=min(values), end=max(values) + SEED_WIDTH))
    return ranges


def seed_in_range(seed, ranges):
    return next((item['run_id'] for item in ranges if item['start'] <= seed < item['end']), None)


def validate_seed_reservation(seeds, ledger, *, run_id=None):
    if not isinstance(seeds, dict) or set(seeds) != set(SEED_PHASES):
        raise ValueError('Seed reservation must contain the shared ledger phases')
    values = [_require_int(seeds[phase], phase) for phase in SEED_PHASES]
    if len(set(values)) != len(values):
        raise ValueError('Duplicate seeds in reservation')
    ranges = archived_seed_ranges(ledger)
    own = None
    if run_id is not None:
        matches = [row for row in ledger.get('runs', []) if row.get('run_id') == run_id]
        if len(matches) != 1 or matches[0].get('seeds') != seeds:
            raise ValueError('Seed reservation must match exactly one ledger run')
        own = run_id
    for phase in ARCHIVED_SEED_PHASES:
        overlapping = seed_in_range(seeds[phase], ranges)
        if overlapping and overlapping != own:
            raise ValueError(f'Reservation reuses archived {phase} seed from {overlapping}')
    for row in planned_roots(seeds):
        overlapping = seed_in_range(row['seed'], ranges)
        if overlapping and overlapping != own:
            raise ValueError(f'Planned root seed overlaps archived range {overlapping}')
    return dict(seeds=dict(seeds), planned=planned_roots(seeds), archived_overlap=False, run_id=own)


def read_model_identity(directory, key):
    root = Path(directory)
    manifest_path = root / f'{key}.json'
    weights_path = root / f'{key}.npz'
    if not manifest_path.is_file() or not weights_path.is_file():
        raise ValueError(f'Missing model files for {key}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    digest = sha256(weights_path.read_bytes()).hexdigest()
    build = manifest.get('build') or {}
    if set(('character_ids', 'card_ids')) - set(build):
        raise ValueError(f'Build identity missing for {key}')
    build_digest = sha256(json.dumps(
        {name: build[name] for name in ('character_ids', 'card_ids')},
        sort_keys=True, separators=(',', ':'),
    ).encode()).hexdigest()
    return dict(
        key=key,
        schema=manifest.get('schema'),
        rule_hash=manifest.get('rule_hash'),
        sha256=manifest.get('sha256'),
        file_sha256=digest,
        build_sha256=manifest.get('build_sha256'),
        computed_build_sha256=build_digest,
        candidate_transform_sha256=manifest.get('candidate_transform_sha256'),
        deck=manifest.get('deck'),
        character_ids=list(build.get('character_ids') or []),
        card_count=len(build.get('card_ids') or []),
        build=dict(character_ids=list(build.get('character_ids') or []),
                   card_ids=list(build.get('card_ids') or [])),
    )


def validate_model_identities(directory, *, rule_hash=None):
    expected = rule_hash if rule_hash is not None else current_rule_identity()
    identities = {}
    for key in ROSTER:
        item = read_model_identity(directory, key)
        if item['deck'] != key:
            raise ValueError(f'Model deck identity mismatch for {key}')
        if item['sha256'] != item['file_sha256']:
            raise ValueError(f'Model weight SHA mismatch for {key}')
        if item['rule_hash'] != expected:
            raise ValueError(f'Rule identity mismatch for {key}')
        if item['schema'] != 'cross_five_grounded_wdl_v1':
            raise ValueError(f'Unsupported model schema for {key}')
        if item['build_sha256'] != item['computed_build_sha256']:
            raise ValueError(f'Build SHA mismatch for {key}')
        from .cross_grounded import fingerprint
        if item['candidate_transform_sha256'] != fingerprint():
            raise ValueError(f'Candidate transform mismatch for {key}')
        if len(item['character_ids']) != 4 or len(set(item['character_ids'])) != 4 or item['card_count'] != 32:
            raise ValueError(f'Build identity missing or incomplete for {key}')
        identities[key] = item
    return dict(rule_hash=expected, models=identities, roster=list(ROSTER))


def validate_output_path(path):
    output = Path(path)
    if output.exists():
        raise ValueError('Output must be new; dry-run does not create it')
    return output


def jsonable(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def accept_root_diagnostics(result):
    """Reject incomplete or non-finite Gumbel32 diagnostics. Selection is not a verdict."""
    if not result or not result.get('complete'):
        raise ValueError('Incomplete search cannot label a root')
    diagnostics = result.get('root_diagnostics')
    if not diagnostics:
        raise ValueError('Root diagnostics are required')
    requested = int(diagnostics['requested_simulations'])
    completed = int(diagnostics['completed_simulations'])
    if requested != completed or completed != int(result.get('simulations', -1)):
        raise ValueError('Requested and completed simulation counts disagree')
    if requested != SEARCH['simulations'] and len(result['actions']) != 1:
        raise ValueError('Gumbel32 evaluation requires 32 completed simulations')
    identities = list(diagnostics['action_identities'])
    if len(identities) != len(result['actions']):
        raise ValueError('Diagnostics omitted a legal action identity')
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate legal action identities')
    prior = [float(value) for value in diagnostics['prior']]
    visits = [int(value) for value in diagnostics['visits']]
    completed_q = [float(value) for value in diagnostics['completed_q']]
    admitted = list(diagnostics['admitted_to_max16'])
    if not (len(prior) == len(visits) == len(completed_q) == len(admitted) == len(identities)):
        raise ValueError('Diagnostic vectors must cover every legal action')
    if any(not _finite(value) or value < 0 for value in prior):
        raise ValueError('Non-finite or negative prior')
    if abs(sum(prior) - 1) > 1e-6:
        raise ValueError('Priors must sum to one')
    if any(value < 0 for value in visits):
        raise ValueError('Negative visit count')
    admitted_indexes = [index for index, flag in enumerate(admitted) if flag]
    if any(not _finite(completed_q[index]) for index in admitted_indexes):
        raise ValueError('Non-finite completed Q on an admitted candidate')
    selected = int(diagnostics['selected_index'])
    if not 0 <= selected < len(identities):
        raise ValueError('Selected action is not legal')
    if diagnostics.get('selected_identity') != identities[selected]:
        raise ValueError('Selected identity does not match the legal list')
    cap = int(diagnostics['candidate_cap'])
    if cap > SEARCH['gumbel_candidates']:
        raise ValueError('Candidate admission exceeded max16')
    if len(admitted_indexes) > cap:
        raise ValueError('More actions were admitted than the Gumbel cap')
    for index, flag in enumerate(admitted):
        if flag and visits[index] <= 0:
            raise ValueError('Admitted candidate has no visits')
        if not flag and visits[index] > 0 and len(identities) > cap:
            raise ValueError('Visited action missing from max16 admission')
    payload = jsonable(diagnostics)
    payload['pi_top'] = int(max(range(len(result['pi'])), key=lambda index: float(result['pi'][index])))
    payload['selected_is_not_a_verdict'] = True
    return payload


def prepare_run(*, seed_reservation, models, output, ledger=None, rule_hash=None, run_id=None,
                seconds=MAX_WALL_SECONDS, max_games=20, max_roots=20):
    """Validate identities and bounds. Never creates output or plays a game."""
    if type(seconds) is not int or not 1 <= seconds <= MAX_WALL_SECONDS:
        raise ValueError('Wall time must be a positive integer <= 3600s')
    if type(max_games) is not int or max_games < 1:
        raise ValueError('max_games must be a positive integer')
    if type(max_roots) is not int or max_roots < 1:
        raise ValueError('max_roots must be a positive integer')
    seeds = json.loads(Path(seed_reservation).read_text(encoding='utf-8')) if not isinstance(seed_reservation, dict) else seed_reservation
    ledger_data = ledger if isinstance(ledger, dict) else load_seed_ledger(ledger)
    output_path = validate_output_path(output)
    reservation = validate_seed_reservation(seeds, ledger_data, run_id=run_id)
    identities = validate_model_identities(models, rule_hash=rule_hash)
    if output_path.exists():
        raise ValueError('Dry-run created output')
    return dict(
        seeds=reservation['seeds'],
        planned=reservation['planned'],
        archived_overlap=False,
        run_id=reservation.get('run_id'),
        identities=identities,
        output=output_path,
        seconds=seconds,
        max_games=max_games,
        max_roots=max_roots,
        models=str(models),
        search=dict(SEARCH),
    )


def dry_run(*, seed_reservation, models, output, ledger=None, rule_hash=None, run_id=None):
    """Validate the next experiment. Never creates output or plays a game."""
    prepared = prepare_run(seed_reservation=seed_reservation, models=models, output=output,
                           ledger=ledger, rule_hash=rule_hash, run_id=run_id)
    output_path = prepared['output']
    return dict(
        dry_run=True,
        runtime_enabled=PAIRED_ROOT_RUNTIME,
        runtime_gap=PAIRED_ROOT_GAP,
        output=str(output_path),
        output_created=False,
        games_played=0,
        roster=list(ROSTER),
        initiatives=list(INITIATIVES),
        search=dict(SEARCH),
        seeds=prepared['seeds'],
        planned=prepared['planned'],
        archived_overlap=False,
        identities=prepared['identities'],
        serving_export=False,
        new_games=False,
        cuda=False,
        training=False,
        optimizer=False,
        full_game_win_rate=False,
    )


def root_is_eligible(state, actor, actions, *, action_cost):
    if not state or state.get('phase') != 'playing':
        return False
    team = (state.get('sides') or {}).get(actor) or {}
    if int(team.get('ap', -1)) != 2:
        return False
    if not any(action.get('type') == 'end_turn' for action in actions):
        return False
    return any(_paid_attack_or_card(action, action_cost(state, actor, action)) for action in actions)


def select_paid_alternative(state, actor, result, diagnostics, *, observe, action_cost):
    """Choose a paying attack or card from the acting player's visible information only."""
    view = observe(state, actor)
    if _hidden_cards_exposed(view, actor):
        raise ValueError('Paid alternative used hidden opponent cards')
    actions = list(result['actions'])
    admitted = list(diagnostics['admitted_to_max16'])
    prior = [float(value) for value in diagnostics['prior']]
    paid = []
    for index, action in enumerate(actions):
        cost = int(action_cost(state, actor, action, view))
        if _paid_attack_or_card(action, cost):
            paid.append((index, prior[index], -index, cost))
    if not paid:
        raise ValueError('Eligible root lost its paying alternative')
    admitted_paid = [item for item in paid if admitted[item[0]]]
    excluded = not admitted_paid
    chosen = max(admitted_paid or paid, key=lambda item: (item[1], item[2]))
    index, _prior, _tie, cost = chosen
    return dict(
        action=deepcopy(actions[index]),
        identity=diagnostics['action_identities'][index],
        index=index,
        ap_cost=cost,
        excluded_top16=excluded,
        visible_only=True,
    )


def pair_forced_root(state, actor, result, diagnostics, *, sample_world, apply_action, clean,
                     observe, action_cost, snapshot, world_seed, continuation, deadline=None):
    """Force end_turn and one paying alternative on identical information-set clones."""
    end_action = next((action for action in result['actions'] if action.get('type') == 'end_turn'), None)
    if end_action is None:
        return _unpaired('missing_end_turn', diagnostics=diagnostics)
    try:
        paid = select_paid_alternative(state, actor, result, diagnostics, observe=observe, action_cost=action_cost)
    except ValueError as exc:
        return _unpaired(str(exc), diagnostics=diagnostics)
    world = sample_world(state, actor, world_seed)
    end_world = deepcopy(world)
    paid_world = deepcopy(world)
    if end_world.get('rng') != paid_world.get('rng'):
        return _unpaired('cloned_rng_mismatch', diagnostics=diagnostics, excluded_top16=paid['excluded_top16'])
    before = snapshot(world)
    try:
        after_end = clean(apply_action(end_world, actor, end_action))
        after_paid = clean(apply_action(paid_world, actor, paid['action']))
    except Exception as exc:
        return _unpaired(f'illegal_forced_action:{exc}', diagnostics=diagnostics, excluded_top16=paid['excluded_top16'])
    if after_end is after_paid or after_end is world or after_paid is world:
        return _unpaired('branches_share_state', diagnostics=diagnostics, excluded_top16=paid['excluded_top16'])
    end_continued = continuation(after_end, 'end_turn', deadline)
    paid_continued = continuation(after_paid, 'paid', deadline)
    if not end_continued.get('complete') or not paid_continued.get('complete'):
        return _unpaired('incomplete_continuation', diagnostics=diagnostics, excluded_top16=paid['excluded_top16'],
                         immediate=_immediate(before, snapshot(after_end), snapshot(after_paid), paid, actor))
    end_final = end_continued['state']
    paid_final = paid_continued['state']
    immediate = _immediate(before, snapshot(after_end), snapshot(after_paid), paid, actor)
    continued = dict(
        end_turn=_delta(before, snapshot(end_final)),
        paid=_delta(before, snapshot(paid_final)),
        steps={'end_turn': end_continued.get('steps', 0), 'paid': paid_continued.get('steps', 0)},
        terminal={'end_turn': _terminal(end_final), 'paid': _terminal(paid_final)},
        diverged=_diverged(snapshot(end_final), snapshot(paid_final)),
    )
    return dict(
        paired=True,
        invalid_reason=None,
        actor=actor,
        ap=int((state.get('sides') or {}).get(actor, {}).get('ap', -1)),
        world_seed=world_seed,
        cloned_rng=world.get('rng'),
        cloned_rng_identical=True,
        diagnostics=diagnostics,
        search_selected=dict(index=int(diagnostics['selected_index']), identity=diagnostics['selected_identity']),
        forced_end_turn=end_action,
        forced_paid=dict(action=paid['action'], identity=paid['identity'], index=paid['index'], ap_cost=paid['ap_cost']),
        excluded_top16=bool(paid['excluded_top16']),
        visible_only=True,
        opponent_observation_only=True,
        immediate=immediate,
        continuation=continued,
        full_game_win_rate=False,
    )


def evaluate_paired_roots(*, seed_reservation, models, output, ledger=None, rule_hash=None, run_id=None,
                          seconds=MAX_WALL_SECONDS, max_games=20, max_roots=20, hooks=None,
                          continuation_actions=CONTINUATION_ACTIONS, require_end_selected=True,
                          clock=time.time):
    """Bounded evaluation-only pairing. Never trains, exports, or claims a win rate."""
    if not PAIRED_ROOT_RUNTIME:
        raise RuntimeError('Paired-root runtime is disabled. ' + PAIRED_ROOT_GAP)
    if hooks is None:
        hooks = real_hooks()
    prepared = prepare_run(seed_reservation=seed_reservation, models=models, output=output,
                           ledger=ledger, rule_hash=rule_hash, run_id=run_id,
                           seconds=seconds, max_games=max_games, max_roots=max_roots)
    prepared['require_end_selected'] = bool(require_end_selected)
    output_path = prepared['output']
    output_path.mkdir(parents=True, exist_ok=False)
    jsonl_path = output_path / 'roots.jsonl'
    started = clock()
    deadline = started + prepared['seconds']
    totals = dict(planned_jobs=len(prepared['planned']), games_started=0, games_played=0, roots_considered=0,
                  roots_eligible=0, roots_end_selected=0, roots_paired=0, roots_unpaired=0, excluded_top16=0,
                  incomplete_search=0, wall_limited=False, illegal_search=0)
    records = []
    try:
        for job in prepared['planned']:
            if totals['roots_paired'] + totals['roots_unpaired'] >= prepared['max_roots']:
                break
            if totals['games_started'] >= prepared['max_games']:
                break
            if clock() >= deadline:
                totals['wall_limited'] = True
                break
            record = _evaluate_job(job, prepared, hooks, clock, deadline, continuation_actions,
                                   require_end_selected, totals)
            records.append(record)
            _append_jsonl(jsonl_path, record)
            if record.get('wall_limited'):
                totals['wall_limited'] = True
                break
    except Exception:
        _write_json(output_path / 'report.json', _report(prepared, totals, records, output_path, started, clock(), failed=True))
        raise
    report = _report(prepared, totals, records, output_path, started, clock())
    _write_json(output_path / 'report.json', report)
    return report


def real_hooks():
    from app.modules.card_game.engine.duel_v2 import acting_side, apply_action, new_game, observe
    from app.modules.card_game.engine.duel_v2.projection import preview
    from app.modules.card_game.rl import cross_runtime as runtime
    from app.modules.card_game.rl.information_search import search, sample_world
    from app.modules.card_game.rl.league_rollout import clean

    def load_policies(directory, left, right):
        return {'a': runtime.model(directory, left), 'b': runtime.model(directory, right)}

    def action_cost(state, side, action, view=None):
        return int(preview(state, side, action).get('ap_cost') or 0)

    def choose_visible(state, side, policies):
        actions, (encoded, encoded_actions) = runtime.decision(state, side)
        scores = runtime.action_scores(policies[side], encoded, encoded_actions)
        return actions[int(scores.argmax())]

    def do_search(state, viewer, policies, **kwargs):
        settings = dict(SEARCH)
        settings.update(kwargs)
        settings.setdefault('runtime', runtime)
        settings.setdefault('diagnostics', True)
        return search(state, viewer, policies, **settings)

    return dict(
        new_game=new_game,
        acting_side=acting_side,
        apply_action=apply_action,
        clean=clean,
        observe=lambda state, side: observe(state, side, include_previews=False),
        search=do_search,
        sample_world=lambda state, viewer, seed: sample_world(state, viewer, seed, runtime=runtime),
        load_policies=load_policies,
        action_cost=action_cost,
        choose_visible=choose_visible,
        snapshot=resource_snapshot,
    )


def resource_snapshot(state):
    payload = dict(phase=state.get('phase'), winner=state.get('winner'), rng=state.get('rng'), sides={})
    for side, team in (state.get('sides') or {}).items():
        characters = team.get('characters') or {}
        if isinstance(characters, dict):
            items = characters.items()
        else:
            items = [(item.get('id'), item) for item in characters]
        payload['sides'][side] = dict(
            hp=team.get('hp'),
            shield=team.get('shield'),
            ap=team.get('ap'),
            front=team.get('front'),
            characters={cid: dict(hp=hero.get('hp'), shield=hero.get('shield'),
                                  alive=int(hero.get('hp') or 0) > 0)
                        for cid, hero in items if cid},
        )
    return payload


def _evaluate_job(job, prepared, hooks, clock, deadline, continuation_actions,
                  require_end_selected, totals):
    totals['games_started'] += 1
    policies = hooks['load_policies'](prepared['models'], job['left'], job['right'])
    builds = {
        'a': policies['a'].serving_deck if hasattr(policies['a'], 'serving_deck') else prepared['identities']['models'][job['left']]['build'],
        'b': policies['b'].serving_deck if hasattr(policies['b'], 'serving_deck') else prepared['identities']['models'][job['right']]['build'],
    }
    state = hooks['new_game'](seed=job['seed'], first_side=job['first'], decks=builds)
    totals['games_played'] += 1
    record = dict(job=dict(job), paired=False, invalid_reason='no_eligible_root', full_game_win_rate=False)
    for step in range(400):
        if clock() >= deadline:
            totals['wall_limited'] = True
            totals['roots_unpaired'] += 1
            record['wall_limited'] = True
            record['invalid_reason'] = 'wall_limit'
            return record
        if state.get('phase') == 'finished':
            totals['roots_unpaired'] += 1
            record['invalid_reason'] = 'game_finished_without_root'
            return record
        actor = hooks['acting_side'](state)
        search_seed = job['seed'] + WORLD_SEED_STRIDE * (step + 1)
        result = hooks['search'](state, actor, policies, seed=search_seed, deadline=deadline, diagnostics=True)
        totals['roots_considered'] += 1
        if not result or not result.get('complete'):
            totals['incomplete_search'] += 1
            totals['roots_unpaired'] += 1
            record['invalid_reason'] = 'incomplete_search'
            if clock() >= deadline:
                totals['wall_limited'] = True
                record['wall_limited'] = True
                record['invalid_reason'] = 'wall_limit'
                return record
            raise ValueError('Incomplete search cannot label a root')
        diagnostics = accept_root_diagnostics(result)
        if not root_is_eligible(state, actor, result['actions'],
                                action_cost=lambda current, side, action, view=None: hooks['action_cost'](current, side, action, view)):
            chosen = result['actions'][int(result['search_choice'])]
            state = hooks['clean'](hooks['apply_action'](state, actor, chosen))
            continue
        totals['roots_eligible'] += 1
        selected = result['actions'][int(result['search_choice'])]
        if selected.get('type') == 'end_turn':
            totals['roots_end_selected'] += 1
        elif require_end_selected:
            state = hooks['clean'](hooks['apply_action'](state, actor, selected))
            continue
        def continuation(branch, _label, branch_deadline):
            return _continue_branch(
                branch, policies, actor=actor, hooks=hooks, deadline=branch_deadline,
                steps=continuation_actions, seed=search_seed, clock=clock,
            )
        paired = pair_forced_root(
            state, actor, result, diagnostics,
            sample_world=hooks['sample_world'],
            apply_action=hooks['apply_action'],
            clean=hooks['clean'],
            observe=hooks['observe'],
            action_cost=lambda current, side, action, view=None: hooks['action_cost'](current, side, action, view),
            snapshot=hooks.get('snapshot', resource_snapshot),
            world_seed=search_seed + 17,
            continuation=continuation,
            deadline=deadline,
        )
        paired.update(job=dict(job), search_seed=search_seed, step=step)
        if paired.get('excluded_top16'):
            totals['excluded_top16'] += 1
        if paired.get('paired'):
            totals['roots_paired'] += 1
        else:
            totals['roots_unpaired'] += 1
        return paired
    totals['roots_unpaired'] += 1
    record['invalid_reason'] = 'action_limit'
    return record


def _continue_branch(state, policies, *, actor, hooks, deadline, steps, seed, clock):
    current = state
    used = 0
    for index in range(steps):
        if current.get('phase') == 'finished':
            break
        if clock() >= deadline:
            return dict(complete=False, reason='wall_limit', state=current, steps=used)
        side = hooks['acting_side'](current)
        result = hooks['search'](current, side, policies, seed=seed + 97 * (index + 1),
                                 deadline=deadline, diagnostics=True,
                                 simulations=SEARCH['simulations'])
        if not result or not result.get('complete'):
            return dict(complete=False, reason='incomplete_search', state=current, steps=used)
        accept_root_diagnostics(result)
        action = result['actions'][int(result['search_choice'])]
        current = hooks['clean'](hooks['apply_action'](current, side, action))
        used += 1
    return dict(complete=True, state=current, steps=used)


def _report(prepared, totals, records, output_path, started, finished, failed=False):
    roots_path = output_path / 'roots.jsonl'
    return dict(
        complete=not failed and not totals['wall_limited']
                 and totals['games_started'] == len(prepared['planned']),
        bounded_run_exited=not failed and not totals['wall_limited'],
        dry_run=False,
        runtime_enabled=True,
        runtime_gap=PAIRED_ROOT_GAP,
        output=str(output_path),
        output_created=True,
        jsonl=str(roots_path),
        roots_sha256=sha256(roots_path.read_bytes()).hexdigest() if roots_path.exists() else None,
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed=finished - started,
        seconds_limit=prepared['seconds'],
        roster=list(ROSTER),
        initiatives=list(INITIATIVES),
        search=dict(SEARCH),
        seeds=prepared['seeds'],
        planned=len(prepared['planned']),
        require_end_selected=prepared['require_end_selected'],
        identities=prepared['identities'],
        denominators=dict(totals),
        roots=len(records),
        planned_job_fraction=totals['games_started'] / len(prepared['planned']),
        paired_identities=[row.get('forced_paid', {}).get('identity') for row in records if row.get('paired')],
        excluded_top16=[row.get('forced_paid', {}).get('identity') for row in records if row.get('excluded_top16')],
        serving_export=False,
        new_games=totals['games_started'] > 0,
        cuda=False,
        training=False,
        optimizer=False,
        full_game_win_rate=False,
        failed=failed,
    )


def _append_jsonl(path, record):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(jsonable(record), ensure_ascii=False) + '\n')


def _write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _paid_attack_or_card(action, cost):
    return action.get('type') in ('attack', 'play_card') and int(cost or 0) >= 1


def _hidden_cards_exposed(view, actor):
    foe = 'b' if actor == 'a' else 'a'
    for card in ((view.get('sides') or {}).get(foe) or {}).get('hand') or []:
        if isinstance(card, dict) and card.get('card_id') and not card.get('hidden') and not card.get('revealed') and not card.get('copy'):
            return True
    return False


def _unpaired(reason, **extra):
    payload = dict(paired=False, invalid_reason=reason, full_game_win_rate=False)
    payload.update(extra)
    return payload


def _immediate(before, after_end, after_paid, paid, actor):
    attacker = (paid['action'] or {}).get('character_id')
    paid_chars = ((after_paid.get('sides') or {}).get(actor) or {}).get('characters') or {}
    survived = None if not attacker else bool((paid_chars.get(attacker) or {}).get('alive'))
    return dict(
        end_turn=_delta(before, after_end),
        paid=_delta(before, after_paid),
        ap_spend=paid.get('ap_cost'),
        attacker_id=attacker,
        attacker_survived=survived,
        diverged=_diverged(after_end, after_paid),
        terminal={'end_turn': _terminal_from_snapshot(after_end), 'paid': _terminal_from_snapshot(after_paid)},
    )


def _delta(before, after):
    payload = {}
    for side in set(before.get('sides') or {}) | set(after.get('sides') or {}):
        left = (before.get('sides') or {}).get(side) or {}
        right = (after.get('sides') or {}).get(side) or {}
        payload[side] = dict(
            hp=(right.get('hp') or 0) - (left.get('hp') or 0),
            shield=(right.get('shield') or 0) - (left.get('shield') or 0),
            ap=(right.get('ap') or 0) - (left.get('ap') or 0),
        )
    payload['phase'] = after.get('phase')
    payload['winner'] = after.get('winner')
    return payload


def _diverged(left, right):
    return jsonable(left) != jsonable(right)


def _terminal(state):
    return dict(phase=state.get('phase'), winner=state.get('winner'), finished=state.get('phase') == 'finished')


def _terminal_from_snapshot(snapshot):
    return dict(phase=snapshot.get('phase'), winner=snapshot.get('winner'), finished=snapshot.get('phase') == 'finished')


def _require_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'Invalid {name} seed')
    return value


def _finite(value):
    return value == value and value not in (float('inf'), float('-inf'))
