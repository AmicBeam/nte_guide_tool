"""Evaluation-only foundation for the next independent paired-root search-label experiment.

This module validates a new seed reservation, the five-preset roster, both
initiatives, and model/build/rule identities. It can also reject incomplete or
non-finite Gumbel32 root diagnostics.

It does not reconstruct archived hidden states, invent future outcomes, or run
paired forced-action games. Runtime execution stays disabled until a sound
paired-root rollout can be implemented against existing search interfaces.
"""
from hashlib import sha256
import json
from pathlib import Path

from .build_acceptance import SEED_PHASES, SEED_WIDTH

ROSTER = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')
INITIATIVES = ('a', 'b')
SEARCH = dict(algorithm='gumbel', simulations=32, gumbel_candidates=16, terminal_horizon=3, noise=False)
ARCHIVED_SEED_PHASES = SEED_PHASES
DEFAULT_LEDGER = Path(__file__).resolve().parents[4] / 'artifacts' / 'rl-seed-ledger.json'
PAIRED_ROOT_RUNTIME = False
PAIRED_ROOT_GAP = (
    'Full-game paired forced-action rollout is not implemented. '
    'information_search.search_game always applies search_choice or a sampled '
    'root action and has no hook to force end_turn and a paying attack or card '
    'from the same root, then continue both lines under the same remaining '
    'Gumbel32 budget. Existing records still lack independent paired values.'
)


def current_rule_identity():
    from .cross_lineup import identity
    return identity()


def planned_roots(seeds):
    """Five presets, both initiatives, seeds from the new foundation reservation."""
    base = _require_int(seeds.get('foundation'), 'foundation')
    planned = []
    for index, preset in enumerate(ROSTER):
        for offset, first in enumerate(INITIATIVES):
            planned.append(dict(preset=preset, first=first, seed=base + index * 2 + offset,
                                opponent=preset, matchup='mirror'))
    if [row['first'] for row in planned] != ['a', 'b'] * len(ROSTER):
        raise ValueError('Both initiatives are required for every preset')
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


def validate_seed_reservation(seeds, ledger):
    if not isinstance(seeds, dict) or set(seeds) != set(SEED_PHASES):
        raise ValueError('Seed reservation must contain the shared ledger phases')
    values = [_require_int(seeds[phase], phase) for phase in SEED_PHASES]
    if len(set(values)) != len(values):
        raise ValueError('Duplicate seeds in reservation')
    ranges = archived_seed_ranges(ledger)
    for phase in ARCHIVED_SEED_PHASES:
        run_id = seed_in_range(seeds[phase], ranges)
        if run_id:
            raise ValueError(f'Reservation reuses archived {phase} seed from {run_id}')
    for row in planned_roots(seeds):
        run_id = seed_in_range(row['seed'], ranges)
        if run_id:
            raise ValueError(f'Planned root seed overlaps archived range {run_id}')
    return dict(seeds=dict(seeds), planned=planned_roots(seeds), archived_overlap=False)


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


def dry_run(*, seed_reservation, models, output, ledger=None, rule_hash=None):
    """Validate the next experiment. Never creates output or plays a game."""
    if PAIRED_ROOT_RUNTIME:
        raise RuntimeError('Runtime flag is set without an implemented paired-root runner')
    seeds = json.loads(Path(seed_reservation).read_text(encoding='utf-8')) if not isinstance(seed_reservation, dict) else seed_reservation
    ledger_data = ledger if isinstance(ledger, dict) else load_seed_ledger(ledger)
    output_path = validate_output_path(output)
    reservation = validate_seed_reservation(seeds, ledger_data)
    identities = validate_model_identities(models, rule_hash=rule_hash)
    if output_path.exists():
        raise ValueError('Dry-run created output')
    return dict(
        dry_run=True,
        runtime_enabled=False,
        runtime_gap=PAIRED_ROOT_GAP,
        output=str(output_path),
        output_created=False,
        games_played=0,
        roster=list(ROSTER),
        initiatives=list(INITIATIVES),
        search=dict(SEARCH),
        seeds=reservation['seeds'],
        planned=reservation['planned'],
        archived_overlap=False,
        identities=identities,
        serving_export=False,
        new_games=False,
        cuda=False,
        training=False,
    )


def _require_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'Invalid {name} seed')
    return value


def _finite(value):
    return value == value and value not in (float('inf'), float('-inf'))
