"""Fail-closed build acceptance. No inference, training, or writes on import."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time

SCHEMA = 'paired_build_acceptance_v1'
SEED_WIDTH = 10_000_000
SEED_PHASES = ('foundation', 'probes', 'selection', 'search', 'adaptation', 'confirmation', 'audit')


def canonical_build(build):
    from app.modules.card_game.content.duel_v2 import validate_deck
    result = validate_deck(deepcopy(build))
    result['card_ids'] = sorted(result['card_ids'])
    return result


def build_identity(build):
    b = canonical_build(build)
    return sha256(json.dumps({'character_ids': b['character_ids'],
                              'cards': sorted(Counter(b['card_ids']).items())},
                             separators=(',', ':')).encode()).hexdigest()


def canonical_candidates(builds):
    result = []; seen = set()
    for build in builds:
        b = canonical_build(build); identity = build_identity(b)
        if identity not in seen:
            seen.add(identity); result.append(b)
    return result


def claim_seeds(path, run_id):
    """Persistent, exclusive reservation; aborted runs never return holdout seeds."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + '.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        data = json.loads(path.read_text()) if path.exists() else {'runs': []}
        if any(r['run_id'] == run_id for r in data['runs']):
            raise ValueError('Run already reserved; do not reuse confirmation/audit seeds')
        start = max([r['end'] for r in data['runs']] + [10_000_000_000])
        end = start + len(SEED_PHASES) * SEED_WIDTH
        if end >= 2**63: raise ValueError('Seed ledger exhausted')
        seeds = {phase: start + i * SEED_WIDTH for i, phase in enumerate(SEED_PHASES)}
        data['runs'].append(dict(run_id=run_id, start=start, end=end, seeds=seeds,
                                 reserved=time.time()))
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(data, indent=2)); temporary.replace(path)
        return seeds
    finally:
        os.close(fd); lock.unlink()


def phase_deadlines(start, seconds):
    if not 600 <= seconds <= 21600: raise ValueError('Total budget must be 600..21600 seconds')
    # Confirmation, audit, and shutdown are reserved before any learning starts.
    fractions = {'foundation': .25, 'search': .15, 'baseline_adaptation': .15,
                 'candidate_adaptation': .15, 'confirmation': .12, 'audit': .15, 'shutdown': .03}
    end = start; result = {}
    for phase, fraction in fractions.items():
        end += seconds * fraction; result[phase] = end
    result['shutdown'] = start + seconds
    return result


def paired_result(rows, *, expected_seeds, opponents, alpha=.05, family_size=3, minimum_gain=.02):
    """Seed-clustered equal-opponent/equal-seat bounded empirical Bernstein interval.

    Fixed sample endpoint and family correction are declared before evaluating.
    Incomplete/duplicate/mismatched results never supply evidence for promotion.
    One observation is the mean paired outcome difference over all opponent/seat
    cells for ONE seed, not each correlated game treated as independent.
    """
    if not 0 < alpha < 1 or family_size < 1 or minimum_gain < 0:
        raise ValueError('Invalid predeclared gate')
    seeds = tuple(expected_seeds); opponents = tuple(opponents)
    if len(seeds) < 32 or len(set(seeds)) != len(seeds) or not opponents or len(set(opponents)) != len(opponents):
        raise ValueError('At least 32 distinct seeds and distinct opponents required')
    expected = {(v, seed, foe, first) for v in ('baseline', 'candidate')
                for seed in seeds for foe in opponents for first in ('a', 'b')}
    observed = {}
    for row in rows:
        k = (row['variant'], row['seed'], row['foe'], row['first'])
        if k not in expected or k in observed: return dict(complete=False, decision='incomplete', reason='identity_or_duplicate')
        if not row.get('complete') or row.get('error') or row.get('winner') not in ('a','b','draw'):
            return dict(complete=False, decision='incomplete', reason='unfinished_or_invalid_game')
        observed[k] = int(row['winner'] == 'a')
    if set(observed) != expected: return dict(complete=False, decision='incomplete', reason='missing_games')
    differences = []; groups = {}
    for foe in opponents:
        for first in ('a','b'):
            b = sum(observed['baseline', seed, foe, first] for seed in seeds)
            c = sum(observed['candidate', seed, foe, first] for seed in seeds)
            groups[f'{foe}/{first}'] = dict(baseline_wins=b, candidate_wins=c, n=len(seeds))
    for seed in seeds:
        differences.append(sum(observed['candidate',seed,foe,first]-observed['baseline',seed,foe,first]
                               for foe in opponents for first in ('a','b'))/(2*len(opponents)))
    n = len(differences); mean = sum(differences)/n
    variance = sum((d-mean)**2 for d in differences)/(n-1)
    # Maurer-Pontil (2009), Thm 4; rescale [0,1] to [-1,1].
    # Two tails plus family correction: log(4 * family_size / alpha).
    logterm = math.log(4/(alpha/family_size))
    radius = math.sqrt(2*variance*logterm/n) + 14*logterm/(3*(n-1))
    lower, upper = max(-1., mean-radius), min(1., mean+radius)
    decision = 'accepted' if lower > minimum_gain else 'rejected' if upper < 0 else 'inconclusive'
    return dict(complete=True, decision=decision, delta=mean, lower=lower, upper=upper,
                seed_clusters=n, improved=sum(d>0 for d in differences),
                regressed=sum(d<0 for d in differences), tied=sum(d==0 for d in differences),
                groups=groups, alpha=alpha, family_size=family_size, minimum_gain=minimum_gain,
                method='bounded_seed_cluster_empirical_bernstein_v1')


def delivery_decision(baseline, proposal, confirmation, audit, *, fair_adaptation, tactical_passed,
                      smoke=False, audit_margin=.05):
    """One locked candidate, one audit. Audit can veto, never select another candidate."""
    selected = canonical_build(baseline)
    status = 'retained_baseline'; reason = 'confirmation_not_accepted'
    if smoke: reason = 'smoke_is_never_quality_evidence'
    elif not fair_adaptation: reason = 'unequal_or_incomplete_adaptation'
    elif build_identity(baseline) == build_identity(proposal): reason = 'no_build_change'
    elif confirmation.get('complete') and confirmation.get('decision') == 'accepted':
        if not audit.get('complete'): reason = 'audit_incomplete'
        elif not tactical_passed: reason = 'tactical_gate_failed'
        elif audit['delta'] < 0 or audit['lower'] <= -audit_margin: reason = 'audit_regression_or_insufficient_evidence'
        else:
            selected = canonical_build(proposal); status = 'accepted_candidate'; reason = 'confirmed_and_audited'
    return dict(schema=SCHEMA, recommendation_status=status, build_quality_approved=status=='accepted_candidate',
                recommended_build=selected, candidate_build=canonical_build(proposal), reason=reason,
                confirmation=confirmation, audit=audit, automatic_serving_approval=False)


def source_disposition(directory):
    """Run-local revocation survives under verified/ without rewriting signed evidence."""
    directory = Path(directory).resolve()
    for parent in (directory, *directory.parents):
        p = parent/'disposition.json'
        if p.is_file():
            local=json.loads(p.read_text())
            if local.get('default_reuse_allowed') is False:return local
    registry=Path(__file__).with_name('experiment_dispositions.json')
    if registry.is_file():
        hashes=set()
        for name in ('starter','weave-rush','quick-rush','zhenhong','midrange'):
            p=directory/(name+'.json')
            if p.is_file():hashes.add(json.loads(p.read_text()).get('sha256'))
        for record in json.loads(registry.read_text()).get('records',[]):
            if hashes.intersection(record.get('weights',{})):
                return record
    return None


def check_source(directory, *, research_warm_start=False, baseline_builds=None):
    disposition = source_disposition(directory)
    if disposition and disposition.get('reuse_revoked'):
        raise ValueError('User revoked reuse of this experiment')
    if disposition and disposition.get('default_reuse_allowed') is False:
        if not research_warm_start or baseline_builds is None:
            raise ValueError('Discarded experiment: explicit research warm start and baseline builds required')
    return disposition
