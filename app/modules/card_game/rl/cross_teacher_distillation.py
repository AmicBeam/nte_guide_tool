"""Isolated old-teacher behavior preservation for starter, weave-rush and quick-rush.

Maps named `fixed_ten_v1` observation and legal-action columns onto current
`cross_five_grounded_wdl_v1` roots. It does not rewrite old rule hashes, does
not read hidden opponent cards, and does not train or export a serving model.
Zhenhong and murk have no old teacher and cannot be claimed improved.
"""
from hashlib import sha256
from pathlib import Path
import json

from .cross_lineup import CARD_IDS, SCHEMA as CURRENT_SCHEMA, all_features, candidate_names
from .fixed_lineup import CARD_IDS as OLD_CARD_IDS, SEATS as OLD_SEATS
from .fixed_lineup import CAND_DIM as OLD_CAND_DIM, SCHEMA as OLD_SCHEMA, candidate_names as old_candidate_names
from .fixed_lineup import all_features as old_all_features, tensor_shapes
from .league_schema import ACT_ATTACK, ACT_END, ACT_PLAY, build_hash

TEAMS = ('starter', 'weave-rush', 'quick-rush')
NO_OLD_TEACHER = ('zhenhong', 'murk')
KL_GRID = (0.0, 0.05, 0.1, 0.2)
MAX_KL = 1.0
MAX_TEACHER_TOP_WEIGHT = 5.0
DAMAGE_IMMUNE_GAP = 20
REPO_TEACHERS = Path(__file__).resolve().parents[1] / 'engine' / 'ai' / 'models'
ARCHIVED_EPISODES = (
    Path(__file__).resolve().parents[4]
    / 'artifacts' / 'rl-evals' / 'five-cross-20260924' / 'formal' / 'episodes'
)
ACCEPTANCE_BOUNDARY = {
    'starter': dict(new_wins=0, games=128),
    'weave-rush': dict(new_wins=0, games=128),
    'quick-rush': dict(new_wins=19, games=128),
}
ACCEPTANCE_NOTE = (
    'New cross models lost 0/128, 0/128 and 19/128 to old adapted strategies '
    'in artifacts/rl-evals/five-cross-20260924/repo-compare-pure-32/report.md. '
    'That comparison is an acceptance boundary, not a serving approval.'
)


def current_features():
    return all_features()


def current_candidates():
    return candidate_names()


def named_index_map(source_names, target_names):
    if len(set(source_names)) != len(source_names):
        raise ValueError('Duplicate source column names')
    if len(set(target_names)) != len(target_names):
        raise ValueError('Duplicate target column names')
    indexes = {name: i for i, name in enumerate(target_names)}
    missing = [name for name in source_names if name not in indexes]
    if missing:
        raise ValueError('Unsupported source column: ' + missing[0])
    return [indexes[name] for name in source_names]


def holdout_seeds(seeds, every=5):
    ordered = tuple(sorted(set(int(seed) for seed in seeds)))
    if every < 2:
        raise ValueError('Holdout stride must keep a fit set')
    return frozenset(seed for index, seed in enumerate(ordered) if index % every == 0)


def split_paired_roots(games, confirmation_seeds=None):
    """Fit vs original selection by paired root seed. Confirmation never selects."""
    paired = {}
    for game in games:
        if game.get('training') is False:
            raise ValueError('Frozen evaluation episode entered the fit set')
        paired.setdefault(int(game['seed']), set()).add(game['first'])
    selection = holdout_seeds(paired, every=5)
    fit = [game for game in games if int(game['seed']) not in selection]
    held = [game for game in games if int(game['seed']) in selection]
    if confirmation_seeds is None:
        confirmation_seeds = holdout_seeds((game['seed'] for game in fit), every=5)
    confirmation_seeds = frozenset(int(seed) for seed in confirmation_seeds)
    if confirmation_seeds & selection:
        raise ValueError('Confirmation seeds overlap the original selection holdout')
    train = [game for game in fit if int(game['seed']) not in confirmation_seeds]
    confirmation = [game for game in fit if int(game['seed']) in confirmation_seeds]
    if {int(game['seed']) for game in train} & selection:
        raise ValueError('A paired seed crossed the fit boundary')
    if {int(game['seed']) for game in train} & confirmation_seeds:
        raise ValueError('Already-viewed confirmation seeds must not enter the fit set')
    return dict(
        train=train, selection=held, confirmation=confirmation,
        selection_seeds=tuple(sorted(selection)),
        confirmation_seeds=tuple(sorted(confirmation_seeds)),
        paired_seeds=len(paired),
        both_seats=sum(len(sides) == 2 for sides in paired.values()),
    )


def validate_kl_grid(values=None):
    grid = tuple(KL_GRID if values is None else values)
    if not grid or 0.0 not in grid:
        raise ValueError('KL grid must include zero')
    if len(set(grid)) != len(grid):
        raise ValueError('KL grid must be unique')
    for value in grid:
        if type(value) is bool or not isinstance(value, (int, float)) or not 0 <= float(value) <= MAX_KL:
            raise ValueError('KL coefficient is outside the predeclared bound')
    return tuple(float(value) for value in grid)


def verify_old_teacher(directory, key):
    """Repository `fixed_ten_v1` identity. Never rewrites the historical rule hash."""
    if key in NO_OLD_TEACHER:
        raise ValueError(f'{key} has no old teacher')
    if key not in TEAMS:
        raise ValueError('Old teacher is only defined for starter, weave-rush and quick-rush')
    root = Path(directory)
    manifest_path = root / f'{key}.json'
    weights_path = root / f'{key}.npz'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    digest = sha256(weights_path.read_bytes()).hexdigest()
    current_fixed = old_all_features()
    current = current_features()
    features = list(manifest.get('features') or ())
    candidates = list(manifest.get('candidates') or ())
    added = sorted(set(current_fixed) - set(features))
    if (manifest.get('schema') != OLD_SCHEMA or manifest.get('deck') != key
            or manifest.get('sha256') != digest
            or manifest.get('build_sha256') != build_hash(manifest.get('build') or {})
            or candidates != list(old_candidate_names())
            or len(set(features)) != len(features) or not set(features) <= set(current)
            or not set(features) <= set(current_fixed)
            or len(added) != DAMAGE_IMMUNE_GAP
            or any(not name.startswith('fixed:damage_immune:') for name in added)):
        raise ValueError('Old numeric identity or named-column migration mismatch')
    hidden = manifest.get('hidden')
    if type(hidden) is not int or not 1 <= hidden <= 256:
        raise ValueError('Invalid old teacher width')
    expected = tensor_shapes(hidden)
    expected['state_net.0.weight'] = (hidden, len(features))
    expected['cand_net.0.weight'] = (hidden, OLD_CAND_DIM)
    import numpy as np
    with np.load(weights_path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected):
            raise ValueError('Old tensor set mismatch')
        weights = {name: saved[name].copy() for name in expected}
    for name, shape in expected.items():
        if weights[name].shape != shape or not np.isfinite(weights[name]).all():
            raise ValueError('Old tensor shape/nonfinite mismatch')
        weights[name].flags.writeable = False
    if manifest.get('automatic_serving_approval') is not False:
        raise ValueError('Old teacher cannot inherit serving approval')
    return dict(
        key=key, schema=OLD_SCHEMA, sha256=digest, file_sha256=digest,
        build_sha256=manifest['build_sha256'],
        source_rule_hash=manifest.get('rule_hash'),
        current_rule_hash=None,
        rule_hash_rewritten=False,
        features=features, candidates=candidates, hidden=hidden,
        weights=weights, build=manifest['build'],
        added_features=added,
        automatic_serving_approval=False,
        serving_export=False,
        observation_index=named_index_map(features, current),
        candidate_index=named_index_map(candidates, current_candidates()),
    )


def verify_repository_teachers(directory=None):
    root = Path(directory or REPO_TEACHERS)
    identities = {key: verify_old_teacher(root, key) for key in TEAMS}
    for key in NO_OLD_TEACHER:
        if (root / f'{key}.json').exists() or (root / f'{key}.npz').exists():
            raise ValueError(f'{key} must not be treated as an old teacher')
    return identities


def project_observation(x, teacher):
    import numpy as np
    vector = np.asarray(x, dtype=np.float32)
    if vector.ndim != 1 or len(vector) != len(current_features()):
        raise ValueError('Current observation width mismatch')
    projected = vector[teacher['observation_index']]
    if projected.shape != (len(teacher['features']),) or not np.isfinite(projected).all():
        raise ValueError('Invalid projected observation')
    return projected


def unsupported_action_mask(candidates, teacher):
    """True where a current legal action cannot be represented by the old teacher."""
    import numpy as np
    current = np.asarray(candidates, dtype=np.float32)
    if current.ndim != 2 or current.shape[1] != len(current_candidates()):
        raise ValueError('Current candidate width mismatch')
    extra = [i for i, name in enumerate(current_candidates()) if name not in set(teacher['candidates'])]
    actor = current[:, 1]
    card = current[:, 2]
    target = current[:, 4]
    unknown = ((actor >= len(OLD_SEATS))
               | ((card >= 0) & (card >= len(OLD_CARD_IDS)))
               | (target >= len(OLD_SEATS)))
    leftover = (
        np.any(np.abs(current[:, extra]) > 1e-8, axis=1)
        if extra else np.zeros(len(current), dtype=bool)
    )
    return np.asarray(unknown | leftover, dtype=bool)


def project_candidates(candidates, teacher):
    import numpy as np
    current = np.asarray(candidates, dtype=np.float32)
    rejected = unsupported_action_mask(current, teacher)
    projected = current[:, teacher['candidate_index']]
    if projected.shape[1] != len(teacher['candidates']) or not np.isfinite(projected).all():
        raise ValueError('Invalid projected candidates')
    return projected, rejected


def teacher_scores(teacher, x, candidates):
    import numpy as np
    weights = teacher['weights']
    hidden = teacher['hidden']

    def linear(value, name):
        return (
            np.einsum('...j,ij->...i', value, weights[name + '.weight'], optimize=False)
            + weights[name + '.bias']
        )

    state = np.maximum(linear(x, 'state_net.0'), 0)
    state = np.maximum(linear(state, 'state_net.2'), 0)
    cand = np.maximum(linear(candidates, 'cand_net.0'), 0)
    cand = np.maximum(linear(cand, 'cand_net.2'), 0)
    scores = np.sum(cand * state, -1) / np.sqrt(np.float32(hidden)) + linear(cand, 'score').reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError('Old scores nonfinite')
    return scores


def overlapping_root(row, teacher):
    """Map one current-rule root onto old teacher actions. Reject unsupported."""
    import numpy as np
    if row.get('key') in NO_OLD_TEACHER:
        raise ValueError(f"{row['key']} has no old teacher")
    if row.get('hidden_opponent_cards'):
        raise ValueError('Old teacher mapping cannot use hidden opponent cards')
    z = int(row['z'])
    if z not in (-1, 0, 1) or row.get('reward_to_go') is not None or row.get('setup_reward', 0):
        raise ValueError('Only actual terminal WDL labels are permitted')
    x = project_observation(row['x'], teacher)
    candidates, rejected = project_candidates(row['c'], teacher)
    keep = np.flatnonzero(~rejected)
    if len(keep) == 0:
        return dict(overlapping=False, rejected=int(rejected.sum()), keep=keep, z=z)
    scores = teacher_scores(teacher, x, candidates[keep])
    return dict(
        overlapping=True, rejected=int(rejected.sum()), keep=keep, z=z,
        x=x, c=candidates[keep], scores=scores,
        teacher_top=int(keep[int(np.argmax(scores))]),
    )


def teacher_distribution(scores):
    import numpy as np
    logits = np.asarray(scores, dtype=np.float64)
    shifted = logits - logits.max()
    prob = np.exp(shifted)
    prob = prob / prob.sum()
    if not np.isfinite(prob).all() or abs(prob.sum() - 1) > 1e-6:
        raise ValueError('Invalid teacher distribution')
    return prob


def teacher_kl_loss(student_logits, mask, teacher_scores_batch, coefficient):
    """KL(student || teacher) on overlapping legal actions. Zero is a true no-op."""
    import torch
    if coefficient == 0:
        return student_logits.new_zeros(()), dict(kl_coefficient=0.0, kl_samples=0)
    if not 0 < float(coefficient) <= MAX_KL:
        raise ValueError('Invalid KL coefficient')
    fill = torch.finfo(student_logits.dtype).min
    log_student = student_logits.masked_fill(~mask, fill).log_softmax(-1)
    teacher = teacher_scores_batch.masked_fill(~mask, fill).softmax(-1)
    student = log_student.exp()
    kl = (student * (log_student - teacher.clamp_min(1e-12).log())).sum(-1)
    loss = coefficient * kl.mean()
    if not torch.isfinite(loss):
        raise ValueError('Non-finite teacher KL')
    return loss, dict(
        kl_coefficient=float(coefficient),
        kl_samples=int(mask.any(-1).sum().item()),
        kl=float(kl.mean().detach()),
    )


def select_variant(measured, *, confirmation_used=False):
    if confirmation_used:
        raise ValueError('Already-viewed confirmation seeds must not select a variant')
    eligible = [row for row in measured if row.get('kl_coefficient') != 0 and row.get('passed')]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row.get('exact_action', 0), -row.get('cross_entropy', 0)))


def _kind(row):
    import numpy as np
    return np.asarray(row['c'], dtype=np.int64)[:, 0]


def _card_name(row, index):
    import numpy as np
    raw = int(np.asarray(row['c'])[index, 2])
    if raw < 0 or raw >= len(CARD_IDS):
        return ''
    return CARD_IDS[raw]


def summarize_agreement(records):
    n = len(records)
    empty = dict(
        n=0, overlapping=0, rejected=0, legal_play=0, exact_action=0, card_agreement=0,
        legal_play_top1=None, teacher_play=0, student_play=0, category_agreement=0,
        attack_called_play=0, end_called_play=0, cross_entropy=None,
        collapse=None, attack_n=0, end_n=0, never_play=None, always_play=None,
    )
    if not n:
        return empty
    legal = [row for row in records if row['legal_play']]
    attack = [row for row in records if row['teacher_kind'] == 'attack']
    end = [row for row in records if row['teacher_kind'] == 'end_turn']
    play_top = sum(row['student_play'] for row in records)
    return dict(
        n=n,
        overlapping=sum(row['overlapping'] for row in records),
        rejected=sum(row['rejected'] for row in records),
        legal_play=len(legal),
        exact_action=sum(row['exact_action'] for row in records),
        card_agreement=sum(row['card_agreement'] for row in records),
        teacher_play=sum(row['teacher_kind'] == 'play_card' for row in records),
        student_play=play_top,
        category_agreement=sum((row['teacher_kind'] == 'play_card') == row['student_play']
                               for row in records),
        legal_play_top1=(sum(row['student_play'] for row in legal) / len(legal)) if legal else None,
        attack_n=len(attack),
        end_n=len(end),
        attack_called_play=sum(row['student_play'] for row in attack),
        end_called_play=sum(row['student_play'] for row in end),
        cross_entropy=float(sum(row['cross_entropy'] for row in records) / n),
        never_play=play_top == 0,
        always_play=play_top >= 0.95 * n,
        collapse=play_top == 0 or play_top >= 0.95 * n,
    )


def agreement_record(row, mapped, student_logits, teacher_prob):
    import numpy as np
    kind = _kind(row)
    student_top = int(np.argmax(student_logits))
    keep = mapped['keep']
    student_keep = np.asarray(student_logits, dtype=np.float64)[keep]
    student_keep = student_keep - student_keep.max()
    student_keep = np.exp(student_keep)
    student_keep = student_keep / student_keep.sum()
    ce = float(-(teacher_prob * np.log(np.clip(student_keep, 1e-12, 1))).sum())
    teacher_top = mapped['teacher_top']
    teacher_kind = {ACT_END: 'end_turn', ACT_ATTACK: 'attack', ACT_PLAY: 'play_card'}.get(
        int(kind[teacher_top]), str(int(kind[teacher_top])))
    return dict(
        overlapping=True,
        rejected=mapped['rejected'],
        legal_play=bool(np.any(kind == ACT_PLAY)),
        exact_action=bool(student_top == teacher_top),
        card_agreement=bool(
            kind[student_top] == ACT_PLAY and kind[teacher_top] == ACT_PLAY
            and _card_name(row, student_top) == _card_name(row, teacher_top)
            and _card_name(row, teacher_top) != ''
        ),
        student_play=bool(kind[student_top] == ACT_PLAY),
        teacher_kind=teacher_kind,
        cross_entropy=ce,
        z=mapped['z'],
    )


def report_without_old_teacher(key):
    if key not in NO_OLD_TEACHER:
        raise ValueError('Old teacher exists for ' + key)
    return dict(
        key=key, no_old_teacher=True, claimed_improved=False,
        reason='Zhenhong and murk have no repository fixed_ten_v1 teacher',
    )


def update_student(net, opt, rows, teachers, *, kl_coefficient=0.0,
                   teacher_top_weight=0.0, value_only=False):
    """Fit Gumbel policy and terminal WDL, with optional old-teacher anchoring."""
    import numpy as np
    import torch
    from .league_learning import tensors
    if any(row['z'] not in (-1, 0, 1) or row.get('reward_to_go') is not None for row in rows):
        raise ValueError('Only actual terminal WDL labels are permitted')
    coefficient = float(kl_coefficient)
    if not np.isfinite(coefficient) or not 0 <= coefficient <= MAX_KL:
        raise ValueError('KL coefficient is outside the predeclared bound')
    top_weight = float(teacher_top_weight)
    if not np.isfinite(top_weight) or not 0 <= top_weight <= MAX_TEACHER_TOP_WEIGHT:
        raise ValueError('Teacher top-action weight is outside the research bound')
    device = next(net.parameters()).device
    x = torch.as_tensor(np.stack([row['x'] for row in rows]), device=device)
    hidden = net.state_net(x)
    context = torch.tensor([[row.get('value_actor', 1)] for row in rows], dtype=hidden.dtype, device=device)
    logits = net.wdl(torch.cat((hidden, context), -1))
    z = torch.tensor([int(row['z']) + 1 for row in rows], device=device)
    weights = torch.tensor([row.get('sample_weight', 1.) for row in rows], dtype=hidden.dtype, device=device)
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid sampling weights')
    value_loss = (weights * torch.nn.functional.cross_entropy(logits, z, reduction='none')).mean()
    policy_loss = logits.sum() * 0
    kl_loss = logits.new_zeros(())
    teacher_top_loss = logits.new_zeros(())
    stats = dict(kl_coefficient=coefficient, kl_samples=0)
    policy_indices = [index for index, row in enumerate(rows) if 'pi' in row] if not value_only else []
    if policy_indices:
        xx, cand, mask = tensors([(rows[index]['x'], rows[index]['c']) for index in policy_indices], device)
        policy, _ = net(xx, cand, mask)
        target = torch.zeros_like(policy)
        for local, index in enumerate(policy_indices):
            pi = np.asarray(rows[index]['pi'], dtype=np.float64)
            if (len(pi) != len(rows[index]['c']) or not np.isfinite(pi).all()
                    or np.any(pi < 0) or not np.isclose(pi.sum(), 1)):
                raise ValueError('Invalid Gumbel policy label')
            target[local, :len(pi)] = torch.as_tensor(pi, device=device)
        ce = -(target * policy.log_softmax(-1).masked_fill(~mask, 0)).sum(-1)
        policy_loss = (weights[policy_indices] * ce).mean()
        if coefficient or top_weight:
            teacher_locals = []
            teacher_score_rows = []
            teacher_top_indexes = []
            for local, index in enumerate(policy_indices):
                row = rows[index]
                if row.get('key') in NO_OLD_TEACHER or row.get('key') not in teachers:
                    continue
                cached = row.get('_teacher_mapped')
                if cached is not None and row.get('_teacher_sha256') == teachers[row['key']]['sha256']:
                    mapped = cached
                else:
                    mapped = overlapping_root(row, teachers[row['key']])
                if not mapped['overlapping']:
                    continue
                teacher_locals.append(local)
                teacher_top_indexes.append(mapped['teacher_top'])
                scores = np.full(len(row['c']), np.finfo(np.float32).min, dtype=np.float32)
                scores[mapped['keep']] = mapped['scores'].astype(np.float32)
                teacher_score_rows.append(scores)
            if teacher_locals and coefficient:
                width = policy.shape[1]
                fill = torch.finfo(policy.dtype).min
                teacher_batch = torch.full((len(teacher_locals), width), fill, device=device)
                overlap = torch.zeros((len(teacher_locals), width), dtype=torch.bool, device=device)
                threshold = np.finfo(np.float32).min / 2
                for index, scores in enumerate(teacher_score_rows):
                    teacher_batch[index, :len(scores)] = torch.as_tensor(scores, device=device)
                    overlap[index, :len(scores)] = torch.as_tensor(
                        np.isfinite(scores) & (scores > threshold), device=device)
                overlap &= mask[teacher_locals]
                kl_loss, stats = teacher_kl_loss(policy[teacher_locals], overlap, teacher_batch, coefficient)
            if teacher_locals and top_weight:
                selected_logits = policy[teacher_locals].masked_fill(
                    ~mask[teacher_locals], torch.finfo(policy.dtype).min)
                targets = torch.tensor(teacher_top_indexes, dtype=torch.long, device=device)
                teacher_top_loss = top_weight * torch.nn.functional.cross_entropy(selected_logits, targets)
    loss = value_loss + policy_loss + kl_loss + teacher_top_loss
    if not torch.isfinite(loss):
        raise ValueError('Non-finite teacher distillation loss')
    prefixes = ('cand_net', 'score', 'state_net')
    owned = [(name, parameter) for name, parameter in net.named_parameters()
             if parameter.requires_grad and name.startswith(prefixes)]
    norms = {}
    if policy_indices and owned:
        grads = torch.autograd.grad(
            policy_loss + kl_loss + teacher_top_loss, [parameter for _, parameter in owned],
            retain_graph=True, allow_unused=True)
        for (name, _), gradient in zip(owned, grads):
            norms[name] = 0.0 if gradient is None else float(gradient.detach().norm())
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    return dict(
        value_loss=float(value_loss.detach()), policy_loss=float(policy_loss.detach()),
        kl_loss=float(kl_loss.detach()), teacher_top_loss=float(teacher_top_loss.detach()),
        teacher_top_weight=top_weight,
        samples=len(rows), policy_samples=len(policy_indices),
        kl_coefficient=stats['kl_coefficient'], kl_samples=stats.get('kl_samples', 0),
        policy_gradient_norms=norms, serving_export=False,
    )


def _episode_jobs(directory):
    import gzip
    jobs = []
    for path in sorted(Path(directory).glob('*.json.gz')):
        payload = json.loads(gzip.open(path, 'rt', encoding='utf-8').read())
        job = payload['job']
        if job.get('training') is False:
            raise ValueError('Frozen evaluation episode entered the fit set')
        if payload.get('winner') not in ('a', 'b', 'draw'):
            raise ValueError('Only completed episodes may be used')
        jobs.append(dict(
            path=path.name, seed=int(job['seed']), first=job['first'],
            left=job['left'], right=job['right'], training=bool(job.get('training', True)),
            winner=payload['winner'],
            keys=sorted({row.get('key') for row in payload.get('rows') or []}),
        ))
    if not jobs:
        raise ValueError('No archived training episodes')
    return jobs


def validate_output_path(path):
    output = Path(path)
    if output.exists():
        raise ValueError('Output must be new; dry-run does not create it')
    return output


def dry_run(*, teachers=None, episodes=None, output, kl_grid=None, confirmation_seeds=None):
    """Validate identities, mapping, seeds and the KL grid. Never trains or writes."""
    teacher_dir = Path(teachers or REPO_TEACHERS)
    episode_dir = Path(episodes or ARCHIVED_EPISODES)
    output_path = validate_output_path(output)
    identities = verify_repository_teachers(teacher_dir)
    before = {key: identities[key]['sha256'] for key in TEAMS}
    after = {key: sha256((teacher_dir / f'{key}.npz').read_bytes()).hexdigest() for key in TEAMS}
    if before != after:
        raise ValueError('Old teacher files were mutated')
    jobs = _episode_jobs(episode_dir)
    split = split_paired_roots(jobs, confirmation_seeds)
    grid = validate_kl_grid(kl_grid)
    if output_path.exists():
        raise ValueError('Dry-run created output')
    for teacher in identities.values():
        if teacher['rule_hash_rewritten'] or teacher['source_rule_hash'] is None:
            raise ValueError('Old rule hashes must be preserved, not rewritten')
    return dict(
        dry_run=True,
        schema=CURRENT_SCHEMA,
        old_schema=OLD_SCHEMA,
        teams=list(TEAMS),
        no_old_teacher=list(NO_OLD_TEACHER),
        kl_grid=list(grid),
        identities={key: {field: identities[key][field] for field in (
            'schema', 'sha256', 'build_sha256', 'source_rule_hash', 'rule_hash_rewritten',
            'hidden', 'automatic_serving_approval')} for key in TEAMS},
        added_features={key: identities[key]['added_features'] for key in TEAMS},
        episodes=len(jobs),
        paired_seeds=split['paired_seeds'],
        both_seats=split['both_seats'],
        selection_seeds=list(split['selection_seeds']),
        confirmation_seeds=list(split['confirmation_seeds']),
        confirmation_selects_variant=False,
        output=str(output_path),
        output_created=False,
        training=False,
        optimizer=False,
        serving_export=False,
        serving_files_rewritten=False,
        hidden_opponent_information=False,
        acceptance_boundary=ACCEPTANCE_BOUNDARY,
        acceptance_note=ACCEPTANCE_NOTE,
        claimed_improved={key: False for key in NO_OLD_TEACHER},
    )
