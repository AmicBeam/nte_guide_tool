"""NumPy serving runtime for phase-isolated seven-team policies.

Old seven loaders stay untouched. This export is opt-in, CPU-serving, and
keeps the current seven observation/action identity.
"""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from . import phase_policy as architecture, seven_grounded as grounded, seven_runtime
from .seven_lineup import all_features, candidate_names, identity, KEYS
from .league_schema import build_hash
from .seven_runtime import decision, determinize, value_for


def export(net, directory, key, build, origin):
    from .recovery_round import write
    from .build_acceptance import canonical_build
    if key not in KEYS or getattr(net, 'policy_schema', None) != architecture.SCHEMA:
        raise ValueError('Phase-isolated seven-team policy required')
    build = canonical_build(build)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / (key + '.npz')
    tmp = root / (key + '.tmp.npz')
    np.savez_compressed(tmp, **{name: tensor.detach().cpu().numpy() for name, tensor in net.state_dict().items()})
    tmp.replace(path)
    write(root / (key + '.json'), dict(
        schema=architecture.SCHEMA, deck=key, hidden=net.hidden, rule_hash=identity(),
        architecture_sha256=architecture.fingerprint(), candidate_transform_sha256=grounded.fingerprint(),
        features=all_features(), candidates=candidate_names(), grounded_features=grounded.transformed_names(),
        source_identity=getattr(net, 'source_identity', None), equal_input_scores='canonical_first_v1',
        sha256=sha256(path.read_bytes()).hexdigest(), build=build, build_sha256=build_hash(build),
        origin=origin, wdl_order=['loss', 'draw', 'win'], value_context=['viewer_is_actor'],
        reward_mode='terminal_wdl_only', automatic_serving_approval=False,
        validation=dict(approved=False)))


class PhaseModel:
    def __init__(self, directory, key):
        self.directory = str(directory)
        root = Path(directory)
        manifest = json.loads((root / (key + '.json')).read_text(encoding='utf8'))
        path = root / (key + '.npz')
        self.version = sha256(path.read_bytes()).hexdigest()
        if (manifest.get('schema') != architecture.SCHEMA or manifest.get('deck') != key
                or manifest.get('rule_hash') != identity()
                or manifest.get('architecture_sha256') != architecture.fingerprint()
                or manifest.get('candidate_transform_sha256') != grounded.fingerprint()
                or manifest.get('features') != all_features()
                or manifest.get('candidates') != candidate_names()
                or manifest.get('grounded_features') != grounded.transformed_names()
                or manifest.get('sha256') != self.version
                or manifest.get('reward_mode') != 'terminal_wdl_only'
                or manifest.get('automatic_serving_approval') is not False
                or manifest.get('equal_input_scores') != 'canonical_first_v1'
                or manifest.get('wdl_order') != ['loss', 'draw', 'win']
                or manifest.get('value_context') != ['viewer_is_actor']
                or (manifest.get('validation') or {}).get('approved') is not False):
            raise ValueError('Phase-isolated numeric identity mismatch')
        self.hidden = manifest['hidden']
        expected = architecture.tensor_shapes(self.hidden)
        self.teacher = None
        self.legacy_extra_indices = None
        if manifest.get('source_identity'):
            from .cross_teacher_distillation import verify_old_teacher, REPO_TEACHERS, named_index_map
            from .preserved_policy import encoded_name
            teacher = verify_old_teacher(REPO_TEACHERS, key)
            source = {name: teacher[name] for name in manifest['source_identity']
                      if name not in ('observation_index', 'candidate_index')}
            source.update(observation_index=named_index_map(teacher['features'], all_features()),
                          candidate_index=named_index_map(teacher['candidates'], candidate_names()))
            if source != manifest['source_identity']:
                raise ValueError('Phase-isolated frozen teacher changed')
            expected.update({'legacy.' + encoded_name(name): value.shape for name, value in teacher['weights'].items()})
            self.teacher = dict(teacher, **source)
            supported = set(source['candidates'])
            self.legacy_extra_indices = np.asarray(
                [i for i, name in enumerate(candidate_names()) if name not in supported], dtype=np.int64)
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(expected):
                raise ValueError('Phase-isolated tensor set mismatch')
            self.weights = {name: arrays[name].copy() for name in arrays.files}
        for name, shape in expected.items():
            if (self.weights[name].shape != shape or self.weights[name].dtype != np.float32
                    or not np.isfinite(self.weights[name]).all()):
                raise ValueError('Phase-isolated tensor shape/nonfinite')
            self.weights[name].flags.writeable = False
        for prefix in ('state_net', 'value_net', 'cand_net', 'opening.state_net', 'opening.cand_net'):
            active = self.weights[prefix + '.0.active']
            if not np.isin(active, [0, 1]).all() or not active.any():
                raise ValueError('Invalid named normalization mask')
        if self.teacher:
            from .preserved_policy import encoded_name
            for name, value in self.teacher['weights'].items():
                if not np.array_equal(value, self.weights['legacy.' + encoded_name(name)]):
                    raise ValueError('Frozen teacher arrays changed')
        self.manifest = manifest
        self.schema = architecture.SCHEMA
        self.serving_deck = manifest['build']
        if build_hash(self.serving_deck) != manifest['build_sha256']:
            raise ValueError('Phase-isolated build mismatch')

    def scores(self, x, c):
        phase = architecture.phase_of_observation(x)
        if not isinstance(phase, str):
            raise ValueError('Serving scores require a single observation')
        scores = architecture.scores_numpy(x, c, self.weights, phase)
        if self.teacher is not None and len(c):
            from .fixed_lineup import SEATS, CARD_IDS
            from .cross_teacher_distillation import teacher_scores
            extra = self.legacy_extra_indices
            unknown = ((c[:, 0] >= 6) | (c[:, 1] >= len(SEATS)) | (c[:, 2] >= len(CARD_IDS))
                       | (c[:, 4] >= len(SEATS)) | (np.abs(c[:, extra]) > 1e-8).any(-1))
            if not unknown.any():
                scores = scores + teacher_scores(
                    self.teacher, x[self.teacher['observation_index']], c[:, self.teacher['candidate_index']])
        return seven_runtime.architecture.canonical_scores_numpy(scores, c)

    scores_only = scores

    def wdl(self, x, is_actor):
        return architecture.wdl_numpy(x, is_actor, self.weights)

    def scores_value(self, x, c):
        probs = self.wdl(x, True)
        return self.scores(x, c), float(probs[2] - probs[0])

    def validate_human(self, build):
        raise ValueError('Phase-isolated research model has no website authorization')


@lru_cache(maxsize=32)
def _load(directory, key, manifest_ns, weights_ns, rule_hash, architecture_hash):
    return PhaseModel(directory, key)


def model(directory, key):
    root = Path(directory)
    return _load(str(root), key, (root / (key + '.json')).stat().st_mtime_ns,
                 (root / (key + '.npz')).stat().st_mtime_ns, identity(), architecture.fingerprint())


def restore(directory, key, device='cpu'):
    import torch
    policy = model(directory, key)
    net = architecture.create_network(policy.hidden, device, key=key, legacy=policy.teacher is not None)
    net.load_state_dict({name: torch.as_tensor(value.copy(), device=device) for name, value in policy.weights.items()})
    return net, policy.serving_deck, policy.manifest['origin']


def migrate(source_directory, target_directory, key):
    source = Path(source_directory).resolve()
    target = Path(target_directory).resolve()
    if target == source:
        raise ValueError('Refusing to overwrite source')
    for suffix in ('.json', '.npz'):
        if (target / (key + suffix)).resolve() == (source / (key + suffix)).resolve():
            raise ValueError('Refusing to overwrite source')
    if any((target / (key + suffix)).exists() for suffix in ('.json', '.npz')):
        raise ValueError('Migration requires new output files')
    old = seven_runtime.model(source, key)
    import torch
    net = architecture.create_network(old.hidden, 'cpu', key=key, legacy=old.teacher is not None)
    payload = net.state_dict()
    playing = {name: tensor for name, tensor in payload.items() if not name.startswith('opening.')}
    if set(old.weights) != set(playing):
        raise ValueError('Source tensor set is not a seven-team policy')
    with torch.no_grad():
        for name, array in old.weights.items():
            tensor = torch.as_tensor(array.copy())
            if tuple(payload[name].shape) != tuple(tensor.shape) or not np.isfinite(array).all():
                raise ValueError('Source tensor shape/nonfinite mismatch')
            payload[name].copy_(tensor)
            if name.startswith(('state_net.', 'cand_net.', 'score.')):
                payload['opening.' + name].copy_(tensor)
    net.load_state_dict(payload)
    origin = dict(old.manifest.get('origin') or {}, phase_isolation=dict(
        source_schema=old.manifest['schema'], source_sha256=old.version,
        opening_copied_from='playing', value_shared=True, optimizer_reset=True,
        prior_ready_adopted=False, prior_quality_adopted=False))
    export(net, target, key, old.serving_deck, origin)
    return net, origin


def predict_encoded(policy, x, c):
    return policy.scores_value(x, c)


def action_scores(policy, x, c):
    return policy.scores(x, c)
