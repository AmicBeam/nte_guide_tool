"""User-authorized serving of checksum-pinned recovery policies; NumPy only.

Training manifests stay byte-identical. This independent adapter validates the
old numeric contract against an explicit current-rule serving binding, without
relaxing the research loaders or upgrading their quality approval.
"""
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from app.modules.card_game.rl import cross_lineup, cross_grounded, residual_policy, recovery_policy
from app.modules.card_game.rl.residual_runtime import ResidualModel
from app.modules.card_game.rl import preserved_policy
from app.modules.card_game.rl.cross_teacher_distillation import (
    REPO_TEACHERS, verify_old_teacher, named_index_map, project_candidates,
    project_observation, teacher_scores,
)
from app.modules.card_game.rl.league_schema import build_hash

RELEASE = 'recovery-20261007'
SCHEMAS = (recovery_policy.PRESERVED_SCHEMA, recovery_policy.RESIDUAL_SCHEMA)
SOURCE_RULE_HASH = '7c928f692ad643ba32bb6d2e70808a137744fde82e58e5ee451e4744d3d5de9e'


class RecoveryServingModel(ResidualModel):
    experimental = True

    def __init__(self, directory, key):
        root = Path(directory)
        binding = json.loads((root / 'serving.json').read_text(encoding='utf-8'))
        pin = binding.get('models', {}).get(key) or {}
        manifest_path = root / f'{key}.json'
        path = root / f'{key}.npz'
        self.manifest = m = json.loads(manifest_path.read_text(encoding='utf-8'))
        self.version = sha256(path.read_bytes()).hexdigest()
        source_fields = ('source_rule_hash', 'source_request', 'source_report')
        source_rule_hash = SOURCE_RULE_HASH
        if any(field in pin for field in source_fields):
            source_rule_hash = pin.get('source_rule_hash')
            if (not isinstance(source_rule_hash, str) or len(source_rule_hash) != 64
                    or any(c not in '0123456789abcdef' for c in source_rule_hash)
                    or any(not isinstance(pin.get(field), str) or not pin[field].strip()
                           for field in ('source_request', 'source_report'))):
                raise ValueError('Recovery per-model source authorization missing')
        if (binding.get('kind') != 'explicit_user_candidate' or not binding.get('request')
                or not binding.get('report') or binding.get('quality_approved') is not False
                or binding.get('runtime_rule_hash') != cross_lineup.identity()
                or binding.get('source_rule_hash') != SOURCE_RULE_HASH
                or m.get('rule_hash') != source_rule_hash
                or pin.get('manifest_sha256') != sha256(manifest_path.read_bytes()).hexdigest()
                or pin.get('model_sha256') != self.version
                or pin.get('build_sha256') != m.get('build_sha256')):
            raise ValueError('Recovery serving authorization mismatch')
        self.schema = m.get('schema')
        if (self.schema not in SCHEMAS or m.get('deck') != key or m.get('sha256') != self.version
                or m.get('architecture_sha256') != residual_policy.fingerprint()
                or m.get('value_architecture_sha256') != recovery_policy.fingerprint()
                or m.get('candidate_transform_sha256') != cross_grounded.fingerprint()
                or m.get('features') != cross_lineup.all_features()
                or m.get('candidates') != cross_lineup.candidate_names()
                or m.get('wdl_order') != ['loss', 'draw', 'win']
                or m.get('value_context') != ['viewer_is_actor']
                or m.get('automatic_serving_approval') is not False
                or (m.get('validation') or {}).get('approved') is not False
                or m.get('reward_mode') != 'terminal_wdl_only'):
            raise ValueError('Recovery serving numeric contract mismatch')
        self.hidden = m.get('hidden')
        if type(self.hidden) is not int or not 1 <= self.hidden <= 256:
            raise ValueError('Invalid recovery width')
        self.separated = True
        expected = residual_policy.tensor_shapes(self.hidden)
        expected.update(recovery_policy.extra_shapes(self.hidden))
        self.teacher = None
        if self.schema == recovery_policy.PRESERVED_SCHEMA:
            from app.modules.card_game.rl.fixed_lineup import tensor_shapes, CAND_DIM
            if (m.get('unsupported_root') != 'residual_only'
                    or m.get('preservation_sha256') != sha256(Path(preserved_policy.__file__).read_bytes()).hexdigest()):
                raise ValueError('Recovery preservation contract mismatch')
            teacher = verify_old_teacher(REPO_TEACHERS, key)
            source = m.get('source_identity') or {}
            required = ('key', 'schema', 'sha256', 'source_rule_hash', 'features', 'candidates',
                        'hidden', 'observation_index', 'candidate_index', 'added_features')
            if set(source) != set(required) or any(source[name] != teacher[name] for name in required):
                raise ValueError('Recovery frozen teacher identity mismatch')
            old = tensor_shapes(source['hidden'])
            old['state_net.0.weight'] = (source['hidden'], len(source['features']))
            old['cand_net.0.weight'] = (source['hidden'], CAND_DIM)
            expected.update({'legacy.' + preserved_policy.encoded_name(name): shape for name, shape in old.items()})
            self.teacher = dict(source, weights=teacher['weights'],
                observation_index=named_index_map(source['features'], cross_lineup.all_features()),
                candidate_index=named_index_map(source['candidates'], cross_lineup.candidate_names()))
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(expected):
                raise ValueError('Recovery tensor set mismatch')
            self.weights = {name: saved[name].copy() for name in expected}
        for name, shape in expected.items():
            value = self.weights[name]
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('Invalid recovery tensor')
            value.flags.writeable = False
        if self.teacher:
            for name, value in self.teacher['weights'].items():
                if not np.array_equal(self.weights['legacy.' + preserved_policy.encoded_name(name)], value):
                    raise ValueError('Recovery embedded teacher changed')
        from app.modules.card_game.content.duel_v2 import validate_deck
        has_override = any(field in pin for field in ('serving_build', 'serving_build_sha256', 'serving_build_request'))
        if has_override:
            request = pin.get('serving_build_request')
            if not isinstance(request, str) or not request.strip():
                raise ValueError('Recovery serving build override authorization missing')
            self.serving_deck = validate_deck(pin.get('serving_build'))
            expected_build_hash = pin.get('serving_build_sha256')
            if self.serving_deck['character_ids'] != m['build']['character_ids']:
                raise ValueError('Recovery serving build override roster mismatch')
        else:
            self.serving_deck = validate_deck(m['build'])
            expected_build_hash = m.get('build_sha256')
        self.serving_build_sha256 = build_hash(self.serving_deck)
        if self.serving_build_sha256 != expected_build_hash:
            raise ValueError('Recovery build mismatch')
        self.serving_authorized = True
        self.deck_id = key
        self.capability = dict(kind=self.schema, experimental=True, human_public_custom=True, bot_preset=key)

    def scores_only(self, x, c):
        residual = residual_policy.scores_numpy(x, c, self.weights)
        if self.teacher and len(c):
            mapped, unknown = project_candidates(c, self.teacher)
            if not unknown.any():
                return residual + teacher_scores(self.teacher, project_observation(x, self.teacher), mapped)
        return residual

    scores = scores_only

    def select_public_action(self, view, actions):
        if not actions:
            raise ValueError('No legal actions')
        x, c = cross_lineup.encode(view, actions)
        return int(self.scores_only(x, c).argmax())

    def validate_human(self, deck):
        from app.modules.card_game.content.duel_v2 import validate_deck, deck_uses_test_characters
        build = validate_deck(deck)
        if deck_uses_test_characters(build):
            raise ValueError('高级人机仅支持公开角色构筑。')
        return self.capability
