"""CPU-only frozen policy selection. Rules still run in engine.duel_v2.

The serving files contain only numeric NumPy arrays and JSON metadata. Neither
PyTorch, CUDA, a compiler, nor a training loop is imported by this module.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from app import config
from app.errors import RuleValidationError
from app.modules.card_game.content.duel_v2 import CARDS, STARTER_DECKS
from app.modules.card_game.rl.gpu_duel.catalog import (
    CARD_IDS, CARD_INDEX, GPU_LOCK, N_SEATS, SEATS, SEAT_INDEX, HAND_SLOTS,
    DISCARD_SLOTS, ACT_ATTACK, ACT_CHOOSE, ACT_END, ACT_PLAY, ACT_ULTIMATE,
)
from app.modules.card_game.rl.gpu_duel.resident_schema import SCHEMA, SIDE_FIELDS, CHAR_FIELDS
from app.modules.card_game.rl.capability import (
    assert_public_encoder_deck, is_exact_preset,
    resolve_model_capability, fixed_team_serving_deck, serving_deck_hash,
)
from app.modules.card_game.rl.rule_ir.layout import OFFSETS
from app.modules.card_game.rl.rule_ir.pack_row import pack_python_row

_CACHE = {}
_LOCK = threading.RLock()
PRESETS = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')
RANDOM_PRESETS = PRESETS
UNTRAINED_STRATEGIES = {}
PENDING_PRESETS = frozenset()


def preset(deck_id):
    if deck_id not in PRESETS:
        raise RuleValidationError('高级人机策略不存在。')
    return next(d for d in STARTER_DECKS if d['id'] == deck_id)


def serving_deck(model, deck_id):
    """Return a copy of the validated model/build pair, with legacy preset fallback."""
    from copy import deepcopy
    return deepcopy(model.serving_deck if hasattr(model, 'serving_deck') else preset(deck_id))


def require_human_deck(model, deck, deck_id):
    """Validate serving scope before creating any room or replay."""
    try:
        from app.modules.card_game.rl.league_schema import SCHEMA as LEAGUE_SCHEMA
        if getattr(model, 'experimental', False):
            return model.validate_human(deck)
        if getattr(model, 'schema', None) in (LEAGUE_SCHEMA, 'fixed_ten_v1'):
            return model.validate_human(deck)
        deck = assert_public_encoder_deck(deck)
        capability = resolve_model_capability(
            {'capability': getattr(model, 'capability', None)}, deck_id)
    except ValueError as exc:
        raise RuleValidationError(str(exc)) from exc
    if capability['kind'] == 'public_candidate_v1':
        raise RuleValidationError('该高级人机尚未通过独立评估，暂不能用于网页对局。')
    if not capability['human_public_custom'] and not is_exact_preset(deck, deck_id):
        raise RuleValidationError('该高级人机尚未开放自由构筑，请选择与对手相同的预组。')
    return capability


def encode(state, side, actions):
    """Encode exactly the training schema, restricted to public information."""
    import numpy as np
    from app.modules.card_game.rl.gpu_duel.catalog import CARD_INDEX as LIVE_CARD_INDEX
    unknown = []
    for team in (state.get('sides') or {}).values():
        for zone in ('hand', 'deck', 'discard'):
            for card in team.get(zone) or []:
                card_id = card.get('card_id')
                if card_id not in LIVE_CARD_INDEX:
                    unknown.append(card_id)
        for hero in (team.get('characters') or {}).values():
            shape = hero.get('shape')
            if shape and shape not in LIVE_CARD_INDEX:
                unknown.append(shape)
    if unknown:
        raise ValueError('Encoder cannot represent ' + ','.join(sorted(set(str(item) for item in unknown if item))))
    row = pack_python_row(state)
    viewer = 'ab'.index(side)
    order = (viewer, 1 - viewer)
    values = [row[OFFSETS['turn']] / 30, float(viewer == row[OFFSETS['first']])]
    for name in SIDE_FIELDS:
        values.extend(row[OFFSETS[name] + s] / 10 for s in order)
    for name in CHAR_FIELDS:
        values.extend(row[OFFSETS[name] + s*N_SEATS + k] / 10 for s in order for k in range(N_SEATS))
    own = row[OFFSETS['hand_kind'] + viewer*HAND_SLOTS:OFFSETS['hand_kind'] + (viewer+1)*HAND_SLOTS]
    zones = [own]
    for s in order:
        n = row[OFFSETS['discard_n'] + s]
        zones.append(row[OFFSETS['discard_kind'] + s*DISCARD_SLOTS:OFFSETS['discard_kind'] + s*DISCARD_SLOTS + n])
    for zone in zones:
        counts = [0.0]*len(CARD_IDS)
        for kind in zone:
            if kind >= 0:
                counts[kind] += .5
        values.extend(counts)
    hand = {c['instance_id']: c for c in state['sides'][side]['hand']}
    choices = {c['instance_id']: c for c in (state.get('pending_choice') or {}).get('cards', [])}
    candidates = []
    for action in actions:
        kind, actor, card_kind, target_side, target_seat = -1, -1, -1, -1, -1
        card = None
        if action['type'] == 'end_turn':
            kind = ACT_END
        elif action['type'] in ('attack', 'ultimate'):
            kind = ACT_ATTACK if action['type'] == 'attack' else ACT_ULTIMATE
            actor = SEAT_INDEX[action['character_id']]
        elif action['type'] == 'play_card':
            kind = ACT_PLAY
            item = hand[action['card_id']]
            card = CARDS[item['card_id']]
            card_kind = CARD_INDEX[item['card_id']]
            actor = SEAT_INDEX[item['character_id']]
            if action.get('target_id'):
                ts, tk = action['target_id'].split(':', 1)
                target_side = 0 if ts == side else 1
                target_seat = -1 if tk == 'player' else SEAT_INDEX[tk]
        elif action['type'] == 'choose':
            kind = ACT_CHOOSE
            item = choices[action['choice_id']]
            card = CARDS[item['card_id']]
            card_kind = CARD_INDEX[item['card_id']]
        else:
            raise ValueError(f'Unsupported model action: {action["type"]}')
        card = card or {}
        candidates.append([kind, actor, card_kind, target_side, target_seat,
                           int(card.get('cost') or 0), int(card.get('attack') or 0),
                           int(card.get('shield') or 0), int(card.get('hp') or 0),
                           int(bool(card.get('instant')))])
    return np.asarray(values, dtype=np.float32), np.asarray(candidates, dtype=np.float32)


class FrozenModel:
    def __init__(self, directory, deck_id):
        import numpy as np
        directory = Path(directory)
        manifest = json.loads((directory / f'{deck_id}.json').read_text(encoding='utf-8'))
        from app.modules.card_game.rl.public_schema import SCHEMA as PUBLIC_SCHEMA, state_dim, rule_identity, CAND_DIM
        from app.modules.card_game.rl.opening_observation import OPENING_SCHEMA,OPENING_CAND_DIM
        self.schema = manifest.get('schema')
        expanded=self.schema in (PUBLIC_SCHEMA,OPENING_SCHEMA)
        if (self.schema not in (SCHEMA, PUBLIC_SCHEMA,OPENING_SCHEMA) or manifest.get('gpu_lock') != GPU_LOCK
                or manifest.get('deck') != deck_id or manifest.get('card_ids') != list(CARD_IDS)
                or manifest.get('seat_ids') != list(SEATS)):
            raise ValueError('Model version does not match current content')
        from app.modules.card_game.rl.rule_ir.completeness import compiled_rule_hash
        runtime_rule_hash = manifest.get('runtime_rule_hash') or (manifest.get('rule_identity') or {}).get('rule_hash')
        expected_rule_hash = (rule_identity(OPENING_SCHEMA) if self.schema==OPENING_SCHEMA else rule_identity()) if expanded else compiled_rule_hash(deck_id)
        if runtime_rule_hash != expected_rule_hash:
            raise ValueError('Model rule version does not match current rules')
        path = directory / f'{deck_id}.npz'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest.get('sha256'):
            raise ValueError('Model checksum mismatch')
        self.capability = resolve_model_capability(manifest, deck_id)
        if self.capability['human_public_custom'] and not expanded:
            raise ValueError('Public custom serving requires an expanded observation/export contract; resident_public_v1 is mirror-only')
        if self.capability['human_public_custom']:
            validation=manifest.get('validation') or {}
            if (validation.get('complete') is not True or validation.get('model_sha256')!=digest
                    or validation.get('rule_hash')!=expected_rule_hash):
                raise ValueError('Public model lacks matching completed evaluation evidence')
        raw_build = manifest.get('serving_build')
        self.serving_deck = fixed_team_serving_deck(raw_build, deck_id) if raw_build is not None else preset(deck_id)
        self.serving_build_sha256 = serving_deck_hash(self.serving_deck)
        if raw_build is None and (manifest.get('validation') or {}).get('serving_build_sha256') is not None:
            raise ValueError('Evaluated serving build is missing')
        if (raw_build is None and self.capability['human_public_custom'] and manifest.get('learner_deck')
                and not is_exact_preset(manifest['learner_deck'],deck_id)):
            raise ValueError('Custom allocation model requires an explicit serving build')
        if raw_build is not None:
            validation = manifest.get('validation') or {}
            if (not expanded or not self.capability['human_public_custom']
                    or validation.get('serving_build_sha256') != self.serving_build_sha256):
                raise ValueError('Serving build lacks matching model/build evaluation evidence')
        if self.schema!=OPENING_SCHEMA and manifest.get('opening_policy','keep_all')!='keep_all':
            raise ValueError('Legacy schema cannot claim learned mulligan')
        if self.schema==OPENING_SCHEMA and manifest.get('opening_policy')!='learned_joint_v1':
            raise ValueError('Mulligan schema requires learned opening policy metadata')
        h = int(manifest['hidden'])
        d = 2 + 2*len(SIDE_FIELDS) + 2*N_SEATS*len(CHAR_FIELDS) + 3*len(CARD_IDS)
        if expanded:
            d = state_dim()
        cand_dim = OPENING_CAND_DIM if self.schema==OPENING_SCHEMA else CAND_DIM if expanded else 10
        shapes = {
            'state_net.0.weight': (h, d), 'state_net.0.bias': (h,),
            'state_net.2.weight': (h, h), 'state_net.2.bias': (h,),
            'cand_net.0.weight': (h, cand_dim), 'cand_net.0.bias': (h,),
            'cand_net.2.weight': (h, h), 'cand_net.2.bias': (h,),
            'score.weight': (1, h), 'score.bias': (1,),
        }
        if not 1 <= h <= 2048 or manifest['state_dim'] != d or manifest['cand_dim'] != cand_dim:
            raise ValueError('Invalid model dimensions')
        with np.load(path, allow_pickle=False) as arrays:
            self.weights = {name: arrays[name].astype(np.float32, copy=True) for name in shapes}
        for name, shape in shapes.items():
            if self.weights[name].shape != shape or not np.isfinite(self.weights[name]).all():
                raise ValueError(f'Invalid model tensor {name}')
            self.weights[name].flags.writeable = False
        self.manifest = manifest
        self.version = digest
        self.hidden = h
        self.deck_id = deck_id

    def select_public_action(self, view, actions):
        """Offline/report policy contract: visible observation in, legal index out."""
        from app.modules.card_game.rl.public_observation import encode_public
        if not actions:
            raise ValueError('No legal actions')
        from app.modules.card_game.rl.opening_observation import OPENING_SCHEMA,encode_opening
        if view.get('phase') == 'mulligan' and self.schema!=OPENING_SCHEMA:
            return next(i for i,a in enumerate(actions) if a['type']=='mulligan' and not a.get('card_ids'))
        state, candidates = encode_opening(view,actions) if self.schema==OPENING_SCHEMA else encode_public(view, actions, schema=self.schema)
        return int(self.scores(state, candidates).argmax())

    def scores(self, state, candidates):
        import numpy as np
        def linear(x, name):
            # Small matrices: avoid a multi-thread BLAS pool per HTTP request.
            return np.einsum('...j,ij->...i', x, self.weights[name+'.weight'], optimize=False) + self.weights[name+'.bias']
        sh = np.maximum(linear(state, 'state_net.0'), 0)
        sh = np.maximum(linear(sh, 'state_net.2'), 0)
        ch = np.maximum(linear(candidates, 'cand_net.0'), 0)
        ch = np.maximum(linear(ch, 'cand_net.2'), 0)
        scores = np.sum(ch*sh, axis=-1)/np.sqrt(np.float32(self.hidden)) + linear(ch, 'score').reshape(-1)
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite model scores')
        return scores


def load_model(deck_id):
    preset(deck_id)
    from .experimental_cross_model import ARCHIVE, EXPERIMENTAL, ExperimentalCrossModel
    from .recovery_model import RELEASE, RecoveryServingModel
    configured = Path(config.DUEL_AI_MODEL_DIR).resolve()
    recovery = configured / RELEASE
    root = recovery if recovery.is_dir() else (ARCHIVE if deck_id in EXPERIMENTAL else configured)
    if deck_id in PENDING_PRESETS and not (root / f'{deck_id}.json').is_file():
        raise RuleValidationError('吱零薄海高级人机尚待训练与验收，请先使用普通人机。')
    key = (str(root), deck_id)
    with _LOCK:
        try:
            signature=tuple((path.stat().st_mtime_ns,path.stat().st_size)
                            for path in ([root/f'{deck_id}.json',root/f'{deck_id}.npz']
                                         + ([root/'serving.json'] if root == recovery else [])))
            if root == recovery:
                from app.modules.card_game.rl.cross_lineup import identity
                signature += (identity(),)
            if key not in _CACHE or getattr(_CACHE[key], '_file_signature', None) != signature:
                manifest = json.loads((root / f'{deck_id}.json').read_text(encoding='utf-8'))
                from app.modules.card_game.rl.league_schema import SCHEMA as LEAGUE_SCHEMA
                if root == recovery:
                    candidate = RecoveryServingModel(root, deck_id)
                elif deck_id in EXPERIMENTAL:
                    candidate = ExperimentalCrossModel(deck_id)
                elif manifest.get('schema') == 'fixed_ten_v1':
                    from .fixed_model import FixedServingModel
                    candidate = FixedServingModel(root, deck_id)
                elif manifest.get('schema') == LEAGUE_SCHEMA:
                    from app.modules.card_game.rl.league_policy import LeagueModel
                    candidate = LeagueModel(root, deck_id, require_approved=True)
                else:
                    candidate = FrozenModel(root, deck_id)
                candidate._file_signature = signature
                _CACHE[key] = candidate
        except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
            raise RuleValidationError('高级人机模型暂不可用，请选择普通人机。') from exc
        return _CACHE[key]


def choose_action(state, side='b'):
    from app.modules.card_game.engine.duel_v2 import legal_actions
    profile = state.get('ai_profile') or {}
    model = load_model(profile.get('deck_id'))
    if profile.get('version') != model.version:
        raise RuleValidationError('高级人机模型版本已变更，请重新开局。')
    actual_build = serving_deck(model, profile.get('deck_id'))
    expected_build_hash = model_build_hash(model, actual_build)
    if (profile.get('build_sha256') is not None and profile['build_sha256'] != expected_build_hash
            or profile.get('build_sha256') is None and not is_exact_preset(actual_build,profile.get('deck_id'))):
        raise RuleValidationError('高级人机构筑版本已变更，请重新开局。')
    actions = [a for a in legal_actions(state, side) if a['type'] != 'concede']
    if not actions:
        raise RuleValidationError('高级人机没有可执行动作。')
    from app.modules.card_game.rl.opening_observation import OPENING_SCHEMA
    from .recovery_model import SCHEMAS as RECOVERY_SCHEMAS
    if state['phase'] == 'mulligan' and getattr(model,'schema',None) not in (OPENING_SCHEMA, 'official_public_v5', 'fixed_ten_v1', *RECOVERY_SCHEMAS):
        # The model was trained with the initial hand retained.
        return next(a for a in actions if not a.get('card_ids'))
    from app.modules.card_game.rl.public_schema import SCHEMA as PUBLIC_SCHEMA
    from app.modules.card_game.rl.league_schema import SCHEMA as LEAGUE_SCHEMA
    if model.schema in (PUBLIC_SCHEMA,OPENING_SCHEMA,LEAGUE_SCHEMA,'fixed_ten_v1', 'cross_five_grounded_wdl_v1', *RECOVERY_SCHEMAS):
        from app.modules.card_game.engine.duel_v2 import observe
        fallback = actions[model.select_public_action(observe(state, side, include_previews=False), actions)]
        from .advanced_search import choose
        return choose(state, side, model, fallback)
    try:
        values, candidates = encode(state, side, actions)
    except (KeyError, ValueError) as exc:
        raise RuleValidationError('高级人机无法编码当前公开构筑，请更换构筑或选择普通人机。') from exc
    return actions[int(model.scores(values, candidates).argmax())]


def model_build_hash(model, build):
    from app.modules.card_game.rl.league_schema import SCHEMA as LEAGUE_SCHEMA, build_hash
    from .recovery_model import SCHEMAS as RECOVERY_SCHEMAS
    return build_hash(build) if getattr(model, 'schema', None) in (LEAGUE_SCHEMA, 'fixed_ten_v1', 'cross_five_grounded_wdl_v1', *RECOVERY_SCHEMAS) else serving_deck_hash(build)
