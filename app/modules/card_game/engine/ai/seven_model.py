"""Explicit adoption of frozen seven-team candidates; CPU NumPy inference only."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from app.modules.card_game.rl import seven_lineup, seven_policy
from app.modules.card_game.rl.seven_runtime import SevenModel

RELEASE = 'seven-20261024'
ADOPTED = frozenset(('starter', 'quick-rush', 'zhenhong', 'murk', 'collapse'))
SCHEMA = seven_policy.SCHEMA


class SevenServingModel(SevenModel):
    experimental = True

    def __init__(self, directory, key):
        root = Path(directory)
        binding = json.loads((root / 'serving.json').read_text(encoding='utf-8'))
        pin = binding.get('models', {}).get(key) or {}
        manifest_bytes = (root / f'{key}.json').read_bytes()
        manifest = json.loads(manifest_bytes)
        if (key not in ADOPTED or binding.get('kind') != 'explicit_user_candidate'
                or not binding.get('request') or binding.get('quality_approved') is not False
                or binding.get('runtime_rule_hash') != seven_lineup.identity()
                or pin.get('manifest_sha256') != sha256(manifest_bytes).hexdigest()
                or pin.get('model_sha256') != sha256((root / f'{key}.npz').read_bytes()).hexdigest()
                or pin.get('build_sha256') != manifest.get('build_sha256')
                or (manifest.get('validation') or {}).get('approved') is not False
                or type(manifest.get('hidden')) is not int or not 1 <= manifest['hidden'] <= 256):
            raise ValueError('Seven serving authorization mismatch')
        super().__init__(root, key)
        if any(value.dtype != np.float32 for value in self.weights.values()):
            raise ValueError('Seven serving tensor dtype mismatch')
        from app.modules.card_game.content.duel_v2 import validate_deck
        self.serving_deck = validate_deck(self.serving_deck)
        self.schema = SCHEMA
        self.deck_id = key
        self.serving_build_sha256 = manifest['build_sha256']
        self.serving_authorized = True
        self.capability = dict(kind=self.schema, experimental=True, human_public_custom=True, bot_preset=key)

    def select_public_action(self, view, actions):
        if not actions:
            raise ValueError('No legal actions')
        x, c = seven_lineup.encode(view, actions)
        return int(self.scores(x, c).argmax())

    def validate_human(self, deck):
        from app.modules.card_game.content.duel_v2 import validate_deck, deck_uses_test_characters
        build = validate_deck(deck)
        if deck_uses_test_characters(build):
            raise ValueError('高级人机仅支持公开角色构筑。')
        return self.capability
