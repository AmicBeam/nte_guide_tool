"""Explicit, checksum-pinned CPU serving for two unapproved cross policies.

These archived weights are user-authorized experimental opponents only.  The
original manifests remain unchanged and cannot acquire quality approval here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from app.modules.card_game.rl import cross_grounded, cross_lineup, cross_runtime
from app.modules.card_game.rl.league_schema import build_hash


ARCHIVE = Path(__file__).resolve().parent / 'models/research/five-cross-20260924'
SOURCE_RULE_HASH = '9f3f368c1d2422092c4bcf483c07c1d2b7e76743d0b5cf914c0fe32033b36261'
COMPATIBLE_RUNTIME_HASH = 'e22c43e1595a86bbe5ebdd2ba5fb8333eb5f05cf243207ce1db9fc6809f8df7f'
EXPERIMENTAL = {
    'zhenhong': (
        '29f6aa981206e1faf6761368c8aceec7bc1f1c183d068f1bf343bdf7e449151b',
        '3c1258b60d9e985a30da55b6d98bc6526064c450a0292ece64c378ebc7f428e1',
    ),
    'murk': (
        '41833567c30d7156368bb74193343b3fffee153277ee91d11c43f0f9cefc90bd',
        '744b4b9d71cca135df6f84ecb0479979cfc1ae6e922e8f62a535a97f902d9f51',
    ),
}


class ExperimentalCrossModel(cross_runtime.CrossModel):
    """Use archived bytes only under the explicitly checked current contract."""

    schema = cross_lineup.SCHEMA
    experimental = True

    def __init__(self, deck_id: str):
        if deck_id not in EXPERIMENTAL:
            raise ValueError('Unknown experimental strategy')
        if cross_lineup.identity() != COMPATIBLE_RUNTIME_HASH:
            raise ValueError('Experimental model runtime has changed')
        manifest_file = ARCHIVE / f'{deck_id}.json'
        weights_file = ARCHIVE / f'{deck_id}.npz'
        expected_manifest, expected_weights = EXPERIMENTAL[deck_id]
        if hashlib.sha256(manifest_file.read_bytes()).hexdigest() != expected_manifest:
            raise ValueError('Experimental model manifest changed')
        if hashlib.sha256(weights_file.read_bytes()).hexdigest() != expected_weights:
            raise ValueError('Experimental model weights changed')
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        if (manifest.get('schema') != cross_lineup.SCHEMA
                or manifest.get('rule_hash') != SOURCE_RULE_HASH
                or manifest.get('sha256') != expected_weights
                or manifest.get('deck') != deck_id
                or manifest.get('features') != cross_lineup.all_features()
                or manifest.get('candidates') != cross_lineup.candidate_names()
                or manifest.get('candidate_transform_sha256') != cross_grounded.fingerprint()
                or manifest.get('wdl_order') != ['loss', 'draw', 'win']
                or manifest.get('automatic_serving_approval') is not False
                or (manifest.get('validation') or {}).get('approved') is not False):
            raise ValueError('Experimental model contract mismatch')
        hidden = manifest.get('hidden')
        if type(hidden) is not int or not 1 <= hidden <= 256:
            raise ValueError('Invalid experimental model width')
        shapes = {
            'state_net.0.weight': (hidden, len(cross_lineup.all_features())),
            'state_net.0.bias': (hidden,),
            'state_net.2.weight': (hidden, hidden),
            'state_net.2.bias': (hidden,),
            'cand_net.0.weight': (hidden, cross_grounded.DIM),
            'cand_net.0.bias': (hidden,),
            'cand_net.2.weight': (hidden, hidden),
            'cand_net.2.bias': (hidden,),
            'score.weight': (1, hidden),
            'score.bias': (1,),
            'value.0.weight': (hidden, hidden),
            'value.0.bias': (hidden,),
            'value.2.weight': (1, hidden),
            'value.2.bias': (1,),
            'wdl.weight': (3, hidden + 1),
            'wdl.bias': (3,),
        }
        with np.load(weights_file, allow_pickle=False) as saved:
            if set(saved.files) != set(shapes):
                raise ValueError('Experimental model tensor set mismatch')
            weights = {name: saved[name].copy() for name in shapes}
        for name, shape in shapes.items():
            if weights[name].shape != shape or not np.isfinite(weights[name]).all():
                raise ValueError('Experimental model tensor mismatch')
            weights[name].flags.writeable = False
        if build_hash(manifest['build']) != manifest.get('build_sha256'):
            raise ValueError('Experimental model build mismatch')
        self.hidden = hidden
        self.weights = weights
        self.serving_deck = manifest['build']
        self.manifest = manifest
        self.version = expected_weights
        self.deck_id = deck_id
        self.capability = {
            'kind': 'experimental_cross_v1',
            'experimental': True,
            'human_public_custom': True,
            'bot_preset': deck_id,
        }

    def select_public_action(self, view, actions):
        if not actions:
            raise ValueError('No legal actions')
        state, candidates = cross_lineup.encode(view, actions)
        return int(self.scores(state, candidates).argmax())

    def validate_human(self, deck):
        from app.modules.card_game.content.duel_v2 import deck_uses_test_characters, validate_deck

        try:
            build = validate_deck(deck)
        except ValueError as exc:
            raise ValueError('Invalid human deck for experimental AI') from exc
        if deck_uses_test_characters(build):
            raise ValueError('Experimental AI accepts public characters only')
        return self.capability
