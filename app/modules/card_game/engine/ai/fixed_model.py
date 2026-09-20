"""Explicitly authorized serving adapter for frozen fixed-lineup numeric models."""
from app.modules.card_game.rl.fixed_lineup import FixedModel, SCHEMA, identity
from app.modules.card_game.rl.league_schema import SEATS as PUBLIC_SEATS
from app.modules.card_game.content.duel_v2 import validate_deck


class FixedServingModel(FixedModel):
    def __init__(self, directory, key):
        super().__init__(directory, key)
        authorization = self.manifest.get('serving_authorization') or {}
        self.serving_build_sha256 = self.manifest['build_sha256']
        self.serving_authorized = bool(
            authorization.get('kind') == 'explicit_user_candidate'
            and authorization.get('request') and authorization.get('report')
            and authorization.get('model_sha256') == self.version
            and authorization.get('build_sha256') == self.serving_build_sha256
            and authorization.get('rule_hash') == identity())
        if not self.serving_authorized:
            raise ValueError('Fixed model lacks explicit serving authorization')
        self.schema = SCHEMA
        self.deck_id = key
        self.capability = dict(kind=SCHEMA, human_public_custom=True, bot_preset=key)

    def validate_human(self, build):
        build = validate_deck(build)
        # Keep website access scoped to the existing public eight-character pool.
        if not set(build['character_ids']) <= set(PUBLIC_SEATS):
            raise ValueError('高级人机仅支持公开角色构筑。')
        return self.capability
