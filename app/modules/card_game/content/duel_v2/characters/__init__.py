"""Load V2 character kits. Importing this package registers passives and card effects."""
from app.modules.card_game.content.duel_v2.registry import KITS, collect_effects, register  # noqa: F401

from . import (  # noqa: F401
    nanali, zero, jiuyuan, xun, anhunqu, canhong, zaowu, lingke, zhenhong, hathor, iloy,
    bohe, baicang, xiaozhi, haiyue, edgar, haniya, yi, adler,
)

EFFECTS, TARGET_POLICIES = collect_effects()

__all__ = ['EFFECTS', 'KITS', 'TARGET_POLICIES']
