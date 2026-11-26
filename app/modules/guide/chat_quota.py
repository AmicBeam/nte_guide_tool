"""Persistent per-account rolling quota; stores timestamps, never chat content."""
from datetime import datetime, timedelta
import json
import math

from peewee import ForeignKeyField, TextField

from app.db import atomic_transaction
from app.models import BaseModel, Player

HOURLY_LIMIT = 10


class GuideChatQuota(BaseModel):
    player = ForeignKeyField(Player, unique=True, on_delete='CASCADE')
    timestamps = TextField(default='[]')


def consume(player_id: int) -> int:
    """Reserve one request atomically, or return seconds until the next slot."""
    with atomic_transaction():
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=1)
        row, _ = GuideChatQuota.get_or_create(player=player_id)
        recent = [datetime.fromisoformat(value) for value in json.loads(row.timestamps)]
        recent = [stamp for stamp in recent if stamp > cutoff]
        if len(recent) >= HOURLY_LIMIT:
            return max(1, math.ceil((recent[0] + timedelta(hours=1) - now).total_seconds()))
        recent.append(now)
        row.timestamps = json.dumps([stamp.isoformat() for stamp in recent])
        row.save(only=[GuideChatQuota.timestamps])
    return 0
