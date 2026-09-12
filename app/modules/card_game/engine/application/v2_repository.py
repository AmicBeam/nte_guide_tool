"""Database adapter for the versioned duel; never interprets combat rules."""
from __future__ import annotations

import json
from datetime import datetime
from secrets import token_hex

from app.models import DuelV2Build, DuelV2Member, DuelV2Room, DuelV2Run, Player


def get_build(player: Player) -> dict | None:
    row = DuelV2Build.get_or_none(DuelV2Build.player == player)
    return json.loads(row.payload) if row else None


def save_build(player: Player, payload: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=False)
    (DuelV2Build.insert(player=player, payload=encoded)
     .on_conflict(conflict_target=[DuelV2Build.player], update={
         DuelV2Build.payload: encoded, DuelV2Build.updated_at: datetime.utcnow(),
     }).execute())


def current_room(player: Player) -> DuelV2Room | None:
    member = (DuelV2Member.select(DuelV2Member, DuelV2Room).join(DuelV2Room)
              .where((DuelV2Member.player == player) & (DuelV2Room.status != 'closed'))
              .order_by(DuelV2Room.updated_at.desc(), DuelV2Room.id.desc()).first())
    return member.room if member else None


def room_by_code(code: str) -> DuelV2Room | None:
    return DuelV2Room.get_or_none(DuelV2Room.room_code == code.upper())


def get_room(room_id: int) -> DuelV2Room | None:
    return DuelV2Room.get_or_none(DuelV2Room.id == room_id)


def create_room(player: Player, mode: str) -> DuelV2Room:
    while True:
        code = token_hex(3).upper()
        if not DuelV2Room.select().where(DuelV2Room.room_code == code).exists():
            return DuelV2Room.create(host=player, mode=mode, room_code=code)


def members(room: DuelV2Room) -> list[DuelV2Member]:
    return list(DuelV2Member.select(DuelV2Member, Player).join(Player)
                .where(DuelV2Member.room == room).order_by(DuelV2Member.side))


def member_for(room: DuelV2Room, player: Player) -> DuelV2Member | None:
    return DuelV2Member.get_or_none((DuelV2Member.room == room) & (DuelV2Member.player == player))


def add_member(room: DuelV2Room, player: Player, side: str, deck: dict, ready: bool) -> DuelV2Member:
    return DuelV2Member.create(room=room, player=player, side=side,
                               deck=json.dumps(deck, ensure_ascii=False), is_ready=ready)


def set_ready(member: DuelV2Member, ready: bool, deck: dict | None = None) -> None:
    member.is_ready = ready
    if deck is not None:
        member.deck = json.dumps(deck, ensure_ascii=False)
    member.save()


def member_deck(member: DuelV2Member) -> dict:
    return json.loads(member.deck)


def remove_member(member: DuelV2Member) -> None:
    member.delete_instance()


def clear_members(room: DuelV2Room) -> None:
    DuelV2Member.delete().where(DuelV2Member.room == room).execute()


def set_status(room: DuelV2Room, status: str) -> None:
    room.status = status
    room.updated_at = datetime.utcnow()
    room.save(only=[DuelV2Room.status, DuelV2Room.updated_at])


def get_run(room_id: int) -> dict | None:
    row = DuelV2Run.get_or_none(DuelV2Run.room == room_id)
    if row is None:
        return None
    return {'generation': row.generation, 'revision': row.revision,
            'envelope': json.loads(row.snapshot)}


def replace_run(room: DuelV2Room, envelope: dict) -> str:
    generation = token_hex(16)
    revision = int(envelope['game']['version'])
    encoded = json.dumps(envelope, ensure_ascii=False)
    (DuelV2Run.insert(room=room, generation=generation, revision=revision, snapshot=encoded)
     .on_conflict(conflict_target=[DuelV2Run.room], update={
         DuelV2Run.generation: generation, DuelV2Run.revision: revision,
         DuelV2Run.snapshot: encoded, DuelV2Run.updated_at: datetime.utcnow(),
     }).execute())
    return generation


def write_snapshot(room_id: int, generation: str, envelope: dict) -> bool:
    """CAS prevents a queued old game or older revision from being restored."""
    revision = int(envelope['game']['version'])
    changed = (DuelV2Run.update(snapshot=json.dumps(envelope, ensure_ascii=False),
                               revision=revision, updated_at=datetime.utcnow())
               .where((DuelV2Run.room == room_id) & (DuelV2Run.generation == generation)
                      & (DuelV2Run.revision < revision)).execute())
    return bool(changed)


def invalidate_run(room: DuelV2Room) -> None:
    # Keep the archived snapshot; only cancel pending writes to its generation.
    DuelV2Run.update(generation=token_hex(16)).where(DuelV2Run.room == room).execute()
