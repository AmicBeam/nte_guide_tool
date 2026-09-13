"""Database adapter for the versioned duel; never interprets combat rules."""
from __future__ import annotations

import json
from datetime import datetime
from secrets import token_hex

from app.models import DuelV2Build, DuelV2Member, DuelV2Replay, DuelV2ReplayStar, DuelV2Room, DuelV2Run, Player

UNFAVORITED_REPLAY_LIMIT = 30


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


def upsert_replay(room: DuelV2Room, payload: dict, meta: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=False)
    winner = meta.get('winner')
    (DuelV2Replay.insert(
        room=room,
        room_code=room.room_code,
        mode=meta.get('mode') or room.mode,
        status=meta.get('status') or 'playing',
        winner=winner,
        player_a_id=meta.get('player_a_id'),
        player_b_id=meta.get('player_b_id'),
        name_a=meta.get('name_a') or '',
        name_b=meta.get('name_b') or '',
        payload=encoded,
    ).on_conflict(conflict_target=[DuelV2Replay.room], update={
        DuelV2Replay.room_code: room.room_code,
        DuelV2Replay.mode: meta.get('mode') or room.mode,
        DuelV2Replay.status: meta.get('status') or 'playing',
        DuelV2Replay.winner: winner,
        DuelV2Replay.player_a_id: meta.get('player_a_id'),
        DuelV2Replay.player_b_id: meta.get('player_b_id'),
        DuelV2Replay.name_a: meta.get('name_a') or '',
        DuelV2Replay.name_b: meta.get('name_b') or '',
        DuelV2Replay.payload: encoded,
        DuelV2Replay.updated_at: datetime.utcnow(),
    }).execute())


def get_replay(room_id: int) -> dict | None:
    row = DuelV2Replay.get_or_none(DuelV2Replay.room == room_id)
    if row is None:
        return None
    return {'row': row, 'payload': json.loads(row.payload)}


def replay_by_code(code: str) -> DuelV2Replay | None:
    return (DuelV2Replay.select().where(DuelV2Replay.room_code == code.upper())
            .order_by(DuelV2Replay.updated_at.desc(), DuelV2Replay.id.desc()).first())


def replay_payload(row: DuelV2Replay) -> dict:
    return json.loads(row.payload)


def list_replays_for(player: Player, limit: int = 12) -> list[DuelV2Replay]:
    return list(DuelV2Replay.select().where(
        (DuelV2Replay.player_a_id == player.id) | (DuelV2Replay.player_b_id == player.id)
    ).order_by(DuelV2Replay.updated_at.desc(), DuelV2Replay.id.desc()).limit(limit))


def _player_replay_clause(player: Player):
    return (DuelV2Replay.player_a_id == player.id) | (DuelV2Replay.player_b_id == player.id)


def starred_replay_ids(player: Player) -> set[int]:
    return {row.replay_id for row in DuelV2ReplayStar.select().where(DuelV2ReplayStar.player == player)}


def is_replay_starred(player: Player, replay: DuelV2Replay) -> bool:
    return DuelV2ReplayStar.select().where(
        (DuelV2ReplayStar.player == player) & (DuelV2ReplayStar.replay == replay)
    ).exists()


def set_replay_star(player: Player, replay: DuelV2Replay, favorited: bool) -> None:
    if favorited:
        (DuelV2ReplayStar.insert(player=player, replay=replay)
         .on_conflict_ignore().execute())
        return
    DuelV2ReplayStar.delete().where(
        (DuelV2ReplayStar.player == player) & (DuelV2ReplayStar.replay == replay)
    ).execute()


def unfavorited_keep_ids(player: Player) -> set[int]:
    starred = starred_replay_ids(player)
    query = DuelV2Replay.select(DuelV2Replay.id).where(_player_replay_clause(player))
    if starred:
        query = query.where(DuelV2Replay.id.not_in(list(starred)))
    return {
        row.id
        for row in query.order_by(DuelV2Replay.updated_at.desc(), DuelV2Replay.id.desc())
        .limit(UNFAVORITED_REPLAY_LIMIT)
    }


def replay_kept_for(player: Player, replay: DuelV2Replay) -> bool:
    if is_replay_starred(player, replay):
        return True
    return replay.id in unfavorited_keep_ids(player)


def list_visible_replays_for(player: Player) -> list[DuelV2Replay]:
    starred = list(
        DuelV2Replay.select()
        .join(DuelV2ReplayStar, on=(DuelV2ReplayStar.replay == DuelV2Replay.id))
        .where(DuelV2ReplayStar.player == player)
        .where(_player_replay_clause(player))
        .order_by(DuelV2Replay.updated_at.desc(), DuelV2Replay.id.desc())
    )
    starred_ids = {row.id for row in starred}
    unfav_query = DuelV2Replay.select().where(_player_replay_clause(player))
    if starred_ids:
        unfav_query = unfav_query.where(DuelV2Replay.id.not_in(list(starred_ids)))
    unfav = list(
        unfav_query.order_by(DuelV2Replay.updated_at.desc(), DuelV2Replay.id.desc())
        .limit(UNFAVORITED_REPLAY_LIMIT)
    )
    return starred + unfav


def delete_replay(row: DuelV2Replay) -> None:
    DuelV2ReplayStar.delete().where(DuelV2ReplayStar.replay == row).execute()
    row.delete_instance()


def trim_unfavorited_replays(player: Player) -> list[int]:
    starred = starred_replay_ids(player)
    query = DuelV2Replay.select().where(
        _player_replay_clause(player) & (DuelV2Replay.status == 'finished')
    )
    if starred:
        query = query.where(DuelV2Replay.id.not_in(list(starred)))
    overflow = list(query.order_by(DuelV2Replay.updated_at.desc(), DuelV2Replay.id.desc()))[
        UNFAVORITED_REPLAY_LIMIT:
    ]
    removed = []
    for row in overflow:
        other_id = row.player_b_id if row.player_a_id == player.id else row.player_a_id
        if other_id:
            other = Player.get_or_none(Player.id == other_id)
            if other is not None and replay_kept_for(other, row):
                continue
        removed.append(row.room_id)
        delete_replay(row)
    return removed
