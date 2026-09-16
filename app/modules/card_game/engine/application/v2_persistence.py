"""Latest-state cache and ordered, generation-guarded V2 snapshot persistence."""
from __future__ import annotations

import atexit
import queue
import threading
import time
from copy import deepcopy
from app.modules.card_game.engine.duel_v2.entities import hydrate_entities

from app.db import atomic_transaction, db
from app.errors import PersistenceError
from app.modules.card_game.engine.application import v2_repository as repository
from app.utils.logger import get_logger

_logger = get_logger('nte.duel_v2.persistence')
_lock = threading.RLock()
_cache: dict[tuple[str, int], dict] = {}
_queue: queue.Queue = queue.Queue()
_started = False


def _key(room_id: int) -> tuple[str, int]:
    return str(db.database), room_id


def load(room_id: int) -> dict | None:
    key = _key(room_id)
    with _lock:
        entry = _cache.get(key)
        if entry is not None:
            return deepcopy(entry['envelope'])
        entry = repository.get_run(room_id)
        if entry is not None:
            hydrate_entities(entry['envelope']['game'])
            _cache[key] = entry
            return deepcopy(entry['envelope'])
    return None


def initialize(room, envelope: dict) -> None:
    envelope = deepcopy(envelope)
    hydrate_entities(envelope['game'])
    generation = repository.replace_run(room, envelope)
    with _lock:
        _cache[_key(room.id)] = {'generation': generation,
                                'revision': envelope['game']['version'],
                                'envelope': deepcopy(envelope)}


def publish(room, envelope: dict, *, final: bool = False) -> None:
    key = _key(room.id)
    with _lock:
        entry = _cache.get(key)
        if entry is None:
            entry = repository.get_run(room.id)
        if entry is None:
            raise PersistenceError('对局快照不存在，请重新进入房间。')
        generation = entry['generation']
    snapshot = deepcopy(envelope)
    if final:
        # Finishing the room and saving the final result are one durable change.
        with atomic_transaction():
            if not repository.write_snapshot(room.id, generation, snapshot):
                raise PersistenceError('对局版本已变化，请刷新后重试。')
            repository.set_status(room, 'finished')
    with _lock:
        _cache[key] = {'generation': generation, 'revision': snapshot['game']['version'],
                       'envelope': snapshot}
    if not final:
        _ensure_worker()
        _queue.put((key, generation, snapshot, 0))


def invalidate(room_id: int) -> None:
    with _lock:
        _cache.pop(_key(room_id), None)


def clear_cache() -> None:
    """Call only after flush when simulating a process restart in tests."""
    with _lock:
        _cache.clear()


def flush(timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while _queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.01)
    return _queue.unfinished_tasks == 0


def _ensure_worker() -> None:
    global _started
    with _lock:
        if not _started:
            threading.Thread(target=_worker, name='nte-duel-v2-persistence', daemon=True).start()
            _started = True


def _latest(key, generation, envelope) -> bool:
    with _lock:
        current = _cache.get(key)
        return bool(current and current['generation'] == generation
                    and current['revision'] == envelope['game']['version'])


def _worker() -> None:
    while True:
        key, generation, envelope, attempts = _queue.get()
        try:
            if key[0] != str(db.database) or not _latest(key, generation, envelope):
                continue
            with atomic_transaction():
                repository.write_snapshot(key[1], generation, envelope)
        except Exception:
            _logger.exception('V2 snapshot write failed room_id=%s revision=%s',
                              key[1], envelope['game']['version'])
            if attempts < 2 and _latest(key, generation, envelope):
                time.sleep(0.05 * (attempts + 1))
                _queue.put((key, generation, envelope, attempts + 1))
        finally:
            if not db.is_closed():
                db.close()
            _queue.task_done()


atexit.register(flush)
