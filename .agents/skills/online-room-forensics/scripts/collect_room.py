"""Read one V2 room without importing app or writing server files."""
import hashlib
import json
import re
import sqlite3
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path


def validate(c):
    if not re.fullmatch(r'[0-9A-F]{6}', c['room']):
        raise ValueError('Room must be six hexadecimal characters')
    date.fromisoformat(c['date'])
    if not 1 <= c['tail_lines'] <= 20000:
        raise ValueError('tail_lines must be 1..20000')
    for relative in c.get('sources', []):
        p = Path(relative)
        if (p.is_absolute() or '..' in p.parts or '\\' in relative
                or not relative.startswith('app/modules/card_game/')
                or p.suffix not in ('.py', '.js', '.css', '.json', '.html')
                or any(part.startswith('.') for part in p.parts)):
            raise ValueError('Source must be ordinary card_game code')


def collect(c):
    validate(c)
    root = Path(c['project']).resolve()
    database = Path(c.get('database') or root / 'nte_board_game.db').resolve()
    result = {'schema_version': 1, 'room_code': c['room'], 'room': None,
              'collected_at_utc': datetime.now(timezone.utc).isoformat(), 'database': str(database)}
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=5) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        row = db.execute('SELECT id, mode, status FROM duel_v2_room WHERE room_code=?', (c['room'],)).fetchone()
        if row:
            result['room'] = dict(zip(('id', 'mode', 'status'), row))
            run = db.execute('SELECT revision, updated_at, snapshot FROM duel_v2_run WHERE room_id=?', (row[0],)).fetchone()
            if run:
                result.update(revision=run[0], updated_at=run[1], game=json.loads(run[2]).get('game'))
            replay = db.execute('SELECT payload FROM duel_v2_replay WHERE room_id=?', (row[0],)).fetchone()
            if replay:
                result['replay_views'] = json.loads(replay[0]).get('views')
    log_path = Path(c.get('log_dir') or root / 'logs') / (c['date'] + '.log')
    lines, total = deque(maxlen=c['tail_lines']), 0
    if log_path.exists():
        with log_path.open(encoding='utf-8', errors='replace') as stream:
            for line in stream:
                total += 1
                lines.append(line.rstrip())
    result['log_scope'] = {'path': str(log_path), 'exists': log_path.exists(),
                           'total_lines': total, 'retained_lines': len(lines), 'truncated': total > len(lines)}
    result['matching_service_logs'] = [line for line in lines if c['room'] in line]
    if c.get('errors'):
        errors, capture = [], 0
        for line in lines:
            if 'V2 request failed' in line or 'V2 snapshot write failed' in line:
                capture = 100
            elif capture and re.match(r'^\d{4}-\d\d-\d\d .* \| (INFO|WARNING|ERROR) \| ', line):
                capture = 0
            if capture:
                errors.append(line)
                capture -= 1
        result['unattributed_recent_v2_errors'] = errors
    result['server_sources'] = {}
    for relative in c.get('sources', []):
        p = (root / relative).resolve()
        p.relative_to(root)
        if not p.is_file():
            result['server_sources'][relative] = {'missing': True}
            continue
        if p.stat().st_size > 2_000_000:
            raise ValueError('Source exceeds 2 MB: ' + relative)
        content = p.read_bytes()
        result['server_sources'][relative] = {'sha256': hashlib.sha256(content).hexdigest(), 'text': content.decode('utf-8')}
    return result
