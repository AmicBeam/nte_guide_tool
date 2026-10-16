"""Apply the configured China-server challenge calendar during offline refresh."""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path


def apply_challenge_schedule(challenges):
    # The opening season spans launch to the first regular rotation.
    periods = {'Abyss_1': ('2026-04-23', '2026-05-29')}
    anchor = date(2026, 10, 16)
    for number in range(2, 15):
        begin = anchor + timedelta(days=14 * (number - 12))
        periods[f'Abyss_{number}'] = (begin.isoformat(), (begin + timedelta(days=14)).isoformat())
    for challenge in challenges:
        identifier = challenge['id']
        if challenge.get('is_permanent') or identifier not in periods:
            continue
        challenge['begin_date'], challenge['end_date'] = periods[identifier]
        if identifier == 'Abyss_14':
            challenge['name'] = '未命名环线'
    return challenges


if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1] / 'app/modules/guide/data/catalog.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    apply_challenge_schedule(data['challenges'])
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    temporary.replace(path)
    for challenge in data['challenges']:
        print(challenge['id'], challenge['name'], challenge.get('begin_date'), challenge.get('end_date'))
