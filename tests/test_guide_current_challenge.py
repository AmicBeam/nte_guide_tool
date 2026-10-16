from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import unittest

from flask import Flask
from app.modules.guide import blueprint
from app.modules.guide.challenge_calendar import is_current_challenge


class CurrentChallengeTest(unittest.TestCase):
    def test_beijing_five_am_boundaries_and_timezone(self):
        record = dict(begin_date='2026-10-16', end_date='2026-10-30', is_permanent=False)
        for timestamp, expected in [
            ('2026-10-16T04:59:59+08:00', False),
            ('2026-10-16T05:00:00+08:00', True),
            ('2026-10-30T04:59:59+08:00', True),
            ('2026-10-30T05:00:00+08:00', False),
            ('2026-10-15T21:00:00+00:00', True),
            ('2026-10-29T21:00:00+00:00', False),
        ]:
            with self.subTest(timestamp=timestamp):
                self.assertEqual(is_current_challenge(record, datetime.fromisoformat(timestamp)), expected)
        for changes in [dict(is_permanent=True), dict(begin_date=None), dict(end_date=None),
                        dict(begin_date='invalid')]:
            self.assertFalse(is_current_challenge(dict(record, **changes)))

    def test_index_uses_fresh_system_time_each_request(self):
        root = Path(__file__).resolve().parents[1]
        app = Flask(__name__, template_folder=str(root / 'app/templates'))
        app.register_blueprint(blueprint)
        client = app.test_client()
        with patch('app.modules.guide.challenge_calendar.datetime', wraps=datetime) as clock:
            for timestamp, identifier in [('2026-10-30T04:59:59+08:00', 'Abyss_12'),
                                          ('2026-10-30T05:00:00+08:00', 'Abyss_13')]:
                clock.now.return_value = datetime.fromisoformat(timestamp)
                response = client.get('/guide?kind=challenges')
                self.assertEqual(response.status_code, 200)
                import re
                highlights = re.findall(r'class="[^"]*guide-challenge-current[^"]*" href="([^"]+)"', response.text)
                self.assertEqual(highlights, ['/guide/challenges/' + identifier])


if __name__ == '__main__':
    unittest.main()
