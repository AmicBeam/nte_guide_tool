"""Challenge dates remain coherent across refresh and public formats."""
import importlib.util
import re
from pathlib import Path
import unittest
from unittest.mock import patch
from datetime import date, datetime
from flask import Flask
from app.modules.guide import blueprint, service

ROOT = Path(__file__).resolve().parents[1]


class ChallengeScheduleTest(unittest.TestCase):
    def test_rotation_and_opening_exception(self):
        spec = importlib.util.spec_from_file_location('guide_challenge_schedule', ROOT / 'scripts/guide_challenge_schedule.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        records = [{'id': f'Abyss_{i}', 'name': '澄明环线', 'begin_date': None, 'end_date': None} for i in range(1, 15)]
        records += [{'id': 'Abyss_Common', 'is_permanent': True, 'begin_date': None},
                    {'id': 'Abyss_15', 'begin_date': None}]
        module.apply_challenge_schedule(records)
        self.assertEqual(records[0]['begin_date'], '2026-04-23')
        self.assertEqual(records[0]['end_date'], '2026-05-29')
        for i in range(1, 14):
            self.assertEqual(records[i-1]['end_date'], records[i]['begin_date'])
            self.assertEqual((date.fromisoformat(records[i]['end_date']) - date.fromisoformat(records[i]['begin_date'])).days, 14)
        self.assertEqual(records[11]['begin_date'], '2026-10-16')
        self.assertEqual(records[13]['name'], '未命名环线')
        self.assertIsNone(records[14]['begin_date'])
        self.assertIsNone(records[15]['begin_date'])

    @patch('app.modules.guide.challenge_calendar.current_time', return_value=datetime.fromisoformat('2026-11-14T12:00:00+08:00'))
    def test_every_challenge_shows_resonance_name_and_effect(self, _clock):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        client = app.test_client()
        for challenge in service.catalog()['challenges']:
            with self.subTest(challenge=challenge['id']):
                response = client.get('/guide/challenges/' + challenge['id'] + '?stage=1')
                self.assertEqual(response.status_code, 200)
                section = response.text.split('id="section-resonance"', 1)[1].split('</section>', 1)[0]
                self.assertIn(challenge['resonance']['name'], section)
                from html import unescape
                self.assertIn(challenge['resonance']['description'], unescape(section))
                self.assertNotIn('built-in method', section)

    @patch('app.modules.guide.challenge_calendar.current_time', return_value=datetime.fromisoformat('2026-11-14T12:00:00+08:00'))
    def test_public_dates_and_unnamed_last_season(self, _clock):
        app = Flask(__name__, template_folder=str(ROOT / 'app/templates'))
        app.register_blueprint(blueprint)
        client = app.test_client()
        data = client.get('/api/guide/challenges/Abyss_12').json['data']
        self.assertEqual((data['begin_date'], data['end_date']), ('2026-10-16', '2026-10-30'))
        last = client.get('/api/guide/challenges/Abyss_14').json['data']
        self.assertEqual(last['name'], '未命名环线')
        self.assertEqual((last['begin_date'], last['end_date']), ('2026-11-13', '2026-11-27'))
        html = client.get('/guide?kind=challenges').text
        self.assertIn('2026-10-16 → 2026-10-30', re.sub(r'<[^>]+>', '', html))
        self.assertIn('未命名环线', html)
        self.assertIn('长期开放', html)
        self.assertNotIn('日期未提供', html)
        md = client.get('/guide/challenges/Abyss_12.md?section=begin_date').text
        self.assertIn('2026-10-16', md)
        detail = client.get('/guide/challenges/Abyss_14?stage=12')
        self.assertEqual(detail.status_code, 200)
        self.assertIn('2026-11-13', detail.text)
        self.assertIn('04:59:59', detail.text)
        self.assertNotIn('built-in method', detail.text)


if __name__ == '__main__':
    unittest.main()
