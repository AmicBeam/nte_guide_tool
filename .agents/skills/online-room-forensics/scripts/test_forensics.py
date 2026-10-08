"""Synthetic offline checks; never connects to a real host."""
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from collect_room import collect
from prepare_query import prepare, deployment_values


class ForensicsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='nte-forensics-test-')
        self.root = Path(self.tmp.name)
        self.db = self.root/'nte_board_game.db'
        with sqlite3.connect(self.db) as db:
            db.executescript('CREATE TABLE duel_v2_room(id,room_code,mode,status); CREATE TABLE duel_v2_run(room_id,revision,updated_at,snapshot); CREATE TABLE duel_v2_replay(room_id,payload);')
            db.executemany('INSERT INTO duel_v2_room VALUES(?,?,?,?)', [(1,'26F33E','advanced','closed'),(2,'ABCDEF','pvp','playing')])
            db.execute('INSERT INTO duel_v2_run VALUES(1,54,?,?)', ('2026-09-16', json.dumps({'game':{'version':54},'requests':['excluded']})))
            db.execute('INSERT INTO duel_v2_replay VALUES(1,?)', (json.dumps({'views':{'a':{'events':[{'seq':1}]}}}),))
        (self.root/'logs').mkdir()
        (self.root/'logs/2026-09-16.log').write_text('old\n26F33E selected\nV2 request failed path=/action\nTraceback synthetic\n')
        self.config = dict(room='26F33E',date='2026-09-16',project=str(self.root),tail_lines=3,errors=True,sources=[])

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_only_and_scoped(self):
        before = hashlib.sha256(self.db.read_bytes()).hexdigest()
        result = collect(self.config)
        self.assertEqual(result['room']['id'],1)
        self.assertEqual(result['game'],{'version':54})
        self.assertNotIn('requests',result)
        self.assertTrue(result['log_scope']['truncated'])
        self.assertEqual(result['matching_service_logs'],['26F33E selected'])
        self.assertEqual(len(result['unattributed_recent_v2_errors']),2)
        self.assertEqual(before,hashlib.sha256(self.db.read_bytes()).hexdigest())
        self.assertIsNone(collect(dict(self.config,room='AAAAAA'))['room'])

    def test_missing_database_not_created(self):
        missing=self.root/'missing.db'
        with self.assertRaises(sqlite3.OperationalError): collect(dict(self.config,database=str(missing)))
        self.assertFalse(missing.exists())

    def test_sources_and_validation(self):
        directory=self.root/'app/modules/card_game';directory.mkdir(parents=True)
        (directory/'sample.py').write_text('print("synthetic")')
        result=collect(dict(self.config,sources=['app/modules/card_game/sample.py']))
        self.assertEqual(len(result['server_sources']['app/modules/card_game/sample.py']['sha256']),64)
        (directory/'outside.py').symlink_to('/etc/hosts')
        for source in ('../outside.py','app/modules/card_game/outside.py','app/modules/card_game/.env'):
            with self.assertRaises(ValueError):collect(dict(self.config,sources=[source]))
        with self.assertRaises(ValueError):collect(dict(self.config,room="x';DROP"))

    def test_helper_stdin_quotes_success_and_failure(self):
        helper,result=prepare(self.config,'fixture-host',self.root/"output ' quoted")
        subprocess.run(['bash','-n',str(helper)],check=True)
        fakebin=self.root/'bin';fakebin.mkdir()
        fake=fakebin/'ssh'
        fake.write_text('#!/usr/bin/env python3\nimport sys\nassert len(sys.argv[-1]) < 2000\nassert "collect(" not in sys.argv[-1]\nexec(sys.stdin.read())\n')
        fake.chmod(0o700)
        env=dict(os.environ,PATH=str(fakebin)+os.pathsep+os.environ['PATH'])
        subprocess.run(['bash',str(helper)],env=env,check=True,stdout=subprocess.PIPE)
        self.assertEqual(json.loads(result.read_text())['revision'],54)
        self.assertEqual(result.stat().st_mode & 0o777,0o600)
        helper2,result2=prepare(self.config,'fixture-host',self.root/'failed')
        fake.write_text('#!/bin/sh\nexit 255\n')
        self.assertNotEqual(subprocess.run(['bash',str(helper2)],env=env).returncode,0)
        self.assertFalse(result2.exists())
        self.assertFalse(list(result2.parent.glob('.partial.*')))

    def test_config_is_not_executed(self):
        config=self.root/'deployment.env'
        config.write_text("NTE_DEPLOY_HOST='user@host'\nNTE_DEPLOY_PROJECT='C:/path with spaces'\nIGNORED=$(false)\n")
        self.assertEqual(deployment_values(config)['NTE_DEPLOY_PROJECT'],'C:/path with spaces')
        config.write_text('NTE_DEPLOY_HOST=$(false)\n')
        with self.assertRaises(ValueError):deployment_values(config)


if __name__ == '__main__':
    unittest.main()
