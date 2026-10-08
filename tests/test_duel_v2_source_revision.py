"""Committed source snapshots must not absorb simultaneous working edits."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from app.modules.card_game.rl.offline_sources import snapshot_revision


class RevisionSnapshotTest(unittest.TestCase):
    def test_pinned_blobs_and_explicit_new_source_exclude_private_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'repo';root.mkdir()
            def git(*args):
                return subprocess.run(['git',*args],cwd=root,check=True,capture_output=True).stdout
            git('init','-q')
            (root/'app').mkdir();(root/'scripts').mkdir()
            (root/'app/rules.py').write_text('current = 1\n')
            (root/'app/.env.py').write_text('excluded fixture')
            (root/'app/external.py').symlink_to('/nonexistent/source')
            git('add','app');git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid',
                                  'commit','-qm','source fixture')
            commit=git('rev-parse','HEAD').decode().strip()
            (root/'app/rules.py').write_text('uncommitted = 2\n')
            (root/'scripts/new_check.py').write_text('new_check = True\n')
            out=Path(temp)/'snapshot'
            manifest=snapshot_revision(root,out,commit,extra=['scripts/new_check.py'])
            self.assertEqual((out/'app/rules.py').read_text(),'current = 1\n')
            self.assertEqual((root/'app/rules.py').read_text(),'uncommitted = 2\n')
            self.assertEqual(set(manifest),{'app/rules.py','scripts/new_check.py'})
            self.assertFalse((out/'app/.env.py').exists())
            self.assertFalse((out/'app/external.py').exists())
            with self.assertRaises(ValueError):snapshot_revision(root,out,commit)
            for name in ('app/rules.py','../outside.py','app/.env.py','app/external.py'):
                with self.subTest(name=name),self.assertRaises(ValueError):
                    snapshot_revision(root,Path(temp)/'invalid',commit,extra=[name])


if __name__=='__main__':unittest.main()


class FrozenSourceSnapshotTest(unittest.TestCase):
    def test_verified_manifest_owns_a_standalone_copy_without_git(self):
        from app.modules.card_game.rl.offline_sources import snapshot
        from hashlib import sha256
        import json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'isolated';root.mkdir();(root/'scripts').mkdir()
            data=b'print("task source")\n';(root/'scripts/check.py').write_bytes(data)
            declared={'scripts/check.py':dict(bytes=len(data),sha256=sha256(data).hexdigest())}
            (root/'source-manifest.json').write_text(json.dumps(declared))
            out=Path(temp)/'nested/output'
            self.assertEqual(snapshot(root,out),declared)
            self.assertEqual((out/'scripts/check.py').read_bytes(),data)
            (root/'scripts/check.py').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Frozen source changed'):snapshot(root,Path(temp)/'changed')

    def test_empty_git_source_still_creates_its_manifest_directory(self):
        from app.modules.card_game.rl.offline_sources import snapshot
        from unittest.mock import patch
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'empty/output'
            with patch('app.modules.card_game.rl.offline_sources.subprocess.run',return_value=subprocess.CompletedProcess([],0,stdout=b'')):
                self.assertEqual(snapshot(root,out),{})
            self.assertTrue((out/'source-manifest.json').is_file())
