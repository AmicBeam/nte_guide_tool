"""Real Git packages and optional PowerShell activation, without a deployment server."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('deploy_app_package', ROOT / 'scripts/deploy_app_package.py')
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)
PWSH = os.environ.get('NTE_TEST_PWSH') or shutil.which('pwsh') or (shutil.which('powershell.exe') if os.name == 'nt' else None)


class DeployFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nte-deploy-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.name', 'Deploy test')
        self.git('config', 'user.email', 'deploy-test@example.invalid')
        self.write('app/__init__.py', b'original app\n')
        self.write('app/static/unchanged.bin', bytes(range(256)) * 4096)
        self.write('app/obsolete.py', b'delete this\n')
        self.write('app/rename.py', b'rename this\n')
        self.write('app/switch', b'file becomes directory\n')
        self.write('app/flatten/child.py', b'directory becomes file\n')
        self.commit()
        self.full = self.root / 'full.zip'
        self.base = self.root / 'base.json'
        self.build(self.full, 'first')
        with zipfile.ZipFile(self.full) as archive:
            self.base.write_bytes(archive.read('state.json'))

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], stderr=subprocess.STDOUT)

    def write(self, path, data):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def commit(self):
        self.git('add', '-A')
        self.git('commit', '-qm', 'Fixture changes')

    def build(self, output, release, base=None):
        return package.build_package(self.repo, output, release, base)

    def manifest(self, path):
        with zipfile.ZipFile(path) as archive:
            return json.loads(archive.read('manifest.json')), archive.namelist()

    def change_all(self):
        self.write('app/__init__.py', b'updated app\n')
        (self.repo / 'app/obsolete.py').unlink()
        self.git('mv', 'app/rename.py', 'app/Rename.py')
        (self.repo / 'app/switch').unlink()
        self.write('app/switch/child.py', b'now a directory\n')
        (self.repo / 'app/flatten/child.py').unlink()
        (self.repo / 'app/flatten').rmdir()
        self.write('app/flatten', b'now a file\n')
        self.write('app/static/中文 name.txt', b'unicode filename\n')
        self.commit()

class PackageTests(DeployFixture):
    def test_full_is_committed_export_only(self):
        self.write('app/__init__.py', b'dirty must not ship')
        self.write('app/local-untracked.txt', b'must not ship')
        output = self.root / 'dirty.zip'
        self.build(output, 'dirty')
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(archive.read('payload/app/__init__.py'), b'original app\n')
            self.assertNotIn('payload/app/local-untracked.txt', archive.namelist())

    def test_delta_add_modify_delete_rename_and_type_changes(self):
        self.change_all()
        delta = self.root / 'delta.zip'
        self.build(delta, 'second', self.base)
        manifest, names = self.manifest(delta)
        self.assertEqual(manifest['kind'], 'delta')
        self.assertEqual(set(manifest['deletes']), {'app/obsolete.py', 'app/rename.py', 'app/switch', 'app/flatten/child.py'})
        self.assertEqual(set(manifest['upserts']), {'app/__init__.py', 'app/Rename.py', 'app/switch/child.py', 'app/flatten', 'app/static/中文 name.txt'})
        self.assertNotIn('payload/app/static/unchanged.bin', names)
        self.assertLess(delta.stat().st_size, self.full.stat().st_size / 2)

    def test_deletion_only_has_no_payload(self):
        (self.repo / 'app/obsolete.py').unlink()
        self.commit()
        delta = self.root / 'delete.zip'
        self.build(delta, 'delete', self.base)
        manifest, names = self.manifest(delta)
        self.assertEqual(manifest['upserts'], [])
        self.assertEqual(manifest['deletes'], ['app/obsolete.py'])
        self.assertEqual(names, ['manifest.json', 'state.json'])

    def test_noop_and_null_base(self):
        delta = self.root / 'noop.zip'
        self.build(delta, 'noop', self.base)
        manifest, _ = self.manifest(delta)
        self.assertEqual((manifest['upserts'], manifest['deletes']), ([], []))
        self.base.write_bytes(b'null\r\n')
        summary = self.build(delta, 'bootstrap', self.base)
        self.assertEqual(summary['kind'], 'full')

    def test_baseline_hash_handles_windows_output_newlines(self):
        raw = self.base.read_bytes()
        self.base.write_bytes(raw + b'\r\n')
        delta = self.root / 'delta.zip'
        self.build(delta, 'newline', self.base)
        manifest, _ = self.manifest(delta)
        self.assertEqual(manifest['base_sha256'], hashlib.sha256(raw).hexdigest())

    def test_unsafe_baseline_and_case_collisions_rejected(self):
        state = json.loads(self.base.read_bytes())
        for name in ['app/../outside', 'app/CON.txt', 'app/f:stream', 'app/trailing.', 'app/back\\slash']:
            with self.subTest(name=name):
                changed = json.loads(json.dumps(state))
                changed['files'][1]['path'] = name
                self.base.write_bytes(package.json_bytes(changed))
                with self.assertRaises(ValueError):
                    self.build(self.root / 'bad.zip', 'bad', self.base)
        state['files'].append(dict(state['files'][0], path='app/__INIT__.py'))
        self.base.write_bytes(package.json_bytes(state))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.build(self.root / 'bad.zip', 'bad', self.base)

    def test_symlinks_are_not_shipped(self):
        if os.name == 'nt':
            self.skipTest('Creating symlinks requires Windows privileges')
        (self.repo / 'app/link').symlink_to('__init__.py')
        self.commit()
        with self.assertRaisesRegex(ValueError, 'regular files'):
            self.build(self.root / 'bad.zip', 'bad')

    def test_export_ignore_respected(self):
        self.write('.gitattributes', b'app/obsolete.py export-ignore\n')
        self.commit()
        output = self.root / 'export.zip'
        self.build(output, 'export', self.base)
        manifest, names = self.manifest(output)
        self.assertIn('app/obsolete.py', manifest['deletes'])
        self.assertNotIn('payload/app/obsolete.py', names)

    def test_backward_deployment_does_not_require_ancestry(self):
        first = self.git('rev-parse', 'HEAD').decode().strip()
        self.change_all()
        second = self.root / 'second.zip'
        self.build(second, 'second', self.base)
        with zipfile.ZipFile(second) as archive:
            self.base.write_bytes(archive.read('state.json'))
        self.git('checkout', '-q', first)
        reverted = self.root / 'backward.zip'
        self.build(reverted, 'backward', self.base)
        manifest, _ = self.manifest(reverted)
        self.assertIn('app/obsolete.py', manifest['upserts'])
        self.assertIn('app/switch/child.py', manifest['deletes'])


@unittest.skipUnless(shutil.which('bash') and shutil.which('shasum'), 'Bash deployment client required')
class ClientTests(DeployFixture):
    def setUp(self):
        super().setUp()
        scripts = self.repo / 'scripts'
        scripts.mkdir()
        for name in ('deploy_windows_app.sh', 'deploy_windows_app.ps1', 'deploy_app_package.py'):
            shutil.copy2(ROOT / 'scripts' / name, scripts / name)
        self.mockbin = self.root / 'bin'
        self.mockbin.mkdir()
        stub = '''#!/usr/bin/env python3
import base64, json, os, pathlib, shutil, sys
root = pathlib.Path(os.environ['NTE_MOCK_ROOT'])
with (root / 'calls.jsonl').open('a') as log:
    log.write(json.dumps(sys.argv) + '\\n')
if pathlib.Path(sys.argv[0]).name == 'ssh':
    control = next((arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('ControlPath=')), None)
    if '-N' in sys.argv:
        if os.environ.get('NTE_MOCK_AUTH_FAIL'):
            sys.exit(1)
        pathlib.Path(control).touch()
        sys.exit(0)
    if '-O' in sys.argv:
        if sys.argv[sys.argv.index('-O') + 1] == 'check':
            sys.exit(0 if control and pathlib.Path(control).exists() else 1)
        pathlib.Path(control).unlink(missing_ok=True)
        sys.exit(0)
    if not control or not pathlib.Path(control).exists() or 'BatchMode=yes' not in sys.argv or 'ProxyCommand=false' not in sys.argv:
        sys.exit('Expected the existing authenticated connection')
    decoded = base64.b64decode(sys.argv[-1].split()[-1]).decode('utf-16le')
    (root / 'remote-command.txt').write_text(decoded)
    if '-ReadState' in decoded:
        print((root / 'base.json').read_text())
        if os.environ.get('NTE_MOCK_DROP_AFTER_READ'):
            pathlib.Path(control).unlink()
    elif os.environ.get('NTE_MOCK_FAIL'):
        sys.exit(1)
    else:
        print('DEPLOY_SUCCESS')
else:
    control = next((arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('ControlPath=')), None)
    if not control or not pathlib.Path(control).exists() or 'BatchMode=yes' not in sys.argv or 'ProxyCommand=false' not in sys.argv:
        sys.exit('Expected the existing authenticated connection')
    for arg in sys.argv[1:-1]:
        if arg.endswith('.zip'):
            shutil.copy2(arg, root / 'captured.zip')
'''
        for name in ('ssh', 'scp'):
            target = self.mockbin / name
            target.write_text(stub)
            target.chmod(0o755)
        self.env = dict(os.environ, NTE_MOCK_ROOT=str(self.root),
                        NTE_DEPLOY_HOST='fake-host', NTE_DEPLOY_PROJECT="C:/A B/'quoted'",
                        NTE_DEPLOY_RESTART_BATCH='', PATH=str(self.mockbin) + os.pathsep + os.environ['PATH'])

    def client(self, *args):
        return subprocess.run(['bash', str(self.repo / 'scripts/deploy_windows_app.sh'), *args],
                              env=self.env, capture_output=True, text=True, timeout=30)

    def test_deploy_uses_server_delta_and_cleans_only_successful_zip(self):
        self.change_all()
        result = self.client('--deploy')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest, _ = self.manifest(self.root / 'captured.zip')
        self.assertEqual(manifest['kind'], 'delta')
        self.assertIn('app/obsolete.py', manifest['deletes'])
        self.assertFalse(list((self.repo / 'dist/deploy').glob('*.zip')))
        self.assertIn("'C:/A B/''quoted'''", (self.root / 'remote-command.txt').read_text())
        calls = [json.loads(line) for line in (self.root / 'calls.jsonl').read_text().splitlines()]
        masters = [call for call in calls if '-N' in call]
        self.assertEqual(len(masters), 1)
        controls = {arg for call in calls for arg in call if arg.startswith('ControlPath=')}
        self.assertEqual(len(controls), 1)
        self.assertEqual(len([call for call in calls if '-O' in call]), 1)
        control = Path(next(iter(controls)).split('=', 1)[1])
        self.assertFalse(control.parent.exists())
        self.assertIn("$ProgressPreference = 'SilentlyContinue';", (self.root / 'remote-command.txt').read_text())
        self.env['NTE_MOCK_FAIL'] = '1'
        result = self.client('--deploy')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(list((self.repo / 'dist/deploy').glob('*.zip'))), 1)
        calls = [json.loads(line) for line in (self.root / 'calls.jsonl').read_text().splitlines()]
        for call in calls:
            for arg in call:
                if arg.startswith('ControlPath='):
                    self.assertFalse(Path(arg.split('=', 1)[1]).parent.exists())

    def test_authentication_failure_never_uploads_or_reprompts(self):
        self.env['NTE_MOCK_AUTH_FAIL'] = '1'
        result = self.client('--deploy')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Could not establish the deployment SSH session. Nothing was uploaded.', result.stderr)
        calls = [json.loads(line) for line in (self.root / 'calls.jsonl').read_text().splitlines()]
        self.assertEqual(len(calls), 1)
        self.assertIn('-N', calls[0])
        self.assertIn('ConnectTimeout=15', calls[0])
        self.assertIn('ConnectionAttempts=1', calls[0])
        control = next(arg.split('=', 1)[1] for arg in calls[0] if arg.startswith('ControlPath='))
        self.assertFalse(Path(control).parent.exists())

    def test_lost_connection_fails_without_second_authentication(self):
        self.env['NTE_MOCK_DROP_AFTER_READ'] = '1'
        result = self.client('--deploy')
        self.assertNotEqual(result.returncode, 0)
        calls = [json.loads(line) for line in (self.root / 'calls.jsonl').read_text().splitlines()]
        self.assertEqual(len([call for call in calls if '-N' in call]), 1)
        self.assertFalse((self.root / 'captured.zip').exists())
        control = next(arg.split('=', 1)[1] for call in calls for arg in call if arg.startswith('ControlPath='))
        self.assertFalse(Path(control).parent.exists())

    def test_reuses_supplied_master_without_closing_it(self):
        control = self.root / 'existing.sock'
        control.touch()
        self.env['NTE_DEPLOY_SSH_CONTROL_PATH'] = str(control)
        result = self.client('--deploy')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(control.exists())
        calls = [json.loads(line) for line in (self.root / 'calls.jsonl').read_text().splitlines()]
        self.assertEqual([call[call.index('-O') + 1] for call in calls if '-O' in call], ['check'])
        self.assertFalse(any('-N' in call for call in calls))
        self.assertTrue(all(f'ControlPath={control}' in call for call in calls))

    def test_unavailable_supplied_master_opens_and_cleans_a_private_master(self):
        control = self.root / 'missing.sock'
        self.env['NTE_DEPLOY_SSH_CONTROL_PATH'] = str(control)
        result = self.client('--deploy')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Opening a new connection.', result.stderr)
        calls = [json.loads(line) for line in (self.root / 'calls.jsonl').read_text().splitlines()]
        self.assertIn('check', calls[0])
        self.assertIn(f'ControlPath={control}', calls[0])
        masters = [call for call in calls if '-N' in call]
        self.assertEqual(len(masters), 1)
        new_option = next(arg for arg in masters[0] if arg.startswith('ControlPath='))
        self.assertNotEqual(new_option, f'ControlPath={control}')
        self.assertTrue(all(new_option in call for call in calls[1:]))
        self.assertIn('exit', calls[-1])
        self.assertFalse(Path(new_option.split('=', 1)[1]).parent.exists())
        self.assertFalse(control.exists())

    def test_failed_fallback_authentication_stops_before_upload(self):
        self.env['NTE_DEPLOY_SSH_CONTROL_PATH'] = str(self.root / 'missing.sock')
        self.env['NTE_MOCK_AUTH_FAIL'] = '1'
        result = self.client('--deploy')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Could not establish the deployment SSH session. Nothing was uploaded.', result.stderr)
        calls = [json.loads(line) for line in (self.root / 'calls.jsonl').read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        self.assertIn('check', calls[0])
        self.assertIn('-N', calls[1])

    def test_check_and_offline_prepare_never_contact_server(self):
        self.assertEqual(self.client('--check').returncode, 0)
        result = self.client('--prepare', '--base-state', str(self.base))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.root / 'calls.jsonl').exists())
        archives = list((self.repo / 'dist/deploy').glob('*.zip'))
        self.assertEqual(len(archives), 1)
        self.assertEqual(self.manifest(archives[0])[0]['upserts'], [])

    def test_first_deploy_full_and_dirty_source_guard(self):
        self.base.write_bytes(b'null')
        result = self.client('--deploy')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.manifest(self.root / 'captured.zip')[0]['kind'], 'full')
        calls = (self.root / 'calls.jsonl').read_text()
        self.write('app/__init__.py', b'dirty app')
        result = self.client('--deploy')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.root / 'calls.jsonl').read_text(), calls)


@unittest.skipUnless(PWSH, 'Set NTE_TEST_PWSH to run real PowerShell activation tests')
class ActivationTests(DeployFixture):
    def setUp(self):
        super().setUp()
        self.project = self.root / 'server'
        (self.project / 'app').mkdir(parents=True)
        (self.project / 'app/old-untracked.py').write_bytes(b'old untracked file')
        self.source = (ROOT / 'scripts/deploy_windows_app.ps1').read_text()
        if os.name != 'nt':
            # Only adapt Windows input-path normalization. Package validation,
            # staging, directory switching, state commit and rollback run unchanged.
            begin = self.source.index('function Resolve-WindowsPath {')
            end = self.source.index('$ProjectRoot = Resolve-WindowsPath')
            self.source = self.source[:begin] + 'function Resolve-WindowsPath { param([string]$Path) return [IO.Path]::GetFullPath($Path) }\n\n' + self.source[end:]

    def activate(self, archive, release, source=None, service=False):
        helper = self.root / f'helper-{release}.ps1'
        helper.write_text(source or self.source)
        args = [PWSH, '-NoProfile', '-NonInteractive', '-File', str(helper),
                '-ProjectRoot', str(self.project), '-ArchivePath', str(archive), '-ReleaseId', release,
                '-ArchiveSha256', hashlib.sha256(archive.read_bytes()).hexdigest()]
        if service:
            args.extend(['-ServiceName', 'mock-service'])
        return subprocess.run(args, capture_output=True, text=True, timeout=45)

    def bootstrap(self):
        result = self.activate(self.full, 'first')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertFalse((self.project / 'app/old-untracked.py').exists())
        self.base.write_bytes((self.project / '.nte-deploy-app.json').read_bytes())

    def assert_target(self):
        state = json.loads((self.project / '.nte-deploy-app.json').read_bytes())
        for entry in state['files']:
            self.assertEqual(hashlib.sha256((self.project / entry['path']).read_bytes()).hexdigest(), entry['sha256'])
        actual = {p.relative_to(self.project).as_posix() for p in (self.project / 'app').rglob('*') if p.is_file()}
        self.assertEqual(actual, {entry['path'] for entry in state['files']})

    def test_real_full_and_delta_activation(self):
        self.bootstrap()
        self.change_all()
        delta = self.root / 'delta.zip'
        self.build(delta, 'second', self.base)
        result = self.activate(delta, 'second')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assert_target()
        self.assertFalse((self.project / 'app/obsolete.py').exists())
        self.assertTrue((self.project / '.nte-deploy/second/backup-app/obsolete.py').exists())

    def test_real_delete_only_then_noop_without_restart(self):
        self.bootstrap()
        (self.repo / 'app/obsolete.py').unlink()
        self.commit()
        delta = self.root / 'delete.zip'
        self.build(delta, 'delete', self.base)
        result = self.activate(delta, 'delete')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assert_target()
        self.base.write_bytes((self.project / '.nte-deploy-app.json').read_bytes())
        noop = self.root / 'noop.zip'
        self.build(noop, 'noop', self.base)
        source = self.source.replace('$DeployLock =', "function Stop-Service { throw 'No-op must not restart' }\n$DeployLock =", 1)
        result = self.activate(noop, 'noop', source, service=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn('no_changes=True', result.stdout)

    def test_live_drift_and_stale_base_fail_before_activation(self):
        self.bootstrap()
        self.change_all()
        delta = self.root / 'delta.zip'
        self.build(delta, 'second', self.base)
        state_before = (self.project / '.nte-deploy-app.json').read_bytes()
        (self.project / 'app/obsolete.py').write_bytes(b'manual server modification')
        result = self.activate(delta, 'second')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('differs from deployment manifest', result.stderr)
        self.assertEqual((self.project / '.nte-deploy-app.json').read_bytes(), state_before)
        state = json.loads(state_before)
        state['release_id'] = 'different-base'
        (self.project / '.nte-deploy-app.json').write_bytes(package.json_bytes(state))
        result = self.activate(delta, 'stale')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('identity mismatch', result.stderr)  # package release differs
        # Keep release matching and use a fresh work directory for the actual base check.
        with zipfile.ZipFile(delta) as z:
            entries = {n: z.read(n) for n in z.namelist()}
        target = json.loads(entries['state.json']); target['release_id'] = 'stale-base'
        entries['state.json'] = package.json_bytes(target)
        with zipfile.ZipFile(delta, 'w') as z:
            for n, data in entries.items(): z.writestr(n, data)
        result = self.activate(delta, 'stale-base')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('base changed', result.stderr)
        full = self.root / 'restore.zip'
        self.build(full, 'restore')
        result = self.activate(full, 'restore')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assert_target()

    def test_restart_failure_restores_app_and_baseline(self):
        self.bootstrap()
        self.change_all()
        delta = self.root / 'delta.zip'
        self.build(delta, 'rollback', self.base)
        before = (self.project / '.nte-deploy-app.json').read_bytes()
        mocks = """
function Stop-Service { param($Name, $ErrorAction) }
function Start-Service { param($Name, $ErrorAction) if (Test-Path (Join-Path $WorkRoot 'backup-app')) { throw 'Injected restart failure' } }
function Get-Service { param($Name) $s = New-Object PSObject; $s | Add-Member NoteProperty Status ([ServiceProcess.ServiceControllerStatus]::Running); $s | Add-Member ScriptMethod WaitForStatus { param($Status, $Timeout) }; return $s }
"""
        source = self.source.replace('$DeployLock =', mocks + '\n$DeployLock =', 1)
        result = self.activate(delta, 'rollback', source, service=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Injected restart failure', result.stderr)
        self.assertEqual((self.project / '.nte-deploy-app.json').read_bytes(), before)
        self.assertEqual((self.project / 'app/__init__.py').read_bytes(), b'original app\n')
        self.assertTrue((self.project / 'app/obsolete.py').exists())

    def test_corrupted_payload_and_unsafe_zip_rejected(self):
        self.bootstrap()
        self.change_all()
        delta = self.root / 'delta.zip'
        self.build(delta, 'corrupt', self.base)
        with zipfile.ZipFile(delta) as z:
            entries = {n: z.read(n) for n in z.namelist()}
        entries['payload/app/__init__.py'] = b'corrupted content'
        with zipfile.ZipFile(delta, 'w') as z:
            for n, data in entries.items(): z.writestr(n, data)
        result = self.activate(delta, 'corrupt')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('differs from deployment manifest', result.stderr)
        with zipfile.ZipFile(delta, 'a') as z:
            z.writestr('payload/app/../../escape.txt', b'escape')
        result = self.activate(delta, 'escape')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unsafe Windows path', result.stderr)
        self.assertFalse((self.project / 'escape.txt').exists())

    def test_missing_deletion_operation_rejected(self):
        self.bootstrap()
        (self.repo / 'app/obsolete.py').unlink()
        self.commit()
        delta = self.root / 'delete.zip'
        self.build(delta, 'missing-delete', self.base)
        with zipfile.ZipFile(delta) as z:
            entries = {n: z.read(n) for n in z.namelist()}
        manifest = json.loads(entries['manifest.json']); manifest['deletes'] = []
        entries['manifest.json'] = package.json_bytes(manifest)
        with zipfile.ZipFile(delta, 'w') as z:
            for n, data in entries.items(): z.writestr(n, data)
        result = self.activate(delta, 'missing-delete')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('operations do not match', result.stderr)
        self.assertTrue((self.project / 'app/obsolete.py').exists())

    def test_waitress_health_failure_rolls_back_and_restarts_previous_app(self):
        self.bootstrap()
        self.change_all()
        delta = self.root / 'health.zip'
        self.build(delta, 'health', self.base)
        batch = self.project / 'mock.bat'; batch.write_text('mock')
        mocks = """
function Stop-PortListener { param($Port) Add-Content (Join-Path $ProjectRoot 'stop-log') 'stop'; return 1 }
function Start-WaitressBatch { param($BatchPath, $WorkingDirectory) Add-Content (Join-Path $ProjectRoot 'start-log') 'start' }
function Get-ListeningProcessIds { param($Port) return @(1) }
function Wait-ForPort { param($Port, $TimeoutSeconds) return -not (Test-Path (Join-Path $WorkRoot 'backup-app')) }
"""
        source = self.source.replace('$DeployLock =', mocks + '\n$DeployLock =', 1)
        # Supply the existing batch to the actual activation parameters.
        source = source.replace('$ProjectRoot = Resolve-WindowsPath',
                                "$RestartBatch = Join-Path $ProjectRoot 'mock.bat'\n$ProjectRoot = Resolve-WindowsPath", 1)
        result = self.activate(delta, 'health', source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('startup verification failed', result.stderr)
        self.assertEqual((self.project / 'start-log').read_text().splitlines(), ['start', 'start'])
        self.assertEqual((self.project / 'stop-log').read_text().splitlines(), ['stop', 'stop'])
        self.assertEqual((self.project / 'app/__init__.py').read_bytes(), b'original app\n')

    def test_state_commit_failure_rolls_back(self):
        self.bootstrap()
        self.change_all()
        delta = self.root / 'state-fail.zip'
        self.build(delta, 'state-fail', self.base)
        before = (self.project / '.nte-deploy-app.json').read_bytes()
        source = self.source.replace('# Publish the new baseline only after successful activation/health checks.',
                                    '[IO.File]::Delete($NewStatePath)', 1)
        result = self.activate(delta, 'state-fail', source)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.project / '.nte-deploy-app.json').read_bytes(), before)
        self.assertEqual((self.project / 'app/__init__.py').read_bytes(), b'original app\n')

    def test_read_state_and_concurrent_lock(self):
        self.bootstrap()
        helper = self.root / 'read-state.ps1'; helper.write_text(self.source)
        args = [PWSH, '-NoProfile', '-NonInteractive', '-File', str(helper), '-ProjectRoot', str(self.project), '-ReadState']
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

        self.assertEqual(json.loads(result.stdout), json.loads(self.base.read_bytes()))
        locker = self.root / 'lock.ps1'
        locker.write_text("""param($LockPath, $ReadyPath)
$l = [IO.File]::Open($LockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try { [IO.File]::WriteAllText($ReadyPath, 'ready'); $null = [Console]::ReadLine() } finally { $l.Dispose() }
""")
        ready = self.root / 'ready'
        with subprocess.Popen([PWSH, '-NoProfile', '-NonInteractive', '-File', str(locker),
                               '-LockPath', str(self.project / '.nte-deploy-app.lock'), '-ReadyPath', str(ready)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
            try:
                deadline = time.monotonic() + 20
                while not ready.exists() and time.monotonic() < deadline:
                    if proc.poll() is not None: break
                    time.sleep(0.05)
                self.assertTrue(ready.exists())
                result = subprocess.run(args, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
            finally:
                proc.communicate('\n', timeout=10)
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


    def read_server_state(self):
        helper = self.root / 'recover.ps1'; helper.write_text(self.source)
        return subprocess.run([PWSH, '-NoProfile', '-NonInteractive', '-File', str(helper),
                               '-ProjectRoot', str(self.project), '-ReadState'],
                              capture_output=True, text=True, timeout=45)

    def test_interrupted_moves_restore_old_app_before_next_state_read(self):
        for point in ('after-old-move', 'before-state-commit'):
            with self.subTest(point=point):
                # Each interruption has an independent app + original baseline.
                archive = self.root / (point + '-full.zip')
                self.build(archive, point + '-base')
                result = self.activate(archive, point + '-base')
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                before = (self.project / '.nte-deploy-app.json').read_bytes()
                self.base.write_bytes(before)
                self.write('app/__init__.py', point.encode())
                self.commit()
                delta = self.root / (point + '.zip')
                self.build(delta, point, self.base)
                if point == 'after-old-move':
                    source = self.source.replace('$OldAppMoved = $true', '$OldAppMoved = $true\n[Environment]::Exit(91)', 1)
                else:
                    source = self.source.replace('# Publish the new baseline only after successful activation/health checks.', '[Environment]::Exit(91)', 1)
                result = self.activate(delta, point, source)
                self.assertEqual(result.returncode, 91, result.stderr + result.stdout)
                self.assertTrue((self.project / '.nte-deploy-app.pending.json').exists())
                recovered = self.read_server_state()
                self.assertEqual(recovered.returncode, 0, recovered.stderr + recovered.stdout)
                self.assertEqual(json.loads(recovered.stdout), json.loads(before))
                self.assertFalse((self.project / '.nte-deploy-app.pending.json').exists())
                self.assert_target()

    def test_interruption_after_state_commit_keeps_new_app(self):
        self.bootstrap()
        self.change_all()
        delta = self.root / 'committed-crash.zip'
        self.build(delta, 'committed-crash', self.base)
        source = self.source.replace('# Cleanup failures must not turn a committed deployment into a reported rollback.',
                                    '[Environment]::Exit(92)', 1)
        result = self.activate(delta, 'committed-crash', source)
        self.assertEqual(result.returncode, 92, result.stderr + result.stdout)
        recovered = self.read_server_state()
        self.assertEqual(recovered.returncode, 0, recovered.stderr + recovered.stdout)
        self.assertEqual(json.loads(recovered.stdout)['release_id'], 'committed-crash')
        self.assert_target()
        self.assertFalse((self.project / '.nte-deploy-app.pending.json').exists())

    def test_rollback_failure_preserves_journal_and_recovers_next_time(self):
        self.bootstrap()
        self.change_all()
        delta = self.root / 'rollback-fail.zip'
        self.build(delta, 'rollback-fail', self.base)
        source = self.source.replace('# Publish the new baseline only after successful activation/health checks.',
                                    '[IO.File]::Delete($NewStatePath)', 1)
        source = source.replace('$DeployLock =', """
function Move-Item {
    param($LiteralPath, $Destination)
    if ($LiteralPath -eq $BackupApp) { throw 'Injected rollback move failure' }
    Microsoft.PowerShell.Management\\Move-Item -LiteralPath $LiteralPath -Destination $Destination
}
$DeployLock =""", 1)
        result = self.activate(delta, 'rollback-fail', source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Rollback incomplete', result.stderr)
        self.assertTrue((self.project / '.nte-deploy-app.pending.json').exists())
        recovered = self.read_server_state()
        self.assertEqual(recovered.returncode, 0, recovered.stderr + recovered.stdout)
        self.assertEqual(json.loads(recovered.stdout)['release_id'], 'first')
        self.assert_target()


if __name__ == '__main__':
    unittest.main()
