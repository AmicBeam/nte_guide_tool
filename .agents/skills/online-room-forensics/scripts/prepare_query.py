"""Generate an interactive SSH helper locally; never connects by itself."""
import argparse
import base64
import json
import re
import shlex
import tempfile
from datetime import date
from pathlib import Path
from collect_room import validate


def deployment_values(path):
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.match(r'^\s*(NTE_DEPLOY_HOST|NTE_DEPLOY_PROJECT)\s*=\s*(.*)$', line)
            if m:
                words = shlex.split(m[2], comments=True)
                if len(words) != 1 or any(c in words[0] for c in ('$', '`', '\n', '\r')):
                    raise ValueError('Use literal deployment values or explicit --host/--project')
                values[m[1]] = words[0]
    return values


def prepare(c, host, output):
    validate(c)
    if not host or host.startswith('-') or any(ch.isspace() for ch in host):
        raise ValueError('Expected SSH alias or user@host without whitespace')
    if any(ch in c['project'] for ch in ('\n', '\r')):
        raise ValueError('Invalid project path')
    project = c['project'].replace("'", "''")
    ps = ("$ErrorActionPreference='Stop'; Set-Location '" + project + "'; "
          "$p=@('.venv/Scripts/python.exe','venv/Scripts/python.exe') | "
          "Where-Object {Test-Path $_} | Select-Object -First 1; "
          "if(-not $p){$p=(Get-Command python -ErrorAction Stop).Source}; "
          "& $p -; exit $LASTEXITCODE")
    command = 'powershell.exe -NoProfile -NonInteractive -EncodedCommand ' + base64.b64encode(ps.encode('utf-16le')).decode()
    if len(command) > 7000:
        raise ValueError('Bootstrap too long for Windows command line')
    output = Path(output).resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    payload, helper, result = output/'query.py', output/'query.sh', output/'evidence.json'
    source = Path(__file__).with_name('collect_room.py').read_text()
    source += '\nprint(json.dumps(collect(json.loads(' + repr(json.dumps(c)) + ')), ensure_ascii=True))\n'
    payload.write_text(source)
    payload.chmod(0o600)
    script = '''#!/bin/bash
set -euo pipefail
umask 077
output=%s
if [[ -e "$output" ]]; then echo "Result already exists: $output" >&2; exit 1; fi
partial=$(mktemp %s)
trap 'rm -f "$partial"' EXIT
ssh -o ConnectTimeout=30 -o ConnectionAttempts=1 %s %s < %s > "$partial"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["room_code"]==sys.argv[2]' "$partial" %s
mv "$partial" "$output"
printf 'Evidence saved: %%s\\n' "$output"
''' % (shlex.quote(str(result)), shlex.quote(str(output/'.partial.XXXXXX')),
       shlex.quote(host), shlex.quote(command), shlex.quote(str(payload)), shlex.quote(c['room']))
    helper.write_text(script)
    helper.chmod(0o700)
    return helper, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('room')
    p.add_argument('--date', default=date.today().isoformat(), help='Server log date YYYY-MM-DD')
    p.add_argument('--host')
    p.add_argument('--project')
    p.add_argument('--database', help='Explicit server DB path; does not inspect .env')
    p.add_argument('--log-dir')
    p.add_argument('--tail-lines', type=int, default=5000)
    p.add_argument('--errors', action='store_true', help='Include bounded unattributed V2 error blocks')
    p.add_argument('--source', action='append', default=[])
    p.add_argument('--output-dir', help='New directory outside repository')
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[4]
    values = deployment_values(repo/'scripts/deploy_windows_app.local.env')
    host, project = a.host or values.get('NTE_DEPLOY_HOST'), a.project or values.get('NTE_DEPLOY_PROJECT')
    if not host or not project:
        p.error('Pass --host/--project or configure deployment target locally')
    c = dict(room=a.room.upper(), date=a.date, project=project, database=a.database,
             log_dir=a.log_dir, tail_lines=a.tail_lines, errors=a.errors, sources=a.source)
    output = Path(a.output_dir).resolve() if a.output_dir else Path(tempfile.mkdtemp(prefix='nte-forensics-'))/a.room.upper()
    if output == repo or repo in output.parents:
        p.error('Evidence must be stored outside repository')
    helper, result = prepare(c, host, output)
    print('No network connection made.')
    print('Run in your terminal: bash ' + shlex.quote(str(helper)))
    print('Then read: ' + str(result))


if __name__ == '__main__':
    main()
