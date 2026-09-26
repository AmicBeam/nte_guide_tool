"""Task-scoped source snapshot for local diagnostics and a future CUDA preflight.

No environment, database, logs, credentials, unrelated untracked files or
repository-external symlink targets are copied. Not a deployment package.
"""
from hashlib import sha256
from pathlib import Path
import json
import subprocess


def snapshot(root, output, extra=()):
    root = Path(root).resolve(); output = Path(output)
    tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=root, check=True, capture_output=True).stdout.decode().split('\0')
    chosen = []
    for name in sorted(set(tracked) | set(extra)):
        parts = Path(name).parts
        if (not name or '.env' in name.lower() or name.startswith(('.git/', 'artifacts/', 'logs/'))
                or any(p.lower() in ('credentials', 'credential_store', 'token_cache', 'auth_cache',
                                     'browser_profile', 'browser_profiles') for p in parts)
                or Path(name).name.lower() in ('credentials.json', 'auth.json', 'tokens.json', 'cookies.json', 'secrets.json')):
            continue
        path = root / name
        is_source = (name.startswith('app/') and path.suffix == '.py'
                     or name.startswith(('app/modules/card_game/content/', 'app/modules/card_game/engine/ai/models/')) and path.suffix == '.json'
                     or name.startswith(('scripts/', 'tests/')) and path.suffix == '.py'
                     or name.startswith('docs/duel-v2') and path.suffix == '.md'
                     or name == 'docs/everness-item-chain-card-design.md'
                     or name in ('AGENTS.md', 'README.md', 'requirements.txt', 'requirements-rl.txt', 'requirements-ai.txt'))
        if not is_source or not path.is_file() or path.is_symlink() or root not in path.resolve().parents:
            continue
        chosen.append(name)
    manifest = {}
    for name in chosen:
        data = (root / name).read_bytes(); target = output / name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        manifest[name] = dict(sha256=sha256(data).hexdigest(), bytes=len(data))
    (output / 'source-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
