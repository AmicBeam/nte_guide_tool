"""Task-scoped source snapshot for local diagnostics and a future CUDA preflight.

No environment, database, logs, credentials, unrelated untracked files or
repository-external symlink targets are copied. Not a deployment package.
"""
from hashlib import sha256
from pathlib import Path
import json
import subprocess


def _allowed_source(name):
    path=Path(name);parts=path.parts
    if (not name or path.is_absolute() or '..' in parts or '.env' in name.lower()
            or name.startswith(('.git/','artifacts/','logs/'))
            or any(p.lower() in ('credentials','credential_store','token_cache','auth_cache',
                                 'browser_profile','browser_profiles') for p in parts)
            or path.name.lower() in ('credentials.json','auth.json','tokens.json','cookies.json','secrets.json')):
        return False
    return (name.startswith('app/') and path.suffix=='.py'
            or name.startswith(('app/modules/card_game/content/','app/modules/card_game/engine/ai/models/')) and path.suffix=='.json'
            or name.startswith(('scripts/','tests/')) and path.suffix=='.py'
            or name.startswith('docs/duel-v2') and path.suffix=='.md'
            or name=='docs/everness-item-chain-card-design.md'
            or name in ('AGENTS.md','README.md','requirements.txt','requirements-rl.txt','requirements-ai.txt'))


def snapshot_revision(root,output,revision,extra=()):
    """Freeze allowlisted committed blobs while concurrent working edits continue.

    Explicit extras are new, untracked source inputs only. They cannot replace
    a committed rule file. Numeric weights require a separate explicit manifest.
    """
    root=Path(root).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('A new source snapshot directory is required')
    if not isinstance(revision,str) or not revision:raise ValueError('Explicit Git revision required')
    def git(*args,input=None):
        return subprocess.run(['git',*args],cwd=root,input=input,capture_output=True,check=True).stdout
    commit=git('rev-parse','--verify','--end-of-options',revision+'^{commit}').decode().strip()
    entries=git('ls-tree','-rz','--full-tree',commit).split(b'\0')
    selected={};tracked=set()
    for entry in entries:
        if not entry:continue
        header,name=entry.split(b'\t',1);mode,kind,oid=header.decode().split(' ')
        name=name.decode('utf-8');tracked.add(name)
        if mode in ('100644','100755') and kind=='blob' and _allowed_source(name):selected[name]=oid
    extras={}
    for name in extra:
        if not isinstance(name,str) or not _allowed_source(name) or name in tracked:
            raise ValueError('Extra must be new allowlisted source: '+str(name))
        source=root/name
        if not source.is_file() or source.is_symlink() or root not in source.resolve().parents:
            raise ValueError('Invalid extra source: '+name)
        extras[name]=source.read_bytes()
    names=sorted(selected)
    raw=git('cat-file','--batch',input=''.join(selected[n]+'\n' for n in names).encode())
    offset=0;blobs={}
    for name in names:
        end=raw.index(b'\n',offset);oid,kind,size=raw[offset:end].decode().split(' ');size=int(size)
        if oid!=selected[name] or kind!='blob':raise ValueError('Git blob identity mismatch')
        start=end+1;stop=start+size
        if raw[stop:stop+1]!=b'\n':raise ValueError('Truncated Git blob')
        blobs[name]=raw[start:stop];offset=stop+1
    if offset!=len(raw):raise ValueError('Unexpected trailing Git blob data')
    blobs.update(extras);output.mkdir(parents=True)
    manifest={}
    for name,data in sorted(blobs.items()):
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        manifest[name]=dict(bytes=len(data),sha256=sha256(data).hexdigest())
    (output/'source-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (output/'source-revision.json').write_text(json.dumps(dict(commit=commit,
        profile='git-object-allowlist-v1',extra_sources=sorted(extras)),indent=2),encoding='utf-8')
    return manifest


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
