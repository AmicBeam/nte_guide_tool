"""Portable report delivery; private per-game logs stay at the run location."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile


def package_report(root, destination):
    root=Path(root).resolve();destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    status=json.loads((root/'status.json').read_text(encoding='utf-8'))
    if status.get('phase')!='complete':raise ValueError('Cannot package an incomplete pipeline as final')
    decisions=json.loads((root/'construction-decisions.json').read_text(encoding='utf-8'))
    if set(decisions)!={'starter','weave-rush'}:raise ValueError('Expected both presets')
    for key,choice in decisions.items():
        selected=choice['selection']
        if selected not in ('original','candidate'):raise ValueError('Invalid construction selection')
        final=root/'final'/key;final.mkdir(parents=True,exist_ok=True)
        source=root/key/('base-export' if selected=='original' else 'adapted-export')
        deck=root/key/('original-deck.json' if selected=='original' else 'cards/candidate.json')
        for incoming,outgoing in [(source/(key+ext),final/(key+ext)) for ext in ('.npz','.json')]+[(deck,final/'deck.json')]:
            if not incoming.resolve().is_relative_to(root) or incoming.is_symlink():raise ValueError('External file in report')
            shutil.copyfile(incoming,outgoing)
        (final/'selection.json').write_text(json.dumps(choice,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    destination.parent.mkdir(parents=True,exist_ok=True)
    manifest={}
    with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):continue
            relative=path.relative_to(root)
            if 'evaluations' in relative.parts and ('raw' in relative.parts or 'public' in relative.parts):continue
            if path.suffix not in ('.json','.jsonl','.csv','.md','.pt','.npz'):continue
            data=path.read_bytes();name=relative.as_posix()
            manifest[name]=hashlib.sha256(data).hexdigest();archive.writestr(name,data)
        source_manifest=Path(__file__).resolve().parents[4]/'source-manifest.json'
        if source_manifest.is_file():
            data=source_manifest.read_bytes();manifest['source-manifest.json']=hashlib.sha256(data).hexdigest()
            archive.writestr('source-manifest.json',data)
        for path in (Path(__file__),Path(__file__).with_name('full_report.py')):
            data=path.read_bytes();name='_report_source/'+path.name
            manifest[name]=hashlib.sha256(data).hexdigest();archive.writestr(name,data)
        archive.writestr('delivery-manifest.json',json.dumps({'files':manifest,'private_raw_records_in_bundle':False,
            'private_raw_records_location':str(root/'evaluations'),
            'note':'Reports, model checkpoints/exports, candidates, and one public sample replay; source-manifest identifies initial training snapshot, _report_source contains delivery-time renderers'},ensure_ascii=False,indent=2)+'\n')
    return {'archive':str(destination),'files':len(manifest),'bytes':destination.stat().st_size,
            'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
