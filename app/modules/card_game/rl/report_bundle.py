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
    chosen_path=root/'selection/selected.json'
    chosen=json.loads(chosen_path.read_text(encoding='utf-8')) if chosen_path.is_file() else None
    if chosen is not None and (not chosen.get('complete') or set(chosen.get('choices',{}))!=set(decisions)):
        raise ValueError('Final held-out selection is incomplete')
    for key,choice in decisions.items():
        selected=choice['selection']
        if selected not in ('original','candidate'):raise ValueError('Invalid construction selection')
        final=root/'final'/key;final.mkdir(parents=True,exist_ok=True)
        source=root/key/('base-export' if selected=='original' else 'adapted-export')
        deck=root/key/('original-deck.json' if selected=='original' else 'cards/candidate.json')
        if selected=='candidate' and (root/key/'selected-deck.json').is_file():
            deck=root/key/'selected-deck.json'
        selected_build=None
        if not chosen and (root/'serving-models'/f'{key}.npz').is_file():
            from .capability import fixed_team_serving_deck,serving_deck_hash
            approved_dir=root/'serving-models'
            approved=json.loads((approved_dir/f'{key}.json').read_text(encoding='utf-8'))
            selected_build=fixed_team_serving_deck(json.loads(deck.read_text(encoding='utf-8')),key)
            digest=hashlib.sha256((source/f'{key}.npz').read_bytes()).hexdigest()
            if (approved.get('validation',{}).get('complete') is not True
                    or approved.get('sha256')!=digest
                    or hashlib.sha256((approved_dir/f'{key}.npz').read_bytes()).hexdigest()!=digest
                    or approved.get('validation',{}).get('model_sha256')!=digest
                    or approved.get('validation',{}).get('serving_build_sha256')!=serving_deck_hash(selected_build)
                    or serving_deck_hash(approved.get('serving_build') or {})!=serving_deck_hash(selected_build)):
                raise ValueError('Approved pipeline model/build mismatch')
            source=approved_dir
        if chosen:
            from .capability import fixed_team_serving_deck,serving_deck_hash
            item=chosen['choices'][key]
            if item['model'] not in ('base-export','adapted-export'):raise ValueError('Unknown selected model source')
            source=root/key/item['model']
            selected_build=fixed_team_serving_deck(item['build'],key)
            if (root/'serving-models'/f'{key}.npz').is_file():
                source=root/'serving-models'
                approved=json.loads((source/f'{key}.json').read_text(encoding='utf-8'))
                if (approved.get('validation',{}).get('serving_build_sha256')!=serving_deck_hash(selected_build)
                        or approved.get('validation',{}).get('complete') is not True):
                    raise ValueError('Approved serving build mismatch')
                if approved.get('sha256')!=item['model_sha256'] or approved.get('validation',{}).get('model_sha256')!=item['model_sha256']:
                    raise ValueError('Approved model identity mismatch')
                selected_build=fixed_team_serving_deck(approved['serving_build'],key)
            if hashlib.sha256((source/f'{key}.npz').read_bytes()).hexdigest()!=item['model_sha256']:
                raise ValueError('Selected model checksum mismatch')
            choice={'selection':item['id'],'model_sha256':item['model_sha256'],'source':str(source.relative_to(root)),
                    'reason':'Independent shortlisted screening followed by held-out validation'}
        copies=[(source/(key+ext),final/(key+ext)) for ext in ('.npz','.json')]
        if selected_build is None:copies.append((deck,final/'deck.json'))
        else:(final/'deck.json').write_text(json.dumps(selected_build,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        for incoming,outgoing in copies:
            if not incoming.resolve().is_relative_to(root) or incoming.is_symlink():raise ValueError('External file in report')
            shutil.copyfile(incoming,outgoing)
        (final/'selection.json').write_text(json.dumps(choice,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    destination.parent.mkdir(parents=True,exist_ok=True)
    manifest={}
    with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):continue
            relative=path.relative_to(root)
            if 'raw' in relative.parts or 'public' in relative.parts:continue
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
            'private_raw_records_location':[str(root/'evaluations'),str(root/'selection')],
            'note':'Reports, model checkpoints/exports, candidates, and one public sample replay; source-manifest identifies initial training snapshot, _report_source contains delivery-time renderers'},ensure_ascii=False,indent=2)+'\n')
    return {'archive':str(destination),'files':len(manifest),'bytes':destination.stat().st_size,
            'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
