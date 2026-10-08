"""Numeric export for the public resident scorer; exporting never trains."""
from pathlib import Path
import hashlib
import json

from .public_schema import SCHEMA, state_dim, rule_identity, CAND_DIM
from .gpu_duel.catalog import CARD_IDS, SEATS, GPU_LOCK
from .capability import PUBLIC_CANDIDATE_KIND


def export_checkpoint(source, output):
    import numpy as np
    import torch
    from .gpu_duel.ppo import CompactScorer
    source, output = Path(source), Path(output)
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    from .opening_observation import OPENING_SCHEMA,OPENING_CAND_DIM,encode_opening
    schema=checkpoint.get('schema')
    cand_dim=OPENING_CAND_DIM if schema==OPENING_SCHEMA else CAND_DIM
    if schema not in (SCHEMA,OPENING_SCHEMA) or checkpoint.get('gpu_lock') != GPU_LOCK:
        raise ValueError('Checkpoint observation/catalog mismatch')
    if checkpoint.get('rule_identity',{}).get('rule_hash') != rule_identity(schema):
        raise ValueError('Checkpoint rule identity mismatch')
    if checkpoint.get('state_dim') != state_dim() or checkpoint.get('cand_dim') != cand_dim:
        raise ValueError('Checkpoint dimensions mismatch')
    deck = checkpoint.get('deck')
    if deck not in ('starter','weave-rush'):
        raise ValueError('Unknown learner preset')
    from .public_training import training_deck
    learner_deck = training_deck(deck, checkpoint.get('learner_deck'))
    hidden = checkpoint['hidden']
    if type(hidden) is not int or not 1 <= hidden <= 2048:
        raise ValueError('Invalid hidden dimension')
    network = CompactScorer(state_dim(),cand_dim,hidden)
    network.load_state_dict(checkpoint['model'])
    network.eval()
    arrays = {k:v.detach().cpu().numpy() for k,v in network.state_dict().items() if not k.startswith('value.')}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Non-finite checkpoint weights')
    if schema==OPENING_SCHEMA and (not checkpoint.get('learn_mulligan') or checkpoint.get('opening_policy')!='learned_joint_v1' or checkpoint.get('update',0)<1):
        raise ValueError('Opening checkpoint must explicitly learn mulligan')
    if output.exists():
        raise ValueError('Choose a new export directory')
    output.mkdir(parents=True)
    path = output/f'{deck}.npz'
    np.savez_compressed(path,**arrays)
    manifest = {k:checkpoint[k] for k in ('schema','deck','hidden','state_dim','cand_dim','gpu_lock','update','rule_identity')}
    manifest['learner_deck'] = learner_deck
    for field in ('training_distribution','opponent_distribution','pinned_opponents','reward_mode'):
        if field in checkpoint:manifest[field]=checkpoint[field]
    if schema==OPENING_SCHEMA:
        manifest['opening_policy']='learned_joint_v1'
    manifest.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    card_ids=list(CARD_IDS),seat_ids=list(SEATS),
                    capability={'kind':PUBLIC_CANDIDATE_KIND,'bot_preset':deck,
                                'human_public_custom':False,'validated_distribution':None})
    (output/f'{deck}.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Verify exported math on varied public observations. This is parity, not strength evaluation.
    from app.modules.card_game.engine.ai.advanced_model import FrozenModel
    from app.modules.card_game.engine.duel_v2 import new_game, observe, legal_actions, acting_side
    from .capability import sample_public_deck,preset_opponent_deck
    from .public_observation import encode_public
    model = FrozenModel(output,deck)
    largest = 0.0
    for seed in range(8):
        state = new_game(seed=seed,skip_mulligan=(schema!=OPENING_SCHEMA or seed%2==0),
                         decks={'a':learner_deck,'b':sample_public_deck(seed)})
        side = acting_side(state)
        actions = [a for a in legal_actions(state,side) if a['type']!='concede']
        values,candidates = encode_opening(observe(state,side),actions) if schema==OPENING_SCHEMA else encode_public(observe(state,side),actions)
        with torch.no_grad():
            logits,_ = network(torch.tensor(values)[None],torch.tensor(candidates)[None],
                               torch.ones(1,len(actions),dtype=torch.bool))
        expected = logits[0].numpy()
        got = model.scores(values,candidates)
        np.testing.assert_allclose(got,expected,rtol=1e-4,atol=1e-5)
        if got.argmax()!=expected.argmax():
            raise ValueError('Export action parity failed')
        largest = max(largest,float(np.max(np.abs(got-expected))))
    result = {'model':str(path),'model_sha256':model.version,'cases':8,'max_error':largest,
              'validated_for_serving':False}
    (output/'export-check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def approve_candidate(model_path, report_path, *, min_pairs=32, min_win_rate=0.5, serving_build=None):
    """Attach completed, matching public evaluation evidence to numeric weights."""
    from .capability import capability_ready_manifest
    model_path,report_path=Path(model_path),Path(report_path)
    manifest_path=model_path.with_suffix('.json')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    report=json.loads(report_path.read_text(encoding='utf-8'))
    from .capability import is_exact_preset
    if serving_build is None and manifest.get('learner_deck') and not is_exact_preset(manifest['learner_deck'],manifest.get('deck')):
        raise ValueError('A custom learner model cannot approve a fixed-preset website model')
    if serving_build is None and report.get('learner_is_preset',True) is not True:
        raise ValueError('A tuned-deck report cannot approve a fixed-preset website model')
    if serving_build is not None:
        from .capability import fixed_team_serving_deck, serving_deck_hash
        serving_build = fixed_team_serving_deck(serving_build, manifest.get('deck'))
        actual_report_build = fixed_team_serving_deck(report.get('learner_build') or {}, manifest.get('deck'))
        if serving_deck_hash(serving_build) != serving_deck_hash(actual_report_build):
            raise ValueError('Report does not match the exact serving build')
    digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
    if type(min_pairs) is not int or min_pairs<1 or not 0<=min_win_rate<=1:
        raise ValueError('Invalid acceptance threshold')
    from .opening_observation import OPENING_SCHEMA
    schema=manifest.get('schema')
    if schema not in (SCHEMA,OPENING_SCHEMA) or manifest.get('sha256')!=digest or int(manifest.get('update',0))<1:
        raise ValueError('Only a trained public checkpoint can be approved')
    identity=rule_identity(schema)
    model=report.get('model') or {}
    if (model.get('sha256'),model.get('schema'),model.get('rule_hash'),report.get('preset')) != (
            digest,schema,identity,manifest.get('deck')):
        raise ValueError('Report does not match the exact model, rules, and preset')
    if (manifest.get('runtime_rule_hash') or (manifest.get('rule_identity') or {}).get('rule_hash'))!=identity:
        raise ValueError('Model rule identity changed')
    if report.get('sampled_opponent') is not True or len(report.get('by_opponent_team') or {})<15:
        raise ValueError('Evaluation must cover all 15 public opponent teams')
    games=report.get('games_index') or []
    if schema==OPENING_SCHEMA and (model.get('opening_policy')!='learned_joint_v1'
            or model.get('opening_override') or report.get('engine',{}).get('skip_mulligan') is not False
            or any('mulligan' not in g.get('action_types',[]) for g in games)):
        raise ValueError('Opening model approval requires actual learned-mulligan evaluation')
    if not games or any(not g.get('terminated') or g.get('truncated') or g.get('error') for g in games):
        raise ValueError('Evaluation contains incomplete/error games')
    from .capability import PUBLIC_CHARACTERS
    teams={tuple(sorted(g.get('opponent_team') or [])) for g in games}
    if len(teams)!=15 or any(len(set(team))!=4 or not set(team)<=set(PUBLIC_CHARACTERS) for team in teams):
        raise ValueError('Game index must cover all 15 actual public teams')
    if len(games)!=int(report.get('pairs',0))*2 or len(games)<min_pairs*2:
        raise ValueError('Not enough complete paired evaluation games')
    pairs={}
    for game in games:
        pair=pairs.setdefault(game['pair'],{})
        if game['position'] in pair:raise ValueError('Duplicate evaluation seat')
        pair[game['position']]=game
    if any(set(pair)!= {'first','second'} or pair['first']['opponent_team']!=pair['second']['opponent_team']
           for pair in pairs.values()):
        raise ValueError('Evaluation pairs are inconsistent')
    wins=sum(g.get('winner')=='a' for g in games)
    if wins/len(games)<min_win_rate:raise ValueError('Evaluation win-rate threshold not met')
    engine=report.get('engine') or {}
    if engine.get('escalation') is not True or engine.get('first_turn_draw') is not True:
        raise ValueError('Evaluation omitted current rules')
    from app.modules.card_game.engine.ai.advanced_model import FrozenModel
    FrozenModel(model_path.parent,manifest['deck'])  # Numeric shape/hash validation before approval.
    manifest['capability']=capability_ready_manifest(manifest['deck'])
    manifest['validation']={'complete':True,'model_sha256':digest,'rule_hash':identity,
                            'games':len(games),'wins':wins,'min_pairs':min_pairs,
                            'min_win_rate':min_win_rate,'opponent_teams':15,
                            'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest()}
    if serving_build is not None:
        manifest['serving_build'] = serving_build
        manifest['validation']['serving_build_sha256'] = serving_deck_hash(serving_build)
    temp=manifest_path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(manifest_path)
    return manifest['validation']
