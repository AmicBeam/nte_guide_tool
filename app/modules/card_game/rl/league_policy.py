"""Numeric-only v5 policies and explicit old-feature warm starts. No auto-training."""
import hashlib,json
from pathlib import Path
import numpy as np
from .league_schema import SCHEMA,SEATS,CARD_IDS,CAND_DIM,PRESETS,feature_names,rule_hash,build_hash,deck

SHARED=('state_net.0','state_net.2','cand_net.0','cand_net.2','score','value.0','value.2')

def shapes(hidden):
    h=hidden;d=len(feature_names())
    return {f'{name}.{part}':shape for name,ins,outs in [('state_net.0',d,h),('state_net.2',h,h),('cand_net.0',CAND_DIM,h),('cand_net.2',h,h),('score',h,1),('value.0',h,h),('value.2',h,1)] for part,shape in [('weight',(outs,ins)),('bias',(outs,))]}

class LeagueModel:
    def __init__(self,directory,key,*,require_approved=False):
        directory=Path(directory);m=json.loads((directory/(key+'.json')).read_text(encoding='utf-8'))
        p=directory/(key+'.npz');digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if key not in PRESETS or m.get('schema')!=SCHEMA or m.get('deck')!=key:raise ValueError('League model schema mismatch')
        runtime = rule_hash()
        compatibility = m.get('runtime_compatibility') or {}
        self.runtime_compatible = bool(
            compatibility.get('kind') == 'explicit_user_compatibility'
            and compatibility.get('request')
            and compatibility.get('source_rule_hash') == m.get('rule_hash')
            and compatibility.get('runtime_rule_hash') == runtime
            and compatibility.get('model_sha256') == digest
            and compatibility.get('build_sha256') == m.get('build_sha256'))
        if m.get('rule_hash') != runtime and not self.runtime_compatible:
            raise ValueError('League model rule/schema mismatch')
        if m.get('sha256')!=digest or m.get('seat_ids')!=list(SEATS) or m.get('card_ids')!=list(CARD_IDS):raise ValueError('League model identity mismatch')
        if m.get('features')!=feature_names() or m.get('cand_dim')!=CAND_DIM:raise ValueError('League feature contract mismatch')
        self.hidden=int(m['hidden'])
        if not 1<=self.hidden<=256:raise ValueError('Invalid hidden width')
        with np.load(p,allow_pickle=False) as z:self.weights={n:z[n].astype(np.float32,copy=True) for n in shapes(self.hidden)}
        for n,s in shapes(self.hidden).items():
            if self.weights[n].shape!=s or not np.isfinite(self.weights[n]).all():raise ValueError('Invalid tensor '+n)
            self.weights[n].flags.writeable=False
        self.serving_deck=m['build'];self.serving_build_sha256=build_hash(self.serving_deck)
        # Serving lineups belong to the model, independently of gift presets.
        if not set(self.serving_deck['character_ids']) <= set(SEATS) or m['build_sha256']!=self.serving_build_sha256:raise ValueError('Build identity mismatch')
        v=m.get('validation') or {}
        self.approved=bool(v.get('approved') and v.get('complete') and v.get('model_sha256')==digest and v.get('build_sha256')==self.serving_build_sha256 and v.get('rule_hash')==m['rule_hash'])
        # Manual candidate activation is separate from quality-gate approval.
        # It is bound to the exact model, build and rules, never a global bypass.
        authorization=m.get('serving_authorization') or {}
        self.user_authorized=bool(
            authorization.get('kind')=='explicit_user_candidate'
            and authorization.get('request')
            and authorization.get('report')
            and authorization.get('model_sha256')==digest
            and authorization.get('build_sha256')==self.serving_build_sha256
            and authorization.get('rule_hash')==m['rule_hash'])
        self.serving_authorized=self.approved or self.user_authorized
        if require_approved and not self.serving_authorized:raise ValueError('League model lacks serving acceptance or explicit candidate authorization')
        self.schema=SCHEMA;self.deck_id=key;self.version=digest;self.manifest=m
        self.capability={'kind':'official_public_v5','human_public_custom':self.serving_authorized,'bot_preset':key}
    def scores_value(self,state,candidates):
        def linear(x,n):return np.einsum('...j,ij->...i',x,self.weights[n+'.weight'],optimize=False)+self.weights[n+'.bias']
        sh=np.maximum(linear(state,'state_net.0'),0);sh=np.maximum(linear(sh,'state_net.2'),0)
        ch=np.maximum(linear(candidates,'cand_net.0'),0);ch=np.maximum(linear(ch,'cand_net.2'),0)
        scores=np.sum(ch*sh,-1)/np.sqrt(np.float32(self.hidden))+linear(ch,'score').reshape(-1)
        value=linear(np.maximum(linear(sh,'value.0'),0),'value.2').item()
        if not np.isfinite(scores).all() or not np.isfinite(value):raise ValueError('Non-finite prediction')
        return scores,float(value)
    def scores(self,state,candidates):return self.scores_value(state,candidates)[0]
    def select_public_action(self,view,actions):
        from .league_observation import encode_league
        return int(self.scores(*encode_league(view,actions)).argmax())
    def validate_human(self,build):
        from app.modules.card_game.content.duel_v2 import validate_deck
        b=validate_deck(build)
        if not self.serving_authorized or not set(b['character_ids'])<=set(SEATS):raise ValueError('Unvalidated league model or unsupported characters')
        return self.capability

def export_model(net,output,key,build,*,origin=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    p=out/(key+'.npz');temp=out/(key+'.tmp.npz')
    weights={k:v.detach().cpu().numpy() for k,v in net.state_dict().items()}
    np.savez_compressed(temp,**weights);temp.replace(p)
    m={'schema':SCHEMA,'deck':key,'rule_hash':rule_hash(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'seat_ids':list(SEATS),'card_ids':list(CARD_IDS),'features':feature_names(),'cand_dim':CAND_DIM,'hidden':net.score.in_features,'build':build,'build_sha256':build_hash(build),'reward_mode':'outcome_only_v1','opening_policy':'learned','validation':{'approved':False},'origin':origin}
    temp=out/(key+'.tmp.json');temp.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(out/(key+'.json'))
    return m

def network_from_old(path,key,device='cuda'):
    import torch
    from .gpu_duel.ppo import CompactScorer
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('schema')!='resident_public_v4' or saved.get('reward_mode')!='outcome_only_v1':raise ValueError('Only explicitly mapped v4 outcome policies may warm-start')
    w=saved.get('network') or saved.get('model')
    if w is None:raise ValueError('Checkpoint has no network weights')
    h=w['state_net.0.weight'].shape[0];net=CompactScorer(len(feature_names()),CAND_DIM,h)
    target=net.state_dict()
    for n,v in w.items():
        if n not in ('state_net.0.weight','cand_net.0.weight'):target[n]=v.clone()
    old=feature_names(old=True);new={n:i for i,n in enumerate(feature_names())}
    if w['state_net.0.weight'].shape[1]!=len(old) or w['cand_net.0.weight'].shape[1]!=60:raise ValueError('Unexpected legacy feature shape')
    target['state_net.0.weight'].zero_();target['cand_net.0.weight'].zero_()
    for i,n in enumerate(old):target['state_net.0.weight'][:,new[n]]=w['state_net.0.weight'][:,i]
    target['cand_net.0.weight'][:,:60]=w['cand_net.0.weight']
    net.load_state_dict(target);return net.to(device),{'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'source_deck':saved['deck'],'warm_start_only':True,'new_features_zero_initialized':True}
