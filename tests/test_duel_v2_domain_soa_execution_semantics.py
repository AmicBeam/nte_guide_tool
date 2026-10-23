"""Full native physical state comparisons; explicit CPU oracle is not CUDA acceptance."""
import random
from copy import deepcopy
import unittest
import torch
from app.modules.card_game.rl.batched_duel.domain_soa_execution import DomainSoAExecution
from app.modules.card_game.rl.batched_duel.domain_soa_codec import _decode_row
from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoABatch,DomainSoALayout,CARD_VOCAB
from app.modules.card_game.content.duel_v2 import STARTER_DECKS,CARDS
from app.modules.card_game.engine.duel_v2.flow import apply_action,legal_actions,acting_side
from app.modules.card_game.engine.duel_v2.state import card_instance
from app.modules.card_game.engine.duel_v2.summons import summon_front
from app.modules.card_game.engine.duel_v2.context import EffectContext
from app.modules.card_game.content.duel_v2.characters.zaowu import summon_gulang
from tests.test_duel_v2_domain_soa_character_queries import state,pack

def differences(a,b,path=''):
 if type(a)!=type(b)and not(isinstance(a,dict)and isinstance(b,dict)):return [(path,repr(a),repr(b))]
 if isinstance(a,dict):
  out=[]
  for k in a.keys()|b.keys():
   if k not in a:out.append((path+'/'+str(k),'missing',repr(b[k])))
   elif k not in b:out.append((path+'/'+str(k),repr(a[k]),'missing'))
   else:out.extend(differences(a[k],b[k],path+'/'+str(k)))
  return out
 if isinstance(a,(list,tuple)):
  if len(a)!=len(b):return[(path+'/length',len(a),len(b))]
  return [d for j,(x,y)in enumerate(zip(a,b))for d in differences(x,y,path+'/'+str(j))]
 return []if a==b else[(path,repr(a),repr(b))]


def fixtures():
 keys=('starter','weave-rush','quick-rush','zhenhong','murk','collapse','lingke');decks={d['id']:d for d in STARTER_DECKS};cases=[];skipped=[]
 for seq,cid in enumerate(CARD_VOCAB):
  spec=CARDS[cid];owner=spec['character_id'];key=next((k for k in keys if owner in decks[k]['character_ids']),None)

  if key is None:
   from app.modules.card_game.engine.duel_v2 import new_game
   from app.modules.card_game.content.duel_v2.catalog import _starter_card_ids
   lineup=[owner,'zero','nanali','jiuyuan'];deck={'id':'fixture-haiyue','name':'海月规则对拍','official':False,'character_ids':lineup,'card_ids':_starter_card_ids(lineup)}
   st=new_game(seed=400+seq,decks={'a':deck,'b':deck},skip_mulligan=True)
  else:st=state(400+seq,a=key,b='starter'if cid=='A01'else key)
  st['active_side']='a';st['phase']='playing';own=st['sides']['a'];foe=st['sides']['b'];own.update(ap=9,normal_attack_available=True,ultimate_available=False)
  for team in (own,foe):
   for h in team['characters'].values():h['hp']=max(1,h['max_hp']-1);h['jingu']=9
  foe['front']=foe['order'][0]
  own['discard'].extend(own['hand']);own['hand']=[];card=card_instance(st,cid,'a');own['hand'].append(card)
  policy=__import__('app.modules.card_game.content.duel_v2.effects',fromlist=['TARGET_POLICIES']).TARGET_POLICIES.get(cid)
  if policy in ('downed_ally','other_downed_ally'):
   target=next(c for c in own['order']if c!=owner);own['characters'][target].update(hp=0,down_turns=1)
  if policy=='downed_enemy':foe['characters'][foe['order'][1]].update(hp=0,down_turns=1)
  if policy=='collapsed_enemy_esper':foe['characters'][foe['order'][0]]['flags']['collapse']={'side':'b','until':2}
  if policy=='any_summon':summon_front(st,'b',name='fixture summon',attack=1,hp=4)
  if cid=='B03':foe['front_debuff']['burn']={'left':2,'by':'a','stacks':1,'source':'a:'+owner,'applied_turn':st['turn']}
  if cid=='I05':foe['front_debuff']['delay']={'left':2,'tick_side':'a','by':'a','slow':0,'slow_turn':st['turn']}
  if cid=='Z01':own['front']=own['order'][0]
  if cid=='S06':summon_gulang(EffectContext(st,'a',owner,{'side':'a'}),'vinyl')
  if spec.get('require')=='surplus':own['surplus']=True
  if spec.get('require')=='harmony_damage':own['harmony_damage']=True
  if spec.get('require')=='own_burn':own['front_debuff']['burn']={'left':2,'by':'b','stacks':1,'source':'b:'+foe['order'][0],'applied_turn':st['turn']}
  legal=[a for a in legal_actions(st,'a')if a['type']!='concede'];choices=[(i,a)for i,a in enumerate(legal)if a['type']=='play_card'and a['card_id']==card['instance_id']]
  if not choices:skipped.append(cid);continue
  for pick,action in ([choices[0]]+[(i,a)for i,a in choices if a.get('option_id')not in (None,'base')][:4]):
   cases.append((cid,st,pick,action))
 return cases,skipped


class NativeExecutionSemanticsTest(unittest.TestCase):
 def test_codec_preserves_continuous_damage_iteration_order(self):
  from app.modules.card_game.engine.duel_v2.continuous import apply_dot
  world=state(200)
  for kind in ('guard','venom','nightmare','etch'):
   apply_dot(world,'b',kind,by='a',source_cid=world['sides']['a']['order'][0],stacks=1)
  layout,fields=pack([world]);got=_decode_row(DomainSoABatch(layout=layout,**{k:v.numpy()for k,v in fields.items()}),0)
  self.assertEqual(list(got['sides']['b']['front_debuff']['dots']),list(world['sides']['b']['front_debuff']['dots']))
 def test_all_public_cards_and_paid_options(self):
  torch.set_num_threads(1)
  cases,skipped=fixtures();self.assertEqual(skipped,[]);self.assertEqual(len(set(c[0] for c in cases)),141)
  layout,fields=pack([c[1]for c in cases]);n=len(cases)
  with DomainSoAExecution(layout,device='cpu',allow_cpu_oracle=True,allow_unqualified=True)as core:
   out=core.launch(core.prepare(fields,torch.zeros(n,dtype=torch.int64),torch.tensor([c[2]for c in cases]),torch.ones(n,dtype=torch.bool)))
   self.assertFalse(out['errors'].any(),out['errors'].tolist())
   batch=DomainSoABatch(layout=layout,**{k:v.numpy()for k,v in out['fields'].items()})
   for row,(card,world,index,action)in enumerate(cases):
    with self.subTest(card=card,action=action):
     gold=apply_action(world,'a',action);gold.pop('events',None);gold.pop('_public_board',None)
     self.assertEqual(differences(gold,_decode_row(batch,row)),[])
 def test_all_seventeen_public_ultimates(self):
  cases=[];seen=set()
  for card,source,index,action in fixtures()[0]:
   owner=next(c['character_id']for c in source['sides']['a']['hand']if c['instance_id']==action['card_id'])
   if owner in seen:continue
   seen.add(owner);world=deepcopy(source);own=world['sides']['a'];own['front']=owner;own['ultimate_available']=True;own['characters'][owner]['energy']=9
   legal=[a for a in legal_actions(world,'a')if a['type']!='concede'];action={'type':'ultimate','character_id':owner};cases.append((owner,world,legal.index(action),action))
  self.assertEqual(len(cases),17);layout,fields=pack([c[1]for c in cases]);n=len(cases)
  with DomainSoAExecution(layout,device='cpu',allow_cpu_oracle=True,allow_unqualified=True)as core:
   out=core.launch(core.prepare(fields,torch.zeros(n,dtype=torch.int64),torch.tensor([c[2]for c in cases]),torch.ones(n,dtype=torch.bool)))
   self.assertFalse(out['errors'].any(),out['errors'].tolist());batch=DomainSoABatch(layout=layout,**{k:v.numpy()for k,v in out['fields'].items()})
   for row,(owner,world,index,action)in enumerate(cases):
    with self.subTest(owner=owner):
     gold=apply_action(world,'a',action);gold.pop('events',None);gold.pop('_public_board',None);self.assertEqual(differences(gold,_decode_row(batch,row)),[])
 def test_fourteen_natural_games_against_formal_engine(self):
  torch.set_num_threads(1);keys=('starter','weave-rush','quick-rush','zhenhong','murk','collapse','lingke');n=14
  worlds=[state(900+i,a=keys[i%7],b=keys[(i+1)%7])for i in range(n)]
  layout,fields=pack(worlds,layout=DomainSoALayout(game_capacity=n,max_characters=16))
  active=torch.ones(n,dtype=torch.bool);sides=torch.zeros(n,dtype=torch.int64);picks=torch.zeros(n,dtype=torch.int64);rng=[random.Random(1000+i)for i in range(n)]
  with DomainSoAExecution(layout,device='cpu',allow_cpu_oracle=True,allow_unqualified=True)as core:
   prepared=core.prepare(fields,sides,picks,active)
   for step in range(350):
    choices=[];counts=[]
    for row,world in enumerate(worlds):
     if not active[row]:choices.append(None);counts.append(0);continue
     actor=acting_side(world);sides[row]=0 if actor=='a'else 1
     legal=[a for a in legal_actions(world,actor)if a['type']!='concede'];picks[row]=rng[row].randrange(len(legal));choices.append(legal[int(picks[row])]);counts.append(len(legal))
    if not active.any():break
    out=core.launch(prepared);self.assertFalse(out['errors'].any(),(step,out['errors'].tolist(),out['error_sites'].tolist()))
    batch=DomainSoABatch(layout=layout,**{k:v.numpy()for k,v in out['fields'].items()})
    for row in range(n):
     if not active[row]:continue
     gold=apply_action(worlds[row],('a','b')[int(sides[row])],choices[row]);gold.pop('_public_board',None);gold.pop('events',None)
     with self.subTest(row=row,step=step,action=choices[row]):
      self.assertEqual(int(out['legal_counts'][row]),counts[row]);self.assertEqual(differences(gold,_decode_row(batch,row)),[])
     gold['events']=[];worlds[row]=gold
     if gold['phase']=='finished':active[row]=False
    for key,value in fields.items():value.copy_(out['fields'][key])
   self.assertFalse(active.any(),'All games must end naturally; prefixes do not count')
