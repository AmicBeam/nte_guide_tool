"""Independent modifier evaluation, not physical rule/search qualification."""
import shutil
import unittest
import torch
from app.modules.card_game.engine.duel_v2.modifiers import add_effect,evaluate
from app.modules.card_game.rl.batched_duel.domain_soa_rule_queries import DomainRuleQueries,PROPERTIES
from tests.test_duel_v2_domain_soa_character_queries import state,pack


def backend(layout,device='cpu'):
    if device=='cpu'and not(shutil.which('clang++')or shutil.which('g++')):raise unittest.SkipTest('Explicit compiler oracle unavailable')
    return DomainRuleQueries(layout,device=device,allow_cpu_oracle=device=='cpu')


def requests(states,layout,device='cpu'):
    C=layout.max_characters;H=2*(C+1);specs=[]
    for h in (0,1,2,C+1,C+2):
        for p in range(len(PROPERTIES)):
            for base in (0.,1.,3.):specs.append((h,p,base))
    B=len(states)
    return (torch.tensor([[x[0]for x in specs]]*B,dtype=torch.int64,device=device),
            torch.tensor([[x[1]for x in specs]]*B,dtype=torch.int32,device=device),
            torch.tensor([[x[2]for x in specs]]*B,dtype=torch.float64,device=device),
            torch.ones(B,len(specs),dtype=torch.bool,device=device),torch.ones(B,len(specs),dtype=torch.bool,device=device))


class RuleQueriesTest(unittest.TestCase):
    def compare(self,states,layout,args,out):
        self.assertFalse(out['errors'].any(),str(out['errors'].cpu().tolist()))
        for b,st in enumerate(states):
            for n in range(args[0].shape[1]):
                h=int(args[0][b,n]);side=('a','b')[h//(layout.max_characters+1)];c=h%(layout.max_characters+1)-1
                subject=st['sides'][side]if c<0 else list(st['sides'][side]['characters'].values())[c]
                prop=PROPERTIES[int(args[1][b,n])];base=float(args[2][b,n])if bool(args[3][b,n])else None
                expected=evaluate(st,side,subject,prop,base)['value']
                self.assertEqual(bool(out['present'][b,n]),expected is not None)
                self.assertEqual(float(out['values'][b,n]),float(expected or 0),(b,n,prop,base))
        self.assertFalse(out['full_duel_qualified'])

    def test_all_properties_sources_scopes_stable_layers_and_bases(self):
        states=[state(10+i,a=k)for i,k in enumerate(('starter','murk','lingke','collapse'))]
        for st in states:
            team=st['sides']['a'];h=team['characters'][team['order'][0]]
            for prop in ('card.cost','card.instant','rule.normal_attack_instant','marker.lingke_taunt'):
                add_effect(team,'test.'+prop,prop,2,scope='allies',source_entity_id=h['entity_id'],layer=1)
                add_effect(h,'test.set.'+prop,prop,3,op='set',layer=0,clear_on=())
                add_effect(team,'test.mul.'+prop,prop,.5,op='multiply',scope='other_allies',source_entity_id=h['entity_id'],layer=1,conditions=[{'id':'source_alive'}])
        layout,fields=pack(states);args=requests(states,layout)
        with backend(layout)as enc:self.compare(states,layout,args,enc.query(fields,*args))

    def test_absent_bases_player_attribute_none_and_expiry(self):
        st=state(23);team=st['sides']['a'];team['turn_count']=4;h=team['characters'][team['order'][0]]
        add_effect(team,'player.none','card.cost',2,conditions=[{'id':'attribute_is'}],clear_on=())
        add_effect(h,'expires','card.cost',9,expiry={'side':'a','turn':4,'phase':'turn_start'},clear_on=())
        layout,fields=pack([st]);args=requests([st],layout)
        args[3][args[1]==3]=False
        with backend(layout)as enc:self.compare([st],layout,args,enc.query(fields,*args))

    def test_live_noncontiguous_invalid_subject_property_and_prior_error(self):
        states=[state(31),state(32)];layout,raw=pack(states);fields={}
        for name,v in raw.items():
            storage=torch.empty((4,)+tuple(v.shape[1:]),dtype=v.dtype);fields[name]=storage[::2];fields[name].copy_(v)
        args=requests(states,layout)
        with backend(layout)as enc:
            p=enc.prepare(fields,*args);self.compare(states,layout,args,enc.launch(p))
            args[0][0,0]=-1;args[1][0,1]=99;args[2][0,2]=float('nan');fields['row_error'][1]=64
            out=enc.launch(p);self.assertEqual(out['errors'][0,:3].tolist(),[1,1,3]);self.assertTrue((out['errors'][1]==64).all());self.assertFalse(out['valid'][1].any())
            args[4][0,:3]=False;out=enc.launch(p);self.assertTrue((out['errors'][0,:3]==0).all());self.assertFalse(out['valid'][0,:3].any())
            fields['catalog_digest'][0,0]^=1;self.assertTrue((enc.launch(p)['errors'][0,3:]==1).all())

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_graph_current_subject_bases_and_errors(self):
        st=state(60,a='lingke');team=st['sides']['a'];h=team['characters']['lingke']
        add_effect(team,'lingke.next_instant','card.instant',True,op='allow',scope='allies',source_entity_id=h['entity_id'],clear_on=(),conditions=[{'id':'card_owner_other'}])
        add_effect(h,'card.discount','card.cost',-.5,scope='allies',source_entity_id=h['entity_id'],clear_on=())
        states=[st]*400;layout,fields=pack(states,device='cuda');args=requests(states,layout,'cuda')
        with backend(layout,'cuda')as enc:
            p=enc.prepare(fields,*args);enc.launch(p);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=enc.launch(p)
            g.replay();torch.cuda.synchronize()
            # Independent full comparison on selected rows, then device changes.
            small={k:v[:2]for k,v in out.items()if isinstance(v,torch.Tensor)};small['full_duel_qualified']=False
            self.compare(states[:2],layout,tuple(a[:2]for a in args),small)
            args[2][0,5]=7;fields['row_error'][1]=64;fields['catalog_digest'][2,0]^=1
            g.replay();torch.cuda.synchronize();self.assertTrue((out['errors'][1]==64).all());self.assertTrue((out['errors'][2]==1).all())
            small={k:v[:1]for k,v in out.items()if isinstance(v,torch.Tensor)};small['full_duel_qualified']=False
            self.compare(states[:1],layout,tuple(a[:1]for a in args),small)
