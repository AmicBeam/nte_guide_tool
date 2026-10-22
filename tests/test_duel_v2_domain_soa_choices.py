"""Choice order and payload parity against the current formal legal_actions."""
from copy import deepcopy
import shutil
import unittest
import torch
from app.modules.card_game.engine.duel_v2.flow import new_game, legal_actions, deck_order_choice
from app.modules.card_game.rl.batched_duel.domain_soa_schema import CHOICE_KINDS
from app.modules.card_game.rl.batched_duel.domain_soa_choices import (
    DomainChoiceActions, INVALID, OVERFLOW, UNRESOLVED_SPEC, supports_full_duel_rules)


def fixture(kind='discard',n=5,**options):
    state=new_game(seed=63,skip_mulligan=True)
    cards=(state['sides']['a']['hand']+state['sides']['a']['deck']+
           state['sides']['b']['hand']+state['sides']['b']['deck'])[:n]
    state.update(phase='choice',pending_choice=dict(side='a',kind=kind,prompt='choice',cards=cards,**options))
    state['operation']=None
    return state


def prepared(states, viewers=None, active=None, resolved=True):
    viewers=viewers or ['a']*len(states)
    active=active if active is not None else [1]*len(states)
    data=[];refs=[]
    for state,viewer,on in zip(states,viewers,active):
        pending=state['pending_choice']
        spec=deck_order_choice(state)
        deck=bool(pending.get('kind')=='heiyu_divination'or pending.get('deck_order'))
        data.append([CHOICE_KINDS.index(pending['kind']),len(pending['cards']),
          int(spec.get('max_select',0)),int(spec.get('min_bottom')or 0),int(spec.get('max_bottom',0)),
          int('max_bottom'in spec),int(bool(spec.get('multi_bottom'))),int(bool(spec.get('hand_placement'))),
          int(deck),int(resolved),int(pending['side']==viewer),int(state['phase']=='finished'),on])
        refs.append([1000+i for i in range(len(pending['cards']))]+[-1]*(64-len(pending['cards'])))
    return torch.tensor(data,dtype=torch.int32),torch.tensor(refs,dtype=torch.int32)


def decode(batch,row,state):
    cards=state['pending_choice']['cards'];out=[]
    for i in range(int(batch.counts[row])):
        token=batch.tokens[row,i].tolist();selection=batch.selections[row,i].tolist()
        if token[0]==9:out.append(dict(type='concede'))
        elif token[0]==4:
            out.append(dict(type='choose',choice_id=cards[batch.references[row].tolist().index(token[1])]['instance_id']))
        elif token[0]==8:
            out.append(dict(type='choose_cards',card_ids=[cards[j]['instance_id']for j in selection[:token[2]]]))
        elif token[0]==6:
            top=token[2];bottom=token[3]
            ids=[cards[j]['instance_id']for j in selection[:top+bottom]]
            action=dict(type='choose_deck_order',card_ids=ids[:top])
            if token[4]:action['bottom_card_ids']=ids[top:]
            else:action['bottom_card_id']=ids[top]if bottom else None
            out.append(action)
        else:raise AssertionError(token)
    return out


class ChoiceActionsTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('clang++')or shutil.which('g++'),'C++ oracle unavailable')
    def test_current_engine_simple_combo_and_deck_order_actions_in_exact_order(self):
        states=[fixture(k,n,**kwargs)for k,n,kwargs in (
          ('discard',0,{}),('inspect_top',5,{}),('enemy_hand',5,{}),('kit',5,{}),
          ('hand_redraw',5,{}),('hand_redraw',16,{}),('kit_multi',40,{'max_select':2}),
          ('kit_multi',64,{'max_select':64}),('kit_multi',2,{'max_select':3}),
          ('deck_order',0,{'deck_order':True}),
          ('deck_order',5,{'deck_order':True,'multi_bottom':True}),
          ('deck_order',6,{'deck_order':True,'max_bottom':0}),
          ('deck_order',6,{'deck_order':True,'hand_placement':True}),
          ('deck_order',4,{'deck_order':True,'min_bottom':3,'max_bottom':1}))]
        x,refs=prepared(states);before=x.clone();before_refs=refs.clone()
        with DomainChoiceActions(device='cpu',allow_cpu_reference=True)as backend:
            out=backend.launch(backend.prepare(x,refs))
            self.assertEqual(out.errors.tolist(),[0]*len(states))
            for row,state in enumerate(states):
                self.assertEqual(decode(out,row,state),legal_actions(state,'a'))
        self.assertTrue(torch.equal(x,before));self.assertTrue(torch.equal(refs,before_refs))
        self.assertFalse(supports_full_duel_rules)

    @unittest.skipUnless(shutil.which('clang++')or shutil.which('g++'),'C++ oracle unavailable')
    def test_overflow_unresolved_and_invalid_do_not_emit_partial_legality(self):
        states=[fixture('hand_redraw',17),fixture('deck_order',6,deck_order=True),
                fixture('kit_multi',5,max_select=-1),fixture('discard',5)]
        x,refs=prepared(states)
        refs[3,1]=refs[3,0]
        with DomainChoiceActions(device='cpu',allow_cpu_reference=True)as backend:
            batch=backend.prepare(x,refs);out=backend.launch(batch)
            self.assertEqual(out.errors.tolist(),[OVERFLOW,OVERFLOW,INVALID,INVALID])
            self.assertEqual(out.counts.tolist(),[0]*4)
            self.assertTrue((out.tokens==-1).all())
            x[:]=prepared([fixture('deck_order',4,deck_order=True)]*4,resolved=False)[0]
            refs[:]=prepared([fixture('deck_order',4,deck_order=True)]*4)[1]
            out=backend.launch(batch)
            self.assertEqual(out.errors.tolist(),[UNRESOLVED_SPEC]*4)
            self.assertEqual(out.counts.tolist(),[0]*4)

    @unittest.skipUnless(shutil.which('clang++')or shutil.which('g++'),'C++ oracle unavailable')
    def test_non_actor_cannot_read_private_choice_and_inactive_or_finished_is_empty(self):
        states=[fixture('deck_order',6,deck_order=True)for _ in range(3)]
        states[2]['phase']='finished'
        x,refs=prepared(states,viewers=['b']*3,active=[1,0,1],resolved=False)
        x[:,1]=-99;refs.fill_(-999)
        with DomainChoiceActions(device='cpu',allow_cpu_reference=True)as backend:
            out=backend.launch(backend.prepare(x,refs))
            self.assertEqual(out.errors.tolist(),[0,0,0])
            self.assertEqual(out.counts.tolist(),[1,0,0])
            self.assertEqual(decode(out,0,states[0]),legal_actions(states[0],'b'))

    @unittest.skipUnless(shutil.which('clang++')or shutil.which('g++'),'C++ oracle unavailable')
    def test_named_fields_follow_live_state_and_reject_missing_multi_count(self):
        from app.modules.card_game.rl.batched_duel.domain_soa_codec import encode_domain_states
        from app.modules.card_game.rl.batched_duel.domain_soa_schema import CHOICE_INT_KEYS
        states=[fixture('discard',5),fixture('hand_redraw',5),fixture('kit_multi',5,max_select=2)]
        fields=encode_domain_states(states,presentation='drop').to_torch(device='cpu')
        viewer=torch.zeros(3,dtype=torch.int64);active=torch.ones(3,dtype=torch.bool)
        with DomainChoiceActions(device='cpu',allow_cpu_reference=True)as backend:
            batch=backend.prepare_from_fields(fields,viewer,active,spec_resolved=active)
            out=backend.launch(batch)
            self.assertEqual(out.errors.tolist(),[0]*3)
            for row,state in enumerate(states):
                self.assertEqual(decode(out,row,state),legal_actions(state,'a'))
            viewer.fill_(1)
            out=backend.launch(batch)
            self.assertEqual(out.counts.tolist(),[1]*3)
            viewer.fill_(0)
            fields.arrays['choice_int_presence'][2,CHOICE_INT_KEYS.index('max_select')]=0
            out=backend.launch(batch)
            self.assertEqual(int(out.errors[2]),INVALID)
            self.assertEqual(int(out.counts[2]),0)

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA graph required')
    def test_400_cuda_graph_matches_formal_actions_and_reads_changed_fields(self):
        choices=[fixture('discard',5),fixture('hand_redraw',5),fixture('kit_multi',5,max_select=2),
                 fixture('inspect_top',4)]
        states=[deepcopy(choices[i%4])for i in range(400)]
        from app.modules.card_game.rl.batched_duel.domain_soa_codec import encode_domain_states
        from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoALayout
        fields=encode_domain_states(states,DomainSoALayout(game_capacity=400),presentation='drop').to_torch(device='cuda')
        viewers=torch.zeros(400,dtype=torch.int64,device='cuda')
        active=torch.ones(400,dtype=torch.bool,device='cuda')
        with DomainChoiceActions(device='cuda')as backend:
            # Deck-order specs are resolved explicitly before this stage.
            manual_states=[fixture('deck_order',4,deck_order=True,multi_bottom=True),
                           fixture('deck_order',6,deck_order=True,max_bottom=0),
                           fixture('deck_order',5,deck_order=True,hand_placement=True)]
            mx,mr=prepared(manual_states)
            manual=backend.launch(backend.prepare(mx.cuda(),mr.cuda()))
            torch.cuda.synchronize()
            host=type(manual)(None,mx,mr,*(getattr(manual,k).cpu()for k in ('tokens','selections','counts','errors')))
            self.assertEqual(host.errors.tolist(),[0]*3)
            for row,state in enumerate(manual_states):
                self.assertEqual(decode(host,row,state),legal_actions(state,'a'))
            batch=backend.prepare_from_fields(fields,viewers,active,spec_resolved=active)
            backend.launch(batch);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):backend.launch(batch)
            for actor in (1,0):
                viewers.fill_(0 if actor else 1)
                graph.replay();torch.cuda.synchronize()
                cpu=type(batch)(None,batch.inputs.cpu(),batch.references.cpu(),*(getattr(batch,k).cpu()for k in ('tokens','selections','counts','errors')))
                self.assertEqual(cpu.errors.tolist(),[0]*400)
                for row,state in enumerate(states):
                    self.assertEqual(decode(cpu,row,state),legal_actions(state,'a'if actor else 'b'))
            fields.arrays['choice_card_index'][0,0]=fields.arrays['card_valid'].shape[1]+5
            viewers.fill_(0)
            graph.replay();torch.cuda.synchronize()
            self.assertEqual(int(batch.errors[0]),INVALID)
            self.assertEqual(int(batch.counts[0]),0)
            self.assertTrue((batch.errors[1:]==0).all())
            viewers.fill_(1)
            graph.replay();torch.cuda.synchronize()
            self.assertEqual(batch.errors.cpu().tolist(),[0]*400)
            self.assertEqual(batch.counts.cpu().tolist(),[1]*400)
            del graph
            torch.cuda.synchronize()

if __name__=='__main__':unittest.main()
