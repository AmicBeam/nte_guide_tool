"""Opt-in compiler structure trial; full-state parity with fallback."""
from contextlib import ExitStack
import copy
import importlib.util
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
class CompiledTrialTest(unittest.TestCase):
    def test_common_sortie_and_fallback_all_fields(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.env import GpuDuelEnv
        from app.modules.card_game.rl.gpu_duel import rules
        from app.modules.card_game.rl.gpu_duel.catalog import SEAT_INDEX, CARD_INDEX
        from app.modules.card_game.rl.gpu_duel.compiled_trial import CompiledSortieTrial
        from app.modules.card_game.rl.rule_ir.lower_cpu import lower_cpu
        env=GpuDuelEnv(6,device='cpu');env.reset(seeds=list(range(6)))
        s=env.state;side=s.active.clone();b=torch.arange(6);foe=1-side;zero=SEAT_INDEX['zero'];nanali=SEAT_INDEX['nanali']
        s.hand_kind.fill_(-1);s.hand_n.zero_();s.front.fill_(-1);s.last_front.fill_(-1)
        s.ch_shape.fill_(-1);s.ch_awakened.zero_();s.ch_harmony.zero_();s.ch_energy.zero_()
        s.hp[b,foe]=30;s.shield.zero_()
        s.ch_hp[b,side,zero]=5;s.ch_shield.zero_();s.ch_base_atk[b,side,zero]=2
        s.hp[0,foe[0]]=1 # lethal player damage must suppress resource award
        s.front[1,foe[1]]=nanali;s.ch_hp[1,foe[1],nanali]=5
        s.hand_kind[2,foe[2],0]=CARD_INDEX['N02'];s.hand_n[2,foe[2]]=1 # response fallback
        s.ch_shape[3,side[3],zero]=CARD_INDEX['Z08'] # equipment fallback
        s.front[4,side[4]]=nanali;s.ch_harmony[4,side[4],nanali]=2 # harmony fallback
        s.front[5,foe[5]]=nanali;s.ch_hp[5,foe[5],nanali]=1 # death fallback
        before=copy.deepcopy(s);expected=copy.deepcopy(s);mask=torch.ones(6,dtype=torch.bool)
        rules.sortie(expected,side,zero,mask)
        with ExitStack() as stack:
            d=stack.enter_context(tempfile.TemporaryDirectory())
            for backend in ['native']+(['cuda'] if torch.cuda.is_available() else []):
                trial=CompiledSortieTrial(d,backend)
                stack.callback(trial.close)
                got=copy.deepcopy(before)
                self.assertEqual(trial.eligible(got,side,zero,mask).tolist(),[True,True,False,False,False,False])
                original_sortie = rules.sortie
                with trial.installed():
                    rules.sortie(got,side,zero,mask)
                self.assertIs(rules.sortie, original_sortie)
                for name in got.__dataclass_fields__:
                    a,e=getattr(got,name),getattr(expected,name)
                    if torch.is_tensor(a):self.assertTrue(torch.equal(a,e),f'{backend}:{name}')
                self.assertEqual(trial.stats['compiled_rows'],2)
                self.assertEqual(trial.stats['fallback_rows'],4)
                # Validate both emitters against tensor interpretation of the SAME IR.
                x=torch.zeros((32,len(trial.program.fields)),dtype=torch.int32)
                index={n:i for i,n in enumerate(trial.program.fields)}
                x[:,index['a_hp']]=5;x[:,index['a_front']]=1;x[:,index['a_atk']]=2;x[:,index['pb_hp']]=30;x[:,index['pa_hp']]=30
                x[:,index['old_front']]=-1;x[:,index['last_front']]=-1
                reference=lower_cpu(trial.program)(x,0)
                device='cuda' if backend=='cuda' else 'cpu';inp=x.to(device);out=torch.empty_like(inp)
                trial.launch(inp,out,1)
                self.assertTrue(torch.equal(out.cpu(),reference))


class IrLayoutValidationTest(unittest.TestCase):
    def test_codegen_rejects_invalid_layout_symbols(self):
        from dataclasses import replace
        from app.modules.card_game.rl.rule_ir.ast import validate
        from app.modules.card_game.rl.rule_ir.ordinary_sortie import ordinary_program
        p = ordinary_program(4)
        for invalid in (replace(p, name='bad;name'), replace(p, fields=p.fields + ('actor',)),
                        replace(p, aliases=(('oops', 'missing', 'a_hp'),))):
            with self.assertRaises(ValueError): validate(invalid)

if __name__=='__main__':unittest.main()
