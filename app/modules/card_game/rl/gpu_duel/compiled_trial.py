"""Opt-in, process-local compiled ordinary-sortie trial with explicit fallback.

Not installed by website/trainer imports. Only CPU GpuState is supported by the
adapter; backend='cuda' includes packet transfer to test the hybrid architecture.
"""
from contextlib import contextmanager
from unittest.mock import patch
import torch
from . import rules
from .catalog import N_SEATS, SEAT_INDEX
from ..rule_ir.ordinary_sortie import ordinary_program
from ..rule_ir.lower_native import compile_native
from ..rule_ir.lower_cuda import compile_cuda


class CompiledSortieTrial:
    def __init__(self, directory, backend='native'):
        self.program = ordinary_program(N_SEATS)
        self.index = {name: i for i, name in enumerate(self.program.fields)}
        if backend not in ('native', 'cuda'): raise ValueError(backend)
        self.backend = backend
        self.launch = compile_native(self.program, directory) if backend == 'native' else compile_cuda(self.program)
        self.original = rules.sortie
        self.stats = {'ordinary_rows': 0, 'compiled_rows': 0, 'fallback_rows': 0, 'batches': 0}

    def close(self):
        if self.backend == 'native': self.launch.close()

    @contextmanager
    def installed(self):
        # A scoped integration experiment; never enabled globally at import time.
        with patch.object(rules, 'sortie', self.sortie):
            yield self

    def eligible(self, s, side, seat, mask):
        b = torch.arange(s.n); foe = 1-side
        good = mask & rules._live(s) & rules._alive(s, side, seat) & (s.hp > 0).all(1)
        allowed = tuple(SEAT_INDEX[c] for c in ('nanali', 'iloy', 'zero', 'jiuyuan'))
        if seat not in allowed: return torch.zeros_like(mask)
        # Reject special effects, including potential response cards regardless of cost.
        special = (s.ch_shape >= 0).flatten(1).any(1) | (s.ch_awakened > 0).flatten(1).any(1)
        for name in ('ch_next_bonus','ch_next_shield','ch_next_followup','ch_allied_hurt',
                     'ch_pending_atk','ch_share_overflow','extra_flower','extra_genesis','weave','burn_left','delay_end'):
            t = getattr(s,name,None)
            if t is not None: special |= (t != 0).reshape(s.n,-1).any(1)
        hand=s.hand_kind[b,foe]; special |= ((hand>=0)&(s.catalog_response[hand.clamp(min=0)]>0)).any(1)
        special |= s.pending_kind >= 0
        # Existing pending deaths must be settled by the general engine.
        present = getattr(s, 'ch_present', torch.ones_like(s.ch_hp)) > 0
        special |= (present&(s.ch_hp<=0)&(s.ch_down==0)).flatten(1).any(1)
        payer=rules._payer(s,side,seat)
        kind=rules._harmony_kind(s,side,seat,payer)
        good &= ~special & (kind==0)
        target=s.front[b,foe]; safe=target.clamp(min=0); has=target>=0
        allowed_target=torch.zeros_like(mask)
        for i in allowed:allowed_target |= target==i
        good &= ~has | allowed_target
        atk=rules.attack_power(s,side,seat)
        counter=torch.zeros_like(atk)
        for i in range(N_SEATS):counter=torch.where(target==i,rules.attack_power(s,foe,i),counter)
        # Death hooks are not in this common-path program: fall back before mutating.
        good &= (s.ch_hp[b,side,seat]+s.ch_shield[b,side,seat]>counter)
        good &= ~has | (s.ch_hp[b,foe,safe]+s.ch_shield[b,foe,safe]>atk)
        return good

    def sortie(self,s,side,seat,mask,**kwargs):
        if kwargs: return self.original(s,side,seat,mask,**kwargs)  # battle-card/response path
        if s.device.type!='cpu': raise ValueError('Compiled trial adapter requires CPU state')
        self.stats['ordinary_rows'] += int(mask.sum())
        if not bool(mask.any()):return self.original(s,side,seat,mask)
        good=self.eligible(s,side,seat,mask)
        count=int(good.sum());self.stats['compiled_rows']+=count
        self.stats['fallback_rows']+=int(mask.sum())-count
        if count:
            self.stats['batches']+=1
            b=torch.arange(s.n);foe=1-side;target=s.front[b,foe];safe=target.clamp(min=0);has=target>=0
            old=s.front[b,side];old_safe=old.clamp(min=0)
            x=torch.zeros((s.n,len(self.program.fields)),dtype=torch.int32)
            def put(name,value):x[:,self.index[name]]=value
            put('a_hp',s.ch_hp[b,side,seat]);put('a_sh',s.ch_shield[b,side,seat]);put('a_atk',rules.attack_power(s,side,seat))
            put('b_hp',s.ch_hp[b,foe,safe]);put('b_sh',s.ch_shield[b,foe,safe]);counter=torch.zeros(s.n,dtype=torch.int32)
            for i in range(N_SEATS):counter=torch.where(target==i,rules.attack_power(s,foe,i),counter)
            put('b_atk',counter);put('pa_hp',s.hp[b,side]);put('pa_sh',s.shield[b,side]);put('pb_hp',s.hp[b,foe]);put('pb_sh',s.shield[b,foe])
            put('a_front',1);put('b_front',has);put('done',~good);put('winner',-1)
            put('actor',seat);put('old_front',old);put('last_front',s.last_front[b,side]);put('old_front_shield',s.ch_shield[b,side,old_safe]);put('harmony',s.ch_harmony[b,side,seat])
            for i in range(N_SEATS):
                put(f'alive_{i}',(s.ch_hp[b,side,i]>0)&(s.ch_down[b,side,i]==0))
                put(f'energy_{i}',s.ch_energy[b,side,i]);put(f'cap_{i}',s.ch_energy_max[b,side,i])
            if self.backend=='cuda':
                gx=x.cuda();gy=torch.empty_like(gx);self.launch(gx,gy,1);y=gy.cpu()
            else:y=torch.empty_like(x);self.launch(x,y,1)
            get=lambda name:y[:,self.index[name]]
            def assign(t,key,value,mask=good):t[key]=torch.where(mask,value,t[key])
            assign(s.ch_shield,(b,side,old_safe),get('old_front_shield'),good&(old>=0)&(old!=seat))
            assign(s.last_front,(b,side),get('last_front'));assign(s.front,(b,side),torch.full_like(old,seat))
            assign(s.ch_hp,(b,side,seat),get('a_hp'));assign(s.ch_shield,(b,side,seat),get('a_sh'))
            assign(s.ch_hp,(b,foe,safe),get('b_hp'),good&has);assign(s.ch_shield,(b,foe,safe),get('b_sh'),good&has)
            assign(s.hp,(b,foe),get('pb_hp'));assign(s.shield,(b,foe),get('pb_sh'))
            assign(s.ch_harmony,(b,side,seat),get('harmony'))
            for i in range(N_SEATS):assign(s.ch_energy,(b,side,i),get(f'energy_{i}'))
            finished=good&(get('done')>0)
            s.phase=torch.where(finished,3,s.phase);s.winner=torch.where(finished,side,s.winner);s.reason=torch.where(finished,1,s.reason)
        # General execution handles all unsupported rows. No rules are omitted.
        self.original(s,side,seat,mask&~good)
