"""Exact resident observation/candidate equality for proposed sampled worlds.

This checker does not sample, execute physical rules, or qualify H3. The numeric
sampler's default gate remains closed. All candidate windows are checked; window
size is a memory bound, never an action pruning or search parameter.
"""
from dataclasses import dataclass, replace
from .domain_soa_actions import DomainSoAActions
from .domain_soa_candidates import DomainSoACandidates
from .domain_soa_observation import PI

supports_full_duel_rules = False
INFORMATION_CHANGED = 256


@dataclass(frozen=True)
class PreparedInformationGuard:
    owner: object
    before: dict
    after: dict
    working: dict
    active: object
    projection: object
    saved: dict
    source_candidates: object
    sample_candidates: object
    source_turns: object
    sample_turns: object
    errors: object
    accepted: object


class DomainInformationGuard:
    """Check seven_sampler's complete model-input invariant without host rules."""
    supports_full_duel_rules = False
    gpu_qualified = False

    def __init__(self, layout=None, *, device='cuda', allow_cpu_oracle=False,
                 max_actions=800, window=16):
        if type(window) is not int or not 1 <= window <= max_actions or max_actions % window:
            raise ValueError('Candidate window must exactly partition the action capacity')
        self.actions = DomainSoAActions(layout, device=device,
            allow_cpu_oracle=allow_cpu_oracle, max_actions=max_actions)
        self.candidates = DomainSoACandidates(device=self.actions.device,
            allow_cpu_oracle=allow_cpu_oracle)
        self.t = self.actions.t
        self.layout = self.actions.layout
        self.device = self.actions.device
        self.max_actions = max_actions
        self.window = window
        self.closed = False

    def prepare(self, before, after, viewers, active):
        t = self.t
        # The action backend validates every named column and device binding.
        working = {name: value.clone(memory_format=t.contiguous_format) for name,value in before.items()}
        projection = self.actions.prepare(working, viewers, active)
        if set(after) != set(working):
            raise ValueError('Sample columns must match the complete source schema')
        for name in self.actions.public.observation.specs:
            self.actions.public.observation.partial._column(after,name,viewers.shape[0])
        parent = projection.playing
        B = viewers.shape[0]
        saved = dict(actions=t.full_like(parent.actions,-1),
            selected=t.full_like(projection.selected,-1), top=t.full_like(projection.top,-1),
            bottom=t.full_like(projection.bottom,-1), counts=t.zeros_like(parent.counts),
            cards=t.zeros_like(parent.public.cards),
            observation=t.zeros_like(parent.public.observation.features))
        source_turns = t.zeros(B,dtype=t.int64,device=self.device)
        sample_turns = t.zeros_like(source_turns)
        errors = t.zeros(B,dtype=t.int32,device=self.device)
        shared = dict(actions=saved['actions'],selected=saved['selected'],top=saved['top'],bottom=saved['bottom'],
            counts=saved['counts'],viewers=viewers,active=active,prior_errors=errors,action_width=self.window)
        src = self.candidates.prepare(cards=saved['cards'],turn_counts=source_turns,**shared)
        dst = self.candidates.prepare(cards=parent.public.cards,turn_counts=sample_turns,**shared)
        return PreparedInformationGuard(self,dict(before),dict(after),working,active,projection,saved,
            src,dst,source_turns,sample_turns,errors,t.zeros(B,dtype=t.bool,device=self.device))

    def launch(self, p):
        if self.closed or not isinstance(p,PreparedInformationGuard) or p.owner is not self:
            raise ValueError('Closed or foreign information-guard buffers')
        t = self.t
        for name,value in p.working.items(): value.copy_(p.before[name])
        original = self.actions.launch(p.projection)
        for key in ('actions','selected','top','bottom','counts'):
            p.saved[key].copy_(original[key])
        p.saved['cards'].copy_(original['public_cards'])
        p.saved['observation'].copy_(original['observation'])
        rows=t.arange(p.active.shape[0],device=self.device)
        viewers=p.projection.playing.public.observation.viewers.clamp(0,1)
        p.source_turns.copy_(p.working['player_int_value'][rows,viewers,PI['turn_count']].long())
        p.errors.copy_(original['errors'])
        for name,value in p.working.items(): value.copy_(p.after[name])
        sampled = self.actions.launch(p.projection)
        p.sample_turns.copy_(p.working['player_int_value'][rows,viewers,PI['turn_count']].long())
        p.errors.bitwise_or_(sampled['errors'])
        changed=(p.saved['observation']!=sampled['observation']).any(-1)
        # Encode the ORIGINAL actions against the sampled public card table,
        # exactly as the formal guard does. Do not replace them by sampled actions.
        for offset in range(0,self.max_actions,self.window):
            source=self.candidates.launch(replace(p.source_candidates,action_start=offset))
            sample=self.candidates.launch(replace(p.sample_candidates,action_start=offset))
            p.errors.bitwise_or_(source['errors']|sample['errors'])
            changed |= (source['features']!=sample['features']).flatten(1).any(-1)
        p.errors.copy_(t.where(changed&p.active&(p.errors==0),INFORMATION_CHANGED,p.errors))
        p.errors.masked_fill_(~p.active,0)
        p.accepted.copy_(p.active&(p.errors==0))
        return dict(errors=p.errors,accepted=p.accepted,full_information_set_guard=True,
            full_duel_qualified=False,sampler_qualified=False,physical_rules_available=False)

    def close(self):
        if not self.closed:
            self.candidates.close(); self.actions.close(); self.closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
