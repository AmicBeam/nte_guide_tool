"""Resident public-deck sampling and frozen-policy league. No automatic training."""
from copy import deepcopy
from collections import deque

from .episode_return import reset_episode_return


def warm_start_weights(checkpoint, network, deck_id):
    """Transfer compatible public input columns; optimizer/history start fresh."""
    import torch
    from .gpu_duel.catalog import CARD_IDS,SEATS,GPU_LOCK
    from .public_schema import SIDE_FIELDS,CHAR_FIELDS,LEGACY_SIDE,LEGACY_CHAR
    if (checkpoint.get('deck')!=deck_id or checkpoint.get('gpu_lock')!=GPU_LOCK
            or checkpoint.get('schema') not in ('resident_public_v1','resident_public_v2','resident_public_v3')):
        raise ValueError('Warm start requires matching legacy preset and card identities')
    def labels(side_fields,char_fields,new):
        keys=[('global','turn'),('global','first')]
        if new:keys += [('global',k) for k in ('escalation','phase','pending_kind','pending_card','pending_count')]
        keys += [('side',k,s) for k in side_fields for s in range(2)]
        keys += [('char',k,s,c) for k in char_fields for s in range(2) for c in SEATS]
        keys += [('zone',z,c) for z in range(5 if new else 3) for c in CARD_IDS]
        return keys
    old_keys=labels(SIDE_FIELDS,CHAR_FIELDS,True) if checkpoint['schema']!='resident_public_v1' else labels(LEGACY_SIDE,LEGACY_CHAR,False)
    new_keys=labels(SIDE_FIELDS,CHAR_FIELDS,True)
    incoming=checkpoint['model'];weights=network.state_dict()
    old=incoming['state_net.0.weight']
    if old.shape[1]!=len(old_keys) or old.shape[0]!=weights['state_net.0.weight'].shape[0]:
        raise ValueError('Warm-start network dimensions differ; use the same --hidden size')
    columns={key:i for i,key in enumerate(new_keys)}
    transferred=torch.zeros_like(weights['state_net.0.weight'])
    transferred[:,[columns[key] for key in old_keys]]=old.to(transferred.device)
    for key in weights:
        if key=='state_net.0.weight':weights[key]=transferred
        elif key=='cand_net.0.weight':
            if incoming[key].shape[1]>weights[key].shape[1]:raise ValueError('Cannot shrink candidate schema during warm start')
            expanded=torch.zeros_like(weights[key]);expanded[:,:incoming[key].shape[1]]=incoming[key].to(expanded.device)
            weights[key]=expanded
        else:weights[key]=incoming[key]
    network.load_state_dict(weights)


class FrozenLeague:
    """A current snapshot plus bounded historical opponents, frozen during each rollout."""
    def __init__(self, max_history=4, *, learn_mulligan=False, greedy=False):
        if max_history < 1:
            raise ValueError('History capacity must be positive')
        self.learn_mulligan=learn_mulligan
        self.greedy=greedy
        self.pinned=[]
        self.history = deque(maxlen=max_history)
        self.revision = 0

    @property
    def policies(self):
        return (*self.pinned,*self.history)

    def add_pinned(self,network):
        frozen=deepcopy(network).eval()
        for parameter in frozen.parameters():parameter.requires_grad_(False)
        self.pinned.append(frozen)

    def add(self, network):
        frozen = deepcopy(network).eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        self.history.append(frozen)
        self.revision += 1

    def actions(self, state, need, assignment):
        import torch
        from .public_observation import public_observe
        from .gpu_duel.opponent import rule_action
        result = torch.where(need, rule_action(state), -1)
        from .opening_observation import opening_observe
        result=torch.where(need & (state.phase==0),torch.zeros_like(result),result)
        obs, cand, mask = opening_observe(state) if self.learn_mulligan else public_observe(state)
        with torch.no_grad():
            for index, policy in enumerate(self.policies, 1):
                selected_mask=need & (assignment==index)
                if self.learn_mulligan and not getattr(policy,'_learn_mulligan',True):selected_mask &= state.phase!=0
                selected = selected_mask.nonzero().flatten()
                if selected.numel():
                    if not bool(mask[selected].any(1).all()):
                        raise RuntimeError('Opponent has no legal candidates')
                    logits, _ = policy(obs[selected], cand[selected], mask[selected])
                    result[selected] = (logits.argmax(1) if self.greedy else torch.distributions.Categorical(logits=logits).sample()).int()
        return result


def tensor_decks_for_seats(seats,device):
    import torch
    from .capability import public_kit_cards
    from .gpu_duel.catalog import SEATS,CARD_INDEX
    n=seats.shape[0]
    kits=torch.tensor([[CARD_INDEX[c] for c in public_kit_cards(cid)] for cid in SEATS],dtype=torch.int32,device=device)
    slots=torch.rand(n,4,16,device=device).argsort(2)[:,:,:8] % 8
    return torch.cat((seats.int(),kits[seats.long()].gather(2,slots).reshape(n,32)),1)


def tensor_public_decks(n, device):
    """[n,36] unrestricted public decks; kept for diagnostics/generalization."""
    import torch
    from .gpu_duel.catalog import SEATS
    return tensor_decks_for_seats(torch.rand(n,len(SEATS),device=device).argsort(1)[:,:4],device)


def tensor_preset_opponents(n,device,*,random_fraction=.2):
    """Device-side fixed-role card variation; no per-game Python or action logs."""
    import torch
    from .candidate_opponents import build_opponent_schedule
    from .gpu_duel.catalog import encode_public_deck_row
    if not 0<=random_fraction<=1:raise ValueError('Invalid random-team fraction')
    # Two small, immutable public deck templates per lineup; manifest cache is run-local.
    templates=build_opponent_schedule(2,seed=0,random_fraction=0,rule_fraction=0,model_ids=('learner',))
    rows=torch.tensor([encode_public_deck_row(p['deck']) for p in templates],device=device,dtype=torch.int32).reshape(2,2,36)
    lineup=torch.randint(0,2,(n,),device=device)
    allocation=torch.randint(0,3,(n,),device=device)
    fixed=rows[lineup,allocation.clamp(max=1)]
    varied=tensor_decks_for_seats(fixed[:,:4].long(),device)
    decks=torch.where((allocation==2)[:,None],varied,fixed)
    general=torch.rand(n,device=device)<random_fraction
    decks=torch.where(general[:,None],tensor_public_decks(n,device),decks)
    return decks,torch.where(general,2,lineup)


def training_deck(deck_id, learner_deck=None):
    from .capability import preset_opponent_deck, assert_public_encoder_deck
    preset = preset_opponent_deck(deck_id)
    deck = assert_public_encoder_deck(learner_deck) if learner_deck is not None else preset
    if deck['character_ids'] != preset['character_ids']:
        raise ValueError('Card adaptation cannot change preset characters or their order')
    return deck


def fixed_deck_rows(deck_id, n, device, learner_deck=None):
    import torch
    from .capability import preset_opponent_deck
    from .gpu_duel.catalog import CARD_INDEX, SEAT_INDEX
    deck = training_deck(deck_id, learner_deck)
    row = [SEAT_INDEX[c] for c in deck['character_ids']] + [CARD_INDEX[c] for c in deck['card_ids']]
    return torch.tensor(row,dtype=torch.int32,device=device)[None].expand(n,-1).contiguous()


class PublicTrainingEnv:
    """Small adapter for the PPO collector; all game rows stay on the chosen device."""
    def __init__(self, n, *, deck_id, league, device='cuda', compiled_dir=None, learner_deck=None,
                 learn_mulligan=False,random_team_fraction=.2,rule_fraction=.2, fixed_opponent_deck=None,
                 curriculum_rows=None, curriculum_fraction=0):
        import torch
        from .gpu_duel.compiled_backend import CompiledStarterBackend
        from .gpu_duel.state import empty_state
        self.n, self.device, self.league = n, torch.device(device), league
        if bool(getattr(league,'learn_mulligan',False))!=learn_mulligan:
            raise ValueError('Opening schema mismatch between environment and league')
        self.learn_mulligan=learn_mulligan
        self.random_team_fraction=random_team_fraction
        self.rule_fraction=rule_fraction
        if not 0<=curriculum_fraction<=.5 or (curriculum_fraction and (not curriculum_rows or fixed_opponent_deck is None)):
            raise ValueError("Curriculum requires fixed matchup states and fraction in [0, .5]")
        self.curriculum_rows=curriculum_rows
        self.curriculum_fraction=curriculum_fraction
        self.curriculum_resets=0
        self.fixed_opponent_deck=fixed_opponent_deck
        if fixed_opponent_deck is not None:
            from .capability import assert_public_encoder_deck
            self.fixed_opponent_deck=assert_public_encoder_deck(fixed_opponent_deck)
            if random_team_fraction or rule_fraction or len(league.policies)!=1:
                raise ValueError("Fixed opponent requires exactly one frozen policy and zero random/rule fractions")
        self.opponent_lineup=torch.zeros(n,device=self.device,dtype=torch.int32)
        self.opening_counts=torch.zeros(24,device=self.device,dtype=torch.int64)
        self.first_action_counts=torch.zeros(42,device=self.device,dtype=torch.int64)
        self.first_action_recorded=torch.zeros(n,device=self.device,dtype=torch.bool)
        self.compiled = CompiledStarterBackend(compiled_dir,
            backend='cuda' if self.device.type=='cuda' else 'native', deck_id=deck_id)
        self.state = empty_state(n, self.device)
        self.fixed = fixed_deck_rows(deck_id,n,self.device,learner_deck)
        self.learner = torch.randint(0,2,(n,),device=self.device,dtype=torch.int32)
        self.assignment = torch.zeros(n,device=self.device,dtype=torch.int32)
        from .training_metrics import TerminalCounters,OpponentPoolCounters
        self.terminal_counters = TerminalCounters(n, self.device)
        self.pool_counters=OpponentPoolCounters(n,self.device)
        self.reset()

    def reset(self, done=None):
        import torch
        self.terminal_counters.record(self.state, self.learner, self.assignment)
        self.pool_counters.record(self.state,self.learner,self.opponent_lineup)
        full_reset = done is None
        if done is None:
            done = torch.ones(self.n,device=self.device,dtype=torch.bool)
        if not bool(done.any()):
            return
        self.terminal_counters.reset(done)
        self.pool_counters.reset(done)
        self.first_action_recorded &= ~done
        seed = torch.randint(0,2**30,(self.n,),device=self.device,dtype=torch.int32)
        seed = torch.where(done,seed,-1)
        if self.fixed_opponent_deck is None:
            player,lineup = tensor_preset_opponents(self.n,self.device,random_fraction=self.random_team_fraction)
        else:
            from .gpu_duel.catalog import encode_public_deck_row
            from .capability import preset_opponent_deck
            row=encode_public_deck_row(self.fixed_opponent_deck)
            player=torch.tensor(row,device=self.device,dtype=torch.int32)[None].expand(self.n,-1).contiguous()
            pool=next((i for i,k in enumerate(("starter","weave-rush")) if preset_opponent_deck(k)["character_ids"]==self.fixed_opponent_deck["character_ids"]),2)
            lineup=torch.full((self.n,),pool,device=self.device,dtype=torch.int32)
        self.opponent_lineup=torch.where(done,lineup.int(),self.opponent_lineup)
        a = torch.where((self.learner==0)[:,None],self.fixed,player)
        b = torch.where((self.learner==1)[:,None],self.fixed,player)
        self.compiled.reset_public_gpu_state(self.state,seed,a,b,escalation=True,mulligan=self.learn_mulligan)
        if self.curriculum_fraction:
            from .rule_ir.pack_row import bind_gpu_rows
            selected=done & (torch.rand(self.n,device=self.device)<self.curriculum_fraction)
            indices=selected.nonzero().flatten()
            if indices.numel():
                pool=torch.tensor(self.curriculum_rows,device=self.device,dtype=torch.int32)
                rows=pool[torch.randint(0,len(pool),(len(indices),),device=self.device)]
                bind_gpu_rows(self.state)[indices]=rows
                self.learner[indices]=0  # Reservoir fixes learner to side A, with both initiatives.
                self.curriculum_resets+=len(indices)
                self.compiled.step_gpu_state(self.state,torch.full((self.n,),-1,device=self.device,dtype=torch.int32))
        reset_episode_return(self, None if full_reset else done)
        # League snapshots update only between rollouts; assignment slots persist until reset.
        assignment = torch.randint(1,len(self.league.policies)+1,(self.n,),device=self.device,dtype=torch.int32) if self.league.policies else torch.zeros_like(self.assignment)
        # Prefer a policy trained for the actual fixed opponent lineup when available.
        for pool,key in enumerate(('starter','weave-rush')):
            eligible=[i for i,net in enumerate(self.league.policies,1) if getattr(net,'_trained_deck',None)==key]
            if eligible:
                choices=torch.tensor(eligible,device=self.device,dtype=torch.int32)
                chosen=choices[torch.randint(0,len(eligible),(self.n,),device=self.device)]
                assignment=torch.where(lineup==pool,chosen,assignment)
        assignment=torch.where(torch.rand(self.n,device=self.device)<self.rule_fraction,0,assignment)
        self.assignment = torch.where(done,assignment,self.assignment)
        self.prepare_decision()

    def reset_finished(self):
        self.reset(self.outcome()[0])

    def outcome(self):
        return self.state.phase == 3, self.state.winner

    def prepare_decision(self):
        for _ in range(64):
            need = (self.state.phase != 3) & (self.state.active != self.learner)
            if not bool(need.any()):
                return
            actions = self.league.actions(self.state,need,self.assignment)
            self.compiled.step_gpu_state(self.state,actions)
        raise RuntimeError('Opponent exceeded bounded action window')

    def opening_summary(self):
        return {'learn_mulligan':self.learn_mulligan,'mulligan_by_position_and_count':self.opening_counts.reshape(3,2,4).sum(0).cpu().tolist(),
                'mulligan_by_lineup_position_count':self.opening_counts.reshape(3,2,4).cpu().tolist(),
                'first_action_by_lineup_position_type':self.first_action_counts.reshape(3,2,7).cpu().tolist(),
                'lineups':['starter','weave-rush','public-random'],'positions':['first','second'],
                'action_types':['end_turn','attack','play_card','ultimate','mulligan','choose','concede']}

    def step_learner(self, actions):
        import torch
        batch=torch.arange(self.n,device=self.device)
        index=actions.long()
        kind=self.state.legal_type[batch,index]
        position=(self.learner!=self.state.first).long()
        swapping=self.state.phase==0
        bits=self.state.legal_hand[batch,index]
        count=sum((bits>>i)&1 for i in range(5)).clamp(0,3).long()
        self.opening_counts.scatter_add_(0,self.opponent_lineup.long()*8+position*4+count,swapping.long())
        first=(self.state.phase==1) & (self.state.turn_count[batch,self.learner.long()]==1) & ~self.first_action_recorded
        self.first_action_counts.scatter_add_(0,self.opponent_lineup.long()*14+position*7+kind.clamp(0,6).long(),first.long())
        self.first_action_recorded |= first
        self.compiled.step_gpu_state(self.state,actions)
        self.prepare_decision()
        self.terminal_counters.record(self.state, self.learner, self.assignment)
        self.pool_counters.record(self.state,self.learner,self.opponent_lineup)

    def close(self):
        self.compiled.close()
