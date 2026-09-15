"""Resident public-deck sampling and frozen-policy league. No automatic training."""
from copy import deepcopy
from collections import deque

from .episode_return import reset_episode_return


def warm_start_weights(checkpoint, network, deck_id):
    """Transfer v1 semantic input columns; optimizer/history always start fresh."""
    import torch
    from .gpu_duel.catalog import CARD_IDS,SEATS,GPU_LOCK
    from .public_schema import SIDE_FIELDS,CHAR_FIELDS,LEGACY_SIDE,LEGACY_CHAR
    if (checkpoint.get('schema'),checkpoint.get('deck'),checkpoint.get('gpu_lock')) != (
            'resident_public_v1',deck_id,GPU_LOCK):
        raise ValueError('Warm start requires matching legacy preset and card identities')
    def labels(side_fields,char_fields,new):
        keys=[('global','turn'),('global','first')]
        if new:keys += [('global',k) for k in ('escalation','phase','pending_kind','pending_card','pending_count')]
        keys += [('side',k,s) for k in side_fields for s in range(2)]
        keys += [('char',k,s,c) for k in char_fields for s in range(2) for c in SEATS]
        keys += [('zone',z,c) for z in range(5 if new else 3) for c in CARD_IDS]
        return keys
    old_keys=labels(LEGACY_SIDE,LEGACY_CHAR,False)
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
            expanded=torch.zeros_like(weights[key]);expanded[:,:10]=incoming[key].to(expanded.device)
            weights[key]=expanded
        else:weights[key]=incoming[key]
    network.load_state_dict(weights)


class FrozenLeague:
    """A current snapshot plus bounded historical opponents, frozen during each rollout."""
    def __init__(self, max_history=4):
        if max_history < 1:
            raise ValueError('History capacity must be positive')
        self.history = deque(maxlen=max_history)
        self.revision = 0

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
        obs, cand, mask = public_observe(state)
        with torch.no_grad():
            for index, policy in enumerate(self.history, 1):
                selected = (need & (assignment == index)).nonzero().flatten()
                if selected.numel():
                    if not bool(mask[selected].any(1).all()):
                        raise RuntimeError('Opponent has no legal candidates')
                    logits, _ = policy(obs[selected], cand[selected], mask[selected])
                    result[selected] = torch.distributions.Categorical(logits=logits).sample().int()
        return result


def tensor_public_decks(n, device):
    """[n,36] public decks: four ordered seats, then their 32 card kinds."""
    import torch
    from .capability import public_kit_cards
    from .gpu_duel.catalog import SEATS, CARD_INDEX
    seats = torch.rand(n, len(SEATS), device=device).argsort(1)[:, :4]
    kits = torch.tensor([[CARD_INDEX[c] for c in public_kit_cards(cid)] for cid in SEATS],
                        dtype=torch.int32, device=device)
    # Each candidate has two copies; sample eight without replacement per owner.
    slots = torch.rand(n,4,16,device=device).argsort(2)[:,:,:8] % 8
    cards = kits[seats].gather(2, slots).reshape(n,32)
    return torch.cat((seats.int(),cards),1)


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
    def __init__(self, n, *, deck_id, league, device='cuda', compiled_dir=None, learner_deck=None):
        import torch
        from .gpu_duel.compiled_backend import CompiledStarterBackend
        from .gpu_duel.state import empty_state
        self.n, self.device, self.league = n, torch.device(device), league
        self.compiled = CompiledStarterBackend(compiled_dir,
            backend='cuda' if self.device.type=='cuda' else 'native', deck_id=deck_id)
        self.state = empty_state(n, self.device)
        self.fixed = fixed_deck_rows(deck_id,n,self.device,learner_deck)
        self.learner = torch.randint(0,2,(n,),device=self.device,dtype=torch.int32)
        self.assignment = torch.zeros(n,device=self.device,dtype=torch.int32)
        from .training_metrics import TerminalCounters
        self.terminal_counters = TerminalCounters(n, self.device)
        self.reset()

    def reset(self, done=None):
        import torch
        self.terminal_counters.record(self.state, self.learner, self.assignment)
        full_reset = done is None
        if done is None:
            done = torch.ones(self.n,device=self.device,dtype=torch.bool)
        if not bool(done.any()):
            return
        self.terminal_counters.reset(done)
        seed = torch.randint(0,2**30,(self.n,),device=self.device,dtype=torch.int32)
        seed = torch.where(done,seed,-1)
        player = tensor_public_decks(self.n,self.device)
        a = torch.where((self.learner==0)[:,None],self.fixed,player)
        b = torch.where((self.learner==1)[:,None],self.fixed,player)
        self.compiled.reset_public_gpu_state(self.state,seed,a,b,escalation=True)
        reset_episode_return(self, None if full_reset else done)
        # League snapshots update only between rollouts; assignment slots persist until reset.
        assignment = torch.randint(0,len(self.league.history)+1,(self.n,),device=self.device,dtype=torch.int32)
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

    def step_learner(self, actions):
        self.compiled.step_gpu_state(self.state,actions)
        self.prepare_decision()
        self.terminal_counters.record(self.state, self.learner, self.assignment)

    def close(self):
        self.compiled.close()
