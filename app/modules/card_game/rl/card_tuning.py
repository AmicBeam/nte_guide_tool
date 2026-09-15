"""Learn card allocation for a user-fixed team; never selects characters."""
from pathlib import Path
import hashlib

import torch
from torch import nn

from .capability import PUBLIC_CHARACTERS, public_kit_cards, assert_public_encoder_deck
from .gpu_duel.catalog import SEAT_INDEX, CARD_INDEX, CARD_IDS, GPU_LOCK

SCHEMA = 'fixed_team_card_tuner_v1'


def validate_team(character_ids):
    ids = list(character_ids)
    if len(ids) != 4 or len(set(ids)) != 4 or not set(ids) <= set(PUBLIC_CHARACTERS):
        raise ValueError('Specify four distinct public characters')
    return ids


class CardTuner(nn.Module):
    """Autoregressive masked policy over 32 card picks, conditioned on prior picks."""
    def __init__(self, character_ids, hidden=64):
        super().__init__()
        self.character_ids = validate_team(character_ids)
        if type(hidden) is not int or not 1 <= hidden <= 1024:
            raise ValueError('Invalid hidden dimension')
        self.hidden = hidden
        self.net = nn.Sequential(nn.Linear(37, hidden), nn.Tanh(), nn.Linear(hidden, 8))
        self.register_buffer('kits', torch.tensor([
            [CARD_INDEX[c] for c in public_kit_cards(cid)] for cid in self.character_ids], dtype=torch.int32),persistent=False)
        self.register_buffer('seats', torch.tensor([SEAT_INDEX[c] for c in self.character_ids],dtype=torch.int32),persistent=False)

    def metadata(self):
        return {'schema':SCHEMA,'character_ids':self.character_ids,'hidden':self.hidden,'gpu_lock':GPU_LOCK}

    def sample(self, n):
        if type(n) is not int or n < 1:
            raise ValueError('Batch size must be positive')
        device = self.seats.device
        counts = torch.zeros(n,4,8,device=device)
        selected, logps, entropies = [], [], []
        batch = torch.arange(n,device=device)
        for owner in range(4):
            owner_features = torch.zeros(n,4,device=device)
            owner_features[:,owner] = 1
            for pick in range(8):
                features = torch.cat((counts.flatten(1)/2,owner_features,
                                      torch.full((n,1),pick/8,device=device)),1)
                logits = self.net(features).masked_fill(counts[:,owner] >= 2, -torch.inf)
                dist = torch.distributions.Categorical(logits=logits)
                choice = dist.sample()
                logps.append(dist.log_prob(choice));entropies.append(dist.entropy())
                selected.append(self.kits[owner,choice])
                # features is a separate cat allocation; later counts cannot mutate its gradient input.
                counts[batch,owner,choice] += 1
        rows = torch.cat((self.seats[None].expand(n,-1),torch.stack(selected,1)),1).contiguous()
        return {'rows':rows,'log_prob':torch.stack(logps).sum(0),'entropy':torch.stack(entropies).mean(0)}

    def deck(self, row, *, name='配牌候选'):
        row = row.detach().cpu().tolist() if torch.is_tensor(row) else list(row)
        if row[:4] != [SEAT_INDEX[c] for c in self.character_ids] or len(row) != 36:
            raise ValueError('Candidate changed the fixed team')
        identifier='tuned-'+hashlib.sha256(','.join(map(str,row)).encode()).hexdigest()[:12]
        return assert_public_encoder_deck({'id':identifier,'name':name,
            'character_ids':self.character_ids,'card_ids':[CARD_IDS[k] for k in row[4:]]})


def update_tuner(tuner, optimizer, sample, rewards, valid, *, entropy_coef=0.01):
    """Only fully finished candidate evaluations contribute terminal rewards."""
    if entropy_coef < 0:raise ValueError('Entropy coefficient must be nonnegative')
    selected = valid.nonzero().flatten()
    if not selected.numel():return {'updated':False,'valid_candidates':0}
    r = rewards[selected].detach()
    baseline = (r.sum()-r)/(len(r)-1) if len(r)>1 else torch.zeros_like(r)
    loss = -((r-baseline)*sample['log_prob'][selected]).mean() - entropy_coef*sample['entropy'][selected].mean()
    if not torch.isfinite(loss):raise ValueError('Non-finite card-tuning loss')
    optimizer.zero_grad();loss.backward()
    nn.utils.clip_grad_norm_(tuner.parameters(),1.0,error_if_nonfinite=True)
    optimizer.step()
    return {'updated':True,'valid_candidates':len(r),'loss':float(loss.detach()),'reward_mean':float(r.mean())}


def load_battle_policy(path, device):
    from .public_schema import SCHEMA as PLAY_SCHEMA,state_dim,CAND_DIM,rule_identity
    from .gpu_duel.ppo import CompactScorer
    from .public_training import warm_start_weights
    path = Path(path)
    checkpoint = torch.load(path,map_location=device,weights_only=True)
    hidden = checkpoint.get('hidden') if isinstance(checkpoint, dict) else None
    if type(hidden) is not int or not 1 <= hidden <= 2048:
        raise ValueError('Invalid battle checkpoint hidden dimension')
    net = CompactScorer(state_dim(),CAND_DIM,hidden).to(device)
    if checkpoint.get('schema') == 'resident_public_v1':
        warm_start_weights(checkpoint,net,checkpoint['deck'])
        transferred = True
    else:
        if (checkpoint.get('schema'),checkpoint.get('gpu_lock'),checkpoint.get('rule_identity',{}).get('rule_hash')) != (PLAY_SCHEMA,GPU_LOCK,rule_identity()):
            raise ValueError('Battle checkpoint schema/catalog/rules mismatch')
        net.load_state_dict(checkpoint['model']);transferred = False
    net.eval()
    for parameter in net.parameters():
        if not torch.isfinite(parameter).all():raise ValueError('Non-finite battle policy')
        parameter.requires_grad_(False)
    return net, {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'legacy_transfer':transferred}


class CardBattleEvaluator:
    """Frozen play policies and a common opponent pool score each candidate batch."""
    def __init__(self, policy, *, opponents=(), device='cuda', compiled_dir=None):
        from .gpu_duel.compiled_backend import CompiledStarterBackend
        self.device = torch.device(device)
        self.policy, self.opponents = policy, list(opponents) or [policy]
        self.backend = CompiledStarterBackend(compiled_dir,
            backend='cuda' if self.device.type=='cuda' else 'native',deck_id='starter')

    @torch.no_grad()
    def evaluate(self, candidates, *, pairs=4, max_actions=400, stop_check=None):
        from .public_training import tensor_public_decks
        from .gpu_duel.state import empty_state
        from .public_observation import public_observe
        from .gpu_duel.opponent import rule_action
        if pairs<1 or max_actions<1:raise ValueError('Positive evaluation budgets required')
        n = candidates.shape[0];trials = pairs*2;total = n*trials
        state = empty_state(total,self.device)
        foes = tensor_public_decks(pairs,self.device).repeat_interleave(2,0)
        seeds = torch.randint(0,2**29,(pairs,),device=self.device,dtype=torch.int32)*2
        seeds = (seeds[:,None]+torch.arange(2,device=self.device,dtype=torch.int32)[None]).flatten()
        assignment = torch.randint(0,len(self.opponents)+1,(pairs,),device=self.device).repeat_interleave(2).repeat(n)
        self.backend.reset_public_gpu_state(state,seeds.repeat(n),candidates.repeat_interleave(trials,0),foes.repeat(n,1))
        for _ in range(max_actions):
            if stop_check:stop_check()
            live = state.phase != 3
            if not bool(live.any()):break
            obs,cand,mask = public_observe(state)
            action = torch.where(live,rule_action(state),-1)
            groups=[(self.policy,live & (state.active==0))]
            groups += [(net,live & (state.active==1) & (assignment==i)) for i,net in enumerate(self.opponents,1)]
            for net,selected in groups:
                index=selected.nonzero().flatten()
                if index.numel():
                    if not mask[index].any(1).all():raise RuntimeError('Live game has no legal action')
                    logits,_=net(obs[index],cand[index],mask[index])
                    action[index]=logits.argmax(1).int()
            self.backend.step_gpu_state(state,action)
        done = (state.phase==3).reshape(n,trials)
        win = (state.winner==0).reshape(n,trials) & done
        loss = (state.winner==1).reshape(n,trials) & done
        reward = (win.float()-loss.float()).mean(1)
        return {'rewards':reward,'valid':done.all(1),'wins':win.sum(1),'losses':loss.sum(1),
                'truncated':(~done).sum(1),'trials':trials}

    def close(self):self.backend.close()
