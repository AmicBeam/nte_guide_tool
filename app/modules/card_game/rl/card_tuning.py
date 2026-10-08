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
    from .opening_observation import OPENING_SCHEMA,OPENING_CAND_DIM
    schema=checkpoint.get('schema')
    if schema==OPENING_SCHEMA and checkpoint.get('opening_policy')!='learned_joint_v1':raise ValueError('Opening policy metadata missing')
    cand_dim=OPENING_CAND_DIM if schema==OPENING_SCHEMA else CAND_DIM
    net = CompactScorer(state_dim(),cand_dim,hidden).to(device)
    net._public_schema=OPENING_SCHEMA if schema==OPENING_SCHEMA else PLAY_SCHEMA
    if checkpoint.get('schema') == 'resident_public_v1':
        warm_start_weights(checkpoint,net,checkpoint['deck'])
        transferred = True
    else:
        if schema not in (PLAY_SCHEMA,OPENING_SCHEMA) or (checkpoint.get('gpu_lock'),checkpoint.get('rule_identity',{}).get('rule_hash')) != (GPU_LOCK,rule_identity(schema)):
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
        self.policy, self.opponents = policy, [policy, *opponents]
        from .opening_observation import OPENING_SCHEMA
        self.learn_mulligan=any(getattr(net,'_public_schema',None)==OPENING_SCHEMA for net in [policy,*opponents])
        self.model_ids = ['learner', *[f'peer-{i}' for i in range(len(opponents))]]
        self.backend = CompiledStarterBackend(compiled_dir,
            backend='cuda' if self.device.type=='cuda' else 'native',deck_id='starter')

    @torch.no_grad()
    def evaluate(self, candidates, *, pairs=4, max_actions=400, stop_check=None, schedule=None,
                 required_lineups=('starter', 'weave-rush')):
        from .public_training import tensor_public_decks
        from .gpu_duel.state import empty_state
        from .public_observation import public_observe
        from .gpu_duel.opponent import rule_action
        if pairs<1 or max_actions<1:raise ValueError('Positive evaluation budgets required')
        if not required_lineups or not set(required_lineups)<= {'starter','weave-rush'} or len(set(required_lineups))!=len(required_lineups):
            raise ValueError('Expected unique public preset lineups')
        n = candidates.shape[0]
        if schedule is None:
            trials=pairs*2
            foes = tensor_public_decks(pairs,self.device).repeat_interleave(2,0)
            seed_base = torch.randint(0,2**29,(pairs,),device=self.device,dtype=torch.int32)*2
            assignment = torch.randint(0,len(self.opponents)+1,(pairs,),device=self.device)
        else:
            from .gpu_duel.catalog import encode_public_deck_row
            if not schedule:raise ValueError('Empty opponent schedule')
            trials=len(schedule)*2
            foes=torch.tensor([encode_public_deck_row(p['deck']) for p in schedule],device=self.device,dtype=torch.int32).repeat_interleave(2,0)
            seed_base=torch.tensor([p['seed'] for p in schedule],device=self.device,dtype=torch.int32)
            if bool((seed_base % 2).any()):raise ValueError('Schedule requires even seed bases')
            assignment=torch.tensor([0 if p['opponent_model']=='rule' else self.model_ids.index(p['opponent_model'])+1
                                     for p in schedule],device=self.device,dtype=torch.int32)
        seeds=(seed_base[:,None]+torch.arange(2,device=self.device,dtype=torch.int32)[None]).flatten()
        assignment=assignment.repeat_interleave(2).repeat(n)
        if n*trials>16384:raise ValueError('Candidate batch exceeds 16384 games; reduce candidates, pairs or random fraction')
        state = empty_state(n*trials,self.device)
        self.backend.reset_public_gpu_state(state,seeds.repeat(n),candidates.repeat_interleave(trials,0),foes.repeat(n,1),mulligan=self.learn_mulligan)
        first = (state.first == 0).reshape(n,trials).clone()
        for _ in range(max_actions):
            if stop_check:stop_check()
            live = state.phase != 3
            if not bool(live.any()):break
            obs,cand,mask = public_observe(state)
            action = torch.where(live,rule_action(state),-1)
            action = torch.where(live & (state.phase==0),0,action)
            from .opening_observation import OPENING_SCHEMA,opening_observe
            opening=opening_observe(state) if self.learn_mulligan else None
            groups=[(self.policy,live & (state.active==0))]
            groups += [(net,live & (state.active==1) & (assignment==i)) for i,net in enumerate(self.opponents,1)]
            for net,selected in groups:
                trained_opening=getattr(net,'_public_schema',None)==OPENING_SCHEMA
                if not trained_opening:selected=selected & (state.phase!=0)
                current_obs,current_cand,current_mask=opening if trained_opening else (obs,cand,mask)
                index=selected.nonzero().flatten()
                if index.numel():
                    if not mask[index].any(1).all():raise RuntimeError('Live game has no legal action')
                    logits,_=net(current_obs[index],current_cand[index],current_mask[index])
                    action[index]=logits.argmax(1).int()
            self.backend.step_gpu_state(state,action)
        done = (state.phase==3).reshape(n,trials)
        win = (state.winner==0).reshape(n,trials) & done
        loss = (state.winner==1).reshape(n,trials) & done
        reward = (win.float()-loss.float()).mean(1)
        result = {'rewards':reward,'valid':done.all(1),'wins':win.sum(1),'losses':loss.sum(1),
                'truncated':(~done).sum(1),'trials':trials,
                'first_wins':(win & first).sum(1),'first_losses':(loss & first).sum(1),
                'first_games':(done & first).sum(1),'second_wins':(win & ~first).sum(1),
                'second_losses':(loss & ~first).sum(1),'second_games':(done & ~first).sum(1),
                'mean_turn':(state.turn.reshape(n,trials).float()*done).sum(1)/done.sum(1).clamp_min(1)}

        if schedule is not None:
            result['by_lineup']={}
            for lineup in sorted({p['lineup'] for p in schedule}):
                columns=torch.tensor([p['lineup']==lineup for p in schedule],device=self.device).repeat_interleave(2)[None]
                group={}
                for position,seat_mask in (('first',first),('second',~first)):
                    mask=columns & seat_mask
                    group[position]={'wins':(win & mask).sum(1),'losses':(loss & mask).sum(1),
                                     'games':(done & mask).sum(1),'draws':(done & ~win & ~loss & mask).sum(1),'truncated':(~done & mask).sum(1)}
                result['by_lineup'][lineup]=group
            result['by_opponent']={}
            for opponent in sorted({p['opponent_model'] for p in schedule}):
                columns=torch.tensor([p['opponent_model']==opponent for p in schedule],device=self.device).repeat_interleave(2)[None]
                result['by_opponent'][opponent]={}
                for position,seat_mask in (('first',first),('second',~first)):
                    mask=columns & seat_mask
                    result['by_opponent'][opponent][position]={'wins':(win & mask).sum(1),'losses':(loss & mask).sum(1),
                        'games':(done & mask).sum(1),'draws':(done & ~win & ~loss & mask).sum(1),'truncated':(~done & mask).sum(1)}
            rates=[];seat_rates=[]
            for lineup in required_lineups:
                group=result['by_lineup'].get(lineup)
                if group is None:raise ValueError('Schedule must cover both public preset lineups')
                rates.append(sum(v['wins'] for v in group.values())/sum(v['games'] for v in group.values()).clamp_min(1))
                seat_rates.extend(v['wins']/v['games'].clamp_min(1) for v in group.values())
            # Equal weight for all four lineup/seat scenarios; no weakest-seat penalty.
            utility=torch.stack(seat_rates).mean(0)
            result['rewards']=2*utility-1
        return result

    def close(self):self.backend.close()
