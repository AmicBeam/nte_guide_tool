"""VM-free CUDA experiment for a guarded, real starter-team combat family.

This is a numeric vertical slice, NOT a replacement for simulate_action. Legal
actions are complete for admitted playing roots. Attack output covers the named
combat fields below, including rout-generated hand counts; it does not emit card
entities, events, action identities or replay records. Never feed it to training
as a complete transition. Unsupported states/actions fail explicitly, without a
Python rule fallback. Packing reads raw state; eligibility and combat run in C++.
"""
from dataclasses import dataclass
import numpy as np

ROSTER = ('nanali', 'iloy', 'zero', 'jiuyuan')
supports_full_duel_rules = False
# Field-major [FIELDS, batch] int32. Character addresses include side, so enemy
# and friendly instances of the same template never alias.
GLOBAL = ('turn', 'escalation', 'actor', 'finished', 'winner')
TEAM = ('hp', 'shield', 'ap', 'normal', 'ultimate_available', 'ultimates_used',
        'seal', 'front', 'last_front', 'family', 'hand_count')
CHAR = ('hp', 'shield', 'energy', 'harmony', 'down', 'attack', 'max_hp', 'ultimate_used_turn')
TF = len(GLOBAL)
CF = TF + 2 * len(TEAM)
FIELDS = CF + 8 * len(CHAR)
LEGAL_WIDTH = 10
ILLEGAL_ACTION, UNSUPPORTED_HARMONY, INVALID_STATE = 1, 2, 4


class UnsupportedNativeAttack(ValueError):
    pass


def _integer(value, label, *, low=0, high=1_000_000):
    if type(value) is not int or not low <= value <= high:
        raise UnsupportedNativeAttack(label + ': bounded exact integer required')
    return value


def pack_roots(states):
    """Validate the entire batch before upload; never modify input snapshots."""
    from app.modules.card_game.engine.duel_v2.state import new_character
    if not states or len(states) > 4096:
        raise UnsupportedNativeAttack('Batch size must be 1..4096')
    rows = []
    templates = {cid: new_character(cid) for cid in ROSTER}
    for template in templates.values():
        template['effects'] = []
    top_known = {'rules_version', 'design_version', 'version', 'phase', 'active_side',
        'first_side', 'turn', 'winner', 'reason', 'rng', 'next_instance', 'event_seq',
        'pending_choice', 'operation', 'mulligan_done', 'current_action_id',
        'current_action_version', 'combat_group', 'escalation_enabled',
        'harmony_residue_version', 'next_entity', 'sides', 'events', '_public_board',
        'deferred_actions'}
    team_known = {'name', 'hp', 'max_hp', 'shield', 'ap', 'normal_attack_available',
        'ultimate_available', 'front', 'last_front', 'order', 'records', 'turn_count',
        'fatigue', 'used', 'harmonized', 'extra_flower', 'extra_delay', 'front_debuff',
        'entity_id', 'side', 'effects', 'attacked_this_turn', 'harmony_damage',
        'genesis_damage_bonus', 'surplus', 'nanali_family', 'nanali_family_played',
        'ultimates_used', 'hand', 'deck', 'discard', 'removed', 'seal_normal_attack', 'characters'}
    dynamic = {'hp', 'shield', 'energy', 'harmony', 'down_turns', 'entity_id', 'side',
        'ultimate_used_turn', 'last_hit_source', 'shield_until'}
    for state in states:
        if (set(state) - top_known or state.get('phase') != 'playing'
                or state.get('active_side') not in ('a', 'b') or state.get('winner') is not None
                or state.get('pending_choice') or state.get('operation')
                or state.get('deferred_actions') or state.get('harmony_residue_version') != 1):
            raise UnsupportedNativeAttack('Only ordinary playing roots without pending mechanics')
        turn = _integer(state['turn'], 'turn', low=1)
        if type(state.get('escalation_enabled')) is not bool:
            raise UnsupportedNativeAttack('Explicit escalation flag required')
        row = [turn, int(state['escalation_enabled']), ('a', 'b').index(state['active_side']), 0, -1]
        chars = []
        for side in ('a', 'b'):
            team = state['sides'][side]
            if (set(team) - team_known or set(team['characters']) != set(ROSTER)
                    or tuple(team.get('order', ())) != ROSTER or team['hand']
                    or team.get('effects') or team.get('front_debuff') or team.get('records')
                    or team.get('extra_flower') or team.get('extra_delay')):
                raise UnsupportedNativeAttack('Starter roster, empty hands and no extra rule effects required')
            values = [_integer(team[k], side + '.' + k) for k in ('hp', 'shield', 'ap')]
            if values[0] == 0:
                raise UnsupportedNativeAttack('Playing root has defeated player')
            for k in ('normal_attack_available', 'ultimate_available'):
                if type(team[k]) is not bool:
                    raise UnsupportedNativeAttack('Exact availability booleans required')
                values.append(int(team[k]))
            values += [_integer(team.get('ultimates_used', 0), 'ultimates_used'),
                       int(bool(team.get('seal_normal_attack', False)))]
            for k in ('front', 'last_front'):
                cid = team.get(k)
                if cid is not None and cid not in ROSTER:
                    raise UnsupportedNativeAttack('Unknown front identity')
                values.append(-1 if cid is None else ROSTER.index(cid))
            values += [_integer(team.get('nanali_family', 0), 'family', high=3), 0]
            row.extend(values)
            for cid in ROSTER:
                h, template = team['characters'][cid], templates[cid]
                # Every non-dynamic field must equal a genuinely unmodified kit.
                # Unknown flags, modifiers, equipment, summons, tutorial overrides,
                # legacy buffs and future metadata are rejected rather than ignored.
                if set(h) - set(template) - dynamic or any(
                        h.get(k) != value for k, value in template.items() if k not in dynamic):
                    raise UnsupportedNativeAttack('Unsupported character modifier or kit state: ' + cid)
                hp = _integer(h['hp'], 'hp')
                down = _integer(h['down_turns'], 'down_turns', high=3)
                if bool(down) != (hp == 0) or hp > h['max_hp']:
                    raise UnsupportedNativeAttack('Inconsistent life/down state')
                chars.extend([hp, _integer(h['shield'], 'shield'),
                    _integer(h['energy'], 'energy', high=6),
                    _integer(h['harmony'], 'harmony', high=2), down,
                    _integer(h['base_stats']['attack'], 'attack'),
                    _integer(h['max_hp'], 'max_hp'),
                    _integer(h.get('ultimate_used_turn', -1), 'ultimate_used_turn', low=-1)])
            for k in ('front', 'last_front'):
                cid = team.get(k)
                if cid and team['characters'][cid]['hp'] <= 0:
                    raise UnsupportedNativeAttack('Dead front/source')
        rows.append(row + chars)
    return np.ascontiguousarray(np.asarray(rows, dtype=np.int32).T)


def decode_legal(tokens):
    """Representation conversion only; no legality decisions on the host."""
    out = []
    for token in tokens:
        token = int(token)
        if token == -1:
            break
        if token in (0, 1):
            out.append({'type': ('concede', 'end_turn')[token]})
        elif 2 <= token < 10:
            out.append({'type': 'attack' if token % 2 == 0 else 'ultimate',
                        'character_id': ROSTER[(token - 2) // 2]})
        else:
            raise ValueError('Invalid device legal token')
    return out


def combat_projection(state):
    """Independent oracle reads only; not used by packing or device execution.

    Compared fields deliberately exclude identities, events and generated-card
    bodies. The latter require a future native entity/queue implementation.
    """
    result = [int(state['phase'] == 'finished'),
              {None: -1, 'a': 0, 'b': 1, 'draw': 2}[state['winner']]]
    for side in ('a', 'b'):
        t = state['sides'][side]
        result += [t['hp'], t['shield'], t['ap'], int(t['normal_attack_available']),
                   -1 if t['front'] is None else ROSTER.index(t['front']),
                   -1 if t['last_front'] is None else ROSTER.index(t['last_front']),
                   t.get('nanali_family', 0), len(t['hand'])]
        for cid in ROSTER:
            h = t['characters'][cid]
            result += [h['hp'], h['shield'], h['energy'], h['harmony'], h['down_turns']]
    return result


def output_projection(column):
    v = list(map(int, column))
    result = v[3:5]
    for side in range(2):
        t = TF + side * len(TEAM)
        result += [v[t + k] for k in (0, 1, 2, 3, 7, 8, 9, 10)]
        for cid in range(4):
            c = CF + (side * 4 + cid) * len(CHAR)
            result += v[c:c + 5]
    return result


def emit_source():
    """One shared arithmetic body for native CPU oracle tests and real CUDA."""
    return r'''
#ifdef __CUDACC__
#define SLICE_DEVICE __device__
#else
#define SLICE_DEVICE
#endif
SLICE_DEVICE int lesser(int a,int b){return a<b?a:b;}
SLICE_DEVICE void slice_row(const int* input,int* output,int* legal,int* status,
                           int rows,int row,int action){
 const int F=91,T=5,C=27,TS=11,CS=8;
 int v[F];for(int f=0;f<F;++f)v[f]=input[f*rows+row];
 for(int k=0;k<10;++k)legal[k*rows+row]=-1;
 bool invalid=v[0]<1||v[1]<0||v[1]>1||v[2]<0||v[2]>1||v[3]!=0||v[4]!=-1;
 for(int f=0;f<F;++f)if(v[f]<-1||v[f]>1000000)invalid=true;
 for(int s=0;s<2;++s){int t=T+s*TS;
  if(v[t]<=0||v[t+1]<0||v[t+2]<0||v[t+3]<0||v[t+3]>1||
     v[t+4]<0||v[t+4]>1||v[t+5]<0||v[t+6]<0||v[t+6]>1||
     v[t+7]<-1||v[t+7]>3||v[t+8]<-1||v[t+8]>3||
     v[t+9]<0||v[t+9]>3||v[t+10]!=0)invalid=true;
  for(int k=0;k<4;++k){int c=C+(s*4+k)*CS;
   if(v[c]<0||v[c+1]<0||v[c+2]<0||v[c+2]>6||v[c+3]<0||v[c+3]>2||
      v[c+4]<0||v[c+4]>3||v[c+5]<0||v[c+6]<1||v[c]>v[c+6]||
      ((v[c]==0)!=(v[c+4]>0)))invalid=true;
  }
  if(v[t+7]>=0&&v[t+7]<4&&v[C+(s*4+v[t+7])*CS]<=0)invalid=true;
  if(v[t+8]>=0&&v[t+8]<4&&v[C+(s*4+v[t+8])*CS]<=0)invalid=true;
 }
 if(invalid){status[row]=4;
  for(int f=0;f<F;++f)output[f*rows+row]=input[f*rows+row];return;
 }
 int actor=v[2],enemy=1-actor,t=T+actor*TS,e=T+enemy*TS;
 int lowered=v[1]&&v[0]>=13,ecap=5-lowered,hcap=2-lowered;
 int ulimit=(v[1]&&v[0]>=6)?2:1,n=0;
 legal[n++*rows+row]=0;legal[n++*rows+row]=1;
 bool can_attack=v[t+3]&&v[t+2]>=1&&!v[t+6];
 for(int k=0;k<4;++k){int c=C+(actor*4+k)*CS;
  if(v[c]<=0)continue;
  if(can_attack)legal[n++*rows+row]=2+2*k;
  if(v[t+4]&&v[t+5]<ulimit&&v[c+2]>=ecap&&v[c+7]!=v[0])
   legal[n++*rows+row]=3+2*k;
 }
 int error=0;
 if(action!=-1){
  if(action<0||action>3||!can_attack||v[C+(actor*4+action)*CS]<=0)error=1;
  else{
   int a=C+(actor*4+action)*CS,front=v[t+7];
   int payer=front>=0?front:v[t+8];
   if(front!=action&&payer>=0&&payer!=action&&
       v[C+(actor*4+payer)*CS]>0&&v[C+(actor*4+payer)*CS+3]>=hcap)error=2;
   if(!error){
    v[t+2]-=1;v[t+3]=0;
    if(front!=action){
     if(front>=0)v[C+(actor*4+front)*CS+1]=0;
     v[t+7]=action;v[t+8]=-1;
    }
    int target=v[e+7],d=target<0?e:C+(enemy*4+target)*CS;
    int atk=v[a+5],counter=target<0?0:v[d+5];
    int absorbed=lesser(v[d+1],atk);v[d+1]-=absorbed;
    v[d]=v[d]-(atk-absorbed);if(v[d]<0)v[d]=0;
    if(target>=0){
     absorbed=lesser(v[a+1],counter);v[a+1]-=absorbed;
     v[a]=v[a]-(counter-absorbed);if(v[a]<0)v[a]=0;
     // Both damage values were locked before either knockdown.
     for(int s=0;s<2;++s){int own=T+s*TS,foe=T+(1-s)*TS;
      int dead=(s==actor)?action:target,live=(s==actor)?target:action;
      int dc=C+(s*4+dead)*CS,lc=C+((1-s)*4+live)*CS;
      if(v[dc]==0){
       if(v[own+7]==dead)v[own+7]=-1;
       if(v[own+8]==dead)v[own+8]=-1;
       v[dc+1]=0;v[dc+4]=3;
       // Nanali's real rout passive also fires on a surviving counter kill.
       if(live==0&&v[lc]>0&&v[foe+7]==live&&v[foe+9]<3){
        ++v[foe+9];++v[foe+10];
       }
      }
     }
    }
    if(v[e]<=0){v[3]=1;v[4]=actor;}
    // Terminal operations and dead attackers gain no basic resources.
    if(!v[3]&&v[a]>0){
     for(int k=0;k<4;++k){int c=C+(actor*4+k)*CS;
      if(v[c]>0)v[c+2]=lesser(ecap,v[c+2]+1);
     }
     v[a+2]=lesser(ecap,v[a+2]+1);
     v[a+3]=lesser(hcap,v[a+3]+1);
    }
   }
  }
 }
 status[row]=error;
 // Atomic rejection: unsupported or illegal lanes keep every input field.
 for(int f=0;f<F;++f)output[f*rows+row]=error?input[f*rows+row]:v[f];
}
#ifdef __CUDACC__
extern "C" __global__ void native_attack_slice(const int* input,int* output,
 int* legal,int* status,const int* actions,int rows){
 int row=blockIdx.x*blockDim.x+threadIdx.x;
 if(row<rows)slice_row(input,output,legal,status,rows,row,actions[row]);
}
#else
extern "C" void native_attack_slice(const int* input,int* output,
 int* legal,int* status,const int* actions,int rows){
 for(int row=0;row<rows;++row)slice_row(input,output,legal,status,rows,row,actions[row]);
}
#endif
'''


@dataclass
class ResidentAttackBatch:
    inputs: object
    outputs: object
    legal: object
    status: object
    actions: object


class NativeAttackSlice:
    """Tiny standalone CUDA module: no VM heap, IR, frames or enlarged stack."""
    def __init__(self, device='cuda:0'):
        from .cuda_runtime import CudaModule
        if FIELDS != 91:
            raise RuntimeError('Native slice layout changed')
        self.module = CudaModule(emit_source(), {'native_attack_slice':
            ('const int32*', 'int32*', 'int32*', 'int32*', 'const int32*', 'int32')}, device)

    def upload(self, packed, actions):
        import torch
        if (not isinstance(packed, np.ndarray) or packed.dtype != np.int32
                or packed.ndim != 2 or packed.shape[0] != FIELDS
                or not packed.flags.c_contiguous or not 1 <= packed.shape[1] <= 4096):
            raise ValueError('Packed field-major batch required')
        a = np.asarray(actions)
        if a.shape != (packed.shape[1],) or a.dtype != np.int32 or np.any((a < -1) | (a > 3)):
            raise ValueError('Exact int32 attack slots or -1 legality-only required')
        x = torch.as_tensor(packed, device=self.module.device)
        return ResidentAttackBatch(x, torch.empty_like(x),
            torch.empty((LEGAL_WIDTH, x.shape[1]), dtype=torch.int32, device=x.device),
            torch.empty(x.shape[1], dtype=torch.int32, device=x.device),
            torch.as_tensor(a, device=x.device))

    def launch(self, batch):
        rows = batch.inputs.shape[1]
        return self.module.launch('native_attack_slice',
            (batch.inputs, batch.outputs, batch.legal, batch.status, batch.actions, rows),
            grid=(rows + 127) // 128, block=128)

    def close(self):
        self.module.close()
