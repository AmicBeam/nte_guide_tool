"""Preallocated GPU ranking for the canonical covered teacher.

Splits expensive teacher selection after battle previews exist: kind codes,
combat mask and the four exact preview scalars stay on device. This is not a
closed search service, not a feature encoder, and not a gameplay shortcut.
Identity is derived from VisibleEngineRulePolicy plus teacher prepare/finish;
unknown ABI or unsupported source is rejected rather than marked ready.
"""
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from app.modules.card_game.rl.contracts import ACTION_TYPES


CLOSED_SERVICE_READY = False
ERR_COUNT, ERR_MASK, ERR_NONFINITE, ERR_EMPTY, ERR_KIND, ERR_BATTLE = 1, 2, 4, 8, 16, 32
KIND_NAMES = ACTION_TYPES + ('choose_cards', 'choose_deck_order', 'exit_ultimate')
_CANONICAL_AST_SHA256 = '5e003f26d30ecee742ecfe107d61d844ae352dd682e3762115fbbab29496c49a'
KIND_CODES = {name: index for index, name in enumerate(KIND_NAMES)}
KIND_MULLIGAN = KIND_CODES['mulligan']
KIND_ATTACK = KIND_CODES['attack']
KIND_PLAY = KIND_CODES['play_card']
KIND_ULTIMATE = KIND_CODES['ultimate']
KIND_CYCLE = KIND_CODES['cycle']
KIND_CHOOSE = KIND_CODES['choose']
KIND_END = KIND_CODES['end_turn']
KIND_CONCEDE = KIND_CODES['concede']

_KERNEL = r'''
__device__ inline int finite_d(double x){
 unsigned long long bits=__double_as_longlong(x);
 return ((bits>>52)&0x7ffULL)!=0x7ffULL;
}
__device__ inline double py_or_d(double x,double fallback){
 unsigned long long bits=__double_as_longlong(x);
 return (bits<<1)==0ULL?fallback:x;
}
extern "C" __global__ void rank_teacher(
 const long long* kinds,const unsigned char* battle,const unsigned char* valid,
 const long long* counts,const unsigned char* active,const int* input_errors,
 const double* attack,const double* attack_count,const double* genesis,const double* counter,
 long long* choices,double* scores,int* errors,int rows,int max_actions,
 int k_mulligan,int k_choose,int k_concede,int k_end,int k_ultimate,int k_cycle,
 int k_attack,int k_play,int num_kinds){
 int row=blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=rows) return;
 long long base=(long long)row*max_actions;
 choices[row]=-1; errors[row]=0;
 for(int j=0;j<max_actions;j++) scores[base+j]=0.0;
 if(!active[row]) return;
 errors[row]=input_errors[row];
 if(errors[row]) return;
 long long n=counts[row];
 if(n<0||n>max_actions){ errors[row]|=1; return; }
 if(n<1){ errors[row]|=8; return; }
 int mask_bad=0, battle_bad=0, battle_padding_bad=0, kind_bad=0;
 for(int j=0;j<max_actions;j++){
  unsigned char v=valid[base+j];
  unsigned char btl=battle[base+j];
  if(((unsigned)j<(unsigned)n)!=(v!=0)) mask_bad=1;
  if(btl && !v) battle_padding_bad=1;
  if(v){
   long long kind=kinds[base+j];
   if(kind<0||kind>=num_kinds) kind_bad=1;
   if(btl && kind!=k_attack && kind!=k_play) battle_bad=1;
  }
 }
 if(mask_bad){ errors[row]|=2; return; }
 if(battle_padding_bad){ errors[row]|=32; return; }
 if(kind_bad){ errors[row]|=16; return; }
 if(battle_bad){ errors[row]|=32; return; }
 for(int j=0;j<n;j++) if(kinds[base+j]==k_mulligan){ choices[row]=j; return; }
 for(int j=0;j<n;j++) if(kinds[base+j]==k_choose){ choices[row]=j; return; }
 int nonfinite=0; int best=-1; double best_score=0.0;
 for(int j=0;j<n;j++){
  long long kind=kinds[base+j];
  int combat=(kind==k_attack)||(kind==k_play && battle[base+j]);
  double s;
  if(kind==k_concede) s=-1000.0;
  else if(kind==k_end) s=-100.0;
  else if(kind==k_ultimate) s=6.0;
  else if(kind==k_cycle) s=1.0;
  else if(combat){
   double a=attack[base+j], ac=attack_count[base+j], g=genesis[base+j], c=counter[base+j];
   if(!finite_d(a)||!finite_d(ac)||!finite_d(g)||!finite_d(c)){ nonfinite=1; s=0.0; }
   else { s=py_or_d(a,0.0)*py_or_d(ac,1.0)+py_or_d(g,0.0)-py_or_d(c,0.0)*0.4+3.0;
    if(!finite_d(s)) nonfinite=1; }
  } else s=2.0;
  scores[base+j]=s;
  if(!nonfinite && (best<0 || s>best_score || (s==best_score && j<best))){
   best=j; best_score=s;
  }
 }
 if(nonfinite){ errors[row]|=4; choices[row]=-1; return; }
 choices[row]=best;
}
'''


@dataclass(frozen=True)
class DeviceTeacherRankingResult:
    choices: Any
    scores: Any
    errors: Any
    identity: str


def _canonical_ast_dump(tree):
    """Preserve the pinned 3.10 AST across empty-field changes in 3.12/3.13."""
    import ast
    import inspect
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if getattr(node, 'type_params', ()):
                raise RuntimeError('Unsupported canonical teacher type parameters')
            # Empty generic metadata was added in 3.12; nonempty metadata is
            # rejected above, so removing this field loses no source semantics.
            node._fields = tuple(name for name in node._fields if name != 'type_params')
    # Python 3.13 omits empty lists by default. Older dumps included them,
    # which is the representation authenticated by the existing source pin.
    kwargs = {'show_empty': True} if 'show_empty' in inspect.signature(ast.dump).parameters else {}
    return ast.dump(tree, include_attributes=False, **kwargs)


def canonical_teacher_ranking_identity():
    """Bind the entire known canonical AST, not matching source fragments."""
    import ast
    import inspect
    from app.modules.card_game.engine.duel_v2.projection import RULE_PREVIEW_SCORE_FIELDS
    from app.modules.card_game.rl.batched_duel.gpu_teacher_preview_parts import (
        teacher_finish, teacher_prepare)
    from app.modules.card_game.rl.policies import VisibleEngineRulePolicy, _action_key
    if RULE_PREVIEW_SCORE_FIELDS != ('attack', '攻击次数', '创生花伤害', 'counter'):
        raise RuntimeError('Unsupported canonical preview score ABI')
    if ACTION_TYPES != ('mulligan', 'attack', 'play_card', 'ultimate', 'cycle',
                        'choose', 'end_turn', 'concede'):
        raise RuntimeError('Unsupported canonical action-kind ABI')
    try:
        source = '\n'.join(_canonical_ast_dump(ast.parse(inspect.getsource(obj)))
                           for obj in (VisibleEngineRulePolicy, _action_key,
                                       teacher_prepare, teacher_finish))
    except (OSError, TypeError, SyntaxError) as exc:
        raise RuntimeError('Unsupported canonical teacher ranking source') from exc
    canonical = sha256((source + repr(RULE_PREVIEW_SCORE_FIELDS)).encode()).hexdigest()
    if canonical != _CANONICAL_AST_SHA256:
        raise RuntimeError('Unsupported canonical teacher ranking source')
    return sha256((canonical + repr(KIND_NAMES) + _KERNEL).encode()).hexdigest()


class DeviceTeacherRanking:
    """Typed CUDA batch ranking over already-visible legal candidates."""

    closed_service_ready = False

    def __init__(self, *, rows, max_actions, device='cuda', allow_cpu_reference=False):
        import torch as t
        if type(allow_cpu_reference) is not bool:
            raise TypeError('Explicit allow_cpu_reference boolean required')
        if any(type(value) is not int or value < 1 for value in (rows, max_actions)):
            raise ValueError('Positive bounded ranking dimensions required')
        self.identity = canonical_teacher_ranking_identity()
        self.t = t
        self.device = t.device(device)
        self.rows, self.max_actions = rows, max_actions
        self.allow_cpu_reference = allow_cpu_reference is True
        self.module = None
        self.closed = False
        if self.device.type == 'cpu' and self.allow_cpu_reference:
            self._allocate()
            return
        if self.device.type != 'cuda' or not t.cuda.is_available():
            raise RuntimeError('Actual CUDA required; CPU is an explicit ranking oracle only')
        if self.device.index is None:
            self.device = t.device('cuda', t.cuda.current_device())
        if t.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare teacher ranking before capture')
        from ..batched_duel.cuda_runtime import CudaModule
        self.module = CudaModule(_KERNEL, {'rank_teacher': [
            'int64*', 'uint8*', 'uint8*', 'int64*', 'uint8*', 'int32*',
            'float64*', 'float64*', 'float64*', 'float64*',
            'int64*', 'float64*', 'int32*',
            'int32', 'int32', 'int32', 'int32', 'int32', 'int32', 'int32',
            'int32', 'int32', 'int32', 'int32']}, device=str(self.device))
        self._allocate()

    def _allocate(self):
        t, d = self.t, self.device
        self.choices = t.full((self.rows,), -1, dtype=t.int64, device=d)
        self.scores = t.zeros((self.rows, self.max_actions), dtype=t.float64, device=d)
        self.errors = t.zeros(self.rows, dtype=t.int32, device=d)
        # Owned persistent tensors only; not a CUDA process peak.
        self.allocated_bytes = sum(x.numel() * x.element_size()
                                   for x in (self.choices, self.scores, self.errors))

    def close(self):
        if not self.closed:
            if self.module is not None:
                self.module.close()
            self.closed = True

    def reset_device(self):
        self.choices.fill_(-1)
        self.scores.zero_()
        self.errors.zero_()

    def _require(self, value, dtype, shape, name):
        t = self.t
        if (not isinstance(value, t.Tensor) or value.device != self.device
                or value.dtype != dtype or tuple(value.shape) != shape
                or not value.is_contiguous()):
            raise ValueError('Exact contiguous %s required' % name)
        return value

    def rank_device(self, kinds, battle, attack, attack_count, genesis_damage, counter,
                    valid, counts, active, input_errors):
        if self.closed:
            raise RuntimeError('Teacher ranking closed')
        t = self.t
        b, m = self.rows, self.max_actions
        kinds = self._require(kinds, t.int64, (b, m), 'kind codes')
        battle = self._require(battle, t.bool, (b, m), 'battle mask')
        valid = self._require(valid, t.bool, (b, m), 'legal mask')
        counts = self._require(counts, t.int64, (b,), 'legal counts')
        active = self._require(active, t.bool, (b,), 'active rows')
        input_errors = self._require(input_errors, t.int32, (b,), 'input errors')
        attack = self._require(attack, t.float64, (b, m), 'attack')
        attack_count = self._require(attack_count, t.float64, (b, m), 'attack_count')
        genesis_damage = self._require(genesis_damage, t.float64, (b, m), 'genesis_damage')
        counter = self._require(counter, t.float64, (b, m), 'counter')
        self.reset_device()
        if self.device.type == 'cpu':
            self._reference(kinds, battle, attack, attack_count, genesis_damage,
                            counter, valid, counts, active, input_errors)
        else:
            capture = t.cuda.is_current_stream_capturing()
            event = self.module.launch(
                'rank_teacher',
                [kinds, battle.view(t.uint8), valid.view(t.uint8), counts,
                 active.view(t.uint8), input_errors, attack, attack_count,
                 genesis_damage, counter, self.choices, self.scores, self.errors,
                 b, m, KIND_MULLIGAN, KIND_CHOOSE, KIND_CONCEDE, KIND_END,
                 KIND_ULTIMATE, KIND_CYCLE, KIND_ATTACK, KIND_PLAY, len(KIND_NAMES)],
                grid=(b + 127) // 128, block=128, capture_safe=capture)
            if not capture:
                t.cuda.current_stream(self.device).wait_event(event)
        return DeviceTeacherRankingResult(self.choices, self.scores, self.errors, self.identity)

    def _reference(self, kinds, battle, attack, attack_count, genesis_damage, counter,
                   valid, counts, active, input_errors):
        import math
        from app.modules.card_game.engine.duel_v2.projection import RULE_PREVIEW_SCORE_FIELDS
        from app.modules.card_game.rl.policies import VisibleEngineRulePolicy
        policy = VisibleEngineRulePolicy()
        attack_key, count_key, genesis_key, counter_key = RULE_PREVIEW_SCORE_FIELDS
        for row in range(self.rows):
            if not bool(active[row]):
                continue
            sticky = int(input_errors[row])
            if sticky:
                self.errors[row] = sticky
                continue
            n = int(counts[row])
            if n < 0 or n > self.max_actions:
                self.errors[row] = ERR_COUNT
                continue
            if n < 1:
                self.errors[row] = ERR_EMPTY
                continue
            row_valid = [bool(valid[row, col]) for col in range(self.max_actions)]
            row_battle = [bool(battle[row, col]) for col in range(self.max_actions)]
            row_kinds = [int(kinds[row, col]) for col in range(self.max_actions)]
            if any(row_valid[col] != (col < n) for col in range(self.max_actions)):
                self.errors[row] = ERR_MASK
                continue
            if any(row_battle[col] and not row_valid[col] for col in range(self.max_actions)):
                self.errors[row] = ERR_BATTLE
                continue
            if any(row_valid[col] and not (0 <= row_kinds[col] < len(KIND_NAMES))
                   for col in range(self.max_actions)):
                self.errors[row] = ERR_KIND
                continue
            if any(row_battle[col] and row_kinds[col] not in (KIND_ATTACK, KIND_PLAY)
                   for col in range(n)):
                self.errors[row] = ERR_BATTLE
                continue
            early = False
            for col in range(n):
                if row_kinds[col] == KIND_MULLIGAN:
                    self.choices[row] = col
                    early = True
                    break
            if not early:
                for col in range(n):
                    if row_kinds[col] == KIND_CHOOSE:
                        self.choices[row] = col
                        early = True
                        break
            actions = []
            entries = []
            hand = []
            nonfinite = False
            for col in range(n):
                kind = KIND_NAMES[row_kinds[col]]
                action = {'type': kind, 'ordinal': col}
                preview = {}
                combat = row_kinds[col] == KIND_ATTACK or (
                    row_kinds[col] == KIND_PLAY and row_battle[col])
                if kind == 'play_card':
                    card_id = 'card-%s' % col
                    action['card_id'] = card_id
                    hand.append({
                        'instance_id': card_id,
                        'type': 'battle' if row_battle[col] else 'tactic',
                    })
                if combat:
                    values = (
                        float(attack[row, col]), float(attack_count[row, col]),
                        float(genesis_damage[row, col]), float(counter[row, col]))
                    if not all(math.isfinite(value) for value in values):
                        nonfinite = True
                    preview = {
                        attack_key: values[0], count_key: values[1],
                        genesis_key: values[2], counter_key: values[3],
                    }
                actions.append(action)
                entries.append({'action': action, 'preview': preview})
            if early:
                continue
            if nonfinite:
                self.errors[row] = ERR_NONFINITE
                continue
            observation = {
                'viewer_side': 'a',
                'legal_actions': entries,
                'sides': {'a': {'hand': hand}},
            }
            self.choices[row] = int(policy(observation, tuple(actions)))
            ranked = policy.ranking_scores(observation, tuple(actions))
            if not all(math.isfinite(score) for score in ranked):
                self.choices[row] = -1
                self.errors[row] = ERR_NONFINITE
                continue
            for col, score in enumerate(ranked):
                self.scores[row, col] = float(score)
