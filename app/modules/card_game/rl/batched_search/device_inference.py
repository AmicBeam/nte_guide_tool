"""Frozen seven-policy inference with device-resident row routing and outputs.

This is a numerical primitive, not a complete device search implementation.
Construction authenticates models through the existing inference bridge; the
hot path never exports observations, candidates, scores or WDL to the host.
"""
from dataclasses import dataclass
from typing import Any


_ALIAS_SOURCE = r'''
extern "C" __global__ void representation_hash(const unsigned int* bits,
 const unsigned char* mask,long long* hashes,int rows,int actions,int width){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*actions)return;
 unsigned long long h=1469598103934665603ULL;
 for(int col=0;col<width;col++)h=(h^bits[(long long)i*width+col])*1099511628211ULL;
 hashes[i]=(long long)((h^mask[i])*1099511628211ULL);
}
extern "C" __global__ void representation_alias(const unsigned int* bits,
 const unsigned char* mask,const long long* hashes,long long* aliases,
 int rows,int actions,int width){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*actions)return;
 int row=i/actions,index=i%actions;aliases[i]=index;
 for(int earlier=0;earlier<index;earlier++){
  int j=row*actions+earlier;if(mask[j]!=mask[i]||hashes[j]!=hashes[i])continue;
  int same=1;for(int col=0;col<width;col++){
   if(bits[(long long)i*width+col]!=bits[(long long)j*width+col]){same=0;break;}
  }
  if(same){aliases[i]=earlier;break;}
 }
}
'''


@dataclass(frozen=True)
class DeviceInferenceResult:
    logits: Any
    wdl: Any
    invalid_rows: Any


class DeviceSevenInference:
    def __init__(self, models, *, device='cuda', max_rows=1600,
                 max_candidate_elements=256 * 256 * 1576,
                 allow_cpu_reference=False):
        import torch
        from .inference import BatchedInference
        from .. import seven_policy, seven_grounded
        if type(max_rows) is not int or max_rows < 1:
            raise ValueError('Positive bounded device row count required')
        if type(max_candidate_elements) is not int or max_candidate_elements < 1:
            raise ValueError('Positive bounded device workspace required')
        self.device = torch.device(device)
        if self.device.type != 'cuda' and not (self.device.type == 'cpu' and allow_cpu_reference):
            raise ValueError('CUDA required; CPU is an explicit numerical oracle only')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required; no CPU fallback')
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.oracle = BatchedInference(models, device=str(self.device))
        self.device = self.oracle._torch_device
        self.keys = tuple(sorted(models))
        self.versions = tuple(models[k].version for k in self.keys)
        if any(models[k].manifest['schema'] != seven_policy.SCHEMA for k in self.keys):
            raise ValueError('Exact seven-policy schema required')
        base = [self.oracle._networks[k] for k in self.keys]
        if len({net.hidden for net in base}) != 1:
            raise ValueError('One compatible hidden width required; never blend architectures')
        self.hidden = base[0].hidden
        self.max_rows = max_rows
        self.max_candidate_elements = max_candidate_elements
        self.feature_dim = self.oracle._contracts[self.keys[0]]['feature_dim']
        self.candidate_dim = self.oracle._contracts[self.keys[0]]['candidate_dim']
        self.torch = torch
        self.categories = seven_grounded.CATEGORIES
        self.front_index = seven_grounded.FRONT_INDEX
        self.actor_index = torch.as_tensor(seven_grounded.INDEX.copy(), device=self.device)
        self.battle_cards = torch.as_tensor(seven_grounded.BATTLE_CARDS.copy(), device=self.device)
        expected = set(seven_policy.tensor_shapes(self.hidden))
        self.weights = {name: torch.stack([net.state_dict()[name].detach().clone()
                                         for net in base]) for name in expected}
        self._prepare_legacy(models)
        self.alias_module = None
        if self.device.type == 'cuda':
            from ..batched_duel.cuda_runtime import CudaModule
            self.alias_module = CudaModule(_ALIAS_SOURCE, {
                'representation_hash': ('uint32*', 'uint8*', 'int64*', 'int32', 'int32', 'int32'),
                'representation_alias': ('uint32*', 'uint8*', 'int64*', 'int64*', 'int32', 'int32', 'int32'),
            }, device=str(self.device))

    def close(self):
        if self.alias_module is not None:
            self.alias_module.close()

    def _aliases(self, candidates, valid):
        t = self.torch; b, m, width = candidates.shape
        if not m:
            return t.empty((b, 0), dtype=t.int64, device=self.device)
        bits = candidates.contiguous().view(t.int32)
        if self.device.type == 'cpu':
            # Explicit numerical oracle only; never the CUDA execution route.
            columns = t.arange(m, device=self.device)[None, :]
            aliases = columns.expand(b, m).clone()
            for earlier in range(m - 1, -1, -1):
                same = ((bits == bits[:, earlier:earlier+1]).all(-1)
                        & (valid == valid[:, earlier:earlier+1]) & (columns >= earlier))
                aliases = t.where(same, t.full_like(aliases, earlier), aliases)
            return aliases
        hashes = t.empty((b, m), dtype=t.int64, device=self.device)
        aliases = t.empty_like(hashes); mask = valid.to(t.uint8).contiguous()
        capture = t.cuda.is_current_stream_capturing()
        self.alias_module.launch('representation_hash', [bits, mask, hashes, b, m, width],
                                 grid=(b*m+127)//128, block=128, capture_safe=capture)
        event = self.alias_module.launch('representation_alias',
                                        [bits, mask, hashes, aliases, b, m, width],
                                        grid=(b*m+127)//128, block=128, capture_safe=capture)
        if event is not None:
            t.cuda.current_stream(self.device).wait_event(event)
        return aliases

    def _prepare_legacy(self, models):
        import numpy as np
        t = self.torch
        teachers = [getattr(models[k], 'teacher', None) for k in self.keys]
        self.has_legacy = any(v is not None for v in teachers)
        if not self.has_legacy:
            return
        live = [v for v in teachers if v is not None]
        self.legacy_hidden = max(v['hidden'] for v in live)
        xd = max(len(v['observation_index']) for v in live)
        cd = max(len(v['candidate_index']) for v in live)
        count = len(teachers)
        observation = np.full((count, xd), self.feature_dim, dtype=np.int64)
        candidate = np.full((count, cd), self.candidate_dim, dtype=np.int64)
        extra = np.zeros((count, self.candidate_dim), dtype=np.bool_)
        scales = np.ones(count, dtype=np.float32)
        enabled = np.zeros(count, dtype=np.bool_)
        shapes = {'state_net.0.weight': (self.legacy_hidden, xd),
                  'state_net.0.bias': (self.legacy_hidden,),
                  'state_net.2.weight': (self.legacy_hidden, self.legacy_hidden),
                  'state_net.2.bias': (self.legacy_hidden,),
                  'cand_net.0.weight': (self.legacy_hidden, cd),
                  'cand_net.0.bias': (self.legacy_hidden,),
                  'cand_net.2.weight': (self.legacy_hidden, self.legacy_hidden),
                  'cand_net.2.bias': (self.legacy_hidden,),
                  'score.weight': (1, self.legacy_hidden), 'score.bias': (1,)}
        banks = {name: np.zeros((count, *shape), dtype=np.float32)
                 for name, shape in shapes.items()}
        from ..seven_lineup import candidate_names
        names = candidate_names()
        for i, teacher in enumerate(teachers):
            if teacher is None:
                continue
            enabled[i] = True
            xindex = teacher['observation_index']; cindex = teacher['candidate_index']
            if (any(type(v) is not int or not 0 <= v < self.feature_dim for v in xindex)
                    or any(type(v) is not int or not 0 <= v < self.candidate_dim for v in cindex)):
                raise ValueError('Named legacy projection outside current schema')
            observation[i, :len(xindex)] = xindex
            candidate[i, :len(cindex)] = cindex
            extra[i] = [name not in teacher['candidates'] for name in names]
            scales[i] = np.sqrt(np.float32(teacher['hidden']))
            for name, target in banks.items():
                value = np.asarray(teacher['weights'][name])
                if value.dtype != np.float32 or value.ndim != len(target.shape) - 1:
                    raise ValueError('Legacy tensor dtype/layout mismatch')
                if any(a > b for a, b in zip(value.shape, target.shape[1:])):
                    raise ValueError('Legacy tensor exceeds named padded bank')
                target[(i, *(slice(0, n) for n in value.shape))] = value
        self.legacy_weights = {k: t.as_tensor(v, device=self.device) for k, v in banks.items()}
        self.legacy_observation = t.as_tensor(observation, device=self.device)
        self.legacy_candidate = t.as_tensor(candidate, device=self.device)
        self.legacy_extra = t.as_tensor(extra, device=self.device)
        self.legacy_scale = t.as_tensor(scales, device=self.device)
        self.legacy_enabled = t.as_tensor(enabled, device=self.device)

    def _linear(self, x, ids, name, *, legacy=False):
        weights = self.legacy_weights if legacy else self.weights
        weight = weights[name + '.weight'].index_select(0, ids)
        bias = weights[name + '.bias'].index_select(0, ids)
        if x.ndim == 2:
            return self.torch.bmm(x[:, None, :], weight.transpose(1, 2))[:, 0, :] + bias
        return self.torch.bmm(x, weight.transpose(1, 2)) + bias[:, None, :]

    def _norm(self, x, ids, name, active=False):
        weight = self.weights[name + '.weight'].index_select(0, ids)
        bias = self.weights[name + '.bias'].index_select(0, ids)
        if x.ndim == 3:
            weight = weight[:, None, :]; bias = bias[:, None, :]
        if active:
            enabled = self.weights[name + '.active'].index_select(0, ids)
            if x.ndim == 3:
                enabled = enabled[:, None, :]
            count = enabled.sum(-1, keepdim=True)
            mean = (x * enabled).sum(-1, keepdim=True) / count
            variance = ((x - mean).square() * enabled).sum(-1, keepdim=True) / count
        else:
            mean = x.mean(-1, keepdim=True)
            variance = (x - mean).square().mean(-1, keepdim=True)
        return (x - mean) * self.torch.rsqrt(variance + 1e-5) * weight + bias

    def _tower(self, x, ids, prefix):
        t = self.torch
        x = t.nn.functional.silu(self._linear(self._norm(x, ids, prefix + '.0', True), ids, prefix + '.1'))
        for block in (3, 4):
            name = prefix + '.' + str(block)
            branch = t.nn.functional.silu(self._linear(self._norm(x, ids, name + '.norm'), ids, name + '.first'))
            x = x + self._linear(branch, ids, name + '.last')
        return self._norm(x, ids, prefix + '.5')

    def _features(self, x, c):
        t = self.torch; blocks = []
        for column, size in self.categories:
            index = c[..., column].long() + (column != 0)
            blocks.append(t.nn.functional.one_hot(index.clamp(0, size - 1), size).to(x.dtype))
        numeric = c[..., 5:].clone(); numeric[..., :4] /= 10
        blocks.append(numeric)
        heroes = x[:, self.actor_index]
        rows = t.arange(x.shape[0], device=self.device)[:, None]
        actors = c[..., 1].long(); sides = c[..., 3].long(); targets = c[..., 4].long()
        battle = self.battle_cards[(c[..., 2].long() + 1).clamp(0, self.battle_cards.shape[0] - 1)]
        # Constants are taken from the same named action schema, not row ownership.
        from ..league_schema import ACT_ATTACK, ACT_PLAY
        combat = (c[..., 0] == ACT_ATTACK) | ((c[..., 0] == ACT_PLAY) & battle)
        implicit = combat & (targets < 0) & (sides < 0)
        targets = t.where(implicit, t.round(x[:, self.front_index] * 10).long()[:, None], targets)
        sides = t.where(implicit, t.ones_like(sides), sides)
        blocks.append(heroes[rows, 0, actors.clamp(0, heroes.shape[2] - 1)] * (actors >= 0)[..., None])
        blocks.append(heroes[rows, sides.clamp(0, 1), targets.clamp(0, heroes.shape[2] - 1)]
                      * ((targets >= 0) & (sides >= 0))[..., None])
        return t.cat(blocks, -1)

    def _legacy(self, x, c, mask, ids):
        t = self.torch
        from ..fixed_lineup import SEATS, CARD_IDS
        extra = self.legacy_extra.index_select(0, ids)
        unknown = ((c[..., 1] >= len(SEATS)) | (c[..., 2] >= len(CARD_IDS))
                   | (c[..., 4] >= len(SEATS)) | ((c.abs() > 1e-8) & extra[:, None, :]).any(-1))
        supported = self.legacy_enabled.index_select(0, ids) & ~(unknown & mask).any(-1)
        xp = t.cat((x, x.new_zeros((x.shape[0], 1))), -1)
        cp = t.cat((c, c.new_zeros((c.shape[0], c.shape[1], 1))), -1)
        old_x = xp.gather(1, self.legacy_observation.index_select(0, ids))
        projection = self.legacy_candidate.index_select(0, ids)
        old_c = cp.gather(2, projection[:, None, :].expand(c.shape[0], c.shape[1], -1))
        state = self._linear(old_x, ids, 'state_net.0', legacy=True).relu()
        state = self._linear(state, ids, 'state_net.2', legacy=True).relu()
        cand = self._linear(old_c, ids, 'cand_net.0', legacy=True).relu()
        cand = self._linear(cand, ids, 'cand_net.2', legacy=True).relu()
        result = ((cand * state[:, None, :]).sum(-1) / self.legacy_scale.index_select(0, ids)[:, None]
                  + self._linear(cand, ids, 'score', legacy=True)[..., 0])
        return t.where(supported[:, None], result, t.zeros_like(result))

    def predict_device(self, x, candidates, valid, model_index, value_actor, *, value_only=None):
        t = self.torch
        for tensor in (x, candidates, valid, model_index, value_actor):
            if not isinstance(tensor, t.Tensor) or tensor.device != self.device:
                raise ValueError('Inputs must already be on the bound device')
        b, m = candidates.shape[:2]
        if (x.dtype != t.float32 or candidates.dtype != t.float32 or valid.dtype != t.bool
                or model_index.dtype != t.int64 or value_actor.dtype != t.float32
                or x.shape != (b, self.feature_dim) or candidates.shape != (b, m, self.candidate_dim)
                or valid.shape != (b, m) or model_index.shape != (b,) or value_actor.shape != (b, 1)
                or not 1 <= b <= self.max_rows or candidates.numel() > self.max_candidate_elements):
            raise ValueError('Exact bounded device input shapes and dtypes required')
        if self.device.type == 'cuda' and t.backends.cuda.matmul.allow_tf32:
            raise ValueError('Explicit FP32 inference requires TF32 disabled')
        if value_only is None:
            value_only = t.zeros(b, dtype=t.bool, device=self.device)
        if value_only.device != self.device or value_only.dtype != t.bool or value_only.shape != (b,):
            raise ValueError('Device value-only mask required')
        invalid = ((model_index < 0) | (model_index >= len(self.keys))
                   | ~t.isfinite(x).all(-1) | ~t.isfinite(candidates).all(-1).all(-1)
                   | ~((value_actor == 0) | (value_actor == 1)).all(-1))
        for column, size in self.categories:
            column_value = candidates[..., column]
            categorical = column_value + (column != 0)
            invalid = invalid | (((column_value != t.round(column_value))
                                   | (categorical < 0) | (categorical >= size)) & valid).any(-1)
        invalid = invalid | (~value_only & ~valid.any(-1)) | (value_only & valid.any(-1))
        ids = model_index.clamp(0, len(self.keys) - 1)
        with t.inference_mode():
            state = self._tower(x, ids, 'state_net')
            cand = self._tower(self._features(x, candidates), ids, 'cand_net')
            context = state[:, None, :].expand_as(cand)
            joint = t.cat((cand, context, cand * context), -1)
            scores = self._linear(t.nn.functional.silu(self._linear(joint, ids, 'score.0')), ids, 'score.2')[..., 0]
            if self.has_legacy:
                scores = scores + self._legacy(x, candidates, valid, ids)
            scores = scores.gather(1, self._aliases(candidates, valid))
            representation = self._tower(x, ids, 'value_net')
            wdl = self._linear(t.cat((representation, value_actor), -1), ids, 'wdl').softmax(-1)
            invalid = invalid | ~t.isfinite(wdl).all(-1) | (~t.isfinite(scores) & valid).any(-1)
            scores = scores.masked_fill(~valid | value_only[:, None], t.finfo(scores.dtype).min)
        return DeviceInferenceResult(scores, wdl, invalid)
