"""Offline FP32 Torch batched inference bridge for verified recovery models.

Importing this module does not import Torch or initialize CUDA. Networks are
only constructed and loaded onto the specified device when BatchedInference
is explicitly instantiated.
"""
from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union
import numpy as np

from app.modules.card_game.rl import cross_lineup

FEATURE_DIM = len(cross_lineup.all_features())
CANDIDATE_DIM = len(cross_lineup.candidate_names())


def _five_team_schemas():
    from app.modules.card_game.rl import residual_policy, recovery_policy, preserved_policy
    return {
        residual_policy.SCHEMA,
        recovery_policy.RESIDUAL_SCHEMA,
        preserved_policy.SCHEMA,
        preserved_policy.PARTIAL_SCHEMA,
        recovery_policy.PRESERVED_SCHEMA,
    }


def named_observation_contract(schema, manifest):
    """Bind exact named feature/candidate widths for one policy family.

    Seven-team models keep seven_lineup widths. Five-team recovery models keep
    cross_lineup widths. There is no old-width default padding in either direction.
    """
    from app.modules.card_game.rl import cross_grounded, cross_lineup, seven_grounded, seven_lineup, seven_policy
    five = _five_team_schemas()
    features = manifest.get('features')
    candidates = manifest.get('candidates')
    if schema in five:
        expected_features = cross_lineup.all_features()
        expected_candidates = cross_lineup.candidate_names()
        grounded = cross_grounded
        family = 'recovery'
    elif schema == seven_policy.SCHEMA:
        expected_features = seven_lineup.all_features()
        expected_candidates = seven_lineup.candidate_names()
        grounded = seven_grounded
        family = 'seven'
    else:
        raise ValueError(f'Unsupported schema: {schema}')
    if features != expected_features:
        raise ValueError(
            f'Model features do not match the {family} named observation contract '
            f'(expected {len(expected_features)} named columns, got {None if features is None else len(features)})'
        )
    if candidates != expected_candidates:
        raise ValueError(
            f'Model candidates do not match the {family} named candidate contract '
            f'(expected {len(expected_candidates)} named columns, got {None if candidates is None else len(candidates)})'
        )
    return {
        'family': family,
        'schema': schema,
        'features': tuple(expected_features),
        'candidates': tuple(expected_candidates),
        'feature_dim': len(expected_features),
        'candidate_dim': len(expected_candidates),
        'categories': tuple(grounded.CATEGORIES),
    }


@dataclass(frozen=True)
class InferenceRequest:
    request_id: Any
    model_key: str
    model_version: str
    x: np.ndarray
    candidates: np.ndarray
    value_actor: Union[bool, int]
    value_only: bool = False
    deadline: Optional[float] = None


@dataclass(frozen=True)
class InferenceResult:
    request_id: Any
    model_key: str
    model_version: str
    logits: np.ndarray
    wdl: np.ndarray
    expired: bool = False


class BatchedInference:
    """Offline FP32 Torch batched inference engine for recovery models."""

    def __init__(
        self,
        models: Mapping[str, Any],
        *,
        device: str = 'cpu',
        max_batch_rows: int = 256,
        max_candidate_elements: int = 256 * 256 * CANDIDATE_DIM,
    ):
        if not isinstance(models, Mapping) or not models:
            raise ValueError("models must be a non-empty Mapping")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty string")
        if type(max_batch_rows) is not int or max_batch_rows < 1:
            raise ValueError("max_batch_rows must be a positive integer")
        if type(max_candidate_elements) is not int or max_candidate_elements < 1:
            raise ValueError("max_candidate_elements must be a positive integer")

        self._device_str = device
        self._max_batch_rows = max_batch_rows
        self._max_candidate_elements = max_candidate_elements

        # Lazy torch import only during explicit instantiation
        import torch
        from app.modules.card_game.rl import residual_policy, recovery_policy, preserved_policy

        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

        self._torch_device = torch.device(device)
        if self._torch_device.type not in ('cpu', 'cuda'):
            raise ValueError('CPU oracle or CUDA device required')
        if self._torch_device.type == 'cuda' and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError('Explicit FP32 contract requires TF32 disabled by the calling process')

        valid_schemas = {
            residual_policy.SCHEMA,
            recovery_policy.RESIDUAL_SCHEMA,
            preserved_policy.SCHEMA,
            preserved_policy.PARTIAL_SCHEMA,
            recovery_policy.PRESERVED_SCHEMA,
        }

        self._models: Dict[str, Any] = {}
        self._model_versions: Dict[str, str] = {}
        self._networks: Dict[str, Any] = {}

        from app.modules.card_game.rl import seven_policy
        valid_schemas = set(valid_schemas)
        valid_schemas.add(seven_policy.SCHEMA)

        total_weight_bytes = 0

        self._contracts: Dict[str, Dict[str, Any]] = {}
        self._families: Dict[str, str] = {}

        for key, model in models.items():
            if not isinstance(key, str) or not key:
                raise ValueError("model key must be a non-empty string")

            manifest = getattr(model, 'manifest', None)
            if not isinstance(manifest, dict):
                raise ValueError(f"Model {key} manifest must be a dict")

            schema = getattr(model, 'schema', manifest.get('schema'))
            if manifest.get('schema') != schema:
                raise ValueError(
                    f"Model {key} schema mismatch between attribute ({schema}) "
                    f"and manifest ({manifest.get('schema')})"
                )
            if schema not in valid_schemas:
                raise ValueError(f"Model {key} has unsupported schema: {schema}")

            version = getattr(model, 'version', None)
            if not isinstance(version, str) or not version:
                raise ValueError(f"Model {key} version must be a non-empty string")
            if manifest.get('sha256') and manifest.get('sha256') != version:
                raise ValueError(
                    f"Model {key} version ({version}) does not match manifest sha256 ({manifest.get('sha256')})"
                )

            contract = named_observation_contract(schema, manifest)
            if self._families and contract['family'] not in set(self._families.values()):
                raise ValueError(
                    'BatchedInference binds one runtime family per instance; '
                    f'recovery and seven models cannot be mixed (got {contract["family"]} after {sorted(set(self._families.values()))})'
                )

            hidden = getattr(model, 'hidden', None) or manifest.get('hidden')
            if type(hidden) is not int or not 16 <= hidden <= 256:
                raise ValueError(f"Model {key} hidden width must be integer in [16, 256], got {hidden}")

            weights = getattr(model, 'weights', None)
            if not isinstance(weights, dict):
                raise ValueError(f"Model {key} weights must be a dict of numpy arrays")

            if contract['family'] == 'seven':
                from app.modules.card_game.rl import seven_policy
                expected_shapes = dict(seven_policy.tensor_shapes(hidden))
                separated = False
                teacher = getattr(model, 'teacher', None)
                if teacher is not None:
                    from app.modules.card_game.rl.preserved_policy import encoded_name
                    expected_shapes.update({
                        'legacy.' + encoded_name(name): value.shape
                        for name, value in teacher['weights'].items()
                    })
            else:
                expected_shapes = dict(residual_policy.tensor_shapes(hidden))
                separated = getattr(model, 'separated', False) or (schema in recovery_policy.SCHEMAS)
                if separated:
                    expected_shapes.update(recovery_policy.extra_shapes(hidden))
                if schema in (preserved_policy.SCHEMA, preserved_policy.PARTIAL_SCHEMA, recovery_policy.PRESERVED_SCHEMA):
                    from app.modules.card_game.rl.fixed_lineup import tensor_shapes as old_tensor_shapes, CAND_DIM as OLD_CAND_DIM
                    teacher = getattr(model, 'teacher', None)
                    if teacher is None:
                        from app.modules.card_game.rl.cross_teacher_distillation import REPO_TEACHERS, verify_old_teacher
                        deck_name = manifest.get('deck') or key
                        teacher = verify_old_teacher(REPO_TEACHERS, deck_name)
                    teacher_hidden = teacher['hidden']
                    old = old_tensor_shapes(teacher_hidden)
                    old['state_net.0.weight'] = (teacher_hidden, len(teacher['features']))
                    old['cand_net.0.weight'] = (teacher_hidden, OLD_CAND_DIM)
                    expected_shapes.update({'legacy.' + preserved_policy.encoded_name(name): s for name, s in old.items()})

            if set(weights.keys()) != set(expected_shapes.keys()):
                missing = set(expected_shapes) - set(weights)
                extra = set(weights) - set(expected_shapes)
                raise ValueError(f"Model {key} weight tensors mismatch: missing={missing}, extra={extra}")

            for name, expected_shape in expected_shapes.items():
                arr = weights[name]
                if arr.shape != expected_shape:
                    raise ValueError(f"Model {key} tensor {name} shape mismatch: expected {expected_shape}, got {arr.shape}")
                if arr.dtype != np.float32:
                    raise ValueError(f"Model {key} tensor {name} dtype mismatch: expected float32, got {arr.dtype}")
                if not np.isfinite(arr).all():
                    raise ValueError(f"Model {key} tensor {name} contains non-finite values")

            # Build Torch network
            if contract['family'] == 'seven':
                from app.modules.card_game.rl import seven_policy
                net = seven_policy.network(hidden, self._torch_device)
                if getattr(model, 'teacher', None) is not None:
                    net = seven_policy.attach_legacy(net, key)
            elif schema in (preserved_policy.SCHEMA, preserved_policy.PARTIAL_SCHEMA, recovery_policy.PRESERVED_SCHEMA):
                deck_name = manifest.get('deck') or key
                net, _ = preserved_policy.create_network(
                    deck_name,
                    hidden=hidden,
                    device=self._torch_device,
                    partial=schema in (preserved_policy.PARTIAL_SCHEMA, recovery_policy.PRESERVED_SCHEMA),
                )
                if separated:
                    net = recovery_policy.attach(net)
            else:
                net = residual_policy.network(hidden, self._torch_device)
                if separated:
                    net = recovery_policy.attach(net)

            state_dict = {
                name: torch.as_tensor(arr.copy(), dtype=torch.float32, device=self._torch_device)
                for name, arr in weights.items()
            }
            net.load_state_dict(state_dict)

            for p in net.parameters():
                p.requires_grad_(False)
            net.eval()

            # Record model weight bytes
            net_bytes = sum(p.numel() * p.element_size() for p in net.parameters()) + sum(
                b.numel() * b.element_size() for b in net.buffers()
            )
            total_weight_bytes += net_bytes

            self._models[key] = model
            self._model_versions[key] = version
            self._networks[key] = net

            self._contracts[key] = contract
            self._families[key] = contract['family']

        self._weight_nbytes = total_weight_bytes
        self._total_batches = 0
        self._total_rows = 0
        self._total_candidate_padding = 0

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def max_batch_rows(self) -> int:
        return self._max_batch_rows

    @property
    def max_candidate_elements(self) -> int:
        return self._max_candidate_elements

    @property
    def models(self) -> Dict[str, Any]:
        return dict(self._models)

    def stats(self) -> Dict[str, Any]:
        return {
            'batches': self._total_batches,
            'rows': self._total_rows,
            'candidate_padding': self._total_candidate_padding,
            'weight_nbytes': self._weight_nbytes,
            'device': self._device_str,

            'feature_dims': {key: row['feature_dim'] for key, row in self._contracts.items()},
            'candidate_dims': {key: row['candidate_dim'] for key, row in self._contracts.items()},
            'schemas': {key: row['schema'] for key, row in self._contracts.items()},
            'runtime_family': sorted(set(self._families.values())),
        }

    def predict(self, requests: Sequence[InferenceRequest]) -> List[InferenceResult]:
        if not isinstance(requests, (list, tuple)):
            raise TypeError("requests must be a sequence of InferenceRequest instances")

        if len(requests) == 0:
            return []

        # Step 1: Strict request validation
        seen_ids: Set[Any] = set()
        validated_requests: List[Tuple[int, InferenceRequest]] = []

        for idx, req in enumerate(requests):
            if not isinstance(req, InferenceRequest):
                raise TypeError(f"Request at index {idx} is not an InferenceRequest, got {type(req)}")

            try:
                if req.request_id in seen_ids:
                    raise ValueError(f"Duplicate request_id: {req.request_id}")
                seen_ids.add(req.request_id)
            except TypeError:
                raise ValueError(f"Unhashable request_id: {req.request_id}")

            if req.model_key not in self._models:
                raise KeyError(f"Unknown model_key: {req.model_key}")

            expected_version = self._model_versions[req.model_key]
            if req.model_version != expected_version:
                raise ValueError(
                    f"Model version mismatch for {req.model_key}: expected {expected_version}, got {req.model_version}"
                )

            contract = self._contracts[req.model_key]
            feature_dim = contract['feature_dim']
            candidate_dim = contract['candidate_dim']

            # Validate observation x: finite float32 shape (FEATURE_DIM,)
            if not isinstance(req.x, np.ndarray):
                raise ValueError(f"Request {req.request_id} observation x must be a numpy array")
            if req.x.dtype != np.float32:
                raise ValueError(f"Request {req.request_id} observation x dtype must be float32, got {req.x.dtype}")
            if req.x.ndim != 1 or req.x.shape != (feature_dim,):
                raise ValueError(
                    f"Request {req.request_id} observation x shape must be ({feature_dim},), got {req.x.shape}"
                )
            if not np.isfinite(req.x).all():
                raise ValueError(f"Request {req.request_id} observation x contains non-finite values")

            # Validate candidates c: finite float32 shape (A, CANDIDATE_DIM) with A >= 1
            if not isinstance(req.candidates, np.ndarray):
                raise ValueError(f"Request {req.request_id} candidates must be a numpy array")
            if req.candidates.dtype != np.float32:
                raise ValueError(f"Request {req.request_id} candidates dtype must be float32, got {req.candidates.dtype}")
            if req.candidates.ndim != 2 or req.candidates.shape[1] != candidate_dim:
                raise ValueError(
                    f"Request {req.request_id} candidates shape must be (A, {candidate_dim}), got {req.candidates.shape}"
                )
            if type(req.value_only) is not bool:
                raise ValueError('value_only must be boolean')
            if req.candidates.shape[0] < 1 and not req.value_only:
                raise ValueError(f'Request {req.request_id} requires at least one policy candidate')
            if req.value_only and req.candidates.shape[0] != 0:
                raise ValueError('Value-only requests use an empty candidate matrix')
            if not np.isfinite(req.candidates).all():
                raise ValueError(f"Request {req.request_id} candidates contains non-finite values")

            # Check candidate element cap
            A = req.candidates.shape[0]
            if A * candidate_dim > self._max_candidate_elements:
                raise ValueError(
                    f"Request {req.request_id} candidate scalar count ({A * candidate_dim}) exceeds max_candidate_elements ({self._max_candidate_elements})"
                )

            for column, size in contract['categories']:
                values = req.candidates[:, column]
                ids = values + (column != 0)
                if np.any(values != np.rint(values)) or np.any((ids < 0) | (ids >= size)):
                    raise ValueError(f'Invalid categorical candidate field {column}')

            # Validate value_actor: only accepts bool or exact 0/1 integer
            if type(req.value_actor) is bool or isinstance(req.value_actor, np.bool_):
                pass
            elif (
                (type(req.value_actor) is int or isinstance(req.value_actor, np.integer))
                and req.value_actor in (0, 1)
            ):
                pass
            else:
                raise ValueError(
                    f"Request {req.request_id} value_actor must be bool or exact int 0/1, got {type(req.value_actor)}: {req.value_actor}"
                )

            validated_requests.append((idx, req))
            if req.deadline is not None and (isinstance(req.deadline,bool) or
                    not isinstance(req.deadline,(int,float)) or not math.isfinite(req.deadline)):
                raise ValueError('Finite monotonic inference deadline required')

        # Step 2: Group by actual model version: (model_key, model_version)
        groups: Dict[Tuple[str, str, bool], List[Tuple[int, InferenceRequest]]] = {}
        for item in validated_requests:
            req = item[1]
            group_key = (req.model_key, req.model_version, req.value_only)
            groups.setdefault(group_key, []).append(item)

        results: List[Optional[InferenceResult]] = [None] * len(requests)

        import torch
        inference_ctx = getattr(torch, 'inference_mode', torch.no_grad)

        # Step 3: Execute microbatches per group
        for (model_key, _, value_only), group_items in groups.items():
            net = self._networks[model_key]

            contract = self._contracts[model_key]
            feature_dim = contract['feature_dim']
            candidate_dim = contract['candidate_dim']

            start_idx = 0
            while start_idx < len(group_items):
                # Bound the padded scalar tensor, not just each row's action count.
                microbatch = []
                M = 0
                for item in group_items[start_idx:start_idx + self._max_batch_rows]:
                    next_m = max(M, item[1].candidates.shape[0])
                    if microbatch and (len(microbatch) + 1) * next_m * candidate_dim > self._max_candidate_elements:
                        break
                    microbatch.append(item)
                    M = next_m
                start_idx += len(microbatch)
                fresh=[]
                for index,req in microbatch:
                    if req.deadline is not None and time.monotonic()>=req.deadline:
                        results[index]=InferenceResult(req.request_id,req.model_key,req.model_version,
                            np.empty(0,dtype=np.float32),np.empty(0,dtype=np.float32),True)
                    else:fresh.append((index,req))
                microbatch=fresh
                if not microbatch:continue
                B = len(microbatch)

                # Prepare padded batch arrays
                x_batch = np.zeros((B, feature_dim), dtype=np.float32)
                c_batch = np.zeros((B, M, candidate_dim), dtype=np.float32)
                mask_batch = np.zeros((B, M), dtype=bool)
                actor_batch = np.zeros((B, 1), dtype=np.float32)

                padding_count = 0
                for i, (_, req) in enumerate(microbatch):
                    x_batch[i] = req.x
                    a_len = req.candidates.shape[0]
                    c_batch[i, :a_len] = req.candidates
                    mask_batch[i, :a_len] = True
                    actor_batch[i, 0] = 1.0 if bool(req.value_actor) else 0.0
                    padding_count += (M - a_len)

                # Forward inference on chosen device
                with inference_ctx():
                    x_t = torch.as_tensor(x_batch, device=self._torch_device, dtype=torch.float32)
                    c_t = torch.as_tensor(c_batch, device=self._torch_device, dtype=torch.float32)
                    mask_t = torch.as_tensor(mask_batch, device=self._torch_device, dtype=torch.bool)
                    actor_t = torch.as_tensor(actor_batch, device=self._torch_device, dtype=torch.float32)

                    if value_only:
                        scores = x_t.new_empty((B, 0))
                        val_repr = getattr(net, 'value_net', net.state_net)(x_t)
                    else:
                        state = net.state_net(x_t)
                        cand_feats = net.candidate_features(x_t, c_t)
                        cand = net.cand_net(cand_feats)
                        context = state.unsqueeze(1).expand_as(cand)
                        joint = torch.cat((cand, context, cand * context), dim=-1)
                        scores = net.score(joint).squeeze(-1)
                        if hasattr(net, 'legacy'):
                            scores = scores + net.legacy(x_t, c_t, mask_t)
                        val_repr = net.value_net(x_t) if hasattr(net, 'value_net') else state

                    wdl_logits = net.wdl(torch.cat((val_repr, actor_t), dim=-1))
                    wdl_probs = torch.softmax(wdl_logits, dim=-1)

                    # CUDA synchronization happens only once for the whole microbatch
                    scores_cpu = scores.detach().to(device='cpu', dtype=torch.float32).numpy()
                    wdl_cpu = wdl_probs.detach().to(device='cpu', dtype=torch.float32).numpy()

                # Slice exact candidate lengths, verify finiteness, and populate results
                for i, (orig_idx, req) in enumerate(microbatch):
                    a_len = req.candidates.shape[0]
                    logits = scores_cpu[i, :a_len].copy()
                    wdl = wdl_cpu[i].copy()

                    if not np.isfinite(logits).all():
                        raise ValueError(f"Non-finite logits detected in inference result for request {req.request_id}")
                    if not np.isfinite(wdl).all() or np.any(wdl < 0) or not np.isclose(wdl.sum(), 1.):
                        raise ValueError(f"Non-finite WDL probabilities detected in inference result for request {req.request_id}")

                    results[orig_idx] = InferenceResult(
                        request_id=req.request_id,
                        model_key=req.model_key,
                        model_version=req.model_version,
                        logits=logits,
                        wdl=wdl,
                        expired=req.deadline is not None and time.monotonic()>=req.deadline,
                    )

                self._total_batches += 1
                self._total_rows += B
                self._total_candidate_padding += padding_count

        return results  # type: ignore[return-value]
