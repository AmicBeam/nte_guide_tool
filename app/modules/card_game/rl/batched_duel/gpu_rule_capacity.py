"""Pure capacity planner and slot lifetime ledger for the CUDA rule backend.

No Torch import. Byte figures are conservative formulas, never observed device
memory. Persistent graph rows stay separate from transient VM query heaps.
Live-state counts come from cooperative search plus nested terminal-horizon
probe parents, not from blindly using the historical 2048-state default.
"""
from __future__ import annotations

from collections import Counter
import json

GPU_RUNTIMES = ('recovery', 'seven')
MAX_GAMES = 1600
MAX_SEARCH_ROOTS = 1600
DEFAULT_MAX_GAMES = 1600
DEFAULT_MAX_STATES = 2048
DEFAULT_MAX_NODES = 16384
DEFAULT_MAX_EDGES = 32768
DEFAULT_MAX_PAYLOAD_BYTES = 131072
DEFAULT_MAX_CANDIDATES = 128
DEFAULT_SIMULATIONS = 32
DEFAULT_QUERY_BATCH_ROWS = 32
MAX_VM_CONSTANT_NODES = 65536
DEFAULT_TERMINAL_HORIZON = 3
ARGUMENT_MAX_NODES = 4096
ARGUMENT_MAX_EDGES = 8192
ARGUMENT_MAX_PAYLOAD_BYTES = 65536
NP_HDR_BYTES = 32
FLOAT32_BYTES = 4
ENCODE_MATRIX_COPIES = 2
ENCODE_INTERMEDIATE_COPIES = 2
NODES_PER_CANDIDATE = 48
NODE_WORKSPACE_BASE = 8192
EDGE_PER_NODE = 2
PER_CANDIDATE_PAYLOAD_SLACK = 256
PROJECTION_INTERMEDIATE_COPIES = 2
DEFAULT_UNMODELED_RESERVE_BYTES = 2 * 1024 * 1024 * 1024
KNOWN_DEVICE_NAME = 'RTX 5060 Ti'
KNOWN_DEVICE_BYTES = 16 * 1024 * 1024 * 1024
VM_OPERATION_MIN_NODES = 262144
VM_OPERATION_MIN_EDGES = 524288
VM_OPERATION_MIN_PAYLOAD_BYTES = 32 * 1024 * 1024
VM_MAX_NODES_CAP = 1048576
VM_MAX_EDGES_CAP = 2097152
VM_MAX_PAYLOAD_BYTES_CAP = 64 * 1024 * 1024
VM_FRAMES = 64
VM_LOCALS = 128
VM_STACK = 256
VM_CELLS = 64
VM_FRAME_HEADER = 14

# Per admitted root, conservative simultaneously live branch refs during a
# cooperative simulation that is also running short_terminal_probe:
# cleaned physical root, sampled/current world, one parent per remaining
# terminal-horizon step, plus overlapping step mid-state and cleaned output.
# Probe action lines are sequential; this does not multiply by legal width.
ROOT_CLEAN_SLOTS = 1
ROOT_SAMPLED_SLOTS = 1
ROOT_SIMULATED_SLOTS = 1
ROOT_TEMPORARY_SLOTS = 1

LIVE_KINDS = ('real', 'root_clean', 'sampled', 'simulated', 'temporary', 'hypothetical')
STATE_REF_ROLES = ('real', 'hypothetical')


class GpuRuleCapacityError(ValueError):
    """Fail-closed capacity or lifetime error. Never truncates."""


def normalize_runtime(runtime):
    if runtime is None:
        runtime = 'recovery'
    if runtime not in GPU_RUNTIMES:
        raise GpuRuleCapacityError("runtime must be 'recovery' or 'seven', got %r" % (runtime,))
    return runtime


def _positive_int(name, value, *, lo=1, hi=None):
    if type(value) is not int or isinstance(value, bool):
        raise TypeError('%s must be an integer, not %s' % (name, type(value).__name__))
    if value < lo or (hi is not None and value > hi):
        if hi is None:
            raise GpuRuleCapacityError('%s must be an integer >= %s, got %s' % (name, lo, value))
        raise GpuRuleCapacityError('%s must be an integer in %s..%s, got %s' % (name, lo, hi, value))
    return value


def _optional_positive_int(name, value, *, lo=1, hi=None):
    if value is None:
        return None
    return _positive_int(name, value, lo=lo, hi=hi)


def graph_state_bytes(max_nodes, max_edges, max_payload_bytes):
    """Host/device graph row size: int32 nodes/edges/counts/roots + uint8 payload."""
    max_nodes = _positive_int('max_nodes', max_nodes)
    max_edges = _positive_int('max_edges', max_edges)
    max_payload_bytes = _positive_int('max_payload_bytes', max_payload_bytes)
    nodes = max_nodes * 4 * 4
    edges = max_edges * 2 * 4
    counts = 3 * 4
    roots = 4
    return int(nodes + edges + max_payload_bytes + counts + roots)


def argument_row_bytes(max_nodes=ARGUMENT_MAX_NODES, max_edges=ARGUMENT_MAX_EDGES,
                       max_payload_bytes=ARGUMENT_MAX_PAYLOAD_BYTES):
    return graph_state_bytes(max_nodes, max_edges, max_payload_bytes)


def state_ref_role(kind):
    if kind == 'real':
        return 'real'
    if kind not in LIVE_KINDS:
        raise GpuRuleCapacityError('Unknown live branch kind %r' % (kind,))
    return 'hypothetical'


def runtime_encoder_dims(runtime, *, obs_dim=None, cand_dim=None):
    """Exact gpu_schema_contract observation/candidate widths. No unknown-runtime fallback."""
    runtime = normalize_runtime(runtime)
    if obs_dim is not None:
        obs_dim = _positive_int('obs_dim', obs_dim)
    if cand_dim is not None:
        cand_dim = _positive_int('cand_dim', cand_dim)
    try:
        from .gpu_schema_contract import GPU_RUNTIME_CONTRACTS
    except Exception as exc:
        raise GpuRuleCapacityError(
            'gpu_schema_contract is required for GPU runtime binding: %s' % exc
        ) from exc
    if not isinstance(GPU_RUNTIME_CONTRACTS, dict) or runtime not in GPU_RUNTIME_CONTRACTS:
        raise GpuRuleCapacityError(
            'gpu_schema_contract has no exact row for runtime %r' % (runtime,)
        )
    row = GPU_RUNTIME_CONTRACTS[runtime]
    if not isinstance(row, dict) or 'obs_dim' not in row or 'cand_dim' not in row:
        raise GpuRuleCapacityError(
            'gpu_schema_contract row for %r is missing obs_dim/cand_dim' % (runtime,)
        )
    try:
        contract_obs = int(row['obs_dim'])
        contract_cand = int(row['cand_dim'])
    except (TypeError, ValueError) as exc:
        raise GpuRuleCapacityError(
            'gpu_schema_contract dims for %r are not integers' % (runtime,)
        ) from exc
    if obs_dim is not None and obs_dim != contract_obs:
        raise GpuRuleCapacityError(
            'obs_dim %s does not match gpu_schema_contract %s for runtime %r'
            % (obs_dim, contract_obs, runtime)
        )
    if cand_dim is not None and cand_dim != contract_cand:
        raise GpuRuleCapacityError(
            'cand_dim %s does not match gpu_schema_contract %s for runtime %r'
            % (cand_dim, contract_cand, runtime)
        )
    schema = row.get('schema')
    schema_sha = row.get('schema_sha')
    return int(contract_obs), int(contract_cand), 'gpu_schema_contract', schema, schema_sha


def per_root_live_slots(terminal_horizon=DEFAULT_TERMINAL_HORIZON):
    terminal_horizon = _positive_int('terminal_horizon', terminal_horizon, lo=0)
    return int(
        ROOT_CLEAN_SLOTS
        + ROOT_SAMPLED_SLOTS
        + terminal_horizon
        + ROOT_SIMULATED_SLOTS
        + ROOT_TEMPORARY_SLOTS
    )


def derived_max_states(games, search_roots, *, terminal_horizon=DEFAULT_TERMINAL_HORIZON):
    """Resident real games plus simultaneously live search/probe branch refs."""
    games = _positive_int('games', games, lo=1, hi=MAX_GAMES)
    search_roots = _positive_int('search_roots', search_roots, lo=1, hi=MAX_SEARCH_ROOTS)
    admitted = min(games, search_roots)
    return int(games + per_root_live_slots(terminal_horizon) * admitted), int(admitted)


def _node_pool_bytes(node_capacity, max_candidates):
    prior = node_capacity * max_candidates * 8
    total = node_capacity * max_candidates * 8
    visits = node_capacity * max_candidates * 4
    value = node_capacity * 8
    count = node_capacity * 8
    return int(prior + total + visits + value + count)


def _vm_frame_bytes(rows, *, frames=VM_FRAMES, locals_=VM_LOCALS, stack=VM_STACK, cells=VM_CELLS):
    stride = VM_FRAME_HEADER + locals_ + stack + cells
    return int(rows * frames * stride * 4)


def split_query_batches(count, query_batch_rows=DEFAULT_QUERY_BATCH_ROWS):
    """Split a pending group into bounded kernel/query microbatches. Not a root cap."""
    if type(count) is not int or isinstance(count, bool) or count < 0:
        raise TypeError('Non-negative batch count required')
    query_batch_rows = _positive_int('query_batch_rows', query_batch_rows)
    if count == 0:
        return []
    return [min(query_batch_rows, count - start) for start in range(0, count, query_batch_rows)]


def derive_vm_workspace(
    max_nodes,
    max_edges,
    max_payload_bytes,
    max_candidates,
    obs_dim,
    cand_dim,
    *,
    vm_max_nodes=None,
    vm_max_edges=None,
    vm_max_payload_bytes=None,
):
    """Source-only bounded VM heap, including multi-view sampling and rule previews."""
    max_nodes = _positive_int('max_nodes', max_nodes)
    max_edges = _positive_int('max_edges', max_edges)
    max_payload_bytes = _positive_int('max_payload_bytes', max_payload_bytes)
    max_candidates = _positive_int('max_candidates', max_candidates)
    obs_dim = _positive_int('obs_dim', obs_dim)
    cand_dim = _positive_int('cand_dim', cand_dim)
    candidate_matrix = max_candidates * cand_dim * FLOAT32_BYTES
    observation_vector = obs_dim * FLOAT32_BYTES
    encode_payload = ENCODE_MATRIX_COPIES * (candidate_matrix + observation_vector + NP_HDR_BYTES)
    derived_payload = encode_payload * (1 + ENCODE_INTERMEDIATE_COPIES) + max_candidates * PER_CANDIDATE_PAYLOAD_SLACK
    derived_nodes = max_nodes + NODE_WORKSPACE_BASE + ENCODE_MATRIX_COPIES * max_candidates * NODES_PER_CANDIDATE
    derived_edges = max(max_edges + NODE_WORKSPACE_BASE * 2, derived_nodes * EDGE_PER_NODE)
    estimates = {
        'encode_candidate_matrix_bytes': int(candidate_matrix),
        'encode_observation_vector_bytes': int(observation_vector),
        'encode_payload_bytes': int(encode_payload),
        'derived_vm_max_nodes': int(max(max_nodes, derived_nodes, VM_OPERATION_MIN_NODES)),
        'derived_vm_max_edges': int(max(max_edges, derived_edges, VM_OPERATION_MIN_EDGES)),
        'derived_vm_max_payload_bytes': int(max(max_payload_bytes, derived_payload, VM_OPERATION_MIN_PAYLOAD_BYTES)),
    }
    chosen_nodes = estimates['derived_vm_max_nodes'] if vm_max_nodes is None else _positive_int('vm_max_nodes', vm_max_nodes)
    chosen_edges = estimates['derived_vm_max_edges'] if vm_max_edges is None else _positive_int('vm_max_edges', vm_max_edges)
    chosen_payload = estimates['derived_vm_max_payload_bytes'] if vm_max_payload_bytes is None else _positive_int('vm_max_payload_bytes', vm_max_payload_bytes)
    if chosen_nodes < max_nodes or chosen_edges < max_edges or chosen_payload < max_payload_bytes:
        raise GpuRuleCapacityError(
            'vm workspace limits must not shrink persistent graph capacity '
            '(vm %s/%s/%s vs persistent %s/%s/%s)'
            % (chosen_nodes, chosen_edges, chosen_payload, max_nodes, max_edges, max_payload_bytes)
        )
    if chosen_nodes < estimates['derived_vm_max_nodes'] or chosen_edges < estimates['derived_vm_max_edges'] or chosen_payload < estimates['derived_vm_max_payload_bytes']:
        raise GpuRuleCapacityError(
            'vm workspace limits are below the conservative encode bound '
            '(vm %s/%s/%s vs derived %s/%s/%s)'
            % (
                chosen_nodes, chosen_edges, chosen_payload,
                estimates['derived_vm_max_nodes'], estimates['derived_vm_max_edges'],
                estimates['derived_vm_max_payload_bytes'],
            )
        )
    if chosen_nodes > VM_MAX_NODES_CAP or chosen_edges > VM_MAX_EDGES_CAP or chosen_payload > VM_MAX_PAYLOAD_BYTES_CAP:
        raise GpuRuleCapacityError(
            'derived vm workspace %s/%s/%s exceeds conservative caps %s/%s/%s'
            % (chosen_nodes, chosen_edges, chosen_payload, VM_MAX_NODES_CAP, VM_MAX_EDGES_CAP, VM_MAX_PAYLOAD_BYTES_CAP)
        )
    estimates.update({
        'vm_max_nodes': int(chosen_nodes),
        'vm_max_edges': int(chosen_edges),
        'vm_max_payload_bytes': int(chosen_payload),
        'vm_state_bytes': graph_state_bytes(chosen_nodes, chosen_edges, chosen_payload),
    })
    return estimates


def _jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError('Capacity plan is not JSON-serializable: %r' % (type(value),))


def plan_gpu_rule_capacity(
    games,
    search_roots,
    *,
    memory_budget_bytes=None,
    max_nodes=DEFAULT_MAX_NODES,
    max_edges=DEFAULT_MAX_EDGES,
    max_payload_bytes=DEFAULT_MAX_PAYLOAD_BYTES,
    max_candidates=DEFAULT_MAX_CANDIDATES,
    runtime='recovery',
    simulations=DEFAULT_SIMULATIONS,
    node_capacity=None,
    obs_dim=None,
    cand_dim=None,
    query_batch_rows=None,
    vm_max_nodes=None,
    vm_max_edges=None,
    vm_max_payload_bytes=None,
    terminal_horizon=None,
    unmodeled_reserve_bytes=None,
):
    """Return a JSON-serializable capacity plan or an explicit rejection.

    max_states is derived from scheduler lifetimes. The historical default of
    2048 states is not sufficient for 1600 simultaneously active roots.
    """
    details = []
    runtime_name = runtime
    try:
        runtime_name = normalize_runtime(runtime)
        games = _positive_int('games', games, lo=1, hi=MAX_GAMES)
        search_roots = _positive_int('search_roots', search_roots, lo=1, hi=MAX_SEARCH_ROOTS)
        max_nodes = _positive_int('max_nodes', max_nodes)
        max_edges = _positive_int('max_edges', max_edges)
        max_payload_bytes = _positive_int('max_payload_bytes', max_payload_bytes)
        max_candidates = _positive_int('max_candidates', max_candidates, lo=1, hi=16384)
        simulations = _positive_int('simulations', simulations)
        memory_budget_bytes = _optional_positive_int('memory_budget_bytes', memory_budget_bytes)
        query_batch_rows = DEFAULT_QUERY_BATCH_ROWS if query_batch_rows is None else query_batch_rows
        query_batch_rows = _positive_int('query_batch_rows', query_batch_rows)
        terminal_horizon = DEFAULT_TERMINAL_HORIZON if terminal_horizon is None else terminal_horizon
        terminal_horizon = _positive_int('terminal_horizon', terminal_horizon, lo=0)
        unmodeled_reserve_bytes = DEFAULT_UNMODELED_RESERVE_BYTES if unmodeled_reserve_bytes is None else unmodeled_reserve_bytes
        unmodeled_reserve_bytes = _positive_int('unmodeled_reserve_bytes', unmodeled_reserve_bytes, lo=0)
        obs_width, cand_width, dim_source, schema, schema_sha = runtime_encoder_dims(
            runtime_name, obs_dim=obs_dim, cand_dim=cand_dim,
        )
        workspace = derive_vm_workspace(
            max_nodes, max_edges, max_payload_bytes, max_candidates, obs_width, cand_width,
            vm_max_nodes=vm_max_nodes, vm_max_edges=vm_max_edges, vm_max_payload_bytes=vm_max_payload_bytes,
        )
    except GpuRuleCapacityError as exc:
        return _jsonable({
            'valid': False,
            'rejected': True,
            'rejection_reason': str(exc),
            'rejection_details': [str(exc)],
            'runtime': runtime_name if runtime_name in GPU_RUNTIMES else None,
            'games': games if type(games) is int and not isinstance(games, bool) else None,
            'search_roots': search_roots if type(search_roots) is int and not isinstance(search_roots, bool) else None,
            'max_states': None,
            'estimated_peak_bytes': None,
            'state_bytes': None,
            'candidate_capacity': max_candidates if type(max_candidates) is int and not isinstance(max_candidates, bool) else None,
            'max_roots': None,
            'estimated_peak_bytes_are_device_observed': False,
        })

    max_states, admitted_roots = derived_max_states(games, search_roots, terminal_horizon=terminal_horizon)
    state_bytes = graph_state_bytes(max_nodes, max_edges, max_payload_bytes)
    if node_capacity is None:
        node_capacity = admitted_roots * (1 + simulations)
    else:
        node_capacity = _positive_int('node_capacity', node_capacity)
    argument_nodes = min(ARGUMENT_MAX_NODES, max_nodes)
    argument_edges = min(ARGUMENT_MAX_EDGES, max_edges)
    argument_payload = min(ARGUMENT_MAX_PAYLOAD_BYTES, max_payload_bytes)
    arg_bytes = argument_row_bytes(argument_nodes, argument_edges, argument_payload)
    lifetime = {
        'real': games,
        'root_clean': admitted_roots * ROOT_CLEAN_SLOTS,
        'sampled': admitted_roots * ROOT_SAMPLED_SLOTS,
        'simulated': admitted_roots * ROOT_SIMULATED_SLOTS,
        'temporary_live': admitted_roots * ROOT_TEMPORARY_SLOTS,
        'probe_parents': admitted_roots * terminal_horizon,
        'admitted_roots': admitted_roots,
        'requested_search_roots': search_roots,
        'query_batch_rows': query_batch_rows,
        'per_root_live_slots': per_root_live_slots(terminal_horizon),
        'terminal_horizon': terminal_horizon,
        'gpu_node_pool': node_capacity,
        'inference_pool_rows': query_batch_rows,
        'notes': (
            'Admitted roots stay live. Kernel/query workspace is query_batch_rows, not roots. '
            'max_states includes real games, cleaned roots, sampled worlds, terminal-horizon '
            'probe parents, and overlapping step mid/output refs.'
        ),
    }
    resident_pool_bytes = max_states * state_bytes
    vm_state_bytes = workspace['vm_state_bytes']
    gather_bytes = query_batch_rows * state_bytes
    bind_dest_bytes = query_batch_rows * vm_state_bytes
    argument_heap_bytes = query_batch_rows * arg_bytes
    vm_scratch_bytes = query_batch_rows * vm_state_bytes
    compact_workspace_bytes = query_batch_rows * vm_state_bytes
    compact_persistent_bytes = query_batch_rows * state_bytes
    compact_bytes = compact_workspace_bytes + compact_persistent_bytes
    frame_bytes = _vm_frame_bytes(query_batch_rows)
    gc_bytes = query_batch_rows * (workspace['vm_max_nodes'] * 12 + MAX_VM_CONSTANT_NODES * 8)
    workspace_bytes = (
        gather_bytes + bind_dest_bytes + argument_heap_bytes + vm_scratch_bytes
        + compact_bytes + frame_bytes + gc_bytes
    )
    observation_batch_bytes = query_batch_rows * obs_width * FLOAT32_BYTES
    candidate_batch_bytes = query_batch_rows * max_candidates * cand_width * FLOAT32_BYTES
    projection_bytes = (observation_batch_bytes + candidate_batch_bytes) * PROJECTION_INTERMEDIATE_COPIES
    node_pool_bytes = _node_pool_bytes(node_capacity, max_candidates)
    inference_bytes = observation_batch_bytes + candidate_batch_bytes
    modeled_bytes = int(
        resident_pool_bytes + workspace_bytes + projection_bytes + node_pool_bytes + inference_bytes
    )
    estimated_peak_bytes = int(modeled_bytes + unmodeled_reserve_bytes)
    default_2048_sufficient = max_states <= DEFAULT_MAX_STATES
    if not default_2048_sufficient:
        details.append(
            'derived max_states=%s exceeds historical default %s; 2048 cannot host %s admitted roots'
            % (max_states, DEFAULT_MAX_STATES, admitted_roots)
        )
    valid = True
    rejection_reason = None
    if memory_budget_bytes is not None and estimated_peak_bytes > memory_budget_bytes:
        valid = False
        rejection_reason = (
            'conservative peak %s bytes exceeds memory_budget_bytes %s; '
            'refusing to truncate states, candidates, or heap limits'
            % (estimated_peak_bytes, memory_budget_bytes)
        )
        details.append(rejection_reason)
    fits_known_device = estimated_peak_bytes <= KNOWN_DEVICE_BYTES
    if not fits_known_device:
        details.append(
            'conservative peak %s bytes exceeds known %s capacity %s; formula is not device coverage'
            % (estimated_peak_bytes, KNOWN_DEVICE_NAME, KNOWN_DEVICE_BYTES)
        )
    return _jsonable({
        'valid': valid,
        'rejected': not valid,
        'rejection_reason': rejection_reason,
        'rejection_details': details,
        'runtime': runtime_name,
        'games': games,
        'search_roots': search_roots,
        'max_games': games,
        'max_roots': admitted_roots,
        'max_states': max_states,
        'state_bytes': state_bytes,
        'candidate_capacity': max_candidates,
        'max_nodes': max_nodes,
        'max_edges': max_edges,
        'max_payload_bytes': max_payload_bytes,
        'max_candidates': max_candidates,
        'argument_max_nodes': argument_nodes,
        'argument_max_edges': argument_edges,
        'argument_max_payload_bytes': argument_payload,
        'obs_dim': obs_width,
        'cand_dim': cand_width,
        'encoder_dim_source': dim_source,
        'schema': schema,
        'schema_sha': schema_sha,
        'query_batch_rows': query_batch_rows,
        'vm_max_nodes': workspace['vm_max_nodes'],
        'vm_max_edges': workspace['vm_max_edges'],
        'vm_max_payload_bytes': workspace['vm_max_payload_bytes'],
        'vm_state_bytes': vm_state_bytes,
        'terminal_horizon': terminal_horizon,
        'unmodeled_reserve_bytes': unmodeled_reserve_bytes,
        'encode_workspace': {
            'encode_candidate_matrix_bytes': workspace['encode_candidate_matrix_bytes'],
            'encode_observation_vector_bytes': workspace['encode_observation_vector_bytes'],
            'encode_payload_bytes': workspace['encode_payload_bytes'],
            'derived_vm_max_nodes': workspace['derived_vm_max_nodes'],
            'derived_vm_max_edges': workspace['derived_vm_max_edges'],
            'derived_vm_max_payload_bytes': workspace['derived_vm_max_payload_bytes'],
        },
        'simulations': simulations,
        'node_capacity': node_capacity,
        'lifetime': lifetime,
        'components': {
            'resident_pool_bytes': resident_pool_bytes,
            'batch_workspace_bytes': workspace_bytes,
            'gather_bytes': gather_bytes,
            'bind_dest_bytes': bind_dest_bytes,
            'argument_heap_bytes': argument_heap_bytes,
            'vm_scratch_bytes': vm_scratch_bytes,
            'compact_bytes': compact_bytes,
            'vm_frame_bytes': frame_bytes,
            'gc_bytes': gc_bytes,
            'observation_batch_bytes': observation_batch_bytes,
            'candidate_batch_bytes': candidate_batch_bytes,
            'projection_bytes': projection_bytes,
            'node_pool_bytes': node_pool_bytes,
            'inference_pool_bytes': inference_bytes,
        },
        'unmodeled_reserve_bytes': unmodeled_reserve_bytes,
        'estimated_modeled_bytes': modeled_bytes,
        'estimated_peak_bytes': estimated_peak_bytes,
        'observed_peak_bytes': None,
        'estimated_peak_bytes_are_device_observed': False,
        'known_device_name': KNOWN_DEVICE_NAME,
        'known_device_bytes': KNOWN_DEVICE_BYTES,
        'fits_known_device': fits_known_device,
        'device_coverage_claimed': False,
        'memory_budget_bytes': memory_budget_bytes,
        'default_2048_states_sufficient': default_2048_sufficient,
        'derived_from_scheduler_lifetimes': True,
        'notes': (
            'estimated_peak_bytes is a conservative formula over resident slots, '
            'query-batch VM workspace, projection/inference intermediates, GPU node '
            'pools, and an unmodeled image/model/compiler reserve. observed bytes are '
            'reported separately by the backend allocator and are not this formula. '
            'This is not measured device memory and does not prove 1600-root coverage.'
        ),
    })


def validate_capacity_plan(plan):
    if not isinstance(plan, dict):
        raise GpuRuleCapacityError('capacity_plan must be a JSON object')
    required = (
        'valid', 'runtime', 'max_states', 'max_games', 'max_roots', 'state_bytes',
        'candidate_capacity', 'max_nodes', 'max_edges', 'max_payload_bytes',
        'estimated_peak_bytes', 'query_batch_rows', 'vm_max_nodes', 'vm_max_edges',
        'vm_max_payload_bytes', 'node_capacity', 'terminal_horizon', 'obs_dim', 'cand_dim',
    )
    missing = [key for key in required if key not in plan]
    if missing:
        raise GpuRuleCapacityError('capacity_plan missing fields: %s' % ','.join(missing))
    if plan.get('valid') is not True or plan.get('rejected') is True:
        raise GpuRuleCapacityError(
            'capacity_plan was rejected: %s' % (plan.get('rejection_reason') or 'invalid plan',)
        )
    normalize_runtime(plan['runtime'])
    for key in ('max_states', 'max_games', 'max_roots', 'state_bytes', 'candidate_capacity',
                'max_nodes', 'max_edges', 'max_payload_bytes', 'query_batch_rows',
                'vm_max_nodes', 'vm_max_edges', 'vm_max_payload_bytes', 'node_capacity',
                'obs_dim', 'cand_dim'):
        _positive_int(key, plan[key])
    _positive_int('terminal_horizon', plan['terminal_horizon'], lo=0)
    _positive_int('estimated_peak_bytes', plan['estimated_peak_bytes'], lo=0)
    if plan.get('estimated_peak_bytes_are_device_observed') is True:
        raise GpuRuleCapacityError('capacity_plan must not claim observed device memory')
    if plan.get('device_coverage_claimed') is True:
        raise GpuRuleCapacityError('capacity_plan must not claim device coverage from formulas')
    if int(plan['max_games']) > MAX_GAMES or int(plan['max_roots']) > MAX_SEARCH_ROOTS:
        raise GpuRuleCapacityError('capacity_plan exceeds 1600 game/root admission')
    expected, admitted = derived_max_states(
        int(plan['max_games']),
        int(plan.get('search_roots', plan['max_roots'])),
        terminal_horizon=int(plan['terminal_horizon']),
    )
    if int(plan['max_states']) < expected:
        raise GpuRuleCapacityError(
            'capacity_plan max_states %s is below derived lifetime requirement %s'
            % (plan['max_states'], expected)
        )
    if int(plan['max_roots']) != admitted:
        raise GpuRuleCapacityError('capacity_plan max_roots does not match admitted search roots')
    if int(plan['vm_max_nodes']) < int(plan['max_nodes']) or int(plan['vm_max_edges']) < int(plan['max_edges']) or int(plan['vm_max_payload_bytes']) < int(plan['max_payload_bytes']):
        raise GpuRuleCapacityError('capacity_plan vm workspace shrinks persistent graph capacity')
    if plan.get('encoder_dim_source') not in (None, 'gpu_schema_contract'):
        raise GpuRuleCapacityError('capacity_plan runtime binding must be gpu_schema_contract')
    return plan


def resolve_gpu_rule_backend_capacity(
    *,
    runtime='recovery',
    capacity_plan=None,
    max_games=DEFAULT_MAX_GAMES,
    max_states=DEFAULT_MAX_STATES,
    max_nodes=DEFAULT_MAX_NODES,
    max_edges=DEFAULT_MAX_EDGES,
    max_payload_bytes=DEFAULT_MAX_PAYLOAD_BYTES,
    max_candidates=DEFAULT_MAX_CANDIDATES,
):
    """Adopt a planner result or keep explicit historical constructor sizes."""
    runtime = normalize_runtime(runtime)
    max_games = _positive_int('max_games', max_games)
    max_states = _positive_int('max_states', max_states)
    max_nodes = _positive_int('max_nodes', max_nodes)
    max_edges = _positive_int('max_edges', max_edges)
    max_payload_bytes = _positive_int('max_payload_bytes', max_payload_bytes)
    max_candidates = _positive_int('max_candidates', max_candidates, lo=1, hi=16384)
    if capacity_plan is None:
        return {
            'runtime': runtime,
            'derived_from_plan': False,
            'max_games': max_games,
            'max_states': max_states,
            'max_roots': None,
            'max_nodes': max_nodes,
            'max_edges': max_edges,
            'max_payload_bytes': max_payload_bytes,
            'max_candidates': max_candidates,
            'candidate_capacity': max_candidates,
            'state_bytes': graph_state_bytes(max_nodes, max_edges, max_payload_bytes),
            'argument_max_nodes': min(ARGUMENT_MAX_NODES, max_nodes),
            'argument_max_edges': min(ARGUMENT_MAX_EDGES, max_edges),
            'argument_max_payload_bytes': min(ARGUMENT_MAX_PAYLOAD_BYTES, max_payload_bytes),
            'query_batch_rows': DEFAULT_QUERY_BATCH_ROWS,
            'vm_max_nodes': max_nodes,
            'vm_max_edges': max_edges,
            'vm_max_payload_bytes': max_payload_bytes,
            'terminal_horizon': DEFAULT_TERMINAL_HORIZON,
            'obs_dim': None,
            'cand_dim': None,
            'estimated_peak_bytes': None,
            'observed_peak_bytes': None,
            'estimated_peak_bytes_are_device_observed': False,
            'device_coverage_claimed': False,
            'default_2048_states_sufficient': None,
            'plan': None,
            'notes': 'legacy explicit sizes; default 2048 states is not a 1600-root proof',
        }
    plan = validate_capacity_plan(capacity_plan)
    if plan['runtime'] != runtime:
        raise GpuRuleCapacityError(
            'capacity_plan runtime %r does not match backend runtime %r' % (plan['runtime'], runtime)
        )
    defaults = {
        'max_games': DEFAULT_MAX_GAMES,
        'max_states': DEFAULT_MAX_STATES,
        'max_nodes': DEFAULT_MAX_NODES,
        'max_edges': DEFAULT_MAX_EDGES,
        'max_payload_bytes': DEFAULT_MAX_PAYLOAD_BYTES,
        'max_candidates': DEFAULT_MAX_CANDIDATES,
    }
    supplied = {
        'max_games': max_games,
        'max_states': max_states,
        'max_nodes': max_nodes,
        'max_edges': max_edges,
        'max_payload_bytes': max_payload_bytes,
        'max_candidates': max_candidates,
    }
    mismatches = []
    for key, value in supplied.items():
        planned = plan.get('candidate_capacity' if key == 'max_candidates' else key)
        if value != defaults[key] and int(value) != int(planned):
            mismatches.append('%s explicit %s != plan %s' % (key, value, planned))
    if mismatches:
        raise GpuRuleCapacityError(
            'explicit sizes disagree with capacity_plan: %s' % '; '.join(mismatches)
        )
    return {
        'runtime': runtime,
        'derived_from_plan': True,
        'max_games': int(plan['max_games']),
        'max_states': int(plan['max_states']),
        'max_roots': int(plan['max_roots']),
        'max_nodes': int(plan['max_nodes']),
        'max_edges': int(plan['max_edges']),
        'max_payload_bytes': int(plan['max_payload_bytes']),
        'max_candidates': int(plan['candidate_capacity']),
        'candidate_capacity': int(plan['candidate_capacity']),
        'state_bytes': int(plan['state_bytes']),
        'argument_max_nodes': int(plan.get('argument_max_nodes', min(ARGUMENT_MAX_NODES, plan['max_nodes']))),
        'argument_max_edges': int(plan.get('argument_max_edges', min(ARGUMENT_MAX_EDGES, plan['max_edges']))),
        'argument_max_payload_bytes': int(plan.get('argument_max_payload_bytes', min(ARGUMENT_MAX_PAYLOAD_BYTES, plan['max_payload_bytes']))),
        'query_batch_rows': int(plan['query_batch_rows']),
        'vm_max_nodes': int(plan['vm_max_nodes']),
        'vm_max_edges': int(plan['vm_max_edges']),
        'vm_max_payload_bytes': int(plan['vm_max_payload_bytes']),
        'terminal_horizon': int(plan['terminal_horizon']),
        'obs_dim': int(plan['obs_dim']),
        'cand_dim': int(plan['cand_dim']),
        'estimated_modeled_bytes': int(plan.get('estimated_modeled_bytes', plan['estimated_peak_bytes'])),
        'unmodeled_reserve_bytes': int(plan.get('unmodeled_reserve_bytes', 0)),
        'estimated_peak_bytes': int(plan['estimated_peak_bytes']),
        'observed_peak_bytes': None,
        'estimated_peak_bytes_are_device_observed': False,
        'device_coverage_claimed': False,
        'default_2048_states_sufficient': bool(plan.get('default_2048_states_sufficient')),
        'plan': plan,
        'notes': plan.get('notes'),
    }


def dumps_capacity_plan(plan):
    return json.dumps(_jsonable(plan), sort_keys=True)


def trace_terminal_probe_liveness(
    ledger,
    *,
    games,
    search_roots,
    terminal_horizon=DEFAULT_TERMINAL_HORIZON,
):
    """Allocate nested probe refs to the configured horizon. Not a device proof."""
    if not isinstance(ledger, GpuRuleSlotLedger):
        raise TypeError('GpuRuleSlotLedger required')
    games = _positive_int('games', games, lo=1, hi=MAX_GAMES)
    search_roots = _positive_int('search_roots', search_roots, lo=1, hi=MAX_SEARCH_ROOTS)
    terminal_horizon = _positive_int('terminal_horizon', terminal_horizon, lo=0)
    admitted = min(games, search_roots)
    game_ids = ['g-%s' % i for i in range(games)]
    ledger.admit_real_games(game_ids)
    real_keys = []
    for game_id in game_ids:
        slot = ledger.allocate(1, 'real')[0]
        real_keys.append(ledger.register(slot, 'real'))
    depths = []
    held = []
    for index in range(admitted):
        clean = ledger.register(ledger.allocate(1, 'root_clean')[0], 'root_clean')
        sampled = ledger.register(ledger.allocate(1, 'sampled')[0], 'sampled')
        parents = []
        current = sampled
        for depth in range(terminal_horizon):
            mid = ledger.register(ledger.allocate(1, 'simulated')[0], 'simulated')
            output = ledger.register(ledger.allocate(1, 'temporary')[0], 'temporary')
            parents.append(current)
            current = output
            depths.append(depth + 1)
            if depth + 1 < terminal_horizon:
                ledger.release_key(mid)
            else:
                held.append(mid)
        held.extend([clean, sampled, current])
        held.extend(parents)
    metrics = ledger.metrics()
    metrics['trace_depth'] = max(depths, default=0)
    metrics['configured_horizon'] = terminal_horizon
    metrics['admitted_roots'] = admitted
    metrics['real_games'] = len(real_keys)
    metrics['formula_max_states'] = games + per_root_live_slots(terminal_horizon) * admitted
    return metrics


class GpuRuleSlotLedger:
    """Pure CPU ledger for GPU state-slot allocate/release/poison lifetimes."""

    def __init__(self, max_states, max_games):
        self.max_states = _positive_int('max_states', max_states)
        self.max_games = _positive_int('max_games', max_games)
        self.free = list(range(self.max_states - 1, -1, -1))
        self.generations = [0] * self.max_states
        self.live = {}
        self.kinds = {}
        self.kind_counts = Counter()
        self.closed = False
        self.poisoned = False
        self.released_count = 0
        self.peak = 0
        self.real_games = set()

    def check(self):
        if self.poisoned:
            raise RuntimeError('Backend is poisoned from a previous failure and cannot be reused')
        if self.closed:
            raise RuntimeError('Backend is closed')

    def abort(self):
        self.poisoned = True
        self.close()

    def close(self):
        self.closed = True
        for key in list(self.live):
            self._release_key(key, require_live=False)
        self.live.clear()
        self.kinds.clear()
        self.kind_counts.clear()
        self.real_games.clear()
        self.free = []

    def allocate(self, n, kind='hypothetical'):
        self.check()
        if kind not in LIVE_KINDS:
            raise GpuRuleCapacityError('Unknown live branch kind %r' % (kind,))
        if type(n) is not int or isinstance(n, bool) or n < 0:
            raise TypeError('Non-negative slot count required')
        if n == 0:
            return []
        if n > len(self.free):
            self.abort()
            raise OverflowError('GPU state slot capacity exceeded')
        return [self.free.pop() for _ in range(n)]

    def register(self, slot, kind):
        self.check()
        if kind not in LIVE_KINDS:
            raise GpuRuleCapacityError('Unknown live branch kind %r' % (kind,))
        if type(slot) is not int or isinstance(slot, bool) or not 0 <= slot < self.max_states:
            raise ValueError('Slot out of range')
        key = (0, slot, self.generations[slot])
        if key in self.live:
            raise GpuRuleCapacityError('Slot already live')
        self.live[key] = kind
        self.kinds[slot] = kind
        self.kind_counts[kind] += 1
        if len(self.live) > self.peak:
            self.peak = len(self.live)
        return key

    def admit_real_games(self, game_ids):
        self.check()
        if len(self.real_games) + len(game_ids) > self.max_games:
            raise OverflowError('Real game admission capacity exceeded')
        for game_id in game_ids:
            if game_id in self.real_games:
                raise ValueError('A game can be imported only once in this backend')
        self.real_games.update(game_ids)

    def validate_key(self, key):
        self.check()
        if key not in self.live:
            raise ValueError('Stale or released state reference')
        return self.live[key]

    def release_key(self, key):
        self.check()
        return self._release_key(key, require_live=True)

    def release_keys(self, keys):
        self.check()
        keys = list(keys)
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate state release')
        for key in keys:
            self.validate_key(key)
        for key in keys:
            self._release_key(key, require_live=True)

    def _release_key(self, key, *, require_live):
        if key not in self.live:
            if require_live:
                raise ValueError('Stale or released state reference')
            return False
        kind = self.live.pop(key)
        worker, slot, generation = key
        if self.kinds.get(slot) == kind:
            self.kinds.pop(slot, None)
        if self.kind_counts[kind]:
            self.kind_counts[kind] -= 1
        if self.generations[slot] == generation:
            self.generations[slot] += 1
            if not self.closed:
                self.free.append(slot)
        self.released_count += 1
        return True

    def metrics(self):
        return {
            'max_states': self.max_states,
            'max_games': self.max_games,
            'live_states': 0 if self.closed else len(self.live),
            'peak_states': self.peak,
            'released_states': self.released_count,
            'free_slots': 0 if self.closed else len(self.free),
            'allocated_slots': 0 if self.closed else (self.max_states - len(self.free)),
            'live_by_kind': {kind: int(self.kind_counts[kind]) for kind in LIVE_KINDS},
            'closed': self.closed,
            'poisoned': self.poisoned,
        }
