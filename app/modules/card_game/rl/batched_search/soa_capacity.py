"""Import-safe resource contract for frozen-weight SoA capacity experiments."""
from dataclasses import asdict, dataclass
import math


CHANNEL_BYTES = {'i64': 8, 'f64': 8, 'i32': 4, 'u8': 1}
DEFAULT_COLUMN_CELLS = {'i64': 4096, 'f64': 1024, 'i32': 16384, 'u8': 32768}
MAX_RULE_LANES = 64


def rule_workspace_minimum_bytes(lanes, *, schema_lengths=None, total_objects=0,
                                 value_cache_entries=0, constant_nodes=0,
                                 max_strings=0, frames=64, local_cells=128,
                                 stack_cells=256, closure_cells=64):
    """Known shared scratch/typed allocations; excludes drivers/models/archives.

    This is a rejection lower bound, never a claim that a batch fits. Runtime
    whole-device checks still apply after all graphs and stores are allocated.
    """
    _integer('lanes',lanes,1,MAX_RULE_LANES)
    for name,value in (('total_objects',total_objects),('value_cache_entries',value_cache_entries),
                       ('constant_nodes',constant_nodes),('max_strings',max_strings),
                       ('frames',frames),('local_cells',local_cells),
                       ('stack_cells',stack_cells),('closure_cells',closure_cells)):
        _integer(name,value,1 if name=='frames' else 0,2**31-1)
    nodes,edges,payload=1048576,2097152,33554432
    row=nodes*16+edges*8+payload+16
    row+=frames*(14+local_cells+stack_cells+closure_cells)*4
    row+=(nodes+constant_nodes)*8+nodes*4
    row+=(total_objects*3+value_cache_entries+constant_nodes)*4
    if schema_lengths is not None:
        if set(schema_lengths)!=set(CHANNEL_BYTES):
            raise ValueError('All four typed columns required')
        for name,value in schema_lengths.items():
            _integer(name,value,0,2**31-1)
        # Shared input plus compactor destination and its mapping tables.
        row+=2*(sum(schema_lengths[k]*v for k,v in CHANNEL_BYTES.items())+4)
        row+=total_objects*9+max_strings*4+4
    return lanes*row


def check_workspace_budget(config, workspace_bytes, *, resident_bytes=0):
    for name,value in (('workspace_bytes',workspace_bytes),('resident_bytes',resident_bytes)):
        _integer(name,value,0,2**63-1)
    if workspace_bytes+resident_bytes > config.gpu_budget_mib << 20:
        raise MemoryError('Known rule workspace plus resident device memory exceeds GPU budget')
    return dict(rule_workspace_minimum_bytes=workspace_bytes,resident_device_bytes=resident_bytes,
                remaining_budget_bytes=(config.gpu_budget_mib<<20)-workspace_bytes-resident_bytes,
                fit_qualified=False)


def _integer(name, value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')


def legacy_dense_state_bytes(schema_lengths, roots):
    """Exact old arena allocation, not a measured capacity or qualification."""
    _integer('roots', roots, 1, 1600)
    if set(schema_lengths) != set(CHANNEL_BYTES):
        raise ValueError('All four typed columns required')
    for key, length in schema_lengths.items():
        _integer(key, length, 0, 2**31-1)
    return (roots*6+8)*(sum(schema_lengths[k]*v for k, v in CHANNEL_BYTES.items())+29)


def archive_capacities(config, decisions):
    """Bound ragged archives by bytes, not every record's maximum action count.

    Appends still reject exhausted prefixes atomically. A smaller byte arena
    never removes candidates or manufactures successful/truncated records.
    """
    _integer('decisions', decisions, 1, config.max_decisions)
    from .device_replay_policy_tape import _layout_bytes
    rows, records = config.roots, config.roots*decisions
    policy_bytes = config.policy_tape_budget_mib << 20
    fixed_policy = _layout_bytes(rows, records+1, 1, config.max_actions)
    legal_entries = min(records*config.max_actions, (policy_bytes-fixed_policy)//20)
    raw_bytes = config.raw_tape_budget_mib << 20
    fixed_raw = 64+rows*37+(records+1)*100+25
    available = raw_bytes-fixed_raw
    nodes = min(records*1024, available//32)
    edges = min(records*1024, available//32)
    payload = min(records*4096, available//4)
    if min(legal_entries, nodes, edges, payload) < 1:
        raise MemoryError('Archive metadata and one complete prefix must fit the declared byte budgets')
    return dict(max_records=records, max_legal_entries=legal_entries,
                node_capacity=nodes, edge_capacity=edges, payload_capacity=payload,
                policy_bytes=_layout_bytes(rows, records+1, legal_entries+1, config.max_actions),
                raw_bytes=fixed_raw+nodes*16+edges*8+payload)


@dataclass(frozen=True)
class SoACapacityConfig:
    roots: int = 8
    rule_lanes: int = 8
    inference_rows: int = 2
    max_actions: int = 800
    max_decisions: int = 400
    state_slots_per_root: int = 6
    state_reserve_slots: int = 8
    action_slots_per_root: int = 6
    action_reserve_slots: int = 128
    gpu_budget_mib: int = 14336
    state_budget_mib: int = 4096
    action_budget_mib: int = 512
    raw_tape_budget_mib: int = 512
    policy_tape_budget_mib: int = 128
    training_tape_budget_mib: int = 512
    execution_seconds: float = 1800.
    proof_seconds: float = 600.

    def __post_init__(self):
        for name, low, high in (
            ('roots', 1, 1600), ('rule_lanes', 1, MAX_RULE_LANES), ('inference_rows', 1, 32),
            ('max_actions', 1, 4096), ('max_decisions', 1, 10000),
            ('state_slots_per_root', 1, 16), ('state_reserve_slots', 0, 1600),
            ('action_slots_per_root', 1, 16), ('action_reserve_slots', 0, 1600),
            ('gpu_budget_mib', 1, 2**20), ('state_budget_mib', 1, 2**20),
            ('action_budget_mib', 1, 2**20), ('raw_tape_budget_mib', 1, 2**20),
            ('policy_tape_budget_mib', 1, 2**20),
            ('training_tape_budget_mib', 1, 2**20),
        ):
            _integer(name, getattr(self, name), low, high)
        for name in ('execution_seconds', 'proof_seconds'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 86400:
                raise ValueError(name+' must be finite and in (0,86400]')
        if self.state_budget_mib > self.gpu_budget_mib or self.action_budget_mib > self.gpu_budget_mib:
            raise ValueError('Each arena must fit the declared total GPU budget')
        check_workspace_budget(self,rule_workspace_minimum_bytes(self.lanes))

    @property
    def lanes(self):
        return min(self.roots, self.rule_lanes)

    @property
    def inference_tile(self):
        return min(self.roots, self.inference_rows)

    @property
    def state_slots(self):
        return self.roots*self.state_slots_per_root+self.state_reserve_slots

    @property
    def action_slots(self):
        return self.roots*self.action_slots_per_root+self.action_reserve_slots

    def plan(self):
        return dict(schema='soa_capacity_h0_frozen_v1', **asdict(self),
            effective_rule_lanes=self.lanes, effective_inference_rows=self.inference_tile,
            rule_workspace_minimum_bytes=rule_workspace_minimum_bytes(self.lanes),
            distinct_roots_required=True,full_module_required=True,minimum_device_stack_bytes=131072,
            state_slots=self.state_slots, action_slots=self.action_slots,
            simulations=32, root_cap=16, max_depth=10, terminal_horizon=0,
            teacher_mode='network_value', opponent_mode='policy',
            world_sampling='per_decision', zhenhong_prior='array_root_logits',
            frozen_weights=True, optimizer_steps=0, hotpath_cpu_rules=False,
            gpu_qualified=False, capacity_qualified=False)
