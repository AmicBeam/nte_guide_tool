"""Prepare same-rule input for complete native array compilation.

The portable image is compiler INPUT only. The native executor must compile
every PC and refuse interpreter fallback; storing an image does not grant
that property or any CUDA qualification. No production module is monkeypatched.
"""
from hashlib import sha256
from pathlib import Path
import types

from ..rule_ir.portable_program import ProgramExporter, FrozenRuleProgram, ValidationError, compute_identity
from .gpu_rule_program import ALLOWED_MODULE_PREFIXES, verify_source_hashes, detect_frozen_runtime
from ...engine.duel_v2.flow import apply_action as _formal_apply_action


def checked_apply_action(state, side, action):
    """Return expected public API rejections through the native result ABI."""
    try:
        return True, _formal_apply_action(state, side, action)
    except ValueError as error:
        return False, 'ValueError', error.args
    except TypeError as error:
        return False, 'TypeError', error.args


def load_native_array_program(path, *, repo_root=None):
    """Load a frozen 3.10 program on any supported host; verify its exact inputs."""
    program = FrozenRuleProgram.from_json(Path(path).read_text(encoding='utf8'))
    program.validate()
    meta = program.data.get('native_array') or {}
    if (meta.get('schema') != 'native_array_rules_v1' or meta.get('runtime') != 'seven'
            or meta.get('interpreter_fallback') is not False
            or meta.get('complete_native_compilation_required') is not True
            or type(meta.get('teacher_entries')) is not bool
            or type(meta.get('policy_query_cache',False)) is not bool
            or meta.get('extra_zhenhong_simulations') != 0):
        raise ValidationError('Native array frozen protocol mismatch')
    if detect_frozen_runtime(program) != 'seven':
        raise ValidationError('Native array sampler runtime mismatch')
    required = {'initialize', 'acting_side', 'finished', 'terminal_value',
                'public_legal_decision', 'apply_action', 'simulate_action', 'clean',
                'sample_world', 'sample_world_guard_views_packet',
                'sample_world_transition_packet', 'checked_apply_action'}
    if not required <= set(program.entries):
        raise ValidationError('Native array frozen entries incomplete')
    teachers = {'teacher_continue', 'rule_decision'}
    if (not meta['teacher_entries'] and teachers & set(program.entries)
            or meta['teacher_entries'] and not teachers <= set(program.entries)):
        raise ValidationError('Native array teacher entry identity mismatch')
    if any(name.startswith('zhenhong_guide') for name in program.entries):
        raise ValidationError('Simulation guide is absent from native array protocol')
    verify_source_hashes(program, repo_root=repo_root)
    return program


class NativeArrayExporter(ProgramExporter):
    """Resolve explicit function substitutions throughout one isolated freeze."""
    def __init__(self, replacements, **kwargs):
        super().__init__(**kwargs)
        self.replacements = dict(replacements)
        if any(not isinstance(a, types.FunctionType) or not isinstance(b, types.FunctionType)
               for a, b in self.replacements.items()):
            raise TypeError('Exact source-function substitutions required')
        self.substituted = set()

    def queue_function(self, fn):
        original = fn.__func__ if isinstance(fn, (staticmethod, classmethod)) else fn
        replacement = self.replacements.get(original) if isinstance(original, types.FunctionType) else None
        ref = super().queue_function(replacement or fn)
        if replacement is not None:
            self.substituted.add(original)
            # Relative imports and function globals must resolve to the same
            # prepared callable, including source aliases imported elsewhere.
            if ref is not None:
                table = self.modules.setdefault(original.__module__, {})
                for alias, value in tuple(original.__globals__.items()):
                    if value is original:
                        table[alias] = ref
        return ref


def capture_native_array_program(*, include_teacher_entries=False, owned_transitions=True,policy_query_cache=False):
    """Freeze public-seven rules; teachers are absent by default, not disabled later.

    State transfer/compaction and typed observation encoders use the existing
    checked array ABI. Sample packets still require BOTH complete encoded
    information guards before a world can be consumed by search.
    """
    if any(type(value) is not bool for value in (include_teacher_entries,owned_transitions,policy_query_cache)):
        raise TypeError('Explicit native-array capture options required')
    from ...content.duel_v2 import CARDS, PUBLIC_CHARACTER_IDS
    from ...engine.duel_v2 import state, flow, simulation
    from .array_rule_sources import array_reduce_max_hp
    from .gpu_owned_transitions import scratch_apply_action, scratch_simulate_action
    from .gpu_policy_projection import policy_public_legal_decision
    from .gpu_sample_guard_views import seven_sample_guard_views_packet
    from .gpu_sample_transition import seven_sample_transition_packet
    from .gpu_rule_entries import entry_functions
    all_entries = entry_functions('seven')
    names = ('initialize', 'acting_side', 'finished', 'terminal_value',
             'public_legal_decision', 'apply_action', 'simulate_action', 'clean',
             'sample_world', 'serving_root_metadata')
    entries = {name: all_entries[name] for name in names}
    entries['public_legal_decision'] = policy_public_legal_decision
    if policy_query_cache:
        from .gpu_cached_policy_projection import cached_policy_public_legal_decision
        entries['public_legal_decision']=cached_policy_public_legal_decision
    entries['sample_world_guard_views_packet'] = seven_sample_guard_views_packet
    entries['sample_world_transition_packet'] = seven_sample_transition_packet
    entries['checked_apply_action'] = checked_apply_action
    if include_teacher_entries:
        entries.update({name: all_entries[name] for name in ('rule_decision', 'teacher_continue')})
    replacements = {state.reduce_max_hp: array_reduce_max_hp}
    if owned_transitions:
        replacements.update({flow.apply_action: scratch_apply_action,
                             simulation.simulate_action: scratch_simulate_action})
    characters = tuple(PUBLIC_CHARACTER_IDS)
    cards = tuple(cid for cid, card in CARDS.items() if card['character_id'] in characters)
    exporter = NativeArrayExporter(replacements, allowed_module_prefixes=ALLOWED_MODULE_PREFIXES,
        scope_character_ids=characters, scope_card_ids=cards)
    program = exporter.run_capture(entries)
    missing = set(replacements) - exporter.substituted
    if missing:
        raise ValueError('Native-array source substitution was not reached: ' +
                         ','.join(sorted(fn.__qualname__ for fn in missing)))
    source_names=['native_array_program.py','array_rule_sources.py','gpu_owned_transitions.py','gpu_policy_projection.py']
    if policy_query_cache:
        source_names.append('gpu_cached_policy_projection.py')
    for name in source_names:
        p = Path(__file__).with_name(name)
        rel = p.relative_to(exporter.repo_root).as_posix()
        program.data['source_hashes'][rel] = sha256(p.read_bytes()).hexdigest()
    program.data['native_array'] = dict(schema='native_array_rules_v1',
        runtime='seven', characters=sorted(characters), cards=sorted(cards),
        owned_workspace=owned_transitions, teacher_entries=include_teacher_entries,
        policy_query_cache=policy_query_cache,
        extra_zhenhong_simulations=0, complete_native_compilation_required=True,
        interpreter_fallback=False, gpu_qualified=False)
    program.data['identity'] = compute_identity(program.data)
    program.validate()
    return program
