"""Build, load, save and validate a frozen GPU-rule source program.

This module compiles canonical formal-rule functions into a FrozenRuleProgram.
It is not a CUDA rule runtime: CPU wrappers are capture inputs, full_duel_rules
stays False, and a missing opcode/intrinsic is reported rather than replaced.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.modules.card_game.rl.batched_duel.gpu_rule_entries import (
    ENTRY_FUNCTIONS,
    FULL_ENTRY_NAMES,
    entry_functions,
    sampler_runtime,
)
from app.modules.card_game.rl.batched_duel.gpu_schema_contract import (
    DEFAULT_RUNTIME,
    GPU_RUNTIME_CONTRACTS,
    GpuSchemaContractError,
    normalize_runtime,
    runtime_metadata,
)
from app.modules.card_game.rl.rule_ir.portable_program import (
    DEFAULT_ALLOWED_MODULE_PREFIXES,
    DEFAULT_EXPLICIT_INTRINSICS,
    ExportError,
    FrozenRuleProgram,
    ValidationError,
    capture_program,
    find_repo_root,
)

PROGRAM_FILENAME = 'frozen_rule_program.json'
MANIFEST_FILENAME = 'identity_source_manifest.json'
ALLOWED_MODULE_PREFIXES = tuple(DEFAULT_ALLOWED_MODULE_PREFIXES) + (
    'app.modules.card_game.engine.ai',
)
HEAVY_ENTRIES = frozenset({
    'public_legal_decision',
    'apply_action',
    'simulate_action',
    'sample_world',
    'public_terminal_probe',
})


class GpuRuleProgramError(Exception):
    """Fail-closed error for GPU rule program capture, load or coverage claims."""


def source_capture_available() -> bool:
    return sys.version_info[:2] == (3, 10)


def catalog_scope() -> tuple[tuple[str, ...], tuple[str, ...]]:
    from app.modules.card_game.content.duel_v2.catalog import CARDS, CHARACTERS
    from app.modules.card_game.content.duel_v2.registry import KITS, _ensure_kits
    _ensure_kits()
    characters = tuple(sorted(CHARACTERS))
    cards = tuple(sorted(CARDS))
    missing_kits = [cid for cid in characters if cid not in KITS]
    if missing_kits:
        raise GpuRuleProgramError('Catalog characters missing kits: ' + ','.join(missing_kits))
    missing_effects = []
    for card_id, card in CARDS.items():
        owner = card.get('character_id')
        kit = KITS.get(owner)
        effects = getattr(kit, 'effects', {}) if kit is not None else {}
        if card_id not in effects:
            missing_effects.append(card_id)
    if missing_effects:
        raise GpuRuleProgramError('Catalog cards missing kit effects: ' + ','.join(missing_effects[:12]))
    return characters, cards


def resolve_entry_names(requested: Iterable[str] | None = None) -> tuple[str, ...]:
    if requested is None:
        return FULL_ENTRY_NAMES
    names = tuple(requested)
    if not names:
        raise GpuRuleProgramError('At least one GPU rule entry is required')
    if len(names) != len(set(names)):
        raise GpuRuleProgramError('Duplicate GPU rule entries are not allowed')
    unknown = [name for name in names if name not in ENTRY_FUNCTIONS]
    if unknown:
        raise GpuRuleProgramError('Unknown GPU rule entries: ' + ','.join(unknown))
    order = {name: index for index, name in enumerate(FULL_ENTRY_NAMES)}
    return tuple(sorted(names, key=lambda name: order[name]))


def coverage_flags(entry_names: Sequence[str]) -> dict[str, Any]:
    all_declared = tuple(entry_names) == FULL_ENTRY_NAMES
    return {
        'entry_set': 'all_declared_entries' if all_declared else 'requested_subset',
        'full_coverage': False,
        'full_duel_rules': False,
        'requested_entries': list(entry_names),
        'declared_entries': list(FULL_ENTRY_NAMES),
    }


def capture_entries(entry_names: Sequence[str], runtime: str | None = None) -> dict[str, Any]:
    functions = entry_functions(runtime)
    missing = [name for name in entry_names if functions.get(name) is None]
    if missing:
        raise GpuRuleProgramError('Missing canonical GPU rule entries: ' + ','.join(missing))
    return {name: functions[name] for name in entry_names}


def detect_frozen_runtime(program: FrozenRuleProgram) -> str:
    names = []
    for item in list(program.functions) + list(program.classes):
        names.append(str(item.get('name') or ''))
        names.append(str(item.get('qualname') or ''))
        names.append(str(item.get('id') or ''))
    blob = ' '.join(names)
    seven = (
        'FrozenSevenRuntime' in blob
        or 'seven_sample_world' in blob
        or 'seven_acting_side' in blob
        or 'seven_public_legal_decision' in blob
    )
    recovery = 'FrozenRecoveryRuntime' in blob
    if seven and recovery:
        raise ValidationError('Frozen program mixes recovery and seven sampler bindings')
    if seven:
        return 'seven'
    return 'recovery'


def capture_gpu_rule_program(
    requested: Iterable[str] | None = None,
    *,
    runtime: str | None = None,
    scope_character_ids: Iterable[str] | None | object = Ellipsis,
    scope_card_ids: Iterable[str] | None | object = Ellipsis,
) -> FrozenRuleProgram:
    if not source_capture_available():
        raise ExportError(
            'Host bytecode export must be run under CPython 3.10, got ' + sys.version
            + '. Windows 3.13 loads an explicit frozen program with source/hash/entry-set checks.'
        )
    runtime = normalize_runtime(runtime)
    entry_names = resolve_entry_names(requested)
    entries = capture_entries(entry_names, runtime=runtime)
    if scope_character_ids is Ellipsis or scope_card_ids is Ellipsis:
        if tuple(entry_names) == FULL_ENTRY_NAMES or set(entry_names) & HEAVY_ENTRIES:
            characters, cards = catalog_scope()
        else:
            characters, cards = None, None
        if scope_character_ids is Ellipsis:
            scope_character_ids = characters
        if scope_card_ids is Ellipsis:
            scope_card_ids = cards
    return capture_program(
        entries,
        allowed_module_prefixes=ALLOWED_MODULE_PREFIXES,
        scope_character_ids=scope_character_ids,
        scope_card_ids=scope_card_ids,
        explicit_intrinsics=DEFAULT_EXPLICIT_INTRINSICS,
    )


def verify_source_hashes(program: FrozenRuleProgram, repo_root: Path | None = None) -> None:
    from app.modules.card_game.rl.offline_sources import _allowed_source
    root = (Path(repo_root) if repo_root is not None else find_repo_root()).resolve()
    mismatches = []
    for relative, digest in program.source_hashes.items():
        if not isinstance(relative, str) or not _allowed_source(relative):
            mismatches.append(str(relative) + ': outside source scope')
            continue
        path = root / relative
        if root not in path.resolve().parents or not path.is_file() or path.is_symlink():
            mismatches.append(relative + ': missing or symlink')
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            mismatches.append(relative)
    if mismatches:
        raise ValidationError('Frozen program source hash mismatch: ' + ', '.join(mismatches[:8]))


def load_program(
    path: str | Path,
    *,
    expected_entries: Iterable[str] | None = None,
    runtime: str | None = None,
    verify_sources: bool = True,
    repo_root: Path | None = None,
) -> FrozenRuleProgram:
    payload = Path(path).read_text()
    program = FrozenRuleProgram.from_json(payload)
    program.validate()
    requested_runtime = normalize_runtime(runtime)
    detected = detect_frozen_runtime(program)
    if detected != requested_runtime:
        raise ValidationError(
            'Frozen program runtime mismatch: expected '
            + requested_runtime + ' got ' + detected
        )
    if expected_entries is not None:
        expected = resolve_entry_names(expected_entries)
        if tuple(sorted(program.entries)) != tuple(sorted(expected)):
            raise ValidationError(
                'Frozen program entry set mismatch: expected '
                + ','.join(expected) + ' got ' + ','.join(program.entries)
            )
    if verify_sources:
        verify_source_hashes(program, repo_root=repo_root)
    return program


def lower_gpu_program(program: FrozenRuleProgram):
    from app.modules.card_game.rl.batched_duel.vm_image import lower_program
    return lower_program(program)


def _supported_vm():
    from app.modules.card_game.rl.batched_duel.cuda_vm import SUPPORTED_INTRINSICS, SUPPORTED_OPS
    return SUPPORTED_OPS, SUPPORTED_INTRINSICS


def program_missing_capabilities(program: FrozenRuleProgram) -> dict[str, list[str]]:
    opcodes, intrinsics = _supported_vm()
    image = lower_gpu_program(program)
    reports = [image.missing_capabilities(name,opcodes=opcodes,intrinsics=intrinsics) for name in image.entries]
    return {
        'opcodes': sorted({item for report in reports for item in report['opcodes']}),
        'intrinsics': sorted({item for report in reports for item in report['intrinsics']}),
    }


def entry_capability_report(image, entry_names: Sequence[str]) -> dict[str, Any]:
    opcodes, intrinsics = _supported_vm()
    reports = {}
    for name in entry_names:
        if name not in image.entries:
            raise GpuRuleProgramError('Lowered image is missing captured entry ' + name)
        missing = image.missing_capabilities(name, opcodes=opcodes, intrinsics=intrinsics)
        reports[name] = {
            'entry': name,
            'compile_ready': True,
            'declared_capabilities_ready': not any(missing.values()),
            'device_ready': False,
            'device_verified': False,
            'missing': missing,
            'full_duel_rules': False,
            'actual_backend': 'source_capture',
        }
    return reports


def program_status(
    program: FrozenRuleProgram | None,
    *,
    requested: Sequence[str],
    image=None,
    error: str | None = None,
    runtime: str | None = None,
) -> dict[str, Any]:
    runtime = normalize_runtime(runtime)
    flags = coverage_flags(requested)
    meta = runtime_metadata(runtime)
    unresolved = list(program.unresolved) if program is not None else []
    validated = bool(program is not None and not unresolved and error is None)
    compile_ready = validated
    entries = {}
    if image is not None and validated:
        entries = entry_capability_report(image, requested)
    elif program is not None:
        for name in requested:
            present = name in program.entries
            entries[name] = {
                'entry': name,
                'compile_ready': bool(validated and present),
                'declared_capabilities_ready': False,
                'device_ready': False,
                'device_verified': False,
                'missing': None if validated and present else {'entries': [name]} if not present else None,
                'full_duel_rules': False,
                'actual_backend': 'source_capture',
                'present': present,
            }
    status = {
        'compiler_python_required': '3.10',
        'runtime': runtime,
        'schema': meta['schema'],
        'observation_schema': meta['observation_schema'],
        'policy_schema': meta['policy_schema'],
        'keys': list(meta['keys']),
        'obs_dim': meta['obs_dim'],
        'cand_dim': meta['cand_dim'],
        'feature_names': list(meta['feature_names']),
        'candidate_names': list(meta['candidate_names']),
        'schema_sha': meta['schema_sha'],
        'candidate_schema_sha': meta['candidate_schema_sha'],
        'sampler': meta['sampler'],
        'sampler_runtime': sampler_runtime(runtime).__name__,
        'sampler_contract': meta['sampler_contract'],
        'host_python': sys.version.split()[0],
        'source_capture_available': source_capture_available(),
        'compile_ready': compile_ready,
        'image_ready': image is not None and validated,
        'device_verified': False,
        'unresolved': unresolved,
        'error': error,
        'identity': None if program is None else program.identity,
        'compiler_python': None if program is None else program.compiler_python,
        'entries': entries,
        'source_hashes': {} if program is None else dict(program.source_hashes),
    }
    status['program_missing'] = program_missing_capabilities(program) if program is not None else {'opcodes': [], 'intrinsics': []}
    status.update(flags)
    return status


def inspect_gpu_rule_program(
    requested: Iterable[str] | None = None,
    *,
    program: FrozenRuleProgram | None = None,
    lower: bool = True,
    runtime: str | None = None,
) -> dict[str, Any]:
    runtime = normalize_runtime(runtime)
    entry_names = resolve_entry_names(requested)
    if program is not None:
        detected = detect_frozen_runtime(program)
        if detected != runtime:
            error = 'Frozen program runtime mismatch: expected %s got %s' % (runtime, detected)
            status = program_status(program, requested=entry_names, image=None, error=error, runtime=runtime)
            status['lower_error'] = None
            return status
    error = None
    image = None
    lower_error = None
    if program is not None:
        try:
            if program.unresolved:
                error = 'Program contains unresolved references'
            else:
                program.validate()
        except (ExportError, ValidationError, GpuRuleProgramError, TypeError, ValueError) as exc:
            error = str(exc)
        if error is None and lower:
            try:
                image = lower_gpu_program(program)
            except (TypeError, ValueError, RuntimeError) as exc:
                lower_error = str(exc)
    status = program_status(program, requested=entry_names, image=image, error=error, runtime=runtime)
    status['lower_error'] = lower_error
    return status


def build_gpu_rule_program(
    requested: Iterable[str] | None = None,
    *,
    lower: bool = True,
    runtime: str | None = None,
) -> tuple[FrozenRuleProgram, dict[str, Any]]:
    runtime = normalize_runtime(runtime)
    entry_names = resolve_entry_names(requested)
    program = capture_gpu_rule_program(entry_names, runtime=runtime)
    status = inspect_gpu_rule_program(entry_names, program=program, lower=lower, runtime=runtime)
    return program, status


def save_program(
    program: FrozenRuleProgram,
    output_dir: str | Path,
    *,
    status: Mapping[str, Any],
) -> dict[str, str]:
    directory = Path(output_dir)
    if directory.exists():
        raise GpuRuleProgramError('Fresh output directory required; refusing to overwrite ' + str(directory))
    directory.mkdir(parents=True)
    program_path = directory / PROGRAM_FILENAME
    manifest_path = directory / MANIFEST_FILENAME
    program_path.write_text(program.to_json())
    manifest = {
        'identity': program.identity,
        'format': program.format,
        'compiler_python': program.compiler_python,
        'source_hashes': dict(program.source_hashes),
        'entries': dict(program.entries),
        'required_intrinsics': list(program.required_intrinsics),
        'status': dict(status),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return {'program': str(program_path), 'manifest': str(manifest_path)}


def check_configuration(requested: Iterable[str] | None = None, runtime: str | None = None) -> dict[str, Any]:
    runtime = normalize_runtime(runtime)
    entry_names = resolve_entry_names(requested)
    status = program_status(None, requested=entry_names, error=None, runtime=runtime)
    status['mode'] = 'check'
    status['torch_imported'] = 'torch' in sys.modules
    status['catalog_characters'] = None
    status['catalog_cards'] = None
    try:
        characters, cards = catalog_scope()
        status['catalog_characters'] = list(characters)
        status['catalog_cards'] = list(cards)
        status['catalog_character_count'] = len(characters)
        status['catalog_card_count'] = len(cards)
    except Exception as exc:
        status['catalog_error'] = str(exc)
    return status
