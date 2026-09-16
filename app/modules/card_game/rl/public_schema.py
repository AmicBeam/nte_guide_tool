"""Versioned numeric public observation contract; optional ML imports stay lazy."""
from .gpu_duel.resident_schema import SIDE_FIELDS as LEGACY_SIDE, CHAR_FIELDS as LEGACY_CHAR

SCHEMA = 'resident_public_v2'
SIDE_FIELDS = (*LEGACY_SIDE, 'fatigue', 'extra_ap', 'surplus', 'ultimate_remaining',
               'burn_left', 'burn_infinite', 'delay_end', 'harmony_damage', 'burn_by')
CHAR_FIELDS = (*LEGACY_CHAR, 'ch_base_max', 'ch_energy_max', 'ch_harmony_max',
               'ch_pending_atk', 'ch_share_overflow', 'ch_hp_floor', 'ch_order', 'ch_collapse_by')
GLOBAL_DIM = 7  # turn, first, escalation, phase, choice kind, resolving card, choice count
CAND_DIM = 11  # legacy ten features plus whether this hand instance is revealed


def state_dim():
    from .gpu_duel.catalog import N_SEATS, CARD_IDS
    return GLOBAL_DIM + 2*len(SIDE_FIELDS) + 2*N_SEATS*len(CHAR_FIELDS) + 5*len(CARD_IDS)


def rule_identity(schema=SCHEMA):
    """Includes official rules, encoder, and compiled rules; no generated artifacts."""
    from hashlib import sha256
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    digest = sha256(schema.encode())
    files = []
    for directory in (root/'engine/duel_v2', root/'content/duel_v2', root/'rl/rule_ir'):
        files.extend(directory.rglob('*.py'))
    files += [root/'content/duel_v2/catalog.json', Path(__file__), root/'rl/public_observation.py',root/'rl/opening_observation.py']
    for path in sorted(set(files)):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_text(encoding='utf-8').encode('utf-8'))
    return digest.hexdigest()
