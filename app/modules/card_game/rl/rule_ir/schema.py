"""Versioned integer layout for the combat slice. Not a second rule source."""

FIELDS = (
    'a_hp', 'a_sh', 'a_atk',
    'b_hp', 'b_sh', 'b_atk',
    'pa_hp', 'pa_sh', 'pb_hp', 'pb_sh',
    'a_front', 'b_front', 'a_down', 'b_down',
    'done', 'winner', 'no_counter', 'penetrate', 'bonus',
)
FIELD_INDEX = {name: i for i, name in enumerate(FIELDS)}
N_FIELDS = len(FIELDS)
SLICE_VERSION = 'combat_zero_v1'

# Side-relative aliases used by the combat program. Values are FIELD names.
SIDE_ALIASES = (
    # attacker char hp/sh/atk, defender char hp/sh/atk,
    # defender player hp/sh, attacker front, defender front, attacker down, defender down
    ('ah', 'a_hp', 'b_hp'),
    ('ash', 'a_sh', 'b_sh'),
    ('atk', 'a_atk', 'b_atk'),
    ('dh', 'b_hp', 'a_hp'),
    ('dsh', 'b_sh', 'a_sh'),
    ('datk', 'b_atk', 'a_atk'),
    ('dphp', 'pb_hp', 'pa_hp'),
    ('dpsh', 'pb_sh', 'pa_sh'),
    ('af', 'a_front', 'b_front'),
    ('df', 'b_front', 'a_front'),
    ('ad', 'a_down', 'b_down'),
    ('dd', 'b_down', 'a_down'),
)
