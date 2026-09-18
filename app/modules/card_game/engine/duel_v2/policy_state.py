"""Public persistent mechanics omitted from display-oriented card descriptions.

Explicit allowlists only: no hands, deck identities, RNG or private choices.
"""
SIDE_KEYS = ('extra_ap', 'extra_genesis_pending', 'extra_genesis_actor', 'surplus', 'harmony_damage',
             'genesis_damage_bonus', 'nanali_family', 'nanali_family_played', 'ultimates_used')
FLAG_KEYS = ('next_bonus', 'next_shield', 'next_followup', 'energy_next',
             'allied_hurt', 'pending_atk', 'share_overflow', 'hp_floor',
             'collapse', 'collapse_count', 'pact', 'atk_buff')


def public_policy_state(state):
    from copy import deepcopy
    from .state import team_order
    result = {'escalation_enabled': bool(state.get('escalation_enabled')), 'sides': {}}
    for side, team in state['sides'].items():
        order = team_order(state, side)
        result['sides'][side] = {
            **{key: deepcopy(team.get(key)) for key in SIDE_KEYS},
            'characters': {
                cid: {'base_attack': h.get('base_attack', 0),
                      'base_max_hp': h.get('base_max_hp', h['max_hp']),
                      'order': order.index(cid),
                      'harmonized': bool((team.get('harmonized') or {}).get(cid)),
                      'flags': {key: deepcopy((h.get('flags') or {}).get(key)) for key in FLAG_KEYS}}
                for cid, h in team['characters'].items() if cid in order
            },
        }
    return result
