"""Public persistent mechanics omitted from display-oriented card descriptions.

Explicit allowlists only: no hands, deck identities, RNG or private choices.
"""
SIDE_KEYS = ('extra_ap', 'delay_turns', 'delay_counted_turn', 'skip_normal_attack', 'extra_genesis_pending', 'extra_genesis_actor', 'genesis_turn', 'genesis_player_hits', 'surplus', 'harmony_damage',
             'surplus_ever', 'empty_draw_wins', 'genesis_damage_bonus', 'nanali_family', 'nanali_family_played', 'ultimates_used')
FLAG_KEYS = ('next_bonus', 'next_shield', 'next_followup', 'energy_next',
             'allied_hurt', 'pending_atk', 'share_overflow', 'hp_floor',
             'collapse', 'collapse_count', 'pact', 'atk_buff', 'timed_attack', 'damage_immunity_until')


def public_policy_state(state):
    from copy import deepcopy
    from .state import team_order
    result = {'escalation_enabled': bool(state.get('escalation_enabled')), 'sides': {}}
    for side, team in state['sides'].items():
        order = team_order(state, side)
        result['sides'][side] = {
            **{key: deepcopy(team.get(key)) for key in SIDE_KEYS},
            'light_resistance': (team.get('next_damage_resistance') or {}).get('光', 0),
            'characters': {
                cid: {'base_attack': h.get('base_attack', 0),
                      'base_max_hp': h.get('base_max_hp', h['max_hp']),
                      'order': order.index(cid),
                      'harmonized': bool((team.get('harmonized') or {}).get(cid)),
                      'jingu_loan': bool(h.get('jingu_loan')),
                      'ultimate_used_turn': h.get('ultimate_used_turn', -1),
                      'attacked_this_turn': h.get('entity_id') in team.get('attacked_this_turn', []),
                      'light_resistance': (h.get('next_damage_resistance') or {}).get('光', 0),
                      'flags': {key: deepcopy((h.get('flags') or {}).get(key)) for key in FLAG_KEYS}}
                for cid, h in team['characters'].items() if cid in order
            },
        }
    return result
