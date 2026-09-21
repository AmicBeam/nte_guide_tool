"""Public persistent mechanics omitted from display-oriented card descriptions.

Explicit allowlists only: no hands, deck identities, RNG or private choices.
"""
SIDE_KEYS = ('extra_ap', 'delay_turns', 'delay_counted_turn', 'skip_normal_attack', 'extra_genesis_pending', 'extra_genesis_actor', 'genesis_turn', 'genesis_player_hits', 'surplus', 'harmony_damage',
             'surplus_ever', 'empty_draw_wins', 'genesis_damage_bonus', 'nanali_family', 'nanali_family_played', 'ultimates_used')
FLAG_KEYS = ('next_bonus', 'next_shield', 'next_followup', 'energy_next',
             'allied_hurt', 'pending_atk', 'share_overflow', 'hp_floor',
             'collapse', 'collapse_count', 'pact', 'atk_buff', 'atk_buff_expiring', 'timed_attack', 'damage_immunity_until')


def public_policy_state(state):
    from copy import deepcopy
    from .state import team_order
    from .equipment import permanent_attack, unarmed_max_hp, stat_layers, public_equipment
    from .modifiers import compatibility_flags, public_effects, resistance_values
    result = {'escalation_enabled': bool(state.get('escalation_enabled')), 'sides': {}}
    for side, team in state['sides'].items():
        order = team_order(state, side)
        result['sides'][side] = {
            **{key: deepcopy(team.get(key)) for key in SIDE_KEYS},
            'light_resistance': resistance_values(team).get('光', 0),
            'characters': {
                cid: {'base_attack': permanent_attack(h),
                      'base_max_hp': unarmed_max_hp(h),
                      'stat_layers': stat_layers(h), 'equipment': public_equipment(h),
                      'order': order.index(cid),
                      'harmonized': bool((team.get('harmonized') or {}).get(cid)),
                      'jingu_loan': bool(h.get('jingu_loan')),
                      'ultimate_used_turn': h.get('ultimate_used_turn', -1),
                      'attacked_this_turn': h.get('entity_id') in team.get('attacked_this_turn', []),
                      'light_resistance': resistance_values(h).get('光', 0),
                      'effects': public_effects(h),
                      'flags': {key: deepcopy(derived_flags.get(key)) for key in FLAG_KEYS}}
                for cid, h in team['characters'].items() if cid in order
                for derived_flags in [compatibility_flags(h, state, side)]
            },
        }
    return result
