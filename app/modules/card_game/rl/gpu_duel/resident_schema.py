"""Dependency-free schema shared by training and CPU serving."""
SCHEMA = 'resident_public_v1'
SIDE_FIELDS = ('hp', 'shield', 'ap', 'front', 'last_front', 'turn_count', 'normal_atk',
               'ultimate_ok', 'used_instant', 'extra_genesis', 'extra_genesis_actor',
               'nanali_seed', 'nanali_seed_played', 'hand_n', 'deck_n', 'discard_n', 'weave')
CHAR_FIELDS = ('ch_present', 'ch_hp', 'ch_max_hp', 'ch_base_atk', 'ch_growth',
               'ch_atk_buff', 'ch_shield', 'ch_harmony', 'ch_energy', 'ch_down',
               'ch_awakened', 'ch_ult_turns', 'ch_shape', 'ch_next_bonus',
               'ch_next_shield', 'ch_next_followup', 'ch_pact', 'ch_harmonized',
               'ch_collapse_count', 'ch_collapse_until', 'ch_allied_hurt', 'ch_energy_next')
