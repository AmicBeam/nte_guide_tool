"""C++ emission for public V2 card effects and catalog-ordered kit dispatch."""

from .domain_soa_kit_source import PUBLIC_KIT_ORDER, emit_source as emit_kit_source

PUBLIC_CARD_EFFECTS = (
    'N01', 'N02', 'N03', 'N04', 'N05', 'N06', 'N07', 'N08', 'NF01',
    'Z01', 'Z02', 'Z03', 'Z04', 'Z05', 'Z06', 'Z07', 'Z08',
    'J01', 'J02', 'J03', 'J04', 'J05', 'J06', 'J07', 'J08',
    'Y01', 'Y02', 'Y03', 'Y04', 'Y05', 'Y06', 'Y07', 'Y08',
    'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08',
    'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
    'Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q06', 'Q07', 'Q08',
    'U01', 'U02', 'U03', 'U04', 'U05', 'U06', 'U07', 'U08',
    'I01', 'I02', 'I03', 'I04', 'I05', 'I06', 'I07', 'I08',
    'R01', 'R02', 'R03', 'R04', 'R05', 'R06', 'R07', 'R08', 'RF01',
    'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'AF01',
    'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08',
    'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08',
    'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08',
    'K01', 'K02', 'K03', 'K04', 'K05', 'K06', 'K07', 'K08',
    'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'FF01',
    'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'PF01',
)


def emit_source():
    return emit_kit_source() + '\n' + _SOURCE


_SOURCE = r'''
DFN void kit_card_shape(DX* x) { d_set_shape(x); }

DFN void kit_card_N01(DX* x) {
  int missing = d_max_hp(x, x->side, x->actor) - d_ci(x, x->side, x->actor, "hp");
  if (missing > 0) d_heal(x, kit_self(x), missing, kit_self(x));
}
DFN void kit_card_N02(DX* x) { d_sortie(x, 0); }
DFN void kit_card_N03(DX* x) { d_sortie(x, 0); }
DFN void kit_card_N04(DX* x) {
  d_set_pi(x, x->side, "genesis_damage_bonus", d_pi(x, x->side, "genesis_damage_bonus") + 1);
  d_event(x);
  d_sortie(x, 0);
}
DFN void kit_card_N05(DX* x) {
  DR t = d_op_target(x, "target_id");
  int amount = 4 + (kit_shape(x, x->actor, "N08") ? 2 : 0);
  d_damage(x,t,amount,kit_self(x),10,0,1,0,0,0);
}
DFN void kit_card_N06(DX* x) { d_revive(x, x->side, x->actor); }
DFN void kit_card_NF01(DX* x) {
  d_context_grow(x, x->side, x->actor, 1, 1);
  kit_grant_harmony(x, x->actor, 1);
  d_set_pi(x, x->side, "nanali_family_played", d_pi(x, x->side, "nanali_family_played") + 1);
}

DFN void kit_card_Z01(DX* x) {
  int front = d_front(x, x->side);
  if (front < 0) return;
  int missing = d_max_hp(x, x->side, front) - d_ci(x, x->side, front, "hp");
  if (missing > 0) d_heal(x,kit_dr(x->side,front),missing,DR{-1,-2});
}
DFN void kit_card_Z02(DX* x) {
  if (d_finished(x)) return;
  d_draw_matching(x, x->side, 2, "form", 1);
}
DFN void kit_card_Z03(DX* x) { d_sortie(x, 0); }
DFN void kit_card_Z04(DX* x) { d_sortie(x, 0); }
DFN void kit_card_Z05(DX* x) { d_sortie(x, 0); }

DFN void kit_card_J01(DX* x) { d_inspect_top(x, x->side, 3); }
DFN void kit_card_J02(DX* x) {
  DR t = d_front_target(x);
  int had_pact = 0, hit_player = t.slot < 0;
  if (t.slot >= 0) had_pact = d_flag(x, t.side, t.slot, "pact");
  d_damage(x, t, 1, kit_self(x), 0, 0, 1, 0, 0, 0);
  if(t.slot>=0)d_attach_pact(x,t);
  if ((had_pact || hit_player) && !d_finished(x)) d_draw(x, x->side, 1, -1);
}
DFN void kit_card_J03(DX* x) {
  DR t = d_op_target(x, "target_id");
  d_set_ci(x, t.side, t.slot, "down_turns", 3);
}
DFN void kit_card_J04(DX* x) {
  DR t = d_op_target(x, "target_id");
  d_damage(x, t, 2, kit_self(x), 0, 0, 1, 0, 0, 0);
  if(t.slot>=0)d_attach_pact(x,t);
}
DFN void kit_card_J05(DX* x) {
  int triggered = d_pi(x, x->side, "genesis_turn") == d_turn(x);
  d_damage(x, kit_player(d_enemy(x->side)), triggered ? 3 : 2, kit_self(x), 0, 0, 1, 0, 0, 0);
}
DFN void kit_card_J06(DX* x) {
  int count = d_pi(x, x->side, "genesis_player_hits"), i;
  for (i = 0; i < count; ++i) {
    if (d_finished(x)) break;
    d_damage(x, d_front_target(x), 1, kit_self(x), 0, 0, 1, 0, 0, 0);
  }
}

DFN void kit_card_Y01(DX* x) {
  int downed = kit_downed_count(x, x->side);
  if (downed) d_damage(x, d_front_target(x), 3 * downed, kit_self(x), 0, 0, 1, 0, 0, 0);
}
DFN void kit_card_Y02(DX* x) {
  DR t = d_op_target(x, "target_id");
  d_draw_owned(x, t.slot, 0);
}
DFN void kit_card_Y03(DX* x) {
  DR t = d_op_target(x, "target_id");
  d_heal(x,t,5,DR{-1,-2});
  if (t.side == x->side && t.slot >= 0) kit_atk_buff(x, t.slot, 1, "Y03");
}
DFN void kit_card_Y04(DX* x) {
  d_set_flag(x, x->side, x->actor, "next_bonus", d_flag(x, x->side, x->actor, "next_bonus") + 1, 0);
  d_set_flag(x, x->side, x->actor, "next_shield", d_flag(x, x->side, x->actor, "next_shield") + 2, 0);
}
DFN void kit_card_Y05(DX* x) {
  int n = d_order_count(x, x->side), i, slot, marked[8], mn = 0;
  for (i = 0; i < n && mn < 8; ++i) {
    slot = d_order_slot(x, x->side, i);
    if (slot == x->actor || !d_alive(x, x->side, slot)) continue;
    d_set_ci(x, x->side, slot, "hp", 0);
    marked[mn++] = slot;
  }
  d_knockdowns(x);
  for (i = 0; i < mn; ++i) {
    slot = marked[i];
    if (d_ci(x, x->side, slot, "down_turns") > 0) d_set_ci(x, x->side, slot, "down_turns", 2);
  }
  d_draw(x, x->side, 2, -1);
  d_context_grow(x, x->side, x->actor, 1, 1);
}
DFN void kit_card_Y06(DX* x) {
  DR t = d_op_target(x, "target_id");
  d_revive(x, t.side, t.slot);
  if (t.side == x->side && t.slot >= 0) kit_atk_buff(x, t.slot, 1, "Y06");
}
DFN void kit_card_Y07(DX* x) { kit_card_shape(x); kit_iloy_y07_heal(x); }

DFN void kit_card_M01(DX* x) { d_context_grow(x, x->side, x->actor, 1, 0); d_sortie(x, 0); }
DFN void kit_card_M02(DX* x) {
  int n = d_order_count(x, x->side), i, slot;
  for (i = 0; i < n; ++i) {
    slot = d_order_slot(x, x->side, i);
    if (d_alive(x, x->side, slot)) kit_atk_buff(x, slot, 1, "M02");
  }
}
DFN void kit_card_M03(DX* x) { DAttack a = kit_attack0(); a.ignore_shield = 1; d_sortie(x, &a); }
DFN void kit_card_M04(DX* x) { d_set_flag(x, x->side, x->actor, "share_overflow", 1, 1); d_sortie(x, 0); }
DFN void kit_card_M05(DX* x) {
  d_set_flag(x, x->side, x->actor, "pending_atk", d_flag(x, x->side, x->actor, "pending_atk") + 2, 0);
}
DFN void kit_card_M06(DX* x) {
  int extra = d_permanent_attack(x, x->side, x->actor) - d_printed_attack(x, x->side, x->actor);
  if (extra < 0) extra = 0;
  d_damage(x, d_front_target(x), 2 + extra, kit_self(x), 0, 0, 1, 0, 0, 0);
}

DFN void kit_card_B01(DX* x) {
  d_damage(x, kit_self(x), 1, kit_self(x), 0, 0, 1, 0, 0, 0);
  d_draw(x, x->side, 1, -1);
}
DFN void kit_card_B02(DX* x) { d_sortie(x, 0); }
DFN void kit_card_B03(DX* x) {
  int foe = d_enemy(x->side);
  if (d_zone_has(x, foe, "burn") && d_zone_int(x, foe, "burn", "by", -1) == x->side) {
    d_zone_set(x, foe, "burn", "infinite", 1);
    d_event(x);
  }
}
DFN void kit_card_B04(DX* x) {
  int missing = d_max_hp(x, x->side, x->actor) - d_ci(x, x->side, x->actor, "hp");
  if (missing <= 0) return;
  int side, i, slot, n, self_ent = d_entity(x, kit_self(x));
  for (side = 0; side < 2; ++side) {
    n = d_order_count(x, side);
    for (i = 0; i < n; ++i) {
      slot = d_order_slot(x, side, i);
      if (d_entity(x, kit_dr(side, slot)) == self_ent) continue;
      if (d_alive(x, side, slot)) d_damage(x, kit_dr(side, slot), missing, kit_self(x), 0, 0, 1, 0, 0, 0);
    }
  }
}
DFN void kit_card_B05(DX* x) { d_set_flag(x, x->side, x->actor, "hp_floor", 1, 1); }
DFN void kit_card_B06(DX* x) {
  DAttack a = kit_attack0();
  a.bonus = d_flag(x, x->side, x->actor, "allied_hurt") ? 1 : 0;
  d_sortie(x, &a);
}
DFN void kit_card_B08(DX* x) {
  kit_card_shape(x);
  if (d_alive(x, x->side, x->actor)) d_damage(x, kit_self(x), 4, kit_self(x), 0, 0, 1, 0, 0, 0);
}

DFN void kit_card_Q01(DX* x) {
  DAttack a = kit_attack0();
  a.bonus = d_op_option_is(x, "boost") ? 3 : 0;
  d_sortie(x, &a);
}
DFN void kit_card_Q02(DX* x) {
  d_sortie(x, 0);
  DR t = d_op_target(x, "last_attack_target");
  if (t.slot == -2 || d_finished(x)) return;
  int hp = t.slot < 0 ? d_pi(x, t.side, "hp") : d_ci(x, t.side, t.slot, "hp");
  if (hp > 0) {
    DEffect e=kit_eff("xiaozhi.light_resistance","damage.resistance",-2,"add");e.stacking="replace";e.stack_key="\x72\x65\x73\x69\x73\x74\x61\x6e\x63\x65\x3a\xe5\x85\x89";e.label="\xe5\x85\x89\xe6\x8a\x97\xe6\x80\xa7\x20\x2d\x32";e.consume_on="damage";e.condition="damage_attribute";e.condition_key="attribute";e.condition_string="\xe5\x85\x89";kit_expiry(&e,x,x->s->active_side[0],0,"turn_end");
    d_effect(x, t, &e);
    d_event(x);
  }
  if (d_op_option_is(x, "release_a") && d_alive(x, x->side, x->actor) && !d_finished(x)) kit_xiaozhi_release(x, "Q01");
}
DFN void kit_card_Q03(DX* x) {
  d_sortie(x, 0);
  if (d_finished(x) || !d_alive(x, x->side, x->actor)) return;
  if (d_op_option_is(x, "release_b") || d_op_option_is(x, "release_ba")) kit_xiaozhi_release(x, "Q02");
  if (d_op_option_is(x, "release_ba") && !d_finished(x) && d_alive(x, x->side, x->actor)) kit_xiaozhi_release(x, "Q01");
}
DFN void kit_card_Q04(DX* x) {
  if (d_response_intercept(x) && kit_shape(x, x->actor, "Q08")) {
    int j = d_ci(x, x->side, x->actor, "jingu");
    if (j < 0) j = 0;
    d_set_flag(x, x->side, x->actor, "counter_bonus", d_flag(x, x->side, x->actor, "counter_bonus") + j / 2, 0);
  }
  d_sortie(x, 0);
}
DFN void kit_card_Q05(DX* x) {
  kit_change_jingu(x, 1, 0);
  if (x->card >= 0 && !d_card_bool(x, x->card, "copy") && !d_finished(x)) {
    int copy = d_make_card(x, x->side, "Q05", x->actor);
    if (!d_bad(x)) { d_insert_deck(x, x->side, copy, d_deck_count(x, x->side)); d_shuffle_zone(x, x->side, 1); d_event(x); }
  }
  d_draw(x, x->side, 1, -1);
}
DFN void kit_card_Q06(DX* x) { d_context_energy(x, x->side, x->actor, d_energy_cap(x, x->side, x->actor)); }

DFN void kit_card_U01(DX* x) { d_sortie(x, 0); }
DFN void kit_card_U02(DX* x) { d_sortie(x, 0); }
DFN void kit_card_U03(DX*x){DCombatSpec spec=combat_spec_from_attack(x,nullptr);spec.target=DR{1-x->side,-1};spec.has_target=1;combat_sortie_ex(x,&spec);}
DFN void kit_card_U04(DX*x){DCombatSpec spec=combat_spec_from_attack(x,nullptr);spec.target=d_op_target(x);spec.has_target=1;spec.base.no_counter=1;spec.bool_presence[1]=2;combat_sortie_ex(x,&spec);}
DFN void kit_card_I01(DX* x) { kit_yi_fangs(x, 4); d_sortie(x, 0); }
DFN void kit_card_I02(DX* x) { kit_yi_fangs(x, 4); d_sortie(x, 0); }
DFN void kit_card_I03(DX* x) { kit_yi_fangs(x, 4); d_draw(x, x->side, 1, -1); }
DFN void kit_card_I04(DX* x) { kit_yi_fangs(x, 2); d_draw(x, x->side, 1, -1); }
DFN void kit_card_I05(DX* x) { d_set_pb(x, d_enemy(x->side), "skip_normal_attack", 1); d_event(x); }
DFN void kit_card_I06(DX* x) {
  DR t = d_op_target(x, "target_id");
  int count = d_pi(x, x->side, "delay_turns");
  if (!count || d_finished(x) || !d_alive(x, t.side, t.slot)) return;
  DEffect e = kit_eff("yi.attack_debuff", "attack.panel", (double)(-count), "add");
  e.expiry_side = t.side; e.expiry_turn = 2; e.expiry_phase = "turn_start";
  d_effect(x, t, &e);
  d_event(x);
}

DFN void kit_card_R01(DX* x) {
  int n = d_order_count(x, x->side), i, slot;
  for (i = 0; i < n; ++i) {
    slot = d_order_slot(x, x->side, i);
    if (slot != x->actor && d_attribute_is_light_family(x, x->side, slot)) d_draw_owned(x, slot, 1);
  }
}
DFN void kit_card_R_battle(DX* x) {
  if (x->card >= 0 && d_card_is(x, x->card, "RF01")) d_draw(x, x->side, 1, -1);
  else if (x->card >= 0 && d_card_is(x, x->card, "R02")) d_draw(x, x->side, 1, 0);
  else if (x->card >= 0 && d_card_is(x, x->card, "R03")) d_draw(x, x->side, 1, 2);
  else if (x->card >= 0 && d_card_is(x, x->card, "R04")) d_draw(x, x->side, 1, 1);
  if (!d_finished(x)) d_sortie(x, 0);
}
DFN void kit_card_RF01(DX* x) { kit_card_R_battle(x); }
DFN void kit_card_R05(DX* x) {
  int n = d_redraw_own_hand(x, x->side, x->actor);
  if (n > 0) d_draw(x, x->side, n, -1);
}
DFN void kit_card_R_shape(DX* x) { d_set_ci(x, x->side, x->actor, "intimidation", 0); d_set_shape(x); }

DFN void kit_card_A01(DX* x) {
  DR t = d_op_target(x, "target_id");
  int tmpl = d_template(x, t.side, t.slot);
  d_set_mimic_template(x, x->side, x->actor, tmpl);
  d_event(x);
  int awoke = kit_awake(x, x->actor);
  d_set_cb(x, x->side, x->actor, "awakened", 0);
  kit_fire_passive_named(x, tmpl, "on_turn_begin", 0);
  d_set_cb(x, x->side, x->actor, "awakened", awoke);
}
DFN void kit_card_A02(DX* x) {
  int stacks = kit_anhunqu_stacks(x);
  DR t = d_op_target(x, "target_id");
  if (stacks) d_damage(x, t, stacks, kit_self(x), 0, 0, 1, 0, 0, 0);
  if (stacks >= 5) d_damage(x, kit_player(d_enemy(x->side)), 3, kit_self(x), 0, 0, 1, 0, 0, 0);
}
DFN void kit_card_A03(DX* x) {
  int foe = d_enemy(x->side);
  if (kit_anhunqu_stacks(x) > 0) { d_trigger_dot(x, foe, "nightmare"); d_zone_remove(x, foe, "nightmare"); d_event(x); }
  d_draw(x, x->side, 1, -1);
}
DFN void kit_card_A04(DX* x) {
  d_apply_dot(x, d_enemy(x->side), "nightmare", x->side, x->actor, 5, -1);
  if (kit_anhunqu_stacks(x) >= 10) {
    int card = d_make_card(x, x->side, "AF01", x->actor);
    if (!d_bad(x)) { d_add_hand(x, x->side, card, 0); d_event(x); }
  }
}
DFN void kit_card_A05(DX*x){DCombatSpec spec=combat_spec_from_attack(x,nullptr);spec.target=d_op_target(x);spec.has_target=1;spec.stay_bench=d_front(x,x->side)!=x->actor;spec.base.no_counter=spec.stay_bench;spec.bool_presence[1]=2;spec.bool_presence[2]=2;combat_sortie_ex(x,&spec);}
DFN void kit_card_A06(DX* x) {
  int front = d_front(x, x->side);
  int attribute = front >= 0 ? combat_attr(x, x->side, front) : ATTR_NONE;
  if (front >= 0 && front != x->actor && !d_collapsed(x, x->side, front) && (attribute == ATTR_SOUL || attribute == ATTR_HEX)) {
    d_set_op_int(x, "skip_nightmare", 1, 1);
    d_enter(x, 1, 0);
    combat_apply_enter_harmony(x,front,attribute == ATTR_HEX ? HK_BURN : HK_STAR,1,0);
    d_sortie(x, 0);
    return;
  }
  d_sortie(x, 0);
}
DFN void kit_card_A07(DX* x) {
  kit_card_shape(x);
  if (d_alive(x, x->side, x->actor)) d_apply_dot(x, d_enemy(x->side), "nightmare", x->side, x->actor, 2, -1);
}
DFN void kit_card_AF01(DX* x) {
  int card = d_make_card(x, x->side, "A03", x->actor);
  if (!d_bad(x)) {
    d_set_card_int(x, card, "ephemeral_turn", d_turn(x));
    d_add_hand(x, x->side, card, 0);
    d_event(x);
  }
  d_sortie(x, 0);
}

DFN void kit_card_C01(DX* x) {
  d_sortie(x, 0);
  if (d_total_stacks(x, x->side) >= 5 && d_alive(x, x->side, x->actor)) {
    int missing = d_max_hp(x, x->side, x->actor) - d_ci(x, x->side, x->actor, "hp");
    if (missing) d_heal(x, kit_self(x), missing, kit_self(x));
  }
}
DFN void kit_card_C02(DX* x) {
  if (d_total_stacks(x, x->side) >= 10) d_draw(x, x->side, 1, -1);
  int foe = d_enemy(x->side);
  int before = kit_living_count(x, foe);
  d_sortie(x, 0);
  if (kit_living_count(x, foe) < before) {
    int card = d_make_card(x, x->side, "C02", x->actor);
    if (!d_bad(x)) { d_add_hand(x, x->side, card, 0); d_event(x); }
  }
}
DFN void kit_card_C03(DX* x) {
  d_sortie(x, 0);
  if (d_pi(x, x->side, "canhong_collapse_used")) return;
  DR t = d_op_target(x, "last_attack_target");
  if (t.slot >= 0 && d_alive(x, t.side, t.slot)) {
    d_force_collapse(x, t.side, t.slot, x->side);
    d_set_pi(x, x->side, "canhong_collapse_used", 1);
    d_event(x);
  }
}
DFN void kit_card_C04(DX* x) {
  int steps = d_total_stacks(x, x->side) / 5;
  if (steps > 4) steps = 4;
  if (steps) d_shield(x, kit_self(x), steps);
  DAttack a = kit_attack0(); a.bonus = steps; d_sortie(x, &a);
}
DFN void kit_card_C05(DX* x) {
  DEffect e = kit_eff("canhong.combat_immune", "rule.combat_immune", 1, "allow");
  e.stacking = "replace"; e.clear_on = "battle";
  d_effect(x, kit_self(x), &e);
  d_sortie(x, 0);
}
DFN void kit_card_C07(DX* x) { d_revive(x, x->side, x->actor); kit_card_shape(x); }

DFN void kit_card_S01(DX*x){d_summon(x,"vinyl");}
DFN void kit_card_S02(DX*x){d_summon(x,"collar");}
DFN void kit_card_S03(DX*x){d_summon(x,"liquid");}
DFN void kit_card_S04(DX* x) { d_damage(x, d_op_target(x, "target_id"), 3, kit_self(x), 0, 0, 1, 0, 0, 0); }
DFN void kit_card_S05(DX* x) {
  int front = d_front(x, x->side);
  if (front >= 0) d_damage(x, kit_dr(x->side, front), 2, kit_self(x), 0, 0, 1, 0, 0, 0);
  int foe = d_enemy(x->side), n = d_order_count(x, foe), i, slot;
  for (i = 0; i < n; ++i) {
    slot = d_order_slot(x, foe, i);
    if (d_cb(x, foe, slot, "summoned") || !d_alive(x, foe, slot)) continue;
    d_damage(x, kit_dr(foe, slot), 2, kit_self(x), 0, 0, 1, 0, 0, 0);
  }
}
DFN void kit_card_S06(DX* x) {
  int g = kit_zaowu_gulang(x);
  if (g >= 0) { d_set_seal_normal_attack(x, d_enemy(x->side), g); d_event(x); }
}
DFN void kit_card_S07(DX* x) { kit_card_shape(x); kit_zaowu_maybe_recall(x); }

DFN void kit_card_D01(DX*x){DR t=d_op_target(x,"target_id");d_damage(x,t,3,kit_self(x));d_shield(x,kit_self(x),3);if(!d_burn_present(x))return;int foe=1-x->side,others[C],n=0;for(int j=0;j<d_order_count(x,foe);j++){int a=d_order_slot(x,foe,j);if(d_alive(x,foe,a)&&!d_cb(x,foe,a,"summoned")&&a!=t.slot)others[n++]=a;}if(n){int a=others[d_random_before(x,n)];d_damage(x,DR{foe,a},3,kit_self(x));d_shield(x,kit_self(x),3);}}
DFN void kit_card_D02(DX* x) {
  d_shield(x, kit_player(x->side), 3);
  int n = d_order_count(x, x->side), i, slot;
  for (i = 0; i < n; ++i) {
    slot = d_order_slot(x, x->side, i);
    if (!d_cb(x, x->side, slot, "summoned") && d_alive(x, x->side, slot)) d_shield(x, kit_dr(x->side, slot), 3);
  }
  kit_adler_guard(x, 3);
}
DFN void kit_card_D03(DX* x) { d_shield(x, d_op_target(x, "target_id"), 4); }
DFN void kit_card_D04(DX* x) {
  d_sortie(x, 0);
  if (!d_burn_present(x)) return;
  int dealt = d_op_int(x, "last_attack_amount", 0);
  if (dealt) d_heal(x, kit_player(x->side), dealt, kit_self(x));
}
DFN void kit_card_D05(DX* x) { d_sortie(x, 0); kit_adler_guard(x, d_burn_present(x) ? 3 : 2); }
DFN void kit_card_D06(DX* x) {
  kit_card_shape(x);
  if (d_alive(x, x->side, x->actor)) d_shield(x, kit_self(x), 2);
}

DFN void kit_card_K01(DX* x) {
  DR t = d_op_target(x, "target_id");
  d_lingke_offer_owned(x, t.slot, 3);
}
DFN void kit_card_K02(DX* x) {
  d_draw(x, x->side, 1, -1);
  if (d_finished(x)) return;
  DEffect e = kit_eff("lingke.next_instant", "card.instant", 1, "allow");
  e.scope="allies";e.stacking="replace";e.consume_on="card_use";e.condition="card_owner_other";e.clear_count=0;e.source_key="K02";e.label="\xe4\xb8\x8b\xe4\xb8\x80\xe5\xbc\xa0\xe9\x9d\x9e\xe7\x81\xb5\xe5\x8f\xaf\xe7\x89\x8c\xe7\x9e\xac\xe5\x8f\x91";
  d_effect(x, kit_player(x->side), &e);
}
DFN void kit_card_K03(DX* x) {
  if (x->card >= 0 && d_card_bool(x, x->card, "_k03_recovery")) { d_add_ap(x, x->side, 1); d_event(x); return; }
  kit_card_shape(x);
}
DFN void kit_card_K04(DX* x) {
  DR t = d_op_target(x, "target_id");
  DEffect e = kit_eff("lingke.taunt", "marker.lingke_taunt", 1, "add");
  e.stacking="replace";e.show_marker=1;e.label="\xe5\xb0\x8f\xe9\xac\xbc\xe6\x89\x93\xe8\xbf\x99\xe9\x87\x8c";e.source_key="K04";
  d_effect(x, t, &e);
  d_event(x);
}
DFN void kit_card_K05(DX* x) {
  DR t = d_op_target(x, "target_id");
  kit_lingke_sync(x, d_attribute_of(x, t.side, t.slot), t.slot, 0);
}
DFN void kit_card_K06(DX* x) {
  int n = d_order_count(x, x->side), i, slot;
  for (i = 0; i < n; ++i) {
    slot = d_order_slot(x, x->side, i);
    if (slot == x->actor || !d_alive(x, x->side, slot) || d_cb(x, x->side, slot, "summoned")) continue;
    if (d_alive(x, x->side, slot)) kit_lingke_sync(x, d_attribute_of(x, x->side, slot), slot, 1);
  }
  kit_lingke_sync(x, d_attribute_of(x, x->side, x->actor), -1, 1);
}

DFN int kit_insight_damage(DX* x) { return 2 + d_pi(x, x->side, "insight_bonus"); }
DFN int kit_support_count(DX* x) {
  if (d_pi(x, x->side, "support_turn") != d_turn(x)) return 0;
  int n = d_pi(x, x->side, "support_count");
  return n > 2 ? 2 : n;
}
DFN void kit_card_F01(DX* x) {
  d_enter(x, 1, 1);
  int count = kit_support_count(x);
  d_set_op_int(x, "card_bonus", d_op_int(x, "card_bonus", 0) + count, 0);
  d_set_op_int(x, "collapse_bonus", count, 0);
  d_sortie(x, 0);
}
DFN void kit_card_F03(DX* x) { DR t = d_op_target(x, "target_id"); d_add_collapse(x, t, 2, x->side); }
DFN void kit_card_F04(DX* x) {
  int foe = d_enemy(x->side), n = d_order_count(x, foe), i, slot;
  for (i = 0; i < n; ++i) {
    slot = d_order_slot(x, foe, i);
    if (!d_cb(x, foe, slot, "summoned")) d_add_collapse(x, kit_dr(foe, slot), 2, x->side);
  }
  d_set_pi(x, x->side, "insight_bonus", d_pi(x, x->side, "insight_bonus") + 1);
  d_event(x);
}
DFN void kit_card_F05(DX*x){int amount=kit_insight_damage(x),repeats=1+d_pi(x,x->side,"insight_played"),foe=1-x->side;for(int j=0;j<repeats&&!d_finished(x);j++){int targets[C+1],n=1;targets[0]=-1;for(int i=0;i<d_order_count(x,foe);i++){int a=d_order_slot(x,foe,i);if(d_alive(x,foe,a))targets[n++]=a;}int a=targets[d_randbelow(x,n)];d_damage(x,DR{foe,a},amount,kit_self(x));}}
DFN void kit_card_F06(DX* x) { DR t = d_op_target(x, "target_id"); d_add_collapse(x, t, 4, x->side); }
DFN void kit_card_F07(DX*x){bmt_state_t rng;d_random_begin(x,&rng,0);int selected[2]={-1,-1};for(int side=0;side<2;side++){int options[C],n=0;for(int j=0;j<d_order_count(x,side);j++){int a=d_order_slot(x,side,j);if(d_alive(x,side,a)&&!d_cb(x,side,a,"summoned"))options[n++]=a;}if(n)selected[side]=options[d_random_index(x,&rng,n)];}d_random_end(x,&rng);for(int side=0;side<2;side++)if(selected[side]>=0)d_force_collapse(x,side,selected[side],x->side);}
DFN void kit_card_FF01(DX* x) {
  DR t = d_op_target(x, "target_id");
  if (d_alive(x, t.side, t.slot) && d_collapsed(x, t.side, t.slot))
    d_damage(x, t, kit_insight_damage(x), kit_self(x), 0, 0, 1, 0, 0, 0);
}

DFN void kit_card_P01(DX* x) {
  DEffect im = kit_eff("heiyu.combat_immune", "rule.combat_immune", 1, "allow");
  im.stacking="replace";im.source_key="P01";im.source_entity=d_entity(x,kit_self(x));im.clear_on="battle";im.clear_on2="operation_end";im.clear_on3="down";im.label="\xe6\x88\x98\xe6\x96\x97\xe5\x85\x8d\xe4\xbc\xa4";im.show_marker=0;
  d_effect(x, kit_self(x), &im);
  d_sortie(x, 0);
}
DFN void kit_card_P02(DX* x) { d_damage(x, d_op_target(x, "target_id"), 3, kit_self(x), 0, 0, 1, 0, 0, 0); }
DFN void kit_card_P03(DX* x) {
  d_draw(x, x->side, 1, -1);
  if (d_zone_count_hand(x, x->side) > 0 && !d_finished(x)) {
    d_set_op_int(x, "heiyu_after_placement", 1, 1);
    d_choice(x, "hand_deck_placement", d_zone_count_hand(x, x->side));
  } else d_sortie(x, 0);
}
DFN void kit_card_P04(DX* x) { d_sortie(x, 0); }
DFN void kit_card_P05(DX*x){int foe=1-x->side,items[C],n=0;for(int j=0;j<d_order_count(x,foe);j++){int a=d_order_slot(x,foe,j);if(d_alive(x,foe,a)&&!d_cb(x,foe,a,"summoned"))items[n++]=a;}int count=d_pi(x,x->side,"heiyu_divinations");if(count>n)count=n;int selected[C];d_sample_before(x,items,n,count,selected);for(int j=0;j<count&&!d_finished(x);j++)if(d_alive(x,foe,selected[j]))d_damage(x,DR{foe,selected[j]},2,kit_self(x));kit_heiyu_start_inspection(x,0,0);}
DFN void kit_card_P06(DX* x) {
  DR t = d_op_target(x, "target_id");
  d_revive(x, t.side, t.slot);
  kit_heiyu_start_inspection(x, 0, 0);
}
DFN void kit_card_PF01(DX* x) { kit_heiyu_start_inspection(x, 1, 1); }

DFN void d_card_effect(DX* x) {
  if (d_finished(x) || d_bad(x)) return;
  int card = x->card;
  if(card==-1){d_fail(x,1);return;}
  if (d_card_is(x, card, "N01")) { kit_card_N01(x); return; }
  if (d_card_is(x, card, "N02")) { kit_card_N02(x); return; }
  if (d_card_is(x, card, "N03")) { kit_card_N03(x); return; }
  if (d_card_is(x, card, "N04")) { kit_card_N04(x); return; }
  if (d_card_is(x, card, "N05")) { kit_card_N05(x); return; }
  if (d_card_is(x, card, "N06")) { kit_card_N06(x); return; }
  if (d_card_is(x, card, "N07") || d_card_is(x, card, "N08")) { kit_card_shape(x); return; }
  if (d_card_is(x, card, "NF01")) { kit_card_NF01(x); return; }
  if (d_card_is(x, card, "Z01")) { kit_card_Z01(x); return; }
  if (d_card_is(x, card, "Z02")) { kit_card_Z02(x); return; }
  if (d_card_is(x, card, "Z03")) { kit_card_Z03(x); return; }
  if (d_card_is(x, card, "Z04")) { kit_card_Z04(x); return; }
  if (d_card_is(x, card, "Z05")) { kit_card_Z05(x); return; }
  if (d_card_is(x, card, "Z06") || d_card_is(x, card, "Z07") || d_card_is(x, card, "Z08")) { kit_card_shape(x); return; }
  if (d_card_is(x, card, "J01")) { kit_card_J01(x); return; }
  if (d_card_is(x, card, "J02")) { kit_card_J02(x); return; }
  if (d_card_is(x, card, "J03")) { kit_card_J03(x); return; }
  if (d_card_is(x, card, "J04")) { kit_card_J04(x); return; }
  if (d_card_is(x, card, "J05")) { kit_card_J05(x); return; }
  if (d_card_is(x, card, "J06")) { kit_card_J06(x); return; }
  if (d_card_is(x, card, "J07") || d_card_is(x, card, "J08")) { kit_card_shape(x); return; }
  if (d_card_is(x, card, "Y01")) { kit_card_Y01(x); return; }
  if (d_card_is(x, card, "Y02")) { kit_card_Y02(x); return; }
  if (d_card_is(x, card, "Y03")) { kit_card_Y03(x); return; }
  if (d_card_is(x, card, "Y04")) { kit_card_Y04(x); return; }
  if (d_card_is(x, card, "Y05")) { kit_card_Y05(x); return; }
  if (d_card_is(x, card, "Y06")) { kit_card_Y06(x); return; }
  if (d_card_is(x, card, "Y07")) { kit_card_Y07(x); return; }
  if (d_card_is(x, card, "Y08")) { kit_card_shape(x); return; }
  if (d_card_is(x, card, "M01")) { kit_card_M01(x); return; }
  if (d_card_is(x, card, "M02")) { kit_card_M02(x); return; }
  if (d_card_is(x, card, "M03")) { kit_card_M03(x); return; }
  if (d_card_is(x, card, "M04")) { kit_card_M04(x); return; }
  if (d_card_is(x, card, "M05")) { kit_card_M05(x); return; }
  if (d_card_is(x, card, "M06")) { kit_card_M06(x); return; }
  if (d_card_is(x, card, "M07") || d_card_is(x, card, "M08")) { kit_card_shape(x); return; }
  if (d_card_is(x, card, "B01")) { kit_card_B01(x); return; }
  if (d_card_is(x, card, "B02")) { kit_card_B02(x); return; }
  if (d_card_is(x, card, "B03")) { kit_card_B03(x); return; }
  if (d_card_is(x, card, "B04")) { kit_card_B04(x); return; }
  if (d_card_is(x, card, "B05")) { kit_card_B05(x); return; }
  if (d_card_is(x, card, "B06")) { kit_card_B06(x); return; }
  if (d_card_is(x, card, "B07")) { kit_card_shape(x); return; }
  if (d_card_is(x, card, "B08")) { kit_card_B08(x); return; }
  if (d_card_is(x, card, "Q01")) { kit_card_Q01(x); return; }
  if (d_card_is(x, card, "Q02")) { kit_card_Q02(x); return; }
  if (d_card_is(x, card, "Q03")) { kit_card_Q03(x); return; }
  if (d_card_is(x, card, "Q04")) { kit_card_Q04(x); return; }
  if (d_card_is(x, card, "Q05")) { kit_card_Q05(x); return; }
  if (d_card_is(x, card, "Q06")) { kit_card_Q06(x); return; }
  if (d_card_is(x, card, "Q07") || d_card_is(x, card, "Q08")) { kit_card_shape(x); return; }
  if (d_card_is(x, card, "U01")) { kit_card_U01(x); return; }
  if (d_card_is(x, card, "U02")) { kit_card_U02(x); return; }
  if (d_card_is(x, card, "U03")) { kit_card_U03(x); return; }
  if (d_card_is(x, card, "U04")) { kit_card_U04(x); return; }
  if (d_card_is(x, card, "U05") || d_card_is(x, card, "U06") || d_card_is(x, card, "U07") || d_card_is(x, card, "U08")) { kit_card_shape(x); return; }
  if (d_card_is(x, card, "I01")) { kit_card_I01(x); return; }
  if (d_card_is(x, card, "I02")) { kit_card_I02(x); return; }
  if (d_card_is(x, card, "I03")) { kit_card_I03(x); return; }
  if (d_card_is(x, card, "I04")) { kit_card_I04(x); return; }
  if (d_card_is(x, card, "I05")) { kit_card_I05(x); return; }
  if (d_card_is(x, card, "I06")) { kit_card_I06(x); return; }
  if (d_card_is(x, card, "I07") || d_card_is(x, card, "I08")) { kit_card_shape(x); return; }
  if (d_card_is(x, card, "R01")) { kit_card_R01(x); return; }
  if (d_card_is(x, card, "R02") || d_card_is(x, card, "R03") || d_card_is(x, card, "R04") || d_card_is(x, card, "R06")) { kit_card_R_battle(x); return; }
  if (d_card_is(x, card, "RF01")) { kit_card_RF01(x); return; }
  if (d_card_is(x, card, "R05")) { kit_card_R05(x); return; }
  if (d_card_is(x, card, "R07") || d_card_is(x, card, "R08")) { kit_card_R_shape(x); return; }
  if (d_card_is(x, card, "A01")) { kit_card_A01(x); return; }
  if (d_card_is(x, card, "A02")) { kit_card_A02(x); return; }
  if (d_card_is(x, card, "A03")) { kit_card_A03(x); return; }
  if (d_card_is(x, card, "A04")) { kit_card_A04(x); return; }
  if (d_card_is(x, card, "A05")) { kit_card_A05(x); return; }
  if (d_card_is(x, card, "A06")) { kit_card_A06(x); return; }
  if (d_card_is(x, card, "A07")) { kit_card_A07(x); return; }
  if (d_card_is(x, card, "A08")) { kit_card_shape(x); return; }
  if (d_card_is(x, card, "AF01")) { kit_card_AF01(x); return; }
  if (d_card_is(x, card, "C01")) { kit_card_C01(x); return; }
  if (d_card_is(x, card, "C02")) { kit_card_C02(x); return; }
  if (d_card_is(x, card, "C03")) { kit_card_C03(x); return; }
  if (d_card_is(x, card, "C04")) { kit_card_C04(x); return; }
  if (d_card_is(x, card, "C05")) { kit_card_C05(x); return; }
  if (d_card_is(x, card, "C06") || d_card_is(x, card, "C08")) { kit_card_shape(x); return; }
  if (d_card_is(x, card, "C07")) { kit_card_C07(x); return; }
  if (d_card_is(x, card, "S01")) { kit_card_S01(x); return; }
  if (d_card_is(x, card, "S02")) { kit_card_S02(x); return; }
  if (d_card_is(x, card, "S03")) { kit_card_S03(x); return; }
  if (d_card_is(x, card, "S04")) { kit_card_S04(x); return; }
  if (d_card_is(x, card, "S05")) { kit_card_S05(x); return; }
  if (d_card_is(x, card, "S06")) { kit_card_S06(x); return; }
  if (d_card_is(x, card, "S07")) { kit_card_S07(x); return; }
  if (d_card_is(x, card, "S08")) { kit_card_shape(x); return; }
  if (d_card_is(x, card, "D01")) { kit_card_D01(x); return; }
  if (d_card_is(x, card, "D02")) { kit_card_D02(x); return; }
  if (d_card_is(x, card, "D03")) { kit_card_D03(x); return; }
  if (d_card_is(x, card, "D04")) { kit_card_D04(x); return; }
  if (d_card_is(x, card, "D05")) { kit_card_D05(x); return; }
  if (d_card_is(x, card, "D06")) { kit_card_D06(x); return; }
  if (d_card_is(x, card, "D07") || d_card_is(x, card, "D08")) { kit_card_shape(x); return; }
  if (d_card_is(x, card, "K01")) { kit_card_K01(x); return; }
  if (d_card_is(x, card, "K02")) { kit_card_K02(x); return; }
  if (d_card_is(x, card, "K03")) { kit_card_K03(x); return; }
  if (d_card_is(x, card, "K04")) { kit_card_K04(x); return; }
  if (d_card_is(x, card, "K05")) { kit_card_K05(x); return; }
  if (d_card_is(x, card, "K06")) { kit_card_K06(x); return; }
  if (d_card_is(x, card, "K07") || d_card_is(x, card, "K08")) { kit_card_shape(x); return; }
  if (d_card_is(x, card, "F01")) { kit_card_F01(x); return; }
  if (d_card_is(x, card, "F02") || d_card_is(x, card, "F08")) { kit_card_shape(x); return; }
  if (d_card_is(x, card, "F03")) { kit_card_F03(x); return; }
  if (d_card_is(x, card, "F04")) { kit_card_F04(x); return; }
  if (d_card_is(x, card, "F05")) { kit_card_F05(x); return; }
  if (d_card_is(x, card, "F06")) { kit_card_F06(x); return; }
  if (d_card_is(x, card, "F07")) { kit_card_F07(x); return; }
  if (d_card_is(x, card, "FF01")) { kit_card_FF01(x); return; }
  if (d_card_is(x, card, "P01")) { kit_card_P01(x); return; }
  if (d_card_is(x, card, "P02")) { kit_card_P02(x); return; }
  if (d_card_is(x, card, "P03")) { kit_card_P03(x); return; }
  if (d_card_is(x, card, "P04")) { kit_card_P04(x); return; }
  if (d_card_is(x, card, "P05")) { kit_card_P05(x); return; }
  if (d_card_is(x, card, "P06")) { kit_card_P06(x); return; }
  if (d_card_is(x, card, "P07") || d_card_is(x, card, "P08")) { kit_card_shape(x); return; }
  if (d_card_is(x, card, "PF01")) { kit_card_PF01(x); return; }
  d_fail(x, 4);
}

DFN int kit_in_catalog(int tmpl) {
  return tmpl == ID_zero || tmpl == ID_zhenhong || tmpl == ID_xiaozhi || tmpl == ID_nanali
      || tmpl == ID_jiuyuan || tmpl == ID_lingke || tmpl == ID_iloy || tmpl == ID_bohe
      || tmpl == ID_yi || tmpl == ID_anhunqu || tmpl == ID_dafutier || tmpl == ID_canhong
      || tmpl == ID_zaowu || tmpl == ID_baicang || tmpl == ID_adler || tmpl == ID_haiyue
      || tmpl == ID_heiyu;
}

DFN void d_fire(DX* x, const char* hook, const DEvt* event, int actor_only, int include_down) {
  if (d_finished(x) || d_bad(x) || d_passives_disabled(x)) return;
  int saved = x->actor;
  int side = x->side;
  const char* CATALOG[17] = {
    "zero","zhenhong","xiaozhi","nanali","jiuyuan","lingke","iloy","bohe","yi",
    "anhunqu","dafutier","canhong","zaowu","baicang","adler","haiyue","heiyu"
  };
  int ci, slot, i, n;
  for (ci = 0; ci < 17; ++ci) {
    if (d_finished(x) || d_bad(x)) break;
    slot = d_hero(x, side, CATALOG[ci]);
    if (slot < 0) continue;
    if (actor_only && slot != saved) continue;
    if (!include_down && !d_alive(x, side, slot)) continue;
    x->actor = slot;
    kit_fire_passive(x, hook, event);
    kit_fire_weapons(x, hook, event);
  }
  n = d_order_count(x, side);
  for (i = 0; i < n; ++i) {
    if (d_finished(x) || d_bad(x)) break;
    slot = d_order_slot(x, side, i);
    if (kit_in_catalog(d_template(x, side, slot))) continue;
    if (actor_only && slot != saved) continue;
    if (!include_down && !d_alive(x, side, slot)) continue;
    x->actor = slot;
    kit_fire_passive(x, hook, event);
    kit_fire_weapons(x, hook, event);
  }
  x->actor = saved;
}

DFN int d_decide(DX* x, const char* hook, int current, const DEvt* event, int actor_only, int include_down) {
  if (d_finished(x) || d_bad(x) || d_passives_disabled(x)) return current;
  int saved = x->actor, side = x->side, value = current, replaced = 0;
  const char* CATALOG[17] = {
    "zero","zhenhong","xiaozhi","nanali","jiuyuan","lingke","iloy","bohe","yi",
    "anhunqu","dafutier","canhong","zaowu","baicang","adler","haiyue","heiyu"
  };
  int ci, slot, i, n;
  for (ci = 0; ci < 17; ++ci) {
    slot = d_hero(x, side, CATALOG[ci]);
    if (slot < 0) continue;
    if (actor_only && slot != saved) continue;
    if (!include_down && !d_alive(x, side, slot)) continue;
    x->actor = slot;
    replaced = 0;
    value = kit_decide_passive(x, hook, value, event, &replaced);
    if (d_template(x, side, slot) == ID_heiyu && kit_weapon_equipped(x, slot, "P07") && kit_hook_is(hook, "combat_overflow")) {
      int wrep = 0; value = kit_weapon_P07_combat_overflow(x, value, event, &wrep);
    }
  }
  n = d_order_count(x, side);
  for (i = 0; i < n; ++i) {
    slot = d_order_slot(x, side, i);
    if (kit_in_catalog(d_template(x, side, slot))) continue;
    if (actor_only && slot != saved) continue;
    if (!include_down && !d_alive(x, side, slot)) continue;
    x->actor = slot;
    replaced = 0;
    value = kit_decide_passive(x, hook, value, event, &replaced);
  }
  x->actor = saved;
  return value;
}

DFN int d_tally(DX* x, const char* hook, const DEvt* event) {
  if (d_finished(x) || d_bad(x)) return 0;
  int saved = x->actor, side = x->side, total = 0;
  const char* CATALOG[17] = {
    "zero","zhenhong","xiaozhi","nanali","jiuyuan","lingke","iloy","bohe","yi",
    "anhunqu","dafutier","canhong","zaowu","baicang","adler","haiyue","heiyu"
  };
  int ci, slot, i, n;
  for (ci = 0; ci < 17; ++ci) {
    slot = d_hero(x, side, CATALOG[ci]);
    if (slot < 0 || !d_alive(x, side, slot)) continue;
    x->actor = slot;
    int mimic = d_mimic_template(x, side, slot);
    int tmpl = (mimic > 0) ? mimic : d_template(x, side, slot);
    if (mimic > 0) {
      int awoke = kit_awake(x, slot);
      d_set_cb(x, side, slot, "awakened", 0);
      total += kit_tally_named(x, tmpl, hook, event);
      d_set_cb(x, side, slot, "awakened", awoke);
    } else {
      total += kit_tally_named(x, tmpl, hook, event);
    }
  }
  n = d_order_count(x, side);
  for (i = 0; i < n; ++i) {
    slot = d_order_slot(x, side, i);
    if (kit_in_catalog(d_template(x, side, slot)) || !d_alive(x, side, slot)) continue;
    x->actor = slot;
    total += kit_tally_named(x, d_template(x, side, slot), hook, event);
  }
  x->actor = saved;
  return total;
}

DFN void d_on_shape(DX* x, const char* hook, const DEvt* event) {
  if (d_finished(x) || d_bad(x) || !d_alive(x, x->side, x->actor)) return;
  kit_on_shape(x, hook, event);
}

DFN int kit_hand_card_rewrite(DX* x, int card) {
  int tmpl = d_template(x, x->side, x->actor);
  if (tmpl == ID_zhenhong && kit_awake(x, x->actor) && !d_card_is(x, card, "unknown")) {
    d_rewrite_hand_as(x, card, "RF01");
    return 1;
  }
  if (tmpl == ID_lingke && kit_shape(x, x->actor, "K03") && d_ap(x, x->side) == 0
      && !d_used_instant(x, x->side) && d_card_owner_template(x, card) == ID_lingke
      && !d_card_is(x, card, "unknown")) {
    d_rewrite_k03_recovery(x, card);
    return 1;
  }
  if (tmpl == ID_heiyu && d_card_owner_template(x, card) == ID_heiyu) {
    d_restore_heiyu_spec(x, card);
    return 1;
  }
  return 0;
}

DFN DPayment d_decide_payment(DX*x,const char*hook,DPayment current,const DEvt*event){(void)event;DPayment p=current;if(!d_streq(hook,"normal_attack_payment"))return p;int kit=d_mimic_template(x,x->side,x->actor);if(!kit)kit=d_template(x,x->side,x->actor);if(kit==ID_xiaozhi&&p.ap<0&&d_cstr_is(x,x->side,x->actor,"shape","Q07")&&d_ci(x,x->side,x->actor,"jingu")>=3){p.ap=0;p.consume_normal=0;p.resource=PAY_JINGU;p.bonus=2;}if(kit==ID_haiyue&&d_cstr_is(x,x->side,x->actor,"shape","U06")&&!d_used(x,x->side,"instant")&&d_teammate_attacked(x,x->side,x->actor)){p.ap=0;p.consume_normal=0;p.resource=PAY_INSTANT;p.bonus=0;}return p;}
DFN DTargetList d_decide_targets(DX* x, const char* hook, const DEvt* event) {
  (void)event;
  DTargetList empty; empty.n = 0;
  if (d_finished(x) || d_bad(x) || d_passives_disabled(x)) return empty;
  int saved = x->actor, side = x->side, ci, slot;
  const char* CATALOG[17] = {
    "zero","zhenhong","xiaozhi","nanali","jiuyuan","lingke","iloy","bohe","yi",
    "anhunqu","dafutier","canhong","zaowu","baicang","adler","haiyue","heiyu"
  };
  DTargetList value = empty;
  for (ci = 0; ci < 17; ++ci) {
    slot = d_hero(x, side, CATALOG[ci]);
    if (slot < 0 || !d_alive(x, side, slot)) continue;
    x->actor = slot;
    DTargetList got = kit_decide_targets(x, hook);
    if (got.n) value = got;
  }
  x->actor = saved;
  return value;
}
''';
