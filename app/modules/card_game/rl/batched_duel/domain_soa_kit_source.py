"""C++ emission for public V2 kit hooks, weapons, clocks and continuations."""

from app.modules.card_game.content.duel_v2.catalog import CHARACTER_ORDER, PUBLIC_CHARACTER_IDS

PUBLIC_KIT_ORDER = tuple(cid for cid in CHARACTER_ORDER if cid in PUBLIC_CHARACTER_IDS)
WEAPON_HOOKS = (
    ('zero', 'Z08', 'on_turn_end'),
    ('zero', 'Z07', 'on_turn_end'),
    ('zero', 'Z06', 'on_turn_end'),
    ('anhunqu', 'A07', 'on_turn_begin'),
    ('dafutier', 'F08', 'on_opponent_sortie_finished'),
    ('heiyu', 'P07', 'combat_overflow'),
)
CHOICE_CONTINUATIONS = ('heiyu.divination',)


def emit_source():
    from app.modules.card_game.content.duel_v2.registry import KITS, _ensure_kits
    _ensure_kits()
    if any(getattr(KITS[cid], "shape_events", None) for cid in PUBLIC_KIT_ORDER):
        raise RuntimeError("Unported public shape-event provider")
    return _SOURCE


_SOURCE = r'''
/* Typed non-integer decision results. Do not truncate these to int. */
#ifndef D_PAYMENT_DEFINED
#define D_PAYMENT_DEFINED
enum { PAY_NONE = 0, PAY_JINGU = 1, PAY_INSTANT = 2 };
struct DPayment { int ap; int consume_normal; int resource; int bonus; };
struct DTargetList { int n; DR items[16]; };
#endif
DFN int d_tally(DX* x, const char* hook, const DEvt* event);
DFN int kit_streq(const char* a, const char* b) {
  if (a == 0 || b == 0) return a == b;
  int i = 0;
  for (; a[i] && b[i]; ++i) if (a[i] != b[i]) return 0;
  return a[i] == b[i];
}
DFN int kit_hook_is(const char* hook, const char* name) { return kit_streq(hook, name); }
DFN DR kit_self(DX* x) { DR r; r.side = x->side; r.slot = x->actor; return r; }
DFN DR kit_player(int side) { DR r; r.side = side; r.slot = -1; return r; }
DFN DR kit_none() { DR r; r.side = -1; r.slot = -2; return r; }
DFN DR kit_dr(int side, int slot) { DR r; r.side = side; r.slot = slot; return r; }
DFN int kit_awake(DX* x, int actor) { return d_cb(x, x->side, actor, "awakened"); }
DFN int kit_shape(DX* x, int actor, const char* card_id) { return d_cstr_is(x, x->side, actor, "shape", card_id); }
DFN int kit_front_is(DX* x, int actor) { return d_front(x, x->side) == actor; }
DFN DAttack kit_attack0() {
  DAttack a;
  a.bonus = 0; a.ignore_shield = 0; a.no_counter = 0; a.no_attack = 0; a.hits = 0;
  a.return_after = 0; a.draw_after = 0; a.low_hp_bonus = 0; a.low_hp_max = 0;
  a.awake_player_bonus = 0; a.has_awake_player_bonus = 0; a.harmony_bonus = 0;
  a.additional_damage_bonus = 0;
  return a;
}
DFN DEffect kit_eff(const char* id, const char* property, double value, const char* op) {
  DEffect e{};
  e.id = id; e.property = property; e.value = value; e.op = op; e.layer = 0;
  e.scope = "self"; e.stacking = "independent"; e.stack_key = nullptr;
  e.expiry_side = -1; e.expiry_turn = -1; e.expiry_phase = nullptr;
  e.consume_on = nullptr; e.clear_on = nullptr; e.condition = nullptr;
  e.condition_key = nullptr; e.condition_string = nullptr; e.condition_number = 0;e.value_kind=(kit_streq(op,"allow")||kit_streq(op,"forbid"))?2:-1;
  return e;
}
DFN void kit_expiry(DEffect* e, DX* x, int side, int turns, const char* phase) {
  e->expiry_side = side;
  e->expiry_turn = d_pi(x, side, "turn_count") + turns;
  e->expiry_phase = phase;
}
DFN void kit_atk_buff(DX* x, int actor, int amount, const char* key) {
  if (amount == 0 || !d_alive(x, x->side, actor)) return;
  DEffect e = kit_eff("attack.buff", "attack.panel", (double)amount, "add");
  e.stack_key = nullptr;
  e.source_key = key;
  int id=d_card_id(key);e.label=id?d_card_default_str(id,KS_name):"\xe6\x94\xbb\xe5\x87\xbb\xe5\xa2\x9e\xe7\x9b\x8a";
  d_effect(x, kit_dr(x->side, actor), &e);if(d_streq(key,"Y03")||d_streq(key,"Y06")||d_streq(key,"Y07"))d_event(x);
}
DFN void kit_grant_harmony(DX* x, int actor, int amount) {
  if (d_finished(x) || d_bad(x) || amount <= 0) return;
  if(!d_alive(x,x->side,actor))return;
  int cap = d_harmony_cap(x);
  int cur = d_ci(x, x->side, actor, "harmony");
  int nxt = cur + amount;
  if (nxt > cap) nxt = cap;
  if(nxt!=cur){d_set_ci(x,x->side,actor,"harmony",nxt);d_event(x);}
}
DFN void kit_grant_seed(DX* x, int actor, int limit) {
  if (d_finished(x) || d_bad(x) || !d_alive(x, x->side, actor)) return;
  int gained = d_pi(x, x->side, "nanali_family");
  if (gained >= limit) return;
  d_set_pi(x, x->side, "nanali_family", gained + 1);
  int card = d_make_card(x, x->side, "NF01", actor);
  if (d_bad(x)) return;
  d_add_hand(x, x->side, card, 0);
}
DFN void kit_followup(DX* x) {
  if (d_finished(x) || d_bad(x)) return;
  int amount = 0;
  if (kit_awake(x, x->actor)) amount += 1;
  int pending = d_flag(x, x->side, x->actor, "next_followup");
  amount+=pending;d_del_flag(x,x->side,x->actor,"next_followup");
  if (x->card >= 0 && d_card_is(x, x->card, "N03")) amount += 1;
  if (kit_shape(x, x->actor, "N08")) amount += 2;
  if (amount <= 0) return;
  DR last = d_op_target(x, "last_attack_target");
  if (last.slot == -2) return;
  if(last.slot>=0&&(!d_alive(x,last.side,last.slot)||d_front(x,last.side)!=last.slot))return;combat_next_group(x);d_damage(x,last,amount,kit_self(x),10);
}
DFN int kit_living_count(DX* x, int side) {
  int n = 0, i, slot, m = d_order_count(x, side);
  for (i = 0; i < m; ++i) {
    slot = d_order_slot(x, side, i);
    if (d_alive(x, side, slot) && !d_cb(x, side, slot, "summoned")) ++n;
  }
  return n;
}
DFN int kit_downed_count(DX* x, int side) {
  int n = 0, i, slot, m = d_order_count(x, side);
  for (i = 0; i < m; ++i) {
    slot = d_order_slot(x, side, i);
    if (d_ci(x, side, slot, "down_turns") > 0) ++n;
  }
  return n;
}
DFN int kit_random_living(DX* x, int side) {
  int n = kit_living_count(x, side);
  if (n <= 0) return -2;
  int pick = d_random_before(x, n), i, slot, seen = 0, m = d_order_count(x, side);
  for (i = 0; i < m; ++i) {
    slot = d_order_slot(x, side, i);
    if (!d_alive(x, side, slot) || d_cb(x, side, slot, "summoned")) continue;
    if (seen == pick) return slot;
    ++seen;
  }
  return -2;
}
DFN int kit_weapon_equipped(DX* x, int actor, const char* card_id) {
  return kit_shape(x, actor, card_id) && d_weapon_matches(x, actor, card_id);
}

DFN void kit_bind_weapon_source(DX* x, int actor) {
  int h = x->side * C + actor;
  if (x->s->character_equipment_presence[h] == 2) x->effect_source_slot = x->s->character_equipment_id_slot[h];
}

DFN void kit_zero_after_card(DX* x, const DEvt* ev) {
  int card = (ev && ev->card >= 0) ? ev->card : x->card;
  if ((ev && ev->actor != x->actor) || card < 0) return;
  if (!(d_card_is(x, card, "Z03") || d_card_is(x, card, "Z04") || d_card_is(x, card, "Z05"))) return;
  if (!d_alive(x, x->side, x->actor)) return;
  kit_grant_harmony(x, x->actor, 2);
}
DFN void kit_weapon_Z08_on_turn_end(DX* x) {
  int front = d_front(x, x->side);
  if (front < 0) return;
  if (!d_alive(x, x->side, front)) return;
  DEffect e = kit_eff("attack.buff", "attack.panel", 1, "add");
  e.source_key = nullptr;
  e.label = "\xe5\xad\xa6\xe5\x9b\xad\xe9\x9a\x8f\xe6\x83\xb3";
  d_effect(x, kit_dr(x->side, front), &e);
  d_event(x);
}
DFN void kit_weapon_Z07_on_turn_end(DX* x) {
  int front = d_front(x, x->side);
  if (front < 0) return;
  d_heal(x, kit_dr(x->side, front), 2, kit_self(x));
}
DFN void kit_weapon_Z06_on_turn_end(DX* x) {
  d_add_ap(x, x->side, 1);
  d_event(x);
}

DFN void kit_nanali_on_turn_begin(DX* x) {
  if (d_pi(x, x->side, "turn_count") == 1) kit_grant_seed(x, x->actor, 3);
}
DFN void kit_nanali_on_rout(DX* x) {
  if (d_alive(x, x->side, x->actor) && kit_front_is(x, x->actor)) kit_grant_seed(x, x->actor, 3);
}
DFN void kit_nanali_after_sortie(DX* x) { kit_followup(x); }
DFN void kit_nanali_on_ultimate_end(DX* x) { (void)x; }

DFN void kit_jiuyuan_on_ultimate(DX* x) {
  int foe = d_enemy(x->side), i, slot, n = d_order_count(x, foe);
  for (i = 0; i < n; ++i) {
    if (d_finished(x) || d_bad(x)) return;
    slot = d_order_slot(x, foe, i);
    if (!d_alive(x, foe, slot)) continue;
    int amount = 1;
    if (d_flag(x, foe, slot, "pact")) {
      amount += 2 + d_tally(x, "attachment_damage_bonus", 0);
      d_del_flag(x, foe, slot, "pact");
      d_event(x);
      if (kit_shape(x, x->actor, "J07")) d_context_energy(x, x->side, x->actor, 1);
    }
    d_damage(x, kit_dr(foe, slot), amount, kit_self(x), 0, 0, 1, 0, 0, 0);
  }
}
DFN void kit_jiuyuan_on_genesis_damage(DX* x, const DEvt* ev) {
  if (!ev || ev->target_side < 0 || ev->target < -1) return;
  int side = ev->target_side, slot = ev->target;
  if (slot >= 0 && !d_alive(x, side, slot)) return;
  if(ev->amount&&slot>=0)d_attach_pact(x,DR{side,slot});
  if (kit_shape(x, x->actor, "J08"))
    d_damage(x, kit_dr(side, slot), 1, kit_self(x), 0, 0, 1, 0, 0, 0);
}
DFN int kit_jiuyuan_genesis_extra(DX* x) { return d_alive(x, x->side, x->actor) ? 1 : 0; }

DFN void kit_iloy_before_hits(DX* x) {
  int bonus = d_flag(x, x->side, x->actor, "next_bonus");
  if (bonus) {
    d_set_op_int(x, "first_bonus", d_op_int(x, "first_bonus", 0) + bonus, 0);
    d_del_flag(x, x->side, x->actor, "next_bonus");
    d_event(x);
  }
  int sh = d_flag(x, x->side, x->actor, "next_shield");
  if (sh) { d_del_flag(x, x->side, x->actor, "next_shield"); d_shield(x, kit_self(x), sh); }
}
DFN void kit_iloy_on_ultimate(DX* x) { d_set_flag(x, x->side, x->actor, "energy_next", 1, 1); d_event(x); }
DFN void kit_iloy_on_ultimate_end(DX* x) { d_del_flag(x, x->side, x->actor, "energy_next"); }
DFN void kit_iloy_y07_heal(DX* x) {
  int n = d_order_count(x, x->side), i, cand[8], cn = 0, slot;
  for (i = 0; i < n && cn < 8; ++i) {
    slot = d_order_slot(x, x->side, i);
    if (d_alive(x, x->side, slot) && d_ci(x, x->side, slot, "hp") < d_max_hp(x, x->side, slot)) cand[cn++] = slot;
  }
  if (cn <= 0) return;
  if(cn>1)d_shuffle_refs(x,cand,cn);slot=cand[0];
  d_heal(x,kit_dr(x->side,slot),2,DR{-1,-2});
  kit_atk_buff(x, slot, 1, "Y07");
}
DFN void kit_iloy_on_turn_begin(DX* x) {
  int pending=d_flag(x,x->side,x->actor,"energy_next");d_del_flag(x,x->side,x->actor,"energy_next");if(!pending)return;
  {
    int n = d_order_count(x, x->side), i, slot;
    for (i = 0; i < n; ++i) {
      slot = d_order_slot(x, x->side, i);
      if (d_alive(x, x->side, slot)) d_context_energy(x, x->side, slot, 1);
    }
  }
  if (kit_shape(x, x->actor, "Y07")) kit_iloy_y07_heal(x);
}
DFN void kit_iloy_after_genesis(DX* x, const DEvt* ev) {
  if (!kit_shape(x, x->actor, "Y08") || !d_alive(x, x->side, x->actor)) return;
  if (d_pb(x, x->side, "extra_genesis_pending")) return;
  d_set_pb(x, x->side, "extra_genesis_pending", 1);
  int at=x->side*PS,entering=ev&&ev->entering>=0?ev->entering:x->actor;x->s->player_str_presence[at]=2;x->s->player_str_slot[at]=x->s->character_id_slot[x->side*C+entering];
  d_event(x);
}

DFN int kit_bohe_combat_overflow(DX* x, int current, const DEvt* ev, int* replaced) {
  int raw = ev ? ev->raw_overflow : 0;
  if (ev && ev->attacker == x->actor && ev->attacker_side == x->side) { *replaced = 1; return raw; }
  if (ev && ev->defender == x->actor && ev->defender_side == x->side && d_flag(x, x->side, x->actor, "share_overflow")) {
    d_del_flag(x, x->side, x->actor, "share_overflow");
    *replaced = 1; return raw;
  }
  return current;
}
DFN int kit_bohe_incoming_damage(DX* x, int current, const DEvt* ev, int* replaced) {
  if (!ev || ev->target != x->actor || !kit_awake(x, x->actor)) return current;
  if(ev->kind==1||ev->kind==9||ev->kind==10)return current;
  if (ev->attacker_side == x->side) return current;
  if(ev->attacker_side>=0||ev->kind==2||ev->kind>=4) { *replaced = 1; return 0; }
  return current;
}
DFN void kit_bohe_after_hits(DX* x) { d_del_flag(x, x->side, x->actor, "share_overflow"); }
DFN void kit_bohe_on_turn_begin(DX* x) {
  d_del_flag(x, x->side, x->actor, "share_overflow");
  int pending = d_flag(x, x->side, x->actor, "pending_atk");
  if (pending) {
    d_del_flag(x, x->side, x->actor, "pending_atk");
    kit_atk_buff(x, x->actor, pending, "M05");
    d_event(x);
  }
  if (!kit_shape(x, x->actor, "M08")) return;
  if (d_finished(x) || !d_alive(x, x->side, x->actor) || d_collapsed(x, x->side, x->actor)) return;
  DAttack a = kit_attack0();
  d_event(x);
  d_sortie(x, &a);
  if (!d_finished(x)) d_finish_operation(x);
}
DFN void kit_bohe_on_enemy_turn_begin(DX* x) { d_del_flag(x, x->side, x->actor, "share_overflow"); }

DFN void kit_baicang_on_damage_taken(DX* x, const DEvt* ev) {
  if (!ev || ev->amount <= 0) return;
  kit_atk_buff(x, x->actor, 1, "ability");
  if (ev->attacker_side == x->side) d_set_flag(x, x->side, x->actor, "allied_hurt", 1, 1);
  d_event(x);
}
DFN void kit_baicang_on_damage_dealt(DX* x, const DEvt* ev) {
  if (!kit_awake(x, x->actor) || !ev || ev->amount <= 0) return;
  if (ev->target_side != d_enemy(x->side) || ev->target < 0) return;
  if (!d_alive(x, ev->target_side, ev->target)) return;
  if (ev->target_side == x->side && ev->target == x->actor) return;
  int hp = d_ci(x, ev->target_side, ev->target, "hp");
  if (hp > 0 && hp <= 2) { d_set_ci(x, ev->target_side, ev->target, "hp", 0); d_event(x); }
}
DFN void kit_baicang_after_hits(DX* x, const DEvt* ev) {
  if (x->card >= 0 && d_card_is(x, x->card, "B02") && ev && ev->loss > 0)
    d_heal(x, kit_self(x), 2, kit_self(x));
}
DFN int kit_baicang_hp_floor(DX* x, int current, const DEvt* ev, int* replaced) {
  if (ev && ev->target == x->actor && d_flag(x, x->side, x->actor, "hp_floor")) {
    *replaced = 1; return current > 1 ? current : 1;
  }
  return current;
}
DFN void kit_baicang_on_turn_begin(DX* x) {
  d_del_flag(x, x->side, x->actor, "allied_hurt");
  d_del_flag(x, x->side, x->actor, "hp_floor");
}
DFN void kit_baicang_on_enemy_turn_begin(DX* x) { d_del_flag(x, x->side, x->actor, "hp_floor"); }

DFN void kit_change_jingu(DX* x, int amount, int allow_downed) {
  if (d_finished(x) || d_bad(x)) return;
  if (!allow_downed && !d_alive(x, x->side, x->actor)) return;
  d_set_ci(x, x->side, x->actor, "jingu", d_ci(x, x->side, x->actor, "jingu") + amount);
  d_event(x);
}
DFN void kit_xiaozhi_mark_card(DX* x, int card) {
  d_set_card_jingu(x, card, d_randbelow(x, 3) - 1, d_entity(x, kit_self(x)));
}
DFN void kit_xiaozhi_clear_marks(DX* x) {
  int n = d_zone_count_hand(x, x->side), i, card;
  for (i = 0; i < n; ++i) { card = d_zone_card_at(x, x->side, 0, i); if (card >= 0) d_clear_card_jingu(x, card); }
}
DFN void kit_xiaozhi_refresh_marks(DX* x) {
  int n = d_zone_count_hand(x, x->side), i, card;
  for (i = 0; i < n; ++i) { card = d_zone_card_at(x, x->side, 0, i); if (card >= 0) kit_xiaozhi_mark_card(x, card); }d_event(x);
}
DFN void kit_xiaozhi_on_game_start(DX* x) { d_set_ci(x, x->side, x->actor, "jingu", 0); }
DFN void kit_xiaozhi_on_turn_begin(DX* x) {
  kit_change_jingu(x, 1, 0);
  if (kit_awake(x, x->actor)) kit_xiaozhi_refresh_marks(x);
}
DFN void kit_xiaozhi_on_damage_dealt(DX* x, const DEvt* ev) {
  if (ev && ev->target_side == d_enemy(x->side) && ev->target == -1 && ev->amount > 0) kit_change_jingu(x, 1, 0);
}
DFN void kit_xiaozhi_on_ultimate(DX* x) {
  if (d_cb(x, x->side, x->actor, "jingu_loan")) kit_change_jingu(x, -3, 0);
  d_set_cb(x, x->side, x->actor, "jingu_loan", 1);
  kit_change_jingu(x, 3, 0);
  kit_xiaozhi_refresh_marks(x);
}
DFN void kit_xiaozhi_on_ultimate_end(DX* x) {
  kit_xiaozhi_clear_marks(x);d_event(x);
  if (d_cb(x, x->side, x->actor, "jingu_loan")) {
    d_del_cb(x,x->side,x->actor,"jingu_loan");
    kit_change_jingu(x, -3, 1);
  }
}
DFN void kit_xiaozhi_on_hand_added(DX* x, const DEvt* ev) {
  if(kit_awake(x,x->actor)&&ev&&ev->card>=0){kit_xiaozhi_mark_card(x,ev->card);d_event(x);}
}
DFN void kit_xiaozhi_after_marked_card(DX*x,const DEvt*ev){(void)ev;if(x->s->op_card_mark_presence[0]!=2||x->s->op_card_mark_source_slot[0]!=d_entity(x,kit_self(x)))return;kit_change_jingu(x,x->s->op_card_mark_delta[0],1);}
DFN int kit_xiaozhi_attack_bonus(DX* x, const DEvt* ev) {
  if (ev && ev->attacker == x->actor && kit_shape(x, x->actor, "Q08") && x->card >= 0) {
    int j = d_ci(x, x->side, x->actor, "jingu");
    return (j < 0 ? 0 : j) / 2;
  }
  return 0;
}

DFN void kit_haiyue_after_sortie(DX* x) { d_set_op_int(x, "return_after_resources", d_entity(x, kit_self(x)), 0); }
DFN void kit_haiyue_after_operation_resources(DX* x) {
  if (d_op_int(x, "return_after_resources", 0) == d_entity(x, kit_self(x))) d_move_out(x, x->side, x->actor, 0);
}
DFN void kit_haiyue_before_attack(DX* x) {
  if (kit_shape(x, x->actor, "U08")) { kit_atk_buff(x, x->actor, 1, "U08"); d_event(x); }
}
DFN int kit_haiyue_dot_damage_bonus(DX* x, const DEvt* ev) {
  return (ev && ev->kind == 6 && kit_shape(x, x->actor, "U05")) ? 1 : 0;
}
DFN int kit_haiyue_sortie_from_bench(DX* x, int current, int* replaced) {
  if (kit_awake(x, x->actor) && !kit_front_is(x, x->actor)) { *replaced = 1; return 1; }
  return current;
}

DFN void kit_yi_fangs(DX* x, int turns) {
  int cur = d_ci(x, x->side, x->actor, "beast_fangs");
  if (turns > cur) d_set_ci(x, x->side, x->actor, "beast_fangs", turns);
  d_event(x);
}
DFN void kit_yi_weapon_damage(DX* x) {
  if (!kit_shape(x, x->actor, "I07") || d_ci(x, x->side, x->actor, "beast_fangs") <= 0) return;
  int foe = d_enemy(x->side), front = d_front(x, foe);
  if (front >= 0 && !d_cb(x, foe, front, "summoned"))
    d_damage(x, kit_dr(foe, front), 1, kit_self(x), 8, 0, 1, 0, 0, 0);
}
DFN void kit_yi_on_ultimate(DX* x) {
  d_set_ci(x, x->side, x->actor, "beast_fangs", d_ci(x, x->side, x->actor, "beast_fangs") + 1);
  d_event(x);
}
DFN int kit_yi_attachment_damage_bonus(DX* x) { return kit_awake(x, x->actor) ? 1 : 0; }
DFN int kit_yi_pause_delay(DX* x, int current, int* replaced) {
  if (d_ci(x, x->side, x->actor, "beast_fangs") > 0) { *replaced = 1; return 1; }
  return current;
}
DFN void kit_yi_on_turn_countdown(DX* x) {
  int f = d_ci(x, x->side, x->actor, "beast_fangs");
  if (f > 0) { d_set_ci(x, x->side, x->actor, "beast_fangs", f - 1); d_event(x); }
}
DFN void kit_yi_after_hits(DX* x) {
  if (!d_zone_has(x, d_enemy(x->side), "delay")) return;
  int by = d_zone_int(x, d_enemy(x->side), "delay", "by", -1);
  int left = d_zone_int(x, d_enemy(x->side), "delay", "left", 0);
  if (by == x->side && left < 2) {
    int nxt = left + 1; if (nxt > 2) nxt = 2;
    d_zone_set(x, d_enemy(x->side), "delay", "left", nxt);
    d_event(x);
  }
}
DFN int kit_yi_can_play(DX* x, int current, int* replaced) {
  if (x->card >= 0 && d_card_is(x, x->card, "I05")) {
    *replaced = 1; return current && d_zone_has(x, d_enemy(x->side), "delay") ? 1 : 0;
  }
  return current;
}

DFN void kit_zhenhong_on_game_start(DX* x) { kit_grant_harmony(x, x->actor, 1); }
DFN void kit_zhenhong_on_surplus(DX* x) {
  if (d_finished(x) || !d_alive(x, x->side, x->actor)) return;
  d_set_ci(x, x->side, x->actor, "surplus_passive_triggers", d_ci(x, x->side, x->actor, "surplus_passive_triggers") + 1);
  DEffect atk = kit_eff("zhenhong.surplus_attack", "attack.panel", 1, "add");
  atk.source_key = "passive"; atk.label = "\xe7\x9c\x9f\xe7\xba\xa2\xe5\xbc\x82\xe8\x83\xbd"; atk.show_marker = 0;
  kit_expiry(&atk, x, x->side, 1, "turn_start");
  d_effect(x, kit_self(x), &atk);
  DEffect lim = kit_eff("zhenhong.surplus_damage_limit", "damage.limit", 1, "set");
  lim.stacking = "refresh"; lim.source_key = "passive"; lim.label = "\xe7\x9c\x9f\xe7\xba\xa2\xe4\xbc\xa4\xe5\xae\xb3\xe9\x99\x90\xe5\x88\xb6"; lim.show_marker = 0;
  kit_expiry(&lim, x, x->side, 1, "turn_start");
  d_effect(x, kit_self(x), &lim);
  d_event(x);
  d_heal(x, kit_self(x), d_max_hp(x, x->side, x->actor), kit_self(x));
  if (d_finished(x)) return;
  d_defer(x, "surplus", x->actor, 0);
  if (kit_shape(x, x->actor, "R07")) {
    int foe = d_enemy(x->side), i, slot, m = d_order_count(x, foe);
    for (i = 0; i < m; ++i) {
      if (d_finished(x)) break;
      slot = d_order_slot(x, foe, i);
      if (d_alive(x, foe, slot) && !d_cb(x, foe, slot, "summoned"))
        d_damage(x, kit_dr(foe, slot), 2, kit_self(x), 0, 0, 1, 0, 0, 0);
    }
  }
}
DFN void kit_zhenhong_on_deferred_action(DX* x, const DEvt* ev) {
  if (!ev || !kit_streq(d_deferred_kind(x, ev), "surplus")) return;
  d_set_ci(x, x->side, x->actor, "extra_attacks", d_ci(x, x->side, x->actor, "extra_attacks") + 1);
  d_set_op_int(x, "skip_combat_harmony", 1, 1);
  d_event(x);
  d_sortie(x, 0);
}
DFN int kit_zhenhong_natural_return(DX* x, int current, int* replaced) {
  if (kit_awake(x, x->actor)) { *replaced = 1; return 0; }
  return current;
}
DFN void kit_zhenhong_on_ultimate(DX* x) { d_event(x); d_enter(x, 0, 1); }
DFN void kit_zhenhong_on_effect_expired(DX* x, const DEvt* ev) {
  if (!ev) return;
  if (ev->effect_definition_slot < 0 || !d_eq(x, ev->effect_definition_slot, "status.ultimate")) return;
  if (ev->effect_source_entity != d_entity(x, kit_self(x))) return;
  int foe = d_enemy(x->side), i, slot, m = d_order_count(x, foe);
  for (i = 0; i < m; ++i) {
    if (d_finished(x)) return;
    slot = d_order_slot(x, foe, i);
    if (d_alive(x, foe, slot)) d_damage(x, kit_dr(foe, slot), 3, kit_self(x), 0, 0, 1, 0, 0, 0);
  }
  if (!d_finished(x)) d_damage(x, kit_player(foe), 3, kit_self(x), 0, 0, 1, 0, 0, 0);
}
DFN void kit_zhenhong_on_ultimate_end(DX* x) { d_event(x); }
DFN void kit_zhenhong_on_leave_front(DX* x) { d_end_ultimate(x, x->actor); }
DFN void kit_zhenhong_on_knockdown(DX* x) { d_set_ci(x, x->side, x->actor, "intimidation", 0); }
DFN void kit_zhenhong_on_turn_end(DX* x) {
  if (kit_shape(x, x->actor, "R08") && !d_attacked_this_turn(x, x->actor)) {
    int v = d_ci(x, x->side, x->actor, "intimidation") + 1;
    if (v > 2) v = 2;
    d_set_ci(x, x->side, x->actor, "intimidation", v);
    d_event(x);
  }
}
DFN void kit_zhenhong_on_sortie_ready(DX* x, const DEvt* ev) {
  if (!ev || ev->actor != x->actor || !kit_shape(x, x->actor, "R08")) return;
  int n = d_ci(x, x->side, x->actor, "intimidation");
  d_set_ci(x, x->side, x->actor, "intimidation", 0);
  d_set_op_int(x, "sortie_attack_bonus", d_op_int(x, "sortie_attack_bonus", 0) + n, 0);
  if (n) kit_grant_harmony(x, x->actor, n);
}

DFN int kit_anhunqu_stacks(DX* x) {
  int foe = d_enemy(x->side);
  if (!d_zone_has(x, foe, "nightmare")) return 0;
  if (d_zone_int(x, foe, "nightmare", "by", -1) != x->side) return 0;
  return d_zone_int(x, foe, "nightmare", "stacks", 0);
}
DFN void kit_weapon_A07_on_turn_begin(DX* x) {
  d_apply_dot(x, d_enemy(x->side), "nightmare", x->side, x->actor, 2, -1);
  d_event(x);
}
DFN void kit_anhunqu_on_ultimate(DX* x) {
  d_apply_dot(x, d_enemy(x->side), "nightmare", x->side, x->actor, 3, -1);
  d_event(x);
}
DFN void kit_anhunqu_after_hits(DX* x) {
  if (d_op_int(x, "skip_nightmare", 0) || !d_alive(x, x->side, x->actor)) return;
  d_apply_dot(x, d_enemy(x->side), "nightmare", x->side, x->actor, 1, -1);
}
DFN void kit_anhunqu_on_dissonance(DX* x, const DEvt* ev) {
  int enemy = ev && ev->target_side >= 0 ? ev->target_side : d_enemy(x->side);
  int front = d_front(x, enemy);
  if (front >= 0 && d_alive(x, enemy, front)) d_damage(x, kit_dr(enemy, front), 1, kit_self(x), 0, 0, 1, 0, 0, 0);
}

DFN void kit_canhong_enter_mirage(DX* x) {
  DEffect e = kit_eff("canhong.mirage", "status.mirage", 1, "add");
  e.stacking = "replace"; e.source_key = "mirage"; e.label = "\xe5\xb9\xbb\xe5\xa2\x83"; e.clear_on = "down";
  kit_expiry(&e, x, x->side, 3, "turn_start");
  d_effect(x, kit_self(x), &e);
  d_event(x);
}
DFN int kit_canhong_mirage(DX* x) { return d_has_effect_id(x, kit_self(x), "canhong.mirage"); }
DFN int kit_canhong_burn_extra_stack(DX* x, int current, int* replaced) {
  (void)current;
  if (d_alive(x, x->side, x->actor)) { *replaced = 1; return 1; }
  return 0;
}
DFN void kit_canhong_on_game_start(DX* x) { kit_grant_harmony(x, x->actor, 2); }
DFN void kit_canhong_on_ultimate(DX* x) {
  d_apply_dot(x, d_enemy(x->side), "venom", x->side, x->actor, 5, -1);
  d_event(x);
}
DFN void kit_canhong_on_card_used(DX* x, const DEvt* ev) {
  if (!ev || ev->actor != x->actor || ev->card < 0) return;
  if (d_card_is(x, ev->card, "C01") || d_card_is(x, ev->card, "C02") || d_card_is(x, ev->card, "C03")
      || d_card_is(x, ev->card, "C04") || d_card_is(x, ev->card, "C05")) {
    if (d_alive(x, x->side, x->actor)) kit_canhong_enter_mirage(x);
  }
}
DFN int kit_canhong_damage_is_additional(DX* x, int current, int* replaced) {
  if (d_alive(x, x->side, x->actor) && kit_canhong_mirage(x)) { *replaced = 1; return 1; }
  return current;
}
DFN void kit_canhong_after_hits(DX* x, const DEvt* ev) {
  if (!kit_canhong_mirage(x) || !d_alive(x, x->side, x->actor)) return;
  int foe = d_enemy(x->side);
  d_apply_dot(x, foe, "etch", x->side, x->actor, 1, -1);
  if (ev && ev->loss > 0) d_add_stack_all(x, foe, x->side, 1);
}
DFN void kit_canhong_on_damage_dealt(DX* x, const DEvt* ev) {
  if (ev && ev->kind != 1 && ev->amount > 0 && ev->target_side == d_enemy(x->side)
      && kit_canhong_mirage(x) && d_alive(x, x->side, x->actor))
    d_add_stack_all(x, ev->target_side, x->side, 1);
}
DFN void kit_canhong_on_replaced(DX* x) {
  if (kit_shape(x, x->actor, "C06")) d_damage(x, d_front_target(x), 1, kit_self(x), 0, 0, 1, 0, 0, 0);
}
DFN void kit_canhong_on_ally_down(DX* x, const DEvt* ev) {
  if (ev && ev->target == x->actor && d_total_stacks(x, x->side) >= 20) d_resolve_release_named(x, "C07");
}
DFN int kit_canhong_hp_floor(DX* x, int current, const DEvt* ev, int* replaced) {
  if (ev && ev->target == x->actor && x->card >= 0 && d_card_is(x, x->card, "C01") && d_total_stacks(x, x->side) >= 5) {
    *replaced = 1; return 1;
  }
  return current;
}
DFN int kit_canhong_combat_overflow(DX* x, int current, const DEvt* ev, int* replaced) {
  if (ev && ev->attacker_side>=0 && ev->attacker>=0 && x->s->character_id_slot[ev->attacker_side*C+ev->attacker]==x->s->character_id_slot[x->side*C+x->actor] && kit_shape(x, x->actor, "C08") && d_total_stacks(x, x->side) >= 5) {
    *replaced = 1; return ev->raw_overflow;
  }
  return current;
}

DFN int kit_zaowu_gulang(DX* x) {
  int n = d_order_count(x, x->side), i, slot;
  for (i = 0; i < n; ++i) {
    slot = d_order_slot(x, x->side, i);
    if (d_alive(x, x->side, slot) && d_cb(x, x->side, slot, "summoned") && d_cstr_is(x,x->side,slot,"name","\xe9\xac\xbc\xe9\x83\x8e\xe4\xb8\xb8")) return slot;
  }
  return -2;
}
DFN void kit_zaowu_maybe_recall(DX* x) {
  if (!kit_shape(x, x->actor, "S07") || d_kind_count(x, x->side) < 4) return;
  if (!d_alive(x, x->side, x->actor)) return;
  int item = d_random_before(x, 3);
  int slot = d_summon(x, item==0?"vinyl":item==1?"collar":"liquid");
  if (slot < 0 || d_bad(x)) return;
  d_damage(x, kit_dr(x->side, slot), 3, kit_self(x), 0, 0, 1, 0, 0, 0);
}
DFN void kit_zaowu_on_turn_begin(DX* x) { kit_zaowu_maybe_recall(x); }
DFN void kit_zaowu_on_ultimate(DX* x) {
  int bonus = kit_shape(x, x->actor, "S08") ? 3 : 2;
  int n = d_order_count(x, x->side), i, slot;
  for (i = 0; i < n; ++i) {
    slot = d_order_slot(x, x->side, i);
    if (slot == x->actor || !d_alive(x, x->side, slot)) continue;
    DEffect e = kit_eff("zaowu.ultimate_atk", "attack.panel", (double)bonus, "add");
    e.stacking = "replace"; e.source_key = "S-ultimate"; e.label = "\xe9\xa5\x95\xe9\xa4\xae\xe5\xae\xb4";
    kit_expiry(&e, x, x->side, 2, "turn_start");
    d_effect(x, kit_dr(x->side, slot), &e);
  }
  d_event(x);
}
DFN int kit_zaowu_can_play(DX* x, int current, int* replaced) {
  if (x->card >= 0 && d_card_is(x, x->card, "S06") && kit_zaowu_gulang(x) < 0) { *replaced = 1; return 0; }
  return current;
}
DFN void kit_zaowu_on_ally_summon_hurt(DX* x, const DEvt* ev) {
  if (!ev || ev->amount <= 0 || ev->target < 0) return;
  if (!d_cstr_is(x,x->side,ev->target,"name","\xe9\xac\xbc\xe9\x83\x8e\xe4\xb8\xb8")) return;
  int holder=d_holder(x,x->side,ev->target);int item=0;if(x->s->flag_str_presence[holder*FS+FS_gulang_item]==2){int slot=x->s->flag_str_slot[holder*FS+FS_gulang_item];if(d_eq(x,slot,"vinyl"))item=1;else if(d_eq(x,slot,"collar"))item=2;else if(d_eq(x,slot,"liquid"))item=3;}
  int foe = d_enemy(x->side);
  if (item == 1) {
    int handn = d_zone_count_hand(x, foe);
    if(handn>0){int card=d_zone_card_at(x,foe,0,d_random_before(x,handn));d_remove_zone_card(x,foe,0,card);d_discard(x,foe,card);d_event(x);}
  } else if (item == 2) {
    int slot = kit_random_living(x, foe);
    if(slot>=0){d_force_collapse(x,foe,slot,x->side);d_event(x);}
  } else if (item == 3) {
    int slot = kit_random_living(x, foe);
    if (slot >= 0) d_damage(x, kit_dr(foe, slot), ev->amount, DR{x->side,ev->target}, 0, 0, 1, 0, 0, 0);
  }
}
DFN void kit_zaowu_on_summon_kill(DX* x) {
  if (!kit_shape(x, x->actor, "S08") || !d_alive(x, x->side, x->actor)) return;
  int missing = d_energy_cap(x, x->side, x->actor) - d_ci(x, x->side, x->actor, "energy");
  if (missing > 0) { d_energy(x, x->side, x->actor, missing); d_event(x); }
}

DFN void kit_adler_guard(DX* x, int turns) { d_apply_dot(x, d_enemy(x->side), "guard", x->side, x->actor, 1, turns); }
DFN int kit_adler_panel_attack(DX* x, int current, int* replaced) {
  (void)current; *replaced = 1; return d_ci(x, x->side, x->actor, "shield");
}
DFN void kit_adler_on_turn_begin(DX* x) { d_shield(x, kit_self(x), d_burn_present(x) ? 3 : 2); }
DFN void kit_adler_on_ultimate(DX* x) {
  kit_adler_guard(x, 2);
  int foe = d_enemy(x->side);
  d_clear_side_shields(x, foe);
  d_event(x);
}
DFN int kit_adler_redirect_overflow(DX* x, int current, const DEvt* ev, int* replaced) {
  if (!kit_shape(x, x->actor, "D06") || (ev && ev->target == x->actor)) return current;
  if (ev && ev->absorbed > 0 && ev->loss > 0 && d_alive(x, x->side, x->actor)) { *replaced = 1; return x->actor; }
  return current;
}
DFN void kit_adler_on_burn_stacks(DX* x, const DEvt* ev) {
  if (!kit_shape(x, x->actor, "D08") || !ev || ev->amount <= 0) return;
  int slot = kit_random_living(x, ev->target_side);
  if (slot < 0) return;
  int choice = d_random_before(x, 3);
  DR t = kit_dr(ev->target_side, slot);
  if (choice == 0) {
    DEffect e = kit_eff("adler.all_resist", "damage.resistance", -1, "add");
    e.stacking = "replace"; e.clear_on = "down"; e.source_key = "D08";
    e.label = "\xe5\x85\xa8\xe5\xb1\x9e\xe6\x80\xa7\xe6\x8a\x97\xe6\x80\xa7\x20\x2d\x31";
    d_effect(x, t, &e);
  } else if (choice == 1) {
    d_set_flag(x, ev->target_side, slot, "collapse_max", 4, 0);
  } else {
    DEffect e = kit_eff("adler.attack_cut", "attack.panel", -1, "add");
    e.stacking = "replace"; e.clear_on = "down"; e.source_key = "D08";
    e.label = "\xe6\x94\xbb\xe5\x87\xbb\x20\x2d\x31";
    d_effect(x, t, &e);
  }
  d_event(x);
}

DFN void kit_lingke_sync(DX* x, const char* attribute, int support_actor, int front_only) {
  if (d_finished(x) || !d_alive(x, x->side, x->actor)) return;
  if (front_only && d_front(x, d_enemy(x->side)) < 0) return;
  d_event(x);
  if (kit_shape(x, x->actor, "K07")) {
    if (kit_streq(attribute, "light")) d_resolve_harmony(x, 1);
    else if (kit_streq(attribute, "curse")) d_resolve_harmony(x, 2);
  }
  d_additional_attack(x, attribute, kit_none(), 1, support_actor, front_only);
}
DFN void kit_lingke_on_card_used(DX* x, const DEvt* ev) {
  if (!ev || ev->actor < 0) return;
  if (ev->actor != x->actor && !d_cb(x, x->side, ev->actor, "summoned")) {
    int count = d_lingke_chain_bump(x, x->actor, ev->actor);
    if(count==3){d_event(x);kit_lingke_sync(x,d_attribute_of(x,x->side,ev->actor),ev->actor,0);}
  }
  if (ev->card < 0) return;
  int foe = d_enemy(x->side), n = d_order_count(x, foe), i, slot;
  for (i = 0; i < n; ++i) {
    slot = d_order_slot(x, foe, i);
    if (!d_has_effect_id(x, kit_dr(foe, slot), "lingke.taunt")) continue;
    if (!d_alive(x, foe, slot) || d_finished(x)) continue;
    d_clear_effect_id(x, kit_dr(foe, slot), "lingke.taunt");
    kit_lingke_sync(x, "spirit", -1, 0);
    if (!d_finished(x)) d_context_energy(x, x->side, x->actor, 1);
  }
}
DFN void kit_lingke_on_deferred_action(DX* x, const DEvt* ev) {
  if(ev&&ev->deferred_kind_slot>=0&&d_eq(x,ev->deferred_kind_slot,"lingke_sync"))kit_lingke_sync(x,d_deferred_attribute(x,ev),ev->deferred_support_actor,0);
}
DFN void kit_lingke_before_attack_damage(DX*x,const DEvt*ev){if(!ev||(!ev->sync_attack&&!(ev->support&&kit_awake(x,x->actor))))return;const char*glyph=d_attribute_text(ev->attribute);char stack[96],label[96];const char*prefix="lingke.resistance:";int n=0;while(prefix[n]){stack[n]=prefix[n];n++;}int j=0;for(;glyph[j];j++)stack[n+j]=glyph[j];stack[n+j]=0;for(j=0;glyph[j];j++)label[j]=glyph[j];const char*suffix="\xe6\x8a\x97\xe6\x80\xa7\x20\x2d\x31";for(int k=0;;k++){label[j+k]=suffix[k];if(!suffix[k])break;}DEffect e=kit_eff("lingke.resistance","damage.resistance",-1,"add");e.stacking="replace";e.stack_key=stack;e.source_key="sync";e.label=label;kit_expiry(&e,x,x->side,1,"turn_start");e.condition="damage_attribute";e.condition_key="attribute";e.condition_string=glyph;d_effect(x,DR{ev->target_side,ev->target},&e);d_event(x);}
DFN int kit_lingke_attack_is_additional(DX* x, int current, const DEvt* ev, int* replaced) {
  if (ev && ev->support && kit_awake(x, x->actor)) { *replaced = 1; return 1; }
  return current;
}
DFN int kit_lingke_additional_attack_damage_bonus(DX* x, int current, int* replaced) {
  if (kit_shape(x, x->actor, "K08")) { *replaced = 1; return current + 1; }
  return current;
}
DFN int kit_lingke_weave_damage(DX* x, int current, int* replaced) { *replaced = 1; return current + 1; }
DFN void kit_lingke_on_choice(DX* x) { d_lingke_finish_kit_choice(x); }

DFN void kit_dafutier_on_collapse_entered(DX* x, const DEvt* ev) {
  if (!ev || ev->target < 0) return;
  if (d_cb(x, ev->target_side, ev->target, "summoned")) return;
  if (ev->target_side == d_enemy(x->side) && d_ci(x, x->side, x->actor, "insight_used_turn") != d_turn(x)) {
    d_set_ci(x, x->side, x->actor, "insight_used_turn", d_turn(x));
    int card = d_make_card(x, x->side, "FF01", x->actor);
    if (!d_bad(x)) d_add_hand(x, x->side, card, 0);
  }
  if (d_template(x, x->side, x->actor) != ID_dafutier) return;
  d_draw_named_from_zones(x, x->side, "F03", 1);
}
DFN void kit_dafutier_on_dissonance(DX* x, const DEvt* ev) {
  int enemy = ev ? ev->target_side : d_enemy(x->side);
  if (enemy != d_enemy(x->side)) return;
  int layers = d_ci(x, x->side, x->actor, "dissonance_layers");
  if (layers >= 2) return;
  d_set_ci(x, x->side, x->actor, "dissonance_layers", layers + 1);
  int n = d_order_count(x, enemy), i, slot;
  for (i = 0; i < n; ++i) {
    slot = d_order_slot(x, enemy, i);
    if (!d_cb(x, enemy, slot, "summoned")) d_set_collapse_reduction(x, enemy, slot, d_entity(x, kit_self(x)), layers + 1);
  }
  d_event(x);
  for (i = 0; i < n; ++i) {
    slot = d_order_slot(x, enemy, i);
    if (!d_cb(x, enemy, slot, "summoned")) d_add_collapse(x, kit_dr(enemy, slot), 0, x->actor);
  }
}
DFN void kit_dafutier_on_card_used(DX* x, const DEvt* ev) {
  if (ev && ev->actor == x->actor && ev->card >= 0 && d_card_is(x, ev->card, "FF01"))
    d_set_pi(x, x->side, "insight_played", d_pi(x, x->side, "insight_played") + 1);
}
DFN void kit_dafutier_after_hits(DX* x, const DEvt* ev) {
  if (x->card >= 0 && d_card_is(x, x->card, "F01") && ev && ev->target >= 0
      && d_alive(x, ev->target_side, ev->target) && !d_cb(x, ev->target_side, ev->target, "summoned"))
    d_add_collapse(x, kit_dr(ev->target_side, ev->target), d_op_int(x, "collapse_bonus", 0), x->actor);
}
DFN void kit_dafutier_on_support_resolved(DX* x, const DEvt* ev) {
  if (!ev || ev->actor != x->actor || !kit_awake(x, x->actor)) return;
  d_damage(x, d_front_target(x), 2, kit_self(x), 0, 0, 1, 0, 0, 0);
  if (kit_shape(x, x->actor, "F02")) d_context_energy(x, x->side, x->actor, d_energy_cap(x, x->side, x->actor));
}
DFN void kit_weapon_F08_on_opponent_sortie_finished(DX* x, const DEvt* ev) {
  if (ev && ev->attacker_side == d_enemy(x->side) && ev->attacker >= 0
      && !d_cb(x, ev->attacker_side, ev->attacker, "summoned"))
    d_add_collapse(x, kit_dr(ev->attacker_side, ev->attacker), 3, x->actor);
}

DFN void kit_heiyu_insert_omen(DX* x) {
  if (kit_awake(x, x->actor) || !d_alive(x, x->side, x->actor) || d_finished(x)) return;
  int card = d_make_card(x, x->side, "PF01", x->actor);
  if (d_bad(x)) return;
  int n = d_deck_count(x, x->side);
  d_insert_deck(x, x->side, card, d_random_before(x, n + 1));
  d_event(x);
}
DFN void kit_heiyu_advance_owned(DX* x) { d_advance_owned_to_top(x, x->side, x->actor); }
DFN void kit_heiyu_resume_inspection(DX* x) {
  int after = d_op_int(x, "heiyu_after_divination_code", 0);
  d_del_op_int(x, "heiyu_after_divination_code");
  if (d_finished(x) || !d_alive(x, x->side, x->actor)) return;
  if (after == 1) d_draw(x, x->side, 1, -1);
  else if (after == 2) d_sortie(x, 0);
  else if (d_op_int(x, "pending_sortie", 0)) d_resume_sortie(x);
}
DFN void kit_heiyu_start_inspection(DX* x, int after, int min_bottom) {
  if (d_finished(x) || !d_alive(x, x->side, x->actor)) return;
  d_set_pi(x, x->side, "heiyu_divinations", d_pi(x, x->side, "heiyu_divinations") + 1);
  if (after) d_set_op_int(x, "heiyu_after_divination_code", after, 0);
  if (kit_shape(x, x->actor, "P08")) kit_heiyu_advance_owned(x);
  int count = kit_living_count(x, x->side);
  if (count < 2) count = 2;
  d_begin_divination(x, count, min_bottom, "heiyu.divination");
}
DFN void kit_heiyu_apply_divination(DX* x, int card, int witch) {
  if (d_finished(x) || card < 0) return;
  if(!(d_card_is(x,card,"P01")||d_card_is(x,card,"P02")||d_card_is(x,card,"P03")||d_card_is(x,card,"P04")||d_card_is(x,card,"P05")||d_card_is(x,card,"P06")||d_card_is(x,card,"P07")||d_card_is(x,card,"P08")))return;d_event(x);
  if (d_card_is(x, card, "P01")) {
    if (witch) {
      d_detach_card(x, x->side, card); d_discard(x, x->side, card, 0);
      DEffect e = kit_eff("heiyu.divination_attack", "attack.hit", 1, "add");
      char source_key[32];source_key[0]='e';d_uint_text(source_key+1,x->s->card_entity_slot[card]);e.source_key=source_key;e.source_entity=d_entity(x,kit_self(x));e.clear_on="battle";e.clear_on2="operation_end";e.clear_on3="down";e.consume_on="attack";e.show_marker=0;e.label="\xe5\x8f\x9b\xe9\x80\x83\xe8\x80\x85";
      d_effect(x, kit_self(x), &e);
      DEffect im = kit_eff("heiyu.combat_immune", "rule.combat_immune", 1, "allow");
      im.stacking="replace";im.source_key="P01";im.source_entity=d_entity(x,kit_self(x));im.clear_on="battle";im.clear_on2="operation_end";im.clear_on3="down";im.label="\xe6\x88\x98\xe6\x96\x97\xe5\x85\x8d\xe4\xbc\xa4";im.show_marker=0;
      d_effect(x, kit_self(x), &im);
    } else d_resolve_release_card(x, card);
  } else if (d_card_is(x, card, "P02")) {
    int slot = kit_random_living(x, d_enemy(x->side));
    if (slot >= 0) d_resolve_release_card_target(x, card, kit_dr(d_enemy(x->side), slot));
  } else if (d_card_is(x, card, "P03")) {
    d_draw(x, x->side, 1, -1);
  } else if (d_card_is(x, card, "P04")) {
    if (witch) { d_detach_card(x, x->side, card); d_discard(x, x->side, card, 0); d_defer(x, "heiyu_release_after_attack", x->actor, card); }
    else d_resolve_release_card(x, card);
  } else if (d_card_is(x, card, "P05")) {
    d_detach_card(x, x->side, card); d_add_hand(x, x->side, card, 1);
  } else if (d_card_is(x, card, "P06")) {
    int n = d_order_count(x, x->side), i, slot, down[8], dn = 0;
    for (i = 0; i < n && dn < 8; ++i) {
      slot = d_order_slot(x, x->side, i);
      if (!d_cb(x, x->side, slot, "summoned") && d_ci(x, x->side, slot, "down_turns") > 0) down[dn++] = slot;
    }
    if (dn) {
      slot = down[d_random_before(x, dn)];
      int before = d_ci(x, x->side, slot, "down_turns");
      d_set_ci(x, x->side, slot, "down_turns", before - 1);
      if (before - 1 == 0) { d_set_ci(x, x->side, slot, "hp", d_max_hp(x, x->side, slot)); d_event(x); }
      else d_event(x);
    }
  } else if (d_card_is(x, card, "P07")) {
    DEffect e = kit_eff("heiyu.instant_sortie", "rule.normal_attack_instant", 1, "allow");
    e.scope="allies";e.stacking="replace";e.source_entity=d_entity(x,kit_self(x));e.source_key="P07";e.clear_count=0;e.label="\xe6\x99\xae\xe9\x80\x9a\xe5\x87\xba\xe5\x87\xbb\xe5\x8f\xaf\xe6\xb6\x88\xe8\x80\x97\xe7\x9e\xac\xe5\x8f\x91\xe6\x9c\xba\xe4\xbc\x9a";kit_expiry(&e,x,x->side,0,"turn_end");
    d_effect(x, kit_player(x->side), &e);
    d_event(x);
  } else if (d_card_is(x, card, "P08")) {
    d_set_used_instant(x, x->side, 0);
    d_event(x);
  }
}
DFN void kit_heiyu_on_deck_reorder(DX* x) {
  if (d_op_int(x, "heiyu_after_placement", 0)) { d_del_op_int(x, "heiyu_after_placement"); d_sortie(x, 0); return; }
  int n = 0, i, card;
  const int* bottoms = d_choice_bottoms(x, &n);
  int witch = kit_awake(x, x->actor) && d_template(x, x->side, x->actor) == ID_heiyu;
  for (i = 0; i < n; ++i) {
    if (d_finished(x) || !d_alive(x, x->side, x->actor)) break;
    card = bottoms[i];
    kit_heiyu_apply_divination(x, card, witch);
  }
  kit_heiyu_resume_inspection(x);
}
DFN void kit_heiyu_on_choice(DX* x) {
  if (!d_op_int(x, "heiyu_hand_to_top", 0)) return;
  d_del_op_int(x, "heiyu_hand_to_top");
  int card = d_choice_selected_card(x);
  d_detach_card(x, x->side, card);
  d_insert_deck(x, x->side, card, 0);
  d_event(x);
  kit_heiyu_start_inspection(x, 2, 0);
}
DFN void kit_heiyu_on_turn_begin(DX* x) { kit_heiyu_insert_omen(x); }
DFN void kit_heiyu_after_card(DX* x, const DEvt* ev) {
  if (kit_awake(x, x->actor)) return;
  int card = ev && ev->card >= 0 ? ev->card : x->card;
  if ((ev && ev->actor != x->actor) || card < 0) return;
  if (!(d_card_is(x, card, "P01") || d_card_is(x, card, "P03") || d_card_is(x, card, "P04"))) return;
  kit_grant_harmony(x, x->actor, 2);
}
DFN int kit_heiyu_stain_duration(DX* x, int current, int* replaced) { (void)x; *replaced = 1; return current * 2; }
DFN int kit_heiyu_dark_star_renewals(DX* x, int current, int* replaced) { (void)x; *replaced = 1; return current > 1 ? current : 1; }
DFN void kit_heiyu_on_ultimate(DX* x) {
  int n = d_order_count(x, x->side), i, slot;
  for (i = 0; i < n; ++i) {
    slot = d_order_slot(x, x->side, i);
    if (!d_alive(x, x->side, slot) || d_cb(x, x->side, slot, "summoned")) continue;
    DEffect e = kit_eff("heiyu.ultimate_atk", "attack.panel", 1, "add");
    e.stacking="replace";e.source_key="P-ultimate";e.label="\xe9\xbb\x91\xe7\xbe\xbd\xe7\xbb\x88\xe7\xbb\x93";kit_expiry(&e,x,x->side,2,"turn_start");
    d_effect(x, kit_dr(x->side, slot), &e);
  }
  d_event(x);
}
DFN int kit_heiyu_can_exit_ultimate(DX* x, int current, int* replaced) {
  if (kit_awake(x, x->actor) && d_template(x, x->side, x->actor) == ID_heiyu) { *replaced = 1; return 1; }
  return current;
}
DFN void kit_heiyu_on_sortie_ready(DX* x, const DEvt* ev) {
  if (ev && ev->response) return;
  if (ev && ev->actor != x->actor) return;
  if (!d_alive(x, x->side, x->actor)) return;
  if (!kit_awake(x, x->actor)) { kit_heiyu_insert_omen(x); return; }
  kit_heiyu_start_inspection(x, 0, 0);
}
DFN void kit_heiyu_on_deferred_action(DX* x, const DEvt* ev) {
  if(ev&&d_streq(d_deferred_kind(x,ev),"heiyu_release_after_attack")&&d_alive(x,x->side,x->actor))d_resolve_release_card_ex(x,ev->card,DR{-1,-2},0);
}
DFN int kit_weapon_P07_combat_overflow(DX* x, int current, const DEvt* ev, int* replaced) {
  if (ev && ev->attacker == x->actor) { *replaced = 1; return ev->raw_overflow; }
  return current;
}
DFN int kit_heiyu_incoming_damage(DX* x, int current, const DEvt* ev, int* replaced) {
  if (ev && ev->kind == 1 && d_evaluate(x, kit_self(x), "rule.combat_immune", 0, ev) != 0) { *replaced = 1; return 0; }
  return current;
}
DFN void kit_continue_heiyu_divination(DX* x) { kit_heiyu_on_deck_reorder(x); }

DFN void kit_fire_weapons(DX* x, const char* hook, const DEvt* ev) {
  int actor = x->actor, tmpl = d_template(x, x->side, actor);int saved_source=x->effect_source_slot;kit_bind_weapon_source(x,actor);
  if (tmpl == ID_zero && kit_hook_is(hook, "on_turn_end")) {
    if (kit_weapon_equipped(x, actor, "Z08")) kit_weapon_Z08_on_turn_end(x);
    if (kit_weapon_equipped(x, actor, "Z07")) kit_weapon_Z07_on_turn_end(x);
    if (kit_weapon_equipped(x, actor, "Z06")) kit_weapon_Z06_on_turn_end(x);
  }
  if (tmpl == ID_anhunqu && kit_hook_is(hook, "on_turn_begin") && kit_weapon_equipped(x, actor, "A07"))
    kit_weapon_A07_on_turn_begin(x);
  if (tmpl == ID_dafutier && kit_hook_is(hook, "on_opponent_sortie_finished") && kit_weapon_equipped(x, actor, "F08"))
    kit_weapon_F08_on_opponent_sortie_finished(x, ev);x->effect_source_slot=saved_source;
}

DFN void kit_fire_passive_named(DX* x, int tmpl, const char* hook, const DEvt* ev) {
  if (tmpl == ID_zero && kit_hook_is(hook, "after_card")) kit_zero_after_card(x, ev);
  if (tmpl == ID_nanali && kit_hook_is(hook, "on_turn_begin")) kit_nanali_on_turn_begin(x);
  if (tmpl == ID_nanali && kit_hook_is(hook, "on_rout")) kit_nanali_on_rout(x);
  if (tmpl == ID_nanali && kit_hook_is(hook, "after_sortie")) kit_nanali_after_sortie(x);
  if (tmpl == ID_nanali && kit_hook_is(hook, "on_ultimate_end")) kit_nanali_on_ultimate_end(x);
  if (tmpl == ID_jiuyuan && kit_hook_is(hook, "on_ultimate")) kit_jiuyuan_on_ultimate(x);
  if (tmpl == ID_jiuyuan && kit_hook_is(hook, "on_genesis_damage")) kit_jiuyuan_on_genesis_damage(x, ev);
  if (tmpl == ID_iloy && kit_hook_is(hook, "before_hits")) kit_iloy_before_hits(x);
  if (tmpl == ID_iloy && kit_hook_is(hook, "on_ultimate")) kit_iloy_on_ultimate(x);
  if (tmpl == ID_iloy && kit_hook_is(hook, "on_ultimate_end")) kit_iloy_on_ultimate_end(x);
  if (tmpl == ID_iloy && kit_hook_is(hook, "on_turn_begin")) kit_iloy_on_turn_begin(x);
  if (tmpl == ID_iloy && kit_hook_is(hook, "after_genesis")) kit_iloy_after_genesis(x, ev);
  if (tmpl == ID_bohe && kit_hook_is(hook, "after_hits")) kit_bohe_after_hits(x);
  if (tmpl == ID_bohe && kit_hook_is(hook, "on_turn_begin")) kit_bohe_on_turn_begin(x);
  if (tmpl == ID_bohe && kit_hook_is(hook, "on_enemy_turn_begin")) kit_bohe_on_enemy_turn_begin(x);
  if (tmpl == ID_baicang && kit_hook_is(hook, "on_damage_taken")) kit_baicang_on_damage_taken(x, ev);
  if (tmpl == ID_baicang && kit_hook_is(hook, "on_damage_dealt")) kit_baicang_on_damage_dealt(x, ev);
  if (tmpl == ID_baicang && kit_hook_is(hook, "after_hits")) kit_baicang_after_hits(x, ev);
  if (tmpl == ID_baicang && kit_hook_is(hook, "on_turn_begin")) kit_baicang_on_turn_begin(x);
  if (tmpl == ID_baicang && kit_hook_is(hook, "on_enemy_turn_begin")) kit_baicang_on_enemy_turn_begin(x);
  if (tmpl == ID_xiaozhi && kit_hook_is(hook, "on_game_start")) kit_xiaozhi_on_game_start(x);
  if (tmpl == ID_xiaozhi && kit_hook_is(hook, "on_turn_begin")) kit_xiaozhi_on_turn_begin(x);
  if (tmpl == ID_xiaozhi && kit_hook_is(hook, "on_damage_dealt")) kit_xiaozhi_on_damage_dealt(x, ev);
  if (tmpl == ID_xiaozhi && kit_hook_is(hook, "on_ultimate")) kit_xiaozhi_on_ultimate(x);
  if (tmpl == ID_xiaozhi && kit_hook_is(hook, "on_ultimate_end")) kit_xiaozhi_on_ultimate_end(x);
  if (tmpl == ID_xiaozhi && kit_hook_is(hook, "on_hand_added")) kit_xiaozhi_on_hand_added(x, ev);
  if (tmpl == ID_xiaozhi && kit_hook_is(hook, "after_marked_card")) kit_xiaozhi_after_marked_card(x, ev);
  if (tmpl == ID_haiyue && kit_hook_is(hook, "after_sortie")) kit_haiyue_after_sortie(x);
  if (tmpl == ID_haiyue && kit_hook_is(hook, "after_operation_resources")) kit_haiyue_after_operation_resources(x);
  if (tmpl == ID_haiyue && kit_hook_is(hook, "before_attack")) kit_haiyue_before_attack(x);
  if (tmpl == ID_yi && kit_hook_is(hook, "on_ultimate")) kit_yi_on_ultimate(x);
  if (tmpl == ID_yi && kit_hook_is(hook, "on_turn_countdown")) kit_yi_on_turn_countdown(x);
  if (tmpl == ID_yi && kit_hook_is(hook, "on_turn_begin")) kit_yi_weapon_damage(x);
  if (tmpl == ID_yi && kit_hook_is(hook, "on_enemy_turn_begin")) kit_yi_weapon_damage(x);
  if (tmpl == ID_yi && kit_hook_is(hook, "after_hits")) kit_yi_after_hits(x);
  if (tmpl == ID_zhenhong && kit_hook_is(hook, "on_game_start")) kit_zhenhong_on_game_start(x);
  if (tmpl == ID_zhenhong && kit_hook_is(hook, "on_surplus")) kit_zhenhong_on_surplus(x);
  if (tmpl == ID_zhenhong && kit_hook_is(hook, "on_deferred_action")) kit_zhenhong_on_deferred_action(x, ev);
  if (tmpl == ID_zhenhong && kit_hook_is(hook, "on_ultimate")) kit_zhenhong_on_ultimate(x);
  if (tmpl == ID_zhenhong && kit_hook_is(hook, "on_effect_expired")) kit_zhenhong_on_effect_expired(x, ev);
  if (tmpl == ID_zhenhong && kit_hook_is(hook, "on_ultimate_end")) kit_zhenhong_on_ultimate_end(x);
  if (tmpl == ID_zhenhong && kit_hook_is(hook, "on_leave_front")) kit_zhenhong_on_leave_front(x);
  if (tmpl == ID_zhenhong && kit_hook_is(hook, "on_knockdown")) kit_zhenhong_on_knockdown(x);
  if (tmpl == ID_zhenhong && kit_hook_is(hook, "on_turn_end")) kit_zhenhong_on_turn_end(x);
  if (tmpl == ID_zhenhong && kit_hook_is(hook, "on_sortie_ready")) kit_zhenhong_on_sortie_ready(x, ev);
  if (tmpl == ID_anhunqu && kit_hook_is(hook, "on_ultimate")) kit_anhunqu_on_ultimate(x);
  if (tmpl == ID_anhunqu && kit_hook_is(hook, "after_hits")) kit_anhunqu_after_hits(x);
  if (tmpl == ID_anhunqu && kit_hook_is(hook, "on_dissonance")) kit_anhunqu_on_dissonance(x, ev);
  if (tmpl == ID_canhong && kit_hook_is(hook, "on_game_start")) kit_canhong_on_game_start(x);
  if (tmpl == ID_canhong && kit_hook_is(hook, "on_ultimate")) kit_canhong_on_ultimate(x);
  if (tmpl == ID_canhong && kit_hook_is(hook, "on_card_used")) kit_canhong_on_card_used(x, ev);
  if (tmpl == ID_canhong && kit_hook_is(hook, "after_hits")) kit_canhong_after_hits(x, ev);
  if (tmpl == ID_canhong && kit_hook_is(hook, "on_damage_dealt")) kit_canhong_on_damage_dealt(x, ev);
  if (tmpl == ID_canhong && kit_hook_is(hook, "on_replaced")) kit_canhong_on_replaced(x);
  if (tmpl == ID_canhong && kit_hook_is(hook, "on_ally_down")) kit_canhong_on_ally_down(x, ev);
  if (tmpl == ID_zaowu && kit_hook_is(hook, "on_turn_begin")) kit_zaowu_on_turn_begin(x);
  if (tmpl == ID_zaowu && kit_hook_is(hook, "on_ultimate")) kit_zaowu_on_ultimate(x);
  if (tmpl == ID_zaowu && kit_hook_is(hook, "on_ally_summon_hurt")) kit_zaowu_on_ally_summon_hurt(x, ev);
  if (tmpl == ID_zaowu && kit_hook_is(hook, "on_summon_kill")) kit_zaowu_on_summon_kill(x);
  if (tmpl == ID_adler && kit_hook_is(hook, "on_turn_begin")) kit_adler_on_turn_begin(x);
  if (tmpl == ID_adler && kit_hook_is(hook, "on_ultimate")) kit_adler_on_ultimate(x);
  if (tmpl == ID_adler && kit_hook_is(hook, "on_burn_stacks")) kit_adler_on_burn_stacks(x, ev);
  if (tmpl == ID_lingke && kit_hook_is(hook, "on_card_used")) kit_lingke_on_card_used(x, ev);
  if (tmpl == ID_lingke && kit_hook_is(hook, "on_deferred_action")) kit_lingke_on_deferred_action(x, ev);
  if (tmpl == ID_lingke && kit_hook_is(hook, "before_attack_damage")) kit_lingke_before_attack_damage(x, ev);
  if (tmpl == ID_lingke && kit_hook_is(hook, "on_choice")) kit_lingke_on_choice(x);
  if (tmpl == ID_dafutier && kit_hook_is(hook, "on_collapse_entered")) kit_dafutier_on_collapse_entered(x, ev);
  if (tmpl == ID_dafutier && kit_hook_is(hook, "on_dissonance")) kit_dafutier_on_dissonance(x, ev);
  if (tmpl == ID_dafutier && kit_hook_is(hook, "on_card_used")) kit_dafutier_on_card_used(x, ev);
  if (tmpl == ID_dafutier && kit_hook_is(hook, "after_hits")) kit_dafutier_after_hits(x, ev);
  if (tmpl == ID_dafutier && kit_hook_is(hook, "on_support_resolved")) kit_dafutier_on_support_resolved(x, ev);
  if (tmpl == ID_heiyu && kit_hook_is(hook, "on_turn_begin")) kit_heiyu_on_turn_begin(x);
  if (tmpl == ID_heiyu && kit_hook_is(hook, "after_card")) kit_heiyu_after_card(x, ev);
  if (tmpl == ID_heiyu && kit_hook_is(hook, "on_ultimate")) kit_heiyu_on_ultimate(x);
  if (tmpl == ID_heiyu && kit_hook_is(hook, "on_sortie_ready")) kit_heiyu_on_sortie_ready(x, ev);
  if (tmpl == ID_heiyu && kit_hook_is(hook, "on_deferred_action")) kit_heiyu_on_deferred_action(x, ev);
  if (tmpl == ID_heiyu && kit_hook_is(hook, "on_choice")) kit_heiyu_on_choice(x);
  if (tmpl == ID_heiyu && kit_hook_is(hook, "on_deck_reorder")) kit_heiyu_on_deck_reorder(x);
}

DFN void kit_fire_passive(DX* x, const char* hook, const DEvt* ev) {
  int own = d_template(x, x->side, x->actor);
  int mimic = d_mimic_template(x, x->side, x->actor);
  int skip_mimic = kit_hook_is(hook, "on_ultimate") || kit_hook_is(hook, "on_ultimate_end");
  if (mimic > 0 && !skip_mimic) {
    int awoke = kit_awake(x, x->actor);
    d_set_cb(x, x->side, x->actor, "awakened", 0);
    kit_fire_passive_named(x, mimic, hook, ev);
    d_set_cb(x, x->side, x->actor, "awakened", awoke);
  } else {
    kit_fire_passive_named(x, own, hook, ev);
  }
}

DFN int kit_decide_named(DX* x, int tmpl, const char* hook, int current, const DEvt* ev, int* replaced) {
  if (tmpl == ID_jiuyuan && kit_hook_is(hook, "genesis_extra")) { *replaced = 1; return kit_jiuyuan_genesis_extra(x); }
  if (tmpl == ID_bohe && kit_hook_is(hook, "combat_overflow")) return kit_bohe_combat_overflow(x, current, ev, replaced);
  if (tmpl == ID_bohe && kit_hook_is(hook, "incoming_damage")) return kit_bohe_incoming_damage(x, current, ev, replaced);
  if (tmpl == ID_baicang && kit_hook_is(hook, "hp_floor")) return kit_baicang_hp_floor(x, current, ev, replaced);
  if (tmpl == ID_haiyue && kit_hook_is(hook, "sortie_from_bench")) return kit_haiyue_sortie_from_bench(x, current, replaced);
  if (tmpl == ID_haiyue && kit_hook_is(hook, "dot_damage_bonus")) { *replaced = 1; return kit_haiyue_dot_damage_bonus(x, ev); }
  if (tmpl == ID_yi && kit_hook_is(hook, "attachment_damage_bonus")) { *replaced = 1; return kit_yi_attachment_damage_bonus(x); }
  if (tmpl == ID_yi && kit_hook_is(hook, "pause_delay")) return kit_yi_pause_delay(x, current, replaced);
  if (tmpl == ID_yi && kit_hook_is(hook, "can_play")) return kit_yi_can_play(x, current, replaced);
  if (tmpl == ID_zhenhong && kit_hook_is(hook, "natural_return")) return kit_zhenhong_natural_return(x, current, replaced);
  if (tmpl == ID_canhong && kit_hook_is(hook, "burn_extra_stack")) return kit_canhong_burn_extra_stack(x, current, replaced);
  if (tmpl == ID_canhong && kit_hook_is(hook, "damage_is_additional")) return kit_canhong_damage_is_additional(x, current, replaced);
  if (tmpl == ID_canhong && kit_hook_is(hook, "hp_floor")) return kit_canhong_hp_floor(x, current, ev, replaced);
  if (tmpl == ID_canhong && kit_hook_is(hook, "combat_overflow")) return kit_canhong_combat_overflow(x, current, ev, replaced);
  if (tmpl == ID_zaowu && kit_hook_is(hook, "can_play")) return kit_zaowu_can_play(x, current, replaced);
  if (tmpl == ID_adler && kit_hook_is(hook, "panel_attack")) return kit_adler_panel_attack(x, current, replaced);
  if (tmpl == ID_adler && kit_hook_is(hook, "redirect_overflow")) return kit_adler_redirect_overflow(x, current, ev, replaced);
  if (tmpl == ID_lingke && kit_hook_is(hook, "attack_is_additional")) return kit_lingke_attack_is_additional(x, current, ev, replaced);
  if (tmpl == ID_lingke && kit_hook_is(hook, "additional_attack_damage_bonus")) return kit_lingke_additional_attack_damage_bonus(x, current, replaced);
  if (tmpl == ID_lingke && kit_hook_is(hook, "weave_damage")) return kit_lingke_weave_damage(x, current, replaced);
  if (tmpl == ID_heiyu && kit_hook_is(hook, "stain_duration")) return kit_heiyu_stain_duration(x, current, replaced);
  if (tmpl == ID_heiyu && kit_hook_is(hook, "dark_star_renewals")) return kit_heiyu_dark_star_renewals(x, current, replaced);
  if (tmpl == ID_heiyu && kit_hook_is(hook, "can_exit_ultimate")) return kit_heiyu_can_exit_ultimate(x, current, replaced);
  if (tmpl == ID_heiyu && kit_hook_is(hook, "incoming_damage")) return kit_heiyu_incoming_damage(x, current, ev, replaced);
  return current;
}

DFN int kit_decide_passive(DX* x, const char* hook, int current, const DEvt* ev, int* replaced) {
  int own = d_template(x, x->side, x->actor);
  int mimic = d_mimic_template(x, x->side, x->actor);
  int skip_mimic = kit_hook_is(hook, "on_ultimate") || kit_hook_is(hook, "on_ultimate_end");
  int value = current;
  if (mimic > 0 && !skip_mimic) {
    int awoke = kit_awake(x, x->actor);
    d_set_cb(x, x->side, x->actor, "awakened", 0);
    value = kit_decide_named(x, mimic, hook, value, ev, replaced);
    d_set_cb(x, x->side, x->actor, "awakened", awoke);
  } else {
    value = kit_decide_named(x, own, hook, value, ev, replaced);
  }
  return value;
}

DFN int kit_tally_named(DX* x, int tmpl, const char* hook, const DEvt* ev) {
  if (tmpl == ID_jiuyuan && kit_hook_is(hook, "genesis_extra")) return kit_jiuyuan_genesis_extra(x);
  if (tmpl == ID_yi && kit_hook_is(hook, "attachment_damage_bonus")) return kit_yi_attachment_damage_bonus(x);
  if (tmpl == ID_haiyue && kit_hook_is(hook, "dot_damage_bonus")) return kit_haiyue_dot_damage_bonus(x, ev);
  if (tmpl == ID_xiaozhi && kit_hook_is(hook, "attack_bonus")) return kit_xiaozhi_attack_bonus(x, ev);
  return 0;
}

DFN void kit_continue_choice(DX* x, const char* name) {
  if (kit_streq(name, "heiyu.divination")) kit_continue_heiyu_divination(x);
}

DFN void kit_on_shape(DX* x, const char* hook, const DEvt* ev) {
  (void)hook; (void)ev;
  if (d_finished(x) || d_bad(x) || !d_alive(x, x->side, x->actor)) return;
}

DFN DPayment kit_decide_payment(DX* x, const char* hook, DPayment current) {
  DPayment p = current;
  if (kit_hook_is(hook, "normal_attack_payment")) {
    if (d_template(x, x->side, x->actor) == ID_xiaozhi && kit_shape(x, x->actor, "Q07")
        && d_ci(x, x->side, x->actor, "jingu") >= 3 && current.resource == 0) {
      p.ap = 0; p.consume_normal = 0; p.resource = PAY_JINGU; p.bonus = 2;
    }
    if (d_template(x, x->side, x->actor) == ID_haiyue && kit_shape(x, x->actor, "U06")
        && !d_used_instant(x, x->side) && d_teammate_attacked(x, x->side, x->actor)) {
      p.ap = 0; p.consume_normal = 0; p.resource = PAY_INSTANT; p.bonus = 0;
    }
  }
  return p;
}
DFN void kit_pay_normal_attack(DX* x, DPayment payment) {
  if (payment.resource == PAY_JINGU) kit_change_jingu(x, -3, 0);
  if (payment.resource == PAY_INSTANT) { d_set_used_instant(x, x->side, 1); d_event(x); }
}

DFN DTargetList kit_decide_targets(DX* x, const char* hook) {
  DTargetList list; list.n = 0;
  if ((kit_hook_is(hook, "battle_targets") || kit_hook_is(hook, "normal_attack_targets"))
      && d_template(x, x->side, x->actor) == ID_canhong
      && kit_shape(x, x->actor, "C08") && d_total_stacks(x, x->side) >= 5) {
    int foe = d_enemy(x->side), n = d_order_count(x, foe), i, slot;
    for (i = 0; i < n && list.n < 16; ++i) {
      slot = d_order_slot(x, foe, i);
      if (d_alive(x, foe, slot) && !d_cb(x, foe, slot, "summoned")) {
        list.items[list.n] = kit_dr(foe, slot); ++list.n;
      }
    }
  }
  return list;
}
''';
