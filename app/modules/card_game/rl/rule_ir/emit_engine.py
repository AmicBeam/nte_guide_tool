"""Emit one scalar C engine for the frozen 创生预组.

CPU and CUDA compile this source. Card identity comes from IR tables; the
generated functions are the only runtime rule implementation.
"""
from __future__ import annotations

from app.modules.card_game.rl.gpu_duel.catalog import (
    ACT_ATTACK, ACT_CHOOSE, ACT_END, ACT_MULLIGAN, ACT_PLAY, ACT_ULTIMATE, CARD_IDS, CARD_INDEX,
    HARMONY_GENESIS, ILOY, JIUYUAN, LEGAL_ATK, LEGAL_END, LEGAL_PLAY, LEGAL_ULT,
    LEGAL_WIDTH, MAX_LEGAL, NANALI, N_SEATS, PLAY_TARGETS, POLICY_ALLY,
    POLICY_DOWN_ALLY, POLICY_DOWN_ENEMY, POLICY_ALLY_FRONT, POLICY_ENEMY, POLICY_ENEMY_FRONT, POLICY_INJURED, POLICY_NONE,
    RESPONSE_ALLY, RESPONSE_SELF, RESPONSE_SELF_LETHAL, REQUIRE_NONE, REQUIRE_OWN_BURN, REQUIRE_SURPLUS, REQUIRE_HARMONY_DAMAGE,
    HARMONY_DELAY, HARMONY_OVERLAY, ZERO, BOHE, BAICANG, SEAT_INDEX, card_table,
)
from app.modules.card_game.rl.rule_ir.completeness import validate_compiled_deck, IMPLEMENTED_TABLES
from app.modules.card_game.rl.rule_ir.layout import ENGINE_NAME, c_defines
from app.modules.card_game.rl.rule_ir.lower_effects import TABLE_KEYS, compile_effect_tables
from app.modules.card_game.rl.rule_ir.starter import STARTER_HOOKS


def _c_array(name: str, values) -> str:
    body = ','.join(str(int(v)) for v in values)
    return f'static const int {name}[{len(values)}] = {{{body}}};'


def _tables(deck_id="starter") -> str:
    catalog = card_table()
    effects = compile_effect_tables(CARD_IDS)
    lines = [
        f'#define N_CARDS {len(CARD_IDS)}',
        f'#define SEAT_NANALI {NANALI}',
        f'#define SEAT_ILOY {ILOY}',
        f'#define SEAT_ZERO {ZERO}',
        f'#define SEAT_JIUYUAN {JIUYUAN}',
        f'#define CARD_NF01 {CARD_INDEX["NF01"]}',
        f'#define CARD_N02 {CARD_INDEX["N02"]}',
        f'#define CARD_N03 {CARD_INDEX["N03"]}',
        f'#define CARD_N07 {CARD_INDEX["N07"]}',
        f'#define CARD_Z07 {CARD_INDEX["Z07"]}',
        f'#define CARD_Z08 {CARD_INDEX["Z08"]}',
        f'#define CARD_Y08 {CARD_INDEX["Y08"]}',
        f'#define CARD_J08 {CARD_INDEX["J08"]}',
        f'#define CARD_J07 {CARD_INDEX["J07"]}',
        f'#define CARD_M07 {CARD_INDEX["M07"]}',
        f'#define CARD_Y07 {CARD_INDEX.get("Y07", -1)}',
        f'#define CARD_Z06 {CARD_INDEX.get("Z06", -1)}',
        f'#define ACT_END {ACT_END}',
        f'#define ACT_ATTACK {ACT_ATTACK}',
        f'#define ACT_PLAY {ACT_PLAY}',
        f'#define ACT_ULTIMATE {ACT_ULTIMATE}',
        f'#define ACT_MULLIGAN {ACT_MULLIGAN}',
        f'#define ACT_CHOOSE {ACT_CHOOSE}',
        f'#define LEGAL_END {LEGAL_END}',
        f'#define LEGAL_ATK {LEGAL_ATK}',
        f'#define LEGAL_ULT {LEGAL_ULT}',
        f'#define LEGAL_PLAY {LEGAL_PLAY}',
        f'#define LEGAL_WIDTH {LEGAL_WIDTH}',
        f'#define PLAY_TARGETS {PLAY_TARGETS}',
        f'#define POLICY_NONE {POLICY_NONE}',
        f'#define POLICY_ALLY {POLICY_ALLY}',
        f'#define POLICY_INJURED {POLICY_INJURED}',
        f'#define POLICY_DOWN_ALLY {POLICY_DOWN_ALLY}',
        f'#define POLICY_ENEMY {POLICY_ENEMY}',
        f'#define POLICY_ENEMY_FRONT {POLICY_ENEMY_FRONT}',
        f'#define POLICY_DOWN_ENEMY {POLICY_DOWN_ENEMY}',
        f'#define POLICY_ALLY_FRONT {POLICY_ALLY_FRONT}',
        f'#define RESPONSE_ALLY {RESPONSE_ALLY}',
        f'#define RESPONSE_SELF {RESPONSE_SELF}',
        f'#define RESPONSE_SELF_LETHAL {RESPONSE_SELF_LETHAL}',
        f'#define REQUIRE_NONE {REQUIRE_NONE}',
        f'#define REQUIRE_OWN_BURN {REQUIRE_OWN_BURN}',
        f'#define REQUIRE_SURPLUS {REQUIRE_SURPLUS}',
        f'#define REQUIRE_HARMONY_DAMAGE {REQUIRE_HARMONY_DAMAGE}',
        f'#define HARMONY_GENESIS {HARMONY_GENESIS}',
        f'#define HARMONY_DELAY {HARMONY_DELAY}',
        f'#define HARMONY_OVERLAY {HARMONY_OVERLAY}',
        f'#define HARMONY_BURN 4',
        f'#define HARMONY_STAR 5',
        f'#define PHASE_MULLIGAN 0',
        f'#define PHASE_PLAYING 1',
        f'#define PHASE_CHOICE 2',
        f'#define PHASE_FINISHED 3',
        f'#define ERR_UNSUPPORTED 1',
        f'#define TURN_END_ATK {int(STARTER_HOOKS["turn_end_atk"])}',
        f'#define TURN_END_HEAL {int(STARTER_HOOKS["turn_end_heal"])}',
        f'#define FOLLOWUP_SHAPE {CARD_INDEX.get(STARTER_HOOKS["payoff_card"], -1)}',
        f'#define FOLLOWUP_FORM {CARD_INDEX.get("N08", -1)}',
        f'#define SCALE_SHAPE {CARD_INDEX.get(STARTER_HOOKS["scale_shape"], -1)}',
        f'#define PAYOFF_CARD {CARD_INDEX.get(STARTER_HOOKS["payoff_card"], -1)}',
    ]
    from .completeness import _deck
    deck = _deck(deck_id)
    deck_kinds = [CARD_INDEX[card_id] for card_id in deck['card_ids']]
    lines.append(_c_array('DECK_SEATS', [SEAT_INDEX[cid] for cid in deck['character_ids']]))
    lines.extend([f'#define SEAT_BOHE {BOHE}', f'#define SEAT_BAICANG {BAICANG}', f'#define CARD_M08 {CARD_INDEX["M08"]}'])
    lines.append(_c_array('STARTER_DECK_KINDS', deck_kinds))
    lines.append(f'#define STARTER_DECK_N {len(deck_kinds)}')
    for key, values in catalog.items():
        lines.append(_c_array(f'CAT_{key.upper()}', values))
    for key in TABLE_KEYS:
        lines.append(_c_array(f'TAB_{key.upper()}', effects[key]))
    unimplemented = [key for key in TABLE_KEYS if key not in IMPLEMENTED_TABLES]
    checks = ' || '.join(f'TAB_{key.upper()}[k]' for key in unimplemented) or '0'
    lines.append(f'static int unsupported_effect(int k){{ return k<0?1:({checks}); }}')
    return '\n'.join(lines)


def emit_engine_source(deck_id="starter") -> str:
    validate_compiled_deck(deck_id)
    return '\n'.join([
        '#include <stdint.h>',
        c_defines(),
        _tables(deck_id),
        ENGINE_BODY,
    ])


ENGINE_BODY = r'''
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))
#define SI(name,s) row[OFF_##name + (s)]
#define CH(name,s,k) row[OFF_##name + ((s)*N_SEATS + (k))]
#define HK(s,i) row[OFF_HAND_KIND + ((s)*HAND_SLOTS + (i))]
#define HI(s,i) row[OFF_HAND_INST + ((s)*HAND_SLOTS + (i))]
#define DK(s,i) row[OFF_DECK_KIND + ((s)*DECK_SLOTS + (i))]
#define DI(s,i) row[OFF_DECK_INST + ((s)*DECK_SLOTS + (i))]
#define XK(s,i) row[OFF_DISCARD_KIND + ((s)*DISCARD_SLOTS + (i))]
#define XI(s,i) row[OFF_DISCARD_INST + ((s)*DISCARD_SLOTS + (i))]
#define PC(i) row[OFF_PENDING_CARD + (i)]
#define PI(i) row[OFF_PENDING_INST + (i)]
#define LT(i) row[OFF_LEGAL_TYPE + (i)]
#define LA(i) row[OFF_LEGAL_ACTOR + (i)]
#define LH(i) row[OFF_LEGAL_HAND + (i)]
#define LS(i) row[OFF_LEGAL_TSIDE + (i)]
#define LK(i) row[OFF_LEGAL_TSEAT + (i)]

static int live(int *row){ return row[OFF_PHASE] != PHASE_FINISHED && row[OFF_ERROR]==0; }
static int lcg(int *rng){
  int x = (*rng * 1103515245 + 12345) & 0x7FFFFFFF;
  *rng = x;
  return x;
}
static int other_side(int s){ return 1-s; }
static int present_alive(int *row, int s, int k){
  return CH(CH_PRESENT,s,k)>0 && CH(CH_HP,s,k)>0 && CH(CH_DOWN,s,k)==0;
}
static int collapsed(int *row, int s, int k){
  return CH(CH_COLLAPSE_UNTIL,s,k)>0;
}
static void fail(int *row, int code){ if(row[OFF_ERROR]==0) row[OFF_ERROR]=code; }

static int attack_power(int *row, int s, int k){
  int form = CH(CH_SHAPE,s,k);
  int perm = CH(CH_BASE_ATK,s,k) - CAT_BASE_ATK[k];
  if(perm<0) perm=0;
  int value;
  if(form>=0 && CAT_HP[form]>0 && CAT_ATK[form]>0) value = CAT_ATK[form] + perm;
  else value = CH(CH_BASE_ATK,s,k) + CH(CH_GROWTH,s,k);
  value += CH(CH_ATK_BUFF,s,k);
  if(value<0) value=0;
  if(form==CARD_M07 && row[OFF_ACTIVE]!=s) value=0;
  return value;
}

static int apply_hp(int *row, int s, int is_player, int seat, int amount){
  if(!live(row) || amount<=0) return 0;
  int *hp = is_player ? &SI(HP,s) : &CH(CH_HP,s,seat);
  int *sh = is_player ? &SI(SHIELD,s) : &CH(CH_SHIELD,s,seat);
  if(!is_player && *hp<=0) return 0;
  int absorb = MIN(*sh, amount);
  *sh -= absorb;
  int loss = amount - absorb;
  int floor = (!is_player && CH(CH_HP_FLOOR,s,seat)>0) ? 1 : 0;
  int nhp = *hp - loss;
  if(nhp<floor) nhp=floor;
  *hp = nhp;
  return loss;
}
static int character_entity_key(int side, int seat){ return side*N_SEATS+seat; }
/* Source-aware damage: character-local hooks, no implicit settlement. */
static void trigger_response(int *row, int atk_side, int atk_seat);
static void trigger_self_lethal(int *row, int ts, int tk);
static int deal_damage(int *row, int ts, int player, int tk, int amount, int ss, int sk, int combat, int bypass){
  if(!live(row) || amount<=0) return 0;
  if(!player && tk==SEAT_BOHE && CH(CH_AWAKENED,ts,tk)>0 && ss!=ts && !combat) return 0;
  int saved=0;
  int *shield_ptr=player? &SI(SHIELD,ts) : &CH(CH_SHIELD,ts,tk);
  if(bypass){ saved=*shield_ptr; *shield_ptr=0; }
  int *hp = player ? &SI(HP,ts) : &CH(CH_HP,ts,tk);
  int *sh = shield_ptr;
  int projected = *hp - (amount - MIN(*sh, amount));
  if(!player && projected<=0) trigger_self_lethal(row, ts, tk);
  int loss=apply_hp(row,ts,player,tk,amount);
  if(bypass) *shield_ptr=saved;
  if(!player){
    CH(LAST_HIT_SIDE,ts,tk)=ss; CH(LAST_HIT_SEAT,ts,tk)=sk;
    if(loss>0 && tk==SEAT_BAICANG && CH(CH_HP,ts,tk)>0){
      CH(CH_ATK_BUFF,ts,tk)+=1;
      if(ss==ts) CH(CH_ALLIED_HURT,ts,tk)=1;
    }
    if(loss>0 && ts!=ss && sk==SEAT_BAICANG && character_entity_key(ts,tk)!=character_entity_key(ss,sk) && present_alive(row,ss,sk)
       && CH(CH_AWAKENED,ss,sk)>0 && CH(CH_HP,ts,tk)>0 && CH(CH_HP,ts,tk)<=2) CH(CH_HP,ts,tk)=0;
  }
  return loss;
}

static void grant_shield(int *row, int s, int seat, int amount){
  if(!live(row) || amount<=0 || !present_alive(row,s,seat)) return;
  if(CH(CH_SHIELD,s,seat)<amount) CH(CH_SHIELD,s,seat)=amount;
}

static void set_last_hit(int *row, int s, int seat, int src_side, int src_seat){
  CH(LAST_HIT_SIDE,s,seat) = src_side;
  CH(LAST_HIT_SEAT,s,seat) = src_seat;
}

static void victory(int *row){
  if(row[OFF_PHASE]==PHASE_FINISHED) return;
  int da = SI(HP,0)<=0, db = SI(HP,1)<=0;
  if(!da && !db) return;
  row[OFF_PHASE]=PHASE_FINISHED;
  row[OFF_REASON]=1;
  row[OFF_WINNER] = (da && db) ? 2 : (da ? 1 : 0);
}

static void add_hand(int *row, int s, int kind, int inst);
static void to_discard(int *row, int s, int kind, int inst);
static void grant_seed(int *row, int s);
static void apply_mulligan(int *row, int s, int mask);
static void resolve_genesis(int *row, int s, int seat, int arm_delay);
static void deal_front(int *row, int s, int amount, int src_seat);
static void play_effects(int *row, int s, int seat, int kind, int tside, int tseat);
static int card_free(int *row, int s, int kind);

static void compact_hand(int *row, int s){
  int n=0;
  for(int i=0;i<HAND_SLOTS;i++){
    if(HK(s,i)>=0){
      if(n!=i){
        HK(s,n)=HK(s,i); HI(s,n)=HI(s,i);
        row[OFF_HAND_REVEALED + (s*HAND_SLOTS + n)] = row[OFF_HAND_REVEALED + (s*HAND_SLOTS + i)];
      }
      n++;
    }
  }
  for(int i=n;i<HAND_SLOTS;i++){
    HK(s,i)=-1; HI(s,i)=0;
    row[OFF_HAND_REVEALED + (s*HAND_SLOTS + i)]=0;
  }
  SI(HAND_N,s)=n;
}
static void compact_deck(int *row, int s){
  int n=0;
  for(int i=0;i<DECK_SLOTS;i++){
    if(DK(s,i)>=0){
      if(n!=i){ DK(s,n)=DK(s,i); DI(s,n)=DI(s,i); }
      n++;
    }
  }
  for(int i=n;i<DECK_SLOTS;i++){ DK(s,i)=-1; DI(s,i)=0; }
  SI(DECK_N,s)=n;
}

static void add_hand(int *row, int s, int kind, int inst){
  if(!live(row) || kind<0) return;
  int n=SI(HAND_N,s);
  if(n>=HAND_SLOTS){ to_discard(row,s,kind,inst); return; }
  HK(s,n)=kind; HI(s,n)=inst;
  row[OFF_HAND_REVEALED + (s*HAND_SLOTS + n)]=0;
  SI(HAND_N,s)=n+1;
}
static void to_discard(int *row, int s, int kind, int inst){
  if(kind<0) return;
  int n=SI(DISCARD_N,s);
  if(n>=DISCARD_SLOTS) return;
  XK(s,n)=kind; XI(s,n)=inst; SI(DISCARD_N,s)=n+1;
}
static int remove_hand_slot(int *row, int s, int slot, int *out_inst){
  if(slot<0 || slot>=HAND_SLOTS) return -1;
  int kind=HK(s,slot);
  if(out_inst) *out_inst = HI(s,slot);
  HK(s,slot)=-1; HI(s,slot)=0;
  row[OFF_HAND_REVEALED + (s*HAND_SLOTS + slot)]=0;
  compact_hand(row,s);
  return kind;
}

static void draw_one(int *row, int s){
  if(!live(row)) return;
  if(SI(DECK_N,s)<=0){
    row[OFF_PHASE]=PHASE_FINISHED;
    row[OFF_WINNER]=other_side(s);
    row[OFF_REASON]=2;
    return;
  }
  int kind=DK(s,0), inst=DI(s,0);
  DK(s,0)=-1; compact_deck(row,s);
  add_hand(row,s,kind,inst);
}
static void draw_n(int *row, int s, int n){
  for(int i=0;i<n && live(row);i++) draw_one(row,s);
}

static void grant_seed(int *row, int s){
  if(SI(NANALI_SEED,s)>=3) return;
  if(!(CH(CH_PRESENT,s,SEAT_NANALI)>0 && CH(CH_HP,s,SEAT_NANALI)>0 && CH(CH_DOWN,s,SEAT_NANALI)==0)) return;
  SI(NANALI_SEED,s) += 1;
  row[OFF_NEXT_INSTANCE] += 1;
  add_hand(row,s,CARD_NF01,row[OFF_NEXT_INSTANCE]);
}

static void move_out(int *row, int s, int seat){
  if(SI(FRONT,s)!=seat) return;
  int hcap=(row[OFF_ESCALATION_ENABLED] && row[OFF_TURN]>=20) ? 1 : 2;
  SI(LAST_FRONT,s)=(CH(CH_HP,s,seat)>0 && CH(CH_DOWN,s,seat)==0 && CH(CH_HARMONY,s,seat)>=hcap) ? seat : -1;
  SI(FRONT,s)=-1;
  CH(CH_SHIELD,s,seat)=0;
}

static void knockdown(int *row){
  if(!live(row)) return;
  int progressed=1;
  while(progressed){
    progressed=0;
    int downed[N_SIDES][N_SEATS]={0};
    for(int s=0;s<N_SIDES;s++){
      for(int k=0;k<N_SEATS;k++){
        if(!(CH(CH_PRESENT,s,k)>0 && CH(CH_HP,s,k)<=0 && CH(CH_DOWN,s,k)==0)) continue;
        progressed=1;
        int killer_side=CH(LAST_HIT_SIDE,s,k);
        int killer_seat=CH(LAST_HIT_SEAT,s,k);
        if(SI(LAST_FRONT,s)==k) SI(LAST_FRONT,s)=-1;
        if(SI(FRONT,s)==k) move_out(row,s,k);
        downed[s][k]=1;
        CH(CH_DOWN,s,k)=3;
        CH(CH_SHIELD,s,k)=0;
        CH(CH_GROWTH,s,k)=0;
        CH(CH_SHAPE,s,k)=-1;
        CH(CH_AWAKENED,s,k)=0;
        CH(CH_ULT_TURNS,s,k)=0;
        CH(CH_ATK_BUFF,s,k)=0;
        CH(CH_NEXT_BONUS,s,k)=0;
        CH(CH_NEXT_SHIELD,s,k)=0;
        CH(CH_NEXT_FOLLOWUP,s,k)=0;
        CH(CH_PACT,s,k)=0;
        CH(CH_ENERGY_NEXT,s,k)=0;
        CH(CH_ALLIED_HURT,s,k)=0;
        CH(CH_PENDING_ATK,s,k)=0;
        CH(CH_SHARE_OVERFLOW,s,k)=0;
        CH(CH_HP_FLOOR,s,k)=0;
        CH(CH_COLLAPSE_COUNT,s,k)=0;
        CH(CH_COLLAPSE_BY,s,k)=-1;
        CH(CH_COLLAPSE_UNTIL,s,k)=0;
        CH(CH_MAX_HP,s,k)=CH(CH_BASE_MAX,s,k);
        if(killer_side==other_side(s) && killer_seat==SEAT_NANALI && present_alive(row,killer_side,SEAT_NANALI)
           && SI(FRONT,killer_side)==SEAT_NANALI){
          grant_seed(row, killer_side);
        }
      }
    }
    /* All simultaneous fronts have left before ally-down effects retarget. */
    for(int s=0;s<N_SIDES;s++){
      for(int k=0;k<N_SEATS;k++){
        if(!downed[s][k] || k==SEAT_ILOY || !present_alive(row,s,SEAT_ILOY)) continue;
        /* Iloy Y07: another ally has finished being knocked down. */
        if(CARD_Y07>=0 && CH(CH_SHAPE,s,SEAT_ILOY)==CARD_Y07){
          int foe=other_side(s);
          int amt=2;
          if(SI(FRONT,foe)>=0){
            int t=SI(FRONT,foe);
            deal_damage(row,foe,0,t,amt,s,SEAT_ILOY,0,0);
            set_last_hit(row,foe,t,s,SEAT_ILOY);
          } else {
            deal_damage(row,foe,1,0,amt,s,SEAT_ILOY,0,0);
          }
        }
      }
    }
  }
  victory(row);
}

static int energy_cap_for(int *row, int seat){
  int cap = CAT_ENERGY_MAX[seat];
  if(row[OFF_ESCALATION_ENABLED] && row[OFF_TURN] >= 13) cap -= 1;
  if(cap<1) cap=1;
  return cap;
}
static int harmony_cap_for(int *row){
  return (row[OFF_ESCALATION_ENABLED] && row[OFF_TURN] >= 20) ? 1 : 2;
}
static int ultimate_limit_for(int *row){
  return (row[OFF_ESCALATION_ENABLED] && row[OFF_TURN] >= 6) ? 2 : 1;
}
static int team_rank(int *row, int s, int k){
  int rank=CH(CH_ORDER,s,k);
  if(rank>0) return rank;
  if(CH(CH_PRESENT,s,k)==0) return 1000 + k;
  return k+1;
}
static int harmony_payer(int *row, int s, int seat){
  if(SI(FRONT,s)==seat) return -1;
  int prev = SI(FRONT,s)>=0 ? SI(FRONT,s) : SI(LAST_FRONT,s);
  if(prev<0 || prev==seat) return -1;
  if(!(CH(CH_HP,s,prev)>0 && CH(CH_HARMONY,s,prev)>=harmony_cap_for(row))) return -1;
  return prev;
}
static int harmony_kind(int *row, int s, int seat){
  int payer=harmony_payer(row,s,seat);
  if(payer<0) return 0;
  int a=CAT_ATTR[seat], b=CAT_ATTR[payer];
  if((a==0&&b==1)||(a==1&&b==0)) return HARMONY_GENESIS;
  if((a==0&&b==2)||(a==2&&b==0)) return HARMONY_DELAY;
  if((a==1&&b==5)||(a==5&&b==1)) return HARMONY_OVERLAY;
  if((a==3&&b==5)||(a==5&&b==3)) return HARMONY_BURN;
  if((a==3&&b==4)||(a==4&&b==3)) return HARMONY_STAR;
  return 0;
}

static void deal_front(int *row, int s, int amount, int src_seat){
  if(!live(row) || amount<=0){ knockdown(row); victory(row); return; }
  int foe=other_side(s);
  if(SI(FRONT,foe)>=0){
    int t=SI(FRONT,foe);
    deal_damage(row,foe,0,t,amount,s,src_seat,0,0);
    set_last_hit(row,foe,t,s,src_seat);
  } else {
    deal_damage(row,foe,1,0,amount,s,src_seat,0,0);
  }
  knockdown(row);
  victory(row);
}

static void resolve_genesis(int *row, int s, int seat, int arm_delay){
  int extra = (CH(CH_HP,s,SEAT_JIUYUAN)>0) ? 1 : 0;
  int ticks = 1 + extra;
  for(int i=0;i<ticks && live(row) && present_alive(row,s,seat);i++){
    int original_target=SI(FRONT,other_side(s));
    SI(HARMONY_DAMAGE,s)=1;
    deal_front(row,s,1,seat);
    if(!live(row) || !present_alive(row,s,seat)) break;
    if(CH(CH_SHAPE,s,SEAT_JIUYUAN)==CARD_J08 && CH(CH_HP,s,SEAT_JIUYUAN)>0){
      int foe=other_side(s);
      int t=original_target;
      if(t<0 || (SI(FRONT,foe)==t && present_alive(row,foe,t))){
        deal_front(row,s,1,SEAT_JIUYUAN);
        if(t>=0 && CH(CH_HP,foe,t)>0) CH(CH_PACT,foe,t)=1;
      }
    }
  }
  if(arm_delay && live(row) && CH(CH_SHAPE,s,SEAT_ILOY)==CARD_Y08 && CH(CH_HP,s,SEAT_ILOY)>0 && SI(EXTRA_GENESIS,s)==0){
    SI(EXTRA_GENESIS,s)=1;
    SI(EXTRA_GENESIS_ACTOR,s)=seat;
  }
  int flowers=SI(EXTRA_FLOWER,s);
  SI(EXTRA_FLOWER,s)=0;
  for(int i=0;i<flowers && live(row) && present_alive(row,s,seat);i++) deal_front(row,s,1,seat);
}

static void enter(int *row, int s, int seat, int as_support){
  (void)as_support;
  if(!live(row) || !present_alive(row,s,seat)) return;
  if(SI(FRONT,s)==seat) return;
  int kind=harmony_kind(row,s,seat);
  int payer=harmony_payer(row,s,seat);
  if(SI(FRONT,s)>=0) move_out(row,s,SI(FRONT,s));
  SI(LAST_FRONT,s)=-1;
  SI(FRONT,s)=seat;
  if(kind && payer>=0){
    CH(CH_HARMONY,s,payer)-=harmony_cap_for(row);
    CH(CH_HARMONIZED,s,seat)=1;
    int foe=other_side(s);
    if(kind==HARMONY_GENESIS) resolve_genesis(row,s,seat,1);
    else if(kind==HARMONY_OVERLAY) SI(WEAVE,foe)=1;
    else if(kind==HARMONY_DELAY) SI(DELAY_END,foe)=1;
    else if(kind==HARMONY_BURN){
      SI(BURN_LEFT,foe)=2;
      SI(BURN_BY,foe)=s;
      SI(BURN_INFINITE,foe)=0;
    } else if(kind==HARMONY_STAR){
      /* public kits do not currently consume 黯星 beyond applying it as a zone flag. */
      SI(DELAY_END,foe)=SI(DELAY_END,foe);
    }
  }
}

static int form_draw_bonus(int *row, int s, int kind){
  if(kind<0) return 0;
  if(CAT_TYPE[kind]==2){
    if(CH(CH_HP,s,SEAT_ZERO)>0 && CH(CH_AWAKENED,s,SEAT_ZERO)>0) return 1;
    int owner=CAT_OWNER[kind];
    if(owner==SEAT_ZERO && CH(CH_HP,s,SEAT_ZERO)>0 && CH(CH_SHAPE,s,SEAT_ZERO)>=0) return 1;
  }
  return 0;
}
static int card_instant(int *row, int s, int kind){
  if(kind<0) return 0;
  return CAT_INSTANT[kind] || (kind==CARD_NF01 && row[OFF_TURN]>=5) || form_draw_bonus(row,s,kind);
}
static int card_free(int *row, int s, int kind){
  return card_instant(row,s,kind) && SI(USED_INSTANT,s)==0;
}

static void trigger_self_lethal(int *row, int ts, int tk){
  if(!live(row) || tk<0) return;
  for(int hs=0; hs<HAND_SLOTS; hs++){
    int kind=HK(ts,hs);
    if(kind<0) continue;
    if(CAT_RESPONSE[kind]!=RESPONSE_SELF_LETHAL) continue;
    int owner=CAT_OWNER[kind];
    if(owner!=tk || !present_alive(row,ts,owner) || collapsed(row,ts,owner)) continue;
    int cost=CAT_COST[kind];
    int afford = card_free(row,ts,kind) || SI(AP,ts)>=cost;
    if(!afford) continue;
    int inst=0;
    remove_hand_slot(row,ts,hs,&inst);
    if(card_free(row,ts,kind)) SI(USED_INSTANT,ts)=1;
    else SI(AP,ts)-=cost;
    play_effects(row,ts,owner,kind,-1,-1);
    to_discard(row,ts,kind,inst);
    return;
  }
}

static void trigger_response(int *row, int atk_side, int atk_seat){
  (void)atk_seat;
  if(!live(row)) return;
  int foe=other_side(atk_side);
  int target=SI(FRONT,foe);
  if(target<0) return;
  for(int hs=0; hs<HAND_SLOTS; hs++){
    int kind=HK(foe,hs);
    if(kind<0) continue;
    int resp=CAT_RESPONSE[kind];
    int owner=CAT_OWNER[kind];
    if(resp==RESPONSE_ALLY && owner==target) continue;
    if(resp==RESPONSE_SELF && owner!=target) continue;
    if(resp==0) continue;
    if(!present_alive(row,foe,owner) || collapsed(row,foe,owner)) continue;
    int cost=CAT_COST[kind];
    int afford = card_free(row,foe,kind) || SI(AP,foe)>=cost;
    if(!afford) continue;
    int inst=0;
    remove_hand_slot(row,foe,hs,&inst);
    if(card_free(row,foe,kind)) SI(USED_INSTANT,foe)=1;
    else SI(AP,foe)-=cost;
    int sh=CAT_SHIELD[kind];
    int bonus=CAT_ATK[kind];
    if(CAT_TYPE[kind]==0){
      if(CH(CH_SHAPE,foe,owner)==SCALE_SHAPE){
        bonus += SI(NANALI_SEED_PLAYED,foe);
        sh += SI(NANALI_SEED_PLAYED,foe);
      }
      grant_shield(row,foe,owner,sh);
    }
    if(TAB_SHARE_OVERFLOW[kind]) CH(CH_SHARE_OVERFLOW,foe,owner)=1;
    enter(row,foe,owner,0);
    if(bonus>0) row[OFF_ACTION]+=bonus;
    to_discard(row,foe,kind,inst);
    return;
  }
}

static void attack_once(int *row, int s, int seat, int bonus, int support, int ignore_shield, int heal_after){
  if(!live(row) || !present_alive(row,s,seat) || SI(FRONT,s)!=seat) return;
  row[OFF_ACTION]=0;
  trigger_response(row,s,seat);
  if(!live(row) || !present_alive(row,s,seat) || SI(FRONT,s)!=seat) return;
  int atk=MAX(0,attack_power(row,s,seat)+bonus);
  int foe=other_side(s), t=SI(FRONT,foe), counter=0, target_loss=0;
  int weave_atk=SI(WEAVE,foe)>0 && (CAT_ATTR[seat]==1 || CAT_ATTR[seat]==5);
  int weave_counter=0;
  if(t>=0){
    counter=(!support && !collapsed(row,foe,t)) ? MAX(0,attack_power(row,foe,t)+row[OFF_ACTION]) : 0;
    weave_counter=!support && !collapsed(row,foe,t) && SI(WEAVE,s)>0 && (CAT_ATTR[t]==1 || CAT_ATTR[t]==5);
    int raw=MAX(0,atk-(ignore_shield?0:CH(CH_SHIELD,foe,t))-CH(CH_HP,foe,t));
    int overflow=(seat==SEAT_BOHE || (t==SEAT_BOHE && CH(CH_SHARE_OVERFLOW,foe,t)))?raw:0;
    if(t==SEAT_BOHE && seat!=SEAT_BOHE) CH(CH_SHARE_OVERFLOW,foe,t)=0;
    target_loss=deal_damage(row,foe,0,t,atk,s,seat,1,ignore_shield);
    deal_damage(row,s,0,seat,counter,foe,t,1,0);
    if(overflow) deal_damage(row,foe,1,0,overflow,s,seat,1,0);
    if(support && present_alive(row,foe,t)){
      CH(CH_COLLAPSE_COUNT,foe,t)+=1;
      if(CH(CH_COLLAPSE_COUNT,foe,t)>=5 && CH(CH_COLLAPSE_UNTIL,foe,t)==0){
        CH(CH_COLLAPSE_COUNT,foe,t)=0;
        CH(CH_COLLAPSE_BY,foe,t)=s;
        CH(CH_COLLAPSE_UNTIL,foe,t)=SI(TURN_COUNT,s)+1;
      }
    }
  } else target_loss=deal_damage(row,foe,1,0,atk,s,seat,1,ignore_shield);
  row[OFF_ACTION]=0;
  if(heal_after>0 && target_loss>0 && CH(CH_DOWN,s,seat)==0)
    CH(CH_HP,s,seat)=MIN(CH(CH_MAX_HP,s,seat),CH(CH_HP,s,seat)+heal_after);
  if(seat==SEAT_BOHE) CH(CH_SHARE_OVERFLOW,s,seat)=0;
  knockdown(row); victory(row);
  if(!live(row)) return;
  if(weave_atk && present_alive(row,s,seat) && (t<0 || (SI(FRONT,foe)==t && present_alive(row,foe,t)))){
    SI(HARMONY_DAMAGE,s)=1;
    deal_damage(row,foe,t<0,MAX(0,t),2,s,seat,0,0);
    knockdown(row); victory(row);
  }
  if(live(row) && weave_counter && present_alive(row,foe,t) && SI(FRONT,s)==seat && present_alive(row,s,seat)){
    SI(HARMONY_DAMAGE,foe)=1;
    deal_damage(row,s,0,seat,2,foe,t,0,0);
    knockdown(row); victory(row);
  }
}

static void grant_combat_resources(int *row, int s, int seat){
  if(!live(row)) return;
  for(int k=0;k<N_SEATS;k++){
    if(CH(CH_HP,s,k)>0 && CH(CH_DOWN,s,k)==0){
      int e=CH(CH_ENERGY,s,k)+1;
      int cap=energy_cap_for(row,k);
      if(e>cap) e=cap;
      CH(CH_ENERGY,s,k)=e;
    }
  }
  if(present_alive(row,s,seat)){
    int e=CH(CH_ENERGY,s,seat)+1;
    int cap=energy_cap_for(row,seat);
    if(e>cap) e=cap;
    CH(CH_ENERGY,s,seat)=e;
    int h=CH(CH_HARMONY,s,seat)+1;
    int hcap=harmony_cap_for(row);
    if(h>hcap) h=hcap;
    CH(CH_HARMONY,s,seat)=h;
  }
}

static void sortie(int *row, int s, int seat, int card_bonus, int followup, int intercept, int ignore_shield, int heal_after){
  if(!live(row) || !present_alive(row,s,seat)) return;
  if(intercept){ enter(row,s,seat,0); return; }
  int support=harmony_kind(row,s,seat)!=0;
  enter(row,s,seat,1);
  int bonus=CH(CH_NEXT_BONUS,s,seat)+card_bonus;
  int extra_sh=CH(CH_NEXT_SHIELD,s,seat);
  CH(CH_NEXT_BONUS,s,seat)=0;
  CH(CH_NEXT_SHIELD,s,seat)=0;
  if(extra_sh) grant_shield(row,s,seat,extra_sh);
  attack_once(row,s,seat,bonus,support,ignore_shield,heal_after);
  int fu=followup;
  if(seat==SEAT_NANALI && CH(CH_AWAKENED,s,seat)>0) fu+=1;
  fu += CH(CH_NEXT_FOLLOWUP,s,seat);
  CH(CH_NEXT_FOLLOWUP,s,seat)=0;
  if(seat==SEAT_NANALI && CH(CH_SHAPE,s,SEAT_NANALI)==FOLLOWUP_FORM) fu+=2;
  if(live(row) && present_alive(row,s,seat) && SI(FRONT,s)==seat && fu>0){
    int foe=other_side(s), target=SI(FRONT,foe);
    /* Followup is ability damage and is not blocked by Bohe's non-battle immunity. */
    deal_damage(row,foe,target<0,MAX(0,target),fu,s,seat,1,0);
    knockdown(row); victory(row);
  }
}

static void play_effects(int *row, int s, int seat, int kind, int tside, int tseat){
  if(kind<0 || !live(row)) return;
  if(unsupported_effect(kind)){ fail(row, ERR_UNSUPPORTED); return; }
  if(TAB_PERM_ATK[kind] || TAB_PERM_HP[kind]){
    CH(CH_BASE_ATK,s,seat) += TAB_PERM_ATK[kind];
    CH(CH_MAX_HP,s,seat) += TAB_PERM_HP[kind];
    CH(CH_BASE_MAX,s,seat) += TAB_PERM_HP[kind];
    CH(CH_HP,s,seat) += TAB_PERM_HP[kind];
  }
  if(TAB_SELF_HARMONY[kind] && CH(CH_HP,s,seat)>0){
    int h=CH(CH_HARMONY,s,seat)+TAB_SELF_HARMONY[kind];
    CH(CH_HARMONY,s,seat)=h<harmony_cap_for(row)?h:harmony_cap_for(row);
  }
  if(TAB_COUNT_SEED[kind]) SI(NANALI_SEED_PLAYED,s) += 1;
  if(TAB_FORM[kind]){
    CH(CH_SHAPE,s,seat)=kind;
    int panel=CAT_HP[kind];
    if(panel>0){
      int perm=CH(CH_BASE_MAX,s,seat)-CAT_BASE_HP[seat];
      if(perm<0) perm=0;
      int nmax=panel+perm;
      CH(CH_MAX_HP,s,seat)=nmax;
      CH(CH_HP,s,seat)=nmax;
    }
  }
  int bonus=CAT_ATK[kind];
  int shield=CAT_SHIELD[kind];
  if(TAB_SORTIE[kind] && CH(CH_SHAPE,s,seat)==CARD_N07){
    bonus += SI(NANALI_SEED_PLAYED,s);
    shield += SI(NANALI_SEED_PLAYED,s);
  }
  if(TAB_SHARE_OVERFLOW[kind]) CH(CH_SHARE_OVERFLOW,s,seat)=1;
  if(TAB_SORTIE_ALLIED_EXTRA[kind] && CH(CH_ALLIED_HURT,s,seat)) bonus+=TAB_SORTIE_ALLIED_EXTRA[kind];
  if(TAB_SELF_DAMAGE[kind]){
    deal_damage(row,s,0,seat,TAB_SELF_DAMAGE[kind],s,seat,0,0);
    knockdown(row); victory(row);
  }
  if(TAB_DRAW_ONE[kind]) draw_one(row,s);
  if(TAB_DAMAGE_PER_DOWNED[kind]){
    int down=0;
    for(int k=0;k<N_SEATS;k++) if(CH(CH_PRESENT,s,k) && CH(CH_DOWN,s,k)>0) down++;
    if(down) deal_front(row,s,TAB_DAMAGE_PER_DOWNED[kind]*down,seat);
  }
  if(TAB_SORTIE[kind]){
    grant_shield(row,s,seat,shield);
    int fu=TAB_FOLLOWUP[kind];
    sortie(row,s,seat,bonus,fu,0,TAB_IGNORE_SHIELD[kind],TAB_HEAL_AFTER_HIT[kind]);
  }
  if(TAB_DRAW_OWNED_SELF[kind]){
    for(int d=0; d<DECK_SLOTS; d++){
      int dk=DK(s,d);
      if(dk>=0 && CAT_OWNER[dk]==seat && CAT_DERIVED[dk]==0){
        int inst=DI(s,d);
        DK(s,d)=-1; compact_deck(row,s);
        add_hand(row,s,dk,inst);
        break;
      }
    }
  }
  if(TAB_DRAW_OWNED_TARGET[kind] && tseat>=0 && tside==s){
    for(int d=0; d<DECK_SLOTS; d++){
      int dk=DK(s,d);
      if(dk>=0 && CAT_OWNER[dk]==tseat && CAT_DERIVED[dk]==0){
        int inst=DI(s,d);
        DK(s,d)=-1; compact_deck(row,s);
        add_hand(row,s,dk,inst);
        break;
      }
    }
  }
  if(TAB_DRAW_FORMS[kind]){
    int found=0, want=TAB_DRAW_FORMS[kind];
    for(int d=0; d<DECK_SLOTS && found<want; d++){
      int dk=DK(s,d);
      if(dk>=0 && CAT_TYPE[dk]==2){
        int inst=DI(s,d);
        DK(s,d)=-1;
        add_hand(row,s,dk,inst);
        found++;
      }
    }
    compact_deck(row,s);
  }
  if(TAB_INSPECT_TOP[kind]){
    int want=TAB_INSPECT_TOP[kind];
    int count=MIN(SI(DECK_N,s), want);
    row[OFF_PENDING_KIND]=1;
    row[OFF_PENDING_SIDE]=s;
    row[OFF_PENDING_COUNT]=count;
    for(int i=0;i<CHOICE_SLOTS;i++){ PC(i)=-1; PI(i)=0; }
    for(int i=0;i<count;i++){
      PC(i)=DK(s,0);
      PI(i)=DI(s,0);
      DK(s,0)=-1; compact_deck(row,s);
    }
    if(count==1){
      add_hand(row,s,PC(0),PI(0));
      row[OFF_PENDING_KIND]=-1;
      row[OFF_PENDING_COUNT]=0;
      PC(0)=-1; PI(0)=0;
    } else if(count>1){
      row[OFF_PHASE]=PHASE_CHOICE;
    } else {
      row[OFF_PENDING_KIND]=-1;
      row[OFF_PENDING_COUNT]=0;
    }
  }
  if(TAB_HEAL_TARGET[kind] && tside==s && tseat>=0){
    int hp=CH(CH_HP,s,tseat)+TAB_HEAL_TARGET[kind];
    if(hp>CH(CH_MAX_HP,s,tseat)) hp=CH(CH_MAX_HP,s,tseat);
    CH(CH_HP,s,tseat)=hp;
    CH(CH_ATK_BUFF,s,tseat) += TAB_BUFF_TARGET[kind];
  }
  if(TAB_HEAL_PLAYER[kind] && tseat<0){
    int hp=SI(HP,s)+TAB_HEAL_PLAYER[kind];
    if(hp>30) hp=30;
    SI(HP,s)=hp;
  }
  if(TAB_REVIVE_TARGET[kind] && CAT_POLICY[kind]==POLICY_NONE) tseat=seat;
  if(TAB_REVIVE_TARGET[kind] && tseat>=0){
    if(SI(LAST_FRONT,s)==tseat) SI(LAST_FRONT,s)=-1;
    CH(CH_DOWN,s,tseat)=0;
    CH(CH_HP,s,tseat)=CH(CH_MAX_HP,s,tseat);
    CH(CH_ATK_BUFF,s,tseat) += TAB_BUFF_TARGET[kind];
  }
  if(TAB_PACT_FRONT[kind]){
    int foe=other_side(s);
    int t=SI(FRONT,foe);
    int had_pact=t>=0 && CH(CH_PACT,foe,t)>0;
    int hit_player=t<0;
    deal_front(row,s,TAB_PACT_FRONT[kind],seat);
    if(t>=0 && CH(CH_HP,foe,t)>0) CH(CH_PACT,foe,t)=1;
    if((had_pact || hit_player) && live(row)) draw_one(row,s);
  }
  if(TAB_PACT_TARGET[kind] && tseat>=0){
    int foe=other_side(s);
    deal_damage(row,foe,0,tseat,TAB_PACT_TARGET[kind],s,seat,0,0);
    set_last_hit(row,foe,tseat,s,seat);
    if(CH(CH_HP,foe,tseat)>0) CH(CH_PACT,foe,tseat)=1;
    knockdown(row); victory(row);
  }
  if(TAB_HEAL_SELF_FULL[kind] && present_alive(row,s,seat)){
    CH(CH_HP,s,seat)=CH(CH_MAX_HP,s,seat);
  }
  if(TAB_HEAL_FRONT_FULL[kind]){
    int front=SI(FRONT,s);
    if(front>=0 && present_alive(row,s,front)){
      CH(CH_HP,s,front)=CH(CH_MAX_HP,s,front);
    }
  }
  if(TAB_NEXT_FOLLOWUP[kind]) CH(CH_NEXT_FOLLOWUP,s,seat) += TAB_NEXT_FOLLOWUP[kind];
  if(TAB_NEXT_BONUS_ATK[kind] || TAB_NEXT_BONUS_SH[kind]){
    CH(CH_NEXT_BONUS,s,seat) += TAB_NEXT_BONUS_ATK[kind];
    CH(CH_NEXT_SHIELD,s,seat) += TAB_NEXT_BONUS_SH[kind];
  }
  if(TAB_SACRIFICE_DRAW[kind]){
    for(int k=0;k<N_SEATS;k++){
      if(k==seat || !present_alive(row,s,k)) continue;
      CH(CH_HP,s,k)=0;
    }
    knockdown(row);
    for(int k=0;k<N_SEATS;k++){
      if(k==seat) continue;
      if(CH(CH_PRESENT,s,k)>0 && CH(CH_DOWN,s,k)>0) CH(CH_DOWN,s,k)=2;
    }
    draw_n(row,s,TAB_SACRIFICE_DRAW[kind]);
  }
  if(TAB_SET_DOWN[kind] && tseat>=0 && tside==other_side(s)){
    /* Python J03 only writes down_turns; knockdowns() skips already-downed units. */
    CH(CH_DOWN,tside,tseat)=TAB_SET_DOWN[kind];
  }
  if(TAB_REVEAL_FRONT_HAND[kind]){
    int foe=other_side(s);
    int front=SI(FRONT,foe);
    if(front>=0){
      for(int i=0;i<HAND_SLOTS;i++){
        int dk=HK(foe,i);
        if(dk>=0 && CAT_OWNER[dk]==front) row[OFF_HAND_REVEALED + (foe*HAND_SLOTS + i)]=1;
      }
    }
  }
  if(TAB_DAMAGE_REVEALED[kind]){
    int foe=other_side(s);
    int count=0;
    for(int i=0;i<HAND_SLOTS;i++){
      if(HK(foe,i)>=0 && row[OFF_HAND_REVEALED + (foe*HAND_SLOTS + i)]) count++;
    }
    if(count) deal_front(row,s,count,seat);
  }
  if(TAB_TEAM_BUFF[kind]){
    for(int k=0;k<N_SEATS;k++){
      if(present_alive(row,s,k)) CH(CH_ATK_BUFF,s,k) += TAB_TEAM_BUFF[kind];
    }
  }
  if(TAB_PENDING_ATK[kind]) CH(CH_PENDING_ATK,s,seat) += TAB_PENDING_ATK[kind];
  if(TAB_HIT_FRONT_BASE[kind]){
    int extra = CH(CH_BASE_ATK,s,seat) - TAB_HIT_FRONT_PRINTED[kind];
    if(extra<0) extra=0;
    deal_front(row,s,TAB_HIT_FRONT_BASE[kind]+extra,seat);
  }
  if(TAB_BURN_INFINITE[kind]){
    int foe=other_side(s);
    if(SI(BURN_LEFT,foe)>0 && SI(BURN_BY,foe)==s) SI(BURN_INFINITE,foe)=1;
  }
  if(TAB_DAMAGE_MISSING_OTHERS[kind]){
    int missing = CH(CH_MAX_HP,s,seat) - CH(CH_HP,s,seat);
    if(missing<0) missing=0;
    if(missing>0){
      for(int os=0; os<N_SIDES; os++){
        for(int rank=1; rank<=N_SEATS; rank++){
          int k=-1;
          for(int cand=0; cand<N_SEATS; cand++){
            if(team_rank(row,os,cand)==rank){ k=cand; break; }
          }
          if(k<0 || character_entity_key(os,k)==character_entity_key(s,seat)) continue;
          if(present_alive(row,os,k)){
            deal_damage(row,os,0,k,missing,s,seat,0,0);
            set_last_hit(row,os,k,s,seat);
          }
        }
      }
      knockdown(row); victory(row);
    }
  }
  if(TAB_HP_FLOOR[kind]) CH(CH_HP_FLOOR,s,seat)=1;
}

static void finish_play(int *row, int s, int kind, int inst, int owner){
  if(kind<0 || !live(row)) return;
  to_discard(row,s,kind,inst);
  if(CAT_TYPE[kind]==0 && present_alive(row,s,SEAT_ZERO) && owner==SEAT_ZERO)
    CH(CH_HARMONY,s,SEAT_ZERO)=harmony_cap_for(row);
}

static void grant_op_resources(int *row, int s, int seat){
  if(!live(row) || !present_alive(row,s,seat) || SI(FRONT,s)!=seat) return;
  grant_combat_resources(row,s,seat);
}

static void play_card(int *row, int s, int hand_slot, int tside, int tseat){
  int inst=0;
  int kind=remove_hand_slot(row,s,hand_slot,&inst);
  if(kind<0) return;
  int owner=CAT_OWNER[kind];
  int cost=CAT_COST[kind];
  int form_bonus = form_draw_bonus(row,s,kind);
  if(card_free(row,s,kind)) SI(USED_INSTANT,s)=1;
  else SI(AP,s)-=cost;
  play_effects(row,s,owner,kind,tside,tseat);
  if(form_bonus && live(row) && present_alive(row,s,owner)) draw_one(row,s);
  if(row[OFF_PHASE]!=PHASE_CHOICE){
    finish_play(row,s,kind,inst,owner);
    if(TAB_SORTIE[kind]) grant_op_resources(row,s,owner);
  } else {
    row[OFF_PENDING_PLAY_KIND]=kind;
    row[OFF_PENDING_PLAY_INST]=inst;
  }
}

static void choose_pending(int *row, int slot){
  if(row[OFF_PHASE]!=PHASE_CHOICE || !live(row)) return;
  int s=row[OFF_PENDING_SIDE];
  if(slot<0 || slot>=row[OFF_PENDING_COUNT]) return;
  int selected=PC(slot);
  add_hand(row,s,selected,PI(slot));
  for(int i=0;i<row[OFF_PENDING_COUNT];i++){
    if(i==slot || PC(i)<0) continue;
    int n=SI(DECK_N,s);
    if(n<DECK_SLOTS){ DK(s,n)=PC(i); DI(s,n)=PI(i); SI(DECK_N,s)=n+1; }
  }
  row[OFF_PHASE]=PHASE_PLAYING;
  row[OFF_PENDING_KIND]=-1;
  row[OFF_PENDING_COUNT]=0;
  for(int i=0;i<CHOICE_SLOTS;i++){ PC(i)=-1; PI(i)=0; }
  int pk=row[OFF_PENDING_PLAY_KIND];
  if(pk>=0){
    int owner=CAT_OWNER[pk];
    finish_play(row,s,pk,row[OFF_PENDING_PLAY_INST],owner);
    if(TAB_SORTIE[pk]) grant_op_resources(row,s,owner);
    row[OFF_PENDING_PLAY_KIND]=-1;
    row[OFF_PENDING_PLAY_INST]=0;
  }
}

static void start_ultimate(int *row, int s, int seat){
  if(!live(row) || !present_alive(row,s,seat)) return;
  CH(CH_ENERGY,s,seat)=0;
  if(CAT_ULT_KIND[seat]==0){
    /* instant: jiuyuan */
    int foe=other_side(s);
    for(int rank=1;rank<=N_SEATS && live(row);rank++){
      int k=-1;
      for(int candidate=0;candidate<N_SEATS;candidate++) if(team_rank(row,foe,candidate)==rank){ k=candidate;break; }
      if(k<0) continue;
      if(CH(CH_HP,foe,k)<=0) continue;
      int amt = CH(CH_PACT,foe,k)>0 ? 3 : 1;
      if(CH(CH_PACT,foe,k)>0 && CH(CH_SHAPE,s,seat)==CARD_J07 && present_alive(row,s,seat))
        CH(CH_ENERGY,s,seat)=MIN(energy_cap_for(row,seat),CH(CH_ENERGY,s,seat)+1);
      if(CH(CH_PACT,foe,k)>0) CH(CH_PACT,foe,k)=0;
      deal_damage(row,foe,0,k,amt,s,seat,0,0);
      set_last_hit(row,foe,k,s,seat);
      knockdown(row); victory(row);
    }
    knockdown(row); victory(row);
  } else {
    CH(CH_AWAKENED,s,seat)=1;
    CH(CH_ULT_TURNS,s,seat)=CAT_ULT_TURNS[seat];
    if(seat==SEAT_ILOY) CH(CH_ENERGY_NEXT,s,SEAT_ILOY)=1;
  }
}

static void clamp_resources(int *row){
  if(!row[OFF_ESCALATION_ENABLED]) return;
  for(int os=0; os<N_SIDES; os++){
    for(int k=0;k<N_SEATS;k++){
      if(CH(CH_PRESENT,os,k)==0) continue;
      int ecap=energy_cap_for(row,k);
      if(CH(CH_ENERGY,os,k)>ecap) CH(CH_ENERGY,os,k)=ecap;
      int hcap=harmony_cap_for(row);
      if(CH(CH_HARMONY,os,k)>hcap) CH(CH_HARMONY,os,k)=hcap;
    }
  }
}

static void begin_escalation(int *row, int s){
  if(!(row[OFF_ESCALATION_ENABLED] && row[OFF_TURN] >= 6)) return;
  int ties[N_SEATS];
  int n_tie=0;
  int best_energy=1<<30;
  for(int rank=1; rank<=N_SEATS; rank++){
    int k=-1;
    for(int cand=0; cand<N_SEATS; cand++){
      if(team_rank(row,s,cand)==rank){ k=cand; break; }
    }
    if(k<0 || !present_alive(row,s,k)) continue;
    int e=CH(CH_ENERGY,s,k);
    if(n_tie==0 || e<best_energy){
      best_energy=e;
      n_tie=1;
      ties[0]=k;
    } else if(e==best_energy){
      ties[n_tie++]=k;
    }
  }
  if(n_tie<=0) return;
  if(n_tie>1){
    for(int i=n_tie-1;i>0;i--){
      int j = lcg(&row[OFF_RNG]) % (i+1);
      int tmp=ties[i]; ties[i]=ties[j]; ties[j]=tmp;
    }
  }
  int best=ties[0];
  int e=CH(CH_ENERGY,s,best)+1;
  int cap=energy_cap_for(row,best);
  if(e>cap) e=cap;
  CH(CH_ENERGY,s,best)=e;
}

static void begin_turn(int *row, int s){
  if(!live(row)) return;
  row[OFF_PHASE]=PHASE_PLAYING;
  row[OFF_ACTIVE]=s;
  row[OFF_TURN]+=1;
  SI(TURN_COUNT,s)+=1;
  SI(SHIELD,s)=0;
  SI(USED_INSTANT,s)=0;
  SI(SURPLUS,s)=0;
  SI(HARMONY_DAMAGE,s)=0;
  SI(WEAVE,0)=0; SI(WEAVE,1)=0;
  CH(CH_ALLIED_HURT,s,SEAT_BAICANG)=0;
  CH(CH_SHARE_OVERFLOW,0,SEAT_BOHE)=0;
  CH(CH_SHARE_OVERFLOW,1,SEAT_BOHE)=0;
  for(int k=0;k<N_SEATS;k++){
    CH(CH_HP_FLOOR,s,k)=0;
    CH(CH_HP_FLOOR,other_side(s),k)=0;
    if(CH(CH_PENDING_ATK,s,k)>0){
      CH(CH_ATK_BUFF,s,k) += CH(CH_PENDING_ATK,s,k);
      CH(CH_PENDING_ATK,s,k)=0;
    }
  }
  clamp_resources(row);
  for(int os=0; os<N_SIDES; os++){
    for(int k=0;k<N_SEATS;k++){
      if(CH(CH_COLLAPSE_UNTIL,os,k)>0 && CH(CH_COLLAPSE_BY,os,k)==s && CH(CH_COLLAPSE_UNTIL,os,k)<=SI(TURN_COUNT,s)){
        CH(CH_COLLAPSE_UNTIL,os,k)=0;
        CH(CH_COLLAPSE_BY,os,k)=-1;
      }
    }
  }
  for(int k=0;k<N_SEATS;k++){
    CH(CH_SHIELD,s,k)=0;
    if(CH(CH_DOWN,s,k)>0){
      if(SI(LAST_FRONT,s)==k) SI(LAST_FRONT,s)=-1;
      CH(CH_DOWN,s,k)-=1;
      if(CH(CH_DOWN,s,k)==0) CH(CH_HP,s,k)=CH(CH_MAX_HP,s,k);
    }
  }
  begin_escalation(row,s);
  if(SI(TURN_COUNT,s)==1) grant_seed(row,s);
  if(present_alive(row,s,SEAT_ILOY) && CH(CH_ENERGY_NEXT,s,SEAT_ILOY)>0){
    CH(CH_ENERGY_NEXT,s,SEAT_ILOY)=0;
    for(int k=0;k<N_SEATS;k++){
      if(CH(CH_HP,s,k)>0){
        int e=CH(CH_ENERGY,s,k)+1;
        int cap=energy_cap_for(row,k);
        if(e>cap) e=cap;
        CH(CH_ENERGY,s,k)=e;
      }
    }
  }
  if(present_alive(row,s,SEAT_BOHE) && CH(CH_SHAPE,s,SEAT_BOHE)==CARD_M08 && !collapsed(row,s,SEAT_BOHE)){
    sortie(row,s,SEAT_BOHE,0,0,0,0,0);
    /* M08 automatic attack grants no combat resources. */
  }
  if(!live(row)) return;
  if(SI(EXTRA_GENESIS,s)>0){
    int actor=SI(EXTRA_GENESIS_ACTOR,s);
    SI(EXTRA_GENESIS,s)=0;
    SI(EXTRA_GENESIS_ACTOR,s)=-1;
    if(actor>=0 && CH(CH_HP,s,actor)>0) resolve_genesis(row,s,actor,0);
  }
  if(!live(row)) return;
  if(SI(BURN_LEFT,s)>0){
    int by = SI(BURN_BY,s)>=0 ? SI(BURN_BY,s) : other_side(s);
    SI(HARMONY_DAMAGE,by)=1;
    int target = SI(FRONT,s);
    if(target>=0) deal_damage(row,s,0,target,1,by,-1,0,0);
    else deal_damage(row,s,1,0,1,by,-1,0,0);
    knockdown(row); victory(row);
    if(!SI(BURN_INFINITE,s)){
      SI(BURN_LEFT,s)-=1;
      if(SI(BURN_LEFT,s)<=0){ SI(BURN_LEFT,s)=0; SI(BURN_BY,s)=-1; SI(BURN_INFINITE,s)=0; }
    }
  }
  if(!live(row)) return;
  for(int k=0;k<N_SEATS;k++){
    if(CH(CH_ULT_TURNS,s,k)>0){
      CH(CH_ULT_TURNS,s,k)-=1;
      if(CH(CH_ULT_TURNS,s,k)<=0){ CH(CH_AWAKENED,s,k)=0; CH(CH_ENERGY_NEXT,s,k)=0; }
    }
  }
  if(SI(FRONT,s)>=0 && !collapsed(row,s,SI(FRONT,s))) move_out(row,s,SI(FRONT,s));
  draw_one(row,s);
  if(!live(row)) return;
  int extra=SI(EXTRA_AP,s);
  SI(EXTRA_AP,s)=0;
  SI(AP,s)=(row[OFF_TURN]==1?1:2)+extra;
  SI(NORMAL_ATK,s)=1;
  SI(ULTIMATES_USED,s)=0;
  SI(ULTIMATE_OK,s)=1;
}

static void end_turn(int *row, int s){
  if(!live(row)) return;
  knockdown(row); victory(row);
  if(!live(row)) return;
  int front=SI(FRONT,s);
  if(CH(CH_SHAPE,s,SEAT_ZERO)==CARD_Z08 && CH(CH_HP,s,SEAT_ZERO)>0 && front>=0)
    CH(CH_ATK_BUFF,s,front) += TURN_END_ATK;
  if(CH(CH_SHAPE,s,SEAT_ZERO)==CARD_Z07 && CH(CH_HP,s,SEAT_ZERO)>0 && front>=0){
    int hp=CH(CH_HP,s,front)+TURN_END_HEAL;
    if(hp>CH(CH_MAX_HP,s,front)) hp=CH(CH_MAX_HP,s,front);
    CH(CH_HP,s,front)=hp;
  }
  if(CH(CH_SHAPE,s,SEAT_ZERO)==CARD_Z06 && CH(CH_HP,s,SEAT_ZERO)>0)
    SI(AP,s) += 1;
  SI(NORMAL_ATK,s)=0;
  SI(ULTIMATE_OK,s)=0;
  SI(EXTRA_FLOWER,s)=0;
  begin_turn(row, other_side(s));
}

static void clear_legal(int *row){
  for(int i=0;i<MAX_LEGAL;i++){ LT(i)=-1; LA(i)=-1; LH(i)=-1; LS(i)=-1; LK(i)=-1; }
  row[OFF_LEGAL_N]=0;
}

static int popcount_bits(int x){
  int n=0;
  while(x){ n += x&1; x >>= 1; }
  return n;
}
static int mulligan_chooser(int *row){
  int done = row[OFF_PENDING_COUNT] & 3;
  if(!(done & 1)) return 0;
  if(!(done & 2)) return 1;
  return 0;
}
static void rebuild_legal(int *row){
  clear_legal(row);
  if(!live(row)) return;
  if(row[OFF_PHASE]==PHASE_CHOICE){
    int n=row[OFF_PENDING_COUNT];
    for(int i=0;i<n && i<CHOICE_SLOTS;i++){
      LT(i)=ACT_CHOOSE; LH(i)=i;
    }
    row[OFF_LEGAL_N]=n;
    return;
  }
  if(row[OFF_PHASE]==PHASE_MULLIGAN){
    row[OFF_ACTIVE]=mulligan_chooser(row);
    int s=row[OFF_ACTIVE];
    if(row[OFF_PENDING_COUNT] & (1<<s)) return;
    int n=SI(HAND_N,s);
    if(n>5) n=5;
    int count=0;
    int limit = 1<<n;
    /* Same subset ordering as Python combinations, including deterministic ties. */
    const int masks[26]={0,1,2,4,8,16,3,5,9,17,6,10,18,12,20,24,7,11,19,13,21,25,14,22,26,28};
    for(int mi=0; mi<26; mi++){
      int mask=masks[mi];
      if(mask>=limit) continue;
      LT(count)=ACT_MULLIGAN;
      LA(count)=-1;
      LH(count)=mask;
      LS(count)=-1;
      LK(count)=-1;
      count++;
    }
    row[OFF_LEGAL_N]=count;
    return;
  }
  if(row[OFF_PHASE]!=PHASE_PLAYING) return;
  int s=row[OFF_ACTIVE];
  int ap=SI(AP,s);
  int foe=other_side(s);
  LT(LEGAL_END)=ACT_END;
  for(int k=0;k<N_SEATS;k++){
    int can_atk = ap>=1 && SI(NORMAL_ATK,s)>0 && present_alive(row,s,k) && !collapsed(row,s,k);
    if(can_atk){ LT(LEGAL_ATK+k)=ACT_ATTACK; LA(LEGAL_ATK+k)=k; }
    int can_ult = SI(ULTIMATE_OK,s)>0 && present_alive(row,s,k)
      && CH(CH_ENERGY,s,k)>=energy_cap_for(row,k)
      && SI(ULTIMATES_USED,s)<ultimate_limit_for(row)
      && CH(CH_ULTIMATE_USED_TURN,s,k)!=row[OFF_TURN];
    if(can_ult){ LT(LEGAL_ULT+k)=ACT_ULTIMATE; LA(LEGAL_ULT+k)=k; }
  }
  for(int hs=0; hs<HAND_SLOTS; hs++){
    int kind=HK(s,hs);
    if(kind<0) continue;
    int owner=CAT_OWNER[kind];
    int cost=CAT_COST[kind];
    int policy=CAT_POLICY[kind];
    if(!(CH(CH_HP,s,owner)>0) && !CAT_PLAYABLE_DOWNED[kind]) continue;
    if(collapsed(row,s,owner)) continue;
    int free = card_free(row,s,kind);
    if(!(free || ap>=cost)) continue;
    int require=CAT_REQUIRE[kind];
    if(require==REQUIRE_OWN_BURN && !(SI(BURN_LEFT,foe)>0 && SI(BURN_BY,foe)==s)) continue;
    if(require==REQUIRE_SURPLUS && SI(SURPLUS,s)==0) continue;
    if(require==REQUIRE_HARMONY_DAMAGE && SI(HARMONY_DAMAGE,s)==0) continue;
    int base=LEGAL_PLAY + hs*PLAY_TARGETS;
    int none_ok = (policy==POLICY_NONE) || (policy==POLICY_ENEMY_FRONT && SI(FRONT,foe)>=0)
      || (policy==POLICY_ALLY_FRONT && SI(FRONT,s)>=0);
    if(none_ok){ LT(base)=ACT_PLAY; LA(base)=owner; LH(base)=hs; }
    for(int ally=0; ally<N_SEATS; ally++){
      int slot=base+1+ally;
      int ok=0;
      if(CH(CH_PRESENT,s,ally)>0){
        if(policy==POLICY_ALLY && CH(CH_HP,s,ally)>0) ok=1;
        if(policy==POLICY_INJURED && CH(CH_HP,s,ally)>0 && CH(CH_HP,s,ally)<CH(CH_MAX_HP,s,ally)) ok=1;
        if(policy==POLICY_DOWN_ALLY && (CH(CH_DOWN,s,ally)>0 || CH(CH_HP,s,ally)<=0)) ok=1;
      }
      if(ok){ LT(slot)=ACT_PLAY; LA(slot)=owner; LH(slot)=hs; LS(slot)=s; LK(slot)=ally; }
    }
    for(int en=0; en<N_SEATS; en++){
      int slot=base+1+N_SEATS+en;
      int ok = 0;
      if(CH(CH_PRESENT,foe,en)>0){
        if(policy==POLICY_ENEMY && CH(CH_HP,foe,en)>0) ok=1;
        if(policy==POLICY_DOWN_ENEMY && CH(CH_DOWN,foe,en)>0) ok=1;
      }
      if(ok){ LT(slot)=ACT_PLAY; LA(slot)=owner; LH(slot)=hs; LS(slot)=foe; LK(slot)=en; }
    }
    int slot=base+1+2*N_SEATS;
    if(policy==POLICY_INJURED && SI(HP,s)<30){
      LT(slot)=ACT_PLAY; LA(slot)=owner; LH(slot)=hs; LS(slot)=s; LK(slot)=-1;
    }
  }
  int last=0;
  for(int i=0;i<MAX_LEGAL;i++) if(LT(i)>=0) last=i+1;
  row[OFF_LEGAL_N]=last;
  if(last>MAX_LEGAL) fail(row, ERR_UNSUPPORTED);
}

static void step_one(int *row, int action){
  rebuild_legal(row);
  if(action<0) return;
  if(!live(row)) return;
  if(action>=row[OFF_LEGAL_N] || LT(action)<0) return;
  int typ=LT(action), actor=LA(action), hand=LH(action), tside=LS(action), tseat=LK(action);
  int s=row[OFF_ACTIVE];
  if(typ==ACT_END) end_turn(row,s);
  else if(typ==ACT_ATTACK){
    SI(AP,s)-=1; SI(NORMAL_ATK,s)=0;
    sortie(row,s,actor,0,0,0,0,0);
    grant_op_resources(row,s,actor);
  } else if(typ==ACT_ULTIMATE){
    CH(CH_ULTIMATE_USED_TURN,s,actor)=row[OFF_TURN];
    SI(ULTIMATES_USED,s)+=1;
    if(SI(ULTIMATES_USED,s) >= ultimate_limit_for(row)) SI(ULTIMATE_OK,s)=0;
    start_ultimate(row,s,actor);
  } else if(typ==ACT_PLAY){
    play_card(row,s,hand,tside,tseat);
  } else if(typ==ACT_CHOOSE){
    choose_pending(row,hand);
  } else if(typ==ACT_MULLIGAN){
    apply_mulligan(row,s,hand);
  }
  rebuild_legal(row);
}

#ifdef __CUDACC__
extern "C" __global__
#endif
#ifndef __CUDACC__
#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
#endif
void starter_engine(int *rows, const int *actions, int n){
#ifdef __CUDACC__
  int row_i = blockIdx.x*blockDim.x+threadIdx.x;
  if(row_i>=n) return;
  int *row = rows + row_i*ROW_WIDTH;
  step_one(row, actions[row_i]);
#else
  for(int row_i=0; row_i<n; row_i++){
    int *row = rows + row_i*ROW_WIDTH;
    step_one(row, actions[row_i]);
  }
#endif
}

static void clear_row(int *row){
  for(int i=0;i<ROW_WIDTH;i++) row[i]=0;
  row[OFF_WINNER]=-1;
  row[OFF_PENDING_KIND]=-1;
  row[OFF_PENDING_SIDE]=-1;
  row[OFF_PENDING_PLAY_KIND]=-1;
  for(int i=0;i<CHOICE_SLOTS;i++){ PC(i)=-1; PI(i)=0; }
  for(int s=0;s<N_SIDES;s++){
    SI(FRONT,s)=-1; SI(LAST_FRONT,s)=-1; SI(EXTRA_GENESIS_ACTOR,s)=-1;
    SI(BURN_BY,s)=-1;
    for(int i=0;i<HAND_SLOTS;i++){ HK(s,i)=-1; row[OFF_HAND_REVEALED + (s*HAND_SLOTS + i)]=0; }
    for(int i=0;i<DECK_SLOTS;i++) DK(s,i)=-1;
    for(int i=0;i<DISCARD_SLOTS;i++) XK(s,i)=-1;
    for(int k=0;k<N_SEATS;k++){
      CH(CH_SHAPE,s,k)=-1;
      CH(LAST_HIT_SIDE,s,k)=-1;
      CH(LAST_HIT_SEAT,s,k)=-1;
      CH(CH_COLLAPSE_BY,s,k)=-1;
    }
  }
  for(int i=0;i<MAX_LEGAL;i++){ LT(i)=-1; LA(i)=-1; LH(i)=-1; LS(i)=-1; LK(i)=-1; }
}

static void shuffle_deck(int *row, int s, int rng){
  int n=SI(DECK_N,s);
  for(int i=n-1;i>0;i--){
    rng = (int)(((unsigned int)rng * 1103515245u + 12345u) & 0x7FFFFFFFu);
    int j = rng % (i+1);
    int tk=DK(s,i), ti=DI(s,i);
    DK(s,i)=DK(s,j); DI(s,i)=DI(s,j);
    DK(s,j)=tk; DI(s,j)=ti;
  }
}
static void shuffle_deck_lcg(int *row, int s){
  int n=SI(DECK_N,s);
  for(int i=n-1;i>0;i--){
    int j = lcg(&row[OFF_RNG]) % (i+1);
    int tk=DK(s,i), ti=DI(s,i);
    DK(s,i)=DK(s,j); DI(s,i)=DI(s,j);
    DK(s,j)=tk; DI(s,j)=ti;
  }
}
static void apply_mulligan(int *row, int s, int mask){
  if(row[OFF_PHASE]!=PHASE_MULLIGAN || !live(row)) return;
  if(row[OFF_PENDING_COUNT] & (1<<s)) return;
  int n=SI(HAND_N,s);
  if(n>5) n=5;
  if(mask<0 || mask>=(1<<n) || popcount_bits(mask)>3) return;
  int removed_k[5];
  int removed_i[5];
  int n_rem=0;
  for(int slot=0; slot<n; slot++){
    if(!(mask & (1<<slot))) continue;
    removed_k[n_rem]=HK(s,slot);
    removed_i[n_rem]=HI(s,slot);
    HK(s,slot)=-1; HI(s,slot)=0;
    row[OFF_HAND_REVEALED + (s*HAND_SLOTS + slot)]=0;
    n_rem++;
  }
  compact_hand(row,s);
  draw_n(row,s,n_rem);
  if(!live(row)) return;
  for(int i=0;i<n_rem;i++){
    int dn=SI(DECK_N,s);
    if(dn>=DECK_SLOTS){ fail(row, ERR_UNSUPPORTED); return; }
    DK(s,dn)=removed_k[i];
    DI(s,dn)=removed_i[i];
    SI(DECK_N,s)=dn+1;
  }
  shuffle_deck_lcg(row,s);
  row[OFF_PENDING_COUNT] = (row[OFF_PENDING_COUNT] | (1<<s)) & 3;
  if((row[OFF_PENDING_COUNT] & 3)==3){
    row[OFF_PENDING_COUNT]=0;
    begin_turn(row, row[OFF_FIRST]);
  } else {
    row[OFF_PHASE]=PHASE_MULLIGAN;
    row[OFF_ACTIVE]=mulligan_chooser(row);
  }
}

static void reset_one_with_decks(int *row, int seed, const int *deck_a, const int *deck_b, int escalation){
  clear_row(row);
  int first = (seed & 0x7FFFFFFF) % 2;
  int enable_escalation = escalation & 1;
  int enable_mulligan = (escalation >> 1) & 1;
  row[OFF_RNG]=seed & 0x7FFFFFFF;
  row[OFF_FIRST]=first;
  row[OFF_ACTIVE]=first;
  row[OFF_ESCALATION_ENABLED]=enable_escalation ? 1 : 0;
  row[OFF_NEXT_INSTANCE]=32;
  const int *decks[2] = {deck_a, deck_b};
  for(int s=0;s<N_SIDES;s++){
    SI(HP,s)=30;
    SI(SHIELD,s)=(s==first)?0:5;
    const int *deck = decks[s];
    for(int slot=0;slot<4;slot++){
      int k=deck[slot];
      CH(CH_PRESENT,s,k)=1;
      CH(CH_ORDER,s,k)=slot+1;
      CH(CH_HP,s,k)=CAT_BASE_HP[k];
      CH(CH_MAX_HP,s,k)=CAT_BASE_HP[k];
      CH(CH_BASE_MAX,s,k)=CAT_BASE_HP[k];
      CH(CH_BASE_ATK,s,k)=CAT_BASE_ATK[k];
      CH(CH_ENERGY_MAX,s,k)=CAT_ENERGY_MAX[k];
      CH(CH_SHAPE,s,k)=-1;
      CH(CH_COLLAPSE_BY,s,k)=-1;
      CH(LAST_HIT_SIDE,s,k)=-1;
      CH(LAST_HIT_SEAT,s,k)=-1;
    }
    for(int i=0;i<32;i++){
      DK(s,i)=deck[4+i];
      DI(s,i)=i+1;
    }
    SI(DECK_N,s)=32;
    shuffle_deck(row,s,(seed & 0x7FFFFFFF) + (s+1)*10007);
    draw_n(row,s,5);
  }
  if(enable_mulligan){
    row[OFF_PHASE]=PHASE_MULLIGAN;
    row[OFF_ACTIVE]=0;
    row[OFF_PENDING_COUNT]=0;
    rebuild_legal(row);
    return;
  }
  row[OFF_PHASE]=PHASE_PLAYING;
  begin_turn(row, first);
  rebuild_legal(row);
}

static void reset_one(int *row, int seed){
  int preset[36];
  for(int i=0;i<4;i++) preset[i]=DECK_SEATS[i];
  for(int i=0;i<STARTER_DECK_N;i++) preset[4+i]=STARTER_DECK_KINDS[i];
  reset_one_with_decks(row, seed, preset, preset, 1);
}
#ifdef __CUDACC__
extern "C" __global__
#endif
#ifndef __CUDACC__
#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
#endif
void starter_reset(int *rows, const int *seeds, int n){
#ifdef __CUDACC__
  int row_i = blockIdx.x*blockDim.x+threadIdx.x;
  if(row_i>=n) return;
  if(seeds[row_i]>=0) reset_one(rows + row_i*ROW_WIDTH, seeds[row_i]);
#else
  for(int row_i=0; row_i<n; row_i++){
    if(seeds[row_i]>=0) reset_one(rows + row_i*ROW_WIDTH, seeds[row_i]);
  }
#endif
}

#ifdef __CUDACC__
extern "C" __global__
#endif
#ifndef __CUDACC__
#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
#endif
void public_reset(int *rows, const int *seeds, const int *deck_a, const int *deck_b, int n, int escalation){
#ifdef __CUDACC__
  int row_i = blockIdx.x*blockDim.x+threadIdx.x;
  if(row_i>=n) return;
  if(seeds[row_i]>=0) reset_one_with_decks(rows + row_i*ROW_WIDTH, seeds[row_i], deck_a + row_i*36, deck_b + row_i*36, escalation);
#else
  for(int row_i=0; row_i<n; row_i++){
    if(seeds[row_i]>=0) reset_one_with_decks(rows + row_i*ROW_WIDTH, seeds[row_i], deck_a + row_i*36, deck_b + row_i*36, escalation);
  }
#endif
}
'''
