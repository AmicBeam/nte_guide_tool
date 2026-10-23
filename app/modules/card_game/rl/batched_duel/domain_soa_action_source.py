"""Root action router for validated physical action table ordinals."""
from .domain_soa_execution_base import _literal


def emit_source():
    from app.modules.card_game.rl.league_schema import OPTION_IDS
    text='DFN const char*d_option_name(int option){switch(option){'+''.join(f'case {i}:return {_literal(name)};'for i,name in enumerate(OPTION_IDS))+'default:return nullptr;}}\n'
    text+='''DFN void d_root_mulligan(DX*c,const int*selected,int count){int side=c->side;DS*f=c->s;if(f->phase[0]!=0||count<0||count>3||f->mulligan_count[0]<0||f->mulligan_count[0]>=2){d_fail(c,1);return;}for(int j=0;j<f->mulligan_count[0];j++)if(f->mulligan_sides[j]==side){d_fail(c,1);return;}int removed[3],n=0,hand=f->zone_count[side*4];for(int j=0;j<hand;j++){int card=f->zone_card_index[(side*4)*Z+j],yes=0;for(int k=0;k<count;k++)if(selected[k]==card)yes++;if(yes){if(yes>1||n>=3){d_fail(c,1);return;}removed[n++]=card;}}if(n!=count){d_fail(c,1);return;}for(int j=0;j<n;j++)d_remove_zone_card(c,side,0,removed[j]);d_event(c);d_draw(c,side,n);for(int j=0;j<n;j++)d_insert_deck(c,side,removed[j],d_deck_count(c,side));d_shuffle_zone(c,side,1);f->mulligan_sides[f->mulligan_count[0]++]=side;if(f->mulligan_count[0]==2)d_begin_turn(c,f->first_side[0]);}
DFN DPayment d_root_payment(DX*c){DPayment p;p.ap=d_pb(c,c->side,"normal_attack_available")?1:-1;p.consume_normal=p.ap>=0;p.resource=PAY_NONE;p.bonus=0;p=d_decide_payment(c,"normal_attack_payment",p,nullptr);if(d_evaluate(c,DR{c->side,c->actor},"rule.normal_attack_instant",0)!=0&&!d_used(c,c->side,"instant")){p.ap=0;p.consume_normal=0;p.resource=PAY_INSTANT;p.bonus=0;}if(p.ap>=0&&d_flag(c,c->side,c->actor,"free_normal_attack_turn")==c->s->turn[0])p.ap=0;return p;}
DFN int d_selection_ref(DX*c,long long value){if(value< -1||value>=2LL*K){d_fail(c,1);return -1;}if(value>=K)value-=K;return (int)value;}
DFN void d_apply_root(DX*c,const long long*action,const long long*selection,int width,const long long*top,const long long*bottom,int bottom_plural){
 if(d_finished(c)||d_bad(c)||!d_valid_side(c,c->side))return;DS*f=c->s;f->choice_serial=0;int kind=(int)action[0];if(kind<0||kind>8||width<1||width>64){d_fail(c,1);return;}
 int phase=f->phase[0];if((phase==1&&f->active_side[0]!=c->side)||(phase==2&&f->choice_side[0]!=c->side)||(phase==0&&kind!=5)){d_fail(c,1);return;}
 if(f->version[0]>=2147483647){d_fail(c,2);return;}f->version[0]++;f->current_action_version[0]=f->version[0];char text[16];text[0]='v';d_uint_text(text+1,f->version[0]);f->current_action_id_slot[0]=d_intern(c,text);f->combat_group[0]=0;
 c->actor=(int)action[1];c->card=(int)action[2];c->target=DR{-1,-2};c->target_card=-1;c->option_slot=-1;
 int target_kind=(int)action[3],target_side=(int)action[4],target=(int)action[5];
 if(target_kind==1||target_kind==2){if(!d_valid_side(c,target_side)||(target_kind==2&&!d_valid_char(c,target_side,target)))return;c->target=DR{target_side,target_kind==1?-1:target};}
 else if(target_kind==3){if(target<0||target>=K||!f->card_valid[target]){d_fail(c,1);return;}c->target_card=target;}
 if(action[6]>=0){const char*opt=d_option_name((int)action[6]);if(!opt){d_fail(c,1);return;}c->option_slot=d_intern(c,opt);}
 if(kind==5){int refs[64],n=0;for(int j=0;j<width;j++)if(selection[j]>=0)refs[n++]=d_selection_ref(c,selection[j]);d_root_mulligan(c,refs,n);}
 else if(kind==4||kind==6||kind==8){if(phase!=2||!f->pending_choice_present[0]){d_fail(c,1);return;}int refs[64],nt=0,nb=0,n=0,t[4],b[4];if(kind==4){if(c->card<0||c->card>=K){d_fail(c,1);return;}refs[n++]=c->card;}else if(kind==8){for(int j=0;j<width;j++)if(selection[j]>=0)refs[n++]=d_selection_ref(c,selection[j]);}else for(int j=0;j<4;j++){if(top[j]>=0)t[nt++]=d_selection_ref(c,top[j]);if(bottom[j]>=0)b[nb++]=d_selection_ref(c,bottom[j]);}if(d_bad(c))return;d_choose(c,f->choice_kind[0],refs,n,t,nt,b,nb,bottom_plural);}
 else if(kind==0){if(phase!=1){d_fail(c,1);return;}c->actor=d_order_slot(c,c->side,0);c->card=-1;d_end_turn(c);}
 else if(kind==2){if(!d_valid_char(c,c->side,c->actor))return;int used=d_pi(c,c->side,"ultimates_used")+1;d_set_pi(c,c->side,"ultimates_used",used);int limit=f->escalation_enabled[0]&&f->turn[0]>=6?2:1;d_set_pb(c,c->side,"ultimate_available",d_pb(c,c->side,"ultimate_available")&&used<limit);d_start_ultimate(c,c->side,c->actor);}
 else if(kind==7){if(!d_valid_char(c,c->side,c->actor))return;d_end_ultimate(c,c->side,c->actor);}
 else if(kind==1){if(!d_valid_char(c,c->side,c->actor))return;c->card=-1;DPayment payment=d_root_payment(c);if(payment.ap<0||payment.ap>d_pi(c,c->side,"ap")){d_fail(c,1);return;}d_set_pi(c,c->side,"ap",d_pi(c,c->side,"ap")-payment.ap);if(payment.consume_normal)d_set_pb(c,c->side,"normal_attack_available",0);if(payment.resource==PAY_INSTANT)d_set_used(c,c->side,"instant",1);d_open_operation(c);d_set_op_int(c,"normal_attack",1,1);kit_pay_normal_attack(c,payment);DCombatSpec spec=combat_spec_from_attack(c,nullptr);spec.base.bonus=payment.bonus;spec.int_presence[0]=2;combat_sortie_ex(c,&spec);if(f->phase[0]!=2)d_finish_operation(c);}
 else if(kind==3){if(phase!=1||c->card<0||c->card>=K||!f->card_valid[c->card]){d_fail(c,1);return;}int physical=c->card;d_remove_zone_card(c,c->side,0,physical);if(c->option_slot>=0){f->op_option_id_presence[0]=2;f->op_option_id_slot[0]=c->option_slot;}c->card=d_effective_card(c,c->side,physical);d_play_card(c,0,0);}
 else d_fail(c,1);
 if(!d_bad(c))d_drain(c);if(!d_bad(c)){d_collect_cards(c);d_collect_strings(c);}
}
'''
    return text
