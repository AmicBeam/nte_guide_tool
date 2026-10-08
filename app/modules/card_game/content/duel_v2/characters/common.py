def set_shape(c):
    c.set_shape()


def draw_owned_card(c, character_id=None, *, reveal=True):
    from app.modules.card_game.engine.duel_v2.state import add_hand, effective_card, finished, hero, log, name
    from app.modules.card_game.engine.duel_v2.presentation import public_card
    if finished(c.state):
        return
    owner = character_id or c.cid
    team = c.state['sides'][c.side]
    for index, card in enumerate(team['deck']):
        if card.get('character_id') == owner and not card.get('copy') and not card.get('derived'):
            team['deck'].pop(index)
            add_hand(c.state, c.side, card, reveal=reveal)
            if not reveal:
                owner_name = hero(c.state, c.side, owner)['name']
                log(c.state, f'{name(c.state, c.side)}抽取了{owner_name}的 1 张牌。', 'draw',
                    side=c.side, actor=f'{c.side}:player', private_side=c.side,
                    private_card=public_card(effective_card(c.state, c.side, card)))
            return


def grant_seed(c, limit=3, *, reason=None):
    from app.modules.card_game.content.duel_v2.registry import KITS
    from app.modules.card_game.engine.duel_v2.state import add_hand, card_instance
    mimic = (c.character.get('flags') or {}).get('mimic') or {}
    kit = KITS.get(mimic.get('id') or c.cid)
    card_id = getattr(kit, 'seed_card', None) if kit else None
    if not card_id:
        return
    team = c.state['sides'][c.side]
    key = getattr(kit, 'seed_key', f'{c.cid}_seed')
    gained = int(team.get(key) or 0)
    if gained >= limit:
        return
    team[key] = gained + 1
    card = card_instance(c.state, card_id, c.side)
    if mimic.get('id'):
        card['original_character_id'] = card['character_id']
        card['character_id'] = c.cid
    add_hand(c.state, c.side, card, reason=reason)


def perm_plus(c, atk=1, hp=1):
    h = c.character
    from app.modules.card_game.engine.duel_v2.equipment import grow_permanently
    grow_permanently(h, atk, hp)
    changes = []
    if atk:
        changes.append(f'永久攻击 +{atk}')
    if hp:
        changes.append(f'生命与生命上限 +{hp}')
    if changes:
        c.history(f"{h['name']}获得{'、'.join(changes)}。")


def revive_self(c):
    revive_character(c, c.side, c.cid)


def revive_character(c, side, cid, *, temporary=False):
    from app.modules.card_game.engine.duel_v2.state import clear_harmony_source, hero, log
    h = hero(c.state, side, cid)
    if h['hp'] > 0 and not h.get('down_turns'):
        return
    clear_harmony_source(c.state, side, cid)
    h['down_turns'] = 0
    h['hp'] = max(int(h['max_hp']), 1)
    if temporary:
        h.setdefault('flags', {})['temp_revive'] = True
    log(c.state, f"{h['name']}被复活。", 'revive', side=side, source=c.cid,
        actor=f'{c.side}:{c.cid}', target=f'{side}:{cid}')


def deal_combat_followup(c, extra=0):
    from app.modules.card_game.content.duel_v2 import CARDS
    from app.modules.card_game.content.duel_v2.registry import KITS
    kit = KITS.get(c.cid)
    amount = extra
    reasons = []
    if c.character.get('awakened'):
        amount += 1
        reasons.append('终结')
    pending = int(c.character.get('flags', {}).pop('next_followup', 0) or 0)
    amount += pending
    if pending:
        reasons.append('预备的追击')
    if kit and c.card and c.card.get('card_id') == getattr(kit, 'payoff_card', None):
        amount += 1
        reasons.append(f"「{c.card['name']}」")
    followup_shape = getattr(kit, 'followup_shape', None) if kit else None
    if followup_shape and c.character.get('shape') == followup_shape:
        amount += 2
        reasons.append(f"「{CARDS[c.character['shape']]['name']}」")
    if amount > 0:
        reason = f"{c.character['name']}的{'、'.join(reasons)}触发追击" if reasons else None
        target = c.operation.get('last_attack_target')
        if target:
            c.followup(target, amount, reason=reason)


def scale_battle_frame(c, attack, shield):
    from app.modules.card_game.content.duel_v2.registry import KITS
    kit = KITS.get(c.cid)
    scale = getattr(kit, 'battle_frame', None)
    if scale:
        attack, shield = scale(c, attack, shield)
    if not kit or c.character.get('shape') != getattr(kit, 'scale_shape', None):
        return attack, shield
    played = int(c.state['sides'][c.side].get(getattr(kit, 'seed_play_key', ''), 0) or 0)
    return attack + played, shield + played


def on_first_turn_seed(c):
    if c.state['sides'][c.side].get('turn_count') == 1:
        grant_seed(c, reason=f"首个己方回合开始，{c.character['name']}的异能触发")


def on_game_start_seed(c):
    grant_seed(c, reason=f"对局开始，{c.character['name']}的异能触发")


def on_rout_seed(c, **_data):
    team = c.state['sides'][c.side]
    if c.character['hp'] > 0 and team.get('front') == c.cid:
        grant_seed(c, reason=f"击退对方异能者，{c.character['name']}的异能触发")


def negative_kinds(character, state=None, side=None) -> list[str]:
    flags = character.get('flags') or {}
    kinds = []
    if flags.get('burn'):
        kinds.append('burn')
    if flags.get('star'):
        kinds.append('star')
    if flags.get('nightmare'):
        kinds.append('nightmare')
    if flags.get('slow'):
        kinds.append('slow')
    if flags.get('brand'):
        kinds.append('brand')
    if state is not None and side and character.get('id') and state['sides'][side].get('front') == character['id']:
        from app.modules.card_game.engine.duel_v2.state import front_debuff
        zone = front_debuff(state, side)
        if zone.get('delay') and 'slow' not in kinds:
            kinds.append('slow')
        if zone.get('burn') and 'burn' not in kinds:
            kinds.append('burn')
        if zone.get('star') and 'star' not in kinds:
            kinds.append('star')
        if zone.get('weave'):
            kinds.append('weave')
        if zone.get('stain'):
            kinds.append('stain')
    return kinds


def add_brand(character, amount=1, cap=3) -> int:
    flags = character.setdefault('flags', {})
    before = int(flags.get('brand') or 0)
    flags['brand'] = min(cap, before + amount)
    return flags['brand'] - before


def grant_nightmare(character, layers, cap=3) -> int:
    flags = character.setdefault('flags', {})
    before = int(flags.get('nightmare') or 0)
    flags['nightmare'] = min(cap, max(before, layers))
    return flags['nightmare'] - before


def settle_nightmare_on(character, layers=1) -> int:
    flags = character.get('flags') or {}
    have = int(flags.get('nightmare') or 0)
    take = have if layers is None else min(have, layers)
    if take <= 0:
        return 0
    remain = have - take
    if remain:
        flags['nightmare'] = remain
    else:
        flags.pop('nightmare', None)
    return take
