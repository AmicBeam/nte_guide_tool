"""Portable, commentable plain-text builds; no persistence or battle rules."""
from collections import Counter
import shlex

from app.modules.card_game.content import duel_v2 as content


def parse(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 16384:
        raise ValueError('请提供不超过 16384 字符的构筑文本。')
    deck = {'id': 'import', 'name': '', 'character_ids': [], 'card_ids': []}
    version = False
    for number, line in enumerate(text.lstrip('\ufeff').splitlines(), 1):
        try:
            fields = shlex.split(line, comments=True)
            if not fields:
                continue
            if fields in (['异能对决构筑', '1'], ['异象对决构筑', '1']) and not version:
                version = True
            elif fields[0] == '名称' and len(fields) == 2 and not deck['name']:
                deck['name'] = fields[1]
            elif fields[0] == '角色' and len(fields) == 2:
                deck['character_ids'].append(fields[1])
            elif fields[0] == '卡牌' and len(fields) == 3 and fields[2] in ('1', '2'):
                deck['card_ids'].extend([fields[1]] * int(fields[2]))
            else:
                raise ValueError('格式错误；卡牌行应为「卡牌 编号 1或2」。')
        except ValueError as exc:
            raise ValueError(f'第 {number} 行：{exc}') from exc
    if not version:
        raise ValueError('缺少或不支持格式版本：异能对决构筑 1。')
    if not deck['name'].strip() or len(deck['name']) > 16:
        raise ValueError('构筑名称须为 1–16 个字符。')
    return content.validate_deck(deck)


def format_build(deck):
    deck = content.validate_deck(deck)
    characters = {item['id']: item for item in content.get_catalog()['characters']}
    cards = {item['id']: item for item in content.get_catalog()['cards']}
    lines = ['# 异能对决构筑；# 后为注释，名称含空格或 # 时保留引号',
             '异能对决构筑 1', '名称 ' + shlex.quote(deck['name'])]
    counts = Counter(deck['card_ids'])
    for character_id in deck['character_ids']:
        lines.append(f"角色 {character_id} # {characters[character_id]['name']}")
        for card_id, count in counts.items():
            if cards[card_id]['character_id'] == character_id:
                lines.append(f"卡牌 {card_id} {count} # {cards[card_id]['name']}")
    return '\n'.join(lines)
