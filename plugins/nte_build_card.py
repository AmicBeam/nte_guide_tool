#!/usr/bin/env python3
"""Render one saved NTE character build as a PNG, without importing the website."""
from __future__ import annotations
import argparse
from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import urllib.parse
import urllib.request

WIDTH = 1200
BACKGROUND = '#101923'
PANEL = '#192632'
INK = '#e9f0f5'
MUTED = '#9eafbf'
ACCENT = '#c9f47a'


def choose_character(payload, character_id, role_id=None):
    profiles = payload.get('data', payload) if isinstance(payload, dict) else payload
    if isinstance(profiles, dict) and isinstance(profiles.get('characters'), list):
        profiles = [profiles]
    if not isinstance(profiles, list) or not all(isinstance(p, dict) for p in profiles):
        raise ValueError('输入须为本站 me 返回值或单个已验证归属的游戏账号资料。')
    if role_id:
        profiles = [profile for profile in profiles if str(profile.get('role_id')) == str(role_id)]
    if not profiles:
        raise ValueError('没有已同步的游戏账号，请先在本站同步角色资料。')
    if len(profiles) != 1:
        raise ValueError('已保存多个游戏账号，请先选择 role_id。')
    characters = [character for character in profiles[0].get('characters', [])
                  if str(character.get('id')) == str(character_id)]
    if len(characters) != 1:
        raise ValueError('此账号没有该角色的已保存资料，请先同步或确认角色ID。')
    return profiles[0], characters[0]


def display(value):
    return '未提供' if value is None or value == '' else str(value)[:180]


def card_content(profile, character, reference=None):
    """Only select display fields; never stringify whole account/snapshot payloads."""
    reference = reference or {}
    labels = {'melee': '普通攻击', 'skill': '变轨攻击', 'ultraskill': '极轨终结',
              'qte': '援护技', 'passive': '天赋', 'city': '都市技能', 'peculiarity': '特性'}
    abilities = {str(ability['id']).casefold(): ability for ability in reference.get('abilities', [])}
    skills = []
    for skill in character.get('skills', [])[:24]:
        ability = abilities.get(str(skill.get('id')).casefold(), {})
        name = skill.get('name') or ability.get('name') or skill.get('id')
        kind = labels.get(ability.get('type'), '都市技能' if skill.get('group') == 'citySkills' else '技能')
        skills.append((f'{kind} · {display(name)}', f'Lv.{display(skill.get("level"))}'))
    equipment = character.get('equipment') or {}
    fork = equipment.get('fork') or {}
    suit = equipment.get('suit') or {}
    def properties(items):
        return [(display(item.get('name') or item.get('id')), display(item.get('value')))
                for item in (items or [])[:24] if isinstance(item, dict)]
    pieces = []
    for category in ('core', 'pie'):
        for item in (suit.get(category) or [])[:16]:
            pieces.append({'title': display(item.get('name') or item.get('id')),
                           'subtitle': f'Lv.{display(item.get("level"))}',
                           'properties': properties(item.get('main_properties')) + properties(item.get('properties'))})
    return {'name': display(character.get('name') or reference.get('name')),
            'id': display(character.get('id')), 'level': display(character.get('level')),
            'awaken': display(character.get('awaken_level')), 'resonance': display(character.get('resonance_level')),
            'account': display(profile.get('role_name')), 'updated': display(profile.get('updated_at')),
            'element': reference.get('element_name', ''), 'properties': properties(character.get('properties')),
            'skills': skills, 'fork': {'title': display(fork.get('name') or fork.get('id')),
                'subtitle': f'等级 {display(fork.get("alev"))}  /  突破 {display(fork.get("blev"))}  /  混频 {display(fork.get("slev"))}',
                'properties': properties(fork.get('properties'))},
            'suit_name': display(suit.get('name')), 'pieces': pieces}


def font_path():
    candidates = [os.environ.get('NTE_GUIDE_FONT', ''), '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Medium.ttc', '/System/Library/Fonts/Supplemental/Songti.ttc',
        'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc']
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise ValueError('未找到中文字体，请设置 NTE_GUIDE_FONT 指向中文字体文件。')


def render_build_card(profile, character, output, reference=None, avatar=None):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    content = card_content(profile, character, reference)
    font = font_path()
    fonts = {size: ImageFont.truetype(font, size) for size in (17, 19, 22, 24, 28, 38, 52)}
    panels = []
    if content['properties']:
        panels.append(('角色面板', content['properties'], 3, None))
    if content['skills']:
        panels.append(('技能 · 同步展示等级', content['skills'], 2, None))
    panels.append(('已装备弧盘', content['fork']['properties'], 2, content['fork']))
    for piece in content['pieces']:
        panels.append(('空幕卡带', piece['properties'], 2, piece))
    if not content['pieces']:
        panels.append(('空幕卡带', [], 2, {'title': '未提供已装备卡带', 'subtitle': ''}))
    heights = [110 + (90 if intro else 0) + max(1, (len(rows) + columns - 1) // columns) * 76
               for title, rows, columns, intro in panels]
    height = 490 + sum(heights) + 18 * len(panels) + 110
    canvas = Image.new('RGB', (WIDTH, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    def text(value, x, y, size=24, color=INK, width=None, lines=1):
        value = str(value).replace('\r', ' ').replace('\n', ' ')[:220]
        wrapped, line = [], ''
        for char in value:
            if width and line and draw.textlength(line + char, font=fonts[size]) > width:
                wrapped.append(line)
                line = ''
                if len(wrapped) >= lines:
                    wrapped[-1] = wrapped[-1][:-1] + '…'
                    break
            line += char
        else:
            if line:
                wrapped.append(line)
        for index, line in enumerate(wrapped or ['']):
            draw.text((x, y + index * (size + 6)), line, font=fonts[size], fill=color)
    def panel(x, y, width, height):
        draw.rounded_rectangle((x, y, x + width, y + height), radius=20, fill=PANEL, outline='#304251', width=1)
    text('NTE TOOLS  /  我的角色配装', 48, 30, 19, ACCENT)
    draw.line((48, 75, WIDTH - 48, 75), fill='#304251', width=1)
    panel(48, 105, 270, 270)
    if avatar is not None:
        picture = ImageOps.fit(avatar.convert('RGB'), (270, 270))
        mask = Image.new('L', (270, 270), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, 270, 270), radius=20, fill=255)
        canvas.paste(picture, (48, 105), mask)
    else:
        text('角色配装', 82, 210, 38, MUTED)
    text(content['name'], 354, 110, 52, width=790)
    text(f'{content["element"]}  ·  ID {content["id"]}', 354, 182, 22, MUTED)
    text('已保存的当前练度', 354, 230, 19, ACCENT)
    for index, (label, value) in enumerate([('角色等级', content['level']), ('觉醒', content['awaken']), ('混频', content['resonance'])]):
        x = 354 + index * 252
        text(label, x, 276, 19, MUTED)
        text(value, x, 310, 38)
    text(f'游戏账号：{content["account"]}', 48, 405, 22, width=1104)
    updated = content['updated']
    try:
        stamp = datetime.fromisoformat(updated.replace('Z', '+00:00'))
        from datetime import timedelta, timezone
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        updated = stamp.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M 北京时间')
    except ValueError:
        pass
    text(f'同步时间：{updated}', 48, 446, 19, MUTED, width=1104)
    y = 490
    for (title, rows, columns, intro), panel_height in zip(panels, heights):
        panel(48, y, 1104, panel_height)
        text(title, 72, y + 20, 22, ACCENT)
        row_y = y + 65
        if intro:
            text(intro['title'], 72, row_y, 28, width=1056)
            text(intro['subtitle'], 72, row_y + 40, 19, MUTED, width=1056)
            row_y += 90
        if not rows:
            text('属性未提供', 72, row_y, 22, MUTED)
        for index, (name, value) in enumerate(rows):
            column_width = 1056 // columns
            x = 72 + index % columns * column_width
            top = row_y + index // columns * 76
            text(name, x, top, 19, MUTED, width=column_width - 28)
            text(value, x, top + 30, 24, INK, width=column_width - 28)
        y += panel_height + 18
    text('仅展示本站已保存快照；技能可能包含额外等级，未进行配装评分。', 48, y + 16, 17, MUTED, width=1104)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format='PNG')
    target.chmod(0o600)
    return {'image_path': str(target.resolve()), 'width': WIDTH, 'height': height}


def public_reference(base_url, character_id):
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('请配置有效的本站根地址。')
    class NoRedirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ValueError('资料请求不接受重定向。')
    opener = urllib.request.build_opener(NoRedirects())
    def fetch(path):
        with opener.open(base_url.rstrip('/') + urllib.parse.quote(path, safe='/'), timeout=10) as response:
            data = response.read(8 * 1024 * 1024 + 1)
            if len(data) > 8 * 1024 * 1024:
                raise ValueError('资料超过图片读取上限。')
            return data
    reference = json.loads(fetch('/api/guide/characters/' + urllib.parse.quote(str(character_id), safe='')))['data']
    avatar = None
    path = reference.get('assets', {}).get('avatar', '')
    if path.startswith('/static/') and not any(part in ('.', '..') for part in path.split('/')):
        from PIL import Image
        avatar = Image.open(BytesIO(fetch(path)))
        if avatar.width > 4096 or avatar.height > 4096:
            raise ValueError('角色图片尺寸超过读取上限。')
        avatar.load()
    return reference, avatar


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Saved profile JSON file, or - for stdin')
    parser.add_argument('--character', required=True, help='Character ID from the saved profile')
    parser.add_argument('--role-id')
    parser.add_argument('--output', required=True, help='Trusted runtime output PNG path')
    parser.add_argument('--base-url', default=os.environ.get('NTE_GUIDE_URL', 'http://127.0.0.1:5001'))
    parser.add_argument('--reference', help='Optional local public character-detail JSON; skips HTTP')
    args = parser.parse_args(argv)
    try:
        source = sys.stdin.read(4 * 1024 * 1024 + 1) if args.input == '-' else Path(args.input).read_text(encoding='utf-8')
        if len(source) > 4 * 1024 * 1024:
            raise ValueError('个人资料文件过大。')
        payload = json.loads(source)
        profile, character = choose_character(payload, args.character, args.role_id)
        if args.reference:
            reference = json.loads(Path(args.reference).read_text(encoding='utf-8'))
            reference = reference.get('data', reference)
            avatar = None
        else:
            try:
                reference, avatar = public_reference(args.base_url, character['id'])
            except (OSError, ValueError, KeyError):
                # Saved values remain renderable when the public site is temporarily unavailable.
                reference, avatar = {}, None
        result = render_build_card(profile, character, args.output, reference, avatar)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        print('配装图片生成失败：请确认账号/角色选择、中文字体和Pillow依赖，以及已同步的资料格式。', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
