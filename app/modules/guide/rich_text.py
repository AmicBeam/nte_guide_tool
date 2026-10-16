"""Render inert, allowlisted description segments; never accept upstream HTML."""
from functools import lru_cache
from itertools import count
import json
from pathlib import Path
import re

from markupsafe import Markup, escape

_IDS = count(1)


@lru_cache(maxsize=1)
def description_data():
    return json.loads((Path(__file__).parent / 'data/character_rich_text.json').read_text(encoding='utf-8'))


def styled_text(run, colors):
    text = escape(run.get('text', ''))
    style = run.get('style', '')
    color = colors.get(style, '')
    if re.fullmatch(r'[\da-fA-F]{6}', color):
        text = Markup('<span style="color:#{}">{}</span>').format(color, text)
    if style == 'italic':
        text = Markup('<em>{}</em>').format(text)
    return text


def description_html(text, character_id):
    data = description_data()
    character = data['characters'].get(str(character_id), {})
    segments = character.get('descriptions', {}).get(text)
    if segments is None:
        return escape(text)
    output = []
    for run in segments:
        content = styled_text(run, data['colors'])
        noun = character.get('nouns', {}).get(run.get('noun'))
        if noun:
            identifier = f'guide-term-{next(_IDS)}'
            body = Markup('').join(styled_text(row, data['colors']) for row in noun['runs'])
            content = Markup('<span class="guide-term-wrap"><button type="button" class="guide-term" '
                'aria-describedby="{}" aria-expanded="false">{}</button><span class="guide-term-tip" '
                'id="{}" role="tooltip" hidden><strong>{}</strong><span>{}</span></span></span>').format(
                    identifier, content, identifier, noun['title'], body)
        output.append(content)
    return Markup('').join(output)
