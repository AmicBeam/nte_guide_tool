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


def parameter_text(text, parameters=()):
    bindings = {str(row['id_value']): row for row in parameters}
    output, previous = [], 0
    for match in re.finditer(r'\{(\d+)\}', text):
        output.append(escape(text[previous:match.start()]))
        row = bindings.get(match[1])
        if row and row.get('values'):
            selected = min(10, len(row['values']))
            identifier = f'guide-term-{next(_IDS)}'
            output.append(Markup('<span class="guide-term-wrap" data-linked-ability="{}" data-linked-values="{}">'
                '<button type="button" class="guide-term" aria-describedby="{}" aria-expanded="false">'
                '<span data-linked-value>{}</span></button><span class="guide-term-tip" id="{}" role="tooltip" hidden>'
                '<strong>{}</strong><span>依据技能：{}</span><span>当前技能等级：<span data-linked-level>{}</span> 级</span>'
                '<span>当前倍率：<span data-linked-value>{}</span>%</span><span>1 级倍率：{}%</span></span></span>').format(
                    row['ability_id'], json.dumps(row['values']), identifier, row['values'][selected - 1],
                    identifier, row['formula'], row['ability_name'], selected, row['values'][selected - 1], row['values'][0]))
        else:
            output.append(escape(match[0]))
        previous = match.end()
    output.append(escape(text[previous:]))
    return Markup('').join(output)


def styled_text(run, colors, parameters=()):
    text = parameter_text(run.get('text', ''), parameters)
    style = run.get('style', '')
    color = colors.get(style, '')
    if re.fullmatch(r'[\da-fA-F]{6}', color):
        text = Markup('<span style="color:#{}">{}</span>').format(color, text)
    if style == 'italic':
        text = Markup('<em>{}</em>').format(text)
    return text


def description_html(text, character_id, parameters=()):
    data = description_data()
    character = data['characters'].get(str(character_id), {})
    segments = character.get('descriptions', {}).get(text)
    if segments is None:
        return parameter_text(text, parameters)
    output = []
    for run in segments:
        content = styled_text(run, data['colors'], parameters)
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
