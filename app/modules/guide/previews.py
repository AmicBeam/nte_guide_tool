"""Offline announcements, kept separate from complete playable character data."""
import json
from pathlib import Path


def characters():
    rows = json.loads((Path(__file__).parent / 'data/character_previews.json').read_text(encoding='utf-8'))
    for row in rows:
        row['status'] = 'preview'
        row['element_icon'] = f'/static/images/elements/{row["element_name"]}.png'
        rank = {4: 'A', 5: 'S'}[row['rarity']]
        row['rarity_icon'] = f'/static/images/rarities/YH_UI_star{rank}.webp'
    return rows
