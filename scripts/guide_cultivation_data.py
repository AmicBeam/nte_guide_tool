"""Build offline character cultivation parameters from data tables."""
from __future__ import annotations
import json
import subprocess
from guide_skill_details import table_rows


def load_payload(path, directory=None):
    if directory is not None:
        return json.loads((directory / path.rsplit('/', 1)[-1]).read_text(encoding='utf-8'))
    response = subprocess.run(['gh', 'api', 'repos/mc-ctrl/NTEData/contents/' + path,
        '-H', 'Accept: application/vnd.github.raw+json'], check=True, capture_output=True)
    return json.loads(response.stdout)


def cultivation_data(characters, items, breakthroughs):
    characters, items, breakthroughs = map(table_rows, (characters, items, breakthroughs))
    books = []
    for index in range(1, 4):
        identifier = f'CharacterUpMaterial_lv{index}'
        row = items[identifier]
        effect = row['ElementData']
        currency, amount = effect['CostGold'].split(':')
        if currency.casefold() != 'gold' or effect['EXP'] <= 0 or int(amount) < 0:
            raise ValueError('Invalid experience book')
        books.append({'id': identifier, 'name': row['ItemName']['LocalizedString'],
                      'exp': effect['EXP'], 'cost_gold': int(amount)})
    breakthroughs = {key.casefold(): value for key, value in breakthroughs.items()}
    caps = {}
    for identifier, row in characters.items():
        pack = row['ElementData']['BreakthroughPackId']
        caps[identifier] = [breakthroughs[f'{pack}_{stage}'.casefold()]['MaxCharacterLevel'] for stage in range(7)]
    return {'exp_books': books, 'level_caps': caps}


def load_cultivation_data(directory=None):
    paths = ['DataTable/Character/DT_Character.json', 'DataTable/Inventory/DT_ItemConfig.json',
             'DataTable/Character/DT_CharacterBreakthroughDataTable.json']
    return cultivation_data(*(load_payload(path, directory) for path in paths))
