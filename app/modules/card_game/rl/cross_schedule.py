"""Explicit initiative coverage, independent of seeds, workers and engine seats."""
from itertools import combinations

ROSTER = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')


def learning_schedule(roster=ROSTER):
    # Fixed seats within a pair; initiative is explicitly swapped in this round.
    return [dict(left=a, right=b, first=first)
            for a, b in combinations(roster, 2) for first in ('a', 'b')]


def matrix_schedule(games_per_cell, roster=ROSTER):
    if type(games_per_cell) is not int or games_per_cell < 2 or games_per_cell % 2:
        raise ValueError('Even games per cell >= 2 required for seat balance')
    # Replicate outermost: all cells get a first sample before any gets a second.
    return [dict(row=row, column=column,
                 left=row if replica % 2 == 0 else column,
                 right=column if replica % 2 == 0 else row,
                 first='a' if replica % 2 == 0 else 'b', replica=replica)
            for replica in range(games_per_cell) for row in roster for column in roster]


def actual_cell(game):
    if game['first'] not in ('a', 'b'):
        raise ValueError('Invalid first side')
    return (game['left'], game['right']) if game['first'] == 'a' else (game['right'], game['left'])


def coverage(games, planned, roster=ROSTER):
    cells = {(a, b): dict(row=a, column=b, planned=0, attempts=0, completed=0,
                         first_wins=0, second_wins=0, draws=0, errors=0, truncated=0)
             for a in roster for b in roster}
    for game in planned:
        cells[actual_cell(game)]['planned'] += 1
    seen = set()
    for game in games:
        if game['id'] in seen:
            raise ValueError('Duplicate game identity')
        seen.add(game['id'])
        cell = cells[actual_cell(game)]
        cell['attempts'] += 1
        if not game['complete']:
            cell['errors' if game.get('error') else 'truncated'] += 1
            continue
        cell['completed'] += 1
        winner = game.get('winner')
        if winner not in ('a', 'b', 'draw'):
            raise ValueError('Missing terminal winner')
        cell['draws'] += winner == 'draw'
        cell['first_wins'] += winner == game['first']
        cell['second_wins'] += winner in ('a', 'b') and winner != game['first']
    for cell in cells.values():
        cell['first_winrate'] = cell['first_wins'] / cell['completed'] if cell['completed'] else None
        cell['complete'] = (cell['planned'] > 0 and cell['completed'] == cell['planned']
                            and not cell['errors'] and not cell['truncated'])
    return list(cells.values())
