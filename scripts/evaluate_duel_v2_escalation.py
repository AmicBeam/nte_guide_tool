#!/usr/bin/env python3
"""Paired rush mirror simulations using the website's visible rule AI, no training."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, acting_side, choose_action


def play(spec):
    seed, first, enabled = spec
    deck = next(d for d in STARTER_DECKS if d['id'] == 'weave-rush')
    state = new_game(seed=seed, decks={'a': deck, 'b': deck}, first_side=first,
                     escalation=enabled)
    counts = Counter()
    firsts = {}
    stage_ultimates = Counter()
    stage_turns = Counter()
    seen_turns = set()
    for actions in range(600):
        if state['phase'] == 'finished':
            break
        stage = '1-5' if state['turn'] < 6 else ('6-12' if state['turn'] < 13 else '13+')
        if state['phase'] == 'playing' and state['turn'] not in seen_turns:
            seen_turns.add(state['turn'])
            stage_turns[stage] += 1
        side = acting_side(state)
        action = choose_action(state, side)
        if action['type'] == 'ultimate':
            counts['ultimates'] += 1
            stage_ultimates[stage] += 1
            firsts.setdefault('ultimate', state['turn'])
        cursor = state['event_seq']
        state = apply_action(state, side, action)
        for event in state['events']:
            if event['seq'] <= cursor:
                continue
            if event['type'] == 'harmony' and 'before' in event and 'harmony' in event['before']:
                counts['harmonies'] += 1
                firsts.setdefault('harmony', state['turn'])
        # Retain recent events and the existing public-board cache, bounding deepcopy cost.
        state['events'] = state['events'][-200:]
    return dict(seed=seed, first=first, enabled=enabled, turns=state['turn'],
                winner=state['winner'], reason=state['reason'], truncated=state['phase'] != 'finished',
                actions=actions, stage_ultimates=dict(stage_ultimates), stage_turns=dict(stage_turns), ultimates=counts['ultimates'], harmonies=counts['harmonies'], firsts=firsts)


def summary(rows):
    turns = sorted(r['turns'] for r in rows)
    stages = {}
    for stage in ('1-5', '6-12', '13+'):
        windows = sum(r['stage_turns'].get(stage, 0) for r in rows)
        uses = sum(r['stage_ultimates'].get(stage, 0) for r in rows)
        stages[stage] = dict(uses=uses, action_windows=windows, per_window=uses/windows if windows else 0, per_game=uses/len(rows))
    return dict(stages=stages, games=len(rows), mean_turns=statistics.mean(turns), median_turns=statistics.median(turns),
                p90_turns=turns[max(0, int(len(turns) * .9 + .999) - 1)],
                first_win_rate=sum(r['winner'] == r['first'] for r in rows) / len(rows),
                mean_ultimates=statistics.mean(r['ultimates'] for r in rows),
                mean_harmonies=statistics.mean(r['harmonies'] for r in rows),
                reach={str(t): sum(r['turns'] >= t for r in rows) / len(rows) for t in (6, 13, 20)},
                endings=dict(Counter(r['reason'] for r in rows)), truncated=sum(r['truncated'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20260915)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1 or args.workers < 1:
        parser.error('pairs and workers must be positive')
    specs = [(args.seed+i, first, enabled) for i in range(args.pairs) for first in ('a', 'b') for enabled in (False, True)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(play, specs))
    baseline = [r for r in rows if not r['enabled']]
    accelerated = [r for r in rows if r['enabled']]
    delta = [b['turns'] - a['turns'] for b, a in zip(baseline, accelerated)]
    result = dict(seed=args.seed, pairs=args.pairs, deck='weave-rush', policy='engine.choose_action',
                  baseline=summary(baseline), escalation=summary(accelerated),
                  paired=dict(mean_turns_saved=statistics.mean(delta), faster=sum(d>0 for d in delta),
                              unchanged=sum(d==0 for d in delta), slower=sum(d<0 for d in delta)), matches=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'matches'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
