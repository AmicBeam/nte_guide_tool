"""Bounded paired Python-oracle evaluation; optional post-training play logs."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, acting_side, legal_actions, observe
from app.modules.card_game.rl.policies import VisibleEngineRulePolicy
from .catalog import preset_by_id
from .state import empty_state
from .resident_obs import resident_observe, SCHEMA
from ..rule_ir.pack_row import pack_python_row, bind_gpu_rows
from ..rule_ir.compiled_oracle import map_python_action


def summarize_games(games):
    def summary(items):
        completed = [g for g in items if g['status'] == 'finished']
        wins = sum(g['winner'] == g['learner_side'] for g in completed)
        draws = sum(g['winner'] == 'draw' for g in completed)
        n = len(completed)
        p = wins / n if n else 0.0
        z = 1.959963984540054
        if n:
            den = 1 + z*z/n
            center = (p + z*z/(2*n))/den
            spread = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
            interval = [center-spread, center+spread]
        else:
            interval = [0.0, 1.0]
        return dict(wins=wins, losses=n-wins-draws, draws=draws, completed=n,
                    attempted=len(items), truncated=len(items)-n, win_rate=p, wilson95=interval)
    first = summary([g for g in games if g['learner_first']])
    second = summary([g for g in games if not g['learner_first']])
    total = summary(games)
    return dict(overall=total, first=first, second=second,
                equal_weight_win_rate=(first['win_rate']+second['win_rate'])/2,
                complete=bool(games) and total['truncated']==0 and first['completed']==second['completed'])


@torch.no_grad()
def evaluate_resident(net, backend, *, deck_id, pairs=64, seed_base=900_000_000,
                      max_seconds=600, max_steps=400, log_path=None, checkpoint='in_memory', run_id='eval'):
    if pairs < 1 or max_seconds <= 0 or max_steps < 1:
        raise ValueError('Evaluation limits must be positive')
    device = next(net.parameters()).device
    deadline = time.monotonic()+max_seconds
    deck = preset_by_id(deck_id)
    rule = VisibleEngineRulePolicy()
    state_t = empty_state(1, 'cpu')
    rows_t = bind_gpu_rows(state_t)
    was_training = net.training
    net.eval()
    games = []
    stream = None
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open('w', encoding='utf-8')
    def emit(record):
        if stream:
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
    try:
        emit(dict(type='metadata', schema='card_play_eval_v1', observation_schema=SCHEMA,
                  run_id=run_id, deck=deck_id, checkpoint=str(checkpoint), rule_ai='VisibleEngineRulePolicy',
                  rule_identity=backend.identity(), pairs=pairs, max_steps=max_steps, max_seconds=max_seconds,
                  card_names={cid: card['name'] for cid, card in CARDS.items()}))
        for pair in range(pairs):
            for first in ('a','b'):
                if time.monotonic() >= deadline:
                    break
                seed = seed_base+pair
                gid = f'{run_id}:{deck_id}:{seed}:{first}'
                state = new_game(seed=seed, first_side=first, skip_mulligan=True, decks={'a':deck,'b':deck})
                info = dict(game_id=gid, seed=seed, deck=deck_id, learner_side='a',
                            first_side=first, learner_first=first=='a', checkpoint=str(checkpoint))
                emit(dict(type='game_start', **info))
                steps = 0
                error = None
                try:
                    for steps in range(max_steps):
                        if state['phase']=='finished' or time.monotonic()>=deadline:
                            break
                        side = acting_side(state)
                        actions = [a for a in legal_actions(state,side) if a['type']!='concede']
                        if not actions:
                            raise RuntimeError('Live oracle has no legal actions')
                        if side=='a':
                            row = backend.step_lists([pack_python_row(state)],[-1])[0]
                            mapped = [map_python_action(row,state,a) for a in actions]
                            rows_t.copy_(torch.tensor([row],dtype=torch.int32))
                            obs = resident_observe(state_t)
                            logits,_ = net(*(v.to(device) for v in obs))
                            scores = logits[0,torch.tensor(mapped,device=device)]
                            action = actions[int(scores.argmax())]
                        else:
                            action = actions[rule(observe(state,side),tuple(actions))]
                        previous_seq = max((e.get('seq',0) for e in state.get('events',[])), default=0)
                        state = apply_action(state,side,action)
                        if stream:
                            for event in state.get('events',[]):
                                if event.get('seq',0)>previous_seq and event.get('type')=='play':
                                    card = event.get('card') or {}
                                    emit(dict(type='successful_play', game_id=gid, seq=event['seq'],
                                              turn=state['turn'], side=event.get('side'),
                                              card_id=card.get('card_id') or card.get('id'),
                                              instance_id=card.get('instance_id'), actor=event.get('actor'),
                                              action_id=event.get('action_id')))
                    status = 'finished' if state['phase']=='finished' else 'truncated'
                except Exception as exc:
                    error = repr(exc)
                    status = 'error'
                game = dict(**info, status=status, winner=state.get('winner') if status=='finished' else None,
                            reason=state.get('reason') if status=='finished' else ('error' if error else 'budget'),
                            steps=steps, error=error)
                games.append(game)
                emit(dict(type='game_end',**game))
                if stream:
                    stream.flush()
                if error:
                    raise RuntimeError(f'Evaluation failed: {game}')
            if time.monotonic()>=deadline:
                break
        report = summarize_games(games)
        report.update(games=games, requested_games=pairs*2, deck=deck_id)
        report['complete'] = report['complete'] and len(games)==pairs*2
        return report
    finally:
        net.train(was_training)
        if stream:
            stream.close()
