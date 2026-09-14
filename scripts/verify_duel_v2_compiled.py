#!/usr/bin/env python3
"""Bounded compile/diff verification for the explicit compiled starter backend."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCENARIOS = (
    'opening_attack',
    'end_turn',
    'response',
    'harmony',
    'equipment',
    'battle_attack',
    'full_hand',
    'choice',
    'empty_deck',
    'simultaneous_death',
    'terminal_before_reset',
    'inactive_row',
    'both_seats',
)


def _report(ok: bool, payload: dict) -> int:
    payload['ok'] = bool(ok)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Compile and diff the frozen 创生预组 backend.')
    parser.add_argument('--deck', choices=('starter','weave-rush'), default='starter')
    parser.add_argument('--backend', choices=('native', 'cuda', 'auto'), default='auto')
    parser.add_argument('--dir', type=Path, default=None)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--trajectories', type=int, default=32)
    args = parser.parse_args(argv)

    from app.modules.card_game.rl.rule_ir.completeness import CompletenessError, validate_compiled_deck
    try:
        manifest = validate_compiled_deck(args.deck)
    except CompletenessError as exc:
        return _report(False, {
            'status': 'completeness_failed',
            'error': str(exc),
            'backend': None,
            'fallback_rows': None,
        })

    backend = args.backend
    if backend == 'auto':
        backend = 'native'
    availability = {
        'native_compiler': True,
        'cuda': False,
        'torch': False,
    }
    try:
        import torch
        availability['torch'] = True
        availability['cuda'] = bool(torch.cuda.is_available())
    except Exception:
        pass
    if backend == 'cuda' and not availability['cuda']:
        return _report(False, {
            'status': 'backend_unavailable',
            'error': 'CUDA compiled backend requested but CUDA is not available',
            'availability': availability,
            'support_manifest': manifest,
            'backend': 'cuda',
            'fallback_rows': 0,
        })

    from app.modules.card_game.rl.gpu_duel.catalog import preset_by_id
    STARTER_DECK = preset_by_id(args.deck)
    from app.modules.card_game.engine.duel_v2 import acting_side, apply_action, legal_actions, new_game
    from app.modules.card_game.rl.gpu_duel.compiled_backend import CompiledStarterBackend
    from app.modules.card_game.rl.rule_ir.compiled_oracle import python_packed_mechan, step_python
    from app.modules.card_game.rl.rule_ir.layout import ENGINE_NAME, ENGINE_VERSION, OFFSETS
    from app.modules.card_game.rl.rule_ir.completeness import compiled_rule_hash

    tmp = None
    directory = args.dir
    if backend == 'native' and directory is None:
        tmp = tempfile.TemporaryDirectory()
        directory = Path(tmp.name)
    env = None
    try:
        if backend == 'cuda':
            from app.modules.card_game.rl.gpu_duel.env import GpuDuelEnv
            env = GpuDuelEnv(
                2, device='cuda', deck=STARTER_DECK, matchup='mirror',
                compiled_backend=backend, compiled_dir=str(directory) if directory else None,
            )
            compiled = env.compiled
        else:
            compiled = CompiledStarterBackend(directory, backend='native', deck_id=args.deck)
    except Exception as exc:
        if tmp:
            tmp.cleanup()
        return _report(False, {
            'status': 'compile_failed',
            'error': str(exc),
            'availability': availability,
            'support_manifest': manifest,
            'backend': backend,
            'fallback_rows': 0,
        })

    failures = []

    try:
        import copy
        scenarios_run = []
        def check(state, action, name):
            packed, got = step_python(compiled, state, action)
            nxt = apply_action(state, acting_side(state), action)
            exp = python_packed_mechan(nxt, compiled)
            if packed[OFFSETS['error']]:
                failures.append(f'{name} compiled error {packed[OFFSETS["error"]]}')
            elif got != exp:
                failures.append(f'{name} snapshot mismatch for {action.get("type")}')
            else:
                scenarios_run.append(name)
            return nxt

        if args.deck == 'starter':
            state = new_game(seed=7, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
            frozen = copy.deepcopy(state)
            side = acting_side(state)
            attack = next(item for item in legal_actions(state, side) if item.get('type') == 'attack')
            packed, got = step_python(compiled, state, attack)
            inactive = compiled.launch.launch_lists([compiled.launch.launch_lists([__import__('app.modules.card_game.rl.rule_ir.pack_row', fromlist=['pack_python_row']).pack_python_row(frozen)], [-1])[0]], [-1])[0]
            nxt = apply_action(state, side, attack)
            if got != python_packed_mechan(nxt, compiled):
                failures.append('opening_attack snapshot mismatch')
            else:
                scenarios_run.append('opening_attack')
            if __import__('app.modules.card_game.rl.rule_ir.pack_row', fromlist=['mechan_from_row']).mechan_from_row(inactive) != python_packed_mechan(frozen, compiled):
                failures.append('inactive_row mutated')
            else:
                scenarios_run.append('inactive_row')

            state = new_game(seed=7, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
            check(state, {'type': 'end_turn'}, 'end_turn')

            state = new_game(seed=11, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
            foe = 'b' if state['active_side'] == 'a' else 'a'
            actor_side = state['active_side']
            n02 = next(c for c in state['sides'][actor_side]['deck'] + state['sides'][actor_side]['hand'] if c.get('card_id') == 'N02')
            state['sides'][foe]['hand'] = [copy.deepcopy(n02)]
            state['sides'][foe]['ap'] = 1
            state['sides'][foe]['front'] = 'nanali'
            attack = next(item for item in legal_actions(state, acting_side(state)) if item.get('type') == 'attack')
            check(state, attack, 'response')

            state = new_game(seed=13, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
            side = state['active_side']
            state['sides'][side]['front'] = 'zero'
            state['sides'][side]['characters']['zero']['harmony'] = 2
            attack = next(item for item in legal_actions(state, acting_side(state)) if item.get('type') == 'attack' and item['character_id'] == 'nanali')
            check(state, attack, 'harmony')

            def first_play(state, card_type):
                side = acting_side(state)
                plays = [item for item in legal_actions(state, side) if item.get('type') == 'play_card']
                for play in plays:
                    card = next(c for c in state['sides'][side]['hand'] if c['instance_id'] == play['card_id'])
                    if card.get('type') == card_type:
                        return play
                return None

            state = new_game(seed=21, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
            play = first_play(state, 'form')
            if play is None:
                failures.append('equipment card missing from opening legal actions')
            else:
                check(state, play, 'equipment')

            state = new_game(seed=21, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
            play = first_play(state, 'battle')
            if play is None:
                failures.append('battle_attack card missing from opening legal actions')
            else:
                check(state, play, 'battle_attack')

            state = new_game(seed=5, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
            side = state['active_side']
            template = state['sides'][side]['hand'][0]
            while len(state['sides'][side]['hand']) < 10:
                extra = dict(template)
                extra['instance_id'] = f'{side}-{1000 + len(state["sides"][side]["hand"])}'
                state['sides'][side]['hand'].append(extra)
            check(state, {'type': 'end_turn'}, 'full_hand')

            state = new_game(seed=3, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
            side = state['active_side']
            j01 = next(c for c in state['sides'][side]['deck'] + state['sides'][side]['hand'] if c.get('card_id') == 'J01')
            if j01 not in state['sides'][side]['hand']:
                state['sides'][side]['deck'] = [c for c in state['sides'][side]['deck'] if c['instance_id'] != j01['instance_id']]
                state['sides'][side]['hand'].append(j01)
            play = next(item for item in legal_actions(state, side) if item.get('type') == 'play_card' and item['card_id'] == j01['instance_id'])
            state = check(state, play, 'choice_play')
            if state.get('phase') == 'choice':
                choose = next(item for item in legal_actions(state, acting_side(state)) if item.get('type') == 'choose')
                check(state, choose, 'choice')

            state = new_game(seed=9, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
            other = 'b' if state['active_side'] == 'a' else 'a'
            state['sides'][other]['deck'] = []
            nxt = check(state, {'type': 'end_turn'}, 'empty_deck')
            if nxt.get('phase') != 'finished':
                failures.append('empty_deck did not finish before reset')
            else:
                scenarios_run.append('terminal_before_reset')

            state = new_game(seed=17, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
            side = state['active_side']
            foe = 'b' if side == 'a' else 'a'
            state['sides'][side]['front'] = 'nanali'
            state['sides'][foe]['front'] = 'nanali'
            state['sides'][side]['characters']['nanali'].update(hp=1, shield=0)
            state['sides'][foe]['characters']['nanali'].update(hp=1, shield=0)
            attack = next(item for item in legal_actions(state, acting_side(state)) if item.get('type') == 'attack' and item['character_id'] == 'nanali')
            check(state, attack, 'simultaneous_death')

            for first in ('a', 'b'):
                state = new_game(seed=8, skip_mulligan=True, first_side=first, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
                side = acting_side(state)
                action = next(item for item in legal_actions(state, side) if item.get('type') != 'concede')
                check(state, action, f'both_seats_{first}')
            scenarios_run.append('both_seats')

        import random
        from app.modules.card_game.rl.rule_ir.compiled_oracle import canonical_row, map_python_action, rebuild
        from app.modules.card_game.rl.rule_ir.pack_row import pack_python_row, legal_entries
        trajectory_steps = 0
        trajectories_finished = 0
        for seed in range(args.trajectories):
            state = new_game(seed=seed, skip_mulligan=True, decks={'a': STARTER_DECK, 'b': STARTER_DECK})
            row = rebuild(compiled, pack_python_row(state))
            rng = random.Random(seed)
            for step in range(args.steps):
                if state['phase'] == 'finished':
                    trajectories_finished += 1
                    break
                actions = [a for a in legal_actions(state, acting_side(state)) if a['type'] != 'concede']
                mapped = [map_python_action(row, state, a) for a in actions]
                if len(set(mapped)) != len(legal_entries(row)) or len(mapped) != len(set(mapped)):
                    failures.append(f'legal action mismatch seed={seed} step={step}')
                    break
                action = rng.choice(actions)
                index = map_python_action(row, state, action)
                row = compiled.step_lists([row], [index])[0]
                state = apply_action(state, acting_side(state), action)
                expected = rebuild(compiled, pack_python_row(state))
                trajectory_steps += 1
                got, want = canonical_row(row), canonical_row(expected)
                if got != want:
                    fields = [name for name in got if got[name] != want[name]]
                    failures.append(f'trajectory seed={seed} step={step} action={action} differs: {fields}')
                    break
            else:
                failures.append(f'trajectory {seed} exceeded step limit')
            if failures:
                break

        compiled.reset_lists([7, 8])
        identity = compiled.identity()
        fallback = int(identity.get('fallback_rows') or 0)
        if fallback:
            failures.append(f'compiled backend reported {fallback} fallback rows')
        missing = [name for name in (SCENARIOS if args.deck=='starter' else ()) if name not in scenarios_run and name not in ('choice_play',)]
        # choice is recorded as 'choice'; opening/inactive already recorded.
        payload = {
            'status': 'ok' if not failures else 'diff_failed',
            'backend': backend,
            'backend_identity': identity,
            'engine_name': ENGINE_NAME,
            'engine_version': ENGINE_VERSION,
            'rule_hash': compiled_rule_hash(args.deck),
            'support_manifest': manifest,
            'availability': availability,
            'fallback_rows': fallback,
            'scenarios': list(SCENARIOS) if args.deck=='starter' else [],
            'scenarios_run': scenarios_run,
            'errors': failures,
            'trajectory_steps': trajectory_steps,
            'trajectories_finished': trajectories_finished,
        }
        if missing:
            failures.append(f'missing scenarios: {missing}')
        return _report(not failures, payload)
    finally:
        compiled.close()
        if env is not None and env.compiled and env.compiled is not compiled:
            env.compiled.close()
        if tmp:
            tmp.cleanup()


if __name__ == '__main__':
    raise SystemExit(main())
