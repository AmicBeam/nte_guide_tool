"""Physical-game records for a cooperative evaluator, never hypothetical worlds.

Uses the existing full-report telemetry and public replay projection. Rules and
policy decisions are supplied by the owning evaluator, not reimplemented here.
"""
from collections import Counter
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path


def state_digest(state):
    return sha256(json.dumps(state,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def _save_gzip_new(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Do not overwrite a physical game record')
    temporary=path.with_name(path.name+'.pending')
    with temporary.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as zipped:
            zipped.write(json.dumps(payload,ensure_ascii=False).encode('utf-8'))
    if path.exists():raise FileExistsError('Physical game record already published')
    temporary.rename(path)


class PhysicalGameRecord:
    def __init__(self,state,*,journal_directory=None):
        from ..report_telemetry import GameTelemetry
        from ..setup_reward import ready
        self.opening=deepcopy(state);self.telemetry=GameTelemetry(state)
        self.actions=[];self.events=[];self.search=[];self.error=None
        self.setup_seen={side:ready(state,side) for side in ('a','b')}
        self.immune_metrics={side:Counter() for side in ('a','b')}
        self.immune_risks={side:[] for side in ('a','b')}
        self._last_digest=state_digest(state)
        self._finished=False
        self._behavior={side:Counter() for side in ('a','b')}
        self._legal_play_turns={side:set() for side in ('a','b')}
        self._journal=Path(journal_directory) if journal_directory is not None else None
        if self._journal is not None:
            _save_gzip_new(self._journal/'opening.json.gz',self.opening)

    def append(self,before,after,side,action,legal_actions,*,search_stats=None):
        """Record one applied physical operation, preserving the engine events."""
        from ..league_rollout import clean
        from ..fixed_lineup_training import track_immune_window
        from ..setup_reward import ready
        if self._finished:raise RuntimeError('Physical record already finalized')
        if side not in ('a','b') or action not in legal_actions:
            raise ValueError('Physical action is not a supplied legal decision')
        if state_digest(before)!=self._last_digest:
            raise ValueError('Physical trace skipped or replaced a state')
        counters=self._behavior[side]
        if any(a.get('type')=='play_card' for a in legal_actions):
            counters['legal_play_decisions']+=1;self._legal_play_turns[side].add(before['turn'])
        kind=action['type']
        if kind=='play_card':counters['manual_play_card_actions']+=1
        elif kind=='attack':counters['ordinary_attack_actions']+=1
        elif kind=='ultimate':counters['ultimate_actions']+=1
        self.telemetry.before(before,side,legal_actions)
        self.telemetry.after(before,after,action,side)
        track_immune_window(before,after,action,side,self.immune_metrics['a'],self.immune_risks['a'])
        swapped_before={**before,'sides':{'a':before['sides']['b'],'b':before['sides']['a']}}
        swapped_after={**after,'sides':{'a':after['sides']['b'],'b':after['sides']['a']}}
        track_immune_window(swapped_before,swapped_after,action,'b' if side=='a' else 'a',
                           self.immune_metrics['b'],self.immune_risks['b'])
        self.events.extend(deepcopy(after.get('events',[])))
        cleaned=clean(deepcopy(after));digest=state_digest(cleaned)
        row=dict(side=side,action=deepcopy(action),turn=before['turn'],state_sha256=digest)
        self.actions.append(row)
        if search_stats is not None:
            self.search.append(dict(deepcopy(search_stats),step=len(self.actions)-1,side=side))
        for owner in ('a','b'):self.setup_seen[owner]=self.setup_seen[owner] or ready(cleaned,owner)
        self._last_digest=digest
        if self._journal is not None:
            # One worker owns each game. A killed final write can be detected by
            # its missing newline; only complete lines are recoverable records.
            entry=dict(row,step=len(self.actions)-1,search_stats=deepcopy(search_stats))
            data=json.dumps(entry,ensure_ascii=False).encode('utf-8')+b'\n'
            with (self._journal/'actions.jsonl').open('ab') as stream:stream.write(data)

    def finish(self,state,job,output,*,public_replay=False):
        from ..league_rollout import clean
        from ...engine.duel_v2 import apply_action
        if self._finished:raise RuntimeError('Physical record already finalized')
        game_id=job.get('id')
        if (not isinstance(game_id,str) or not game_id or
            any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in game_id)):
            raise ValueError('Safe nonempty game id required')
        if state_digest(state)!=self._last_digest:raise ValueError('Final physical state mismatch')
        replayed=deepcopy(self.opening);verified=False;error=self.error
        try:
            for row in self.actions:
                replayed=clean(apply_action(replayed,row['side'],row['action']))
                if state_digest(replayed)!=row['state_sha256']:
                    raise ValueError('Physical replay action/state mismatch')
            verified=True
        except Exception as exc:error=repr(exc)
        complete=state['phase']=='finished' and error is None and verified
        behavior={side:{**{key:int(self._behavior[side][key]) for key in
            ('legal_play_decisions','manual_play_card_actions','ordinary_attack_actions','ultimate_actions')},
            'legal_play_turns':len(self._legal_play_turns[side])} for side in ('a','b')}
        payload=dict(job,behavior=behavior,complete=complete,winner=state.get('winner'),turn=state['turn'],
            error=error,replay_verified=verified,telemetry=self.telemetry.finish(state),
            actions=self.actions,events=self.events,opening_state=self.opening,
            search_stats=self.search,search_decisions=len(self.search),
            search_simulations=sum(row.get('simulations',0) for row in self.search),
            setup_achieved=self.setup_seen,
            passive_triggers={side:int(state['sides'][side]['characters'].get('zhenhong',{}).get('surplus_passive_triggers',0))
                              for side in ('a','b')},
            immune_window={side:dict(data) for side,data in self.immune_metrics.items()},
            immune_risks=self.immune_risks,gradient_updates=0)
        out=Path(output);_save_gzip_new(out/'raw'/(game_id+'.json.gz'),payload)
        if public_replay and complete:
            from ..offline_analysis import _public_payload
            final=deepcopy(state);final['events']=self.events
            replay=_public_payload(self.opening,final,game_id=game_id)
            _save_gzip_new(out/'replays'/(game_id+'.json.gz'),replay)
        self._finished=True
        return {key:value for key,value in payload.items()
                if key not in ('telemetry','actions','events','opening_state','search_stats')}
