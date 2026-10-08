"""Fail-closed evaluation audit; never changes legal actions or policy scores."""


class InactivePolicyError(RuntimeError):
    pass


class BehaviorGate:
    def __init__(self, *, stop_games=8):
        if type(stop_games) is not int or stop_games<1:raise ValueError('Positive distinct-game stop threshold required')
        self.stop_games=stop_games;self._seen=set();self.cases=[];self._tripped=False

    def observe(self, records):
        for record in records:
            if not record.get('complete'):continue
            if not record.get('replay_verified') or record.get('error'):raise ValueError('Behavior audit requires verified normal games')
            gid=record['id']
            if gid in self._seen:raise ValueError('Duplicate behavior game')
            self._seen.add(gid)
            behavior=record.get('behavior')
            if not isinstance(behavior,dict) or set(behavior)!= {'a','b'}:raise ValueError('Missing physical behavior evidence')
            for side,row in behavior.items():
                required=('legal_play_decisions','legal_play_turns','manual_play_card_actions','ordinary_attack_actions','ultimate_actions')
                if any(type(row.get(k)) is not int or row[k]<0 for k in required):raise ValueError('Invalid physical behavior counters')
                if (row['legal_play_decisions']>=2 and row['legal_play_turns']>=2
                    and sum(row[k] for k in required[2:])==0):
                    self.cases.append(dict(id=gid,side=side,model=str(record.get('policies',{}).get(side)),**row))
        if len({x['id'] for x in self.cases})>=self.stop_games and not self._tripped:
            self._tripped=True
            raise InactivePolicyError('P0: verified entire-game inactivity despite legal plays in at least '+str(self.stop_games)+' distinct games')

    def report(self):
        games=len({x['id'] for x in self.cases})
        return dict(behavior_approved=not self.cases,verified_normal_games=len(self._seen),inactive_games=games,
                    inactive_sides=len(self.cases),stop_games=self.stop_games,p0_stop=games>=self.stop_games,
                    cases=self.cases,scope='Whole game; at least two legal-play decisions in two turns and no play/attack/ultimate; no score or reward changes')
