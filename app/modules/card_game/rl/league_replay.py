"""Bounded replay of terminal comparisons the current policy still gets wrong."""
from dataclasses import dataclass
import hashlib
import numpy as np
from .league_preferences import validate_pairs

@dataclass
class OptimizerBudget:
    branch_steps: int = 0
    ordinary_steps: int = 0

    def can_update_ordinary(self):
        return self.branch_steps >= 3*(self.ordinary_steps+1)

    def record_ordinary(self, steps):
        if steps not in (0,1) or self.branch_steps < 3*(self.ordinary_steps+steps):
            raise ValueError('Ordinary optimizer budget exceeded')
        self.ordinary_steps += steps

    def record_branch(self, steps):
        if steps not in (0,1):raise ValueError('Unexpected branch optimizer count')
        self.branch_steps += steps

class ErrorReplay:
    def __init__(self, capacity=256, ttl=5, max_uses=3, min_gap=2.5):
        if min(capacity,ttl,max_uses)<=0 or min_gap<=0:raise ValueError('Invalid replay bounds')
        self.capacity=capacity;self.ttl=ttl;self.max_uses=max_uses;self.min_gap=min_gap
        self.entries={}

    def add(self,x,c,selected,q,cycle,auxiliary=()):
        x=np.asarray(x,dtype=np.float32);c=np.asarray(c,dtype=np.float32)
        selected=tuple(int(i) for i in selected);q=np.asarray(q,dtype=np.float32)
        if (x.ndim!=1 or c.ndim!=2 or q.shape!=(len(selected),) or len(selected)<2
                or len(set(selected))!=len(selected) or min(selected)<0 or max(selected)>=len(c)
                or not all(np.isfinite(v).all() for v in (x,c,q))):
            raise ValueError('Invalid completed branch sample')
        auxiliary=validate_pairs(auxiliary,q)
        if np.ptp(q)<self.min_gap and not auxiliary:return False
        key=hashlib.sha256(x.tobytes()+c.tobytes()).hexdigest()
        self.entries[key]={'x':x.copy(),'c':c.copy(),'selected':selected,'q':q.copy(),
                           'auxiliary':auxiliary,'cycle':cycle,'uses':0,'last_used':-1}
        while len(self.entries)>self.capacity:
            self.entries.pop(min(self.entries,key=lambda k:self.entries[k]['cycle']))
        return True

    def take(self,score_fn,cycle):
        best=None;priority=0.
        for key,e in list(self.entries.items()):
            if cycle-e['cycle']>self.ttl or e['uses']>=self.max_uses:
                del self.entries[key];continue
            if e['last_used']==cycle:continue
            scores=np.asarray(score_fn(e['x'],e['c']))
            if scores.shape!=(len(e['c']),) or not np.isfinite(scores).all():raise ValueError('Invalid policy scores')
            chosen=int(scores.argmax())
            if chosen not in e['selected']:continue # Unassessed moves have unknown value.
            gap=float(e['q'].max()-e['q'][e['selected'].index(chosen)])
            weak=max([w for i,j,w,reason in e['auxiliary'] if j==e['selected'].index(chosen)]+[0.])
            if gap<self.min_gap and not weak:continue
            gap=gap if gap>=self.min_gap else weak
            p=np.exp(scores-scores.max());p/=p.sum()
            weight=gap*(.5+float(p[chosen]))
            if weight>priority:best=e;priority=weight
        if best is None:return None
        best['uses']+=1;best['last_used']=cycle
        return {**best,'priority':priority}

    def clear(self):
        self.entries.clear()
