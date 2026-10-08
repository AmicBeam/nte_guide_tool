"""Device-resident terminal counters; no action logs and no per-step CPU copies."""


class TerminalCounters:
    def __init__(self, n, device):
        import torch
        self.counts = torch.zeros(12, dtype=torch.int64, device=device)
        self.recorded = torch.zeros(n, dtype=torch.bool, device=device)

    def record(self, state, learner, assignment):
        import torch
        done = state.phase == 3
        fresh = done & ~self.recorded
        outcome = torch.where(state.winner == learner, 0, torch.where(state.winner == 1-learner, 1, 2))
        index = (assignment > 0).long()*6 + (learner != state.first).long()*3 + outcome
        self.counts.scatter_add_(0, index.long(), fresh.long())
        self.recorded |= done

    def reset(self, mask):
        self.recorded &= ~mask

    def summary(self):
        values = self.counts.reshape(2, 2, 3).cpu().tolist()
        result = {}
        for i, opponent in enumerate(('rule', 'frozen_models')):
            result[opponent] = {}
            for j, position in enumerate(('first', 'second')):
                win, loss, draw = values[i][j]
                games = win+loss+draw
                result[opponent][position] = dict(games=games, win=win, loss=loss, draw=draw,
                    win_rate=win/games if games else None)
        return result


class OpponentPoolCounters(TerminalCounters):
    """Terminal results by actual sampling pool; independent of policy identity."""
    names=('starter','weave-rush','public-random')

    def __init__(self,n,device):
        super().__init__(n,device)
        import torch
        self.counts=torch.zeros(len(self.names)*6,dtype=torch.int64,device=device)

    def record(self,state,learner,pool):
        import torch
        done=state.phase==3;fresh=done & ~self.recorded
        outcome=torch.where(state.winner==learner,0,torch.where(state.winner==1-learner,1,2))
        index=pool.long()*6+(learner!=state.first).long()*3+outcome
        self.counts.scatter_add_(0,index.long(),fresh.long())
        self.recorded |= done

    def summary(self):
        result={}
        for name,positions in zip(self.names,self.counts.reshape(len(self.names),2,3).cpu().tolist()):
            result[name]={}
            for position,(win,loss,draw) in zip(('first','second'),positions):
                games=win+loss+draw
                result[name][position]={'games':games,'win':win,'loss':loss,'draw':draw,'win_rate':win/games if games else None}
        return result
