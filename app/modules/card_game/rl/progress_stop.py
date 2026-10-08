"""Conservative, preconfigured no-progress detector over scalar window means."""
import math


class ProgressStop:
    def __init__(self, window_seconds=300, patience=6):
        self.window_seconds=window_seconds;self.patience=patience
        self.last_window=0.;self.rows=[];self.reference=None;self.stale=0;self.inactive=0;self.last_summary=None

    def observe(self, seconds, row):
        self.rows.append(row)
        if seconds-self.last_window<self.window_seconds:return False
        keys=('value_loss','explained_variance','entropy','approx_kl','clip_fraction') if 'value_loss' in row else ('reward_mean','loss')
        if not all(all(k in r and math.isfinite(float(r[k])) for k in keys) for r in self.rows):
            raise ValueError('Invalid progress metrics')
        mean={k:sum(float(r[k]) for r in self.rows)/len(self.rows) for k in keys}
        activity=sum(r.get('finished_games',r.get('valid_candidates',0)) for r in self.rows)
        self.rows=[];self.last_window=seconds
        self.inactive=self.inactive+1 if activity<=0 else 0
        if self.inactive>=self.patience:raise RuntimeError("No completed games or valid candidates across six progress windows")
        tolerances={'value_loss':max(.01,abs(mean.get('value_loss',0))*.01),
                    'explained_variance':.01,'entropy':.01,'approx_kl':.0001,'clip_fraction':.01,
                    'reward_mean':.01,'loss':max(.01,abs(mean.get('loss',0))*.01)}
        # Any material change postpones the stop, including deterioration: this is not a convergence certificate.
        changed=self.reference is None or any(abs(mean[k]-self.reference[k])>tolerances[k] for k in keys)
        if changed or activity<=0:
            self.stale=0;self.reference=mean
        else:self.stale+=1
        self.last_summary={'seconds':seconds,'window_means':mean,'tolerances':tolerances,
                           'stale_windows':self.stale,'activity':activity,'low_entropy':mean.get('entropy',1)<.01}
        return self.stale>=self.patience
