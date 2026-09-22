"""Require measured formal-search throughput before committing a long-run budget."""
import math,time


def validate_capacity(report,config,now=None):
    now=time.time() if now is None else now
    if report.get('rule_hash')!=config['rule_hash'] or report.get('simulations')!=config['simulations']:
        raise ValueError('Capacity measurement must use the current rules and formal search budget')
    if report.get('search_algorithm','puct')!=config.get('search_algorithm','puct'):
        raise ValueError('Capacity measurement must use the planned search algorithm')
    if config.get('search_algorithm')=='gumbel' and report.get('gumbel_candidates')!=config.get('gumbel_candidates',16):
        raise ValueError('Capacity measurement must use the planned candidate limit')
    if report.get('workers')!=config['workers'] or not report.get('complete'):
        raise ValueError('Capacity measurement must be complete and use the planned worker count')
    if not 0<=now-report.get('measured_at',0)<=86400:
        raise ValueError('Capacity measurement must be at most 24 hours old')
    seconds=report.get('estimated_stage_seconds',{})
    starts={'foundation':config['created'],'search':config['deadlines']['foundation'],
            'adaptation':config['deadlines']['search'],'confirmation':config['deadlines']['candidate_adaptation'],
            'audit':config['deadlines']['confirmation'],'evaluation':config['deadlines']['audit']}
    ends={**config['deadlines'],'adaptation':config['deadlines']['candidate_adaptation']}
    for phase,start in starts.items():
        value=seconds.get(phase)
        if not isinstance(value,(float,int)) or not math.isfinite(value) or value<=0:
            raise ValueError('Missing measured capacity estimate for '+phase)
        if value*1.25>ends[phase]-start:
            raise ValueError('Insufficient time for '+phase+' with 25% capacity margin')
    return True
