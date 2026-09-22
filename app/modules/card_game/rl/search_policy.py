"""Finite-budget Gumbel planning formulas, implemented with NumPy.

Reference: Danihelka et al., Policy improvement by planning with Gumbel (ICLR 2022),
https://openreview.net/forum?id=bERaNdoegnO and Google DeepMind Mctx's
qtransform_completed_by_mix_value / sequential-halving / interior selection.
This module implements the mathematical procedure, not a perfect-information
convergence claim for our approximate information-set sampler.
"""
import math
import numpy as np


def normalized_exp(x):
    x=np.asarray(x,dtype=np.float64);p=np.exp(x-x.max());return p/p.sum()


def completed_q(prior,visits,total,raw_value):
    prior=np.asarray(prior,dtype=np.float64);visits=np.asarray(visits)
    visited=visits>0;q=np.divide(total,visits,out=np.zeros(len(visits)),where=visited)
    mass=prior[visited].sum()
    mean=float(np.dot(prior[visited],q[visited])/mass) if mass>0 else raw_value
    mixed=(raw_value+visits.sum()*mean)/(1+visits.sum())
    q=np.where(visited,q,mixed)
    return (q-q.min())/max(float(q.max()-q.min()),1e-8)*(.1*(50+visits.max(initial=0)))


def visit_schedule(actions,budget):
    if actions==1:return list(range(budget))
    phases=math.ceil(math.log2(actions));counts=[0]*actions;result=[];active=actions
    while len(result)<budget:
        rounds=max(1,budget//(phases*active))
        for _ in range(rounds):
            result.extend(counts[:active])
            for i in range(active):counts[i]+=1
        active=max(2,active//2)
    return result[:budget]


def improved_policy(node):
    return normalized_exp(np.log(np.maximum(node.prior,1e-300))+completed_q(node.prior,node.visits,node.total,node.raw_value))


def choose_root(node,noise,visit):
    score=np.log(np.maximum(node.prior,1e-300))+noise+completed_q(node.prior,node.visits,node.total,node.raw_value)
    eligible=node.visits==visit
    if not eligible.any():raise ValueError('Invalid sequential-halving visit schedule')
    return int(np.argmax(np.where(eligible,score,-np.inf)))


def choose_interior(node):
    return int(np.argmax(improved_policy(node)-node.visits/(1+node.visits.sum())))
