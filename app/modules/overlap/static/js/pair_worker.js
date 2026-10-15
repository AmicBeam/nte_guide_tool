'use strict';
importScripts(`pair_engine.js${self.location.search}`, `pair_multi_engine.js${self.location.search}`);
let characters, plans;
self.onmessage = ({data}) => {
    try {
        if (data.type === 'variants') {
            self.postMessage({variants: self.OverlapMultiEngine.variants(plans.get(data.planId), characters), planId: data.planId});
            return;
        }
        characters = data.characters;
        const result = self.OverlapMultiEngine.match(characters, data.options, progress => self.postMessage({progress}));
        plans = new Map(result.plans.map(plan => [plan.id, plan]));
        self.postMessage({result: {...result, plans: result.plans.map(plan => ({id: plan.id, groups: plan.groups,
            actions: plan.actions, period: plan.period, multipliers: plan.multipliers,
            totalMultiplier: plan.totalMultiplier, multiplierRate: plan.multiplierRate, variants: null}))}});
    } catch (error) { self.postMessage({error: error.message}); }
};
