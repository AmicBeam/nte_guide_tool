'use strict';
importScripts(`pair_engine.js${self.location.search}`, `pair_multi_engine.js${self.location.search}`);
let characters, plans, search;
self.onmessage = ({data}) => {
    try {
        if (data.type === 'variants') {
            self.postMessage({variants: self.OverlapMultiEngine.variants(plans.get(data.planId), characters), planId: data.planId});
            return;
        }
        if (data.type !== 'more') {
            characters = data.characters;
            search = self.OverlapMultiEngine.createTopSearch(characters, data.options, progress => self.postMessage({progress}));
        }
        const result = search.extend(data.limit || 20);
        plans = new Map(result.plans.map(plan => [plan.id, plan]));
        self.postMessage({result: {...result, plans: result.plans.map(plan => ({id: plan.id, groups: plan.groups,
            actions: plan.actions, period: plan.period, multipliers: plan.multipliers,
            totalMultiplier: plan.totalMultiplier, multiplierRate: plan.multiplierRate, variants: null}))}});
    } catch (error) { self.postMessage({error: error.message}); }
};
