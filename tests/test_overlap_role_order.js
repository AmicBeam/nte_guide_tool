/* Swapping UI roles must preserve the searched cycles and their objective. */
'use strict';
const assert = require('node:assert/strict');
const engine = require('../app/modules/overlap/static/js/pair_multi_engine.js');
const data = require('../app/modules/overlap/static/data/tables.json');
const chars = names => names.map(name => data.characters.find(c => c.name === name));
const reverseSort = sort => ({'first-rate':'second-rate','second-rate':'first-rate'}[sort] || sort);
const identity = (plan, reverse = false) => JSON.stringify([
    (reverse ? [...plan.groups].reverse() : plan.groups).map(groups => groups.map(g => g.key).sort()),
    plan.period, plan.totalMultiplier,
]);
function validate(plan, characters) {
    assert(plan.segments.every(s => s.end > s.start));
    assert.equal(plan.segments[0].start, 0);
    assert.equal(plan.segments.at(-1).end, plan.period);
    const projected = engine.materialize(plan, characters);
    assert.deepEqual(projected.rows.map(row => row.actions), plan.actions);
    for (const segment of plan.segments) {
        const role = segment.role, offset = plan.offsets[role];
        assert(projected.rows[role].timeline.some(step => [-plan.period,0,plan.period].some(shift =>
            offset + step.start_seconds + shift <= segment.start + 1e-8
            && offset + (step.recovery_end_seconds ?? step.end_seconds) + shift > segment.start + 1e-8)), 'live entry');
    }
    const cuts = plan.segments.slice(1).map(s => s.start);
    if (!plan.sameEnd) cuts.push(plan.period);
    cuts.forEach((t,i) => assert(cuts[(i+1)%cuts.length] + (i+1===cuts.length ? plan.period : 0) - t >= 1.2-1e-8));
}
let comparisons = 0;
function verify(pair, options = {}, exact = false) {
    const sort = options.sort || 'rate';
    const reverse = [...pair].reverse();
    const swapped = {...options, sort:reverseSort(sort), containsGroups:options.containsGroups?.slice().reverse()};
    const forwardSearch = engine.createTopSearch(pair, options), reverseSearch = engine.createTopSearch(reverse, swapped);
    for (const k of [20,40]) {
        const a = forwardSearch.extend(k).plans, b = reverseSearch.extend(k).plans;
        assert.equal(a.length, b.length);
        const normalized = b.map(plan => ({...plan, actions:[...plan.actions].reverse(), multipliers:[...plan.multipliers].reverse()}));
        // Different group IDs may break an otherwise exact ranking tie differently.
        const compare = engine.comparePlans(sort);
        for (let i=0;i<a.length;i++) assert.equal(compare({...a[i],id:''},{...normalized[i],id:''}), 0,
            `${pair.map(c=>c.name)} ${JSON.stringify(options)} Top${k} #${i}`);
        if (exact && a.length) {
            const boundary = {...a.at(-1),id:''};
            const above = (plans,normalize) => plans.filter((_,i) => compare({...normalize[i],id:''},boundary)<0);
            assert.deepEqual(above(a,a).map(p=>identity(p)).sort(), above(b,normalized).map(p=>identity(p,true)).sort());
        }
        for (const [plans, characters] of [[a,pair],[b,reverse]]) for (const plan of plans.slice(0,2)) {
            validate(plan, characters);
            const variants = engine.variants(plan, characters);
            assert(variants.length > 0);
            variants.forEach(v => { assert.equal(v.period,plan.period); validate(v,characters); });
        }
        comparisons++;
    }
}
for (let a=0;a<data.characters.length;a++) for (let b=a+1;b<data.characters.length;b++) {
    verify([data.characters[a],data.characters[b]]);
}
const pair=chars(['Exe','黑羽']);
for (const options of [{}, {withoutDodge:true}, {noForegroundIdle:true}, {groupCount:'single'},
    {groupCount:'both-two'}, {containsGroups:[JSON.stringify(['a1','a2','a3','a4']),JSON.stringify(['魔e','魔z2','魔a3','魔a4','魔a5'])]},
    ...['first-rate','second-rate','time','actions'].map(sort=>({sort}))]) verify(pair,options,true);
for (const characters of [pair,[...pair].reverse()]) {
    const options={maxGroups:1,containsGroups:characters.map(c => JSON.stringify(c.name==='Exe'
        ? ['a1','a2','a3','a4'] : ['魔e','魔z2','魔a3','魔a4','魔a5']))};
    const full=engine.match(characters,options).plans;
    const top=engine.createTopSearch(characters,options).extend(20).plans;
    assert.equal(full.length,1);
    assert.equal(full[0].period,5.3);
    assert(Math.abs(full[0].multiplierRate-15.235368/5.3)<1e-10);
    assert.deepEqual(top.map(p=>identity(p)),full.map(p=>identity(p)));
    validate(top[0],characters);
}
console.log(`PASS: ${comparisons} role-swap Top20/40 comparisons and both 5.3s cycle anchors`);
