/* Exact on-demand Top-K against exhaustive matching under the same filters. */
'use strict';
const assert = require('node:assert/strict');
const engine = require('../app/modules/overlap/static/js/pair_multi_engine.js');
const data = require('../app/modules/overlap/static/data/tables.json');
const sorts = ['rate','first-rate','second-rate','time','actions'];
const identity = plan => [plan.id,plan.period,plan.actions,plan.multipliers,plan.recipe];
function assertNoForegroundIdle(plan,characters) {
    const projected=engine.materialize(plan,characters),tick=n=>Math.round(n*1000);
    for(const segment of plan.segments) {
        const offset=plan.offsets[segment.role], spans=[];
        for(const step of projected.rows[segment.role].timeline) for(const shift of [-plan.period,0,plan.period]) {
            const start=tick(offset+step.start_seconds+shift);
            spans.push([start,start+tick(step.base_duration_seconds ?? step.duration_seconds)]);
        }
        spans.sort((a,b)=>a[0]-b[0]);
        let cursor=tick(segment.start),end=tick(segment.end);
        for(const [left,right] of spans) {
            if(right<=cursor) continue;
            if(left>cursor) break;
            cursor=Math.max(cursor,right);
            if(cursor>=end) break;
        }
        assert(cursor>=end,`foreground idle at ${cursor/1000}s in ${plan.id}`);
    }
}
let comparisons=0, cached=0;
function verify(characters, options={}, reference=null) {
    const full=reference || engine.match(characters,options);
    for(const sort of sorts) {
        const expected=[...full.plans].sort(engine.comparePlans(sort));
        const search=engine.createTopSearch(characters,{...options,sort});
        const first=search.extend(20);
        assert.deepEqual(first.plans.map(identity),expected.slice(0,20).map(identity),`${characters.map(c=>c.name)} ${JSON.stringify(options)} ${sort}`);
        assert.equal(first.filtered+first.pruned+first.evaluated,first.checked);
        const again=search.extend(20);
        assert.equal(again.solved,first.solved);
        assert.equal(again.completed,first.completed);
        if(options.noForegroundIdle) first.plans.forEach(p=>assertNoForegroundIdle(p,characters));
        const more=search.extend(40);
        assert.deepEqual(more.plans.map(identity),expected.slice(0,40).map(identity),`more ${sort}`);
        assert(more.completed>=first.completed);
        if(options.noForegroundIdle) more.plans.forEach(p=>assertNoForegroundIdle(p,characters));
        assert.equal(more.filtered+more.pruned+more.evaluated,more.checked);
        const small=engine.createTopSearch(characters,{...options,sort}).extend(1);
        assert.deepEqual(small.plans.map(identity),expected.slice(0,1).map(identity),`Top1 ${sort}`);
        if(expected.length<=40) assert.equal(more.hasMore,false);
        if(sort==='rate' && characters[0].name==='黑羽' && characters[1].name==='明音凛' && !Object.keys(options).length) {
            assert(first.topPruned>0);
            assert(first.solved<full.solved);
            console.log(`Heiyu/Akane: ${full.solved} -> ${first.solved} solved structures, ${first.completed}/${first.buckets} group buckets`);
        }
        // Returned recipes must still reconstruct their exact chosen timing slots.
        for(const plan of first.plans.slice(0,2)) {
            const projected=engine.materialize(plan,characters);
            assert.deepEqual(projected.rows.map(r=>r.actions),plan.actions);
            const variants=engine.variants(plan,characters);
            assert(variants.length>0);
            assert(variants.every(v=>v.period===plan.period));
            if(options.noForegroundIdle) variants.forEach(v=>assertNoForegroundIdle(v,characters));
        }
        comparisons+=3;cached++;
    }
}
for(let a=0;a<data.characters.length;a++) for(let b=a+1;b<data.characters.length;b++) {
    const pair=[data.characters[a],data.characters[b]];
    verify(pair);
    verify(pair,{noForegroundIdle:true});
    console.log(`PASS Top-K: ${pair.map(c=>c.name).join('/')}`);
}
const heiyu=data.characters.find(c=>c.name==='黑羽'),akane=data.characters.find(c=>c.name==='明音凛');
const pair=[heiyu,akane];
verify(pair);
for(const options of [{withoutDodge:true}, {groupCount:'single'}, {groupCount:'with-two'},
    {groupCount:'both-two'}, {maxGroups:1},
    {containsGroups:[JSON.stringify(['魔a1','魔a2','魔a3','魔a4','魔a5']),JSON.stringify(['a1','a2','a3','a4'])]},
    {maxGroups:1,groupCount:'both-two'},
    {noForegroundIdle:true,withoutDodge:true,groupCount:'both-two'},
    {noForegroundIdle:true,containsGroups:[JSON.stringify(['魔a1','魔a2','魔a3','魔a4','魔a5']),JSON.stringify(['a1','a2','a3','a4'])]}]) verify(pair,options);
assert.throws(()=>engine.createTopSearch(pair,{sort:'unknown'}),/排序/);
assert.throws(()=>engine.createTopSearch(pair).extend(0),/正整数/);
console.log(`PASS: ${comparisons} exhaustive Top1/20/40 comparisons, ${cached} cached extensions`);
