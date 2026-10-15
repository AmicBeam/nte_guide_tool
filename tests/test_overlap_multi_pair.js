'use strict';
const assert = require('node:assert/strict');
const engine = require('../app/modules/overlap/static/js/pair_multi_engine.js');
const core = require('../app/modules/overlap/static/js/pair_engine.js');
const data = require('../app/modules/overlap/static/data/tables.json');
const chars = names => names.map(name => data.characters.find(c => c.name === name));
const ms = value => Math.round(value * 1000);
const near = (a,b) => Math.abs(a-b)<1e-7;
function check(plan, characters) {
    const cuts = plan.segments.slice(1).map(s=>s.start);
    if (!plan.sameEnd) cuts.push(plan.period);
    cuts.forEach((time,i)=>assert((cuts[(i+1)%cuts.length]+(i+1===cuts.length ? plan.period : 0))-time>=1.2-1e-7));
    assert(near(plan.segments[0].start,0));
    assert(near(plan.segments.at(-1).end,plan.period));
    plan.segments.forEach((s,i)=>{
        assert(s.end>s.start);
        if(i){assert(near(s.start,plan.segments[i-1].end));assert.notEqual(s.role,plan.segments[i-1].role);}
    });
    let totalMultiplier=0;
    for(let role=0;role<2;role++) {
        const jobs=plan.jobs.filter(j=>j.role===role).sort((a,b)=>a.occurrence-b.occurrence);
        assert(jobs.length>=1 && jobs.length<=2);
        const rows=jobs.map(j=>engine.rowById(characters[role],j.rowId));
        assert.equal(new Set(rows.map(r=>JSON.stringify(r.actions))).size,jobs.length);
        assert.equal(plan.groups[role].length,jobs.length);
        jobs.forEach((job,i)=>{
            const row=rows[i], next=jobs[(i+1)%jobs.length], nextRow=rows[(i+1)%jobs.length];
            assert(!row.requires_three_characters);
            const gap=engine.transition(characters[role],row,nextRow);
            assert.notEqual(gap,null);
            const nextInput=next.offset+(nextRow.timeline[0].input_seconds??nextRow.timeline[0].start_seconds)+(i+1===jobs.length?plan.period:0);
            assert(nextInput-job.offset>=row.release_end_seconds-1e-7);
            assert(nextInput-job.offset-row.timeline.at(-1).end_seconds>=gap/1000-1e-7);
            const spans=plan.segments.filter(s=>s.role===role);
            row.foreground_intervals.forEach(([start,end])=>assert(spans.some(s=>s.start<=job.offset+start+1e-7 && s.end>=job.offset+end-1e-7)));
            row.timeline.forEach(step=>{
                if(step.input_seconds==null)return;
                const time=job.offset+step.input_seconds;
                assert(spans.some(s=>s.start<=time+1e-7 && s.end>=time-1e-7),'manual input must be on field');
            });
            totalMultiplier+=row.multipliers.atk||0;
        });
    }
    assert(near(totalMultiplier,plan.totalMultiplier));
    assert(near(totalMultiplier/plan.period,plan.multiplierRate));
    assert(plan.overlap>0);
    assert(near(plan.foregroundSeconds[0]+plan.foregroundSeconds[1],plan.period));
    const projected=engine.materialize(plan,characters);
    assert.deepEqual(projected.rows.map(r=>r.actions),plan.actions);
    plan.segments.forEach((segment,index)=>{
        const time=ms(segment.start),offset=plan.offsets[segment.role];
        assert(projected.rows[segment.role].timeline.some(action=>[-ms(plan.period),0,ms(plan.period)].some(shift=>{
            const start=ms(offset+action.start_seconds)+shift,end=ms(offset+action.end_seconds)+shift;
            return start<=time && time<end;
        })),`empty entry at ${segment.start}s for role ${segment.role}`);
    });
}
assert.deepEqual(core.optimize(3,[[0,1,10],[1,0,0],[0,2,10],[2,0,0]],[-1,1,0],[-1,0,1]),[0,10,10]);
const pair=chars(['安魂曲','残虹']);
let count=0,variants=0;
for(const groupCount of ['all','single','with-two','both-two']) {
    const result=engine.match(pair,{groupCount,maxGroups:groupCount==='single'?1:2});
    assert(result.plans.length>0);
    for(const plan of result.plans) {
        check(plan,pair); count++;
        const sizes=plan.groups.map(g=>g.length);
        if(groupCount==='single')assert.deepEqual(sizes,[1,1]);
        if(groupCount==='with-two')assert(sizes.includes(2));
        if(groupCount==='both-two')assert.deepEqual(sizes,[2,2]);
    }
    for(const plan of result.plans.slice(0,20)) {
        const choices=engine.variants(plan,pair);
        assert(choices.length>0);
        choices.forEach(v=>{check(v,pair);assert.equal(v.period,plan.period);variants++;});
    }
}
// Contains-group uses exact membership, not action-subsequence matching, and
// applies before Pareto pruning so the only witness cannot disappear.
const required=JSON.stringify(['a1远','a2远']);
const filtered=engine.match(pair,{containsGroups:[required,''],groupCount:'with-two'});
assert(filtered.plans.length>0);
assert(filtered.plans.every(p=>p.groups[0].some(g=>g.key===required)));
const noDodge=engine.match(pair,{withoutDodge:true,groupCount:'both-two'});
assert(noDodge.plans.every(p=>p.actions.flat().every(a=>!a.endsWith('闪'))));
assert(!engine.groupOptions(pair[0]).some(g=>g.actions.includes('e')));
assert(engine.groupOptions(pair[1]).every(g=>g.actions.every(a=>!a.startsWith('幻'))));
// Magic-E may start a group only if the previous OWN group ends in magic-a5.
const heiyu=chars(['黑羽','明音凛']);
const eGroup=JSON.stringify(['魔e','魔z2']);
const ePlans=engine.match(heiyu,{containsGroups:[eGroup,''],groupCount:'with-two',withoutDodge:true});
assert(ePlans.plans.length>0);
for(const plan of ePlans.plans.slice(0,40)){check(plan,heiyu);assert.equal(plan.groups[0].length,2);}
assert(ePlans.plans.some(p=>p.jobs.some(j=>j.role===0 && j.rowId.includes('~0.6'))),'cross-group magic-E must reuse recorded detached a5 duration');
const badE=engine.match(heiyu,{containsGroups:[eGroup,''],groupCount:'single',maxGroups:1});
assert.equal(badE.plans.length,0);
// Exercise all other pairings with two selected real groups, including dodge
// filtering and the larger Heiyu catalog without an unbounded test runtime.
for (const names of [['安魂曲','明音凛'],['安魂曲','黑羽'],['残虹','明音凛'],['残虹','黑羽']]) {
    const selected=chars(names);
    const containsGroups=selected.map(c=>engine.groupOptions(c,{withoutDodge:true})[0].key);
    const result=engine.match(selected,{containsGroups,groupCount:'both-two',withoutDodge:true});
    assert(result.plans.length>0);
    result.plans.slice(0,40).forEach(p=>{check(p,selected);assert.deepEqual(p.groups.map(g=>g.length),[2,2]);});
    result.plans.slice(0,5).forEach(p=>engine.variants(p,selected).forEach(v=>check(v,selected)));
}
// Cross-group restart delays: a4 -> a1 waits; a5 -> a1 is a recorded restart.
const raw=engine.rawRows(pair[0]);
const a4=raw.find(r=>r.actions.at(-1)==='a4远'),a5=raw.find(r=>r.actions.at(-1)==='a5远');
assert.equal(engine.transition(pair[0],a4,a5),500);
assert.equal(engine.transition(pair[0],a5,a4),0);
// Entry windows contain signed period edges, so feasibility is not monotone in
// T. Scan from an independent lower bound and enumerate every entry disjunction.
let checkedMinimum=0;
for(const plan of engine.match(pair,{groupCount:'both-two'}).plans.slice(0,20)) {
    const own=plan.recipe.sequences.map((ids,role)=>{
        const rows=ids.map(id=>engine.rowById(pair[role],id));
        return {rows,groups:rows.map(r=>JSON.stringify(r.actions)),gaps:rows.map((r,i)=>engine.transition(pair[role],r,rows[(i+1)%rows.length]))};
    });
    const model=engine.model(own,plan.recipe.order), period=ms(plan.period);
    const chosen=engine.specialize(model,plan.recipe.entryChoices);
    assert(core.solve(chosen.size,engine.edgesAt(chosen,period)));
    assert.equal(core.solve(chosen.size,engine.edgesAt(chosen,period-1)),null);
    let low=0,high=period;
    const baseEdges=t=>[...model.constant,...model.periodic.map(([u,v,w])=>[u,v,w+t])];
    while(low<high){const mid=Math.floor((low+high)/2);if(core.solve(model.size,baseEdges(mid)))high=mid;else low=mid+1;}
    while(low<period && !engine.entryModelsAt(model,low).length)low++;
    assert.equal(period,low);checkedMinimum++;
}
// The known one-group recovery allocation remains available under the single filter.
const original=engine.match(pair,{groupCount:'single',maxGroups:1,containsGroups:[JSON.stringify(['a1远','a2远','a3远']),JSON.stringify(['a1','z','a5'])]});
assert(original.plans.length>0);
assert.equal(original.plans[0].period,2.8);
const allocations=engine.variants(original.plans[0],pair);
assert(near(Math.max(...allocations.map(p=>p.foregroundSeconds[1])),1.6));
allocations.forEach(p=>check(p,pair));
// The screenshot's idle start (0 -> 0.1s before the first action) is forbidden.
assert(allocations.every(p=>near(p.offsets[0],0)));
assert(allocations.some(p=>{
    const segment=p.segments.find(s=>s.role===1),first=p.jobs.find(j=>j.role===1).offset;
    return first-segment.start>=.6-1e-7 && near(p.foregroundSeconds[1],1.6);
}),'returning during a5 recovery must remain possible');
const idle={...allocations[0],jobs:allocations[0].jobs.map(j=>j.role===0?{...j,offset:j.offset+.1}:j),offsets:[.1,allocations[0].offsets[1]]};
assert.throws(()=>check(idle,pair),/empty entry/);
console.log(`PASS: ${count} multi-group plans, ${variants} allocations, ${checkedMinimum} exact period oracle checks`);
