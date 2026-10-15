/* Run with node tests/test_overlap_pair.js. Uses synthetic timing and real catalog. */
'use strict';
const assert = require('node:assert/strict');
const engine = require('../app/modules/overlap/static/js/pair_engine.js');
const data = require('../app/modules/overlap/static/data/tables.json');
function row(intervals, release, extra = {}) {
    return {requires_three_characters:false, foreground_intervals:intervals,
        release_end_seconds:release, actions:['a1','a2'],
        timeline:[{action:'a1', start_seconds:0, end_seconds:release, duration_seconds:release}], ...extra};
}
function check(plan) {
    assert(plan);
    const {period, segments, offsets, rows} = plan;
    assert(period > 0);
    for (let i = 1; i < segments.length; i++) {
        assert(Math.abs(segments[i].start - segments[i-1].end) < 1e-7);
        assert.notEqual(segments[i].role, segments[i-1].role);
    }
    const cuts = segments.slice(1).map(s => s.start);
    if (!plan.sameEnd) cuts.push(period);
    cuts.forEach((cut, i) => {
        const next = i+1 < cuts.length ? cuts[i+1] : cuts[0]+period;
        assert(next-cut >= 1.2-1e-7, `cyclic cooldown ${next-cut}`);
    });
    rows.forEach((r, role) => {
        assert(period >= r.release_end_seconds-1e-7);
        r.foreground_intervals.forEach(([start,end], index) => {
            const s=segments.find(s => s.role===role && s.index===index);
            assert(s.start <= offsets[role]+start+1e-7);
            assert(s.end >= offsets[role]+end-1e-7);
        });
        r.timeline.forEach(step => {
            if (step.input_seconds == null) return;
            const time=offsets[role]+step.input_seconds;
            assert(segments.some(s=>s.role===role && time>=s.start-1e-7 && time<=s.end+1e-7), `input outside foreground ${time}`);
        });
    });
}
// Earliest cut 0.2s must be delayed. Both sides of the seam need 1.2s.
const delayed=engine.schedule(row([[0,.2]],2), row([[0,.2]],2));
check(delayed);
assert.equal(delayed.period,2.4);
assert.equal(delayed.segments[0].end,1.2);
assert.equal(delayed.segments[1].end,2.4);
// A route requiring a third actor must never enter the matcher.
assert.equal(engine.schedule(row([[0,.2]],2,{requires_three_characters:true}),row([[0,.2]],2)),null);
// A returning A can end the cycle only after a legal full finish back to its start.
const returning=row([[0,.4],[2,2.4]],2.4,{hold_reason:'cycle_to_start'});
const same=engine.schedule(returning,row([[0,1.2]],1.8));
check(same); assert(same.sameEnd);
assert.equal(engine.schedule({...returning,hold_reason:'no_successor'}, row([[0,1.2]],1.8)),null);
// Foreground windows that cannot interleave remain impossible, even with extra period.
assert.equal(engine.schedule(row([[0,.8],[2,3]],3),row([[0,2],[2.5,3]],3)),null);
// One does not call a wholly serial pair an overlap plan.
assert.equal(engine.schedule(row([[0,2]],2),row([[0,2]],2)),null);
// Same action sequence -> minimum period; extra ordered actions at no time cost dominate.
const p=(actions,period)=>({actions,period});
assert.deepEqual(engine.prune([p([['a'],['b']],3),p([['a'],['b']],2),p([['a','c'],['b']],2)]),[p([['a','c'],['b']],2)]);
assert.equal(engine.prune([p([['a','c'],['b']],2),p([['c','a'],['b']],2)]).length,2);
assert.throws(()=>engine.match([data.characters[0],data.characters[0]]),/不同/);
let total=0;
for (let i=0;i<data.characters.length;i++) for(let j=i+1;j<data.characters.length;j++) {
    const result=engine.match([data.characters[i],data.characters[j]]);
    assert(result.plans.length>0);
    result.plans.forEach(plan=>{
        check(plan);assert(plan.rows.every(r=>!r.requires_three_characters));
        assert.equal(plan.totalMultiplier,plan.rows.reduce((sum,r)=>sum+(r.multipliers.atk||0),0));
        assert.equal(plan.multiplierRate,plan.totalMultiplier/plan.period);
        plan.multipliers.forEach((m,role)=>assert.equal(m.atk,plan.rows[plan.roles.indexOf(role)].multipliers.atk));
    });
    total+=result.plans.length;
}
const hei=engine.availableRows(data.characters[2]);
assert(hei.length>0);
assert(hei.some(r=>r.actions[0]==='魔e' && r.actions.at(-1)==='魔a5'));
assert(hei.every(r=>r.actions.every((a,i,n)=>a!=='魔e' || n[(i+n.length-1)%n.length]==='魔a5')));
const heiFixture={name:'黑羽',actions:[{name:'魔e'},{name:'魔a5'},{name:'魔a4'}],rows:[
 row([[0,1.4]],2,{actions:['魔e','魔a5']}),
 row([[0,1.4]],2,{actions:['魔e','魔a4']}),
 row([[0,1.4]],2,{actions:['魔a5','魔e']}),
 row([[0,1.4]],2,{actions:['魔a5','魔a4','魔e']}),
]};
assert.deepEqual(engine.availableRows(heiFixture).map(r=>r.actions),[['魔e','魔a5'],['魔a5','魔e']]);
assert(engine.availableRows(data.characters[0]).every(r=>!r.original_actions.includes('e')));
assert(engine.availableRows(data.characters[1]).every(r=>r.original_actions.every(n=>!n.startsWith('幻'))));
for (const conditions of [{required_buff_key:'state'},{required_additional_buff_keys:['ready']},{cooldown_ticks:120},{energy_cost:1},{personal_resource_cost:{charge:1}}]) assert(!engine.unrestricted({conditions}));
assert(engine.unrestricted({conditions:{cooldown_ticks:0,energy_cost:0,personal_resource_cost:{}}}));
const clean=engine.match([data.characters[2],data.characters[1]],{withoutDodge:true});
assert(clean.plans.every(p=>p.rows.every(r=>!r.has_dodge)));
console.log(`PASS: synthetic seam/delay/dominance cases and ${total} real pair plans`);
