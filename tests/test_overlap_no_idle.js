'use strict';
const assert=require('node:assert/strict');
const engine=require('../app/modules/overlap/static/js/pair_multi_engine.js');
function character(id,duration,recovery=0) {
    const names=[`${id}1`,`${id}2`];
    return {id,name:id,actions:names.map((name,i)=>({name,type:'普攻',successors:i?[]:[{to:names[1]}],recovery_seconds:i?recovery:0})),
        rows:[{id,actions:names,original_actions:names,requires_three_characters:false,
            foreground_intervals:[[0,.2]],foreground_seconds:.2,release_end_seconds:duration,multipliers:{atk:2},
            timeline:[{action:names[0],input_seconds:0,start_seconds:0,end_seconds:.2,duration_seconds:.2,base_duration_seconds:.2,recovery_seconds:0},
                {action:names[1],input_seconds:.1,start_seconds:.2,end_seconds:duration,duration_seconds:duration-.2,
                    base_duration_seconds:duration-.2,recovery_seconds:recovery}]}]};
}
// A live entry alone is insufficient: B's body ends before its 1.2s foreground turn.
const idlePair=[character('long',2),character('short',.5)];
assert(engine.match(idlePair,{maxGroups:1}).plans.length>0);
assert.equal(engine.match(idlePair,{maxGroups:1,noForegroundIdle:true}).plans.length,0);
assert.equal(engine.createTopSearch(idlePair,{maxGroups:1,noForegroundIdle:true}).extend(20).plans.length,0);
// Entering during recovery is legal normally, but recovery is idle under this filter.
const recoveryPair=[character('recovery',2.1,.5),character('other',1.2)];
const sequences=recoveryPair.map(c=>engine.sequences(c,{maxGroups:1})[0]);
const order=[{role:0,occurrence:0,segment:0},{role:1,occurrence:0,segment:0}];
for(const noForegroundIdle of [false,true]) {
    const model=engine.model(sequences,order,{noForegroundIdle});
    model.constant.push([0,1,100],[1,0,-100]);
    const result=engine.minimum(model);
    if(noForegroundIdle) assert.equal(result,null);
    else assert.equal(result.period,2600);
}
// Consecutive groups may cover one foreground block when their bodies touch.
const connectedPair=[character('A',1.2),character('B',1.2)];
const result=engine.match(connectedPair,{noForegroundIdle:true});
assert(result.plans.some(p=>p.groups[0].length===2 && p.groups[1].length===1));
console.log('PASS: idle tail excluded, recovery excluded, connected foreground groups retained');
