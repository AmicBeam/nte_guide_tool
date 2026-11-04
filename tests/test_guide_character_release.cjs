const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.resolve(__dirname,'../app/modules/guide/static/character_release.js'),'utf8');
const settle=()=>new Promise(resolve=>setImmediate(resolve));
function setup(responses,clientYear=2099,nodes=1){
  const state={reloads:0,timers:[],requests:[],visibility:null};
  const NativeDate=Date;
  class WrongClientDate extends NativeDate { constructor(){super(clientYear,0,1);} static now(){return new NativeDate(clientYear,0,1).getTime();} }
  const context=vm.createContext({Date:WrongClientDate,Number,
    document:{visibilityState:'visible',querySelectorAll:()=>Array.from({length:nodes},(_,i)=>({dataset:{characterReleaseUrl:'/api/guide/characters/'+i+'/release'}})),
      addEventListener(name,fn){state.visibility=fn;}},
    window:{location:{reload(){state.reloads++;}}},
    clearTimeout(){},setTimeout(fn,delay){state.timers.push({fn,delay});return state.timers.length;},
    fetch:async(url,options)=>{state.requests.push({url,options});const data=responses.shift();
      return {ok:!!data,json:async()=>({data})};}
  });
  vm.runInContext(source,context);return state;
}
test('a future client clock cannot release content while the server says no',async()=>{
  const state=setup([{content_released:false,release_at:'2026-11-05T10:00:00+08:00',check_after_ms:1000}],2099);
  await settle();assert.equal(state.reloads,0);assert.equal(state.timers[0].delay,1000);
  assert.equal(state.requests[0].options.cache,'no-store');
});
test('server transition releases a page even when the client clock is in the past',async()=>{
  const state=setup([{content_released:false,check_after_ms:500},{content_released:true}],1970);
  await settle();await state.timers[0].fn();await settle();assert.equal(state.reloads,1);
});
test('a failed status read retries and returning to the tab checks immediately',async()=>{
  const state=setup([null,{content_released:true}]);await settle();
  assert.equal(state.reloads,0);assert.equal(state.timers[0].delay,30000);
  await state.visibility();await settle();assert.equal(state.reloads,1);
});
test('several released characters trigger only one reload',async()=>{
  const state=setup([{content_released:true},{content_released:true}],2099,2);
  await settle();assert.equal(state.reloads,1);
});
