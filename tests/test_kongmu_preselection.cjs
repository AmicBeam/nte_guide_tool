const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.resolve(__dirname,'../app/modules/kongmu/static/js/kongmu.js'),'utf8')
  .replace('bootstrapKongmu();','');

async function bootstrap(query, allowed) {
  const elements = new Map();
  const context = vm.createContext({URLSearchParams,window:{location:{search:query}},
    document:{getElementById(id){
      if (!elements.has(id)) elements.set(id,{textContent:'',hidden:true,disabled:false});
      return elements.get(id);
    }},catalog:{characters:[{id:'1003'},...(allowed?[{id:'1090',merged_ids:['char_elyms']}]:[])],
      cartridges:[{id:'cosmos'}]},calls:0});
  vm.runInContext(source,context);
  vm.runInContext(`kongmuRequest=async()=>catalog;
    renderKongmuPickers=()=>{};bindKongmuEvents=()=>{};
    runKongmuPlan=async()=>{calls++;};`,context);
  await vm.runInContext('bootstrapKongmu()',context);
  return {id:vm.runInContext('kongmuState.selectedCharacterId',context),calls:context.calls,elements};
}

test('an inaccessible linked character is not replaced by the first available character',async()=>{
  const result=await bootstrap('?character=1090&cartridge=cosmos',false);
  assert.equal(result.id,'');assert.equal(result.calls,0);
  assert.equal(result.elements.get('kongmu-plan-btn').disabled,true);
  assert.match(result.elements.get('kongmu-empty-result').textContent,/暂无权限/);
});
for (const id of ['1090','char_elyms']) test(`an authorized ${id} link calculates canonical Elyms`,async()=>{
  const result=await bootstrap(`?character=${id}&cartridge=cosmos`,true);
  assert.equal(result.id,'1090');assert.equal(result.calls,1);
});
test('an unlinked visit preserves manual calculation',async()=>{
  const result=await bootstrap('',false);assert.equal(result.id,'1003');assert.equal(result.calls,0);
});
