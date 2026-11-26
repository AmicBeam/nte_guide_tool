const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

test('signal is a single character in the personal resource stack for either side and replay', () => {
  const context = vm.createContext({window: {matchMedia:()=>({matches:false})}, console,
    document:{getElementById:()=>null}, requestAnimationFrame(){}});
  for (const file of ['v2_api.js','v2_table/helpers.js','v2_table/render.js']) {
    vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/'+file,'utf8'),context);
  }
  const attrs={};
  const node={style:{},classList:{toggle(){},remove(){},add(){}},
    setAttribute(k,v){attrs[k]=v},getAttribute(k){return attrs[k]},removeAttribute(k){delete attrs[k]},
    querySelectorAll(){return []},querySelector(){return null},innerHTML:''};
  for (const side of ['a','b']) for (const usePortrait of [false,true]) {
    for (let level=0;level<3;level++) {
      const data=JSON.parse(JSON.stringify({id:'elyms',name:'曳尔沐',attack:2,hp:5,max_hp:5,signal_level:level}));
      context.window.NTE_V2_TABLE.fillCharacterCard(node,data,{side,usePortrait});
      assert.match(node.innerHTML, /v2-attack-stack[\s\S]*is-signal/);
      assert.match(node.innerHTML, new RegExp('is-signal[^>]*title="信号强度：'+['低信号','中信号','高信号'][level]+'"[^>]*>[\\s\\S]*?<em>'+['低','中','高'][level]+'</em>'));
      const signal = node.innerHTML.match(/<span class="v2-combat-stat is-signal"[^>]*>([\s\S]*?)<\/span>/)[1];
      assert.ok(signal.indexOf('<em>') < signal.indexOf('<svg'));
      assert.ok(signal.includes('data-signal-level="' + level + '"'));
      assert.equal((signal.match(/<path /g) || []).length, level + 1);
      assert.doesNotMatch(node.innerHTML, /<em>(低信号|中信号|高信号)<\/em>/);
    }
    context.window.NTE_V2_TABLE.fillCharacterCard(node,{id:'zero',name:'零',hp:5},{side,usePortrait});
    assert.doesNotMatch(node.innerHTML,/is-signal/);
  }
});
