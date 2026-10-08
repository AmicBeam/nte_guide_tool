const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const clone = value => JSON.parse(JSON.stringify(value));
const apiSource = fs.readFileSync('app/modules/card_game/static/js/v2_api.js', 'utf8');
function fixture() {
  return {characters: [
    {id:'anhunqu',name:'安魂曲',mechanisms:[{id:'mechanism-nightmare',character_id:'anhunqu',type:'mechanism',name:'噩梦',description:'上限 10 层。'}]},
    {id:'canhong',name:'残虹',mechanisms:[{id:'mechanism-etch-venom',character_id:'canhong',type:'mechanism',name:'蚀心与鸩火',reference_names:['蚀心','鸩火'],description:'各自上限 10 层。'}]},
    {id:'adler',name:'阿德勒',mechanisms:[{id:'mechanism-guard',character_id:'adler',type:'mechanism',name:'诛恶护持',description:'无视护盾。'}]},
  ],cards:[{id:'C02',name:'渊底之吻',type:'battle',character_id:'canhong'}]};
}
async function load(catalog=fixture()) {
  const context = vm.createContext({window:{localStorage:{getItem:()=>''}},
    fetch:async()=>({ok:true,status:200,headers:{get:()=>null},json:async()=>catalog})});
  vm.runInContext(apiSource, context);
  const api=context.window.NTE_V2;
  await api.getCatalog();
  return api;
}
test('character and card descriptions resolve private mechanisms alongside real cards', async()=>{
  const api=await load();
  assert.deepEqual(clone(api.referencedCards({name:'残虹',description:'施加「蚀心」。终结施加「鸩火」。获得「渊底之吻」。'})).map(c=>c.id), ['mechanism-etch-venom','C02']);
  assert.equal(api.referencedCards({name:'A04',description:'施加「噩梦」。'})[0].type,'mechanism');
  assert.equal(api.referencedCards({name:'阿德勒',description:'施加「诛恶护持」。'})[0].name,'诛恶护持');
  assert.equal(api.cardTypeLabel('mechanism'),'机制说明');
  assert.equal(api.cardFrameStat({type:'mechanism'}),'');
});
test('reference cards do not recursively reference themselves and escape explanation text', async()=>{
  const api=await load();
  assert.equal(api.referencedCards({id:'mechanism-etch-venom',name:'蚀心与鸩火',description:'「蚀心」与「鸩火」。'}).length,0);
  assert.equal(api.cardDescriptionMarkup('<img onerror="bad">'), '&lt;img onerror=&quot;bad&quot;&gt;');
});
test('codex appends the character explanation after the owned skill cards', async()=>{
  const catalog=fixture();
  catalog.characters[2].passive='护盾即攻击';catalog.characters[2].awakened_passive='施加「诛恶护持」。';
  catalog.cards.push({id:'D08',name:'正心',type:'form',character_id:'adler',description:'浊燃强化。'});
  const api=await load(catalog);
  const nodes={};
  const node=id=>nodes[id]||(nodes[id]={innerHTML:'',listeners:{},addEventListener(kind,fn){this.listeners[kind]=fn;},querySelectorAll:()=>[],focus(){},getBoundingClientRect:()=>({})});
  const context=vm.createContext({window:{NTE_V2:{...api,ensureLogin:()=>true,getCatalog:async()=>catalog}},
    document:{getElementById:node}});
  vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/v2_codex.js','utf8'),context);
  await new Promise(resolve=>setImmediate(resolve));
  nodes['v2-codex-nav'].listeners.click({target:{closest:()=>({getAttribute:()=> 'adler'})}});
  const html=nodes['v2-codex-body'].innerHTML;
  assert.ok(html.indexOf('正心') < html.indexOf('is-mechanism'));
  assert.match(html,/is-mechanism[\s\S]*机制说明[\s\S]*诛恶护持/);
});

test('solitary resolves from passive and R06 without portrait or owner in its reference card', async()=>{
  const catalog=JSON.parse(fs.readFileSync('app/modules/card_game/content/duel_v2/catalog.json','utf8'));
  catalog.characters.forEach(c=>(c.mechanisms||[]).forEach(m=>Object.assign(m,{type:'mechanism',character_id:c.id})));
  const api=await load(catalog);
  const hero=catalog.characters.find(c=>c.id==='zhenhong');
  const r06=catalog.cards.find(c=>c.id==='R06');
  assert.equal(api.referencedCards({name:hero.name,description:hero.passive})[0].id,'mechanism-solitary');
  const mechanism=api.referencedCards(r06)[0];
  assert.equal(mechanism.name,'独行');
  assert.match(mechanism.description,/回复真红的全部生命/);
  const source=fs.readFileSync('app/modules/card_game/static/js/v2_table/interaction.js','utf8');
  const renderer=source.slice(source.indexOf('    function hoverCardMarkup('),source.indexOf('    function hoverCharacterMarkup('));
  const context=vm.createContext({V2:api,render:null});
  vm.runInContext(renderer+'\nrender = hoverCardMarkup;',context);
  const markup=context.render(mechanism);
  assert.match(markup,/v2-hover-card is-mechanism/);
  assert.doesNotMatch(markup,/v2-hover-art|真红.webp|v2-card-owner|<img/);
  assert.match(markup,/机制说明/);
  assert.doesNotMatch(markup,/v2-frame-stats|v2-cost-badge/);
});

test('summons use their own portrait and pact uses the debuff icon', ()=>{
  const context=vm.createContext({window:{matchMedia:()=>({matches:false})},
    document:{getElementById:()=>null},requestAnimationFrame(){}});
  for (const file of ['v2_api.js','v2_table/helpers.js','v2_table/render.js'])
    vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/'+file,'utf8'),context);
  const node={style:{},classList:{toggle(){},remove(){}},setAttribute(){},getAttribute(){},removeAttribute(){},
    querySelector:()=>null,querySelectorAll:()=>[]};
  const render=context.window.NTE_V2_TABLE.fillCharacterCard;
  render(node,{id:'summon_1',summoned:true,name:'鬼郎丸',hp:4,max_hp:4,
    portrait:'/static/images/characters/portrait/鬼郎丸.webp'},{side:'a',usePortrait:true});
  assert.match(node.innerHTML,/鬼郎丸.webp/);
  assert.doesNotMatch(node.innerHTML,/⌛/);
  render(node,{id:'summon_2',summoned:true,name:'塔吉多',hp:4,max_hp:4},{side:'a',usePortrait:true});
  assert.match(node.innerHTML,/⌛/);
  render(node,{id:'zero',name:'零',hp:5,max_hp:5,pact:true,
    effect_markers:[{id:'pact',name:'枚约',kind:'debuff',description:'清算',clears_on_down:true}]},
    {side:'b',usePortrait:true});
  assert.match(node.innerHTML,/v2-effect-icon is-debuff/);
  assert.doesNotMatch(node.innerHTML,/<span class="">枚约<\/span>/);
});
