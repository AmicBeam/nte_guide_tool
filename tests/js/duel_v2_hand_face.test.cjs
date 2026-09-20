const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

test('view-as keeps physical hand face, effective art and cost, and character references', async () => {
  const catalog = JSON.parse(fs.readFileSync('app/modules/card_game/content/duel_v2/catalog.json', 'utf8'));
  const context = {window: {localStorage:{getItem:()=>''}}, fetch: async()=>({ok:true,status:200,headers:{get:()=>''},json:async()=>catalog})};
  vm.createContext(context);
  for (const file of ['v2_api.js','v2_table/helpers.js','v2_table/render.js']) vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/'+file,'utf8'),context);
  const api=context.window.NTE_V2, table=context.window.NTE_V2_TABLE;
  await api.getCatalog();
  const original={card_id:'I04',name:'划定常规的决议',description:'原牌说明',character_id:'yi',type:'tactic'};
  const card={instance_id:'c1',card_id:'RF01',character_id:'zhenhong',name:'升腾之赤',description:'有效效果',type:'battle',attack:2,cost:1,action_point_free:false,hand_face:original};
  const html=table.handCardHtml(card,{legal:true});
  assert.match(html,/划定常规的决议/); assert.match(html,/原牌说明/);
  assert.match(html,/data-card-type="tactic"/); assert.match(html,/真红.webp/);
  assert.doesNotMatch(html,/有效效果|<em>\+2<\/em>|is-ap-free/);
  assert.equal(api.handCardFace(card).cost,1);
  assert.equal(card.type,'battle'); assert.equal(card.attack,2);
  assert.doesNotMatch(table.handCardHtml({...card,hidden:true}),/划定|真红.webp/);
  const hero=catalog.characters.find(c=>c.id==='zhenhong');
  const refs=api.referencedCards({name:hero.name,description:[hero.passive,hero.awakened_passive].join('。')});
  assert.ok(refs.some(c=>c.id==='RF01' && c.name==='升腾之赤' && c.attack===4 && c.shield===4));
  assert.match(hero.awakened_passive,/真红•「升腾之赤」/);
  const sorted=table.sortHandCards([card,{instance_id:'c2',character_id:'zero',type:'form'}],{characters:[{id:'zero'},{id:'yi'},{id:'zhenhong'}]});
  assert.equal(sorted[0].instance_id,'c2');
});
