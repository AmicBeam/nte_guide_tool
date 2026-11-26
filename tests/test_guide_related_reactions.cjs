const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('app/modules/guide/static/related_axes.js','utf8');
const settle = async () => { for(let i=0;i<6;i++) await new Promise(resolve=>setImmediate(resolve)); };
async function page({logged=true, owner=false, visibility='public'}={}) {
  const nodes = new Map();
  const get = id => {
    if (!nodes.has(id)) nodes.set(id,{value:'',innerHTML:'',textContent:'',dataset:{},
      setAttribute(){},addEventListener(){},add(){},replaceChildren(){this.innerHTML='';}});
    return nodes.get(id);
  };
  get('section-related-axes').dataset={guideCharacterId:'1042',guideCharacterName:'黑羽'};
  get('related-scope').value='public';get('related-sort').value='new';
  let expired=false, fail=false, hold=null;
  const item={id:17,title:'合成排轴',visibility,is_owner:owner,team:[],liked:false,disliked:false,favorited:false,
    like_count:0,dislike_count:0,favorite_count:0};
  const calls=[];
  const fetch=async(url,options={})=>{
    calls.push({url,options});
    let payload={}, status=200;
    if(url==='/api/me') {
      if(hold) await hold;
      if(expired) {status=401;payload={error:'expired'};}
    } else if(url==='/api/shaft/catalog') payload={characters:[{id:'heiyu',name:'黑羽'}]};
    else if(url.startsWith('/api/shaft/market') || url.startsWith('/api/shaft/me/')) {
      payload={items:url.startsWith('/api/shaft/me/favorites')&&!item.favorited?[]:[{...item}]};
    } else {
      if(fail) {status=500;payload={error:'合成失败'};}
      else {
        const kind=url.split('/').at(-1), active=options.method==='POST';
        const flag={like:'liked',dislike:'disliked',favorite:'favorited'}[kind];
        item[flag]=active;
        if(active && kind==='like') item.disliked=false;
        if(active && kind==='dislike') item.liked=false;
        item.like_count=Number(item.liked);item.dislike_count=Number(item.disliked);item.favorite_count=Number(item.favorited);
        payload={...item};
      }
    }
    return {status,ok:status<400,json:async()=>payload};
  };
  vm.runInNewContext(source,{document:{getElementById:get,querySelector:()=>null},
    localStorage:{getItem:()=>logged?'synthetic-token':''},location:{pathname:'/guide/characters/1042/builds',search:'',hash:''},
    window:{addEventListener(){}},fetch,URLSearchParams,Option:function(){}});
  await settle();
  const click=kind=>get('related-list').onclick({target:{closest:selector=>selector==='[data-axis-reaction]'?{dataset:{axisReaction:kind,reactionAxis:'17'}}:null}});
  return {get,calls,item,click,expire:()=>{expired=true;},fail:()=>{fail=true;},hold:promise=>{hold=promise;}};
}
test('anonymous and private cards do not offer reaction controls',async()=>{
  const anonymous=await page({logged:false});
  assert.doesNotMatch(anonymous.get('related-list').innerHTML,/data-axis-reaction=/);
  assert.match(anonymous.get('related-list').innerHTML,/登录后评价与收藏/);
  anonymous.click('favorite');await settle();
  assert.equal(anonymous.calls.filter(c=>c.options.method==='POST').length,0);
  const privatePage=await page({visibility:'private'});
  assert.doesNotMatch(privatePage.get('related-list').innerHTML,/data-axis-reaction=/);
});
test('own axes cannot be newly favorited',async()=>{
  const own=await page({owner:true});own.click('favorite');await settle();
  assert.equal(own.calls.filter(c=>/\/favorite$/.test(c.url)).length,0);
  assert.match(own.get('related-list').innerHTML,/不能收藏自己的排轴/);
});
test('like/dislike toggle mutually, and favorite can be cancelled in favorites scope',async()=>{
  const p=await page();
  p.click('like');await settle();assert.equal(p.item.liked,true);
  p.click('dislike');await settle();assert.equal(p.item.liked,false);assert.equal(p.item.disliked,true);
  p.click('dislike');await settle();assert.equal(p.item.disliked,false);
  p.click('like');await settle();p.click('like');await settle();assert.equal(p.item.liked,false);
  p.click('favorite');await settle();assert.equal(p.item.favorited,true);
  p.get('related-scope').value='favorites';p.get('related-scope').onchange();await settle();
  p.click('favorite');await settle();assert.equal(p.item.favorited,false);
  assert.equal(p.get('related-list').innerHTML,'');
  const writes=p.calls.filter(c=>/\/(like|dislike|favorite)$/.test(c.url));
  assert.equal(writes.at(-1).options.method,'DELETE');
  assert.ok(writes.every(c=>c.options.headers.Authorization==='Bearer synthetic-token'));
  assert.ok(writes.every(c=>!c.options.headers['X-Shaft-Visitor-Key']&&!c.options.body));
});
test('expired login and failed writes preserve state; double clicks make one write',async()=>{
  const expired=await page();expired.expire();expired.click('favorite');await settle();
  assert.equal(expired.calls.filter(c=>/\/favorite$/.test(c.url)).length,0);
  assert.doesNotMatch(expired.get('related-list').innerHTML,/data-axis-reaction=/);
  const failed=await page();failed.fail();failed.click('like');await settle();
  assert.equal(failed.item.liked,false);assert.match(failed.get('related-list').innerHTML,/合成失败/);
  const pending=await page();let release;pending.hold(new Promise(r=>{release=r;}));
  pending.click('like');pending.click('like');await settle();release();await settle();
  assert.equal(pending.calls.filter(c=>/\/like$/.test(c.url)).length,1);
});
