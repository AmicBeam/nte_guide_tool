const test = require('node:test');
const assert = require('node:assert/strict');
const state = require('../app/modules/guide/static/chat_state.js');
test('six hours creates a new context only after the boundary',()=>{
 const session={messages:[{timestamp:1000}]};
 assert.equal(state.expired(session,1000+state.GAP_MS),false);
 assert.equal(state.expired(session,1001+state.GAP_MS),true);
 assert.equal(state.expired({messages:[]},999999999),false);
});
test('bounded model window preserves stored messages and excludes page metadata',()=>{
 const messages=Array.from({length:40},(_,i)=>({role:i%2?'assistant':'user',text:'message '+i,timestamp:i,images:[]}));
 messages[39].page_title='temporary page';
 const before=JSON.stringify(messages),context=state.context(messages);
 assert.equal(context.length,32);assert.equal(context[0].text,'message 8');
 assert.equal('page_title' in context[31],false);assert.equal(JSON.stringify(messages),before);
});
test('latest six images are sent while old local images remain intact',()=>{
 const messages=Array.from({length:8},(_,i)=>({role:'user',text:'image '+i,timestamp:i,images:[{data:'data:'+i}]}));
 const context=state.context(messages);
 assert.equal(context.flatMap(m=>m.images).length,6);
 assert.match(context[0].text,/早前图片/);assert.deepEqual(context[7].images,['data:7']);
 assert.equal(messages[0].images.length,1);
});
test('long answers stay within the server limit and retain references for follow-up',()=>{
 const messages=[{role:'assistant',text:'长'.repeat(24000),timestamp:1,
   sources:[{title:'配队攻略',url:'https://www.taptap.cn/moment/123456789012345678'}]}];
 const context=state.context(messages);
 assert.equal(context[0].text.length,12000);
 assert.match(context[0].text,/配队攻略: https:\/\/www.taptap.cn\/moment\/123456789012345678/);
 assert.equal(messages[0].text.length,24000);
 messages[0].sources=Array.from({length:20},()=>({title:'标题'.repeat(100),url:'https://example.com/'+ 'x'.repeat(600)}));
 assert.equal(state.context(messages)[0].text.length,12000);
});
test('interrupted answers stay marked when included in the next model context',()=>{
 const messages=[{role:'assistant',text:'未说完',timestamp:1,incomplete:true}];
 assert.match(state.context(messages)[0].text,/上次回答中断/);
 assert.equal(messages[0].text,'未说完');
});
