const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const script = fs.readFileSync('app/modules/guide/static/article_editor.js', 'utf8');
const ident = '12345678-1234-4234-8234-123456789abc';
class Element {
  constructor(tag='div') { this.tag=tag;this.children=[];this.listeners={};this.value='';this.hidden=true;this.textContent='';this.disabled=false;this.dataset={};this.selectionStart=0;this.selectionEnd=0; }
  addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
  async emit(name) { for (const fn of this.listeners[name] || []) await fn({preventDefault(){}}); }
  append(child) { this.children.push(child); }
  replaceChildren(...children) { this.children = children; }
  get options() { return this.children.flatMap(child => child.tag === 'option' ? [child] : child.options); }
  focus() {}
  setRangeText(text, start, end) { this.value=this.value.slice(0,start)+text+this.value.slice(end); }
  setSelectionRange(start,end) { this.selectionStart=start;this.selectionEnd=end; }
}
const tick = async () => { for(let i=0;i<12;i++) await new Promise(resolve=>setImmediate(resolve)); };
async function harness(storage=new Map([['nte_token','synthetic-token']]), remote={row:null, puts:[], mode:'normal', release:null}) {
  const elements = new Map();
  const element = name => { if (!elements.has(name)) elements.set(name,new Element());return elements.get(name); };
  element('article-target').append(new Element('option'));
  const timers = new Map();let timerId=0;
  const events={};
  const location = new URL('http://example.test/guide/write?id='+ident);
  const reply=(status,payload)=>({ok:status<400,status,json:async()=>payload});
  const fetch = async (path, options={}) => {
    if(path==='/api/me')return reply(200,{player_uid:'synthetic-author'});
    if(path==='/api/guide/article-targets')return reply(200,{items:[]});
    if(path==='/api/guide/me/articles')return reply(200,{items:remote.row?[remote.row]:[]});
    if(path==='/api/guide/articles/preview')return reply(200,{html:'<p>preview</p>',toc:[]});
    if(options.method==='PUT') {
      const value=JSON.parse(options.body);remote.puts.push(value);
      if(remote.mode==='offline')throw new TypeError('network failed');
      if(remote.mode==='delay') { remote.mode='normal';await new Promise(resolve=>{remote.release=resolve;}); }
      if(value.revision!==(remote.row?.revision||0))return reply(409,{error:{message:'另一页面已保存'}});
      remote.row={...value,id:ident,revision:value.revision+1};return reply(200,{data:{...remote.row}});
    }
    return remote.row ? reply(200,{data:{...remote.row}}) : reply(404,{error:{message:'not found'}});
  };
  const context={console,URL,URLSearchParams,AbortController,location,
    document:{getElementById:element,createElement:tag=>new Element(tag),querySelectorAll:()=>[],addEventListener(){}},
    localStorage:{getItem:key=>storage.get(key)||null,setItem:(key,value)=>storage.set(key,value)},
    history:{replaceState(_a,_b,url){ location.href=String(url); }},
    window:{guideAuth:{ready:Promise.resolve('synthetic-token')},crypto:{randomUUID:()=>ident},addEventListener(name,fn){events[name]=fn;}},
    fetch,setTimeout(fn,delay){timers.set(++timerId,{fn,delay});return timerId;},clearTimeout(id){timers.delete(id);}};
  vm.runInNewContext(script,context);await tick();
  return {element,remote,storage,events,async timers(delay){for(const [id,value] of [...timers])if(value.delay===delay){timers.delete(id);value.fn();}await tick();},
    async type(name,value){element('article-'+name).value=value;await element('article-'+name).emit('input');await tick();}};
}

test('failed autosave keeps a local backup and resumes when writes recover',async()=>{
  const h=await harness();h.remote.mode='offline';
  await h.type('title','本机草稿');await h.type('markdown','# 离线正文');await h.timers(800);
  assert.equal(h.remote.row,null);
  assert.match(h.element('article-save-state').textContent,/本机备份已保留/);
  const backup=JSON.parse(h.storage.get('nte-guide-article:synthetic-author:'+ident));
  assert.equal(backup.markdown,'# 离线正文');
  h.remote.mode='normal';
  const reloaded=await harness(h.storage,h.remote);
  assert.equal(reloaded.element('article-markdown').value,'# 离线正文');
  await reloaded.timers(800);
  assert.equal(h.remote.row.markdown,'# 离线正文');
  assert.equal(h.remote.row.revision,1);
});

test('edits while a save is pending are serialized with the returned revision',async()=>{
  const h=await harness();h.remote.mode='delay';
  await h.type('title','第一版');await h.timers(800);
  assert.equal(h.remote.puts.length,1);
  await h.type('markdown','保存途中继续输入的最新版');await h.timers(800);
  assert.equal(h.remote.puts.length,1);
  h.remote.release();await tick();
  assert.equal(h.remote.puts.length,2);
  assert.equal(h.remote.puts[1].revision,1);
  assert.equal(h.remote.row.markdown,'保存途中继续输入的最新版');
  assert.equal(h.remote.row.revision,2);
});

test('another-tab conflict preserves local work and asks which version to restore',async()=>{
  const h=await harness();await h.type('title','初始标题');await h.timers(800);
  h.remote.row={...h.remote.row,title:'其他标签页',revision:2};
  await h.type('title','本页尚未保存');await h.timers(800);
  assert.equal(h.remote.row.title,'其他标签页');
  assert.equal(h.element('article-reload').hidden,false);
  await h.element('article-reload').emit('click');await tick();
  assert.equal(h.element('article-recovery').hidden,false);
  assert.equal(h.element('article-title').value,'其他标签页');
  await h.element('article-restore').emit('click');await h.timers(800);
  assert.equal(h.remote.row.title,'本页尚未保存');
  assert.equal(h.remote.row.revision,3);
});
