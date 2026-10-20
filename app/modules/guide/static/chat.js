(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const widget = $('guide-chat-widget');
  if (!widget) return;
  const Store = window.GuideChatState;
  const state = {user:'', token:'', db:null, sessions:[], current:null, attachments:[], open:false, busy:false,
    controller:null, available:false, checking:null, queue:Promise.resolve(), epoch:0, storageError:''};
  const token = () => { try { return localStorage.getItem('nte_token') || ''; } catch { return ''; } };
  const status = text => { $('guide-chat-status').textContent = text || state.storageError; };
  const node = (tag, text, className) => { const element = document.createElement(tag); if(text) element.textContent=text; if(className) element.className=className; return element; };
  function safeUrl(value) {
    try {
      if (typeof value !== 'string' || /[\u0000-\u0020]/.test(value)) return null;
      const url = new URL(value, location.origin);
      return ['https:','http:'].includes(url.protocol) && !url.username && !url.password ? url.href : null;
    } catch { return null; }
  }
  function link(title, href) {
    const url = safeUrl(href); if (!url) return node('span',title);
    const element = node('a',title);element.href=url;element.target='_blank';element.rel='noopener noreferrer';return element;
  }
  function richText(element, text) {
    const regex = /\[([^\]\n]+)\]\(([^)\s]+)\)|\*\*([^*\n]+)\*\*|`([^`\n]+)`/g;
    let previous=0;
    for (const match of text.matchAll(regex)) {
      element.append(document.createTextNode(text.slice(previous,match.index)));
      element.append(match[1] ? link(match[1],match[2]) : node(match[3] ? 'strong':'code',match[3] || match[4]));
      previous=match.index+match[0].length;
    }
    element.append(document.createTextNode(text.slice(previous)));
  }
  function fit() {
    const mobile = matchMedia('(max-width:650px)').matches;
    document.documentElement.classList.toggle('guide-chat-mobile-open',state.open && mobile);
    $('guide-chat-panel').setAttribute('aria-modal',String(mobile));
    if (mobile) widget.style.setProperty('--guide-chat-height',`${window.visualViewport?.height || innerHeight}px`);
  }
  function open(value) {
    state.open=value;$('guide-chat-panel').hidden=!value;
    $('guide-chat-launcher').setAttribute('aria-expanded',String(value));fit();
    if(value) $('guide-chat-input').focus();
  }
  function controls() {
    $('guide-chat-send').disabled=state.busy || !state.available || !state.current;
    $('guide-chat-send').hidden=state.busy;$('guide-chat-stop').hidden=!state.busy;
    for (const id of ['guide-chat-new','guide-chat-clear','guide-chat-session','guide-chat-files']) $(id).disabled=state.busy || !state.current;
  }
  function persist() {
    const session=structuredClone(state.current);
    if (!session || !state.db) return Promise.resolve();
    state.queue=state.queue.catch(()=>{}).then(()=>Store.save(state.db,session)).catch(error=>{state.storageError=error.message;status(error.message);});
    return state.queue;
  }
  function sessionOptions() {
    const selected=state.current?.id;
    state.sessions.sort((a,b)=>b.updatedAt-a.updatedAt);
    $('guide-chat-session').replaceChildren(...state.sessions.map(session => {
      const date=new Date(session.updatedAt).toLocaleString('zh-CN',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'});
      const option=new Option(`${session.title || '新对话'} · ${date}`,session.id);option.selected=session.id===selected;return option;
    }));
  }
  function render() {
    const container=$('guide-chat-messages');container.replaceChildren();
    if (!state.current?.messages.length) container.append(node('p',window.GuideInfographicEditor ? '可以直接告诉我怎么改这张一图流，例如：改成竖排、隐藏能力雷达、调整配装建议。修改后可撤销；也可以照常查询攻略。' : '可以问我角色怎么玩、怎么配队，查阅技能、在线排轴或TapTap攻略。也可以发送截图一起分析。','guide-chat-empty'));
    for (const message of state.current?.messages || []) {
      const article=node('article','',`guide-chat-message ${message.role}`);
      article.append(node('small',message.role==='user'?'你':message.incomplete?'攻略助手 · 未完成':'攻略助手','guide-chat-message-label'));
      const body=node('div','','guide-chat-message-body');richText(body,message.text);article.append(body);
      if (message.images?.length) {
        const images=node('div','','guide-chat-message-images');
        for(const attachment of message.images) { const img=node('img');img.src=attachment.data;img.alt=attachment.name || '发送的图片';images.append(img); }
        article.append(images);
      }
      if(message.sources?.length) {
        const sources=node('div','','guide-chat-sources');
        for(const source of message.sources) sources.append(link(source.title,source.url));article.append(sources);
      }
      container.append(article);
    }
    container.scrollTop=container.scrollHeight;sessionOptions();controls();
  }
  function fresh() {
    const now=Date.now();state.current={user:state.user,id:crypto.randomUUID?.() || `${now}-${Math.random().toString(36).slice(2)}`,title:'',updatedAt:now,messages:[]};
    state.sessions.push(state.current);state.attachments=[];renderAttachments();render();persist();
  }
  function renderAttachments() {
    $('guide-chat-attachments').replaceChildren(...state.attachments.map((attachment,index)=>{
      const span=node('span','','guide-chat-attachment');const img=node('img');img.src=attachment.data;img.alt=attachment.name;
      const button=node('button','×');button.type='button';button.setAttribute('aria-label',`移除图片 ${attachment.name}`);
      button.onclick=()=>{state.attachments.splice(index,1);renderAttachments();};span.append(img,button);return span;
    }));
  }
  async function compress(file) {
    if(!['image/png','image/jpeg','image/webp'].includes(file.type) || file.size>12*1024*1024) throw new Error('请选择不超过12MB的PNG、JPEG或WebP图片。');
    const url=URL.createObjectURL(file);
    try {
      const img=new Image();img.src=url;await img.decode();
      if(img.width*img.height>40000000) throw new Error('图片尺寸过大，请先缩小。');
      const scale=Math.min(1,1800/Math.max(img.width,img.height));const canvas=document.createElement('canvas');
      canvas.width=Math.max(1,Math.round(img.width*scale));canvas.height=Math.max(1,Math.round(img.height*scale));
      const context=canvas.getContext('2d');context.fillStyle='#fff';context.fillRect(0,0,canvas.width,canvas.height);context.drawImage(img,0,0,canvas.width,canvas.height);
      let data=canvas.toDataURL('image/jpeg',.85);
      if(data.length>1400000) data=canvas.toDataURL('image/jpeg',.6);
      if(data.length>2000000) throw new Error('图片压缩后仍过大，请裁剪后重试。');
      return {name:file.name || '粘贴的截图',data};
    } finally {URL.revokeObjectURL(url);}
  }
  async function attach(files) {
    if(state.busy) return;const epoch=state.epoch;
    try {
      for(const file of files) {const image=await compress(file);if(epoch!==state.epoch) return;
        if(state.attachments.length>=2) throw new Error('每条消息最多附加2张图片。');state.attachments.push(image);}
      status('图片已就绪，发送后会一起交给模型读取。');
    } catch(error) {status(error.message);} finally {renderAttachments();$('guide-chat-files').value='';}
  }
  async function packet(response) {
    try {const value=await response.json();return typeof value.error==='string'?value.error:value.error?.message || '请求失败，请重试。';}
    catch{return '服务暂不可用，请稍后重试。';}
  }
  async function send(event) {
    event?.preventDefault();if(state.busy || !state.available || !state.current) return;
    if(token()!==state.token) {await authenticate();return;}
    const text=$('guide-chat-input').value.trim();if(!text && !state.attachments.length) return;
    const attachments=state.attachments;state.attachments=[];
    if(Store.expired(state.current)) fresh();
    const session=state.current;const epoch=state.epoch;
    session.messages.push({role:'user',text,timestamp:Date.now(),images:attachments});
    session.updatedAt=Date.now();if(!session.title) session.title=(text || '图片问答').slice(0,24);
    state.busy=true;state.controller=new AbortController();
    $('guide-chat-input').value='';renderAttachments();render();await persist();status('正在连接攻略助手…');
    let draft='',draftArticle=null,draftBody=null;
    const resetDraft=()=>{draft='';draftArticle?.remove();draftArticle=null;draftBody=null;};
    try {
      const response=await fetch('/api/guide/chat',{method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${state.token}`},
        body:JSON.stringify({messages:Store.context(session.messages),page:location.pathname+location.search,
          page_title:document.title,...(window.GuideInfographicEditor ? {infographic:window.GuideInfographicEditor.snapshot()} : {})}),signal:state.controller.signal});
      if(!response.ok) {if(response.status===401) {open(false);widget.hidden=true;state.user='';}throw new Error(await packet(response));}
      const reader=response.body.getReader();const decoder=new TextDecoder();let pending='',answered=false,problem='',editorStatus='';
      const accept=line=>{
        if(!line.trim()) return;const value=JSON.parse(line);
        if(epoch!==state.epoch || state.current!==session) return;
        if(value.type==='infographic_patch' && window.GuideInfographicEditor) {
          editorStatus=window.GuideInfographicEditor.apply(value)?'一图流已更新，可在编辑页撤销。':'画布已发生变化，助手修改未覆盖，请重新发送修改要求。';
        }
        if(value.type==='status') status(value.text);
        if(value.type==='error') problem=value.text;
        if(value.type==='answer_reset') resetDraft();
        if(value.type==='answer_delta' && !answered) {
          const text=String(value.text || '');draft+=text;
          const container=$('guide-chat-messages');
          const follow=container.scrollHeight-container.scrollTop-container.clientHeight<70;
          if(!draftArticle) {
            draftArticle=node('article','','guide-chat-message assistant');
            draftArticle.append(node('small','攻略助手 · 正在回答','guide-chat-message-label'));
            draftBody=node('div','','guide-chat-message-body');draftArticle.append(draftBody);container.append(draftArticle);
          }
          draftBody.replaceChildren();richText(draftBody,draft);status('正在回答…');
          if(follow) container.scrollTop=container.scrollHeight;
        }
        if(value.type==='answer' && !answered) {
          resetDraft();
          answered=true;session.messages.push({role:'assistant',text:String(value.text),timestamp:Date.now(),sources:value.sources || []});
          session.updatedAt=Date.now();persist();render();status(editorStatus);
        }
      };
      while(true) {const chunk=await reader.read();if(chunk.done) break;pending+=decoder.decode(chunk.value,{stream:true});let newline;
        while((newline=pending.indexOf('\n'))>=0) {accept(pending.slice(0,newline));pending=pending.slice(newline+1);}}
      pending+=decoder.decode();if(pending.trim()) accept(pending);
      if(!answered) throw new Error(problem || '回答未完成，请重新发送问题。');
    } catch(error) {if(epoch===state.epoch) {
      if(draft) {
        session.messages.push({role:'assistant',text:draft,timestamp:Date.now(),incomplete:true});
        session.updatedAt=Date.now();persist();render();
      }
      status(error.name==='AbortError'?'已停止，已收到的内容保存在浏览器，可继续提问。':error.message);
    }}
    finally {if(epoch===state.epoch) {state.busy=false;state.controller=null;controls();}}
  }
  async function authenticate() {
    if(state.checking) return state.checking;
    state.checking=(async()=>{
      const current=token();if(current===state.token && state.user) return;
      state.epoch++;state.controller?.abort();state.busy=false;state.token=current;state.user='';state.current=null;state.sessions=[];state.attachments=[];open(false);widget.hidden=true;
      if(!current) return;
      try {
        const response=await fetch('/api/me',{headers:{Authorization:`Bearer ${current}`},cache:'no-store'});
        if(!response.ok || token()!==current) return;
        const player=await response.json();if(!player.player_uid) return;
        state.user=String(player.player_uid);
        state.db=state.db || await Store.open();state.sessions=await Store.list(state.db,state.user);
        state.sessions.sort((a,b)=>b.updatedAt-a.updatedAt);state.current=state.sessions[0] || null;
        if(!state.current) fresh();else render();
        const config=await fetch('/api/guide/chat/config',{headers:{Authorization:`Bearer ${current}`},cache:'no-store'});
        const options=await config.json();state.available=config.ok && options.available;controls();widget.hidden=false;
        status(state.available?'':'对话服务尚未配置，请联系管理员。');
      } catch(error) {widget.hidden=!state.user;state.available=false;controls();status(error.message || '浏览器无法保存对话。');}
    })().finally(()=>{state.checking=null;});return state.checking;
  }
  $('guide-chat-launcher').onclick=()=>open(!state.open);$('guide-chat-close').onclick=()=>open(false);
  $('guide-chat-form').onsubmit=send;$('guide-chat-stop').onclick=()=>state.controller?.abort();
  $('guide-chat-new').onclick=()=>{fresh();status('已开启新对话，旧记录仍保存在历史列表中。');};
  $('guide-chat-session').onchange=event=>{state.current=state.sessions.find(session=>session.id===event.target.value);state.attachments=[];renderAttachments();render();status('');};
  $('guide-chat-clear').onclick=async()=>{
    if(!state.current || !confirm('删除当前浏览器中的这段对话？')) return;
    try {await state.queue;await Store.remove(state.db,state.user,state.current.id);state.sessions=state.sessions.filter(s=>s.id!==state.current.id);
      state.current=state.sessions[0] || null;if(state.current) render();else fresh();status('对话已删除。');} catch(error) {status(error.message);}
  };
  $('guide-chat-files').onchange=event=>attach([...event.target.files]);
  $('guide-chat-input').addEventListener('paste',event=>{const files=[...(event.clipboardData?.files || [])];if(files.length) {event.preventDefault();attach(files);}});
  $('guide-chat-input').addEventListener('keydown',event=>{
    if(event.key==='Enter' && !event.isComposing && !event.shiftKey && (event.ctrlKey || event.metaKey || !matchMedia('(max-width:650px)').matches)) {event.preventDefault();send();}
  });
  document.addEventListener('keydown',event=>{if(event.key==='Escape' && state.open) {open(false);$('guide-chat-launcher').focus();}});
  window.addEventListener('storage',event=>{if(event.key==='nte_token') authenticate();});
  window.addEventListener('pageshow',()=>authenticate());window.addEventListener('focus',()=>authenticate());
  window.addEventListener('resize',fit);window.visualViewport?.addEventListener('resize',fit);
  authenticate();
}());
