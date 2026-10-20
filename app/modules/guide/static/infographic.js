window.startGuideInfographic=function (data,account,allowed) {
  'use strict';
  const Store=window.GuideInfographicState,Scripts=window.GuideInfographicScript;
  const session=()=>{if(!allowed())throw new Error('请登录后继续编辑。');};
  const $=id=>document.getElementById(id), key=`nte-infographic-v2:${account}:${data.character_id}`;
  let controlsSignature='',renderTask=null,displayedScript='';
  let config=Store.defaults(data),revision=0,history=[],saveTimer,renderNumber=0,ready=Promise.resolve(),exporting=false;
  const status=message=>{$('ig-status').textContent=message;};
  const node=(tag,text)=>{const e=document.createElement(tag);if(text)e.textContent=text;return e;};
  function input(label,path,{choices=null,type='text',limit=200,rows=3}={}) {
    const wrapper=node('label',label),element=node(choices?'select':type==='textarea'?'textarea':'input');
    element.dataset.field=path;
    if(choices) {for(const [id,name]of choices)element.add(new Option(name,id));}
    else if(type==='textarea'){element.rows=rows;element.maxLength=limit;}
    else {element.type=type;element.maxLength=limit;}
    wrapper.append(element);return wrapper;
  }
  function pathValue(path){return path.split('.').reduce((v,k)=>v[k],config);}
  function writePath(next,path,value){const keys=path.split('.'),last=keys.pop();let obj=next;for(const k of keys)obj=obj[k];obj[last]=value;}
  function moduleList() {
    $('ig-modules').replaceChildren(...config.order.map((key,index)=>{
      const row=node('div');row.className='ig-module';const label=node('label'),check=node('input');check.type='checkbox';check.dataset.field=`modules.${key}`;check.checked=config.modules[key];label.append(check,document.createTextNode(data.modules[key]));row.append(label);
      for(const [direction,symbol]of [[-1,'↑'],[1,'↓']]){const b=node('button',symbol);b.type='button';b.setAttribute('aria-label',`${direction===-1?'上移':'下移'}${data.modules[key]}`);b.disabled=index+direction<0||index+direction>=config.order.length;
        b.onclick=()=>{const next=Store.clone(config);[next.order[index],next.order[index+direction]]=[next.order[index+direction],next.order[index]];commit(next);sync();};row.append(b);}return row;
    }));
  }
  for(const [id,title]of Object.entries(data.modules)) {
    const details=node('details');details.dataset.module=id;details.open=['equipment','teams'].includes(id);const summary=node('summary',title),fields=node('div');fields.className='ig-fields';details.append(summary,fields);$('ig-editors').append(details);
    if(id==='equipment') {
      for(let i=0;i<2;i++){
        fields.append(node('h3',i?'备选方案':'主要方案'),input('方案名称',`plans.${i}.name`,{limit:40}),
          input('弧盘',`plans.${i}.arc`,{choices:[['','不添加弧盘'],...data.arcs.map(a=>[a.id,`${a.rarity}星 · ${a.name}`])]}),
          input('核心卡带',`plans.${i}.core`,{choices:[['','不添加卡带'],...data.cores.map(a=>[a.id,a.name])]}),
          input('卡带主词条建议',`plans.${i}.main`),input('副词条优先级',`plans.${i}.sub`),
          input('方案说明 / 适用场景',`plans.${i}.note`,{type:'textarea',limit:400}));
      }
    } else if(id==='teams') {
      for(let i=0;i<2;i++) {
        fields.append(node('h3',i?'备选队伍':'推荐队伍'),input('队伍名称',`teams.${i}.name`,{limit:40}));
        const slots=node('div');slots.id=`ig-team-slots-${i}`;fields.append(slots);
        fields.append(input('队伍职能 / 可替换角色 / 连携思路',`teams.${i}.note`,{type:'textarea',limit:500}));
      }
    } else if(id==='radar') {
      fields.append(node('p','0表示尚未评分；仅为作者主观评价。'));
      data.radar_labels.forEach((label,i)=>{fields.append(input(`维度 ${i+1} 名称`,`radar_labels.${i}`,{limit:12}));const wrapper=input(label,`radar.${i}`,{type:'range'});wrapper.className='ig-score';const slider=wrapper.querySelector('input');slider.min='0';slider.max='100';slider.step='1';wrapper.append(node('output','0'));fields.append(wrapper);});
    } else {
      const labels={stats:'目标面板与适用条件（每行一项）',skills:'技能优先级 / 关键觉醒与共鸣',rotation:'起手、输出循环与操作要点',advice:'适用场景、养成与抽取建议'};
      fields.append(input(labels[id],id,{type:'textarea',limit:Store.limits[id],rows:6}));
    }
  }
  function refreshTeamSlots() {
    config.teams.forEach((team,i)=>{
      const container=$(`ig-team-slots-${i}`),opened=new Map([...container.querySelectorAll('details')].map(e=>[e.dataset.slot,e.open]));
      container.replaceChildren();
      team.members.forEach((primary,j)=>{
        const details=node('details'),summary=node('summary'),contents=node('div');details.dataset.slot=String(j);details.open=opened.get(String(j))??j===0;
        const count=team.candidates[j].filter(Boolean).length;
        summary.textContent=`格子 ${j+1} · ${data.characters.find(c=>c.id===primary)?.name||'未选首选'}${count?` ＋${count}位候选`:''}`;
        contents.className='ig-slot-fields';details.append(summary,contents);container.append(details);
        const main=input('首选角色',`teams.${i}.members.${j}`,{choices:[['','留空'],...data.characters.map(c=>[c.id,`${c.name} · ${c.element}`])]});
        main.querySelector('select').setAttribute('aria-label',`队伍${i+1}格子${j+1}首选`);contents.append(main);
        contents.append(input('折叠候选为小头像',`teams.${i}.folded.${j}`,{type:'checkbox'}));
        const candidates=team.candidates[j];
        candidates.forEach((id,k)=>{
          const row=node('div');row.className='ig-candidate-row';
          const excluded=new Set([primary,...candidates.filter((_,n)=>n!==k)].filter(Boolean));
          const selector=input(`候选 ${k+1}`,`teams.${i}.candidates.${j}.${k}`,{choices:[['','请选择角色'],...data.characters.filter(c=>!excluded.has(c.id)).map(c=>[c.id,c.name])]});
          selector.querySelector('select').setAttribute('aria-label',`队伍${i+1}格子${j+1}候选${k+1}`);
          const remove=node('button','移除');remove.type='button';remove.setAttribute('aria-label',`移除队伍${i+1}格子${j+1}候选${k+1}`);
          remove.onclick=()=>{const next=Store.clone(config);next.teams[i].candidates[j].splice(k,1);commit(next);sync();};
          row.append(selector,remove);contents.append(row);
        });
        const add=node('button','＋ 添加候选');add.type='button';add.setAttribute('aria-label',`队伍${i+1}格子${j+1}添加候选`);
        add.disabled=candidates.length>=data.characters.length-(primary?1:0);
        add.onclick=()=>{const next=Store.clone(config);next.teams[i].candidates[j].push('');commit(next);sync();};contents.append(add);
      });
    });
  }
  function sync() {
    const spec=Scripts.validate(config.script,config.script_params);
    for(const details of document.querySelectorAll('[data-builtin]'))details.hidden=!spec.builtin.includes(details.dataset.builtin);
    const signature=JSON.stringify(spec.controls);
    if(signature!==controlsSignature){
      controlsSignature=signature;$('ig-script-controls').replaceChildren();
      for(const item of spec.controls){const field=input(item.label,`script_params.${item.key}`,{type:item.type==='boolean'?'checkbox':item.type==='text'?'textarea':item.type,limit:2000,choices:item.type==='select'?item.options.map(v=>[v,v]):null});const element=field.querySelector('input,select,textarea');if(item.type==='number')for(const k of ['min','max','step'])if(item[k]!==undefined)element[k]=item[k];$('ig-script-controls').append(field);}
    }
    if(displayedScript!==config.script){$('ig-script-source').value=config.script;displayedScript=config.script;}
    refreshTeamSlots();
    for(const e of $('ig-form').querySelectorAll('[data-field]')) {
      const value=pathValue(e.dataset.field);if(e.type==='checkbox')e.checked=value;else e.value=value;
      if(e.type==='range'&&e.parentElement.querySelector('output'))e.parentElement.querySelector('output').textContent=value;
    }
    config.radar_labels.forEach((label,i)=>{document.querySelector(`[data-field="radar.${i}"]`).parentElement.firstChild.textContent=label;});
    for(const e of $('ig-editors').children)e.hidden=!config.modules[e.dataset.module]||!spec.builtin.includes(e.dataset.module);
    moduleList();$('ig-undo').disabled=!history.length;
  }
  function save(manual=false) {
    try{session();localStorage.setItem(key,JSON.stringify(Store.pack(config,data)));if(manual)status('草稿已保存到当前浏览器。');}
    catch{status('浏览器无法保存草稿，请导出配置文件保留修改。');}
  }
  function redraw() {
    session();renderTask?.cancel?.();
    const number=++renderNumber,snapshot=Store.clone(config),buffer=document.createElement('canvas');
    renderTask=window.GuideInfographicRuntime.render(buffer,snapshot,data);
    ready=renderTask.then(result=>{
      if(number!==renderNumber)return;
      const canvas=$('ig-canvas');canvas.width=result.width;canvas.height=result.height;canvas.getContext('2d').drawImage(buffer,0,0);
      $('ig-size').textContent=`${result.width} × ${result.height}`;board.setScene(result);
      if(result.failed)status('部分图片未加载成功，导出前请重试或移除对应图片。');
      return result;
    }).catch(error=>{if(number===renderNumber)status(error.message);throw error;});
    ready.catch(()=>{});
  }
  function commit(next) {
    session();const clean=Store.validate(next,data);history.push(Store.clone(config));if(history.length>50)history.shift();config=clean;revision++;
    $('ig-undo').disabled=false;redraw();clearTimeout(saveTimer);saveTimer=setTimeout(()=>save(),350);
  }
  $('ig-form').onsubmit=event=>event.preventDefault();
  $('ig-form').addEventListener('input',event=>{
    const e=event.target;if(!e.dataset.field)return;
    try {const next=Store.clone(config);if(['orientation','layout'].includes(e.dataset.field))next.positions={};writePath(next,e.dataset.field,e.type==='checkbox'?e.checked:['range','number'].includes(e.type)?Number(e.value):e.value);if(/^teams\.\d+\.members\.\d+$/.test(e.dataset.field)){
        const [,team,,slot]=e.dataset.field.split('.');next.teams[team].candidates[slot]=next.teams[team].candidates[slot].filter(id=>id!==e.value);
      }
      commit(next);
      if(e.type==='checkbox'||e.dataset.field.startsWith('teams.'))sync();if(e.type==='range'&&e.parentElement.querySelector('output'))e.parentElement.querySelector('output').textContent=e.value;
    }catch(error){status(error.message);}
  });
  const board=window.GuideInfographicBoard.create({getConfig:()=>config,data,status,
    change(next){commit(next);sync();},writeField(path,value){const next=Store.clone(config);writePath(next,path,value);commit(next);sync();}});
  $('ig-auto-layout').onclick=()=>{const next=Store.clone(config);next.positions={};board.close();commit(next);sync();status('已恢复自动排版。');};
  $('ig-save').onclick=()=>save(true);
  $('ig-undo').onclick=()=>{if(!history.length)return;config=history.pop();revision++;sync();redraw();save();status('已撤销上一次修改。');};
  $('ig-reset').onclick=()=>{if(confirm('将当前一图流恢复为初始模板？可使用撤销恢复。')){commit(Store.defaults(data));sync();status('已恢复初始模板。');}};
  function download(blob,filename){const url=URL.createObjectURL(blob),a=node('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);}
  $('ig-download-js').onclick=()=>{session();download(new Blob([config.script],{type:'text/javascript'}),`${data.character_id}-一图流模板.js`);};
  $('ig-export-json').onclick=()=>{session();download(new Blob([JSON.stringify(Store.pack(config,data),null,2)],{type:'application/json'}),`${data.character_id}-一图流模板.json`);status('配置文件已导出，可在此角色页面重新导入。');};
  $('ig-import').onchange=async event=>{
    const file=event.target.files?.[0];if(!file)return;
    try{if(file.size>256*1024)throw new Error('模板文件须小于256KB。');const raw=await file.text();const next=/\.js$/i.test(file.name)?{...Store.clone(config),script:raw,script_params:Scripts.defaults(raw,config.script_params)}:Store.unpack(JSON.parse(raw),data);await verifiedApply(next,revision); status('配置已导入，原内容可撤销恢复。');}
    catch(error){status(error.message || '配置导入失败。');}finally{event.target.value='';}
  };
  async function exportBlob() {
    session();const snapshot=Store.clone(config),rendered=document.createElement('canvas');
    await document.fonts.ready;
    const result=await window.GuideInfographicRuntime.render(rendered,snapshot,data);
    if(result.failed)throw new Error('图片加载失败，请检查网络后重试。');
    const desired=Number($('ig-quality').value),scale=Math.min(desired,8192/result.height,8192/result.width,Math.sqrt(30000000/(result.width*result.height)));
    const output=document.createElement('canvas');output.width=Math.round(result.width*scale);output.height=Math.round(result.height*scale);
    output.getContext('2d').drawImage(rendered,0,0,output.width,output.height);
    const blob=await new Promise((resolve,reject)=>output.toBlob(b=>b?resolve(b):reject(new Error('图片生成失败，请缩短内容后重试。')),'image/png'));
    return {blob,width:output.width,height:output.height};
  }
  async function exportingImage(copy) {
    if(exporting)return;exporting=true;$('ig-export').disabled=true;$('ig-copy').disabled=true;status('正在生成高清图片…');
    try{
      if(copy && (!window.isSecureContext||!navigator.clipboard?.write||!window.ClipboardItem))throw new Error('当前浏览器或访问地址不支持复制图片，请使用导出高清 PNG。');
      // Supply the promise during the click gesture, including on browsers with strict clipboard activation.
      const result=exportBlob();
      if(copy){await navigator.clipboard.write([new ClipboardItem({'image/png':result.then(r=>r.blob)})]);await result;status('高清图片已复制，可粘贴分享。');}
      else{const {blob,width,height}=await result;download(blob,`${data.character_id}-角色一图流.png`);status(`已导出 ${width} × ${height} PNG。`);}
    }catch(error){status(error.message || '生成失败，请改用导出图片。');}
    finally{exporting=false;$('ig-export').disabled=false;$('ig-copy').disabled=false;}
  }
  $('ig-export').onclick=()=>exportingImage(false);$('ig-copy').onclick=()=>exportingImage(true);
  // Only this page installs the bridge. The shared chat widget never exposes it to CLI tools.
  async function verifiedApply(next,expected,signal){
    session();const clean=Store.validate(next,data),test=document.createElement('canvas');
    if(signal?.aborted)throw new Error('已停止，JS未应用。');
    const task=window.GuideInfographicRuntime.render(test,clean,data),cancel=()=>task.cancel();signal?.addEventListener('abort',cancel,{once:true});
    let result;try{result=await task;}finally{signal?.removeEventListener('abort',cancel);}
    if(signal?.aborted)throw new Error('已停止，JS未应用。');if(result.failed)throw new Error('素材未加载成功，JS未应用。');
    session();if(revision!==expected)throw new Error('验证脚本期间画布已变化，本次修改未覆盖。');commit(clean);sync();
  }
  $('ig-apply-script').onclick=async()=>{
    const source=$('ig-script-source').value,expected=revision;
    try{status('正在验证绘图JS…');await verifiedApply({...Store.clone(config),script:source,script_params:Scripts.defaults(source,config.script_params)},expected);status('绘图JS已应用并保存在浏览器，可撤销。');}catch(error){status(error.message);}
  };
  window.GuideInfographicEditor={snapshot:()=>{session();return {character_id:data.character_id,revision,config:Store.clone(config)};},
    async apply(message,{signal}={}){
      if(message.character_id!==data.character_id||message.revision!==revision){status('助手回答期间你已修改画布，本次修改未覆盖。请重新告诉助手要调整的内容。');return false;}
      try{await verifiedApply(message.config,message.revision,signal);status('攻略助手已更新一图流，可使用“撤销修改”恢复。');return true;}
      catch(error){status(`助手修改未应用：${error.message}`);return false;}
    }};
  try{let raw=localStorage.getItem(key);
    const legacy=`nte-infographic-v1:${data.character_id}`,claim=`nte-infographic-claimed:${data.character_id}`;
    if(!raw&&!localStorage.getItem(claim)){raw=localStorage.getItem(legacy);if(raw){localStorage.setItem(claim,account);localStorage.setItem(key,raw);}}
if(raw){config=Store.unpack(JSON.parse(raw),data);status('已恢复此角色的浏览器草稿。');}else status('模板已就绪，编辑后会自动保存草稿。');}
  catch{status('旧草稿无法读取，已打开初始模板。原草稿未覆盖，可重新导入配置。');}
  window.addEventListener('pagehide',()=>renderTask?.cancel?.());
  sync();redraw();
};
