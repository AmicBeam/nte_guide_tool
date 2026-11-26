(function(root, factory) {
  const api = factory(); if (typeof module === 'object' && module.exports) module.exports = api; else root.GuideInfographicState = api;
}(typeof window === 'undefined' ? globalThis : window, function() {
  'use strict';
  const Scripts=typeof module==='object'&&module.exports?require('./infographic_script.js'):window.GuideInfographicScript;
  const modules = ['equipment','stats','radar','skills','teams','rotation','advice'];
  const limits = {title:60,subtitle:100,author:40,version:40,disclaimer:160,stats:650,skills:650,rotation:1200,advice:1000};
  const clone = x => JSON.parse(JSON.stringify(x));
  function defaults(data) {
    const c = data.characters.find(c => c.id === data.character_id);
    return {script:data.default_script||Scripts.fallback,script_params:Scripts.defaults(data.default_script||Scripts.fallback),title:c.name,subtitle:'角色攻略 · 配装与实战',author:'',version:data.game_version,
      disclaimer:'攻略作者个人建议，请结合队伍与实战环境调整。',orientation:'landscape',layout:'editorial',theme:'night',
      portrait:'fullbody',art_scale:1,modules:Object.fromEntries(modules.map(k=>[k,true])),order:[...modules],positions:{},radar_labels:[...(data.radar_labels||['爆发','持续','倾陷','生存','辅助','机动'])],radar:[0,0,0,0,0,0],
      stats:'',skills:'',rotation:'',advice:'',
      plans:[{name:'主要方案',arc:data.recommended_arc,core:data.recommended_core,main:'',sub:'',note:''},
        {name:'备选方案',arc:'',core:'',main:'',sub:'',note:''}],
      teams:[{name:'推荐队伍',members:[c.id,'','',''],candidates:[[],[],[],[]],folded:[true,true,true,true],note:''},{name:'备选队伍',members:['','','',''],candidates:[[],[],[],[]],folded:[true,true,true,true],note:''}]};
  }
  const exact = (v, keys) => v && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length===keys.length && keys.every(k=>Object.hasOwn(v,k));
  const text = (v,limit) => typeof v==='string' && [...v].length<=limit;
  function validate(value,data) {
    const fail = () => {throw new Error('配置格式、长度或选项无效，请检查导入文件或修改内容。');};
    if (!exact(value,Object.keys(defaults(data)))) fail();
    for(const [key,limit] of Object.entries(limits)) if(!text(value[key],limit)) fail();
    for(const [key,choices] of Object.entries({orientation:['landscape','portrait'],layout:['editorial','balanced','compact'],theme:['night','paper','teal'],portrait:['fullbody','portrait','avatar']})) if(!choices.includes(value[key])) fail();
    if(!Number.isFinite(value.art_scale)||value.art_scale<.7||value.art_scale>1.6) fail();
    if(!exact(value.modules,modules)||Object.values(value.modules).some(v=>typeof v!=='boolean')) fail();
    if(!Array.isArray(value.order)||value.order.length!==modules.length||new Set(value.order).size!==modules.length||value.order.some(k=>!modules.includes(k))) fail();
    if(!Array.isArray(value.radar_labels)||value.radar_labels.length!==6||value.radar_labels.some(v=>!text(v,12)||!v.trim())) fail();
    if(!value.positions||typeof value.positions!=='object'||Array.isArray(value.positions)||Object.keys(value.positions).length>64||Object.keys(value.positions).some(k=>!Scripts.identifier(k))) fail();
    for(const pos of Object.values(value.positions)) if(!exact(pos,['x','y'])||![pos.x,pos.y].every(v=>Number.isFinite(v)&&v>=0&&v<=16000)) fail();
    if(!Array.isArray(value.radar)||value.radar.length!==6||value.radar.some(v=>!Number.isInteger(v)||v<0||v>100)) fail();
    if(!Array.isArray(value.plans)||value.plans.length!==2||!Array.isArray(value.teams)||value.teams.length!==2) fail();
    for(const p of value.plans) {
      if(!exact(p,['name','arc','core','main','sub','note'])||![...data.arcs.map(a=>a.id),''].includes(p.arc)||![...data.cores.map(a=>a.id),''].includes(p.core)) fail();
      for(const [k,l] of [['name',40],['main',200],['sub',200],['note',400]]) if(!text(p[k],l)) fail();
    }
    for(const t of value.teams) {
      if(!exact(t,['name','members','candidates','folded','note'])||!text(t.name,40)||!text(t.note,500)||!Array.isArray(t.members)||t.members.length!==4||t.members.some(id=>![...data.characters.map(c=>c.id),''].includes(id))) fail();
      if(!Array.isArray(t.candidates)||t.candidates.length!==4||!Array.isArray(t.folded)||t.folded.length!==4||t.folded.some(v=>typeof v!=='boolean')) fail();
      t.candidates.forEach((ids,i)=>{
        if(!Array.isArray(ids)||ids.length>data.characters.length||ids.some(id=>typeof id!=='string'||![...data.characters.map(c=>c.id),''].includes(id))) fail();
        const selected=[t.members[i],...ids].filter(Boolean);if(new Set(selected).size!==selected.length) fail();
      });
    }
    Scripts.validate(value.script,value.script_params);
    return clone(value);
  }
  function patch(config,changes,data) {
    if(!changes || typeof changes!=='object'||Array.isArray(changes)||Object.keys(changes).some(k=>!Object.hasOwn(config,k))) throw new Error('包含不可编辑的字段。');
    return validate({...config,...changes},data);
  }
  function pack(config,data) {return {format:'nte-character-infographic',version:1,character_id:data.character_id,config:clone(config)};}
  function unpack(raw,data) {
    if(!raw||raw.format!=='nte-character-infographic'||raw.version!==1||raw.character_id!==data.character_id) throw new Error('配置文件不属于当前角色或版本不支持。');
    const config=clone(raw.config);
    // Older browser drafts predate direct canvas editing.
    if(!Object.hasOwn(config,'positions'))config.positions={};
    if(!Object.hasOwn(config,'radar_labels'))config.radar_labels=[...(data.radar_labels||['爆发','持续','倾陷','生存','辅助','机动'])];
    for(const t of config.teams||[]){if(!Object.hasOwn(t,'candidates'))t.candidates=[[],[],[],[]];if(!Object.hasOwn(t,'folded'))t.folded=[true,true,true,true];}
    if(!Object.hasOwn(config,'script'))config.script=data.default_script||Scripts.fallback;
    if(!Object.hasOwn(config,'script_params'))config.script_params=Scripts.defaults(config.script);
    return validate(config,data);
  }
  return {defaults,validate,patch,pack,unpack,modules,limits,clone};
}));
