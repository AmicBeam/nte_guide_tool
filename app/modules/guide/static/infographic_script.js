(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.GuideInfographicScript=api;}(typeof window==='undefined'?globalThis:window,function(){
  'use strict';
  const groups=['canvas','identity','modules','equipment','stats','radar','skills','teams','rotation','advice'];
  const identifier=v=>typeof v==='string'&&/^[A-Za-z][A-Za-z0-9_]{0,39}$/.test(v)&&!['constructor','prototype'].includes(v);
  const fallback='/* @editor {"builtin":[],"controls":[]} */\nwindow.GuideInfographicCanvas={render:async(canvas)=>{canvas.width=1000;canvas.height=800;return {width:1000,height:800};}};';
  function parameter(item,value){
    const valid=item.type==='boolean'?typeof value==='boolean':item.type==='number'?Number.isFinite(value)&&value>=Math.max(-1e12,item.min??-1e12)&&value<=Math.min(1e12,item.max??1e12):item.type==='text'?typeof value==='string'&&[...value].length<=2000:item.type==='color'?typeof value==='string'&&/^#[a-f\d]{6}$/i.test(value):typeof value==='string'&&item.options.includes(value);
    if(!valid)throw new Error('自定义参数值不符合脚本声明。');
  }
  function declaration(source){
    if(typeof source!=='string'||[...source].length>80000)throw new Error('绘图JS须为不超过80000字的文本。');
    const matches=[...source.matchAll(/\/\*\s*@editor\s+([\s\S]*?)\*\//g)];
    if(matches.length!==1||matches[0][1].length>16000)throw new Error('绘图JS须包含一个 @editor JSON参数声明。');
    let spec;try{spec=JSON.parse(matches[0][1]);}catch{throw new Error('@editor参数声明不是有效JSON。');}
    if(!spec||Object.keys(spec).sort().join(',')!=='builtin,controls'||!Array.isArray(spec.builtin)||spec.builtin.length>groups.length||new Set(spec.builtin).size!==spec.builtin.length||spec.builtin.some(k=>!groups.includes(k))||!Array.isArray(spec.controls)||spec.controls.length>32)throw new Error('@editor控件声明无效。');
    const keys=new Set();
    for(const item of spec.controls){
      if(!item||typeof item!=='object'||Array.isArray(item)||Object.keys(item).some(k=>!['key','label','type','default','min','max','step','options'].includes(k))||!['key','label','type','default'].every(k=>Object.hasOwn(item,k))||!identifier(item.key)||keys.has(item.key)||typeof item.label!=='string'||!item.label.length||[...item.label].length>40||!['text','number','color','boolean','select'].includes(item.type))throw new Error('自定义参数字段无效。');
      keys.add(item.key);
      if(item.type==='number'&&(['min','max','step'].some(k=>Object.hasOwn(item,k)&&!Number.isFinite(item[k]))||(item.min??-1e12)>(item.max??1e12)||(item.step??1)<=0))throw new Error('数值参数范围无效。');
      if(item.type==='select'&&(!Array.isArray(item.options)||!item.options.length||item.options.length>50||item.options.some(v=>typeof v!=='string'||[...v].length>80)))throw new Error('选择参数options无效。');
      parameter(item,item.default);
    }
    return spec;
  }
  function defaults(source,old={}){const spec=declaration(source);return Object.fromEntries(spec.controls.map(item=>{try{parameter(item,old[item.key]);return [item.key,old[item.key]];}catch{return [item.key,item.default];}}));}
  function validate(source,params){const spec=declaration(source);if(!params||typeof params!=='object'||Array.isArray(params)||Object.keys(params).sort().join(',')!==spec.controls.map(i=>i.key).sort().join(','))throw new Error('脚本参数须与 @editor 声明一致。');for(const item of spec.controls)parameter(item,params[item.key]);return spec;}
  return {declaration,validate,defaults,identifier,fallback};
}));
