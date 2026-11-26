(function(){
  'use strict';
  const images=new Map();
  function assets(data){return new Set([...data.characters.flatMap(c=>[...Object.values(c.assets),c.element_icon,c.rarity_icon]),...data.arcs.map(a=>a.image),...data.cores.map(a=>a.image)].filter(v=>typeof v==='string'&&v.startsWith('/static/')));}
  function load(src){
    if(!images.has(src))images.set(src,new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>{images.delete(src);reject(new Error('本站素材加载失败。'));};image.src=src;}));return images.get(src);
  }
  function scene(raw,width,height,config){
    if(!raw||typeof raw!=='object')return {boards:[],fields:[],slots:[]};
    const box=v=>v&&['x','y','w','h'].every(k=>Number.isFinite(v[k]))&&v.x>=0&&v.y>=0&&v.w>0&&v.h>0&&v.x+v.w<=width+1&&v.y+v.h<=height+1;
    const path=value=>{
      if(typeof value!=='string'||value.length>180||!value.split('.').every(k=>/^(?:[A-Za-z][A-Za-z0-9_]*|\d+)$/.test(k)&&!['constructor','prototype','__proto__'].includes(k)))return false;
      let ref=config;for(const k of value.split('.')){if(!ref||!Object.hasOwn(ref,k))return false;ref=ref[k];}return typeof ref==='string';
    };
    const unique=new Set();
    const boards=(Array.isArray(raw.boards)?raw.boards:[]).slice(0,64).filter(v=>box(v)&&window.GuideInfographicScript.identifier(v.key)&&!unique.has(v.key)&&(unique.add(v.key),true)).map(v=>({key:v.key,label:typeof v.label==='string'?v.label.slice(0,80):v.key,x:v.x,y:v.y,w:v.w,h:v.h}));
    const fields=(Array.isArray(raw.fields)?raw.fields:[]).slice(0,256).filter(v=>box(v)&&path(v.path)).map(v=>({path:v.path,x:v.x,y:v.y,w:v.w,h:v.h,size:Number.isFinite(v.size)?Math.max(8,Math.min(160,v.size)):20}));
    const slots=(Array.isArray(raw.slots)?raw.slots:[]).slice(0,8).filter(v=>box(v)&&/^teams\.[01]\.folded\.[0-3]$/.test(v.path)).map(v=>({path:v.path,label:String(v.label||'候选').slice(0,80),folded:!!v.folded,x:v.x,y:v.y,w:v.w,h:v.h}));
    return {boards,fields,slots};
  }
  function render(canvas,config,data){
    window.GuideInfographicScript.validate(config.script,config.script_params);
    const frame=document.createElement('iframe');frame.sandbox='allow-scripts';frame.hidden=true;frame.src='/static/guide/infographic_sandbox.html';
    const channel=crypto.randomUUID?.()||Math.random().toString(36).slice(2),allowed=assets(data);let done=false,count=0,timeout;
    let cancel;
    const promise=new Promise((resolve,reject)=>{
      const finish=(error,result)=>{if(done)return;done=true;clearTimeout(timeout);window.removeEventListener('message',receive);frame.contentWindow?.postMessage({type:'cancel',channel},'*');setTimeout(()=>frame.remove(),0);error?reject(error):resolve(result);};
      cancel=()=>finish(new Error('绘图任务已取消。'));
      async function receive(event){
        if(event.source!==frame.contentWindow||event.data?.channel!==channel||done)return;
        const m=event.data;
        if(m.type==='asset'){
          if(typeof m.src!=='string'||!allowed.has(m.src)||!Number.isInteger(m.id)||++count>128){frame.contentWindow.postMessage({type:'asset',channel,id:m.id,error:'只能读取本页已提供的本站素材。'},'*');return;}
          try{const bitmap=await createImageBitmap(await load(m.src));if(done){bitmap.close();return;}frame.contentWindow.postMessage({type:'asset',channel,id:m.id,bitmap},'*',[bitmap]);}
          catch{if(!done)frame.contentWindow.postMessage({type:'asset',channel,id:m.id,error:'图片加载失败。'},'*');}
        }else if(m.type==='error')finish(new Error(String(m.message||'绘图失败。').slice(0,300)));
        else if(m.type==='result'){
          try{
            if(!(m.bitmap instanceof ImageBitmap)||!Number.isInteger(m.width)||!Number.isInteger(m.height)||m.width<1||m.height<1||m.width>16000||m.height>16000||m.width*m.height>24000000||m.bitmap.width!==m.width||m.bitmap.height!==m.height)throw new Error('脚本返回的画布无效。');
            const clean=scene(m.scene,m.width,m.height,config);canvas.width=m.width;canvas.height=m.height;canvas.getContext('2d').drawImage(m.bitmap,0,0);m.bitmap.close();
            finish(null,{width:m.width,height:m.height,scene:clean,failed:Number.isFinite(m.failed)?m.failed:0});
          }catch(error){m.bitmap?.close?.();finish(error);}
        }
      }
      window.addEventListener('message',receive);
      frame.onload=()=>{if(!done)frame.contentWindow.postMessage({type:'start',channel,source:config.script,config,data},'*');};
      frame.onerror=()=>finish(new Error('绘图沙箱未能加载。'));
      timeout=setTimeout(()=>finish(new Error('绘图JS运行超过8秒，已停止；可以修改后重试或撤销。')),8000);document.body.append(frame);
    });
    promise.cancel=()=>cancel();return promise;
  }
  window.GuideInfographicRuntime={render};
}());
