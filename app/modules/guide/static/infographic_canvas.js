(function () {
  'use strict';
  const palettes = {night:{bg:'#10181f',panel:'#19252e',line:'#32434d',ink:'#f0f3ed',dim:'#a6b6bc',accent:'#c8f082'},
    paper:{bg:'#f2eee5',panel:'#fffdf7',line:'#d8d0c0',ink:'#292d30',dim:'#736f66',accent:'#af4535'},
    teal:{bg:'#0b242b',panel:'#11343c',line:'#2c535a',ink:'#e6f6ee',dim:'#99bcbf',accent:'#80e5ce'}};
  const cache = new Map();
  const FONT = '"PingFang SC", "Microsoft YaHei", sans-serif';
  function image(src) {
    if(!src) return Promise.resolve(null);
    if(!cache.has(src)) cache.set(src,new Promise(resolve=>{
      const img = new Image(); img.onload=()=>resolve(img); img.onerror=()=>{cache.delete(src);resolve(null);};img.src=src;
    }));
    return cache.get(src);
  }
  async function render(canvas, config, data) {
    const character = data.characters.find(c=>c.id===data.character_id), c=config, p=palettes[c.theme];
    const urls = [character.assets[c.portrait],character.element_icon,character.rarity_icon, ...c.plans.flatMap(plan=>[data.arcs.find(a=>a.id===plan.arc)?.image,data.cores.find(a=>a.id===plan.core)?.image]),
      ...c.teams.flatMap(t=>t.members.map(id=>data.characters.find(a=>a.id===id)?.assets.avatar))];
    const images=new Map(await Promise.all(urls.filter(Boolean).map(async src=>[src,await image(src)])));
    const W=c.orientation==='landscape'?1600:1000, margin=40, gap=22;
    const sidebar=c.layout==='editorial' && c.orientation==='landscape';
    const x0=sidebar?450:margin, areaW=W-x0-margin;
    const columns=c.orientation==='portrait' && c.layout==='balanced'?1:c.layout==='compact' && c.orientation==='landscape'?3:2;
    const cardW=(areaW-gap*(columns-1))/columns;
    const ctx=canvas.getContext('2d'), fields=[];
    function font(size=22,bold=false) {ctx.font=`${bold?'600':'400'} ${size}px ${FONT}`;}
    function lines(text,width,size=22,bold=false) {
      font(size,bold);const result=[];
      for(const paragraph of String(text||'').split('\n')) {
        if(!paragraph) {result.push('');continue;}
        let row='';
        for(const letter of paragraph) {if(ctx.measureText(row+letter).width>width && row){result.push(row);row=letter;}else row+=letter;}
        result.push(row);
      }
      return result;
    }
    function text(value,x,y,width,size=22,color=p.ink,bold=false,draw=true,path=null) {
      const rows=lines(value,width,size,bold);font(size,bold);
      if(draw) {if(path)fields.push({path,x,y,w:width,h:rows.length*size*1.55,size});ctx.fillStyle=color;ctx.textBaseline='top';rows.forEach((row,i)=>ctx.fillText(row,x,y+i*size*1.55));}
      return rows.length*size*1.55;
    }
    function rect(x,y,w,h,fill,r=16) {ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fillStyle=fill;ctx.fill();}
    function photo(src,x,y,w,h,scale=1) {
      const img=images.get(src);if(!img) return;
      ctx.save();ctx.beginPath();ctx.rect(x,y,w,h);ctx.clip();
      const fit=Math.min(w/img.width,h/img.height)*scale;
      ctx.drawImage(img,x+(w-img.width*fit)/2,y+(h-img.height*fit)/2,img.width*fit,img.height*fit);ctx.restore();
    }
    // Measuring and drawing share exactly the same wrapping and block geometry.
    function card(key,x,y,width,draw) {
      const left=x+26, inner=width-52;let cy=y+24;
      cy+=text(data.modules[key],left,cy,inner,25,p.accent,true,draw)+16;
      function body(value,size=21,color=p.ink,bold=false,path=null) {cy+=text(value,left,cy,inner,size,color,bold,draw,path)+12;}
      if(key==='equipment') {
        const selected=c.plans.filter(plan=>plan.arc||plan.core||plan.main||plan.sub||plan.note);
        if(!selected.length) body('选择弧盘与卡带，写下你的配装思路。',21,p.dim);
        selected.forEach((plan,index)=>{
          if(index) cy+=16;
          const planIndex=c.plans.indexOf(plan);
          body(plan.name||`方案 ${index+1}`,23,p.ink,true,`plans.${planIndex}.name`);
          for(const [label,item] of [['弧盘',data.arcs.find(a=>a.id===plan.arc)],['卡带',data.cores.find(a=>a.id===plan.core)]]) {
            if(!item) continue;
            if(draw) {rect(left,cy,70,70,p.bg,10);photo(item.image,left,cy,70,70);}
            const h=text(`${label} / ${item.name}`,left+84,cy+4,inner-84,21,p.ink,false,draw);
            cy+=Math.max(70,h)+12;
          }
          if(plan.main) body(`主词条  ${plan.main}`,21,p.ink,false,`plans.${planIndex}.main`);
          if(plan.sub) body(`副词条  ${plan.sub}`,21,p.ink,false,`plans.${planIndex}.sub`);
          if(plan.note) body(plan.note,20,p.dim,false,`plans.${planIndex}.note`);
        });
      } else if(key==='teams') {
        const selected=c.teams.filter(team=>team.members.some(Boolean)||team.note);
        selected.forEach((team,index)=>{
          const teamIndex=c.teams.indexOf(team);
          if(index) cy+=20;body(team.name||`队伍 ${index+1}`,23,p.ink,true,`teams.${teamIndex}.name`);
          const members=team.members.map(id=>data.characters.find(a=>a.id===id)).filter(Boolean);
          const cell=(inner-12*3)/4;let namesH=0;
          members.forEach((member,i)=>{
            if(draw) {rect(left+i*(cell+12),cy,cell,cell,p.bg,10);photo(member.assets.avatar,left+i*(cell+12),cy,cell,cell);}
            namesH=Math.max(namesH,text(member.name,left+i*(cell+12),cy+cell+8,cell,18,p.ink,false,draw));
          });
          if(members.length)cy+=cell+8+namesH+14;
          if(team.note)body(team.note,21,p.dim,false,`teams.${teamIndex}.note`);
        });
        if(!selected.length)body('选择队员，补充战术职能与替代思路。',21,p.dim);
      } else if(key==='radar') {
        const centerX=x+width/2,centerY=cy+170,r=Math.min(inner/2-85,110);
        if(draw) {
          ctx.save();ctx.strokeStyle=p.line;
          const polygon=(values,fill)=>{ctx.beginPath();values.forEach((v,i)=>{const a=-Math.PI/2+i*Math.PI/3;const px=centerX+Math.cos(a)*r*v,py=centerY+Math.sin(a)*r*v;i?ctx.lineTo(px,py):ctx.moveTo(px,py);});ctx.closePath();if(fill){ctx.fillStyle=p.accent+'35';ctx.fill();}ctx.stroke();};
          for(const level of [.25,.5,.75,1])polygon(Array(6).fill(level));
          ctx.strokeStyle=p.accent;ctx.lineWidth=2;polygon(c.radar.map(v=>v/100),true);
          ctx.restore();
          c.radar_labels.forEach((label,i)=>{
            const a=-Math.PI/2+i*Math.PI/3,labelW=Math.min(110,inner/3);
            const lx=centerX+Math.cos(a)*(r+38)-labelW/2,ly=centerY+Math.sin(a)*(r+38)-24;
            text(label,lx,ly,labelW,17,p.dim,false,true,`radar_labels.${i}`);
            text(String(c.radar[i]),lx,ly+lines(label,labelW,17).length*26,labelW,18,p.accent,true);
          });
        }
        cy+=370;body('作者主观评分 · 0–100',18,p.dim);

      } else {
        const placeholders={stats:'填写目标面板，并注明等级、配装与战斗条件。',skills:'填写技能升级顺序、关键觉醒与共鸣建议。',rotation:'填写起手、切人顺序、输出窗口与操作注意事项。',advice:'填写适用场景、养成优先级与抽取建议。'};
        body(c[key]||placeholders[key],22,c[key]?p.ink:p.dim,false,key);
        if(key==='stats' && c.stats)body('以上为作者填写的参考目标',18,p.dim);
      }
      return cy-y+14;
    }
    const titleWidth=sidebar?318:W-420;
    const titleHeight=text(c.title,0,0,titleWidth,sidebar?45:52,p.ink,true,false);
    const subHeight=text(c.subtitle,0,0,titleWidth,23,p.dim,false,false);
    const headerHeight=sidebar?0:Math.max(c.layout==='compact'?175:245,110+titleHeight+subHeight);
    const top=sidebar?margin:headerHeight+margin;
    const bottoms=Array(columns).fill(top), placements=[];
    for(const key of c.order.filter(key=>c.modules[key])) {
      const column=bottoms.indexOf(Math.min(...bottoms)),x=x0+column*(cardW+gap),y=bottoms[column];
      const h=card(key,x,y,cardW,false),pos=c.positions[key];
      placements.push({key,x:pos?Math.min(W-margin-cardW,Math.max(margin,pos.x)):x,y:pos?Math.max(margin,Math.min(15000-h,pos.y)):y,h});bottoms[column]+=h+gap;
    }
    const positionOrder=Object.keys(c.positions);
    placements.sort((a,b)=>positionOrder.indexOf(a.key)-positionOrder.indexOf(b.key));
    const footerWidth=W-80;
        const footerColumn=(footerWidth-40)/3;
    const footerH=Math.max(...[c.author?`制作 / ${c.author}`:'',c.version?`版本 / ${c.version}`:'','攻略数据库 · NTE TOOLS'].map(v=>text(v,0,0,footerColumn,20,p.dim,false,false)))+text(c.disclaimer,0,0,footerWidth,18,p.dim,false,false)+38;
    const heroMin=titleHeight+subHeight+660;
    const H=Math.ceil(Math.max(c.orientation==='landscape'?900:1400,...placements.map(item=>item.y+item.h+gap),sidebar?heroMin:0)+footerH+margin);
    if(H>16000) throw new Error('内容过长，请精简文字或改用横排布局。');
    canvas.width=W;canvas.height=H;ctx.textAlign='left';ctx.fillStyle=p.bg;ctx.fillRect(0,0,W,H);
    if(sidebar) {
      rect(margin,margin,370,H-footerH-margin*2,p.panel);
      let y=70;text('CHARACTER / GUIDE',66,y,318,16,p.accent,true);y+=42;
      y+=text(c.title,66,y,318,45,p.ink,true,true,'title')+14;
      y+=text(c.subtitle,66,y,318,23,p.dim,false,true,'subtitle')+24;
      photo(character.assets[c.portrait],46,y,358,430,c.art_scale);y+=450;
      photo(character.element_icon,66,y,40,40);photo(character.rarity_icon,124,y,58,40);y+=58;
      text(character.tags.join(' · '),66,y,318,21,p.dim);
    } else {
      rect(margin,margin,W-80,headerHeight-16,p.panel);
      text('CHARACTER / GUIDE',66,65,titleWidth,16,p.accent,true);
      let y=105;y+=text(c.title,66,y,titleWidth,52,p.ink,true,true,'title')+8;
      text(c.subtitle,66,y,titleWidth,23,p.dim,false,true,'subtitle');
      photo(character.element_icon,W-400,56,28,28);photo(character.rarity_icon,W-358,56,40,28);
      photo(character.assets[c.portrait],W-315,45,240,headerHeight-26,c.art_scale);
    }
    placements.forEach(item=>{rect(item.x,item.y,cardW,item.h,p.panel);card(item.key,item.x,item.y,cardW,true);});
    let fy=H-footerH;ctx.fillStyle=p.accent;ctx.fillRect(margin,fy-14,44,3);
    const parts=[['author',c.author?`制作 / ${c.author}`:''],['version',c.version?`版本 / ${c.version}`:''],'攻略数据库 · NTE TOOLS'];
    const partW=(footerWidth-40)/3;
    const heights=parts.map((part,i)=>Array.isArray(part)?text(part[1],margin+i*(partW+20),fy,partW,20,p.dim,false,true,part[0]):text(part,margin+i*(partW+20),fy,partW,20,p.dim));
    fy+=Math.max(...heights)+10;
    text(c.disclaimer,margin,fy,footerWidth,18,p.dim,false,true,'disclaimer');
    return {width:W,height:H,scene:{fields,boards:placements.map(item=>({...item,w:cardW}))},failed:[...images].filter(([,img])=>!img).length};
  }
  window.GuideInfographicCanvas={render};
}());
