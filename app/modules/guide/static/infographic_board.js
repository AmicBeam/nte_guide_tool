(function () {
  'use strict';
  function create({getConfig,change,writeField,status,data}) {
    const stage=document.getElementById('ig-stage'),layer=document.getElementById('ig-board-layer'),editor=document.getElementById('ig-inline-editor'),canvas=document.getElementById('ig-canvas');
    let scene=null,selected='',gesture=null;
    const element=(tag,text)=>{const node=document.createElement(tag);if(text)node.textContent=text;return node;};
    const style=(node,box)=>Object.assign(node.style,{left:`${box.x/scene.width*100}%`,top:`${box.y/scene.height*100}%`,width:`${box.w/scene.width*100}%`,height:`${box.h/scene.height*100}%`});
    function finishEdit(){editor.hidden=true;editor.replaceChildren();}
    function edit(field) {
      if(gesture)return;
      finishEdit();const config=getConfig();let value=config;for(const key of field.path.split('.'))value=value[key];
      const input=element('textarea'),buttons=element('div'),save=element('button','保存'),cancel=element('button','取消');
      const source=document.querySelector(`[data-field="${field.path}"]`);
      input.value=value;input.maxLength=source?.maxLength>0?source.maxLength:1200;input.setAttribute('aria-label','编辑画板文字');input.rows=4;
      save.type=cancel.type='button';buttons.append(save,cancel);editor.append(input,buttons);editor.hidden=false;
      const width=stage.clientWidth,px=field.x/scene.width*width,py=field.y/scene.width*width;
      Object.assign(editor.style,{left:`${Math.max(0,Math.min(px,width-Math.min(320,width)))}px`,top:`${py}px`,width:`${Math.min(320,width)}px`});
      const apply=()=>{try{writeField(field.path,input.value);finishEdit();status('文字已更新，可撤销恢复。');}catch(error){status(error.message);}};
      save.onclick=apply;cancel.onclick=finishEdit;input.onkeydown=event=>{if(event.key==='Escape'){event.preventDefault();finishEdit();}if(event.key==='Enter'&&(event.ctrlKey||event.metaKey)){event.preventDefault();apply();}};
      input.focus();input.select();
    }
    function position(board,x,y) {
      return {x:Math.round(Math.max(40,Math.min(scene.width-40-board.w,x))),y:Math.round(Math.max(40,Math.min(15000-board.h,y)))};
    }
    function move(board,pos) {const next=structuredClone(getConfig());delete next.positions[board.key];next.positions[board.key]=pos;change(next);status('板块位置已更新，可撤销或恢复自动排版。');}
    function select(key){selected=key;for(const node of layer.querySelectorAll('.ig-board-handle')){node.classList.toggle('is-selected',node.dataset.key===key);node.setAttribute('aria-pressed',String(node.dataset.key===key));}}
    function handle(board) {
      const button=element('button','⠿');button.type='button';button.dataset.key=board.key;button.className='ig-board-handle';button.setAttribute('aria-label',`拖动${(data.modules[board.key]||board.label||board.key)}板块`);button.title='拖动移动 · 方向键微调';style(button,{...board,h:30});
      button.onclick=()=>select(board.key);
      button.onkeydown=event=>{
        const delta={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[event.key];if(!delta)return;
        event.preventDefault();const step=event.altKey?1:10;select(board.key);move(board,position(board,board.x+delta[0]*step,board.y+delta[1]*step));
      };
      button.onpointerdown=event=>{
        if(event.button!==0)return;event.preventDefault();finishEdit();select(board.key);button.focus();button.setPointerCapture(event.pointerId);
        gesture={board,pointer:event.pointerId,startX:event.clientX,startY:event.clientY,x:board.x,y:board.y,moved:false};
      };
      button.onpointermove=event=>{
        if(!gesture||gesture.pointer!==event.pointerId)return;
        const ratio=scene.width/canvas.getBoundingClientRect().width,dx=(event.clientX-gesture.startX)*ratio,dy=(event.clientY-gesture.startY)*ratio;
        if(!gesture.moved&&Math.hypot(dx,dy)<5)return;
        if(!gesture.moved){
          const ghost=element('canvas');ghost.className='ig-drag-ghost';ghost.width=Math.ceil(board.w);ghost.height=Math.ceil(board.h);
          ghost.getContext('2d').drawImage(canvas,board.x,board.y,board.w,board.h,0,0,board.w,board.h);layer.append(ghost);gesture.ghost=ghost;gesture.moved=true;
        }
        Object.assign(gesture,position(board,board.x+dx,board.y+dy));style(gesture.ghost,{...board,x:gesture.x,y:gesture.y});
      };
      const stop=(event,apply)=>{
        if(!gesture||gesture.pointer!==event.pointerId)return;
        const ended=gesture;gesture=null;ended.ghost?.remove();if(button.hasPointerCapture(event.pointerId))button.releasePointerCapture(event.pointerId);
        if(apply&&ended.moved)move(board,{x:ended.x,y:ended.y});
      };
      button.onpointerup=e=>stop(e,true);button.onpointercancel=e=>stop(e,false);button.onlostpointercapture=e=>stop(e,false);
      return button;
    }
    return {setScene(result){
      const focusKey=document.activeElement?.classList.contains('ig-board-handle')?document.activeElement.dataset.key:null;
      scene={...result.scene,width:result.width,height:result.height};
      layer.replaceChildren();
      for(const board of scene.boards)layer.append(handle(board));
      for(const field of scene.fields){
        const button=element('button');button.type='button';button.className='ig-text-target';button.dataset.path=field.path;button.title='双击编辑文字';button.setAttribute('aria-label',`编辑${document.querySelector(`[data-field="${field.path}"]`)?.parentElement.firstChild?.textContent||field.path}`);style(button,field);
        button.ondblclick=e=>{e.preventDefault();edit(field);};button.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();edit(field);}};layer.append(button);
      }
      for(const slot of scene.slots||[]){
        const button=element('button');button.type='button';button.className='ig-slot-toggle';button.setAttribute('aria-label',`${slot.folded?'展开':'折叠'}${slot.label}`);button.title=slot.folded?'展开本格候选':'折叠为小头像';style(button,slot);
        button.onclick=()=>{const next=structuredClone(getConfig());const [,i,,j]=slot.path.split('.');next.teams[i].folded[j]=!next.teams[i].folded[j];change(next);};layer.append(button);
      }
      select(selected);if(focusKey)layer.querySelector(`[data-key="${focusKey}"]`)?.focus({preventScroll:true});
    },close:finishEdit};
  }
  window.GuideInfographicBoard={create};
}());
