(function(){
  'use strict';
  const gate=document.getElementById('ig-auth-gate'),workspace=document.getElementById('ig-editor-workspace');if(!gate)return;
  const login='/login?next='+encodeURIComponent(location.pathname);document.getElementById('ig-login-link').href=login;
  let token='';try{token=localStorage.getItem('nte_token')||'';}catch{}
  if(!token){gate.querySelector('p').textContent='登录后可制作一图流，JS和模板保存在当前浏览器。';return;}
  async function boot(){
    try{
      const response=await fetch('/api/guide/characters/'+workspace.dataset.characterId+'/infographic',{headers:{Authorization:`Bearer ${token}`},cache:'no-store'});
      if(!response.ok)throw new Error(response.status===401?'登录已失效，请重新登录。':'编辑器资料暂未加载，请刷新重试。');
      const value=await response.json();if(localStorage.getItem('nte_token')!==token)return;
      const allowed=()=>localStorage.getItem('nte_token')===token&&!workspace.hidden;
      gate.hidden=true;workspace.hidden=false;window.startGuideInfographic(value.data,value.account,allowed);
      const revoke=()=>{workspace.hidden=true;gate.hidden=false;gate.querySelector('p').textContent='登录状态已变化，请重新登录或刷新页面。本机模板已保留。';window.GuideInfographicEditor=undefined;};
      window.addEventListener('storage',event=>{if(event.key==='nte_token'&&localStorage.getItem('nte_token')!==token)revoke();});
      window.addEventListener('focus',async()=>{
        if(localStorage.getItem('nte_token')!==token){revoke();return;}
        try{const response=await fetch('/api/me',{headers:{Authorization:`Bearer ${token}`},cache:'no-store'});if(response.status===401)revoke();}catch{}
      });
    }catch(error){gate.querySelector('p').textContent=error.message;}
  }
  boot();
}());
