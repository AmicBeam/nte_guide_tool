'use strict';
(async()=>{
  const edit=document.querySelector('[data-article-edit]');
  if(!edit)return;
  const token=await window.guideAuth?.ready;
  if(!token)return;
  try{
    const response=await fetch('/api/guide/me/articles/'+edit.dataset.articleEdit,{headers:{Authorization:'Bearer '+token},cache:'no-store'});
    if(response.ok)edit.hidden=false;
  }catch(_){/* Keep public reading available when the owner check cannot finish. */}
})();
