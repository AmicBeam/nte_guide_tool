(function () {
  'use strict';
  const section = document.getElementById('section-related-axes');
  if (!section) return;
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const format = value => Number(value || 0).toLocaleString('zh-CN', {maximumFractionDigits:0});
  const token = () => { try { return localStorage.getItem('nte_token') || ''; } catch { return ''; } };
  let catalog, character, currentPage = 1, requestId = 0, previewRequest = 0;
  let items = [];
  async function request(url) {
    const response = await fetch(url, {headers: token() ? {Authorization:`Bearer ${token()}`} : {}, cache:'no-store'});
    const payload = await response.json();
    if (!response.ok || payload.error) throw new Error(response.status === 401 ? '登录已失效，请重新登录后读取个人排轴。' : String(payload.error || '读取失败，请重试。'));
    return payload;
  }
  function compareUrl(axisId) {
    const query = new URLSearchParams({character:character.id, guide_character:section.dataset.guideCharacterId, locked:'1'});
    if (axisId) query.set('axis', axisId);
    return `/shaft/compare?${query}`;
  }
  function teamHtml(team) {
    return `<div class="guide-axis-team">${(team || []).map(member => `<span>${member.character_avatar ? `<img src="${esc(member.character_avatar)}" alt="" loading="lazy">` : ''}${esc(member.character_name)}<small>${esc(member.awakening_nodes?.length ?? member.awakening ?? 0)}觉</small></span>`).join('')}</div>`;
  }
  function render() {
    $('related-list').innerHTML = items.map(item => `<article class="guide-axis-card"><h3>${esc(item.title)}</h3><span class="guide-axis-author">${esc(item.owner?.nickname || '作者')} · 保存版本 ${esc(item.source_version || '未提供')}</span>${teamHtml(item.team)}<p class="guide-axis-note" title="${esc(item.description)}">${esc(item.description || '暂无备注')}</p><div class="guide-axis-metrics"><span><strong>${format(item.dps)}</strong> DPS</span><span><strong>${Number(item.duration_seconds || 0).toFixed(1)}</strong> 秒</span><span>${item.loop_enabled ? '循环轴' : '非循环轴'}</span></div><div class="guide-axis-actions"><button type="button" data-axis-preview="${Number(item.id)}">预览操作轴</button><a href="${compareUrl(item.id)}">用此轴比较配装 ↗</a></div></article>`).join('');
  }
  async function load(more = false) {
    if (!character) return;
    const id = ++requestId;
    const scope = $('related-scope').value;
    $('related-more').hidden = true;
    if (scope !== 'public' && !token()) {
      $('related-refresh').disabled = false;
      $('related-list').replaceChildren();
      $('related-list').setAttribute('aria-busy','false');
      $('related-status').innerHTML = `登录后可查看自己的排轴和收藏。<a href="/login?next=${encodeURIComponent(location.pathname + '#section-related-axes')}">前往登录 ↗</a>`;
      return;
    }
    $('related-list').setAttribute('aria-busy','true');
    $('related-status').textContent = '正在读取相关排轴…';
    $('related-refresh').disabled = true;
    const query = new URLSearchParams({character_id:character.id, sort:$('related-sort').value, page_size:'6', page:String(more ? currentPage + 1 : 1)});
    if ($('related-teammate').value) query.append('character_id',$('related-teammate').value);
    const route = scope === 'public' ? '/api/shaft/market' : `/api/shaft/me/${scope === 'mine' ? 'axes' : 'favorites'}`;
    try {
      const payload = await request(`${route}?${query}`);
      if (id !== requestId) return;
      const incoming = payload.items || [];
      items = more ? [...new Map([...items,...incoming].map(item => [item.id,item])).values()] : incoming;
      currentPage = more ? currentPage + 1 : 1;
      render();
      $('related-status').textContent = items.length ? `已显示 ${items.length} 条相关排轴；数值为保存时结果，比较配装时会重新计算。` : '暂未找到符合条件的排轴。可调整队友或范围，也可以前往收益对比使用浏览器当前轴。';
      $('related-more').hidden = scope !== 'public' || !payload.has_more;
    } catch(error) { if (id === requestId) { $('related-list').replaceChildren(); $('related-status').textContent = error.message; } }
    finally { if (id === requestId) { $('related-refresh').disabled = false; $('related-list').setAttribute('aria-busy','false'); } }
  }
  async function preview(axisId) {
    const id = ++previewRequest;
    const dialog = $('related-preview');
    $('related-preview-title').textContent = '正在读取排轴…'; $('related-preview-body').replaceChildren(); $('related-preview-compare').hidden = true;
    if (!dialog.open) dialog.showModal();
    try {
      const payload = await request(`/api/shaft/axes/${axisId}`);
      if (id !== previewRequest || !dialog.open) return;
      $('related-preview-title').textContent = payload.title;
      const axis = payload.axis;
      const actions = new Map(catalog.actions.map(action => [action.id,action]));
      $('related-preview-body').innerHTML = `${teamHtml(payload.team)}<p class="guide-muted">${esc(payload.description || '暂无备注')}</p><p class="guide-muted">原轴 ${Number(payload.duration_seconds || 0).toFixed(1)} 秒 · ${format(payload.dps)} DPS。下列为轴内编辑位置，零秒时停的展示位置不计入真实轴长。</p><ol class="guide-axis-sequence">${[...(axis.steps || [])].sort((a,b) => Number(a.start_tick) - Number(b.start_tick)).map(step => `<li>${(Number(step.start_tick || 0)/10).toFixed(1)}s · ${esc(payload.team.find(member => Number(member.slot) === Number(step.slot))?.character_name)} · ${esc(actions.get(step.action_id)?.name || step.action_name || step.action_id)}${step.placement === 'background' ? '（后台）' : ''}</li>`).join('')}</ol>`;
      $('related-preview-compare').href = compareUrl(axisId); $('related-preview-compare').hidden = false;
    } catch(error) { if (id === previewRequest) $('related-preview-title').textContent = error.message; }
  }
  async function init() {
    $('related-refresh').onclick = () => load(); $('related-more').onclick = () => load(true);
    for (const id of ['related-scope','related-teammate','related-sort']) $(id).onchange = () => load();
    $('related-list').onclick = event => { const button = event.target.closest('[data-axis-preview]'); if (button) preview(Number(button.dataset.axisPreview)); };
    $('related-preview').addEventListener('close', () => { ++previewRequest; });
    try {
      catalog = await request('/api/shaft/catalog');
      // Both protagonist entries use the same existing shaft character.
      const name = ['1046','1051'].includes(section.dataset.guideCharacterId) ? '主角' : section.dataset.guideCharacterName;
      character = catalog.characters.find(item => item.name === name);
      if (!character) { $('related-status').textContent = '排轴模块暂未接入该角色。'; $('related-list').setAttribute('aria-busy','false'); return; }
      $('related-compare').href = compareUrl();
      $('related-compare').removeAttribute('aria-disabled');
      for (const teammate of catalog.characters.filter(item => item.id !== character.id && !item.selection_disabled)) $('related-teammate').add(new Option(teammate.name,teammate.id));
      await load();
    } catch(error) { $('related-status').textContent = error.message; $('related-list').setAttribute('aria-busy','false'); }
  }
  init();
}());
