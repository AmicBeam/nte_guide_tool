(function () {
  'use strict';
  const section = document.getElementById('section-related-axes');
  if (!section) return;
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const format = value => Number(value || 0).toLocaleString('zh-CN', {maximumFractionDigits:0});
  const token = () => { try { return localStorage.getItem('nte_token') || ''; } catch { return ''; } };
  let catalog, character, currentPage = 1, requestId = 0, previewRequest = 0;
  let items = [], signedIn = false;
  const pendingAxes = new Set();
  const reactionFeedback = new Map();
  const loginUrl = () => `/login?next=${encodeURIComponent(location.pathname + location.search + location.hash)}`;
  async function request(url, {method = 'GET', bearer = token()} = {}) {
    const response = await fetch(url, {method, headers: bearer ? {Authorization:`Bearer ${bearer}`} : {}, cache:'no-store'});
    let payload;
    try { payload = await response.json(); } catch (_) { throw new Error('网站服务暂时不可用，请稍后重试。'); }
    if (!response.ok || payload.error) throw new Error(response.status === 401 ? '登录已失效，请重新登录后读取个人排轴。' : String(payload.error?.message || payload.error || '读取失败，请重试。'));
    return payload;
  }
  function compareUrl(axisId) {
    return `/guide/characters/${section.dataset.guideCharacterId}/axes/${axisId}/compare`;
  }
  function teamHtml(team) {
    return `<div class="guide-axis-team">${(team || []).map(member => `<span>${member.character_avatar ? `<img src="${esc(member.character_avatar)}" alt="" loading="lazy">` : ''}${esc(member.character_name)}<small>${esc(member.awakening_nodes?.length ?? member.awakening ?? 0)}觉</small></span>`).join('')}</div>`;
  }
  function render() {
    $('related-list').innerHTML = items.map(item => `<article class="guide-axis-card" data-axis-id="${Number(item.id)}"><h3>${esc(item.title)}</h3><span class="guide-axis-author">${esc(item.owner?.nickname || '作者')} · 保存版本 ${esc(item.source_version || '未提供')}</span>${teamHtml(item.team)}<p class="guide-axis-note" title="${esc(item.description)}">${esc(item.description || '暂无备注')}</p><div class="guide-axis-metrics"><span><strong>${format(item.dps)}</strong> DPS</span><span><strong>${Number(item.duration_seconds || 0).toFixed(1)}</strong> 秒</span><span>${item.loop_enabled ? '循环轴' : '非循环轴'}</span></div><div class="guide-axis-actions"><button type="button" data-axis-preview="${Number(item.id)}">预览操作轴</button><a href="${compareUrl(item.id)}">用此轴比较配装 ↗</a></div>${reactionHtml(item)}</article>`).join('');
  }
  function reactionHtml(item) {
    if (item.visibility !== 'public') return '';
    if (!signedIn) return `<div class="guide-axis-reactions"><span>赞 ${format(item.like_count)} · 踩 ${format(item.dislike_count)} · 收藏 ${format(item.favorite_count)}</span><a href="${loginUrl()}">登录后评价与收藏 ↗</a></div>`;
    const controls = [['like', 'liked', '赞', 'like_count'], ['dislike', 'disliked', '踩', 'dislike_count'], ['favorite', 'favorited', '收藏', 'favorite_count']];
    return `<div class="guide-axis-reactions">${controls.map(([kind, flag, label, count]) => {
      const ownFavorite = kind === 'favorite' && item.is_owner && !item.favorited;
      return `<button type="button" data-axis-reaction="${kind}" data-reaction-axis="${Number(item.id)}" aria-pressed="${Boolean(item[flag])}" ${pendingAxes.has(Number(item.id)) || ownFavorite ? 'disabled' : ''}${ownFavorite ? ' title="不能收藏自己的排轴"' : ''}>${item[flag] ? '取消' : ''}${label} ${format(item[count])}</button>`;
    }).join('')}</div><p class="guide-axis-feedback" role="status">${esc(reactionFeedback.get(Number(item.id)) || '')}</p>`;
  }
  async function react(axisId, kind) {
    const flag = {like:'liked', dislike:'disliked', favorite:'favorited'}[kind];
    const item = items.find(axis => Number(axis.id) === axisId);
    if (!flag || !item || item.visibility !== 'public' || pendingAxes.has(axisId) || !signedIn) return;
    if (kind === 'favorite' && item.is_owner && !item.favorited) return;
    const bearer = token();
    pendingAxes.add(axisId); reactionFeedback.delete(axisId); render();
    try {
      if (!bearer) throw new Error('请先登录后再评价或收藏。');
      // Favorite also supports anonymous visitors elsewhere; this page requires a valid account.
      await request('/api/me', {bearer});
      if (token() !== bearer) throw new Error('登录账号已变化，请刷新页面后重试。');
      const updated = await request(`/api/shaft/axes/${axisId}/${kind}`, {method:item[flag] ? 'DELETE' : 'POST', bearer});
      items = items.map(axis => Number(axis.id) === axisId ? {...axis, ...updated} : axis);
      if ($('related-scope').value === 'favorites' && !updated.favorited) {
        items = items.filter(axis => Number(axis.id) !== axisId);
        $('related-status').textContent = items.length ? '已取消收藏。' : '已取消收藏，此筛选下暂无收藏排轴。';
      } else {
        const label = {like:'点赞', dislike:'踩', favorite:'收藏'}[kind];
        reactionFeedback.set(axisId, item[flag] ? `已取消${label}。` : `已${label}。`);
      }
    } catch (error) {
      reactionFeedback.set(axisId, error.message);
      if (!bearer || /登录/.test(error.message)) {
        signedIn = false;
        $('related-status').innerHTML = `登录已失效。<a href="${loginUrl()}">重新登录 ↗</a>`;
      }
    } finally {
      pendingAxes.delete(axisId); render();
      const button = document.querySelector(`[data-axis-id="${axisId}"] [data-axis-reaction="${kind}"]`);
      if (button && !button.disabled) button.focus({preventScroll:true});
    }
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
      $('related-status').textContent = items.length ? `已显示 ${items.length} 条相关排轴；数值为保存时结果，比较配装时会重新计算。` : '暂未找到符合条件的排轴，请调整队友或范围。找到轴后可使用该轴比较配装。';
      $('related-more').hidden = scope !== 'public' || !payload.has_more;
    } catch(error) { if (id === requestId) { $('related-list').replaceChildren(); $('related-status').textContent = error.message; } }
    finally { if (id === requestId) { $('related-refresh').disabled = false; $('related-list').setAttribute('aria-busy','false'); } }
  }
  async function preview(axisId) {
    const id = ++previewRequest;
    const dialog = $('related-preview');
    $('related-preview-title').textContent = '正在读取排轴…'; $('related-preview-frame').src = 'about:blank'; $('related-preview-compare').hidden = true; $('related-preview-simulator').hidden = true;
    if (!dialog.open) dialog.showModal();
    try {
      const payload = await request(`/api/shaft/axes/${axisId}`);
      if (id !== previewRequest || !dialog.open) return;
      $('related-preview-title').textContent = payload.title;
      $('related-preview-frame').src = `/shaft/rotation?preview=${axisId}&embedded=1`;
      $('related-preview-simulator').href = `/shaft/rotation?preview=${axisId}`; $('related-preview-simulator').hidden = false;
      $('related-preview-compare').href = compareUrl(axisId); $('related-preview-compare').hidden = false;
    } catch(error) { if (id === previewRequest) $('related-preview-title').textContent = error.message; }
  }
  async function init() {
    $('related-refresh').onclick = () => load(); $('related-more').onclick = () => load(true);
    for (const id of ['related-scope','related-teammate','related-sort']) $(id).onchange = () => load();
    $('related-list').onclick = event => {
      const action = event.target.closest('[data-axis-reaction]');
      if (action) { react(Number(action.dataset.reactionAxis), action.dataset.axisReaction); return; }
      const button = event.target.closest('[data-axis-preview]');
      if (button) preview(Number(button.dataset.axisPreview));
    };
    $('related-preview').addEventListener('close', () => { ++previewRequest; $('related-preview-frame').src = 'about:blank'; });
    window.addEventListener('message', event => {
      if (event.origin === location.origin && event.source === $('related-preview-frame').contentWindow && event.data?.type === 'shaft-preview-close') $('related-preview').close();
    });
    try {
      if (token()) {
        try { await request('/api/me'); signedIn = true; } catch (_) { signedIn = false; }
      }
      catalog = await request('/api/shaft/catalog');
      // Both protagonist entries use the same existing shaft character.
      const name = ['1046','1051'].includes(section.dataset.guideCharacterId) ? '主角' : section.dataset.guideCharacterName;
      character = catalog.characters.find(item => item.name === name);
      if (!character) { $('related-status').textContent = '排轴模块暂未接入该角色。'; $('related-list').setAttribute('aria-busy','false'); return; }
      for (const teammate of catalog.characters.filter(item => item.id !== character.id && !item.selection_disabled)) $('related-teammate').add(new Option(teammate.name,teammate.id));
      await load();
    } catch(error) { $('related-status').textContent = error.message; $('related-list').setAttribute('aria-busy','false'); }
  }
  init();
}());
