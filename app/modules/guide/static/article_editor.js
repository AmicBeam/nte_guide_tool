'use strict';
(() => {
  const $ = id => document.getElementById('article-' + id);
  const title = $('title'), body = $('markdown'), target = $('target');
  if (!body) return;
  const state = $('save-state');
  let token = '', account = '', id = '', revision = 0, saved = '', task = null;
  let saveTimer, previewTimer, previewAbort, previewSequence = 0, blocked = false, recovering = false;
  let restoring = null, backupOk = true, lastActiveId = '', navigating = false;
  const status = text => { state.textContent = text; };
  const fields = () => {
    const separator = target.value.indexOf('/');
    return {title: title.value, markdown: body.value,
      target_kind: separator < 0 ? '' : target.value.slice(0, separator),
      target_id: separator < 0 ? '' : target.value.slice(separator + 1)};
  };
  const stamp = value => JSON.stringify(value);
  const backupKey = () => `nte-guide-article:${account}:${id}`;
  const readStorage = key => { try { return localStorage.getItem(key); } catch (_) { return null; } };
  const uuid = () => window.crypto.randomUUID ? window.crypto.randomUUID() : '10000000-1000-4000-8000-100000000000'.replace(/[018]/g, digit => (Number(digit) ^ window.crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> Number(digit) / 4).toString(16));
  const dirty = () => stamp(fields()) !== saved;
  function backup() {
    try {
      localStorage.setItem(backupKey(), JSON.stringify({...fields(), revision}));
      localStorage.setItem('nte-guide-article-last:' + account, id);
      backupOk = true;
    } catch (_) { backupOk = false; }
  }
  function sessionValid() {
    if (readStorage('nte_token') === token) return true;
    blocked = true;
    $('fields').disabled = true;
    $('publish').disabled = true;
    status('登录账号已变化，请重新打开编辑器。本机草稿已保留。');
    return false;
  }
  async function request(url, options = {}) {
    if (!sessionValid()) throw new Error('登录账号已变化。');
    const response = await fetch(url, {...options, cache: 'no-store', headers: {
      Authorization: 'Bearer ' + token, 'Content-Type': 'application/json', ...options.headers}});
    let value;
    try { value = await response.json(); } catch (_) { throw new Error('服务暂时无法读取，请稍后重试。'); }
    if (!response.ok) {
      const error = new Error(value.error?.message || (typeof value.error === 'string' ? value.error : '读取或保存失败。'));
      error.status = response.status;
      if (response.status === 401) { blocked = true; status('登录已失效。本机备份已保留，请重新登录。'); }
      if (response.status === 409) { blocked = true; $('reload').hidden = false; }
      throw error;
    }
    return value;
  }
  function apply(value) {
    title.value = value.title || '';
    body.value = value.markdown || '';
    const targetValue = value.target_kind ? value.target_kind + '/' + value.target_id : '';
    if (targetValue && !Array.from(target.options).some(option => option.value === targetValue)) {
      const option = document.createElement('option'); option.value = targetValue;
      option.textContent = '原挂载页面（请核对是否仍开放）'; target.append(option);
    }
    target.value = targetValue;
    $('word-count').textContent = body.value.length + ' 字';
  }
  function updateOption() {
    let option = Array.from($('drafts').options).find(item => item.value === id);
    if (!option) { option = document.createElement('option'); option.value = id; $('drafts').append(option); }
    option.textContent = title.value.trim() || '未命名草稿';
    $('drafts').value = id;
  }
  async function draftList() {
    const result = await request('/api/guide/me/articles');
    $('drafts').replaceChildren();
    for (const row of result.items) {
      const option = document.createElement('option'); option.value = row.id;
      option.textContent = (row.title || '未命名草稿') + (row.published ? ' · 已发布' : '');
      $('drafts').append(option);
    }
    updateOption();
  }
  async function preview() {
    const sequence = ++previewSequence;
    previewAbort?.abort(); previewAbort = new AbortController();
    try {
      const result = await request('/api/guide/articles/preview', {method: 'POST', signal: previewAbort.signal,
        body: JSON.stringify({markdown: body.value})});
      if (sequence !== previewSequence) return;
      // Only the server's escaped Markdown renderer may supply preview HTML.
      $('preview').innerHTML = result.html;
      $('preview-state').textContent = '';
    } catch (error) {
      if (error.name !== 'AbortError' && sequence === previewSequence) $('preview-state').textContent = '预览暂不可用，正文仍会保存。';
    }
  }
  function queuePreview() { clearTimeout(previewTimer); previewTimer = setTimeout(preview, 400); }
  async function flush() {
    clearTimeout(saveTimer);
    if (task) { await task; return dirty() && !blocked && !recovering ? flush() : undefined; }
    if (blocked || recovering || !dirty() || !sessionValid()) return;
    task = (async () => {
      while (dirty() && !blocked && !recovering) {
        const snapshot = fields();
        status('正在保存草稿…');
        const result = await request('/api/guide/me/articles/' + id, {method: 'PUT',
          body: JSON.stringify({...snapshot, revision})});
        revision = result.data.revision;
        saved = stamp(snapshot);
        backup(); updateOption();
      }
      status(backupOk ? '草稿已自动保存' : '草稿已保存到云端；本机备份空间不足。');
    })();
    try { await task; }
    catch (error) {
      status(error.status === 409 ? error.message : error.message + (backupOk ? ' 本机备份已保留。' : ' 本机备份也未成功，请复制正文留存。'));
      throw error;
    } finally { task = null; }
  }
  const quietSave = () => flush().catch(() => {});
  function changed() {
    if (!sessionValid()) return;
    $('word-count').textContent = body.value.length + ' 字';
    if (recovering) return;
    backup(); updateOption(); queuePreview();
    if (blocked) return;
    status(backupOk ? '已备份到本机，等待同步…' : '本机备份空间不足，正在同步…');
    clearTimeout(saveTimer); saveTimer = setTimeout(quietSave, 800);
  }
  async function openDraft(nextId) {
    previewSequence++; previewAbort?.abort();
    clearTimeout(saveTimer); clearTimeout(previewTimer);
    blocked = false; recovering = false; restoring = null;
    $('recovery').hidden = true; $('reload').hidden = true;
    id = nextId; revision = 0;
    let value = {title:'', markdown:'', target_kind:'', target_id:''};
    try {
      const result = await request('/api/guide/me/articles/' + id);
      value = result.data; revision = value.revision;
    } catch (error) { if (error.status !== 404) throw error; }
    apply(value); saved = stamp(fields());
    let local = null;
    try { local = JSON.parse(readStorage(backupKey()) || 'null'); } catch (_) { /* Ignore invalid local backups. */ }
    if (local && typeof local.title === 'string' && typeof local.markdown === 'string'
        && typeof local.target_kind === 'string' && typeof local.target_id === 'string') {
      const localFields = {title:local.title, markdown:local.markdown, target_kind:local.target_kind, target_id:local.target_id};
      if (stamp(localFields) !== saved) {
        if (local.revision === revision) { apply(local); }
        else { restoring = local; recovering = true; $('recovery').hidden = false; }
      }
    }
    lastActiveId = id;
    const url = new URL(location.href); url.searchParams.set('id', id);
    url.searchParams.delete('target_kind'); url.searchParams.delete('target_id'); history.replaceState(null, '', url);
    const link = $('published-link'); link.hidden = !value.published; link.href = '/guide/articles/' + id;
    updateOption(); queuePreview();
    if (recovering) status('请选择本机或云端草稿。');
    else if (dirty()) { status('已恢复本机未同步内容。'); changed(); }
    else { backup(); status(revision ? '已读取保存的草稿' : '开始撰写，输入后自动保存'); }
  }
  async function switching(nextId) {
    if (navigating) return;
    navigating = true; $('fields').disabled = true; $('publish').disabled = true;
    try { await flush(); if (dirty() || blocked || recovering) { $('drafts').value = lastActiveId; return; } await openDraft(nextId); }
    catch (error) { status(error.message + ' 当前内容已保留。'); $('drafts').value = lastActiveId; }
    finally { navigating = false; $('fields').disabled = false; $('publish').disabled = false; }
  }
  function insert(text, selectionStartOffset = text.length, selectionEndOffset = selectionStartOffset) {
    const start = body.selectionStart;
    body.setRangeText(text, start, body.selectionEnd, 'end');
    body.focus(); body.setSelectionRange(start + selectionStartOffset, start + selectionEndOffset); changed();
  }
  title.addEventListener('input', changed); body.addEventListener('input', changed); target.addEventListener('change', changed);
  $('save').addEventListener('click', quietSave);
  $('drafts').addEventListener('change', () => switching($('drafts').value));
  $('new').addEventListener('click', () => switching(uuid()));
  $('reload').addEventListener('click', () => { if (!task) openDraft(id).catch(error => status(error.message)); });
  $('restore').addEventListener('click', () => { apply(restoring); recovering = false; restoring = null; $('recovery').hidden = true; changed(); });
  $('discard-backup').addEventListener('click', () => { recovering = false; restoring = null; $('recovery').hidden = true; backup(); status('已使用云端草稿'); });
  $('publish').addEventListener('click', async () => {
    if (recovering || blocked) { status('请先处理草稿版本提示。'); return; }
    $('fields').disabled = true; $('publish').disabled = true; $('drafts').disabled = true; $('new').disabled = true;
    try {
      await flush(); if (dirty() || blocked) throw new Error('草稿尚未保存成功，请先保存后再发布。');
      const result = await request('/api/guide/me/articles/' + id + '/publish', {method:'POST', body: JSON.stringify({revision})});
      revision = result.data.revision; backup();
      $('published-link').href = result.data.url; $('published-link').hidden = false;
      status('攻略已发布'); await draftList();
    } catch (error) { status(error.message); }
    finally { $('fields').disabled = false; $('publish').disabled = false; $('drafts').disabled = false; $('new').disabled = false; }
  });
  for (const button of document.querySelectorAll('[data-insert]')) button.addEventListener('click', () => {
    const selected = body.value.slice(body.selectionStart, body.selectionEnd);
    const presets = {heading:'\n## ' + (selected || '章节标题') + '\n', bold:'**' + (selected || '重点内容') + '**',
      list:'\n- ' + (selected || '列表内容') + '\n', link:'[' + (selected || '链接说明') + '](https://)', image:'![' + (selected || '图片说明') + '](https://)'};
    insert(presets[button.dataset.insert]);
  });
  $('insert-video').addEventListener('click', () => {
    const input = $('video').value.trim();
    let bvid = /^BV[0-9A-Za-z]{10}$/.test(input) ? input : '';
    if (!bvid) {
      try {
        const url = new URL(input);
        if (['https:', 'http:'].includes(url.protocol) && ['bilibili.com', 'www.bilibili.com'].includes(url.hostname)
            && !url.username && !url.password) bvid = url.pathname.match(/^\/video\/(BV[0-9A-Za-z]{10})\/?$/)?.[1] || '';
      } catch (_) { /* A complete BV or URL is required. */ }
    }
    if (!bvid) { status('请输入有效的B站视频链接或BV号。'); $('video').focus(); return; }
    insert('\n\n```bilibili\n' + bvid + '\n```\n\n'); $('video').value = '';
  });
  body.addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); quietSave(); }
  });
  window.addEventListener('online', quietSave);
  document.addEventListener('visibilitychange', () => { if (document.hidden && token) { if (!recovering) backup(); quietSave(); } });
  window.addEventListener('beforeunload', event => { if (token && dirty()) { if (!recovering) backup(); event.preventDefault(); event.returnValue = ''; } });
  window.addEventListener('storage', event => { if (event.key === 'nte_token' && token) sessionValid(); });
  (async () => {
    token = await window.guideAuth?.ready || '';
    if (!token) { $('login').hidden = false; $('login-link').href = '/login?next=' + encodeURIComponent(location.pathname + location.search); status('登录后可保存草稿'); return; }
    try {
      const me = await request('/api/me'); account = me.player_uid;
      const options = await request('/api/guide/article-targets');
      const groups = new Map();
      for (const item of options.items) {
        let group = groups.get(item.kind);
        if (!group) { group = document.createElement('optgroup'); group.label = item.kind_name; groups.set(item.kind, group); target.append(group); }
        const option = document.createElement('option'); option.value = item.kind + '/' + item.id; option.textContent = item.name; group.append(option);
      }
      const params = new URLSearchParams(location.search);
      const hasTarget = params.has('target_kind') && params.has('target_id');
      const candidate = params.get('id') || (!hasTarget ? readStorage('nte-guide-article-last:' + account) : '') || uuid();
      if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(candidate)) throw new Error('草稿地址无效，请从攻略页重新进入。');
      await openDraft(candidate); await draftList();
      if (hasTarget && revision === 0 && !recovering) {
        const choice = params.get('target_kind') + '/' + params.get('target_id');
        if (Array.from(target.options).some(option => option.value === choice)) { target.value = choice; changed(); }
      }
      $('workspace').hidden = false;
    } catch (error) { status(error.message); $('login').hidden = false; }
  })();
})();
