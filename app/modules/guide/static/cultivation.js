'use strict';

async function readGuidePrivateResponse(response) {
  const loginMessage = '请先登录本站，或重新登录后同步。';
  if (response.status === 401 || (response.redirected && new URL(response.url, location.href).pathname === '/login')) {
    throw new Error(loginMessage);
  }
  let payload;
  try { payload = await response.json(); } catch (_) {
    throw new Error(response.status === 404 ? '同步接口暂不可用，请联系管理员。' : '网站服务暂时不可用，请稍后重试。');
  }
  if (!response.ok) {
    const message = typeof payload?.error === 'string' ? payload.error : payload?.error?.message;
    throw new Error(typeof message === 'string' ? message : '读取失败，请稍后重试。');
  }
  if (!payload || typeof payload !== 'object' || (!Object.hasOwn(payload, 'data') && payload.ok !== true)) {
    throw new Error('网站返回的资料格式异常，请稍后重试。');
  }
  return payload.data;
}

function guideSavedCharacterLines(saved, labels = {}) {
  const fork = saved.equipment?.fork || {};
  const suit = saved.equipment?.suit || {};
  const mix = typeof fork.slev === 'number' || typeof fork.slev === 'string' ? String(fork.slev) : '';
  const mixLabel = /^[1-5]$/.test(mix) ? mix : '未提供';
  const skillLabels = Object.fromEntries(Object.entries(labels).map(([id, label]) => [id.toLowerCase(), label]));
  const lines = [`角色 ${saved.level ?? '未提供'} 级 · 觉醒 ${saved.awaken_level ?? '未提供'}`,
    `弧盘：${fork.name || '未提供'}${fork.alev ? ` · ${fork.alev}级` : ''} · 混频 ${mixLabel}`,
    '技能等级（接口展示值，计算前请核对基础等级）：'];
  if (saved.skills?.length) {
    for (const skill of saved.skills) {
      lines.push(`${skillLabels[String(skill.id || '').toLowerCase()] || skill.name || skill.id || '未知技能'}：${skill.level == null ? '等级未提供' : `${skill.level}级`}`);
    }
  } else {
    lines.push('已保存数据没有技能等级，请重新同步或手动填写。');
  }
  lines.push(`空幕 / 卡带：${suit.name || '未提供'}`);
  for (const item of [...(suit.core || []), ...(suit.pie || [])]) {
    const props = [...(item.main_properties || []), ...(item.properties || [])].map(p => `${p.name || p.id} ${p.value}`).join('、');
    lines.push(`${item.name || item.id} · ${item.level ?? '未提供'}级${props ? ' · ' + props : ''}`);
  }
  return lines;
}

const cultivationForm = document.querySelector('form[data-boundary-levels]');
if (cultivationForm) {
  const boundaries = JSON.parse(cultivationForm.dataset.boundaryLevels);
  for (const field of cultivationForm.querySelectorAll('[data-boundary-side]')) {
    const level = cultivationForm.elements.namedItem(`${field.dataset.boundarySide}_level`);
    const select = field.querySelector('select');
    const update = () => {
      const boundary = boundaries.includes(Number(level.value));
      field.hidden = !boundary;
      select.disabled = !boundary;
    };
    level.addEventListener('input', () => {
      select.value = '0';
      update();
    });
    update();
  }
}


if (cultivationForm) {
  for (const group of cultivationForm.querySelectorAll('.cultivation-sliders')) {
    const current = group.querySelector('[name^="current_skill_"]');
    const target = group.querySelector('[name^="target_skill_"]');
    const update = () => {
      if (Number(target.value) < Number(current.value)) target.value = current.value;
      for (const input of [current, target]) {
        document.getElementById(`${input.id}_value`).textContent = input.value;
        input.setAttribute('aria-valuetext', input.value === '0' ? '0级，未解锁' : `${input.value}级`);
      }
    };
    current.addEventListener('input', update);
    target.addEventListener('input', update);
    update();
  }
}

(() => {
  const root = document.querySelector('[data-account-profiles], [data-character-id]');
  if (!root) return;
  const accountPage = root.hasAttribute('data-account-profiles');
  if (!accountPage && !cultivationForm) return;
  const $ = (id) => document.getElementById(id);
  let profiles = [], sessionId = '', choosing = false, busy = false, cooldownUntil = 0;
  let token = '';
  try { token = localStorage.getItem('nte_token') || ''; } catch (_) { /* anonymous works */ }
  const status = (message) => { $('personal-status').textContent = message; };
  const api = async (path, body, method) => {
    let response;
    try {
      response = await fetch('/api/guide/me/' + path, {
        method: method || (body ? 'POST' : 'GET'), cache: 'no-store',
        headers: { Authorization: `Bearer ${token}`, ...(body ? { 'Content-Type': 'application/json' } : {}) },
        ...(body ? { body: JSON.stringify(body) } : {})
      });
    } catch (_) {
      throw new Error('网络连接失败，请检查网络后重试。');
    }
    return readGuidePrivateResponse(response);
  };
  const selected = () => profiles.find(p => p.role_id === $('profile-select').value);
  const savedCharacter = () => selected()?.characters.find(c => c.id === root.dataset.characterId);
  const equipment = () => {
    if (accountPage) { renderAccount(); return; }
    const saved = savedCharacter();
    $('saved-equipment').hidden = !saved;
    if (!saved) return;
    const labels = JSON.parse(root.dataset.skillLabels || '{}');
    $('saved-equipment-content').textContent = guideSavedCharacterLines(saved, labels).join('\n');
    $('use-profile').disabled = false;
  };
  const renderAccount = () => {
    const profile = selected();
    const list = $('account-characters');
    list.replaceChildren();
    $('account-summary').textContent = profile ? `账号 ${profile.role_id} · ${profile.characters.length} 名角色 · 同步于 ${new Date(profile.updated_at).toLocaleString('zh-CN')}` : '';
    if (!profile) { window.guideAccountDetails.close(); return; }
    const catalog = JSON.parse($('account-catalog').textContent);
    window.guideAccountDetails.close();
    for (const saved of profile.characters) {
      const local = catalog[saved.id];
      const card = document.createElement('button');
      card.type = 'button'; card.className = 'account-character';
      card.setAttribute('aria-haspopup', 'dialog');
      card.setAttribute('aria-controls', 'account-detail-dialog');
      if (local?.avatar) {
        const image = document.createElement('img');
        image.src = local.avatar; image.alt = ''; image.loading = 'lazy';
        card.append(image);
      }
      const title = document.createElement('span');
      const name = document.createElement('strong');
      name.textContent = local?.name || saved.name || saved.id;
      const level = document.createElement('small');
      level.textContent = `Lv.${saved.level ?? '未提供'} · 觉醒 ${saved.awaken_level ?? '未提供'}`;
      title.append(name, level); card.append(title);
      card.addEventListener('click', () => window.guideAccountDetails.open(saved, profile, local));
      list.append(card);
    }
    if (!profile.characters.length) list.textContent = '此账号暂无已保存角色，请重新同步。';
  };
  const refresh = async () => {
    const previous = $('profile-select').value;
    profiles = await api('profiles');
    $('profile-select').replaceChildren(...profiles.map(p => new Option(p.role_name || p.role_id, p.role_id)));
    if (profiles.some(p => p.role_id === previous)) $('profile-select').value = previous;
    $('personal-controls').hidden = profiles.length === 0;
    if (profiles.length) {
      status(accountPage ? '选择游戏账号，点击角色查看练度、技能与装备详情。' : `已保存 ${profiles.length} 个游戏账号。选择账号后，可将当前角色的练度填入下方。`);
      equipment();
    } else {
      status(accountPage ? '尚未同步游戏数据。点击“同步游戏数据”，短信验证后即可保存和查看。' : '尚未同步游戏数据。可短信验证后保存，也可直接手填计算。');
      if (accountPage) renderAccount();
      else $('saved-equipment').hidden = true;
    }
  };
  const useProfile = async () => {
    if (!savedCharacter()) { status('此游戏账号没有当前角色的可用数据，请手动填写或重新同步。'); return; }
    try {
      const data = await api(`profiles/${encodeURIComponent(selected().role_id)}/characters/${encodeURIComponent(root.dataset.characterId)}/prefill`);
      for (const [key, value] of Object.entries(data.values)) {
        const input = cultivationForm.elements.namedItem(key);
        if (input) { input.value = String(value); input.dispatchEvent(new Event('input')); }
      }
      $('prefill-confirm-label').hidden = false;
      $('prefill-confirm').checked = false;
      $('prefill-confirm').required = true;
      $('prefill-notice').hidden = false;
      $('prefill-notice').textContent = ['已填入保存的当前练度。目标练度可自行调整。', ...data.warnings].join('\n');
      status(`已使用 ${selected().role_name || selected().role_id} 的数据；同步于 ${new Date(selected().updated_at).toLocaleString('zh-CN')}。`);
    } catch (error) { status(error.message); }
  };
  $('profile-select').addEventListener('change', equipment);
  if (!accountPage) $('use-profile').addEventListener('click', useProfile);
  $('delete-profile').addEventListener('click', async () => {
    if (!selected() || !confirm('删除此游戏账号保存的角色练度和装备数据？')) return;
    try { await api(`profiles/${encodeURIComponent(selected().role_id)}`, null, 'DELETE'); await refresh(); }
    catch (error) { status(error.message); }
  });
  $('sync-open').addEventListener('click', async () => {
    if (window.guideAuth) token = await window.guideAuth.check();
    if (!token) { location.href = window.guideAuth ? window.guideAuth.loginUrl() : '/login?next=' + encodeURIComponent(location.pathname + location.search + location.hash); return; }
    $('sync-dialog').showModal();
  });
  $('sync-close').addEventListener('click', () => { if (!busy) $('sync-dialog').close(); });
  $('sync-dialog').addEventListener('cancel', e => { if (busy) e.preventDefault(); });
  const setBusy = (value) => {
    busy = value;
    $('sync-submit').disabled = value;
    $('sync-mobile').disabled = value || !!sessionId;
    $('sync-close').disabled = value;
    $('sms-send').disabled = value || Date.now() < cooldownUntil;
  };
  const updateCooldown = () => {
    const remaining = Math.max(0, Math.ceil((cooldownUntil - Date.now()) / 1000));
    $('sms-send').textContent = remaining ? `${remaining}秒后重发` : '获取验证码';
    $('sms-send').disabled = busy || remaining > 0;
  };
  $('sync-reset').addEventListener('click', () => {
    if (busy) return;
    sessionId = ''; choosing = false;
    $('sync-mobile').disabled = false; $('sync-mobile').value = '';
    $('sync-code').value = ''; $('sync-role-label').hidden = true;
    $('sync-submit').textContent = '验证并同步';
    $('sync-status').textContent = '请重新填写手机号并获取验证码。';
    $('sync-mobile').focus();
  });
  $('sms-send').addEventListener('click', async () => {
    setBusy(true);
    $('sync-status').textContent = '正在发送验证码…';
    try {
      const data = await api('sync/sms', { mobile: $('sync-mobile').value.trim() });
      sessionId = data.session_id; choosing = false;
      $('sync-role-label').hidden = true;
      $('sync-submit').textContent = '验证并同步';
      cooldownUntil = Date.now() + 60000;
      const timer = setInterval(() => { updateCooldown(); if (Date.now() >= cooldownUntil) clearInterval(timer); }, 1000);
      $('sync-status').textContent = '验证码已发送，10分钟内有效。';
      $('sync-code').focus();
    } catch (error) { $('sync-status').textContent = error.message; }
    finally { setBusy(false); updateCooldown(); }
  });
  $('sync-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!sessionId) { $('sync-status').textContent = '请先获取短信验证码。'; return; }
    setBusy(true); $('sync-status').textContent = '正在验证并同步角色资料…';
    try {
      const data = await api(choosing ? 'sync/select-role' : 'sync/verify', {
        session_id: sessionId, code: $('sync-code').value.trim(), role_id: $('sync-role').value
      });
      $('sync-code').value = '';
      if (data.status === 'select_role') {
        choosing = true;
        $('sync-role').replaceChildren(...data.roles.map(r => new Option(r.role_name || r.role_id, r.role_id)));
        $('sync-role-label').hidden = false;
        $('sync-status').textContent = '验证成功，请选择要同步的游戏账号。';
        $('sync-submit').textContent = '保存所选账号';
      } else {
        sessionId = ''; choosing = false; $('sync-mobile').value = '';
        await refresh(); $('profile-select').value = data.profile.role_id; equipment(); if (!accountPage) await useProfile();
        $('sync-dialog').close();
      }
    } catch (error) { $('sync-status').textContent = error.message; }
    finally { setBusy(false); }
  });
  const ready = window.guideAuth ? window.guideAuth.ready : Promise.resolve(token);
  ready.then(value => {
    token = value;
    if (token) refresh().catch(error => status(error.message));
  });
})();
