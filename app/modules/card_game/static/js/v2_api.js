(() => {
  const API_ROOT = '/api/duel-v2';
  const TOKEN_KEY = 'nte_token';
  const LOGIN_REDIRECT_KEY = 'nte_login_redirect';
  const CHARACTER_ORDER = ['zero', 'xun', 'zhenhong', 'nanali', 'jiuyuan', 'lingke', 'iloy', 'hathor', 'anhunqu', 'canhong', 'zaowu'];
  const CHARACTER_FALLBACK = {
    baicang: { id: 'baicang', portrait: '/static/images/characters/portrait/白藏.webp', avatar: '/static/images/characters/avatar/白藏.webp' },
    bohe: { id: 'bohe', portrait: '/static/images/characters/portrait/薄荷.webp', avatar: '/static/images/characters/avatar/薄荷.webp' },
    nanali: { id: 'nanali', name: '娜娜莉', attribute: '灵', attack: 2, max_hp: 5, avatar: '/static/images/characters/avatar/娜娜莉.webp', portrait: '/static/images/characters/portrait/娜娜莉.webp' },
    zero: { id: 'zero', name: '零', attribute: '光', attack: 2, max_hp: 5, avatar: '/static/images/characters/avatar/鉴定师.webp', portrait: '/static/images/characters/portrait/鉴定师.webp' },
    jiuyuan: { id: 'jiuyuan', name: '九原', attribute: '灵', attack: 3, max_hp: 4, avatar: '/static/images/characters/avatar/九原.webp', portrait: '/static/images/characters/portrait/九原.webp' },
    xun: { id: 'xun', name: '浔', attribute: '光', attack: 2, max_hp: 5, avatar: '/static/images/characters/avatar/浔.webp', portrait: '/static/images/characters/portrait/浔.webp' },
    anhunqu: { id: 'anhunqu', name: '安魂曲', attribute: '暗', attack: 2, max_hp: 5, avatar: '/static/images/characters/avatar/安魂曲.webp', portrait: '/static/images/characters/portrait/安魂曲.webp' },
    canhong: { id: 'canhong', name: '残虹', attribute: '咒', attack: 2, max_hp: 5, avatar: '/static/images/characters/avatar/残虹.png', portrait: '/static/images/characters/portrait/残虹.webp' },
    zaowu: { id: 'zaowu', name: '早雾', attribute: '咒', attack: 3, max_hp: 4, avatar: '/static/images/characters/avatar/早雾.webp', portrait: '/static/images/characters/portrait/早雾.webp' },
    lingke: { id: 'lingke', name: '灵可', attribute: '灵', attack: 3, max_hp: 4, avatar: '/static/images/characters/avatar/灵可.png', portrait: '/static/images/characters/portrait/灵可.webp' },
    zhenhong: { id: 'zhenhong', name: '真红', attribute: '光', attack: 3, max_hp: 4, avatar: '/static/kongmu/images/characters/player_zhenhong_256.webp', portrait: '/static/images/characters/portrait/真红.webp' },
    hathor: { id: 'hathor', name: '哈索尔', attribute: '相', attack: 2, max_hp: 5, avatar: '/static/images/characters/avatar/哈索尔.webp', portrait: '/static/images/characters/portrait/哈索尔.webp' },
    iloy: { id: 'iloy', name: '伊洛伊', attribute: '灵', attack: 1, max_hp: 6, avatar: '/static/kongmu/images/characters/player_yiluoyi_256.webp', portrait: '/static/images/characters/portrait/伊洛伊.webp' },
  };
  // Presentation only: equipment identity comes from the public board (also in replays).
  // Sources and user selections: docs/character-image-assets-manifest.json.
  const EQUIPMENT_PORTRAITS = {
    N07: { characterId: 'nanali', fashion: 'Fashion_1010_3' },
    N08: { characterId: 'nanali', fashion: 'Fashion_1010_nighty' },
    Z07: { characterId: 'zero', fashion: 'Fashion_1051_2' },
    J08: { characterId: 'jiuyuan', fashion: 'Fashion_1055_nighty' },
    X08: { characterId: 'xun', fashion: 'Fashion_1052_nighty' },
    A07: { characterId: 'anhunqu', fashion: 'Fashion_1004_1' },
    A08: { characterId: 'anhunqu', fashion: 'Fashion_1004_nighty' },
    C07: { characterId: 'canhong', fashion: 'Fashion_1036_1' },
    C08: { characterId: 'canhong', fashion: 'Fashion_1036_nighty' },
    S08: { characterId: 'zaowu', fashion: 'Fashion_1003_1' },
    K07: { characterId: 'lingke', fashion: 'Fashion_1072_school' },
    K08: { characterId: 'lingke', fashion: 'Fashion_1072_nighty' },
    R07: { characterId: 'zhenhong', fashion: 'Fashion_1076_1' },
    R08: { characterId: 'zhenhong', fashion: 'Fashion_1076_nighty' },
    Y07: { characterId: 'iloy', fashion: 'Fashion_1075_rpg_level3' },
    Y08: { characterId: 'iloy', fashion: 'Fashion_1075_nighty' },
    M07: { characterId: 'bohe', fashion: 'Fashion_1019_rpg_level3' },
    M08: { characterId: 'bohe', fashion: 'Fashion_1019_nighty' },
    Q07: { characterId: 'xiaozhi', fashion: 'Fashion_1073_1' },
    Q08: { characterId: 'xiaozhi', fashion: 'Fashion_1073_nighty' },
  };
  const CHARACTER_IMAGE_VERSION = 'ntedata-lingke-20261008-v5';

  // Server-provided identity/assets are authoritative; the static table only
  // supports old snapshots. Do not cache mutable combat stats across sides.
  const characterAssets = Object.create(null);
  const cardsByName = Object.create(null);

  function registerCharacters(characters) {
    (Array.isArray(characters) ? characters : []).forEach(function (character) {
      if (!character || typeof character.id !== 'string' || !character.id) return;
      const entry = characterAssets[character.id] || { id: character.id };
      ['name', 'attribute', 'avatar', 'portrait'].forEach(function (key) {
        if (typeof character[key] === 'string' && character[key]) entry[key] = character[key];
      });
      characterAssets[character.id] = entry;
      (character.mechanisms || []).forEach(function (mechanism) {
        [mechanism.name].concat(mechanism.reference_names || []).forEach(function (name) {
          if (name) cardsByName[name] = Object.assign({}, mechanism);
        });
      });
    });
  }

  function registerPayloadCharacters(payload) {
    registerCharacters(payload && payload.characters);
    ((payload && payload.cards) || []).forEach(function (card) {
      if (card && card.name) cardsByName[card.name] = Object.assign({}, card);
    });
    const boards = [payload && payload.game, payload && payload.replay && payload.replay.opening_board];
    boards.forEach(function (board) {
      Object.values((board && board.sides) || {}).forEach(function (side) {
        registerCharacters(side && side.characters);
      });
    });
  }

  const CARD_TYPE_ORDER = { tactic: 0, battle: 1, form: 2 };
  const CARD_TYPE_LABELS = {
    battle: '战斗',
    tactic: '战术',
    form: '武备',
    mechanism: '机制说明',
  };

  function getToken() {
    if (typeof window.getToken === 'function') {
      return window.getToken() || '';
    }
    try {
      return window.localStorage.getItem(TOKEN_KEY) || '';
    } catch (error) {
      return '';
    }
  }

  function currentRelativeUrl() {
    return window.location.pathname + window.location.search + window.location.hash;
  }

  function loginUrlForCurrentPage() {
    if (typeof window.loginUrlForCurrentPage === 'function') {
      return window.loginUrlForCurrentPage();
    }
    const target = currentRelativeUrl();
    try {
      window.localStorage.setItem(LOGIN_REDIRECT_KEY, target);
    } catch (error) {
      // query 参数仍可用于登录后回跳。
    }
    return '/login?next=' + encodeURIComponent(target);
  }

  function redirectToLogin() {
    if (typeof window.redirectToLogin === 'function') {
      window.redirectToLogin();
      return;
    }
    try {
      window.localStorage.removeItem(TOKEN_KEY);
    } catch (error) {
      // 无法清理本地登录态时仍跳转登录页。
    }
    window.location.replace(loginUrlForCurrentPage());
  }

  function ensureLogin() {
    if (typeof window.ensureLogin === 'function') {
      return window.ensureLogin();
    }
    if (!getToken()) {
      redirectToLogin();
      return false;
    }
    return true;
  }

  function makeRequestId() {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID();
    }
    const random = function () {
      return Math.random().toString(16).slice(2).padEnd(8, '0');
    };
    return 'v2-' + Date.now().toString(16) + '-' + random() + '-' + random();
  }

  function nextLogId() {
    if (typeof window.nextLogId === 'function') {
      return window.nextLogId();
    }
    return 'v2-' + Date.now();
  }

  class V2ApiError extends Error {
    constructor(message, options) {
      super(message);
      this.name = 'V2ApiError';
      options = options || {};
      this.status = options.status || 0;
      this.payload = options.payload || {};
      this.logId = options.logId || '';
    }
  }

  async function request(path, options) {
    options = options || {};
    const headers = Object.assign({}, options.headers || {});
    const token = getToken();
    if (token) {
      headers.Authorization = 'Bearer ' + token;
    }
    headers['X-Log-Id'] = headers['X-Log-Id'] || nextLogId();
    if (options.body !== undefined && !headers['Content-Type']) {
      headers['Content-Type'] = 'application/json';
    }
    const fetchOptions = Object.assign({}, options, { headers: headers });
    if (options.body !== undefined && typeof options.body !== 'string') {
      fetchOptions.body = JSON.stringify(options.body);
    }
    let response;
    try {
      response = await fetch(API_ROOT + path, fetchOptions);
    } catch (error) {
      throw new V2ApiError('网络不可用，请检查连接后重试。', {
        status: 0,
        payload: { error: 'network' },
      });
    }
    let payload = {};
    try {
      payload = await response.json();
    } catch (error) {
      payload = {};
    }
    const logId = response.headers.get('X-Log-Id') || headers['X-Log-Id'];
    if (response.status === 401) {
      redirectToLogin();
      throw new V2ApiError('未登录或登录态已失效。', {
        status: 401,
        payload: payload,
        logId: logId,
      });
    }
    if (!response.ok || payload.error) {
      const fallback = response.status === 404 ? '当前没有房间。' : '请求失败';
      const message = payload.error || fallback;
      if (String(message).indexOf('token') >= 0 || String(message).indexOf('未登录') >= 0) {
        redirectToLogin();
      }
      throw new V2ApiError(logId ? message + '（logId: ' + logId + '）' : message, {
        status: response.status,
        payload: payload,
        logId: logId,
      });
    }
    registerPayloadCharacters(payload);
    return payload;
  }

  function getCatalog() {
    return request('/catalog');
  }

  function getState() {
    return request('/state');
  }

  function listReplays() {
    return request('/replays');
  }

  function getReplay(roomCode) {
    return request('/replay/' + encodeURIComponent(String(roomCode || '').trim()));
  }

  function starReplay(roomCode, favorited) {
    return request('/replay-star', {
      method: 'POST',
      body: {
        room_code: String(roomCode || '').trim(),
        favorited: favorited !== false,
      },
    });
  }

  function importReplay(payload) {
    return request('/replay-import', { method: 'POST', body: payload || {} });
  }

  function startGame(body) {
    return request('/start', { method: 'POST', body: body || {} });
  }

  function skipTutorial(body) {
    return request('/tutorial-skip', { method: 'POST', body: body || {} });
  }

  function joinRoom(roomCode) {
    return request('/join', { method: 'POST', body: { room_code: String(roomCode || '').trim() } });
  }

  function setReady(isReady) {
    return request('/ready', { method: 'POST', body: { is_ready: isReady !== false } });
  }

  function roomStart() {
    return request('/room-start', { method: 'POST', body: {} });
  }

  function saveBuild(build) {
    return request('/build', { method: 'POST', body: build });
  }

  function deleteBuild(buildId) {
    return request('/build-delete', { method: 'POST', body: { id: buildId } });
  }

  function selectBuild(buildId) {
    return request('/build-select', { method: 'POST', body: { id: buildId } });
  }

  function reorderBuilds(ids) {
    return request('/build-reorder', { method: 'POST', body: { ids: ids || [] } });
  }

  function sendAction(params) {
    params = params || {};
    const id = params.requestId || makeRequestId();
    const expected = Number(params.expectedVersion);
    if (!Number.isInteger(expected) || expected < 0) {
      return Promise.reject(new V2ApiError('当前对局版本未知，请刷新后再操作。'));
    }
    return request('/action', {
      method: 'POST',
      body: {
        room_code: params.roomCode,
        request_id: id,
        expected_version: expected,
        action: params.action,
      },
    }).then(function (payload) {
      return Object.assign({ request_id: id }, payload);
    });
  }

  function leaveRoom() {
    return request('/leave', { method: 'POST', body: {} });
  }

  function advanceAi(params) {
    return request('/ai-step', {
      method: 'POST',
      body: {
        room_code: params.roomCode,
        request_id: params.requestId,
        expected_version: params.expectedVersion,
      },
    });
  }

  function formatReplayTime(value) {
    if (!value) {
      return '';
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return '';
    }
    const pad = function (n) {
      return String(n).padStart(2, '0');
    };
    return date.getFullYear() + '-' + pad(date.getMonth() + 1) + '-' + pad(date.getDate())
      + ' ' + pad(date.getHours()) + ':' + pad(date.getMinutes());
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, function (char) {
      return ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
      })[char];
    });
  }

  function cardDescriptionMarkup(text, fallback) {
    const source = text == null ? fallback : text;
    return String(source || '').split(/(「[^」]+」)/g).map(function (part) {
      const referenced = part.startsWith('「') && cardsByName[part.slice(1, -1)];
      const html = escapeHtml(part).replace(/(?:[二三四五六七八九十百X]|[0-9]+)选一|瞬发|响应|穿透|远程|可重复使用|不消耗行动力/g, function (keyword) {
        return '<strong>' + keyword + '</strong>';
      });
      return referenced && referenced.tooltip
        ? '<span title="' + escapeAttr(referenced.tooltip) + '">' + html + '</span>' : html;
    }).join('');
  }

  function handCardFace(card) {
    if (!card || card.hidden || !card.hand_face) return card;
    const face = Object.assign({}, card);
    // Clear absent physical stats rather than inheriting the effective battle frame.
    ['name', 'type', 'description', 'stat', 'stat_value', 'shield', 'attack', 'hp', 'attack_mode', 'derived'].forEach(function (key) {
      delete face[key];
      if (Object.prototype.hasOwnProperty.call(card.hand_face, key)) face[key] = card.hand_face[key];
    });
    return face;
  }

  function referencedCards(card) {
    const result = [];
    const seen = new Set();
    const regex = /「([^」]+)」/g;
    const text = (card && card.description) || '';
    let match;
    while ((match = regex.exec(text))) {
      const name = match[1];
      const referenced = cardsByName[name];
      const identity = referenced && (referenced.id || referenced.card_id || name);
      if (referenced && name !== card.name && (!card.id || referenced.id !== card.id) && !seen.has(identity)) {
        result.push(Object.assign({}, referenced));
        seen.add(identity);
      }
    }
    return result;
  }

  function cardDescriptionHtml(text, fallback, className) {
    const html = cardDescriptionMarkup(text, fallback);
    if (!html) {
      return '';
    }
    return '<p' + (className ? ' class="' + escapeAttr(className) + '"' : '') + '>' + html + '</p>';
  }

  function escapeAttr(value) {
    return escapeHtml(value).replace(/\u0060/g, '&#96;');
  }

  function elementIcon(attribute) {
    const token = String(attribute || '').trim();
    if (!token) {
      return '';
    }
    return '/static/images/elements/' + encodeURIComponent(token) + '.png';
  }

  function characterFallback(characterId) {
    const id = String(characterId || '');
    const local = characterAssets[id];
    const fallback = CHARACTER_FALLBACK[id];
    return local ? Object.assign({}, fallback || {}, local) : fallback || null;
  }

  function characterAsset(character, kind) {
    kind = kind || 'avatar';
    const fallback = characterFallback(character && character.id);
    return characterImageUrl((character && character[kind]) || (fallback && fallback[kind]) || '');
  }

  function characterImageUrl(src) {
    if (!src) return '';
    if (src.startsWith('/static/images/characters/') && !/[?&]nte_art=/.test(src)) {
      return src + (src.includes('?') ? '&' : '?') + 'nte_art=' + CHARACTER_IMAGE_VERSION;
    }
    return src;
  }

  function battlePortrait(character) {
    const data = character || {};
    const costume = EQUIPMENT_PORTRAITS[data.shape_id];
    if (!data.summoned && !(Number(data.down_turns) > 0) && costume && costume.characterId === data.id) {
      return characterImageUrl('/static/images/characters/costume/' + costume.fashion + '.webp');
    }
    return characterAsset(data, 'portrait') || characterAsset(data, 'avatar');
  }

  function imageMarkup(src, alt, className) {
    src = characterImageUrl(src);
    if (!src) {
      return '<span class="' + escapeAttr(className || '') + ' v2-image-fallback" aria-hidden="true"></span>';
    }
    return '<img class="' + escapeAttr(className || '') + '" src="' + escapeAttr(src) + '" alt="' + escapeAttr(alt || '') + '" draggable="false">';
  }

  function attributeMarkup(attribute) {
    const icon = elementIcon(attribute);
    if (!icon) {
      return escapeHtml(attribute || '');
    }
    return '<span class="v2-attribute"><img src="' + escapeAttr(icon) + '" alt="">' + escapeHtml(attribute) + '</span>';
  }

  function cardTypeLabel(type, terminal) {
    return CARD_TYPE_LABELS[type] || type || '卡牌';
  }

  function cardTypeToken(type) {
    return type === 'battle' || type === 'tactic' || type === 'form' ? type : '';
  }

  function cardTimingClass(card) {
    const instant = !!(card && card.instant);
    const response = !!(card && card.response);
    if (instant && response) {
      return 'is-instant-response';
    }
    if (response) {
      return 'is-response';
    }
    if (instant) {
      return 'is-instant';
    }
    return '';
  }

  function cardCostBadge(card) {
    return '';
  }

  function cardTypeChip(type, terminal) {
    const token = cardTypeToken(type);
    return '<span class="v2-chip"' +
      (token ? ' data-card-type="' + escapeAttr(token) + '"' : '') + '>' +
      escapeHtml(cardTypeLabel(type, terminal)) + '</span>';
  }

  function cardDerivedChip(card) {
    if (!card || !card.derived) {
      return '';
    }
    return '<span class="v2-chip">不可构筑</span>';
  }

  function cardFrameStatText(card) {
    if (!card) {
      return '';
    }
    const parts = [];
    if (card.type === 'form') {
      if (Number(card.attack) > 0) {
        parts.push('攻击 ' + card.attack);
      }
      if (Number(card.hp) > 0) {
        parts.push('生命 ' + card.hp);
      }
    }
    if (card.type === 'battle') {
      if (Number(card.attack) !== 0 && Number.isFinite(Number(card.attack))) {
        parts.push('攻击 ' + (card.attack_mode !== 'set' && Number(card.attack) > 0 ? '+' : '') + card.attack);
      }
      if (Number(card.shield) > 0) {
        parts.push('护盾 ' + card.shield);
      }
    }
    return parts.join(' · ');
  }

  function statIconSvg(kind) {
    if (kind === 'fang') {
      return '<svg class="v2-stat-icon" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M2 2h5c1 5-1 9-4 12L2 2zm7 0h5l-1 12c-3-3-5-7-4-12z"/></svg>';
    }
    if (kind === 'aura') {
      return '<svg class="v2-stat-icon" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M6 3l8-2v10h-2V4L8 5v8H6V3z"/><ellipse cx="4.5" cy="12.5" rx="3" ry="2.5" fill="currentColor"/><ellipse cx="10.5" cy="10.5" rx="3" ry="2.5" fill="currentColor"/></svg>';
    }
    if (kind === 'jingu') {
      return '<svg class="v2-stat-icon" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.5" fill="currentColor"/><path d="M6 5h4v6H6z" fill="none" stroke="#79520c" stroke-width="1.5"/><path d="M8 2v2M8 12v2" stroke="#79520c"/></svg>';
    }
    if (kind === 'hp') {
      return '<svg class="v2-stat-icon" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 14s-6-3.6-6-7.2C2 4.6 3.6 3 5.4 3c1.1 0 2.1.6 2.6 1.5C8.5 3.6 9.5 3 10.6 3 12.4 3 14 4.6 14 6.8 14 10.4 8 14 8 14z"/></svg>';
    }
    if (kind === 'atk') {
      return '<svg class="v2-stat-icon" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M13.7 2.3 9 3.8 7.2 5.6 6 5.3 4.6 6.7l1.8 1.8-3.7 3.7-.9 2 2-.9 3.7-3.7 1.8 1.8 1.4-1.4-.3-1.2 1.8-1.8 1.5-4.7z"/></svg>';
    }
    return '<svg class="v2-stat-icon" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 2c1.6 1.2 3.2 1.8 4.8 1.8V8c0 3.1-2.1 5.4-4.8 6.2C5.3 13.4 3.2 11.1 3.2 8V3.8C4.8 3.8 6.4 3.2 8 2z"/></svg>';
  }

  function statBadge(kind, shown, title) {
    return '<span class="v2-stat-badge is-' + kind + '" title="' + escapeAttr(title) + '">' +
      statIconSvg(kind) + '<em>' + escapeHtml(shown) + '</em></span>';
  }

  function energyMaxForAttribute(attribute) {
    const token = String(attribute || '').trim();
    return token === '暗' || token === '魂' || token === '咒' ? 6 : 5;
  }

  function characterEnergyMax(character) {
    if (character && character.energy_max != null && character.energy_max !== '') {
      const value = Number(character.energy_max);
      if (Number.isFinite(value) && value > 0) {
        return value;
      }
    }
    return energyMaxForAttribute(character && character.attribute);
  }

  function characterStatMarkup(character) {
    const attack = character && character.attack != null ? character.attack : '-';
    const hp = character && character.max_hp != null ? character.max_hp : '-';
    return '<span class="v2-frame-stats v2-character-stats">' +
      statBadge('atk', String(attack), '攻击 ' + attack) +
      statBadge('hp', String(hp), '生命 ' + hp) +
      '</span>';
  }

  function characterUltimateLabel(character) {
    return '终结（' + characterEnergyMax(character) + '）';
  }

  function cardFrameStat(card) {
    if (!card) {
      return '';
    }
    const parts = [];
    if (card.type === 'form') {
      if (Number(card.attack) > 0) {
        parts.push(statBadge('atk', String(card.attack), '攻击 ' + card.attack));
      }
      if (Number(card.hp) > 0) {
        parts.push(statBadge('hp', String(card.hp), '生命 ' + card.hp));
      }
    }
    if (card.type === 'battle') {
      if (Number(card.attack) !== 0 && Number.isFinite(Number(card.attack))) {
        const value = (card.attack_mode !== 'set' && Number(card.attack) > 0 ? '+' : '') + card.attack;
        parts.push(statBadge('atk', value, (card.attack_mode === 'set' ? '设置攻击为 ' : '攻击 ') + value));
      }
      if (Number(card.shield) > 0) {
        parts.push(statBadge('shield', String(card.shield), '护盾 ' + card.shield));
      }
    }
    if (!parts.length) {
      return '';
    }
    return '<span class="v2-frame-stats">' + parts.join('') + '</span>';
  }

  function sortedCharacters(characters) {
    const list = Array.isArray(characters) ? characters : [];
    if (list.length) {
      return list.map(function (item) {
        return Object.assign({}, characterFallback(item && item.id) || { id: item && item.id }, item || {});
      });
    }
    return CHARACTER_ORDER.map(function (id) {
      return Object.assign({}, characterFallback(id) || { id: id });
    });
  }

  function indexById(list) {
    const map = {};
    (Array.isArray(list) ? list : []).forEach(function (item) {
      if (item && item.id) {
        map[item.id] = item;
      }
    });
    return map;
  }

  function cardsForCharacter(cards, characterId, options) {
    const buildableOnly = options && options.buildableOnly;
    return (Array.isArray(cards) ? cards : []).filter(function (card) {
      if (!card || card.character_id !== characterId) {
        return false;
      }
      if (buildableOnly && card.derived) {
        return false;
      }
      return true;
    }).sort(function (left, right) {
      const derived = Number(Boolean(left && left.derived)) - Number(Boolean(right && right.derived));
      if (derived) {
        return derived;
      }
      const leftType = CARD_TYPE_ORDER[left && left.type];
      const rightType = CARD_TYPE_ORDER[right && right.type];
      const byType = (leftType == null ? 9 : leftType) - (rightType == null ? 9 : rightType);
      if (byType) {
        return byType;
      }
      return String(left.id || '').localeCompare(String(right.id || ''));
    });
  }

  function countIds(ids) {
    const counts = {};
    (Array.isArray(ids) ? ids : []).forEach(function (id) {
      const key = String(id || '');
      if (!key) {
        return;
      }
      counts[key] = (counts[key] || 0) + 1;
    });
    return counts;
  }

  function expandCounts(counts, order) {
    const ids = [];
    (Array.isArray(order) ? order : Object.keys(counts || {})).forEach(function (id) {
      const copies = Math.max(0, Number((counts || {})[id] || 0));
      for (let index = 0; index < copies; index += 1) {
        ids.push(id);
      }
    });
    return ids;
  }

  function legalEntries(game) {
    return Array.isArray(game && game.legal_actions) ? game.legal_actions : [];
  }

  function findLegalEntries(game, predicate) {
    return legalEntries(game).filter(function (entry) {
      return predicate((entry && entry.action) || {}, entry);
    });
  }

  function findLegalEntry(game, predicate) {
    return findLegalEntries(game, predicate)[0] || null;
  }

  function sameIdSet(left, right) {
    const a = (Array.isArray(left) ? left : []).map(String).slice().sort();
    const b = (Array.isArray(right) ? right : []).map(String).slice().sort();
    if (a.length !== b.length) {
      return false;
    }
    return a.every(function (value, index) {
      return value === b[index];
    });
  }

  function actionFingerprint(action) {
    try {
      return JSON.stringify(action || {});
    } catch (error) {
      return String(action || '');
    }
  }

  function previewLines(preview) {
    if (!preview || typeof preview !== 'object') {
      return [];
    }
    const lines = [];
    const known = [
      ['ap_cost', '行动力'],
      ['attack', '攻击'],
      ['counter', '反击'],
      ['harmony', '环合'],
      ['will_switch', '换人'],
    ];
    known.forEach(function (pair) {
      const key = pair[0];
      const label = pair[1];
      const value = preview[key];
      if (value === undefined || value === null || value === '') {
        return;
      }
      if (key === 'harmony' && typeof value === 'boolean') {
        lines.push(value ? '环合：触发' : '环合：不触发');
        return;
      }
      if (key === 'will_switch') {
        lines.push(value ? '换人：是' : '换人：否');
        return;
      }
      lines.push(label + ' ' + value);
    });
    Object.keys(preview).forEach(function (key) {
      if (['ap_cost', 'attack', 'counter', 'harmony', 'will_switch'].indexOf(key) >= 0) {
        return;
      }
      const value = preview[key];
      if (value === undefined || value === null || value === '') {
        return;
      }
      if (typeof value === 'object') {
        return;
      }
      lines.push(key + ' ' + value);
    });
    return lines;
  }

  function previewMarkup(preview) {
    const lines = previewLines(preview);
    if (!lines.length) {
      return '<p class="subtle">暂无额外预览。</p>';
    }
    return '<ul class="v2-preview-list">' + lines.map(function (line) {
      return '<li>' + escapeHtml(line) + '</li>';
    }).join('') + '</ul>';
  }

  function roomStatusLabel(room, game) {
    if (game && game.winner) {
      if (game.winner === 'draw') {
        return '对局平局';
      }
      return game.winner === game.viewer_side ? '对局胜利' : '对局失败';
    }
    const status = String((room && room.status) || '');
    if (game) {
      if (game.phase === 'mulligan') {
        return '起手换牌';
      }
      if (game.phase === 'choice') {
        return '等待选择';
      }
      if (game.phase === 'finished') {
        return '对局结束';
      }
      return game.is_my_turn ? '你的回合' : '对手回合';
    }
    if (status === 'waiting' || status === 'ready') {
      return '房间等待中';
    }
    if (status === 'playing') {
      return '对局进行中';
    }
    return status || '待开始';
  }

  function viewerSide(game) {
    return String((game && game.viewer_side) || 'a');
  }

  function opponentSide(game) {
    return viewerSide(game) === 'b' ? 'a' : 'b';
  }

  function sideState(game, side) {
    return ((game && game.sides) || {})[side] || null;
  }

  function setText(id, value) {
    const node = typeof id === 'string' ? document.getElementById(id) : id;
    if (node) {
      node.textContent = value == null ? '' : String(value);
    }
  }

  function setHidden(node, hidden) {
    if (!node) {
      return;
    }
    node.hidden = Boolean(hidden);
    node.setAttribute('aria-hidden', hidden ? 'true' : 'false');
  }

  function showBanner(node, message, kind) {
    if (!node) {
      return;
    }
    if (node._bannerTimer) {
      window.clearTimeout(node._bannerTimer);
      node._bannerTimer = 0;
    }
    node.textContent = message || '';
    node.dataset.kind = kind || (message ? 'info' : '');
    setHidden(node, !message);
    if (message && kind === 'ok') {
      node._bannerTimer = window.setTimeout(function () {
        node._bannerTimer = 0;
        if (node.dataset.kind === 'ok') {
          showBanner(node, '', '');
        }
      }, 2400);
    }
  }

  window.NTE_V2 = {
    API_ROOT: API_ROOT,
    CHARACTER_ORDER: CHARACTER_ORDER,
    CHARACTER_FALLBACK: CHARACTER_FALLBACK,
    CARD_TYPE_LABELS: CARD_TYPE_LABELS,
    CARD_TYPE_ORDER: CARD_TYPE_ORDER,
    V2ApiError: V2ApiError,
    getToken: getToken,
    ensureLogin: ensureLogin,
    redirectToLogin: redirectToLogin,
    makeRequestId: makeRequestId,
    request: request,
    getCatalog: getCatalog,
    getState: getState,
    listReplays: listReplays,
    getReplay: getReplay,
    starReplay: starReplay,
    importReplay: importReplay,
    skipTutorial: skipTutorial,
    startGame: startGame,
    joinRoom: joinRoom,
    setReady: setReady,
    roomStart: roomStart,
    saveBuild: saveBuild,
    deleteBuild: deleteBuild,
    selectBuild: selectBuild,
    reorderBuilds: reorderBuilds,
    sendAction: sendAction,
    advanceAi: advanceAi,
    leaveRoom: leaveRoom,
    formatReplayTime: formatReplayTime,
    escapeHtml: escapeHtml,
    cardDescriptionMarkup: cardDescriptionMarkup,
    cardDescriptionHtml: cardDescriptionHtml,
    referencedCards: referencedCards,
    handCardFace: handCardFace,
    escapeAttr: escapeAttr,
    elementIcon: elementIcon,
    characterFallback: characterFallback,
    characterAsset: characterAsset,
    battlePortrait: battlePortrait,
    imageMarkup: imageMarkup,
    attributeMarkup: attributeMarkup,
    cardTypeLabel: cardTypeLabel,
    cardTypeToken: cardTypeToken,
    cardTypeChip: cardTypeChip,
    cardDerivedChip: cardDerivedChip,
    cardTimingClass: cardTimingClass,
    cardCostBadge: cardCostBadge,
    cardFrameStatText: cardFrameStatText,
    cardFrameStat: cardFrameStat,
    statIconSvg: statIconSvg,
    energyMaxForAttribute: energyMaxForAttribute,
    characterEnergyMax: characterEnergyMax,
    characterStatMarkup: characterStatMarkup,
    characterUltimateLabel: characterUltimateLabel,
    sortedCharacters: sortedCharacters,
    indexById: indexById,
    cardsForCharacter: cardsForCharacter,
    countIds: countIds,
    expandCounts: expandCounts,
    legalEntries: legalEntries,
    findLegalEntries: findLegalEntries,
    findLegalEntry: findLegalEntry,
    sameIdSet: sameIdSet,
    actionFingerprint: actionFingerprint,
    previewLines: previewLines,
    previewMarkup: previewMarkup,
    roomStatusLabel: roomStatusLabel,
    viewerSide: viewerSide,
    opponentSide: opponentSide,
    sideState: sideState,
    setText: setText,
    setHidden: setHidden,
    showBanner: showBanner,
  };
})();
