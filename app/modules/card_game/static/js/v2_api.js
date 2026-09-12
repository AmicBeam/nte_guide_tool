(() => {
  const API_ROOT = '/api/duel-v2';
  const TOKEN_KEY = 'nte_token';
  const LOGIN_REDIRECT_KEY = 'nte_login_redirect';
  const CHARACTER_ORDER = ['nanali', 'zero', 'jiuyuan', 'xun'];
  const CHARACTER_FALLBACK = {
    nanali: {
      id: 'nanali',
      name: '娜娜莉',
      attribute: '灵',
      attack: 3,
      max_hp: 7,
      avatar: '/static/images/characters/avatar/娜娜莉.webp',
      portrait: '/static/images/characters/portrait/娜娜莉.webp',
      role: '主要攻击手，自身成长提高攻击',
    },
    zero: {
      id: 'zero',
      name: '主角「零」',
      attribute: '光',
      attack: 2,
      max_hp: 9,
      avatar: '/static/images/characters/avatar/鉴定师.webp',
      portrait: '/static/images/characters/portrait/鉴定师.webp',
      role: '快速积累自己的环合值',
    },
    jiuyuan: {
      id: 'jiuyuan',
      name: '九原',
      attribute: '灵',
      attack: 2,
      max_hp: 8,
      avatar: '/static/images/characters/avatar/九原.webp',
      portrait: '/static/images/characters/portrait/九原.webp',
      role: '强化全队创生',
    },
    xun: {
      id: 'xun',
      name: '浔',
      attribute: '光',
      attack: 2,
      max_hp: 8,
      avatar: '/static/images/characters/avatar/浔.webp',
      portrait: '/static/images/characters/portrait/浔.webp',
      role: '记录队友牌并以终结战斗牌兑现',
    },
  };
  const CARD_TYPE_LABELS = {
    battle: '战斗',
    tactic: '战术',
    form: '形态',
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
    return payload;
  }

  function getCatalog() {
    return request('/catalog');
  }

  function getState() {
    return request('/state');
  }

  function startGame(body) {
    return request('/start', { method: 'POST', body: body || {} });
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
    return CHARACTER_FALLBACK[String(characterId || '')] || null;
  }

  function characterAsset(character, kind) {
    kind = kind || 'avatar';
    const fallback = characterFallback(character && character.id);
    return (character && character[kind]) || (fallback && fallback[kind]) || '';
  }

  function imageMarkup(src, alt, className) {
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
    const base = CARD_TYPE_LABELS[type] || type || '卡牌';
    return terminal ? base + '·终结' : base;
  }

  function sortedCharacters(characters) {
    const byId = {};
    (Array.isArray(characters) ? characters : []).forEach(function (item) {
      if (item && item.id) {
        byId[item.id] = item;
      }
    });
    return CHARACTER_ORDER.map(function (id) {
      return Object.assign({}, CHARACTER_FALLBACK[id] || { id: id }, byId[id] || {});
    }).concat((Array.isArray(characters) ? characters : []).filter(function (item) {
      return item && item.id && CHARACTER_ORDER.indexOf(item.id) < 0;
    }));
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

  function cardsForCharacter(cards, characterId) {
    return (Array.isArray(cards) ? cards : []).filter(function (card) {
      return card && card.character_id === characterId;
    }).sort(function (left, right) {
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
    node.textContent = message || '';
    node.dataset.kind = kind || (message ? 'info' : '');
    setHidden(node, !message);
  }

  window.NTE_V2 = {
    API_ROOT: API_ROOT,
    CHARACTER_ORDER: CHARACTER_ORDER,
    CHARACTER_FALLBACK: CHARACTER_FALLBACK,
    CARD_TYPE_LABELS: CARD_TYPE_LABELS,
    V2ApiError: V2ApiError,
    getToken: getToken,
    ensureLogin: ensureLogin,
    redirectToLogin: redirectToLogin,
    makeRequestId: makeRequestId,
    request: request,
    getCatalog: getCatalog,
    getState: getState,
    startGame: startGame,
    joinRoom: joinRoom,
    setReady: setReady,
    roomStart: roomStart,
    saveBuild: saveBuild,
    sendAction: sendAction,
    leaveRoom: leaveRoom,
    escapeHtml: escapeHtml,
    escapeAttr: escapeAttr,
    elementIcon: elementIcon,
    characterFallback: characterFallback,
    characterAsset: characterAsset,
    imageMarkup: imageMarkup,
    attributeMarkup: attributeMarkup,
    cardTypeLabel: cardTypeLabel,
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
