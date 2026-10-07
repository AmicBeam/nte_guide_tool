const V2 = window.NTE_V2;

if (V2.ensureLogin()) {
  bootstrapHome();
}

async function bootstrapHome() {
  const banner = document.getElementById('v2-home-banner');
  const resumeCard = document.getElementById('v2-resume-card');
  const resumeTitle = document.getElementById('v2-resume-title');
  const resumeCopy = document.getElementById('v2-resume-copy');
  const resumeEyebrow = document.getElementById('v2-resume-eyebrow');
  const resumeBtn = document.getElementById('v2-resume-btn');
  const leaveBtn = document.getElementById('v2-leave-btn');
  const resumeReplay = document.getElementById('v2-resume-replay');
  const starterGrid = document.getElementById('v2-starter-grid');
  const starterCopy = document.getElementById('v2-starter-copy');
  const buildSummary = document.getElementById('v2-build-summary');
  const startBtn = document.getElementById('v2-start-btn');
  const tutorialStart = document.getElementById('v2-tutorial-start');
  const tutorialTitle = document.getElementById('v2-tutorial-level-title');
  const tutorialPrev = document.getElementById('v2-tutorial-prev');
  const tutorialNext = document.getElementById('v2-tutorial-next');
  const joinBtn = document.getElementById('v2-join-btn');
  const roomInput = document.getElementById('v2-room-code-input');
  const nameNode = document.getElementById('v2-home-name');
  const uidNode = document.getElementById('v2-home-uid');
  const soloDeck = document.getElementById('v2-solo-deck');
  const soloPicker = document.getElementById('v2-solo-picker');
  const aiDeck = document.getElementById('v2-ai-deck');
  const aiPicker = document.getElementById('v2-advanced-picker');
  const aiDeckHint = document.getElementById('v2-ai-deck-hint');
  let selectedMode = 'solo';
  let busy = false;
  let catalog = null;
  let roomState = null;

  function showError(error) {
    V2.showBanner(banner, error && error.message ? error.message : '无法读取主页数据。', 'error');
  }

  function setBusy(next) {
    busy = Boolean(next);
    startBtn.disabled = busy;
    joinBtn.disabled = busy;
    ['solo', 'advanced', 'pvp'].forEach(function (mode) {
      document.getElementById('v2-mode-' + mode).disabled = busy;
    });
    aiDeck.disabled = busy;
    soloDeck.disabled = busy;
    if (tutorialStart) tutorialStart.disabled = busy;
    if (leaveBtn) leaveBtn.disabled = busy;
    if (resumeBtn) resumeBtn.classList.toggle('is-disabled', busy);
  }

  function renderProfile() {
    nameNode.textContent = '已登录';
    uidNode.textContent = '进度已保存';
  }

  function renderStarter() {
    const byId = V2.indexById(V2.sortedCharacters(catalog && catalog.characters));
    const saved = catalog && catalog.saved_build;
    const starter = catalog && catalog.starter_deck;
    const deck = saved || starter || {};
    const ids = Array.isArray(deck.character_ids) && deck.character_ids.length
      ? deck.character_ids.slice(0, 4)
      : ['nanali', 'zero', 'jiuyuan', 'iloy'];
    starterGrid.innerHTML = ids.map(function (id) {
      const character = Object.assign({}, V2.characterFallback(id) || { id: id }, byId[id] || {});
      const portrait = V2.characterAsset(character, 'portrait') || V2.characterAsset(character, 'avatar');
      return (
        '<article class="v2-home-esper">' +
          '<div class="v2-home-esper-art">' + V2.imageMarkup(portrait, character.name, 'v2-portrait') + '</div>' +
          '<h3>' + V2.escapeHtml(character.name || id) + '</h3>' +
        '</article>'
      );
    }).join('');
    starterCopy.textContent = '每方 4 名角色、单人战斗区、玩家生命胜负。首次进入会获赠官方预组，可在构筑页更换或删除。';
  }

  function renderBuildSummary() {
    const saved = catalog && catalog.saved_build;
    const starter = catalog && catalog.starter_deck;
    const previousSolo = soloDeck.value;
    soloDeck.innerHTML = '<option value="random">随机</option><option value="mirror">镜像对局</option>' + ((catalog && catalog.starter_decks) || []).map(function (deck) {
      return '<option value="' + V2.escapeAttr(deck.id) + '">' + V2.escapeHtml(deck.name) + '</option>';
    }).join('');
    if (Array.from(soloDeck.options).some(function (option) { return option.value === previousSolo; })) soloDeck.value = previousSolo;
    if (selectedMode === 'advanced') {
      const opponentName = aiDeck.options[aiDeck.selectedIndex].text;
      const playerDeck = saved || starter;
      buildSummary.textContent = '我方「' + ((playerDeck && playerDeck.name) || '当前构筑') + '」· 对手「' + opponentName + '」';
      aiDeckHint.textContent = '使用当前保存的构筑对战，仅支持公开角色。五套高级人机均已启用新模型与搜索，属于尚未通过新规则完整强度验收的实验性策略。';
      return;
    }
    if (selectedMode === 'solo') {
      const playerDeck = saved || starter;
      const opponentName = soloDeck.options[soloDeck.selectedIndex].text;
      buildSummary.textContent = '我方「' + ((playerDeck && playerDeck.name) || '当前构筑') + '」· 对手「' + opponentName + '」';
      return;
    }
    if (saved && Array.isArray(saved.card_ids)) {
      buildSummary.textContent = '当前构筑「' + (saved.name || '自定义') + '」· ' + saved.card_ids.length + ' 张';
      return;
    }
    if (starter && Array.isArray(starter.card_ids)) {
      buildSummary.textContent = '将使用基础预组「' + (starter.name || '置顶预组') + '」';
      return;
    }
    buildSummary.textContent = '正在读取牌组。';
  }

  function renderResume() {
    const room = roomState && roomState.room;
    const game = roomState && roomState.game;
    if (!room) {
      V2.setHidden(resumeCard, true);
      return;
    }
    V2.setHidden(resumeCard, false);
    const finished = Boolean((game && (game.phase === 'finished' || game.winner)) || room.status === 'finished');
    resumeTitle.textContent = room.room_code ? ('房间 ' + room.room_code) : '已有房间';
    if (resumeEyebrow) resumeEyebrow.textContent = finished ? '已结束' : '进行中';
    const tutorialRoom = room && room.mode === 'tutorial';
    resumeCopy.textContent = tutorialRoom
      ? (finished ? '本关已结束。可离开后继续下一关。' : '可直接返回牌桌继续新手教学。')
      : V2.roomStatusLabel(room, game) + (finished ? '。可观看回放，或离开房间后开始新对局。' : '。可直接返回牌桌继续。');
    if (resumeBtn && tutorialRoom && !finished) {
      resumeBtn.textContent = '继续新手教学';
    }
    if (resumeBtn) {
      resumeBtn.textContent = finished ? '离开房间' : '返回牌桌';
      if (finished) {
        resumeBtn.removeAttribute('href');
        resumeBtn.setAttribute('role', 'button');
        resumeBtn.setAttribute('tabindex', '0');
      } else {
        resumeBtn.setAttribute('href', '/table');
        resumeBtn.removeAttribute('role');
        resumeBtn.removeAttribute('tabindex');
      }
    }
    V2.setHidden(leaveBtn, finished);
    const replayCode = room.replay_code || room.room_code;
    if (resumeReplay) {
      const canReplay = Boolean(room.has_replay && replayCode);
      V2.setHidden(resumeReplay, !canReplay);
      if (canReplay) resumeReplay.href = '/table?replay=' + encodeURIComponent(replayCode);
    }
  }

  function goTable() {
    window.location.href = '/table';
  }

  async function refreshState() {
    try {
      roomState = await V2.getState();
      renderResume();
    } catch (error) {
      roomState = null;
      renderResume();
      if (!(error && error.status === 404)) {
        showError(error);
      }
    }
  }

  async function startSelected() {
    if (busy) {
      return;
    }
    setBusy(true);
    V2.showBanner(banner, '', '');
    try {
      const payload = { mode: selectedMode };
      if (selectedMode === 'advanced') {
        payload.ai_deck = aiDeck.value;
      } else if (selectedMode === 'solo') {
        payload.ai_deck = soloDeck.value;
      }
      await V2.startGame(payload);
      goTable();
    } catch (error) {
      showError(error);
    } finally {
      setBusy(false);
    }
  }

  async function joinRoom() {
    if (busy) {
      return;
    }
    const code = String(roomInput.value || '').trim().toUpperCase();
    roomInput.value = code;
    if (!/^[0-9A-F]{6}$/.test(code)) {
      V2.showBanner(banner, '请输入 6 位房间码。', 'error');
      return;
    }
    setBusy(true);
    try {
      await V2.joinRoom(code);
      goTable();
    } catch (error) {
      showError(error);
    } finally {
      setBusy(false);
    }
  }

  function selectMode(mode) {
    if (busy) return;
    selectedMode = mode;
    ['solo', 'advanced', 'pvp'].forEach(function (value) {
      const button = document.getElementById('v2-mode-' + value);
      button.classList.toggle('selected', value === mode);
      button.setAttribute('aria-pressed', String(value === mode));
    });
    V2.setHidden(aiPicker, mode !== 'advanced');
    V2.setHidden(soloPicker, mode !== 'solo');
    startBtn.textContent = mode === 'advanced' ? '挑战高级人机' : (mode === 'pvp' ? '创建 1v1 房间' : '开始人机对局');
    renderStarter();
    renderBuildSummary();
  }
  ['solo', 'advanced', 'pvp'].forEach(function (mode) {
    document.getElementById('v2-mode-' + mode).addEventListener('click', function () { selectMode(mode); });
  });
  soloDeck.addEventListener('change', renderBuildSummary);
  aiDeck.addEventListener('change', function () { renderStarter(); renderBuildSummary(); });
  let selectedTutorialId = '';

  function selectableTutorialLevels() {
    const info = catalog && catalog.tutorial;
    const levels = (info && info.levels) || [];
    return levels.filter(function (item) { return item.available; });
  }

  function renderTutorialCard() {
    const info = catalog && catalog.tutorial;
    const available = selectableTutorialLevels();
    if (!available.length) {
      if (tutorialTitle) tutorialTitle.textContent = '第一关 牌桌与回合';
      if (tutorialStart) tutorialStart.disabled = true;
      if (tutorialPrev) tutorialPrev.disabled = true;
      if (tutorialNext) tutorialNext.disabled = true;
      return;
    }
    const preferred = selectedTutorialId
      || (info && info.current_level)
      || available[0].id;
    let index = available.findIndex(function (item) { return item.id === preferred; });
    if (index < 0) {
      index = available.length - 1;
    }
    const current = available[index];
    selectedTutorialId = current.id;
    if (tutorialTitle) tutorialTitle.textContent = current.title;
    if (tutorialStart) {
      tutorialStart.disabled = false;
      tutorialStart.textContent = '进入';
    }
    if (tutorialPrev) tutorialPrev.disabled = index <= 0;
    if (tutorialNext) tutorialNext.disabled = index >= available.length - 1;
  }

  function flipTutorial(delta) {
    const available = selectableTutorialLevels();
    const index = available.findIndex(function (item) { return item.id === selectedTutorialId; });
    const next = available[index + delta];
    if (!next) {
      return;
    }
    selectedTutorialId = next.id;
    renderTutorialCard();
  }

  async function startTutorial() {
    if (busy) return;
    setBusy(true);
    V2.showBanner(banner, '', '');
    try {
      const payload = { mode: 'tutorial' };
      if (selectedTutorialId) {
        payload.scenario = selectedTutorialId;
      }
      await V2.startGame(payload);
      goTable();
    } catch (error) {
      showError(error);
    } finally {
      setBusy(false);
    }
  }

  if (tutorialStart) tutorialStart.addEventListener('click', startTutorial);
  if (tutorialPrev) tutorialPrev.addEventListener('click', function () { flipTutorial(-1); });
  if (tutorialNext) tutorialNext.addEventListener('click', function () { flipTutorial(1); });
  startBtn.addEventListener('click', startSelected);
  joinBtn.addEventListener('click', joinRoom);
  roomInput.addEventListener('keydown', function (event) {
    if (event.key === 'Enter') {
      joinRoom();
    }
  });
  async function leaveCurrentRoom() {
    if (busy) {
      return;
    }
    setBusy(true);
    try {
      await V2.leaveRoom();
      roomState = null;
      renderResume();
      await refreshState();
      V2.showBanner(banner, '已离开房间。', 'ok');
    } catch (error) {
      showError(error);
      await refreshState();
    } finally {
      setBusy(false);
    }
  }

  if (resumeBtn) {
    resumeBtn.addEventListener('click', function (event) {
      const game = roomState && roomState.game;
      const room = roomState && roomState.room;
      const finished = Boolean((game && (game.phase === 'finished' || game.winner)) || (room && room.status === 'finished'));
      if (!finished) {
        return;
      }
      event.preventDefault();
      leaveCurrentRoom();
    });
  }
  leaveBtn.addEventListener('click', function () {
    leaveCurrentRoom();
  });

  renderProfile();
  try {
    catalog = await V2.getCatalog();
    renderStarter();
    renderBuildSummary();
    renderTutorialCard();
  } catch (error) {
    showError(error);
    starterCopy.textContent = '角色资料暂未到达，仍可使用固定预组入口。';
    renderStarter();
    renderBuildSummary();
  }
  await refreshState();
}
