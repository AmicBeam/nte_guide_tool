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
  const tutorialSkip = document.getElementById('v2-tutorial-skip');
  const tutorialCopy = document.getElementById('v2-tutorial-copy');
  const joinBtn = document.getElementById('v2-join-btn');
  const roomInput = document.getElementById('v2-room-code-input');
  const nameNode = document.getElementById('v2-home-name');
  const uidNode = document.getElementById('v2-home-uid');
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
      await V2.startGame({ mode: selectedMode });
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

  document.getElementById('v2-mode-solo').addEventListener('click', function () {
    selectedMode = 'solo';
    document.getElementById('v2-mode-solo').classList.add('selected');
    document.getElementById('v2-mode-pvp').classList.remove('selected');
    startBtn.textContent = '开始人机对局';
  });
  document.getElementById('v2-mode-pvp').addEventListener('click', function () {
    selectedMode = 'pvp';
    document.getElementById('v2-mode-pvp').classList.add('selected');
    document.getElementById('v2-mode-solo').classList.remove('selected');
    startBtn.textContent = '创建 1v1 房间';
  });
  function renderTutorialCard() {
    const info = catalog && catalog.tutorial;
    if (!tutorialCopy || !tutorialStart) {
      return;
    }
    if (!info || info.completed || info.skipped) {
      tutorialCopy.textContent = info && info.completed ? '可以再看一遍第一关。' : '逐步学习出击与回合。';
      tutorialStart.textContent = info && info.completed ? '再看一遍' : '新手教学';
      if (tutorialSkip) tutorialSkip.hidden = Boolean(info && (info.completed || info.skipped));
      return;
    }
    tutorialCopy.textContent = info.completed_levels && info.completed_levels.length
      ? ('继续第 ' + (info.completed_levels.length + 1) + ' 关')
      : '逐步学习出击与回合。';
    tutorialStart.textContent = info.completed_levels && info.completed_levels.length ? '继续新手教学' : '新手教学';
  }

  async function startTutorial() {
    if (busy) return;
    setBusy(true);
    V2.showBanner(banner, '', '');
    try {
      await V2.startGame({ mode: 'tutorial' });
      goTable();
    } catch (error) {
      showError(error);
    } finally {
      setBusy(false);
    }
  }

  async function skipTutorial() {
    if (busy) return;
    setBusy(true);
    try {
      catalog = catalog || {};
      catalog.tutorial = await V2.skipTutorial({});
      renderTutorialCard();
      V2.showBanner(banner, '已跳过新手教学。', 'ok');
    } catch (error) {
      showError(error);
    } finally {
      setBusy(false);
    }
  }

  if (tutorialStart) tutorialStart.addEventListener('click', startTutorial);
  if (tutorialSkip) tutorialSkip.addEventListener('click', skipTutorial);
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
