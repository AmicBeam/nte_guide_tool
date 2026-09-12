const V2 = window.NTE_V2;

if (V2.ensureLogin()) {
  bootstrapHome();
}

async function bootstrapHome() {
  const banner = document.getElementById('v2-home-banner');
  const resumeCard = document.getElementById('v2-resume-card');
  const resumeTitle = document.getElementById('v2-resume-title');
  const resumeCopy = document.getElementById('v2-resume-copy');
  const leaveBtn = document.getElementById('v2-leave-btn');
  const starterGrid = document.getElementById('v2-starter-grid');
  const starterCopy = document.getElementById('v2-starter-copy');
  const buildSummary = document.getElementById('v2-build-summary');
  const startBtn = document.getElementById('v2-start-btn');
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
    leaveBtn.disabled = busy;
  }

  function renderProfile() {
    nameNode.textContent = '已登录';
    uidNode.textContent = '构筑与对局进度自动保存。';
  }

  function renderStarter() {
    const characters = V2.sortedCharacters(catalog && catalog.characters);
    const cards = catalog && catalog.cards || [];
    starterGrid.innerHTML = characters.map(function (character) {
      const owned = V2.cardsForCharacter(cards, character.id);
      const portrait = V2.characterAsset(character, 'portrait') || V2.characterAsset(character, 'avatar');
      return (
        '<article class="v2-starter-card" data-character-id="' + V2.escapeAttr(character.id) + '">' +
          '<div class="v2-starter-art">' + V2.imageMarkup(portrait, character.name, 'v2-portrait') + '</div>' +
          '<div>' +
            '<h3>' + V2.escapeHtml(character.name) + '</h3>' +
            V2.attributeMarkup(character.attribute) +
          '</div>' +
          '<p class="card-meta">' + V2.escapeHtml(character.role || character.passive || '固定预组成员') + '</p>' +
          '<p class="subtle">攻击 ' + V2.escapeHtml(character.attack) + ' · 生命 ' + V2.escapeHtml(character.max_hp) + ' · 候选 ' + owned.length + ' 种</p>' +
        '</article>'
      );
    }).join('');
    const starter = catalog && catalog.starter_deck;
    starterCopy.textContent = starter && starter.name
      ? '基础预组「' + starter.name + '」：' + characters.map(function (item) { return item.name; }).join('、') + '。'
      : '固定展示顺序为娜娜莉、主角「零」、九原、浔。';
  }

  function renderBuildSummary() {
    const saved = catalog && catalog.saved_build;
    const starter = catalog && catalog.starter_deck;
    if (saved && Array.isArray(saved.card_ids)) {
      buildSummary.textContent = '已保存构筑「' + (saved.name || '自定义') + '」，共 ' + saved.card_ids.length + ' 张。';
      return;
    }
    if (starter && Array.isArray(starter.card_ids)) {
      buildSummary.textContent = '尚未保存自定义构筑，将使用基础预组「' + (starter.name || '置顶预组') + '」。';
      return;
    }
    buildSummary.textContent = '正在读取牌组，请稍候。';
  }

  function renderResume() {
    const room = roomState && roomState.room;
    const game = roomState && roomState.game;
    if (!room) {
      V2.setHidden(resumeCard, true);
      return;
    }
    V2.setHidden(resumeCard, false);
    resumeTitle.textContent = room.room_code ? ('房间 ' + room.room_code) : '已有房间';
    resumeCopy.textContent = V2.roomStatusLabel(room, game) + (game && game.phase === 'finished' ? '。可查看结果，或开始新对局。' : '。可直接返回牌桌继续。');
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
  startBtn.addEventListener('click', startSelected);
  joinBtn.addEventListener('click', joinRoom);
  roomInput.addEventListener('keydown', function (event) {
    if (event.key === 'Enter') {
      joinRoom();
    }
  });
  leaveBtn.addEventListener('click', async function () {
    if (busy) {
      return;
    }
    setBusy(true);
    try {
      await V2.leaveRoom();
      roomState = null;
      renderResume();
      V2.showBanner(banner, '已离开房间。', 'ok');
    } catch (error) {
      showError(error);
    } finally {
      setBusy(false);
    }
  });

  renderProfile();
  try {
    catalog = await V2.getCatalog();
    renderStarter();
    renderBuildSummary();
  } catch (error) {
    showError(error);
    starterCopy.textContent = '角色资料暂未到达，仍可使用固定预组入口。';
    renderStarter();
    renderBuildSummary();
  }
  await refreshState();
}
