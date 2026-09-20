const V2 = window.NTE_V2;
const TABLE = window.NTE_V2_TABLE;

if (V2.ensureLogin()) {
  bootstrapTable();
}

async function bootstrapTable() {
  const page = document.getElementById('v2-table-page');
  const banner = document.getElementById('v2-table-banner');
  const emptyState = document.getElementById('v2-empty-state');
  const emptyTitle = document.getElementById('v2-empty-title');
  const emptyCopy = document.getElementById('v2-empty-copy');
  const roomPanel = document.getElementById('v2-room-panel');
  const tableLayout = document.getElementById('v2-table-layout');
  const handDock = document.getElementById('v2-hand-dock');
  const toast = document.getElementById('v2-toast');
  const ghost = document.getElementById('v2-drag-ghost');
  const previewBox = document.getElementById('v2-drag-preview');
  const choiceOverlay = document.getElementById('v2-choice-overlay');
  const pileModal = document.getElementById('v2-pile-modal');
  const cardPreview = document.getElementById('v2-card-preview');
  const historyDrawer = document.getElementById('v2-history-drawer');
  const menuOverlay = document.getElementById('v2-menu-overlay');
  const castZone = document.getElementById('v2-cast-zone');
  const targetMask = document.getElementById('v2-target-mask');
  const targetCopy = document.getElementById('v2-target-copy');
  const targetCancel = document.getElementById('v2-target-cancel');
  const targetLine = document.getElementById('v2-target-line');
  const pool = document.getElementById('v2-entity-pool');
  const nodes = {
    roundLabel: document.getElementById('v2-round-label'),
    phaseChip: document.getElementById('v2-phase-chip'),
    roomChip: document.getElementById('v2-room-chip'),
    connectionChip: document.getElementById('v2-connection-chip'),
    replayLink: document.getElementById('v2-replay-link'),
    readyBtn: document.getElementById('v2-ready-btn'),
    roomStartBtn: document.getElementById('v2-room-start-btn'),
    endTurnBtn: document.getElementById('v2-end-turn-btn'),
    concedeBtn: document.getElementById('v2-concede-btn'),
    homeLink: document.getElementById('v2-home-link'),
    resultOverlay: document.getElementById('v2-result-overlay'),
    resultTitle: document.getElementById('v2-result-title'),
    resultCopy: document.getElementById('v2-result-copy'),
    resultClose: document.getElementById('v2-result-close'),
    resultStar: document.getElementById('v2-result-star'),
    resultLeave: document.getElementById('v2-result-leave'),
    opponentHp: document.getElementById('v2-opponent-hp'),
    opponentShield: document.getElementById('v2-opponent-shield'),
    opponentAp: document.getElementById('v2-opponent-ap'),
    opponentName: document.getElementById('v2-opponent-name'),
    opponentLife: document.getElementById('v2-opponent-life'),
    opponentFront: document.getElementById('v2-opponent-front-slot'),
    opponentBench: document.getElementById('v2-opponent-bench'),
    opponentDeck: document.getElementById('v2-opponent-deck'),
    opponentDiscard: document.getElementById('v2-opponent-discard'),
    opponentHandCount: document.getElementById('v2-opponent-hand-count'),
    playerHp: document.getElementById('v2-player-hp'),
    playerShield: document.getElementById('v2-player-shield'),
    playerAp: document.getElementById('v2-player-ap'),
    playerName: document.getElementById('v2-player-name'),
    playerLife: document.getElementById('v2-player-life'),
    playerFront: document.getElementById('v2-player-front-slot'),
    playerBench: document.getElementById('v2-player-bench'),
    playerDeck: document.getElementById('v2-player-deck'),
    playerDiscard: document.getElementById('v2-player-discard'),
    playerHandCount: document.getElementById('v2-player-hand-count'),
    turnCopy: document.getElementById('v2-turn-copy'),
    waitChip: document.getElementById('v2-wait-chip'),
    records: document.getElementById('v2-records'),
    playHistoryDetail: document.getElementById('v2-play-history-detail'),
    logList: document.getElementById('v2-log-list'),
    normalAttack: document.getElementById('v2-normal-attack'),
    ultimate: document.getElementById('v2-ultimate'),
    mulliganToggle: document.getElementById('v2-mulligan-toggle'),
    mulliganBtn: document.getElementById('v2-mulligan-btn'),
    mulliganStage: document.getElementById('v2-mulligan-stage'),
    mulliganRail: document.getElementById('v2-mulligan-rail'),
    mulliganStageCopy: document.getElementById('v2-mulligan-stage-copy'),
    opponentHand: document.getElementById('v2-opponent-hand'),
    playerHand: document.getElementById('v2-player-hand'),
    roomTitle: document.getElementById('v2-room-title'),
    roomCopy: document.getElementById('v2-room-copy'),
    memberList: document.getElementById('v2-member-list'),
    playerFrontZone: document.getElementById('v2-player-front'),
    opponentFrontZone: document.getElementById('v2-opponent-front'),
    playerFrontDebuff: document.getElementById('v2-player-front-debuff'),
    opponentFrontDebuff: document.getElementById('v2-opponent-front-debuff'),
    tutorialMask: document.getElementById('v2-tutorial-mask'),
    tutorialModal: document.getElementById('v2-tutorial-modal'),
    tutorialCue: document.getElementById('v2-tutorial-cue'),
    playArea: document.getElementById('v2-play-area'),
    targetMask: targetMask,
    targetCopy: targetCopy,
    targetCancel: targetCancel,
  };

  const options = TABLE.loadOptions();
  const ctx = {
    page: page,
    banner: banner,
    emptyState: emptyState,
    emptyTitle: emptyTitle,
    emptyCopy: emptyCopy,
    roomPanel: roomPanel,
    tableLayout: tableLayout,
    handDock: handDock,
    toast: toast,
    ghost: ghost,
    previewBox: previewBox,
    choiceOverlay: choiceOverlay,
    pileModal: pileModal,
    cardPreview: cardPreview,
    historyDrawer: historyDrawer,
    menuOverlay: menuOverlay,
    castZone: castZone,
    targetMask: targetMask,
    targetCopy: targetCopy,
    targetCancel: targetCancel,
    targetLine: targetLine,
    tutorialMask: document.getElementById('v2-tutorial-mask'),
    tutorialModal: document.getElementById('v2-tutorial-modal'),
    tutorialCue: document.getElementById('v2-tutorial-cue'),
    pool: pool,
    nodes: nodes,
    room: null,
    game: null,
    displayGame: null,
    lastVersion: -1,
    seenCursor: null,
    busy: false,
    playingEvents: false,
    pendingRequest: null,
    pendingEnvelope: null,
    playHistory: [],
    replayMode: false,
    playerGeneration: 0,
    dragState: null,
    localChoice: null,
    mulliganIds: [],
    raisedCardId: null,
    suppressClick: false,
    toastTimer: 0,
    pollTimer: 0,
    options: options,
  };

  function showToast(message) {
    if (!message) return;
    toast.textContent = message;
    toast.classList.add('open');
    toast.setAttribute('aria-hidden', 'false');
    window.clearTimeout(ctx.toastTimer);
    ctx.toastTimer = window.setTimeout(function () {
      toast.classList.remove('open');
      toast.setAttribute('aria-hidden', 'true');
    }, 2200);
  }
  ctx.showToast = showToast;

  TABLE.bindSync(ctx);
  TABLE.bindInteraction(ctx);
  if (typeof TABLE.bindTutorial === 'function') {
    TABLE.bindTutorial(ctx);
  }

  function setMenuOpen(open) {
    menuOverlay.classList.toggle('open', Boolean(open));
    menuOverlay.setAttribute('aria-hidden', open ? 'false' : 'true');
  }

  function setHistoryOpen(open) {
    historyDrawer.classList.toggle('open', Boolean(open));
    historyDrawer.setAttribute('aria-hidden', open ? 'false' : 'true');
  }

  const mobileLayout = window.matchMedia('(pointer: coarse) and (max-width: 1200px)');

  function applyPlayerOptions() {
    const fontSize = Number.isInteger(ctx.options.fontSize) && ctx.options.fontSize >= 0 && ctx.options.fontSize <= 4 ? ctx.options.fontSize : 2;
    ctx.options.fontSize = fontSize;
    const layoutScale = [0.8, 1, 1.5, 2].includes(ctx.options.layoutScale) ? ctx.options.layoutScale : 1;
    ctx.options.layoutScale = layoutScale;
    const baseScale = mobileLayout.matches ? 0.8 : 1;
    page.style.setProperty('--table-layout-scale', layoutScale * baseScale);
    page.style.setProperty('--table-font-scale', [0.8, 0.9, 1, 1.1, 1.2][fontSize] * layoutScale * baseScale);
    page.querySelectorAll('[data-layout-scale]').forEach(function (button) {
      button.setAttribute('aria-pressed', String(Number(button.dataset.layoutScale) === layoutScale));
    });
    page.querySelectorAll('[data-font-size]').forEach(function (button) {
      const selected = Number(button.dataset.fontSize) === fontSize;
      button.setAttribute('aria-pressed', String(selected));
      button.classList.toggle('primary-btn', selected);
    });
    TABLE.saveOptions(ctx.options);
    if (ctx.player && typeof ctx.player.setOptions === 'function') {
      ctx.player.setOptions(ctx.options);
    }
    const dragToggle = document.getElementById('v2-drag-targeting-toggle');
    dragToggle.checked = !mobileLayout.matches && ctx.options.dragTargeting !== false;
    dragToggle.disabled = mobileLayout.matches;
    document.getElementById('v2-drag-targeting-help').textContent = mobileLayout.matches
      ? '手机端左右滑动浏览手牌，向上拖到战斗区松开，再选择目标或效果。点击空白处或手牌数展开、收起手牌。'
      : '开启：有目标牌拖到高亮目标。无目标牌或关闭开关时，拖到战斗大区松开，需要目标时再点选。';
    document.getElementById('v2-mute-toggle').checked = Boolean(ctx.options.muted);
    document.getElementById('v2-motion-toggle').checked = Boolean(ctx.options.reducedMotion);
    document.getElementById('v2-speed-1').classList.toggle('primary-btn', ctx.options.speed !== 2);
    document.getElementById('v2-speed-2').classList.toggle('primary-btn', ctx.options.speed === 2);
  }

  mobileLayout.addEventListener('change', applyPlayerOptions);

  document.getElementById('v2-menu-btn').addEventListener('click', function () { setMenuOpen(true); });
  document.getElementById('v2-menu-close').addEventListener('click', function () { setMenuOpen(false); });
  document.getElementById('v2-history-btn').addEventListener('click', function () { setHistoryOpen(true); });
  document.getElementById('v2-history-close').addEventListener('click', function () { setHistoryOpen(false); });
  document.getElementById('v2-drag-targeting-toggle').addEventListener('change', function (event) {
    if (mobileLayout.matches) return;
    ctx.endBoardTargetSelect();
    ctx.options.dragTargeting = Boolean(event.target.checked);
    applyPlayerOptions();
  });
  page.querySelectorAll('[data-font-size]').forEach(function (button) {
    button.addEventListener('click', function () {
      ctx.options.fontSize = Number(button.dataset.fontSize);
      applyPlayerOptions();
    });
  });
  page.querySelectorAll('[data-layout-scale]').forEach(function (button) {
    button.addEventListener('click', function () {
      ctx.options.layoutScale = Number(button.dataset.layoutScale);
      applyPlayerOptions();
    });
  });
  document.getElementById('v2-mute-toggle').addEventListener('change', function (event) {
    ctx.options.muted = Boolean(event.target.checked);
    applyPlayerOptions();
  });
  document.getElementById('v2-motion-toggle').addEventListener('change', function (event) {
    ctx.options.reducedMotion = Boolean(event.target.checked);
    applyPlayerOptions();
  });
  document.getElementById('v2-speed-1').addEventListener('click', function () {
    ctx.options.speed = 1;
    applyPlayerOptions();
  });
  document.getElementById('v2-speed-2').addEventListener('click', function () {
    ctx.options.speed = 2;
    applyPlayerOptions();
  });
  menuOverlay.addEventListener('click', function (event) {
    if (event.target === menuOverlay) setMenuOpen(false);
  });
  function leaveToHome() {
    const href = (nodes.homeLink && nodes.homeLink.getAttribute('href')) || '/card-game';
    return V2.leaveRoom().catch(function () { return null; }).then(function () {
      window.location.href = href;
    });
  }

  function resultDialogKey() {
    const code = ctx.room && ctx.room.room_code;
    const version = ctx.game && ctx.game.version;
    return String(code || '') + ':' + String(version == null ? '' : version);
  }

  function gameHasEnded() {
    return Boolean(!ctx.replayMode && ctx.game && (ctx.game.phase === 'finished' || ctx.game.winner));
  }

  function resultCopy() {
    const reason = ctx.game && ctx.game.reason;
    if (reason === 'concede') {
      return ctx.game.winner === ctx.game.viewer_side ? '对方认输。' : '你已认输。';
    }
    if (reason === 'empty_draw_win') {
      return (ctx.game.winner === ctx.game.viewer_side ? '你' : '对方') + '在无牌可抽时触发卡牌效果，获得胜利。';
    }
    if (reason === 'empty_draw') {
      return '牌库已空，无法抽牌。';
    }
    return '';
  }

  function fillResultDialog() {
    if (nodes.resultTitle) {
      nodes.resultTitle.textContent = V2.roomStatusLabel(ctx.room, ctx.game) || '对局结束';
    }
    if (nodes.resultCopy) {
      nodes.resultCopy.textContent = resultCopy();
      V2.setHidden(nodes.resultCopy, !resultCopy());
    }
    const replayCode = ctx.room && (ctx.room.replay_code || ctx.room.room_code);
    const canStar = Boolean(ctx.room && ctx.room.has_replay && replayCode);
    if (nodes.resultStar) {
      nodes.resultStar.disabled = !canStar || ctx.busy;
      nodes.resultStar.textContent = ctx.room && ctx.room.favorited ? '已收藏' : '收藏录像';
    }
  }

  function setResultOpen(open) {
    if (!nodes.resultOverlay) return;
    nodes.resultOverlay.classList.toggle('open', Boolean(open));
    nodes.resultOverlay.setAttribute('aria-hidden', open ? 'false' : 'true');
    if (open) fillResultDialog();
  }

  function hasUnplayedPresentation() {
    if (ctx.playingEvents) {
      return true;
    }
    if (ctx.seenCursor == null) {
      return false;
    }
    const game = ctx.game;
    if (!game || typeof TABLE.eventsAfter !== 'function') {
      return false;
    }
    const presentation = TABLE.presentationFrom({ game: game }, game);
    const fresh = TABLE.eventsAfter(presentation.events, ctx.seenCursor);
    return Boolean(fresh.length);
  }

  function syncResultDialog() {
    if (!gameHasEnded()) {
      setResultOpen(false);
      return;
    }
    if (hasUnplayedPresentation()) {
      return;
    }
    if (ctx.room && ctx.room.mode === 'tutorial') {
      setResultOpen(false);
      advanceTutorialCampaign();
      return;
    }
    if (ctx.resultDialogDismissed === resultDialogKey()) {
      return;
    }
    setResultOpen(true);
  }

  function advanceTutorialCampaign() {
    if (ctx.tutorialAdvancing) {
      return;
    }
    ctx.tutorialAdvancing = true;
    const next = ctx.game && ctx.game.tutorial && ctx.game.tutorial.next_scenario;
    const home = (nodes.homeLink && nodes.homeLink.getAttribute('href')) || '/card-game';
    V2.leaveRoom().catch(function () { return null; }).then(function () {
      if (!next) {
        window.location.href = home;
        return null;
      }
      return V2.startGame({ mode: 'tutorial', scenario: next });
    }).then(function (payload) {
      if (payload) {
        window.location.href = '/table';
      }
    }).catch(function () {
      window.location.href = home;
    });
  }
  ctx.syncResultDialog = syncResultDialog;

  function openResultFromEvent() {
    if (ctx.replayMode) {
      return;
    }
    if (ctx.resultDialogDismissed === resultDialogKey()) {
      return;
    }
    setResultOpen(true);
  }
  ctx.openResultFromEvent = openResultFromEvent;

  if (nodes.homeLink) {
    nodes.homeLink.addEventListener('click', function (event) {
      event.preventDefault();
      leaveToHome();
    });
  }
  if (nodes.resultClose) {
    nodes.resultClose.addEventListener('click', function () {
      ctx.resultDialogDismissed = resultDialogKey();
      setResultOpen(false);
    });
  }
  if (nodes.resultOverlay) {
    nodes.resultOverlay.addEventListener('click', function (event) {
      if (event.target !== nodes.resultOverlay) return;
      ctx.resultDialogDismissed = resultDialogKey();
      setResultOpen(false);
    });
  }
  if (nodes.resultLeave) {
    nodes.resultLeave.addEventListener('click', function () {
      if (ctx.busy) return;
      ctx.busy = true;
      nodes.resultLeave.disabled = true;
      leaveToHome().finally(function () {
        ctx.busy = false;
        nodes.resultLeave.disabled = false;
      });
    });
  }
  if (nodes.resultStar) {
    nodes.resultStar.addEventListener('click', async function () {
      const replayCode = ctx.room && (ctx.room.replay_code || ctx.room.room_code);
      if (!replayCode || ctx.busy) return;
      const next = !(ctx.room && ctx.room.favorited);
      ctx.busy = true;
      nodes.resultStar.disabled = true;
      try {
        await V2.starReplay(replayCode, next);
        if (ctx.room) ctx.room.favorited = next;
        if (ctx.room) ctx.room.has_replay = true;
        fillResultDialog();
      } catch (error) {
        V2.showBanner(banner, error.message || '无法收藏录像。', 'error');
      } finally {
        ctx.busy = false;
        fillResultDialog();
      }
    });
  }

  ctx.nodes.readyBtn.addEventListener('click', async function () {
    if (ctx.busy) return;
    ctx.busy = true;
    try {
      const selfMember = (ctx.room && ctx.room.members || []).find(function (member) { return member.is_self; });
      const payload = await V2.setReady(!(selfMember && selfMember.is_ready));
      await ctx.applyEnvelope(payload, { force: true });
    } catch (error) {
      V2.showBanner(banner, error.message || '准备失败。', 'error');
    } finally {
      ctx.busy = false;
      ctx.renderRoomChrome();
    }
  });
  ctx.nodes.roomStartBtn.addEventListener('click', async function () {
    if (ctx.busy) return;
    ctx.busy = true;
    try {
      const payload = await V2.roomStart();
      await ctx.applyEnvelope(payload, { force: true, initial: true });
    } catch (error) {
      V2.showBanner(banner, error.message || '开局失败。', 'error');
    } finally {
      ctx.busy = false;
      ctx.renderRoomChrome();
      if (ctx.game) ctx.alignAuthoritative();
    }
  });

  applyPlayerOptions();
  // Reference previews need card definitions even when those cards are not in hand.
  await V2.getCatalog().catch(function () {});
  const replayCode = TABLE.replayQueryCode();
  if (replayCode) {
    await TABLE.startReplay(ctx, replayCode);
    return;
  }
  await ctx.refreshState({ force: true, initial: true });
  ctx.startPolling();
}
