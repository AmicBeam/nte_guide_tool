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
  const targetLine = document.getElementById('v2-target-line');
  const pool = document.getElementById('v2-entity-pool');
  const nodes = {
    roundLabel: document.getElementById('v2-round-label'),
    phaseChip: document.getElementById('v2-phase-chip'),
    roomChip: document.getElementById('v2-room-chip'),
    connectionChip: document.getElementById('v2-connection-chip'),
    readyBtn: document.getElementById('v2-ready-btn'),
    roomStartBtn: document.getElementById('v2-room-start-btn'),
    endTurnBtn: document.getElementById('v2-end-turn-btn'),
    concedeBtn: document.getElementById('v2-concede-btn'),
    opponentHp: document.getElementById('v2-opponent-hp'),
    opponentShield: document.getElementById('v2-opponent-shield'),
    opponentAp: document.getElementById('v2-opponent-ap'),
    opponentName: document.getElementById('v2-opponent-name'),
    opponentLife: document.getElementById('v2-opponent-life'),
    opponentFront: document.getElementById('v2-opponent-front-slot'),
    opponentBench: document.getElementById('v2-opponent-bench'),
    opponentDeck: document.getElementById('v2-opponent-deck'),
    opponentDiscard: document.getElementById('v2-opponent-discard'),
    playerHp: document.getElementById('v2-player-hp'),
    playerShield: document.getElementById('v2-player-shield'),
    playerAp: document.getElementById('v2-player-ap'),
    playerName: document.getElementById('v2-player-name'),
    playerLife: document.getElementById('v2-player-life'),
    playerFront: document.getElementById('v2-player-front-slot'),
    playerBench: document.getElementById('v2-player-bench'),
    playerDeck: document.getElementById('v2-player-deck'),
    playerDiscard: document.getElementById('v2-player-discard'),
    turnCopy: document.getElementById('v2-turn-copy'),
    waitChip: document.getElementById('v2-wait-chip'),
    records: document.getElementById('v2-records'),
    playHistory: document.getElementById('v2-play-history'),
    playHistoryDetail: document.getElementById('v2-play-history-detail'),
    logList: document.getElementById('v2-log-list'),
    normalAttack: document.getElementById('v2-normal-attack'),
    handCount: document.getElementById('v2-hand-count'),
    mulliganHint: document.getElementById('v2-mulligan-hint'),
    mulliganBtn: document.getElementById('v2-mulligan-btn'),
    opponentHand: document.getElementById('v2-opponent-hand'),
    playerHand: document.getElementById('v2-player-hand'),
    roomTitle: document.getElementById('v2-room-title'),
    roomCopy: document.getElementById('v2-room-copy'),
    memberList: document.getElementById('v2-member-list'),
    playerFrontZone: document.getElementById('v2-player-front'),
    opponentFrontZone: document.getElementById('v2-opponent-front'),
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
    targetLine: targetLine,
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

  function setMenuOpen(open) {
    menuOverlay.classList.toggle('open', Boolean(open));
    menuOverlay.setAttribute('aria-hidden', open ? 'false' : 'true');
  }

  function setHistoryOpen(open) {
    historyDrawer.classList.toggle('open', Boolean(open));
    historyDrawer.setAttribute('aria-hidden', open ? 'false' : 'true');
  }

  function applyPlayerOptions() {
    TABLE.saveOptions(ctx.options);
    if (ctx.player && typeof ctx.player.setOptions === 'function') {
      ctx.player.setOptions(ctx.options);
    }
    document.getElementById('v2-mute-toggle').checked = Boolean(ctx.options.muted);
    document.getElementById('v2-motion-toggle').checked = Boolean(ctx.options.reducedMotion);
    document.getElementById('v2-speed-1').classList.toggle('primary-btn', ctx.options.speed !== 2);
    document.getElementById('v2-speed-2').classList.toggle('primary-btn', ctx.options.speed === 2);
  }

  function skipPresentation() {
    if (ctx.settlePresentation) ctx.settlePresentation();
    else {
      if (ctx.player && typeof ctx.player.cancel === 'function') ctx.player.cancel();
      ctx.playingEvents = false;
      page.classList.remove('presentation-busy');
      ctx.alignAuthoritative();
    }
  }

  document.getElementById('v2-menu-btn').addEventListener('click', function () { setMenuOpen(true); });
  document.getElementById('v2-menu-close').addEventListener('click', function () { setMenuOpen(false); });
  document.getElementById('v2-history-btn').addEventListener('click', function () { setHistoryOpen(true); });
  document.getElementById('v2-history-close').addEventListener('click', function () { setHistoryOpen(false); });
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
  document.getElementById('v2-skip-btn').addEventListener('click', skipPresentation);
  document.getElementById('v2-skip-inline-btn').addEventListener('click', skipPresentation);
  menuOverlay.addEventListener('click', function (event) {
    if (event.target === menuOverlay) setMenuOpen(false);
  });

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
  await ctx.refreshState({ force: true, initial: true });
  ctx.startPolling();
}
