(function (root) {
  const V2 = root.NTE_V2;
  const TABLE = root.NTE_V2_TABLE = root.NTE_V2_TABLE || {};
  function replayQueryCode() {
    try {
      return new URLSearchParams(window.location.search).get('replay') || '';
    } catch (error) {
      return '';
    }
  }

  function gameAt(opening, events, index) {
    let game = TABLE.clone(opening) || {};
    const applied = [];
    const limit = Math.max(0, Math.min(index, (events || []).length));
    for (let i = 0; i < limit; i += 1) {
      const event = events[i];
      game = TABLE.applyPatch(game, (event && event.patch) || {}, event);
      applied.push(event);
    }
    game.legal_actions = [];
    game.is_my_turn = false;
    game.events = TABLE.clone(applied);
    game.logs = applied.map(function (event) { return (event && event.text) || ''; });
    const cursor = applied.length ? applied[applied.length - 1].seq : 0;
    game.presentation = {
      schema_version: 1,
      cursor: cursor,
      oldest_seq: applied.length ? applied[0].seq : 0,
      events: TABLE.clone(applied),
    };
    return game;
  }

  function stepLabel(events, index) {
    if (!index) {
      return '开局';
    }
    const event = events[index - 1] || {};
    if (event.type === 'discard') return '';
    const seq = event.seq != null ? ('#' + event.seq + ' ') : '';
    return seq + (event.text || event.type || '公开事件');
  }

  function startReplay(ctx, roomCode) {
    const bar = document.getElementById('v2-replay-bar');
    const prevBtn = document.getElementById('v2-replay-prev');
    const nextBtn = document.getElementById('v2-replay-next');
    const playBtn = document.getElementById('v2-replay-play');
    const pauseBtn = document.getElementById('v2-replay-pause');
    const stepNode = document.getElementById('v2-replay-step');
    const timeNode = document.getElementById('v2-replay-time');
    const exitLink = document.getElementById('v2-replay-exit');
    const replayLink = document.getElementById('v2-replay-link');

    ctx.replayMode = true;
    ctx.page.classList.add('replay-mode');
    V2.setHidden(bar, false);
    V2.setHidden(replayLink, true);
    if (ctx.castZone) V2.setHidden(ctx.castZone, true);

    function replayViewer() {
      if (typeof ctx.viewer === 'function') {
        return ctx.viewer();
      }
      const game = ctx.displayGame || ctx.game;
      return (game && game.viewer_side) || ctx.viewer || 'a';
    }

    function replayOpponent() {
      if (typeof ctx.opponent === 'function') {
        return ctx.opponent();
      }
      return replayViewer() === 'b' ? 'a' : 'b';
    }

    function replayRenderCtx() {
      return {
        game: ctx.displayGame,
        room: ctx.room,
        viewer: replayViewer(),
        opponent: replayOpponent(),
        nodes: ctx.nodes,
        pool: ctx.pool,
        busy: true,
        mulliganIds: ctx.mulliganIds || [],
        raisedCardId: null,
        playHistory: ctx.playHistory,
        page: ctx.page,
        showMulligan: Boolean(ctx.replayShowMulligan),
        findAction: function () { return null; },
        playEntries: function () { return []; },
      };
    }

    function rebuildHistory(index) {
      ctx.playHistory = [];
    }

    function paintChrome() {
      const events = ctx.replayEvents || [];
      const max = events.length;
      ctx.index = Math.max(0, Math.min(ctx.index || 0, max));
      ctx.nodes.roomChip.textContent = ctx.room && ctx.room.room_code ? ('回放 ' + ctx.room.room_code) : '回放';
      ctx.nodes.phaseChip.textContent = '对局回放';
      ctx.nodes.connectionChip.textContent = ctx.playing ? '播放中' : '已暂停';
      ctx.nodes.connectionChip.dataset.state = ctx.playing ? 'busy' : 'live';
      ctx.nodes.endTurnBtn.disabled = true;
      ctx.nodes.concedeBtn.disabled = true;
      V2.setHidden(ctx.nodes.mulliganBtn, true);
      const inMulligan = Boolean(ctx.game && ctx.game.phase === 'mulligan');
      if (ctx.nodes.mulliganStageCopy) {
        ctx.nodes.mulliganStageCopy.textContent = inMulligan ? '回放起手换牌。' : '最多换 3 张。点选不要的牌，再确认。';
      }
      const current = ctx.game && ctx.game.events ? ctx.game.events[ctx.game.events.length - 1] : null;
      ctx.nodes.turnCopy.textContent = current && current.text ? current.text : '起手公开局面';
      if (ctx.nodes.waitChip) ctx.nodes.waitChip.textContent = ctx.playing ? '回放播放中' : '回放已暂停';
      stepNode.textContent = stepLabel(events, ctx.index) + ' · ' + ctx.index + '/' + max;
      prevBtn.disabled = ctx.index <= 0 || ctx.playing;
      nextBtn.disabled = ctx.index >= max || ctx.playing;
      V2.setHidden(playBtn, ctx.playing);
      V2.setHidden(pauseBtn, !ctx.playing);
    }

    function paint() {
      const events = ctx.replayEvents || [];
      ctx.index = Math.max(0, Math.min(ctx.index || 0, events.length));
      ctx.game = gameAt(ctx.replayOpening, events, ctx.index);
      ctx.displayGame = TABLE.clone(ctx.game);
      ctx.replayShowMulligan = Boolean(ctx.index === 0 && ctx.game && ctx.game.phase === 'mulligan');
      ctx.mulliganIds = [];
      rebuildHistory(ctx.index);
      V2.setHidden(ctx.emptyState, true);
      V2.setHidden(ctx.roomPanel, true);
      V2.setHidden(ctx.tableLayout, false);
      V2.setHidden(ctx.handDock, false);
      TABLE.renderGame(replayRenderCtx());
      paintChrome();
    }

    function stopPlay() {
      ctx.playing = false;
      ctx.replayPlayToken = (ctx.replayPlayToken || 0) + 1;
      if (ctx.player && typeof ctx.player.cancel === 'function') {
        ctx.player.cancel();
      }
    }

    async function presentNext() {
      const events = ctx.replayEvents || [];
      if (ctx.index >= events.length) {
        return false;
      }
      const event = events[ctx.index];
      ctx.index += 1;
      if (ctx.player && typeof ctx.player.play === 'function') {
        await ctx.player.play([event]);
      } else {
        ctx.displayGame = TABLE.applyPatch(ctx.displayGame || ctx.replayOpening, (event && event.patch) || {}, event);
        TABLE.renderGame(replayRenderCtx());
      }
      ctx.game = gameAt(ctx.replayOpening, events, ctx.index);
      paintChrome();
      return ctx.index < events.length;
    }

    function startPlay() {
      const events = ctx.replayEvents || [];
      if (ctx.index >= events.length) {
        ctx.index = 0;
        paint();
      }
      ctx.playing = true;
      ctx.replayPlayToken = (ctx.replayPlayToken || 0) + 1;
      const token = ctx.replayPlayToken;
      if (ctx.player && typeof ctx.player.setOptions === 'function') {
        ctx.player.setOptions(Object.assign({}, ctx.options || {}, { presentOwnPlays: true }));
      }
      paintChrome();
      (async function loop() {
        while (ctx.playing && token === ctx.replayPlayToken) {
          const more = await presentNext();
          if (!more) {
            break;
          }
        }
        if (token === ctx.replayPlayToken) {
          ctx.playing = false;
          paintChrome();
        }
      })();
    }

    prevBtn.addEventListener('click', function () {
      stopPlay();
      ctx.index = Math.max(0, (ctx.index || 0) - 1);
      paint();
    });
    nextBtn.addEventListener('click', function () {
      if (ctx.playing) return;
      presentNext();
    });
    playBtn.addEventListener('click', startPlay);
    pauseBtn.addEventListener('click', function () {
      stopPlay();
      paint();
    });

    return V2.getReplay(roomCode).then(function (payload) {
      ctx.room = payload.room || { room_code: roomCode, status: 'finished' };
      ctx.replayOpening = payload.replay && payload.replay.opening_board;
      ctx.replayEvents = (payload.replay && payload.replay.events) || [];
      ctx.index = 0;
      if (typeof ctx.bindPlayer === 'function') {
        ctx.bindPlayer();
      }
      if (ctx.player && typeof ctx.player.setOptions === 'function') {
        ctx.player.setOptions(Object.assign({}, ctx.options || {}, { presentOwnPlays: true }));
      }
      if (timeNode) {
        const when = V2.formatReplayTime(payload.replay && payload.replay.updated_at);
        timeNode.textContent = when;
        V2.setHidden(timeNode, !when);
      }
      if (exitLink) {
        exitLink.href = '/table';
        exitLink.textContent = ctx.room.status === 'playing' ? '返回牌桌' : '返回主页';
        if (ctx.room.status !== 'playing') exitLink.href = '/card-game';
      }
      paint();
      if (ctx.showToast) ctx.showToast('回放从起手换牌开始，双方出牌都会演出。');
      startPlay();
    }).catch(function (error) {
      V2.setHidden(bar, true);
      ctx.renderEmpty('无法打开回放', (error && error.message) || '未找到这场对局回放。', 'error');
    });
  }

  TABLE.replayQueryCode = replayQueryCode;
  TABLE.gameAt = gameAt;
  TABLE.startReplay = startReplay;
}(window));
