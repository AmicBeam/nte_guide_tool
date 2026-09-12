(function (root) {
  const V2 = root.NTE_V2;
  const TABLE = root.NTE_V2_TABLE = root.NTE_V2_TABLE || {};
  const POLL_MS = 1600;

  function bindSync(ctx) {
    function findAction(type, extra) {
      extra = extra || {};
      return V2.findLegalEntry(ctx.game, function (action) {
        if (action.type !== type) return false;
        return Object.keys(extra).every(function (key) {
          if (key === 'card_ids') return V2.sameIdSet(action.card_ids, extra.card_ids);
          return String(action[key] || '') === String(extra[key] || '');
        });
      });
    }

    function actionsOf(type, matcher) {
      return V2.findLegalEntries(ctx.game, function (action, entry) {
        if (action.type !== type) return false;
        return matcher ? matcher(action, entry) : true;
      });
    }

    ctx.findAction = findAction;
    ctx.actionsOf = actionsOf;
    ctx.playEntries = function (cardId) {
      return actionsOf('play_card', function (action) {
        return String(action.card_id) === String(cardId);
      });
    };
    ctx.viewer = function () { return TABLE.viewerSide(ctx.game); };
    ctx.opponent = function () { return TABLE.opponentSide(ctx.game); };
    ctx.playHistory = ctx.playHistory || [];
    ctx.playerGeneration = ctx.playerGeneration || 0;

    function setConnection(state, label) {
      const chip = ctx.nodes.connectionChip;
      if (!chip) return;
      chip.dataset.state = state || 'live';
      chip.textContent = label || '已同步';
    }

    function interactionLocked() {
      return Boolean(ctx.busy || ctx.dragState || ctx.localChoice || ctx.choiceOverlay.classList.contains('open') || ctx.playingEvents);
    }
    ctx.interactionLocked = interactionLocked;

    function renderCtx() {
      const locked = ctx.busy || ctx.playingEvents;
      return {
        game: ctx.displayGame || ctx.game,
        room: ctx.room,
        viewer: ctx.viewer(),
        opponent: ctx.opponent(),
        nodes: ctx.nodes,
        pool: ctx.pool,
        busy: locked,
        mulliganIds: ctx.mulliganIds,
        raisedCardId: ctx.raisedCardId,
        playHistory: ctx.playHistory,
        findAction: locked ? function () { return null; } : findAction,
        playEntries: locked ? function () { return []; } : ctx.playEntries,
      };
    }

    function bindPlayer() {
      ctx.playerGeneration += 1;
      const generation = ctx.playerGeneration;
      ctx.player = TABLE.createPlayer({
        root: ctx.page,
        applyPatch: function (patch, event) {
          if (generation !== ctx.playerGeneration) return;
          ctx.applyDisplayPatch(patch, event);
        },
        onBusy: function (busy) {
          if (generation !== ctx.playerGeneration) return;
          ctx.playingEvents = Boolean(busy);
          ctx.page.classList.toggle('presentation-busy', Boolean(busy));
          if (ctx.nodes.waitChip) ctx.nodes.waitChip.textContent = busy ? '演出中' : (ctx.game && ctx.game.is_my_turn ? '你的回合' : '对手回合');
          if (ctx.nodes.connectionChip) {
            ctx.nodes.connectionChip.dataset.state = busy ? 'busy' : 'live';
            ctx.nodes.connectionChip.textContent = busy ? '演出中' : '已同步';
          }
        },
        onPublicCard: function (card, event) {
          if (generation !== ctx.playerGeneration) return;
          ctx.playHistory = TABLE.appendPublicPlay(ctx.playHistory, event || { type: 'play', card: card });
          TABLE.renderPlayHistory(renderCtx());
        },
      });
      if (ctx.player && typeof ctx.player.setOptions === 'function') {
        ctx.player.setOptions(ctx.options);
      }
    }
    ctx.bindPlayer = bindPlayer;
    if (!ctx.player) bindPlayer();

    function resetRoomPresentation() {
      ctx.seenCursor = null;
      ctx.displayGame = null;
      ctx.playHistory = [];
      ctx.pendingEnvelope = null;
      ctx.mulliganIds = [];
      ctx.raisedCardId = null;
      ctx.playingEvents = false;
      ctx.lastVersion = -1;
      ctx.page.classList.remove('presentation-busy');
      if (ctx.player && typeof ctx.player.cancel === 'function') {
        ctx.player.cancel();
      }
      bindPlayer();
    }
    ctx.resetRoomPresentation = resetRoomPresentation;

    function renderMembers() {
      const members = Array.isArray(ctx.room && ctx.room.members) ? ctx.room.members : [];
      if (!members.length) {
        ctx.nodes.memberList.innerHTML = '<p class="subtle">等待成员信息。</p>';
        return;
      }
      ctx.nodes.memberList.innerHTML = members.map(function (member) {
        const name = member.name || member.nickname || member.player_uid || '玩家';
        const ready = member.is_ready || member.ready;
        const host = member.is_host || member.host;
        return '<div class="v2-chip">' + V2.escapeHtml(name) + (host ? ' · 房主' : '') + (ready ? ' · 已准备' : ' · 未准备') + '</div>';
      }).join('');
    }

    function renderRoomChrome() {
      const statusLabel = V2.roomStatusLabel(ctx.room, ctx.game);
      ctx.nodes.roomChip.textContent = (ctx.room && ctx.room.room_code) ? ('房间 ' + ctx.room.room_code) : '无房间';
      ctx.nodes.phaseChip.textContent = statusLabel;
      const lobby = Boolean(ctx.room) && !ctx.game && ctx.room.mode === 'pvp' && (ctx.room.status === 'waiting' || ctx.room.status === 'ready');
      const canReady = lobby;
      const canStart = Boolean(lobby && ctx.room.can_start);
      V2.setHidden(ctx.nodes.readyBtn, !canReady);
      V2.setHidden(ctx.nodes.roomStartBtn, !canStart);
      ctx.nodes.readyBtn.disabled = ctx.busy || !canReady;
      ctx.nodes.roomStartBtn.disabled = ctx.busy || !canStart;
      const selfMember = ((ctx.room && ctx.room.members) || []).find(function (member) { return member.is_self; });
      ctx.nodes.readyBtn.textContent = (selfMember && selfMember.is_ready) ? '取消准备' : '准备';
      if (lobby || (ctx.room && !ctx.game)) {
        V2.setHidden(ctx.roomPanel, false);
        ctx.nodes.roomTitle.textContent = ctx.room.room_code ? ('房间 ' + ctx.room.room_code) : '等待对手';
        ctx.nodes.roomCopy.textContent = (ctx.room.mode === 'solo' ? '人机对战' : '1v1 对战') + ' · ' + statusLabel + '。双方准备后，由房主开始。';
        renderMembers();
      } else {
        V2.setHidden(ctx.roomPanel, true);
      }
    }

    function renderEmpty(title, copy, kind) {
      ctx.emptyTitle.textContent = title;
      ctx.emptyCopy.textContent = copy;
      V2.setHidden(ctx.emptyState, false);
      V2.setHidden(ctx.tableLayout, true);
      V2.setHidden(ctx.handDock, true);
      V2.setHidden(ctx.roomPanel, true);
      ctx.nodes.endTurnBtn.disabled = true;
      ctx.nodes.concedeBtn.disabled = true;
      V2.setHidden(ctx.nodes.readyBtn, true);
      V2.setHidden(ctx.nodes.roomStartBtn, true);
      if (kind) V2.showBanner(ctx.banner, copy, kind);
    }

    function renderMulliganChrome() {
      const inMulligan = Boolean(ctx.game && ctx.game.phase === 'mulligan');
      const canMulligan = inMulligan && actionsOf('mulligan').length > 0;
      V2.setHidden(ctx.nodes.mulliganBtn, !canMulligan);
      ctx.nodes.mulliganHint.textContent = canMulligan
        ? ('已选 ' + ctx.mulliganIds.length + ' / 3，点选手牌后确认。')
        : (inMulligan ? '已完成换牌，等待对手。' : '行动力支付出牌；能量只用于主动觉醒。');
      ctx.nodes.mulliganBtn.disabled = ctx.busy || !canMulligan;
    }

    function renderChoiceFromServer() {
      const pending = ctx.game && ctx.game.pending_choice;
      if (ctx.localChoice || ctx.dragState) return;
      if (!pending || pending.side !== ctx.viewer()) {
        if (!ctx.localChoice) ctx.closeOverlay(ctx.choiceOverlay);
        return;
      }
      ctx.openChoice({
        title: pending.prompt || '请选择',
        choices: pending.choices || [],
        onPick: function (choice) {
          const entry = findAction('choose', { choice_id: choice.id }) || {
            action: { type: 'choose', choice_id: choice.id },
            label: choice.label,
          };
          ctx.submitEntry(entry);
        },
      });
    }

    function renderAuthoritativeControls() {
      const endTurn = findAction('end_turn');
      const concede = findAction('concede');
      ctx.nodes.endTurnBtn.disabled = ctx.busy || ctx.playingEvents || !endTurn;
      ctx.nodes.concedeBtn.disabled = ctx.busy || !concede;
      renderMulliganChrome();
      if (!ctx.playingEvents) renderChoiceFromServer();
    }

    function paintBoard() {
      V2.setHidden(ctx.emptyState, true);
      V2.setHidden(ctx.tableLayout, false);
      V2.setHidden(ctx.handDock, false);
      TABLE.renderGame(renderCtx());
      renderAuthoritativeControls();
    }

    function applyDisplayPatch(patch, event) {
      ctx.displayGame = TABLE.applyPatch(ctx.displayGame || ctx.game, patch || {}, event);
      if (event) ctx.playHistory = TABLE.appendPublicPlay(ctx.playHistory, event);
      TABLE.renderGame(renderCtx());
    }

    function alignAuthoritative() {
      ctx.displayGame = TABLE.clone(ctx.game);
      if (ctx.game) paintBoard();
      else renderRoomChrome();
    }

    function consumePresentation(payload, options) {
      options = options || {};
      const generation = ctx.playerGeneration;
      const presentation = TABLE.presentationFrom(payload, payload && payload.game);
      const nextCursor = Number(presentation.cursor);
      if (!ctx.game) {
        ctx.seenCursor = Number.isFinite(nextCursor) ? nextCursor : ctx.seenCursor;
        return Promise.resolve();
      }
      if (options.initial || ctx.seenCursor == null) {
        ctx.seenCursor = Number.isFinite(nextCursor) ? nextCursor : ctx.seenCursor;
        ctx.displayGame = TABLE.clone(ctx.game);
        return Promise.resolve();
      }
      if (Number.isFinite(presentation.oldest_seq) && ctx.seenCursor != null && ctx.seenCursor + 1 < presentation.oldest_seq) {
        ctx.seenCursor = nextCursor;
        ctx.displayGame = TABLE.clone(ctx.game);
        return Promise.resolve();
      }
      const fresh = TABLE.eventsAfter(presentation.events, ctx.seenCursor);
      if (!fresh.length) {
        ctx.seenCursor = Number.isFinite(nextCursor) ? Math.max(ctx.seenCursor, nextCursor) : ctx.seenCursor;
        return Promise.resolve();
      }
      ctx.playingEvents = true;
      ctx.page.classList.add('presentation-busy');
      setConnection('busy', '演出中');
      return ctx.player.play(fresh, { instant: Boolean(options.instant) }).then(function () {
        if (generation !== ctx.playerGeneration) return;
        ctx.seenCursor = Number.isFinite(nextCursor) ? nextCursor : ctx.seenCursor;
      }).catch(function () {
        if (generation !== ctx.playerGeneration) return;
        ctx.seenCursor = Number.isFinite(nextCursor) ? nextCursor : ctx.seenCursor;
      }).finally(function () {
        if (generation !== ctx.playerGeneration) return;
        ctx.playingEvents = false;
        ctx.page.classList.remove('presentation-busy');
        setConnection('live', '已同步');
        alignAuthoritative();
      });
    }

    function applyEnvelope(payload, options) {
      options = options || {};
      if (!payload) return Promise.resolve();
      const nextGame = payload.game || null;
      const nextVersion = Number(nextGame && nextGame.version);
      const incomingCode = payload.room && payload.room.room_code;
      const currentCode = ctx.room && ctx.room.room_code;
      const sameRoom = Boolean(currentCode && incomingCode && currentCode === incomingCode);
      const roomChanged = Boolean(incomingCode && currentCode && incomingCode !== currentCode);
      if (roomChanged) {
        ctx.game = null;
        resetRoomPresentation();
      }
      if (sameRoom && ctx.game && (!nextGame || nextVersion < Number(ctx.game.version))) {
        return Promise.resolve();
      }
      if (sameRoom && ctx.game && nextVersion === Number(ctx.game.version) && options.ignoreIfBusy) {
        ctx.room = payload.room || ctx.room;
        renderRoomChrome();
        return Promise.resolve();
      }
      if (options.ignoreIfBusy && interactionLocked() && !options.force) {
        if (ctx.pendingEnvelope && ctx.pendingEnvelope.game && nextGame &&
            Number(ctx.pendingEnvelope.game.version) > nextVersion) {
          return Promise.resolve();
        }
        ctx.pendingEnvelope = payload;
        ctx.room = payload.room || ctx.room;
        renderRoomChrome();
        return Promise.resolve();
      }
      ctx.pendingEnvelope = null;
      ctx.room = payload.room || ctx.room;
      const firstGame = roomChanged || (!ctx.game && Boolean(nextGame));
      ctx.game = nextGame;
      if (!Number.isNaN(nextVersion)) ctx.lastVersion = nextVersion;
      renderRoomChrome();
      if (!ctx.room && !ctx.game) {
        renderEmpty('没有进行中的房间', '可以从主页开始人机、创建房间或输入房间码加入。');
        return Promise.resolve();
      }
      if (!ctx.game) {
        V2.setHidden(ctx.emptyState, true);
        V2.setHidden(ctx.tableLayout, true);
        V2.setHidden(ctx.handDock, true);
        V2.setHidden(ctx.roomPanel, false);
        ctx.nodes.endTurnBtn.disabled = true;
        ctx.nodes.concedeBtn.disabled = true;
        return Promise.resolve();
      }
      if (!ctx.displayGame) ctx.displayGame = TABLE.clone(ctx.game);
      return consumePresentation(payload, { initial: firstGame || options.initial, instant: options.instant }).then(function () {
        if (!ctx.playingEvents) alignAuthoritative();
        if (ctx.pendingEnvelope && !interactionLocked()) flushPendingEnvelope();
      });
    }

    function flushPendingEnvelope() {
      if (!ctx.pendingEnvelope || interactionLocked()) return;
      const queued = ctx.pendingEnvelope;
      ctx.pendingEnvelope = null;
      applyEnvelope(queued, { force: true });
    }
    ctx.flushPendingEnvelope = flushPendingEnvelope;

    function settlePresentation() {
      if (ctx.playingEvents && ctx.player && typeof ctx.player.cancel === 'function') {
        ctx.player.cancel();
      }
      ctx.playingEvents = false;
      ctx.page.classList.remove('presentation-busy');
      if (ctx.game) alignAuthoritative();
    }
    ctx.settlePresentation = settlePresentation;

    async function submitEntry(entry, options) {
      options = options || {};
      let retryMulligan = null;
      if (!entry || !entry.action) {
        ctx.showToast('没有合法动作。');
        return;
      }
      if (ctx.busy) return;
      if (ctx.playingEvents && ctx.player && typeof ctx.player.cancel === 'function') {
        ctx.player.cancel();
        alignAuthoritative();
      }
      const expectedVersion = Number(ctx.game && ctx.game.version);
      const actionRoomCode = ctx.room && ctx.room.room_code;
      if (!Number.isInteger(expectedVersion) || expectedVersion < 0) {
        ctx.showToast('当前对局版本未知，请刷新后再操作。');
        return;
      }
      const requestId = (ctx.pendingRequest && ctx.pendingRequest.fingerprint === V2.actionFingerprint(entry.action) && ctx.pendingRequest.id)
        ? ctx.pendingRequest.id
        : V2.makeRequestId();
      ctx.pendingRequest = {
        id: requestId,
        fingerprint: V2.actionFingerprint(entry.action),
        expectedVersion: expectedVersion,
        action: entry.action,
      };
      ctx.busy = true;
      renderRoomChrome();
      if (ctx.game) {
        ctx.nodes.endTurnBtn.disabled = true;
        ctx.nodes.concedeBtn.disabled = true;
        ctx.nodes.mulliganBtn.disabled = true;
      }
      try {
        const payload = await V2.sendAction({
          roomCode: actionRoomCode,
          action: entry.action,
          expectedVersion: ctx.pendingRequest.expectedVersion,
          requestId: requestId,
        });
        ctx.pendingRequest = null;
        ctx.localChoice = null;
        window.clearTimeout(ctx.toastTimer);
        ctx.toast.classList.remove('open');
        ctx.toast.setAttribute('aria-hidden', 'true');
        await applyEnvelope(payload, { force: true });
        V2.showBanner(ctx.banner, '', '');
        setConnection('live', '已同步');
      } catch (error) {
        if (error && error.status === 409) {
          await refreshState({ force: true });
          if (entry.action.type === 'mulligan' && !options.retryingMulligan && ctx.room && ctx.room.room_code === actionRoomCode) {
            retryMulligan = actionsOf('mulligan').find(function (candidate) {
              return V2.sameIdSet(candidate.action.card_ids, entry.action.card_ids);
            }) || null;
          }
          if (!retryMulligan) ctx.showToast(error.message || '版本已更新，请按最新局面操作。');
        } else if (error && error.status === 404) {
          renderEmpty('房间已结束', error.message || '当前没有房间。', 'error');
        } else {
          V2.showBanner(ctx.banner, error && error.message ? error.message : '操作失败。', 'error');
          ctx.showToast(error && error.message ? error.message : '操作失败');
          setConnection('error', '同步失败');
        }
      } finally {
        ctx.busy = false;
        if (ctx.pendingEnvelope && !interactionLocked()) flushPendingEnvelope();
        else if (ctx.game) alignAuthoritative();
        else renderRoomChrome();
      }
      if (retryMulligan) return submitEntry(retryMulligan, { retryingMulligan: true });
    }
    ctx.submitEntry = submitEntry;

    function toggleMulligan(cardId) {
      if (!actionsOf('mulligan').length) return;
      const id = String(cardId || '');
      const index = ctx.mulliganIds.indexOf(id);
      if (index >= 0) ctx.mulliganIds.splice(index, 1);
      else if (ctx.mulliganIds.length < 3) ctx.mulliganIds.push(id);
      else ctx.showToast('起手最多换 3 张。');
      TABLE.renderHands(renderCtx());
      renderMulliganChrome();
    }
    ctx.toggleMulligan = toggleMulligan;

    function submitMulligan() {
      const exact = findAction('mulligan', { card_ids: ctx.mulliganIds });
      if (exact) return submitEntry(exact);
      const keepAll = findAction('mulligan', { card_ids: [] });
      if (!ctx.mulliganIds.length && keepAll) return submitEntry(keepAll);
      const matched = actionsOf('mulligan').find(function (entry) {
        return V2.sameIdSet(entry.action.card_ids, ctx.mulliganIds);
      });
      if (matched) return submitEntry(matched);
      ctx.showToast('当前选择不是合法换牌组合。');
    }
    ctx.submitMulligan = submitMulligan;

    async function refreshState(options) {
      options = options || {};
      try {
        const payload = await V2.getState();
        await applyEnvelope(payload, options);
        if (!payload.room && !payload.game) {
          renderEmpty('没有进行中的房间', '可以从主页开始人机、创建房间或输入房间码加入。');
        }
        V2.showBanner(ctx.banner, ctx.banner.dataset.kind === 'error' && interactionLocked() ? ctx.banner.textContent : '', ctx.banner.dataset.kind === 'error' ? 'error' : '');
        setConnection('live', '已同步');
      } catch (error) {
        if (error && error.status === 404) {
          ctx.room = null;
          ctx.game = null;
          renderEmpty('没有进行中的房间', error.message || '当前没有房间。可以从主页开始或加入。', 'error');
          return;
        }
        if (error && error.status === 401) {
          renderEmpty('登录已失效', '请重新登录后再进入牌桌。', 'error');
          return;
        }
        V2.showBanner(ctx.banner, error && error.message ? error.message : '网络不可用，牌桌骨架仍保留。', 'error');
        setConnection('error', '网络中断');
      }
    }
    ctx.refreshState = refreshState;
    ctx.applyEnvelope = applyEnvelope;
    ctx.applyDisplayPatch = applyDisplayPatch;
    ctx.alignAuthoritative = alignAuthoritative;
    ctx.renderRoomChrome = renderRoomChrome;
    ctx.renderEmpty = renderEmpty;
    ctx.paintBoard = paintBoard;
    ctx.setConnection = setConnection;

    ctx.startPolling = function () {
      ctx.pollTimer = window.setInterval(function () {
        if (document.hidden || interactionLocked()) return;
        refreshState({ ignoreIfBusy: true });
      }, POLL_MS);
      window.addEventListener('beforeunload', function () { window.clearInterval(ctx.pollTimer); });
      document.addEventListener('visibilitychange', function () {
        if (!document.hidden && !interactionLocked()) refreshState({ ignoreIfBusy: true });
      });
    };
  }

  TABLE.bindSync = bindSync;
}(window));
