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
    ctx.viewer = function () { return ctx.replayMode && ctx.replayViewerSide || TABLE.viewerSide(ctx.game); };
    ctx.opponent = function () { return ctx.viewer() === 'b' ? 'a' : 'b'; };
    ctx.playHistory = ctx.playHistory || [];
    ctx.playerGeneration = ctx.playerGeneration || 0;
    ctx.aiStepTimer = 0;
    ctx.aiStepFailed = false;

    function scheduleAiStep() {
      if (ctx.replayMode || document.hidden || ctx.aiStepTimer || ctx.aiStepFailed || interactionLocked() ||
          !ctx.room || !ctx.room.ai_pending || !ctx.game) return;
      ctx.aiStepTimer = window.setTimeout(runAiStep, 0);
    }

    async function runAiStep() {
      ctx.aiStepTimer = 0;
      if (ctx.replayMode || document.hidden || interactionLocked() || !ctx.room || !ctx.room.ai_pending || !ctx.game) return;
      const roomCode = ctx.room.room_code;
      const request = ctx.pendingAiRequest || {
        roomCode: roomCode,
        expectedVersion: Number(ctx.game.version),
        requestId: V2.makeRequestId(),
      };
      ctx.pendingAiRequest = request;
      ctx.busy = true;
      setConnection('busy', '对手思考中');
      if (ctx.nodes.waitChip) ctx.nodes.waitChip.textContent = '对手思考中';
      try {
        const payload = await V2.advanceAi(request);
        if (!ctx.room || ctx.room.room_code !== roomCode) return;
        ctx.pendingAiRequest = null;
        await applyEnvelope(payload, { force: true });
      } catch (error) {
        if (error && error.status === 409) {
          ctx.pendingAiRequest = null;
          await refreshState({ force: true });
        } else {
          ctx.aiStepFailed = true;
          V2.showBanner(ctx.banner, (error && error.message || '人机操作未完成。') + ' 请刷新重试。', 'error');
          const retry = document.createElement('button');
          retry.type = 'button';
          retry.textContent = '重试这一步';
          retry.addEventListener('click', function () {
            ctx.aiStepFailed = false;
            V2.showBanner(ctx.banner, '', '');
            scheduleAiStep();
          });
          ctx.banner.appendChild(retry);
          setConnection('error', '人机操作未完成');
        }
      } finally {
        ctx.busy = false;
        if (ctx.pendingEnvelope && !interactionLocked()) flushPendingEnvelope();
        else if (ctx.game) alignAuthoritative();
        scheduleAiStep();
      }
    }

    function setConnection(state, label) {
      const chip = ctx.nodes.connectionChip;
      if (!chip) return;
      chip.dataset.state = state || 'live';
      chip.textContent = label || '已同步';
    }

    function interactionLocked() {
      return Boolean(ctx.replayMode || ctx.busy || ctx.dragState || ctx.localChoice || ctx.choiceOverlay.classList.contains('open') || ctx.playingEvents);
    }
    ctx.interactionLocked = interactionLocked;

    function currentViewer() {
      if (typeof ctx.viewer === 'function') {
        return ctx.viewer();
      }
      const game = ctx.displayGame || ctx.game;
      return ctx.viewer || TABLE.viewerSide(game) || 'a';
    }

    function currentOpponent() {
      if (typeof ctx.opponent === 'function') {
        return ctx.opponent();
      }
      return ctx.opponent || (currentViewer() === 'b' ? 'a' : 'b');
    }

    function renderCtx() {
      const targeting = Boolean(ctx.localChoice && (ctx.localChoice.kind === 'board-target' || ctx.localChoice.kind === 'choice-target'));
      const locked = ctx.busy || ctx.playingEvents || targeting;
      return {
        game: ctx.displayGame || ctx.game,
        room: ctx.room,
        viewer: currentViewer(),
        opponent: currentOpponent(),
        nodes: ctx.nodes,
        pool: ctx.pool,
        busy: locked,
        mulliganIds: ctx.mulliganIds,
        raisedCardId: ctx.raisedCardId,
        playHistory: ctx.playHistory,
        page: ctx.page,
        toast: ctx.toast,
        dragState: ctx.dragState,
        tutorialMask: ctx.tutorialMask,
        tutorialModal: ctx.tutorialModal,
        tutorialCue: ctx.tutorialCue,
        tutorialModalDismissed: ctx.tutorialModalDismissed,
        sendAction: ctx.sendAction,
        showMulligan: ctx.forceMulliganStage ? true : (ctx.mulliganFinished ? false : undefined),
        findAction: locked ? function () { return null; } : findAction,
        playEntries: locked ? function () { return []; } : ctx.playEntries,
      };
    }

    function bindPlayer() {
      if (ctx.player && typeof ctx.player.cancel === 'function') {
        ctx.player.cancel();
      }
      if (ctx.page && typeof ctx.page.querySelectorAll === 'function') {
        Array.prototype.slice.call(ctx.page.querySelectorAll('.v2-fx-overlay')).forEach(function (node) {
          if (node && node.parentNode) node.parentNode.removeChild(node);
        });
      }
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
          if (!busy && typeof ctx.syncResultDialog === 'function') ctx.syncResultDialog();
          if (ctx.nodes.connectionChip) {
            ctx.nodes.connectionChip.dataset.state = busy ? 'busy' : 'live';
            ctx.nodes.connectionChip.textContent = busy ? '演出中' : '已同步';
          }
        },
        onPublicCard: function () {},
        onFinish: function () {
          if (generation !== ctx.playerGeneration) return;
          if (typeof ctx.openResultFromEvent === 'function') ctx.openResultFromEvent();
        },
      });
      if (ctx.player && typeof ctx.player.setOptions === 'function') {
        ctx.player.setOptions(ctx.options);
      }
    }
    ctx.bindPlayer = bindPlayer;
    if (!ctx.player) bindPlayer();

    function resetRoomPresentation() {
      if (ctx.aiStepTimer) window.clearTimeout(ctx.aiStepTimer);
      ctx.aiStepTimer = 0;
      ctx.pendingAiRequest = null;
      ctx.aiStepFailed = false;
      ctx.seenCursor = null;
      ctx.displayGame = null;
      ctx.playHistory = [];
      ctx.pendingEnvelope = null;
      ctx.mulliganIds = [];
      ctx.mulliganHold = null;
      ctx.mulliganFinished = false;
      ctx.forceMulliganStage = false;
      ctx.raisedCardId = null;
      ctx.playingEvents = false;
      ctx.initiativePlayed = false;
      if (ctx.initiativeTimer) {
        window.clearTimeout(ctx.initiativeTimer);
        ctx.initiativeTimer = null;
      }
      if (ctx.endBoardTargetSelect) ctx.endBoardTargetSelect();
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
      const replayCode = ctx.room && (ctx.room.replay_code || ctx.room.room_code);
      const canReplay = Boolean(!ctx.replayMode && ctx.room && ctx.room.has_replay && replayCode && ctx.nodes.replayLink);
      if (ctx.nodes.replayLink) {
        V2.setHidden(ctx.nodes.replayLink, !canReplay);
        if (canReplay) ctx.nodes.replayLink.href = '/table?replay=' + encodeURIComponent(replayCode);
      }
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
        ctx.nodes.roomCopy.textContent = (ctx.room.mode === 'advanced' ? '高级人机' : (ctx.room.mode === 'solo' ? '人机对战' : '1v1 对战')) + ' · ' + statusLabel + '。双方准备后，由房主开始。';
        renderMembers();
      } else {
        V2.setHidden(ctx.roomPanel, true);
      }
      if (typeof ctx.syncResultDialog === 'function') ctx.syncResultDialog();
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
      const inMulligan = Boolean(ctx.forceMulliganStage || (ctx.game && ctx.game.phase === 'mulligan' && !ctx.mulliganFinished));
      const canMulligan = inMulligan && actionsOf('mulligan').length > 0;
      ctx.page.classList.toggle('is-mulligan', inMulligan);
      V2.setHidden(ctx.nodes.mulliganToggle, !inMulligan);
      if (!inMulligan) {
        ctx.page.classList.remove('mulligan-collapsed');
        ctx.nodes.mulliganToggle.setAttribute('aria-expanded', 'true');
        ctx.nodes.mulliganToggle.textContent = '收起换牌';
      }
      if (ctx.nodes.mulliganStage) V2.setHidden(ctx.nodes.mulliganStage, !inMulligan);
      V2.setHidden(ctx.nodes.mulliganBtn, !canMulligan);
      const selected = ctx.mulliganIds.length;
      if (ctx.nodes.mulliganStageCopy) {
        ctx.nodes.mulliganStageCopy.textContent = canMulligan
          ? ('已选 ' + selected + ' / 3。点选不要的牌，确认后立即换入新牌。')
          : '已完成换牌，等待对手。';
      }
      ctx.nodes.mulliganBtn.disabled = ctx.busy || !canMulligan;
    }

    function renderChoiceFromServer() {
      if (ctx.replayMode) return;
      const pending = ctx.game && ctx.game.pending_choice;
      if (ctx.localChoice || ctx.dragState) return;
      if (!pending || pending.side !== ctx.viewer()) {
        if (!ctx.localChoice) ctx.closeOverlay(ctx.choiceOverlay);
        return;
      }
      ctx.openChoice({
        title: pending.prompt || '请选择',
        choices: pending.choices || [],
        maxSelect: pending.max_select,
        onPick: function (choice) {
          if (pending.max_select) {
            ctx.submitEntry(findAction('choose_cards', { card_ids: choice.ids }));
            return;
          }
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
      updateInitiative(ctx.game);
      if (ctx.reapplyBoardTargetSelect) ctx.reapplyBoardTargetSelect();
      renderAuthoritativeControls();
    }

    function updateInitiative(game) {
      const overlay = document.getElementById('v2-initiative-overlay');
      const chip = document.getElementById('v2-initiative-chip');
      const label = document.getElementById('v2-initiative-label');
      const copy = document.getElementById('v2-initiative-copy');
      if (!overlay || !chip) {
        return;
      }
      if (!game || !game.first_side || !game.viewer_side) {
        V2.setHidden(overlay, true);
        V2.setHidden(chip, true);
        return;
      }
      const isFirst = game.first_side === game.viewer_side;
      const title = isFirst ? '先手' : '后手';
      if (label) {
        label.textContent = title;
      }
      if (copy) {
        copy.textContent = isFirst ? '你先行动。' : '对手先行动。你有 5 点开局护盾。';
      }
      chip.textContent = title;
      chip.classList.toggle('is-second', !isFirst);
      overlay.classList.toggle('is-second', !isFirst);
      if (game.phase === 'mulligan') {
        V2.setHidden(chip, false);
        if (ctx.replayMode) {
          V2.setHidden(overlay, true);
          return;
        }
        if (!ctx.initiativePlayed) {
          ctx.initiativePlayed = true;
          overlay.classList.remove('is-leaving');
          overlay.classList.add('is-playing');
          V2.setHidden(overlay, false);
          if (ctx.initiativeTimer) {
            window.clearTimeout(ctx.initiativeTimer);
          }
          ctx.initiativeTimer = window.setTimeout(function () {
            overlay.classList.add('is-leaving');
            ctx.initiativeTimer = window.setTimeout(function () {
              V2.setHidden(overlay, true);
              overlay.classList.remove('is-playing', 'is-leaving');
            }, 280);
          }, 1400);
        }
        return;
      }
      V2.setHidden(overlay, true);
      V2.setHidden(chip, true);
      overlay.classList.remove('is-playing', 'is-leaving');
    }

    function applyDisplayPatch(patch, event) {
      const wasMulligan = !ctx.mulliganFinished && ctx.displayGame && ctx.displayGame.phase === 'mulligan';
      const firsts = wasMulligan && patch && patch.phase && patch.phase !== 'mulligan' ? captureMulliganRects() : null;
      ctx.displayGame = TABLE.applyPatch(ctx.displayGame || ctx.game, patch || {}, event);
      ctx.displayGame = TABLE.appendPresentedEvent(ctx.displayGame, event);
      if (!ctx.replayMode && !ctx.playingEvents) {
        ctx.displayGame = TABLE.copyViewerHand(ctx.displayGame, ctx.game, currentViewer());
      }
      TABLE.renderGame(renderCtx());
      if (firsts) {
        dealMulliganToHand(firsts);
      }
      if (ctx.reapplyBoardTargetSelect) ctx.reapplyBoardTargetSelect();
    }

    function alignAuthoritative() {
      ctx.displayGame = TABLE.clone(ctx.game);
      if (ctx.game) paintBoard();
      else renderRoomChrome();
    }

    function lastEventSeq(events, fallback) {
      const seqs = (events || []).map(function (event) { return Number(event && event.seq); }).filter(function (value) {
        return Number.isFinite(value);
      });
      if (!seqs.length) {
        return fallback;
      }
      return Math.max.apply(null, seqs);
    }

    function waitMulliganHandDeal(firsts) {
      if (ctx.options && ctx.options.reducedMotion) {
        dealMulliganToHand(firsts);
        return Promise.resolve();
      }
      return new Promise(function (resolve) {
        requestAnimationFrame(function () {
          requestAnimationFrame(function () {
            if (ctx.nodes.playerHand) {
              void ctx.nodes.playerHand.offsetWidth;
            }
            dealMulliganToHand(firsts);
            window.setTimeout(resolve, 720);
          });
        });
      });
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
      return playOpeningMulligan(fresh, options).then(function () {
        if (generation !== ctx.playerGeneration) return;
        ctx.seenCursor = Number.isFinite(nextCursor) ? nextCursor : ctx.seenCursor;
      }).catch(function () {
        if (generation !== ctx.playerGeneration) return;
        ctx.seenCursor = Number.isFinite(nextCursor) ? nextCursor : ctx.seenCursor;
      }).finally(function () {
        if (generation !== ctx.playerGeneration) return;
        ctx.forceMulliganStage = false;
        ctx.playingEvents = false;
        ctx.page.classList.remove('presentation-busy');
        setConnection('live', '已同步');
        alignAuthoritative();
      });
    }

    async function playOpeningMulligan(events, options) {
      const playOptions = { instant: Boolean(options.instant) };
      const hold = ctx.mulliganHold;
      ctx.mulliganHold = null;
      const viewer = currentViewer();
      const start = hold && events.findIndex(function (event) { return event.type === 'mulligan' && event.side === viewer; });
      if (!hold || start < 0 || !ctx.nodes.mulliganRail) return ctx.player.play(events, playOptions);
      ctx.forceMulliganStage = true;
      ctx.displayGame = TABLE.clone(hold.snapshot);
      TABLE.renderGame(renderCtx());
      const generation = ctx.playerGeneration;
      if (start) await ctx.player.play(events.slice(0, start), playOptions);
      if (generation !== ctx.playerGeneration) return;
      const event = events[start];
      const count = Number(event.amount) || 0;
      const draws = events.slice(start + 1, start + 1 + count);
      if (draws.length !== count || draws.some(function (draw) { return draw.type !== 'draw' || draw.side !== viewer || !draw.card; })) {
        ctx.forceMulliganStage = false;
        return ctx.player.play(events.slice(start), playOptions);
      }
      const rail = ctx.nodes.mulliganRail;
      const slots = Array.from(rail.querySelectorAll('[data-card-id]'));
      const selected = (event.card_ids || []).map(String);
      const outgoing = slots.map(function (node) { return node.getAttribute('data-card-id'); }).filter(function (id) { return selected.includes(id); });
      ctx.forceMulliganStage = true;
      ctx.playingEvents = true;
      ctx.page.classList.add('presentation-busy');
      // Keep the original nodes and their geometry until each replacement arrives.
      await TABLE.animateMulliganSwap({
        rail: rail,
        outgoingIds: outgoing,
        incomingIds: draws.map(function (draw) { return String(draw.card.instance_id); }),
        reducedMotion: playOptions.instant || (ctx.options && ctx.options.reducedMotion),
        cancelled: function () { return generation !== ctx.playerGeneration; },
        replaceOne: function (oldId, newId, index) {
          const old = slots.find(function (node) { return node.getAttribute('data-card-id') === oldId; });
          if (!old) return;
          const style = window.getComputedStyle(old);
          const bounds = { width: parseFloat(style.width), height: parseFloat(style.height) };
          const holder = document.createElement('div');
          holder.innerHTML = TABLE.handCardHtml(draws[index].card, { phase: 'mulligan' });
          const next = holder.firstElementChild;
          next.classList.toggle('v2-hand-group-start', old.classList.contains('v2-hand-group-start'));
          next.style.width = bounds.width + 'px';
          next.style.height = bounds.height + 'px';
          next.style.flex = '0 0 ' + bounds.width + 'px';
          old.replaceWith(next);
        },
      });
      if (generation !== ctx.playerGeneration) return;
      // Apply the corresponding private patches only after the last slot has settled.
      const team = TABLE.sideState(ctx.displayGame, viewer);
      team.hand = (team.hand || []).filter(function (card) { return !selected.includes(String(card.instance_id)); });
      [event].concat(draws).forEach(function (item) {
        ctx.displayGame = TABLE.applyPatch(ctx.displayGame, item.patch || {}, item);
        ctx.displayGame = TABLE.appendPresentedEvent(ctx.displayGame, item);
      });
      const firsts = captureMulliganRects();
      ctx.forceMulliganStage = false;
      ctx.mulliganFinished = true;
      TABLE.renderGame(renderCtx());
      await waitMulliganHandDeal(firsts);
      if (generation !== ctx.playerGeneration) return;
      await ctx.player.play(events.slice(start + 1 + count), playOptions);
    }

    function captureMulliganRects() {
      const map = {};
      const rail = ctx.nodes.mulliganRail;
      if (!rail) return map;
      Array.prototype.forEach.call(rail.querySelectorAll('[data-card-id]'), function (node) {
        const id = node.getAttribute('data-card-id');
        if (id) map[id] = node.getBoundingClientRect();
      });
      return map;
    }

    function dealMulliganToHand(firsts) {
      if (!firsts || (ctx.options && ctx.options.reducedMotion)) return;
      const hand = ctx.nodes.playerHand;
      if (!hand) return;
      Array.prototype.forEach.call(hand.querySelectorAll('[data-card-id]'), function (node) {
        const first = firsts[node.getAttribute('data-card-id')];
        const last = node.getBoundingClientRect();
        if (!first || last.width < 40 || last.height < 40) {
          node.classList.add('mulligan-arrive');
          return;
        }
        const dx = first.left - last.left;
        const dy = first.top - last.top;
        const sx = first.width / Math.max(last.width, 1);
        const sy = first.height / Math.max(last.height, 1);
        node.style.transformOrigin = 'top left';
        node.style.transition = 'none';
        node.style.transform = 'translate(' + dx + 'px,' + dy + 'px) scale(' + sx + ',' + sy + ')';
        node.style.zIndex = '8';
        requestAnimationFrame(function () {
          requestAnimationFrame(function () {
            node.style.transition = 'transform 420ms cubic-bezier(.2,.8,.2,1)';
            node.style.transform = 'none';
          });
        });
        node.addEventListener('transitionend', function cleanup(event) {
          if (event.propertyName && event.propertyName !== 'transform') return;
          node.style.transform = '';
          node.style.transition = '';
          node.style.zIndex = '';
          node.style.transformOrigin = '';
          node.removeEventListener('transitionend', cleanup);
        });
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
        scheduleAiStep();
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

    ctx.submitEntry = submitEntry;
    async function submitEntry(entry, options) {
      options = options || {};
      let retryMulligan = null;
      if (ctx.replayMode) return;
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
      const wireAction = entry.entity_action || entry.action;
      const requestId = (ctx.pendingRequest && ctx.pendingRequest.fingerprint === V2.actionFingerprint(entry.action) && ctx.pendingRequest.id)
        ? ctx.pendingRequest.id
        : V2.makeRequestId();
      ctx.pendingRequest = {
        id: requestId,
        fingerprint: V2.actionFingerprint(entry.action),
        expectedVersion: expectedVersion,
        action: wireAction,
      };
      if (entry.action.type === 'mulligan') {
        ctx.mulliganHold = {
          outgoingIds: (entry.action.card_ids || []).map(String),
          snapshot: TABLE.clone(ctx.displayGame || ctx.game),
        };
      }
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
          action: wireAction,
          expectedVersion: ctx.pendingRequest.expectedVersion,
          requestId: requestId,
        });
        ctx.pendingRequest = null;
        ctx.localChoice = null;
        if (entry.action.type === 'mulligan') ctx.mulliganIds = [];
        window.clearTimeout(ctx.toastTimer);
        ctx.toast.classList.remove('open');
        ctx.toast.setAttribute('aria-hidden', 'true');
        await applyEnvelope(payload, { force: true });
        V2.showBanner(ctx.banner, '', '');
        setConnection('live', '已同步');
      } catch (error) {
        ctx.mulliganHold = null;
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
        scheduleAiStep();
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
        if (!ctx.aiStepFailed) {
          V2.showBanner(ctx.banner, ctx.banner.dataset.kind === 'error' && interactionLocked() ? ctx.banner.textContent : '', ctx.banner.dataset.kind === 'error' ? 'error' : '');
          setConnection('live', '已同步');
        }
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
      if (ctx.replayMode) return;
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
