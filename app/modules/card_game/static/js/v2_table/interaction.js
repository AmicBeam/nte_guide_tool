(function (root) {
  const V2 = root.NTE_V2;
  const TABLE = root.NTE_V2_TABLE = root.NTE_V2_TABLE || {};
  const CLICK_DISTANCE = 8;

  function bindInteraction(ctx) {
    const page = ctx.page;
    const nodes = ctx.nodes;
    const ghost = ctx.ghost;
    const previewBox = ctx.previewBox;
    const choiceOverlay = ctx.choiceOverlay;
    const pileModal = ctx.pileModal;
    const cardPreview = ctx.cardPreview;
    const targetLine = ctx.targetLine;
    const castZone = ctx.castZone;

    function showPreview(entry) {
      if (!entry) {
        V2.setHidden(previewBox, true);
        return;
      }
      previewBox.innerHTML = '<strong>' + V2.escapeHtml(entry.label || '预览') + '</strong>' + V2.previewMarkup(entry.preview);
      V2.setHidden(previewBox, false);
    }

    function closeOverlay(node) {
      node.classList.remove('open');
      node.setAttribute('aria-hidden', 'true');
      node.innerHTML = '';
    }

    function openChoice(config) {
      const choices = config.choices || [];
      choiceOverlay.innerHTML =
        '<div class="v2-choice-dialog">' +
          '<div class="section-title-row"><h2>' + V2.escapeHtml(config.title || '请选择') + '</h2><button class="icon-btn" id="v2-choice-close" type="button" aria-label="关闭">×</button></div>' +
          '<div class="v2-choice-grid">' + (choices.length ? choices.map(function (choice) {
            const card = choice.card;
            return (
              '<button class="secondary-btn" type="button" data-choice-id="' + V2.escapeAttr(choice.id) + '">' +
                V2.escapeHtml(choice.label || (card && card.name) || choice.id) +
                (card && card.description ? '<small>' + V2.escapeHtml(card.description) + '</small>' : '') +
              '</button>'
            );
          }).join('') : '<p class="subtle">没有候选。</p>') + '</div>' +
        '</div>';
      choiceOverlay.classList.add('open');
      choiceOverlay.setAttribute('aria-hidden', 'false');
      document.getElementById('v2-choice-close').addEventListener('click', function () {
        ctx.localChoice = null;
        closeOverlay(choiceOverlay);
        ctx.flushPendingEnvelope();
      });
      choiceOverlay.querySelectorAll('[data-choice-id]').forEach(function (button) {
        button.addEventListener('click', function () {
          const picked = choices.find(function (item) { return String(item.id) === button.getAttribute('data-choice-id'); });
          ctx.localChoice = null;
          closeOverlay(choiceOverlay);
          if (picked) config.onPick(picked);
        });
      });
    }

    function openPile(title, cards) {
      pileModal.innerHTML =
        '<div class="v2-pile-dialog">' +
          '<div class="section-title-row"><h2>' + V2.escapeHtml(title) + '</h2><button class="icon-btn" id="v2-pile-close" type="button" aria-label="关闭">×</button></div>' +
          '<div class="v2-pile-grid">' + ((cards || []).length ? cards.map(function (card) {
            if (card.hidden) return '<div class="v2-hidden-card"></div>';
            return '<article class="v2-codex-card"><strong>' + V2.escapeHtml(card.name || '卡牌') + '</strong><p class="subtle">' + V2.escapeHtml(V2.cardTypeLabel(card.type, card.terminal)) + '</p><p>' + V2.escapeHtml(card.description || '') + '</p></article>';
          }).join('') : '<p class="subtle">空</p>') + '</div>' +
        '</div>';
      pileModal.classList.add('open');
      pileModal.setAttribute('aria-hidden', 'false');
      document.getElementById('v2-pile-close').addEventListener('click', function () { closeOverlay(pileModal); });
    }

    function showCardDetail(card, extraActions) {
      extraActions = extraActions || [];
      cardPreview.innerHTML =
        '<div class="v2-detail-card">' +
          '<div class="section-title-row"><h2>' + V2.escapeHtml(card.name || '卡牌') + '</h2><button class="icon-btn" id="v2-preview-close" type="button" aria-label="关闭">×</button></div>' +
          '<p class="subtle">' + V2.escapeHtml(V2.cardTypeLabel(card.type, card.terminal)) + ' · 费用 ' + V2.escapeHtml(card.cost) + (card.copy ? ' · 复制' : '') + '</p>' +
          '<p>' + V2.escapeHtml(card.description || '正在读取卡牌效果。') + '</p>' +
          (extraActions.length ? '<div class="hero-actions">' + extraActions.map(function (item, index) {
            return '<button class="' + (item.primary ? 'primary-btn' : 'secondary-btn') + '" type="button" data-extra-action="' + index + '">' + V2.escapeHtml(item.label || '操作') + '</button>';
          }).join('') + '</div>' : '') +
        '</div>';
      cardPreview.classList.add('open');
      cardPreview.setAttribute('aria-hidden', 'false');
      document.getElementById('v2-preview-close').addEventListener('click', function () { closeOverlay(cardPreview); });
      cardPreview.querySelectorAll('[data-extra-action]').forEach(function (button) {
        button.addEventListener('click', function () {
          const action = extraActions[Number(button.getAttribute('data-extra-action'))];
          closeOverlay(cardPreview);
          if (action && action.onPick) action.onPick();
          else if (action && action.entry) ctx.resolveAndSubmit(action.entry, action.entries);
        });
      });
    }

    function showCharacterDetail(character, isOpponent) {
      const found = character || {};
      const attack = !isOpponent ? ctx.findAction('attack', { character_id: found.id }) : null;
      const awaken = !isOpponent ? ctx.findAction('awaken', { character_id: found.id }) : null;
      showCardDetail({
        name: found.name,
        type: 'form',
        cost: '',
        description: ((found.awakened ? found.awakened_passive : found.passive) || '') + '\n' +
          (found.shape_description ? '形态：' + found.shape_description + '\n' : '') +
          TABLE.resourceLine(found),
      }, [
        attack ? { label: attack.label || '确认出击', entry: attack, primary: true } : null,
        awaken ? { label: awaken.label || '主动觉醒（1 行动力）', entry: awaken } : null,
      ].filter(Boolean));
    }

    function groupedPlay(cardId) {
      return TABLE.groupPlayEntries(ctx.game, ctx.playEntries(cardId));
    }

    function highlightTargets(ids, on) {
      (ids || []).forEach(function (id) {
        const entity = TABLE.canonicalEntityId(id);
        if (!entity) return;
        const node = page.querySelector('[data-entity-id="' + entity + '"]')
          || (entity.indexOf(':player') >= 0 ? page.querySelector('[data-player-side="' + entity.split(':')[0] + '"]') : null);
        if (node) node.classList.toggle('target-legal', Boolean(on));
      });
    }

    function drawTargetLine(fromNode, toNode) {
      if (!targetLine || !fromNode || !toNode) {
        if (targetLine) V2.setHidden(targetLine, true);
        return;
      }
      const a = fromNode.getBoundingClientRect();
      const b = toNode.getBoundingClientRect();
      const x1 = a.left + a.width / 2;
      const y1 = a.top + a.height / 2;
      const x2 = b.left + b.width / 2;
      const y2 = b.top + b.height / 2;
      const length = Math.hypot(x2 - x1, y2 - y1);
      const angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;
      targetLine.style.left = x1 + 'px';
      targetLine.style.top = y1 + 'px';
      targetLine.style.width = length + 'px';
      targetLine.style.transform = 'rotate(' + angle + 'deg)';
      V2.setHidden(targetLine, false);
    }

    function clearCastUi() {
      V2.setHidden(castZone, true);
      V2.setHidden(targetLine, true);
      page.querySelectorAll('.target-legal').forEach(function (node) { node.classList.remove('target-legal'); });
      page.querySelectorAll('.v2-hand-card.raised').forEach(function (node) { node.classList.remove('raised'); });
      ctx.raisedCardId = null;
    }

    function legalDrop(kind, id, zone) {
      const grouped = kind === 'card' ? groupedPlay(id) : null;
      const attackEntry = kind === 'character' ? ctx.findAction('attack', { character_id: id }) : null;
      return TABLE.legalDropFor(kind, grouped, zone, attackEntry);
    }

    function dropZoneAt(x, y) {
      ghost.style.pointerEvents = 'none';
      const el = document.elementFromPoint(x, y);
      ghost.style.pointerEvents = '';
      if (!el || !el.closest) return null;
      return el.closest('[data-drop-role], [data-entity-id], [data-player-side]');
    }

    function highlightDrop(zone, legal) {
      [nodes.playerFrontZone, nodes.opponentFrontZone, castZone].forEach(function (node) {
        if (node) node.classList.toggle('drop-target', Boolean(legal && zone === node));
      });
      page.querySelectorAll('[data-entity-id]').forEach(function (node) {
        node.classList.toggle('drop-target', Boolean(legal && zone === node));
      });
    }

    function moveGhost(x, y) {
      ghost.style.transform = 'translate(' + (x - 60) + 'px, ' + (y - 80) + 'px)';
    }

    function beginDrag(event, kind, id, sourceNode) {
      if (ctx.playingEvents && ctx.settlePresentation) ctx.settlePresentation();
      if (ctx.busy || ctx.dragState || event.button || ctx.localChoice || choiceOverlay.classList.contains('open')) return;
      if (ctx.game && (ctx.game.phase === 'mulligan' || ctx.game.phase === 'choice')) return;
      if (event.pointerType === 'mouse' && event.buttons !== 1) return;
      if (kind === 'card') {
        const grouped = groupedPlay(id);
        const known = grouped.groups.cast.length + grouped.groups.target.length + grouped.groups.form.length;
        if (!known) {
          return;
        }
      }
      event.preventDefault();
      if (typeof sourceNode.setPointerCapture === 'function') sourceNode.setPointerCapture(event.pointerId);
      ctx.dragState = {
        pointerId: event.pointerId,
        kind: kind,
        id: id,
        sourceNode: sourceNode,
        startX: event.clientX,
        startY: event.clientY,
        moved: false,
      };
      sourceNode.classList.add('dragging-source');
      ghost.innerHTML = sourceNode.innerHTML;
      ghost.classList.add('open');
      document.body.classList.add('v2-dragging');
      if (kind === 'card') {
        ctx.raisedCardId = id;
        sourceNode.classList.add('raised');
        const grouped = groupedPlay(id);
        const needsCast = grouped.groups.cast.length || grouped.groups.form.length;
        V2.setHidden(castZone, !needsCast);
        highlightTargets([].concat.apply([], grouped.groups.target.concat(grouped.groups.form).map(function (entry) {
          return (entry.interaction && entry.interaction.target_ids) || [];
        })), true);
      }
      moveGhost(event.clientX, event.clientY);
      sourceNode.addEventListener('pointermove', onDragMove);
      sourceNode.addEventListener('pointerup', onDragEnd);
      sourceNode.addEventListener('pointercancel', onDragCancel);
    }

    function cleanupDrag() {
      if (!ctx.dragState) return null;
      ctx.dragState.sourceNode.classList.remove('dragging-source', 'raised');
      ctx.dragState.sourceNode.removeEventListener('pointermove', onDragMove);
      ctx.dragState.sourceNode.removeEventListener('pointerup', onDragEnd);
      ctx.dragState.sourceNode.removeEventListener('pointercancel', onDragCancel);
      ghost.classList.remove('open');
      ghost.innerHTML = '';
      document.body.classList.remove('v2-dragging');
      highlightDrop(null, false);
      V2.setHidden(previewBox, true);
      const state = ctx.dragState;
      ctx.dragState = null;
      return state;
    }

    function onDragMove(event) {
      if (!ctx.dragState || event.pointerId !== ctx.dragState.pointerId) return;
      moveGhost(event.clientX, event.clientY);
      const distance = Math.hypot(event.clientX - ctx.dragState.startX, event.clientY - ctx.dragState.startY);
      if (distance > CLICK_DISTANCE) ctx.dragState.moved = true;
      const zone = dropZoneAt(event.clientX, event.clientY);
      const legal = legalDrop(ctx.dragState.kind, ctx.dragState.id, zone);
      highlightDrop(zone, Boolean(legal));
      if (ctx.dragState.kind === 'character') {
        showPreview(ctx.findAction('attack', { character_id: ctx.dragState.id }));
      } else {
        showPreview(legal || ctx.playEntries(ctx.dragState.id)[0] || null);
        if (zone && zone.getAttribute && zone.getAttribute('data-entity-id')) {
          drawTargetLine(ctx.dragState.sourceNode, zone);
        }
      }
    }

    function onDragCancel() {
      cleanupDrag();
      clearCastUi();
      ctx.flushPendingEnvelope();
    }

    function onDragEnd(event) {
      if (!ctx.dragState || event.pointerId !== ctx.dragState.pointerId) return;
      const state = cleanupDrag();
      const distance = Math.hypot(event.clientX - state.startX, event.clientY - state.startY);
      if (!state.moved || distance < CLICK_DISTANCE) {
        ctx.suppressClick = true;
        window.setTimeout(function () { ctx.suppressClick = false; }, 0);
        handleActivate(state.kind, state.id, state.sourceNode);
        clearCastUi();
        ctx.flushPendingEnvelope();
        return;
      }
      ctx.suppressClick = true;
      window.setTimeout(function () { ctx.suppressClick = false; }, 0);
      const zone = dropZoneAt(event.clientX, event.clientY);
      const legal = legalDrop(state.kind, state.id, zone);
      clearCastUi();
      if (!legal) {
        ctx.showToast('未放到合法区域，已回弹。');
        ctx.flushPendingEnvelope();
        return;
      }
      if (state.kind === 'character') ctx.submitEntry(legal);
      else ctx.resolveAndSubmit(legal, [legal]);
      ctx.flushPendingEnvelope();
    }

    function resolveAndSubmit(entry, entries) {
      const list = entries || (entry ? [entry] : []);
      if (!list.length) {
        ctx.showToast('没有合法动作。');
        return;
      }
      if (list.length === 1 && !list[0].action.target_id) {
        ctx.submitEntry(list[0]);
        return;
      }
      if (list.length === 1 && list[0].action.target_id) {
        ctx.submitEntry(list[0]);
        return;
      }
      ctx.localChoice = { kind: 'target' };
      openChoice({
        title: '选择完整目标',
        choices: list.map(function (item, index) {
          return { id: String(index), label: item.label || item.action.target_id || ('候选 ' + (index + 1)) };
        }),
        onPick: function (choice) {
          ctx.localChoice = null;
          ctx.submitEntry(list[Number(choice.id)]);
        },
      });
    }
    ctx.resolveAndSubmit = resolveAndSubmit;

    function handleActivate(kind, id, sourceNode) {
      if (ctx.playingEvents && ctx.settlePresentation) ctx.settlePresentation();
      if (ctx.game && ctx.game.phase === 'mulligan' && kind === 'card') {
        ctx.toggleMulligan(id);
        return;
      }
      if (kind === 'character') {
        const parsed = TABLE.parseEntityId(sourceNode && sourceNode.getAttribute('data-entity-id'));
        const isOpponent = parsed.side === ctx.opponent();
        const character = TABLE.characterMap(TABLE.sideState(ctx.game, parsed.side || ctx.viewer()))[id]
          || TABLE.characterMap(TABLE.sideState(ctx.game, ctx.viewer()))[id]
          || TABLE.characterMap(TABLE.sideState(ctx.game, ctx.opponent()))[id]
          || V2.characterFallback(id);
        showCharacterDetail(character, isOpponent);
        return;
      }
      const grouped = groupedPlay(id);
      const cycle = ctx.findAction('cycle', { card_id: id });
      const card = TABLE.cardByInstance(ctx.game, id) || { name: '卡牌', description: '' };
      const extras = [];
      if (grouped.groups.cast.length === 1 && !grouped.groups.target.length && !grouped.groups.form.length) {
        extras.push({ label: grouped.groups.cast[0].label || '施放', entry: grouped.groups.cast[0], entries: grouped.groups.cast, primary: true });
      } else {
        grouped.entries.forEach(function (entry) {
          extras.push({ label: entry.label || '打出', entry: entry, entries: grouped.entries });
        });
      }
      if (cycle) extras.push({ label: cycle.label || '弃一抽一', entry: cycle });
      showCardDetail(card, extras);
    }

    function bindPointer(container, kindAttr, idAttr) {
      container.addEventListener('pointerdown', function (event) {
        const source = event.target.closest('[' + kindAttr + ']');
        if (!source || ctx.busy || event.target.closest('.v2-inspect, .v2-attack-btn, .v2-awaken-btn, [data-inspect-character]')) return;
        const kind = source.getAttribute(kindAttr);
        const id = source.getAttribute(idAttr);
        if (!kind || !id) return;
        beginDrag(event, kind === 'character' ? 'character' : 'card', id, source);
      });
      container.addEventListener('click', function (event) {
        if (ctx.suppressClick) {
          ctx.suppressClick = false;
          event.preventDefault();
          return;
        }
        const attackBtn = event.target.closest('[data-attack-character]');
        if (attackBtn) {
          event.preventDefault();
          ctx.submitEntry(ctx.findAction('attack', { character_id: attackBtn.getAttribute('data-attack-character') }));
          return;
        }
        const awakenBtn = event.target.closest('[data-awaken-id]');
        if (awakenBtn) {
          event.preventDefault();
          ctx.submitEntry(ctx.findAction('awaken', { character_id: awakenBtn.getAttribute('data-awaken-id') }));
          return;
        }
        const inspect = event.target.closest('[data-inspect-character]');
        if (inspect) return;
        const source = event.target.closest('[' + idAttr + ']');
        if (!source || ctx.busy || ctx.dragState) return;
        handleActivate(source.getAttribute(kindAttr) === 'character' ? 'character' : 'card', source.getAttribute(idAttr), source);
      });
    }

    page.addEventListener('click', function (event) {
      const inspect = event.target.closest('[data-inspect-character]');
      if (!inspect || ctx.busy) return;
      const isOpponent = inspect.dataset.opponent === 'true';
      const character = TABLE.characterMap(TABLE.sideState(ctx.game, isOpponent ? ctx.opponent() : ctx.viewer()))[inspect.dataset.inspectCharacter];
      if (character) showCharacterDetail(character, isOpponent);
      const playChip = event.target.closest('[data-play-seq]');
      if (playChip && ctx.historyDrawer) {
        ctx.historyDrawer.classList.add('open');
        ctx.historyDrawer.setAttribute('aria-hidden', 'false');
      }
    });
    nodes.opponentHand.addEventListener('click', function (event) {
      const button = event.target.closest('[data-public-copy]');
      if (!button) return;
      const card = (TABLE.sideState(ctx.game, ctx.opponent()).hand || []).find(function (item) {
        return item.instance_id === button.dataset.publicCopy;
      });
      if (card) showCardDetail(card, []);
    });

    bindPointer(nodes.playerBench, 'data-drag-kind', 'data-character-id');
    bindPointer(nodes.playerFront, 'data-drag-kind', 'data-character-id');
    bindPointer(nodes.playerHand, 'data-drag-kind', 'data-card-id');
    function inspectOpponent(event) {
      const source = event.target.closest('[data-character-id]');
      if (!source || ctx.busy) return;
      if (event.target.closest('.v2-inspect')) return;
      const character = TABLE.characterMap(TABLE.sideState(ctx.game, ctx.opponent()))[source.getAttribute('data-character-id')];
      if (character) showCharacterDetail(character, true);
    }
    nodes.opponentBench.addEventListener('click', inspectOpponent);
    nodes.opponentFront.addEventListener('click', inspectOpponent);

    nodes.endTurnBtn.addEventListener('click', function () { ctx.submitEntry(ctx.findAction('end_turn')); });
    nodes.concedeBtn.addEventListener('click', function () { ctx.submitEntry(ctx.findAction('concede')); });
    nodes.mulliganBtn.addEventListener('click', ctx.submitMulligan);
    nodes.playerDiscard.addEventListener('click', function () { openPile('我方弃牌', TABLE.sideState(ctx.game, ctx.viewer()).discard || []); });
    nodes.opponentDiscard.addEventListener('click', function () { openPile('对手弃牌', TABLE.sideState(ctx.game, ctx.opponent()).discard || []); });
    nodes.playerDeck.addEventListener('click', function () {
      openPile('我方牌库', [{ name: '牌库 ' + (TABLE.sideState(ctx.game, ctx.viewer()).deck_count || 0) + ' 张', description: '牌序保密。' }]);
    });
    nodes.opponentDeck.addEventListener('click', function () {
      openPile('对手牌库', [{ name: '牌库 ' + (TABLE.sideState(ctx.game, ctx.opponent()).deck_count || 0) + ' 张', description: '牌序保密。' }]);
    });
    choiceOverlay.addEventListener('click', function (event) {
      if (event.target === choiceOverlay) {
        ctx.localChoice = null;
        closeOverlay(choiceOverlay);
      }
    });
    pileModal.addEventListener('click', function (event) { if (event.target === pileModal) closeOverlay(pileModal); });
    cardPreview.addEventListener('click', function (event) { if (event.target === cardPreview) closeOverlay(cardPreview); });

    ctx.openChoice = openChoice;
    ctx.closeOverlay = closeOverlay;
    ctx.showCardDetail = showCardDetail;
    ctx.showCharacterDetail = showCharacterDetail;
    ctx.showPreview = showPreview;
    ctx.clearCastUi = clearCastUi;
    ctx.handleActivate = handleActivate;
    ctx.legalDrop = legalDrop;
    ctx.groupedPlay = groupedPlay;
  }

  TABLE.bindInteraction = bindInteraction;
}(window));
