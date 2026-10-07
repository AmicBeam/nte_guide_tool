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
    let optionZones = null;
    const battleArea = page.querySelector('[data-drop-role="battle-area"]');
    const mobileLayout = window.matchMedia('(pointer: coarse) and (max-width: 1200px)');
    let ignoreHandToggleClick = false;
    page.addEventListener('pointerdown', function () { ignoreHandToggleClick = false; }, true);

    function dragTargetingEnabled() {
      return !mobileLayout.matches && ctx.options.dragTargeting !== false;
    }

    function setMobileHandOpen(open) {
      if (ctx.dragState) return;
      page.classList.toggle('mobile-hand-open', Boolean(open));
      if (ctx.game && ctx.game.tutorial && TABLE.renderTutorial) TABLE.renderTutorial(ctx);
    }

    mobileLayout.addEventListener('change', function () {
      setMobileHandOpen(false);
      syncHandButton();
    });

    function syncHandButton() {
      if (nodes.playerHandCount) nodes.playerHandCount.setAttribute('aria-expanded', String(
        !mobileLayout.matches || page.classList.contains('mobile-hand-open')
      ));
    }
    new MutationObserver(syncHandButton).observe(page, { attributes: true, attributeFilter: ['class'] });
    syncHandButton();

    page.addEventListener('click', function (event) {
      if (!mobileLayout.matches || event.defaultPrevented || ctx.dragState || ctx.suppressClick
          || ignoreHandToggleClick || ctx.localChoice
          || page.classList.contains('is-mulligan') || page.classList.contains('tutorial-hide-hand')
          || choiceOverlay.classList.contains('open')) return;
      if (!event.target.closest('[data-drop-role="battle-area"], .v2-hud-top, #v2-hand-dock')) return;
      const handButton = event.target.closest('#v2-player-hand-count');
      if (!handButton && event.target.closest('button, a, input, select, label, .v2-character-card, .v2-hand-stack, .v2-player-hud, .v2-piles, .v2-side-actions, .v2-combat-core, .v2-zone-effects, .v2-mulligan-stage, .v2-history-drawer, .v2-hover-preview')) return;
      hideHoverPreview({ force: true });
      setMobileHandOpen(!page.classList.contains('mobile-hand-open'));
    });

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

    function isTouchLike(pointerType) {
      if (pointerType === 'mouse' || pointerType === 'keyboard') {
        return false;
      }
      if (pointerType === 'touch') {
        return true;
      }
      try {
        return window.matchMedia('(hover: none), (pointer: coarse)').matches;
      } catch (_error) {
        return false;
      }
    }

    function hideHoverPreview(options) {
      options = options || {};
      if (ctx.hoverPinned && !options.force) {
        return;
      }
      ctx.hoverPinned = false;
      cardPreview.className = 'v2-hover-preview';
      cardPreview.setAttribute('aria-hidden', 'true');
      cardPreview.innerHTML = '';
    }

    function hoverCardMarkup(card) {
      card = V2.handCardFace(card);
      if (card.type === 'mechanism') {
        return '<article class="v2-hover-card is-mechanism"><div class="v2-stat-row">' +
          V2.cardTypeChip(card.type) + '</div><h2>' + V2.escapeHtml(card.name) + '</h2>' +
          V2.cardDescriptionHtml(card.description) + '</article>';
      }
      if (card.selection_kind === 'character') {
        return '<article class="v2-hover-card is-character"><div class="v2-hover-art">' +
          V2.imageMarkup(card.portrait || card.avatar, card.name, 'v2-portrait') + '</div><h2>' +
          V2.escapeHtml(card.name) + '</h2>' + V2.cardDescriptionHtml(card.description) + '</article>';
      }
      const owner = (V2.characterFallback(card.character_id) || {}).name || '';
      const portrait = card.portrait || V2.characterAsset(V2.characterFallback(card.character_id) || {}, 'portrait')
        || V2.characterAsset(V2.characterFallback(card.character_id) || {}, 'avatar');
      return (
        '<article class="v2-hover-card' + (card.type === 'mechanism' ? ' is-mechanism' : '') + '"' + (V2.cardTypeToken(card.type) ? ' data-card-type="' + V2.escapeAttr(V2.cardTypeToken(card.type)) + '"' : '') + '>' +
          '<div class="v2-hover-art">' + V2.imageMarkup(portrait, card.name, 'v2-portrait') + V2.cardFrameStat(card) + '</div>' +
          V2.cardCostBadge(card) +
          '<div class="v2-stat-row">' + V2.cardTypeChip(card.type, card.terminal) +
            V2.cardDerivedChip(card) +
            (card.copy ? '<span class="v2-chip copy">复制</span>' : '') + '</div>' +
          (owner ? '<span class="v2-card-owner">' + V2.escapeHtml(owner) + '</span>' : '') +
          '<h2>' + V2.escapeHtml(card.name || '卡牌') + '</h2>' +
          V2.cardDescriptionHtml(card.description, '正在读取卡牌效果。') +
        '</article>'
      );
    }

    function hoverCharacterMarkup(character) {
      const found = character || {};
      const portrait = V2.battlePortrait(found);
      const lines = [
        found.flavor ? '<p><em>' + V2.escapeHtml(found.flavor) + '</em></p>' : '',
        '<p>异能：' + V2.cardDescriptionMarkup(found.passive || '尚未提供') + '</p>',
        '<p>终结：' + V2.cardDescriptionMarkup(found.awakened_passive || '尚未提供') + '</p>',
      ];
      if (found.shape_description) {
        lines.push('<p>弧盘：' + V2.escapeHtml(found.shape_description) + '</p>');
      }
      if (Array.isArray(found.effect_markers) && found.effect_markers.length) {
        lines.push('<section class="v2-effect-details"><h3>额外效果</h3>' + found.effect_markers.map(function (marker) {
          const detail = [marker.description, marker.timing, marker.clears_on_down ? '倒地时失去' : '倒地后保留'].filter(Boolean).join(' · ');
          return '<p><strong>' + V2.escapeHtml(marker.name) + '</strong><br>' + V2.escapeHtml(detail) + '</p>';
        }).join('') + '</section>');
      }
      const resources = TABLE.resourceLine(found);
      if (resources) {
        lines.push('<p class="subtle">' + V2.escapeHtml(resources) + '</p>');
      }
      return (
        '<article class="v2-hover-card is-character">' +
          '<div class="v2-hover-art">' + V2.imageMarkup(portrait, found.name, 'v2-portrait') + '</div>' +
          '<h2>' + V2.escapeHtml(found.name || '角色') + '</h2>' +
          (found.attribute ? V2.attributeMarkup(found.attribute) : '') +
          (found.faction ? '<span class="v2-faction">' + V2.escapeHtml(found.faction) + '</span>' : '') +
          '<p class="subtle">攻击 ' + V2.escapeHtml(found.attack == null ? '-' : found.attack) +
            ' · 生命 ' + V2.escapeHtml((found.hp == null ? '-' : found.hp) + '/' + (found.max_hp == null ? '-' : found.max_hp)) + '</p>' +
          lines.join('') +
        '</article>'
      );
    }

    function payloadFromNode(node) {
      if (!node || !node.getAttribute) {
        return null;
      }
      const publicCopy = node.getAttribute('data-public-copy');
      if (publicCopy) {
        const card = ((TABLE.sideState(ctx.game, ctx.opponent()).hand) || []).find(function (item) {
          return item.instance_id === publicCopy;
        });
        return card && !card.hidden ? { kind: 'card', card: card } : null;
      }
      const dragKind = node.getAttribute('data-drag-kind');
      const cardId = node.getAttribute('data-card-id');
      if (dragKind === 'card' || (cardId && node.classList && node.classList.contains('v2-hand-card'))) {
        const card = TABLE.cardByInstance(ctx.game, cardId);
        return card && !card.hidden ? { kind: 'card', card: card } : null;
      }
      const characterId = node.getAttribute('data-character-id');
      if (characterId) {
        const parsed = TABLE.parseEntityId(node.getAttribute('data-entity-id'));
        const side = parsed.side || ctx.viewer();
        const character = TABLE.characterMap(TABLE.sideState(ctx.game, side))[characterId]
          || TABLE.characterMap(TABLE.sideState(ctx.game, ctx.viewer()))[characterId]
          || TABLE.characterMap(TABLE.sideState(ctx.game, ctx.opponent()))[characterId]
          || V2.characterFallback(characterId);
        return character ? { kind: 'character', character: character } : null;
      }
      return null;
    }

    function showHoverPreview(payload, sourceNode, options) {
      options = options || {};
      if (!payload || ctx.dragState) {
        return;
      }
      ctx.hoverPinned = Boolean(options.pinned);
      const touchLike = isTouchLike(options.pointerType);
      const rect = sourceNode && sourceNode.getBoundingClientRect ? sourceNode.getBoundingClientRect() : null;
      const placeLeft = Boolean(rect && rect.left > (window.innerWidth * 0.55));
      cardPreview.className = 'v2-hover-preview open' + (placeLeft ? ' is-left' : ' is-right') +
        (ctx.hoverPinned ? ' is-pinned' : '') + (touchLike ? ' is-touch' : '');
      cardPreview.setAttribute('aria-hidden', 'false');
      const body = payload.kind === 'character' ? hoverCharacterMarkup(payload.character) : hoverCardMarkup(payload.card);
      const closeBtn = (ctx.hoverPinned || touchLike)
        ? '<button class="icon-btn preview-close" type="button" data-hover-close="1" aria-label="关闭">×</button>'
        : '';
      const referenceSource = payload.kind === 'card' ? payload.card : {name: payload.character.name, description: [payload.character.passive, payload.character.awakened_passive, payload.character.shape_description].filter(Boolean).join('。')};
      const references = V2.referencedCards(referenceSource);
      cardPreview.classList.toggle('has-references', references.length > 0);
      const referenceHtml = references.length ? '<aside class="v2-hover-references" aria-label="关联卡牌">' + references.map(hoverCardMarkup).join('') + '</aside>' : '';
      cardPreview.innerHTML = closeBtn + referenceHtml + body;
    }

    function openChoice(config) {
      const choices = config.choices || [];
      const selectedIds = new Set();
      const maxSelect = Number(config.maxSelect) || 0;
      const hasCards = choices.some(function (choice) {
        return Boolean(choice && choice.card && !choice.card.hidden);
      });
      choiceOverlay.innerHTML =
        '<div class="v2-choice-dialog' + (hasCards ? ' is-cards' : '') + (maxSelect ? ' is-multi' : '') + '">' +
          '<div class="section-title-row"><h2>' + V2.escapeHtml(config.title || '请选择') + '</h2><button class="icon-btn" id="v2-choice-close" type="button" aria-label="关闭">×</button></div>' +
          '<div class="v2-choice-grid' + (hasCards ? ' has-cards' : '') + '">' + (choices.length ? choices.map(function (choice) {
            const card = choice.card;
            if (card && !card.hidden) {
              return (
                '<button class="v2-choice-card" type="button" data-choice-id="' + V2.escapeAttr(choice.id) + '">' +
                  hoverCardMarkup(card) +
                '</button>'
              );
            }
            return (
              '<button class="secondary-btn" type="button" data-choice-id="' + V2.escapeAttr(choice.id) + '">' +
                V2.escapeHtml(choice.label || (card && card.name) || choice.id) +
                (card && card.description ? '<small>' + V2.cardDescriptionMarkup(card.description) + '</small>' : '') +
              '</button>'
            );
          }).join('') : '<p class="subtle">没有候选。</p>') + '</div>' +
          (maxSelect ? '<button class="primary-btn" id="v2-choice-confirm" type="button">确认调度（0 / ' + maxSelect + '）</button>' : '') +
        '</div>';
      choiceOverlay.classList.add('open');
      choiceOverlay.setAttribute('aria-hidden', 'false');
      document.getElementById('v2-choice-close').addEventListener('click', function () {
        markRaisedCard(ctx.localChoice && ctx.localChoice.cardId, false);
        ctx.localChoice = null;
        closeOverlay(choiceOverlay);
        ctx.flushPendingEnvelope();
      });
      const confirm = choiceOverlay.querySelector('#v2-choice-confirm');
      if (confirm) confirm.addEventListener('click', function () {
        ctx.localChoice = null;
        closeOverlay(choiceOverlay);
        config.onPick({ ids: Array.from(selectedIds) });
      });
      choiceOverlay.querySelectorAll('[data-choice-id]').forEach(function (button) {
        button.addEventListener('click', function () {
          const picked = choices.find(function (item) { return String(item.id) === button.getAttribute('data-choice-id'); });
          if (maxSelect && picked) {
            if (selectedIds.has(picked.id)) selectedIds.delete(picked.id);
            else if (selectedIds.size < maxSelect) selectedIds.add(picked.id);
            ctx.localChoice = { kind: 'server-multi' };
            button.classList.toggle('is-selected', selectedIds.has(picked.id));
            button.setAttribute('aria-pressed', String(selectedIds.has(picked.id)));
            confirm.textContent = '确认调度（' + selectedIds.size + ' / ' + maxSelect + '）';
            return;
          }
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
            return '<article class="v2-codex-card"' + (V2.cardTypeToken(card.type) ? ' data-card-type="' + V2.escapeAttr(V2.cardTypeToken(card.type)) + '"' : '') + '><strong>' + V2.escapeHtml(card.name || '卡牌') + '</strong><p class="subtle">' + V2.cardTypeChip(card.type, card.terminal) + '</p>' + V2.cardDescriptionHtml(card.description) + '</article>';
          }).join('') : '<p class="subtle">空</p>') + '</div>' +
        '</div>';
      pileModal.classList.add('open');
      pileModal.setAttribute('aria-hidden', 'false');
      document.getElementById('v2-pile-close').addEventListener('click', function () { closeOverlay(pileModal); });
    }

    function showCardDetail(card, extraActions, sourceNode) {
      showHoverPreview({ kind: 'card', card: card || {} }, sourceNode, { pinned: true, pointerType: 'touch' });
    }

    function showCharacterDetail(character, isOpponent, sourceNode) {
      showHoverPreview(
        { kind: 'character', character: character || {} },
        sourceNode,
        { pinned: isTouchLike(), pointerType: isTouchLike() ? 'touch' : 'mouse' }
      );
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

    function clearOptionZones() {
      if (optionZones) optionZones.remove();
      optionZones = null;
    }

    function showOptionZones(card, entries) {
      clearOptionZones();
      if (!battleArea || !card || !(card.play_options || []).length) return;
      const rect = battleArea.getBoundingClientRect();
      optionZones = document.createElement('div');
      optionZones.className = 'v2-play-option-zones';
      optionZones.setAttribute('aria-label', '选择卡牌效果');
      optionZones.style.cssText = 'left:' + rect.left + 'px;top:' + rect.top + 'px;width:' + rect.width + 'px;height:' + rect.height + 'px;';
      optionZones.innerHTML = card.play_options.map(function (option, index) {
        const available = entries.some(function (entry) { return entry.action.option_id === option.id; });
        return '<div class="v2-play-option-zone' + (available ? '' : ' is-disabled') + '" data-play-option="' + V2.escapeAttr(option.id) + '" aria-disabled="' + !available + '">' +
          '<strong>' + (index + 1) + ' · ' + V2.escapeHtml(option.label) + '</strong>' +
          V2.cardDescriptionHtml(option.description) + (available ? '' : '<small>资源不足</small>') + '</div>';
      }).join('');
      page.appendChild(optionZones);
    }

    function clearCastUi() {
      clearOptionZones();
      page.classList.remove('v2-playing-card', 'v2-playing-attack-front');
      V2.setHidden(castZone, true);
      V2.setHidden(targetLine, true);
      if (nodes.playArea) nodes.playArea.classList.remove('drop-target', 'drop-target-top');
      [nodes.playerFrontZone, nodes.opponentFrontZone, castZone].forEach(function (node) {
        if (node) node.classList.remove('drop-target');
      });
      page.querySelectorAll('.target-legal').forEach(function (node) { node.classList.remove('target-legal'); });
      page.querySelectorAll('.v2-hand-card.raised').forEach(function (node) { node.classList.remove('raised'); });
      ctx.raisedCardId = null;
    }

    function characterFromFront(sourceNode) {
      return Boolean(sourceNode && sourceNode.closest && sourceNode.closest('#v2-player-front, [data-drop-role="player-front"]'));
    }

    function inPlayArea(zone) {
      return Boolean(zone && zone.closest && zone.closest('[data-drop-role="play-area"]'));
    }

    function inBattleArea(zone) {
      return Boolean(zone && zone.closest && zone.closest('[data-drop-role="battle-area"]'));
    }

    function inPlayAreaTopHalf(point) {
      if (!point || !nodes.playArea) {
        return false;
      }
      const box = nodes.playArea.getBoundingClientRect();
      return point.y < box.top + box.height / 2;
    }

    function legalDrop(kind, id, zone, point, fromFront) {
      const grouped = kind === 'card' ? groupedPlay(id) : null;
      if (kind === 'card' && grouped) {
        const card = TABLE.cardByInstance(ctx.game, id);
        if (!mobileLayout.matches && (card && card.play_options || []).length) {
          return TABLE.playOptionAtPoint(card, grouped.entries, battleArea && battleArea.getBoundingClientRect(), point);
        }
        if (!dragTargetingEnabled()) {
          return inBattleArea(zone) ? grouped.entries[0] || null : null;
        }
        const byId = boardTargetEntries(grouped.entries);
        if (Object.keys(byId).length) {
          const entity = zone && zone.closest('[data-entity-id], [data-player-side]');
          const id = entity && (entity.getAttribute('data-entity-id') || entity.getAttribute('data-player-side') + ':player');
          return byId[TABLE.canonicalEntityId(id)] || null;
        }
        return inBattleArea(zone) ? grouped.entries[0] || null : null;
      }
      const attackEntry = kind === 'character' ? ctx.findAction('attack', { character_id: id }) : null;
      const front = fromFront != null ? fromFront : Boolean(ctx.dragState && ctx.dragState.fromFront);
      if (kind === 'character' && front && inPlayArea(zone)) {
        const aimed = ctx.actionsOf('attack', function (action) {
          return String(action.character_id) === String(id) && action.target_id;
        });
        if (aimed.length) {
          const entity = zone && zone.closest && zone.closest('[data-entity-id]');
          const eid = entity && TABLE.canonicalEntityId(entity.getAttribute('data-entity-id'));
          return aimed.filter(function (entry) {
            return TABLE.canonicalEntityId(entry.action.target_id) === eid;
          })[0] || null;
        }
        if (attackEntry) {
          return inPlayAreaTopHalf(point) ? attackEntry : null;
        }
      }
      return TABLE.legalDropFor(kind, grouped, zone, attackEntry, { fromFront: front });
    }

    function dropZoneAt(x, y) {
      ghost.style.pointerEvents = 'none';
      const el = document.elementFromPoint(x, y);
      ghost.style.pointerEvents = '';
      if (!el || !el.closest) return null;
      return el.closest('[data-drop-role], [data-entity-id], [data-player-side]');
    }

    function highlightDrop(zone, legal) {
      const playArea = nodes.playArea;
      const kind = ctx.dragState && ctx.dragState.kind;
      const fromFront = Boolean(ctx.dragState && ctx.dragState.fromFront);
      const centerEnabled = kind === 'character' && !fromFront;
      const wholeCenter = Boolean(centerEnabled && legal && inPlayArea(zone));
      const opponentHalf = Boolean(kind === 'character' && fromFront && legal);
      if (playArea) {
        playArea.classList.toggle('drop-target', wholeCenter);
        playArea.classList.toggle('drop-target-top', opponentHalf);
      }
      if (nodes.opponentFrontZone) {
        nodes.opponentFrontZone.classList.toggle('drop-target', opponentHalf);
      }
      [nodes.playerFrontZone, castZone].forEach(function (node) {
        if (node) node.classList.remove('drop-target');
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
        if (!(grouped.entries && grouped.entries.length)) {
          return;
        }
      }
      if (kind === 'character' && !ctx.findAction('attack', { character_id: id })) {
        return;
      }
      const mobileHandGesture = kind === 'card' && mobileLayout.matches && event.pointerType === 'touch';
      // Let the browser own horizontal panning; only an upward gesture starts a cast.
      if (!mobileHandGesture) event.preventDefault();
      if (typeof sourceNode.setPointerCapture === 'function') sourceNode.setPointerCapture(event.pointerId);
      const fromFront = kind === 'character' && characterFromFront(sourceNode);
      ctx.dragState = {
        pointerId: event.pointerId,
        kind: kind,
        id: id,
        sourceNode: sourceNode,
        startX: event.clientX,
        startY: event.clientY,
        moved: false,
        mobileHandGesture: mobileHandGesture,
        panIntent: false,
        fromFront: fromFront,
      };
      sourceNode.addEventListener('pointermove', onDragMove);
      sourceNode.addEventListener('pointerup', onDragEnd);
      sourceNode.addEventListener('pointercancel', onDragCancel);
    }

    function showDragUi() {
      const { kind, id, sourceNode, fromFront } = ctx.dragState;
      hideHoverPreview({ force: true });
      sourceNode.classList.add('dragging-source');
      ghost.innerHTML = sourceNode.innerHTML;
      ghost.classList.add('open');
      const ghostType = sourceNode.getAttribute && sourceNode.getAttribute('data-card-type');
      if (ghostType) ghost.setAttribute('data-card-type', ghostType);
      else ghost.removeAttribute('data-card-type');
      document.body.classList.add('v2-dragging');
      if (typeof TABLE.drawTutorialDragArrow === 'function') {
        TABLE.drawTutorialDragArrow(ctx);
      }
      if (kind === 'card' || (kind === 'character' && !fromFront)) {
        if (kind === 'card') {
          ctx.raisedCardId = id;
          sourceNode.classList.add('raised');
        }
        page.classList.add('v2-playing-card');
      } else if (kind === 'character' && fromFront) {
        page.classList.add('v2-playing-attack-front');
      }
      if (kind === 'card') {
        const byId = !dragTargetingEnabled() ? {} : boardTargetEntries(groupedPlay(id).entries);
        const hasTargets = Object.keys(byId).length > 0;
        page.classList.remove('v2-playing-card');
        page.classList.add('is-targeting');
        page.classList.toggle('is-drag-cast', !hasTargets);
        page.classList.add('is-drag-targeting');
        if (hasTargets) applyTargetClasses(byId);
        if (ctx.targetCopy) ctx.targetCopy.textContent = hasTargets
          ? '拖到高亮目标松开；松在其他位置回弹' : '拖到战斗大区松开出牌';
        if (ctx.targetCancel) V2.setHidden(ctx.targetCancel, true);
        if (ctx.targetMask) V2.setHidden(ctx.targetMask, false);
        const card = TABLE.cardByInstance(ctx.game, id);
        if (!mobileLayout.matches && (card && card.play_options || []).length) {
          showOptionZones(card, groupedPlay(id).entries);
          if (ctx.targetCopy) ctx.targetCopy.textContent = '拖入对应区域，选择要打出的效果';
        }
      }
    }

    function cleanupDrag() {
      if (!ctx.dragState) return null;
      ctx.dragState.sourceNode.classList.remove('dragging-source', 'raised');
      ctx.dragState.sourceNode.removeEventListener('pointermove', onDragMove);
      ctx.dragState.sourceNode.removeEventListener('pointerup', onDragEnd);
      ctx.dragState.sourceNode.removeEventListener('pointercancel', onDragCancel);
      ghost.classList.remove('open');
      ghost.innerHTML = '';
      ghost.removeAttribute('data-card-type');
      document.body.classList.remove('v2-dragging');
      if (typeof TABLE.drawTutorialDragArrow === 'function') {
        TABLE.drawTutorialDragArrow(ctx);
      }
      highlightDrop(null, false);
      V2.setHidden(previewBox, true);
      const state = ctx.dragState;
      clearOptionZones();
      ctx.dragState = null;
      if (page.classList.contains('is-drag-targeting')) endBoardTargetSelect();
      return state;
    }

    function onDragMove(event) {
      if (!ctx.dragState || event.pointerId !== ctx.dragState.pointerId) return;
      const distance = Math.hypot(event.clientX - ctx.dragState.startX, event.clientY - ctx.dragState.startY);
      if (!ctx.dragState.moved) {
        if (distance <= CLICK_DISTANCE) return;
        if (ctx.dragState.mobileHandGesture) {
          const dx = event.clientX - ctx.dragState.startX;
          const dy = event.clientY - ctx.dragState.startY;
          if (ctx.dragState.panIntent) return;
          if (Math.abs(dx) >= Math.abs(dy) || dy > 0) {
            ctx.dragState.panIntent = true;
            return;
          }
          if (-dy < Math.abs(dx) * 1.2) return;
        }
        ctx.dragState.moved = true;
        showDragUi();
      }
      moveGhost(event.clientX, event.clientY);
      const zone = dropZoneAt(event.clientX, event.clientY);
      const legal = legalDrop(ctx.dragState.kind, ctx.dragState.id, zone, { x: event.clientX, y: event.clientY });
      highlightDrop(zone, Boolean(legal));
      if (optionZones) optionZones.querySelectorAll('[data-play-option]').forEach(function (node) {
        node.classList.toggle('is-active', Boolean(legal && legal.action.option_id === node.getAttribute('data-play-option')));
      });
      if (ctx.dragState.kind === 'character') {
        showPreview(ctx.findAction('attack', { character_id: ctx.dragState.id }));
      } else {
        showPreview(legal || ctx.playEntries(ctx.dragState.id)[0] || null);
        V2.setHidden(targetLine, true);
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
      ignoreHandToggleClick = true;
      if (state.panIntent) {
        ctx.suppressClick = true;
        window.setTimeout(function () { ctx.suppressClick = false; }, 0);
        clearCastUi();
        ctx.flushPendingEnvelope();
        return;
      }
      const distance = Math.hypot(event.clientX - state.startX, event.clientY - state.startY);
      if (!state.moved || distance < CLICK_DISTANCE) {
        ctx.suppressClick = true;
        window.setTimeout(function () { ctx.suppressClick = false; }, 0);
        handleActivate(state.kind, state.id, state.sourceNode, event.pointerType);
        clearCastUi();
        ctx.flushPendingEnvelope();
        return;
      }
      ctx.suppressClick = true;
      window.setTimeout(function () { ctx.suppressClick = false; }, 0);
      const zone = dropZoneAt(event.clientX, event.clientY);
      const legal = legalDrop(state.kind, state.id, zone, { x: event.clientX, y: event.clientY }, state.fromFront);
      clearCastUi();
      const directTarget = state.kind === 'card' && dragTargetingEnabled()
        && Object.keys(boardTargetEntries(groupedPlay(state.id).entries)).length > 0;
      if (!legal) {
        const message = directTarget ? '未放到合法目标，已回弹。'
          : state.kind === 'character' && state.fromFront ? '未放到对手区域，已回弹。'
          : state.kind === 'card' ? '未放到战斗大区，已回弹。' : '未放到中央区域，已回弹。';
        ctx.showToast(message);
        ctx.flushPendingEnvelope();
        return;
      }
      if (mobileLayout.matches && state.kind === 'card') setMobileHandOpen(false);
      const optionCard = state.kind === 'card' && TABLE.cardByInstance(ctx.game, state.id);
      if (!mobileLayout.matches && (optionCard && optionCard.play_options || []).length) {
        const selected = groupedPlay(state.id).entries.filter(function (entry) { return entry.action.option_id === legal.action.option_id; });
        if (selected.length > 1) startChoiceSelect(selected, optionCard);
        else ctx.submitEntry(legal);
      } else if (state.kind === 'character' || directTarget) {
        ctx.submitEntry(legal);
      } else {
        beginPlayCard(state.id, null);
      }
      ctx.flushPendingEnvelope();
    }

    function isBoardTargetId(value) {
      const token = TABLE.canonicalEntityId(value);
      if (!token || token.indexOf(':') < 0) return '';
      const id = token.split(':')[1];
      if (!id) return '';
      return token;
    }

    function boardTargetEntries(entries) {
      const map = {};
      (entries || []).forEach(function (entry) {
        const actionId = isBoardTargetId(entry.action && entry.action.target_id);
        if (actionId) {
          map[actionId] = entry;
          return;
        }
        ((entry.interaction && entry.interaction.target_ids) || []).forEach(function (raw) {
          const id = isBoardTargetId(raw);
          if (id && !map[id] && (entry.action && entry.action.target_id)) {
            map[id] = entry;
          }
        });
      });
      return map;
    }

    function applyTargetClasses(byId) {
      const ids = Object.keys(byId || {});
      page.querySelectorAll('.v2-character-card, .v2-player-hud').forEach(function (node) {
        const eid = TABLE.canonicalEntityId(node.getAttribute('data-entity-id'));
        const legal = Boolean(eid && ids.indexOf(eid) >= 0);
        node.classList.toggle('target-legal', legal);
        if (node.classList.contains('v2-character-card')) {
          node.classList.toggle('target-dim', !legal);
        }
      });
    }

    function markRaisedCard(cardId, on) {
      if (on) ctx.raisedCardId = cardId || null;
      else if (!cardId || String(ctx.raisedCardId) === String(cardId)) ctx.raisedCardId = null;
      page.querySelectorAll('.v2-hand-card.raised').forEach(function (node) {
        if (!on || String(node.getAttribute('data-card-id')) !== String(cardId)) {
          node.classList.remove('raised');
        }
      });
      if (on && cardId) {
        const handCard = page.querySelector('.v2-hand-card[data-card-id="' + cardId + '"]');
        if (handCard) handCard.classList.add('raised');
      }
      if (!ctx.dragState) {
        page.classList.remove('v2-playing-card');
      }
    }

    function endBoardTargetSelect() {
      if (ctx.localChoice && ctx.localChoice.kind === 'board-target') {
        markRaisedCard(ctx.localChoice.cardId, false);
        ctx.localChoice = null;
      }
      page.classList.remove('is-targeting', 'is-drag-targeting', 'is-drag-cast');
      if (ctx.targetCancel) V2.setHidden(ctx.targetCancel, false);
      page.querySelectorAll('.target-legal, .target-dim').forEach(function (node) {
        node.classList.remove('target-legal');
        node.classList.remove('target-dim');
      });
      if (ctx.targetMask) V2.setHidden(ctx.targetMask, true);
    }

    function reapplyBoardTargetSelect() {
      if (!ctx.localChoice || ctx.localChoice.kind !== 'board-target') return;
      const cardId = ctx.localChoice.cardId;
      const grouped = cardId ? groupedPlay(cardId) : { entries: [] };
      const byId = boardTargetEntries(grouped.entries || []);
      if (!Object.keys(byId).length) {
        endBoardTargetSelect();
        return;
      }
      ctx.localChoice.byId = byId;
      page.classList.add('is-targeting');
      applyTargetClasses(byId);
      markRaisedCard(cardId, true);
      if (ctx.targetMask) V2.setHidden(ctx.targetMask, false);
    }

    function startBoardTargetSelect(entries, card) {
      const byId = boardTargetEntries(entries);
      const ids = Object.keys(byId);
      if (!ids.length) {
        ctx.submitEntry(entries[0]);
        return;
      }
      hideHoverPreview({ force: true });
      ctx.localChoice = { kind: 'board-target', byId: byId, cardId: card && card.instance_id };
      page.classList.add('is-targeting');
      applyTargetClasses(byId);
      markRaisedCard(card && card.instance_id, true);
      if (ctx.targetCopy) {
        ctx.targetCopy.textContent = (card && card.name ? '选择「' + card.name + '」的目标' : '选择一个目标');
      }
      if (ctx.targetMask) V2.setHidden(ctx.targetMask, false);
    }

    function startChoiceSelect(entries, card) {
      hideHoverPreview({ force: true });
      ctx.localChoice = { kind: 'choice-target', cardId: card && card.instance_id };
      markRaisedCard(card && card.instance_id, true);
      ctx.openChoice({
        title: card && card.name ? '选择「' + card.name + '」的目标' : '选择一个目标',
        choices: entries.map(function (entry, index) {
          const tid = entry.action && entry.action.target_id;
          const found = TABLE.cardByInstance(ctx.game, tid);
          return {
            id: String(index),
            label: entry.label || (found && found.name) || tid || ('目标 ' + (index + 1)),
            card: found,
            entry: entry,
          };
        }),
        onPick: function (picked) {
          markRaisedCard(card && card.instance_id, false);
          ctx.localChoice = null;
          if (picked && picked.entry) ctx.submitEntry(picked.entry);
        },
      });
    }

    function beginPlayCard(cardId, zone) {
      const grouped = groupedPlay(cardId);
      const list = grouped.entries || [];
      if (!list.length) {
        ctx.showToast('没有合法动作。');
        return;
      }
      const card = TABLE.cardByInstance(ctx.game, cardId);
      if ((card && card.play_options || []).length) {
        ctx.openChoice({title: '选择「' + card.name + '」的效果', choices: list.map(function (entry, index) {
          return {id: String(index), label: entry.label, entry: entry};
        }), onPick: function (choice) { if (choice) ctx.submitEntry(choice.entry); }});
        return;
      }
      const entityNode = zone && zone.closest ? zone.closest('[data-entity-id]') : null;
      const droppedOn = TABLE.canonicalEntityId(entityNode && entityNode.getAttribute && entityNode.getAttribute('data-entity-id'));
      if (droppedOn) {
        const direct = list.find(function (entry) { return TABLE.entryMatchesTarget(entry, droppedOn); });
        if (direct) {
          ctx.submitEntry(direct);
          return;
        }
      }
      const byId = boardTargetEntries(list);
      const needsBoardPick = Object.keys(byId).length > 0 && list.some(function (entry) {
        return Boolean(isBoardTargetId(entry.action && entry.action.target_id));
      });
      if (needsBoardPick) {
        startBoardTargetSelect(list, TABLE.cardByInstance(ctx.game, cardId));
        return;
      }
      const choiceKeys = [];
      list.forEach(function (entry) {
        const tid = entry.action && entry.action.target_id;
        if (tid && !isBoardTargetId(tid) && choiceKeys.indexOf(tid) < 0) choiceKeys.push(tid);
      });
      if (choiceKeys.length > 1) {
        startChoiceSelect(list, TABLE.cardByInstance(ctx.game, cardId));
        return;
      }
      ctx.submitEntry(list[0]);
    }

    function resolveAndSubmit(entry, entries) {
      const list = entries || (entry ? [entry] : []);
      if (!list.length) {
        ctx.showToast('没有合法动作。');
        return;
      }
      beginPlayCard((entry && entry.action && entry.action.card_id) || (list[0].action && list[0].action.card_id), null);
    }
    ctx.resolveAndSubmit = resolveAndSubmit;
    ctx.endBoardTargetSelect = endBoardTargetSelect;
    ctx.reapplyBoardTargetSelect = reapplyBoardTargetSelect;

    function handleActivate(kind, id, sourceNode, pointerType) {
      if (ctx.playingEvents && ctx.settlePresentation) ctx.settlePresentation();
      if (ctx.game && ctx.game.phase === 'mulligan' && kind === 'card') {
        ctx.toggleMulligan(id);
        return;
      }
      const payload = payloadFromNode(sourceNode) || (
        kind === 'card'
          ? { kind: 'card', card: TABLE.cardByInstance(ctx.game, id) || { name: '卡牌', description: '' } }
          : { kind: 'character', character: V2.characterFallback(id) }
      );
      showHoverPreview(payload, sourceNode, { pinned: isTouchLike(pointerType), pointerType: pointerType });
    }

    function bindPointer(container, kindAttr, idAttr) {
      container.addEventListener('pointerdown', function (event) {
        const source = event.target.closest('[' + kindAttr + ']');
        if (!source || ctx.busy || event.target.closest('.v2-inspect, .v2-awaken-btn, [data-inspect-character]')) return;
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
        const awakenBtn = event.target.closest('[data-ultimate-id], [data-awaken-id]');
        if (awakenBtn) {
          event.preventDefault();
          ctx.submitEntry(ctx.findAction('ultimate', {
            character_id: awakenBtn.getAttribute('data-ultimate-id') || awakenBtn.getAttribute('data-awaken-id'),
          }));
          return;
        }
        const inspect = event.target.closest('[data-inspect-character]');
        if (inspect) return;
        const source = event.target.closest('[' + idAttr + ']');
        if (!source || ctx.busy || ctx.dragState) return;
        handleActivate(source.getAttribute(kindAttr) === 'character' ? 'character' : 'card', source.getAttribute(idAttr), source, event.pointerType);
      });
    }

    page.addEventListener('pointerover', function (event) {
      if (event.pointerType === 'touch' || ctx.dragState || ctx.busy) return;
      if (ctx.game && ctx.game.phase === 'mulligan') return;
      const source = event.target.closest('[data-drag-kind], [data-public-copy], [data-character-id], .v2-hand-card');
      if (!source || source.classList.contains('v2-hidden-card')) return;
      const payload = payloadFromNode(source);
      if (payload) showHoverPreview(payload, source, { pointerType: event.pointerType || 'mouse' });
    });
    page.addEventListener('pointerout', function (event) {
      const source = event.target.closest('[data-drag-kind], [data-public-copy], [data-character-id], .v2-hand-card');
      if (!source) return;
      if (source.contains(event.relatedTarget)) return;
      hideHoverPreview();
    });
    page.addEventListener('focusin', function (event) {
      const source = event.target.closest('[data-drag-kind], [data-public-copy], [data-character-id], .v2-hand-card');
      const payload = source ? payloadFromNode(source) : null;
      if (payload) showHoverPreview(payload, source, { pointerType: 'keyboard' });
    });
    page.addEventListener('focusout', function (event) {
      const source = event.target.closest('[data-drag-kind], [data-public-copy], [data-character-id], .v2-hand-card');
      if (!source || source.contains(event.relatedTarget)) return;
      hideHoverPreview();
    });

    page.addEventListener('click', function (event) {
      const inspect = event.target.closest('[data-inspect-character]');
      if (!inspect || ctx.busy) return;
      const isOpponent = inspect.dataset.opponent === 'true';
      const character = TABLE.characterMap(TABLE.sideState(ctx.game, isOpponent ? ctx.opponent() : ctx.viewer()))[inspect.dataset.inspectCharacter];
      if (character) showCharacterDetail(character, isOpponent, inspect.closest('[data-character-id]'));
    });
    function opponentPublicCard(button) {
      return (TABLE.sideState(ctx.game, ctx.opponent()).hand || []).find(function (item) {
        return item.instance_id === button.dataset.publicCopy;
      });
    }
    nodes.opponentHand.addEventListener('click', function (event) {
      const button = event.target.closest('[data-public-copy]');
      if (!button) return;
      const card = opponentPublicCard(button);
      if (card) showCardDetail(card, [], button);
    });
    nodes.opponentHand.addEventListener('mouseover', function (event) {
      const button = event.target.closest('[data-public-copy]');
      if (!button || ctx.hoverPinned) return;
      const card = opponentPublicCard(button);
      if (card) showHoverPreview({ kind: 'card', card: card }, button, { pointerType: 'mouse' });
    });

    bindPointer(nodes.playerBench, 'data-drag-kind', 'data-character-id');
    bindPointer(nodes.playerFront, 'data-drag-kind', 'data-character-id');
    bindPointer(nodes.playerHand, 'data-drag-kind', 'data-card-id');
    if (nodes.mulliganRail) bindPointer(nodes.mulliganRail, 'data-drag-kind', 'data-card-id');
    function inspectOpponent(event) {
      const source = event.target.closest('[data-character-id]');
      if (!source || ctx.busy) return;
      if (event.target.closest('.v2-inspect')) return;
      const character = TABLE.characterMap(TABLE.sideState(ctx.game, ctx.opponent()))[source.getAttribute('data-character-id')];
      if (character) showCharacterDetail(character, true, source);
    }
    nodes.opponentBench.addEventListener('click', inspectOpponent);
    nodes.opponentFront.addEventListener('click', inspectOpponent);

    nodes.endTurnBtn.addEventListener('click', function () { ctx.submitEntry(ctx.findAction('end_turn')); });
    nodes.concedeBtn.addEventListener('click', function () { ctx.submitEntry(ctx.findAction('concede')); });
    nodes.mulliganToggle.addEventListener('click', function () {
      const collapsed = ctx.page.classList.toggle('mulligan-collapsed');
      nodes.mulliganToggle.setAttribute('aria-expanded', String(!collapsed));
      nodes.mulliganToggle.textContent = collapsed ? '展开换牌' : '收起换牌';
    });
    nodes.mulliganBtn.addEventListener('click', ctx.submitMulligan);
    nodes.playerDiscard.addEventListener('click', function () { openPile('我方弃牌', TABLE.sideState(ctx.game, ctx.viewer()).discard || []); });
    nodes.opponentDiscard.addEventListener('click', function () { openPile('对手弃牌', TABLE.sideState(ctx.game, ctx.opponent()).discard || []); });
    nodes.playerDeck.addEventListener('click', function () {
      const mine = TABLE.sideState(ctx.game, ctx.viewer());
      const cards = Array.isArray(mine.visible_deck) ? mine.visible_deck.map(function (card, index) {
        return Object.assign({}, card, { name: (index + 1) + '. ' + card.name });
      }) : [{ name: '牌库 ' + (mine.deck_count || 0) + ' 张', description: '牌序保密。' }];
      openPile(Array.isArray(mine.visible_deck) ? '我方牌库 · 从顶到底' : '我方牌库', cards);
    });
    nodes.opponentDeck.addEventListener('click', function () {
      openPile('对手牌库', [{ name: '牌库 ' + (TABLE.sideState(ctx.game, ctx.opponent()).deck_count || 0) + ' 张', description: '牌序保密。' }]);
    });
    choiceOverlay.addEventListener('click', function (event) {
      if (event.target === choiceOverlay) {
        markRaisedCard(ctx.localChoice && ctx.localChoice.cardId, false);
        ctx.localChoice = null;
        closeOverlay(choiceOverlay);
      }
    });
    pileModal.addEventListener('click', function (event) { if (event.target === pileModal) closeOverlay(pileModal); });
    cardPreview.addEventListener('click', function (event) {
      if (event.target.closest('[data-hover-close]') || event.target === cardPreview) {
        hideHoverPreview({ force: true });
      }
    });
    function cancelBoardTarget() {
      endBoardTargetSelect();
      ctx.flushPendingEnvelope();
    }
    page.addEventListener('click', function (event) {
      if (!ctx.localChoice || ctx.localChoice.kind !== 'board-target') return;
      if (ctx.suppressClick) {
        ctx.suppressClick = false;
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      if (event.target.closest('#v2-card-preview')) return;
      if (event.target.closest('#v2-target-cancel')) {
        event.preventDefault();
        event.stopPropagation();
        cancelBoardTarget();
        return;
      }
      if (event.target.closest('.v2-target-bar')) return;
      const legal = event.target.closest('.v2-character-card.target-legal, .v2-player-hud.target-legal');
      if (legal) {
        event.preventDefault();
        event.stopPropagation();
        const eid = TABLE.canonicalEntityId(legal.getAttribute('data-entity-id'))
          || (legal.getAttribute('data-player-side') ? legal.getAttribute('data-player-side') + ':player' : '');
        const entry = ctx.localChoice.byId[eid];
        endBoardTargetSelect();
        if (entry) ctx.submitEntry(entry);
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      cancelBoardTarget();
    }, true);
    if (ctx.targetCancel) {
      ctx.targetCancel.addEventListener('click', function (event) {
        event.preventDefault();
        cancelBoardTarget();
      });
    }
    document.addEventListener('keydown', function (event) {
      if (event.key !== 'Escape') return;
      if (ctx.localChoice && ctx.localChoice.kind === 'board-target') {
        cancelBoardTarget();
        return;
      }
      hideHoverPreview({ force: true });
    });
    document.addEventListener('pointerdown', function (event) {
      if (!cardPreview.classList.contains('is-pinned') && !cardPreview.classList.contains('is-touch')) return;
      if (event.target.closest('#v2-card-preview') || event.target.closest('[data-drag-kind], [data-public-copy], [data-character-id], .v2-hand-card')) return;
      hideHoverPreview({ force: true });
    });

    ctx.openChoice = openChoice;
    ctx.closeOverlay = closeOverlay;
    ctx.showCardDetail = showCardDetail;
    ctx.showCharacterDetail = showCharacterDetail;
    ctx.showHoverPreview = showHoverPreview;
    ctx.hideHoverPreview = hideHoverPreview;
    ctx.showPreview = showPreview;
    ctx.clearCastUi = clearCastUi;
    ctx.handleActivate = handleActivate;
    ctx.legalDrop = legalDrop;
    ctx.groupedPlay = groupedPlay;
  }

  TABLE.bindInteraction = bindInteraction;
}(window));
