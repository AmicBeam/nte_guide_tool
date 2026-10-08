(function (root) {
  const V2 = root.NTE_V2;
  const TABLE = root.NTE_V2_TABLE = root.NTE_V2_TABLE || {};

  function sameOrder(left, right) {
    return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
      left.every(function (id, index) { return String(id) === String(right[index]); });
  }

  function moveCard(draft, id, row, index) {
    const next = Object.assign({}, draft, {topIds: draft.topIds.slice(), bottomIds: (draft.bottomIds || []).slice()});
    const fromTop = next.topIds.indexOf(id);
    const fromBottom = next.bottomIds.indexOf(id);
    if ((fromTop < 0 && fromBottom < 0) || (row !== 'top' && row !== 'bottom')) return next;
    if (row === 'bottom' && fromBottom >= 0 && !Number.isInteger(index)) return next;
    if (fromTop >= 0) next.topIds.splice(fromTop, 1);
    if (fromBottom >= 0) next.bottomIds.splice(fromBottom, 1);
    // Preserve a pending old single-bottom snapshot; new choices have no swap.
    if (row === 'bottom' && draft.maxBottom === 1 && next.bottomIds.length) {
      const displaced = next.bottomIds.pop();
      next.topIds.splice(Math.max(0, fromTop), 0, displaced);
    }
    const target = row === 'top' ? next.topIds : next.bottomIds;
    const at = Number.isInteger(index) ? Math.max(0, Math.min(index, target.length)) : target.length;
    target.splice(at, 0, id);
    return next;
  }

  function choiceKey(pending) {
    return pending ? [pending.side, pending.kind, pending.min_bottom || 0, pending.max_bottom,
      (pending.choices || []).map(function (c) { return c.id; }).join('|')].join(':') : '';
  }

  function findEntry(game, draft) {
    return V2.findLegalEntry(game, function (action) {
      const bottoms = Array.isArray(action.bottom_card_ids) ? action.bottom_card_ids :
        (action.bottom_card_id ? [action.bottom_card_id] : []);
      return action.type === 'choose_deck_order' && sameOrder(action.card_ids, draft.topIds) &&
        sameOrder(bottoms, draft.bottomIds || []);
    });
  }

  function effectText(card, name) {
    const text = String(card && card.description || '');
    const marker = '【' + name + '】';
    const start = text.indexOf(marker);
    if (start < 0) return '';
    return text.slice(start + marker.length).split(/\n|【/)[0].trim();
  }

  function bindDivination(ctx, cardMarkup) {
    const overlay = ctx.choiceOverlay;
    let drag = null;

    function clearDrag() {
      if (drag && drag.ghost) drag.ghost.remove();
      if (drag && overlay.hasPointerCapture && overlay.hasPointerCapture(drag.pointerId)) {
        overlay.releasePointerCapture(drag.pointerId);
      }
      overlay.querySelectorAll('.is-divination-dragging, .is-divination-over').forEach(function (node) {
        node.classList.remove('is-divination-dragging', 'is-divination-over');
      });
      drag = null;
    }

    function render(message, focusId) {
      const state = ctx.divinationDraft;
      if (!state) return;
      const byId = state.byId;
      function card(id, position) {
        const choice = byId[id];
        if (!choice || !choice.card) return '';
        return '<div class="v2-divination-card" tabindex="0" role="button" data-divination-id="' + V2.escapeAttr(id) +
          '" aria-label="' + V2.escapeAttr(choice.card.name + '，可拖动排序或移至另一行') + '">' +
          '<span class="v2-divination-position">' + position + '</span>' +
          '<button type="button" class="v2-divination-detail" data-divination-detail="' + V2.escapeAttr(id) +
          '" aria-label="查看' + V2.escapeAttr(choice.card.name) + '详情">详情</button>' +
          cardMarkup(choice.card).replace('</article>',
            '<p class="v2-divination-benefit is-divination is-current"><strong>占卜效果：</strong>' +
            V2.escapeHtml(effectText(choice.card, '占卜效果') || '无占卜效果') + '</p></article>') + '</div>';
      }
      overlay.innerHTML = '<div class="v2-choice-dialog v2-divination-dialog" role="dialog" aria-modal="true" aria-labelledby="v2-divination-title">' +
        '<header><h2 id="v2-divination-title">占卜</h2><p class="subtle">两行均可拖动排序；第二行可放多张牌。' + (state.minBottom ? '谕石至少置底 1 张。' : '可以不置底。') + '</p></header>' +
        '<section class="v2-divination-zone" data-divination-row="top" aria-label="牌库顶">' +
        '<div class="v2-divination-zone-label"><strong>牌库顶</strong><span>从左到右依次抽取</span></div>' +
        '<div class="v2-divination-row">' + state.topIds.map(function (id, index) { return card(id, index + 1); }).join('') +
        (state.topIds.length ? '' : '<span class="v2-divination-placeholder">拖到这里放回牌库顶</span>') + '</div></section>' +
        '<section class="v2-divination-zone" data-divination-row="bottom" aria-label="牌库底">' +
        '<div class="v2-divination-zone-label"><strong>牌库底</strong><span>基础形态按从左到右结算占卜效果</span></div>' +
        '<div class="v2-divination-row">' + (state.bottomIds.length ? state.bottomIds.map(function (id, index) { return card(id, '底' + (index + 1)); }).join('') :
          '<span class="v2-divination-placeholder">' + (state.minBottom ? '至少拖入 1 张牌' : '可拖入多张牌，也可以留空') + '</span>') + '</div></section>' +
        '<footer><span class="v2-divination-status" role="status" aria-live="polite">' + V2.escapeHtml(message || '') +
        '</span><button class="primary-btn" id="v2-divination-confirm" type="button">确定</button></footer></div>';
      overlay.classList.add('open');
      overlay.classList.add('is-divination');
      overlay.setAttribute('aria-hidden', 'false');
      const confirm = overlay.querySelector('#v2-divination-confirm');
      confirm.disabled = Boolean(ctx.busy || !findEntry(ctx.game, state));
      if (focusId) {
        const node = Array.from(overlay.querySelectorAll('[data-divination-id]')).find(function (item) {
          return item.getAttribute('data-divination-id') === focusId;
        });
        if (node) node.focus({ preventScroll: true });
      }
    }

    function move(id, row, index) {
      const state = ctx.divinationDraft;
      if (!state) return;
      const next = moveCard(state, id, row, index);
      state.topIds = next.topIds;
      state.bottomIds = next.bottomIds;
      const name = state.byId[id].card.name;
      const message = '已调整「' + name + '」在' + (row === 'bottom' ? '牌库底' : '牌库顶') + '的顺序。';
      render(message, id);
    }

    ctx.openDivinationChoice = function (pending) {
      clearDrag();
      ctx.hideHoverPreview({ force: true });
      const choices = (pending.choices || []).filter(function (choice) { return choice.card && !choice.card.hidden; });
      const byId = {};
      choices.forEach(function (choice) { byId[String(choice.id)] = choice; });
      const team = ((ctx.game || {}).sides || {})[pending.side] || {};
      const heiyu = (team.characters || []).find(function (character) { return character.id === 'heiyu'; });
      ctx.divinationDraft = { key: choiceKey(pending), topIds: choices.map(function (c) { return String(c.id); }), bottomIds: [], minBottom: Number(pending.min_bottom || 0), maxBottom: pending.max_bottom == null ? choices.length : Number(pending.max_bottom), byId: byId, witch: Boolean(heiyu && heiyu.awakened) };
      ctx.localChoice = { kind: 'server-divination', key: choiceKey(pending) };
      render();
    };

    overlay.addEventListener('pointerdown', function (event) {
      if (!ctx.divinationDraft || !ctx.localChoice || ctx.localChoice.kind !== 'server-divination' ||
          ctx.busy || event.button > 0 || event.target.closest('button')) return;
      const node = event.target.closest('[data-divination-id]');
      if (!node) return;
      event.preventDefault();
      drag = { id: node.getAttribute('data-divination-id'), node: node, pointerId: event.pointerId,
        startX: event.clientX, startY: event.clientY, ghost: null };
      if (overlay.setPointerCapture) overlay.setPointerCapture(event.pointerId);
    });

    overlay.addEventListener('pointermove', function (event) {
      if (!drag || event.pointerId !== drag.pointerId) return;
      event.preventDefault();
      if (!drag.ghost && Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < 7) return;
      if (!drag.ghost) {
        const rect = drag.node.getBoundingClientRect();
        drag.ghost = drag.node.cloneNode(true);
        drag.ghost.classList.add('v2-divination-ghost');
        drag.ghost.removeAttribute('data-divination-id');
        drag.ghost.setAttribute('aria-hidden', 'true');
        drag.ghost.style.width = rect.width + 'px';
        drag.ghost.style.height = rect.height + 'px';
        document.body.appendChild(drag.ghost);
        drag.node.classList.add('is-divination-dragging');
      }
      drag.ghost.style.left = event.clientX + 'px';
      drag.ghost.style.top = event.clientY + 'px';
      const hit = document.elementFromPoint(event.clientX, event.clientY);
      const zone = hit && hit.closest('[data-divination-row]');
      overlay.querySelectorAll('[data-divination-row]').forEach(function (node) {
        node.classList.toggle('is-divination-over', node === zone);
      });
    });

    overlay.addEventListener('pointerup', function (event) {
      if (!drag || event.pointerId !== drag.pointerId) return;
      const id = drag.id;
      const moved = Boolean(drag.ghost);
      const hit = document.elementFromPoint(event.clientX, event.clientY);
      const zone = hit && hit.closest('[data-divination-row]');
      let index = 0;
      if (zone) {
        const cards = Array.from(zone.querySelectorAll('[data-divination-id]')).filter(function (node) {
          return node.getAttribute('data-divination-id') !== id;
        });
        index = cards.findIndex(function (node) {
          const rect = node.getBoundingClientRect();
          return event.clientX < rect.left + rect.width / 2;
        });
        if (index < 0) index = cards.length;
      }
      clearDrag();
      if (moved && zone) move(id, zone.getAttribute('data-divination-row'), index);
    });
    overlay.addEventListener('pointercancel', clearDrag);

    overlay.addEventListener('keydown', function (event) {
      if (!ctx.divinationDraft || ctx.busy || event.target.closest('button')) return;
      const node = event.target.closest('[data-divination-id]');
      if (!node) return;
      const id = node.getAttribute('data-divination-id');
      const state = ctx.divinationDraft;
      const bottom = state.bottomIds.indexOf(id) >= 0;
      const row = bottom ? 'bottom' : 'top';
      const ids = bottom ? state.bottomIds : state.topIds;
      const index = ids.indexOf(id);
      if (event.key === 'ArrowDown' && !bottom) { event.preventDefault(); move(id, 'bottom'); }
      else if (event.key === 'ArrowUp' && bottom) { event.preventDefault(); move(id, 'top', state.topIds.length); }
      else if (event.key === 'ArrowLeft' && index > 0) { event.preventDefault(); move(id, row, index - 1); }
      else if (event.key === 'ArrowRight' && index >= 0 && index < ids.length - 1) { event.preventDefault(); move(id, row, index + 1); }
    });

    overlay.addEventListener('click', function (event) {
      if (!ctx.localChoice || ctx.localChoice.kind !== 'server-divination') return;
      const detail = event.target.closest('[data-divination-detail]');
      if (detail) {
        const choice = ctx.divinationDraft.byId[detail.getAttribute('data-divination-detail')];
        if (choice) ctx.showCardDetail(choice.card, null, detail);
        return;
      }
      if (!event.target.closest('#v2-divination-confirm') || ctx.busy) return;
      const entry = findEntry(ctx.game, ctx.divinationDraft);
      if (!entry) { ctx.showToast('当前排列已失效，请刷新后重试。'); return; }
      clearDrag();
      ctx.localChoice = null;
      ctx.divinationDraft = null;
      ctx.closeOverlay(overlay);
      ctx.submitEntry(entry);
    });
  }

  TABLE.sameDeckOrder = sameOrder;
  TABLE.moveDivinationCard = moveCard;
  TABLE.divinationChoiceKey = choiceKey;
  TABLE.findDivinationEntry = findEntry;
  TABLE.bindDivination = bindDivination;
}(window));
