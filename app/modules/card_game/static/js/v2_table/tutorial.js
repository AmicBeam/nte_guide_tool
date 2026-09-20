(function (root) {
  const V2 = root.NTE_V2;
  const TABLE = root.NTE_V2_TABLE = root.NTE_V2_TABLE || {};

  function tutorial(game) {
    return (game && game.tutorial) || null;
  }

  function resolveNode(ctx, token) {
    if (!token || !ctx.page) {
      return null;
    }
    const raw = String(token);
    if (raw === 'escalation') return ctx.page.querySelector('#v2-escalation');
    if (raw === 'ultimate_count') return ctx.nodes.ultimate;
    if (raw === 'round') return ctx.nodes.roundLabel;
    if (raw === 'end_turn') {
      return ctx.nodes.endTurnBtn;
    }
    if (raw === 'play_area') {
      return ctx.nodes.playArea;
    }
    if (raw === 'mulligan') {
      return document.getElementById('v2-mulligan-stage') || document.getElementById('v2-mulligan-btn');
    }
    if (raw.indexOf('deck:') === 0) {
      const side = raw.slice(5);
      return side === ctx.viewer ? ctx.nodes.playerDeck : ctx.nodes.opponentDeck;
    }
    if (raw === 'turn') {
      return ctx.nodes.waitChip || ctx.nodes.turnCopy || ctx.nodes.roundLabel;
    }
    if (raw === 'log') {
      return ctx.nodes.logList || ctx.nodes.turnCopy;
    }
    if (raw.indexOf('ap:') === 0) {
      const side = raw.slice(3);
      return side === ctx.viewer ? ctx.nodes.playerAp : ctx.nodes.opponentAp;
    }
    if (raw.indexOf('player_hp:') === 0) {
      const side = raw.slice(10);
      return document.querySelector('[data-player-side="' + CSS.escape(side) + '"]');
    }
    if (raw.indexOf('bench:') === 0) {
      const side = raw.slice(6);
      return side === ctx.viewer ? ctx.nodes.playerBench : ctx.nodes.opponentBench;
    }
    if (raw.indexOf('zone:') === 0) {
      const parts = raw.split(':');
      const side = parts[1];
      if (parts[2] === 'front') {
        return side === ctx.viewer ? ctx.nodes.playerFrontZone : ctx.nodes.opponentFrontZone;
      }
      if (parts[2] === 'cast') {
        return ctx.castZone;
      }
    }
    if (raw.indexOf('hand:') === 0) {
      return ctx.page.querySelector('[data-card-id="' + CSS.escape(raw.slice(5)) + '"]');
    }
    if (raw.indexOf('ultimate:') === 0) {
      const parts = raw.split(':');
      const character = resolveNode(ctx, 'character:' + parts[1] + ':' + parts[2]);
      return character && character.querySelector('[data-ultimate-id]');
    }
    if (raw.indexOf('character:') === 0) {
      const parts = raw.split(':');
      return document.querySelector('[data-entity-id="' + CSS.escape(parts[1] + ':' + parts[2]) + '"]');
    }
    return document.querySelector('[data-tutorial-id="' + CSS.escape(raw) + '"]');
  }

  function visibleRect(node) {
    if (!node || typeof node.getBoundingClientRect !== 'function') {
      return null;
    }
    const rect = node.getBoundingClientRect();
    const style = window.getComputedStyle(node);
    if (rect.width < 8 || rect.height < 8 || style.display === 'none' || style.visibility === 'hidden') {
      return null;
    }
    return rect;
  }

  function clearCues() {
    document.querySelectorAll('.v2-tutorial-focus-frame, .v2-tutorial-drag-cue, .v2-tutorial-click-cue').forEach(function (node) {
      node.remove();
    });
  }

  function addFocusFrame(rect) {
    const frame = document.createElement('div');
    frame.className = 'v2-tutorial-focus-frame';
    frame.style.left = Math.max(4, rect.left - 6) + 'px';
    frame.style.top = Math.max(4, rect.top - 6) + 'px';
    frame.style.width = Math.max(24, rect.width + 12) + 'px';
    frame.style.height = Math.max(24, rect.height + 12) + 'px';
    document.body.appendChild(frame);
  }

  function addDragCue(fromNode, toNode) {
    const from = visibleRect(fromNode);
    const to = visibleRect(toNode);
    if (!from || !to) {
      return;
    }
    const x1 = from.left + from.width / 2;
    const y1 = from.top + from.height / 2;
    const x2 = to.left + to.width / 2;
    const y2 = to.top + to.height / 2;
    const dx = x2 - x1;
    const dy = y2 - y1;
    const length = Math.hypot(dx, dy);
    if (length < 16) {
      return;
    }
    const cue = document.createElement('div');
    cue.className = 'v2-tutorial-drag-cue';
    cue.style.left = x1 + 'px';
    cue.style.top = y1 + 'px';
    cue.style.width = length + 'px';
    cue.style.transform = 'rotate(' + (Math.atan2(dy, dx) * 180 / Math.PI) + 'deg)';
    cue.innerHTML = '<span class="v2-tutorial-drag-cue-runner"></span>';
    document.body.appendChild(cue);
  }

  function addClickCue(node) {
    const rect = visibleRect(node);
    if (!rect) {
      return;
    }
    const cue = document.createElement('div');
    cue.className = 'v2-tutorial-click-cue';
    cue.style.left = (rect.left + rect.width / 2) + 'px';
    cue.style.top = (rect.top + rect.height / 2) + 'px';
    document.body.appendChild(cue);
  }

  function roundedHole(x, y, w, h, radius) {
    const r = Math.max(0, Math.min(radius, w / 2, h / 2));
    return 'M' + (x + r) + ',' + y +
      'H' + (x + w - r) +
      'A' + r + ',' + r + ' 0 0 1 ' + (x + w) + ',' + (y + r) +
      'V' + (y + h - r) +
      'A' + r + ',' + r + ' 0 0 1 ' + (x + w - r) + ',' + (y + h) +
      'H' + (x + r) +
      'A' + r + ',' + r + ' 0 0 1 ' + x + ',' + (y + h - r) +
      'V' + (y + r) +
      'A' + r + ',' + r + ' 0 0 1 ' + (x + r) + ',' + y + 'Z';
  }

  function paintMaskHoles(svg, holes) {
    const width = window.innerWidth;
    const height = window.innerHeight;
    svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
    svg.setAttribute('width', String(width));
    svg.setAttribute('height', String(height));
    let shade = svg.querySelector('[data-shade="1"]');
    if (!shade) {
      shade = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      shade.setAttribute('data-shade', '1');
      shade.setAttribute('fill', 'rgba(1, 4, 9, 0.46)');
      shade.setAttribute('fill-rule', 'evenodd');
      svg.appendChild(shade);
    }
    let path = 'M0,0H' + width + 'V' + height + 'H0Z';
    holes.forEach(function (hole) {
      path += roundedHole(hole.x, hole.y, hole.w, hole.h, 12);
    });
    shade.setAttribute('d', path);
  }

  function applyHide(ctx, hide) {
    const page = ctx.page;
    const keys = hide || [];
    page.classList.toggle('tutorial-hide-hand', keys.indexOf('hand') >= 0);
    page.classList.toggle('tutorial-hide-deck', keys.indexOf('deck') >= 0);
    page.classList.toggle('tutorial-hide-harmony', keys.indexOf('harmony') >= 0);
    page.classList.toggle('tutorial-hide-energy', keys.indexOf('energy') >= 0);
    page.classList.toggle('tutorial-hide-ultimate', keys.indexOf('ultimate') >= 0);
    page.classList.toggle('tutorial-hide-mulligan', keys.indexOf('mulligan') >= 0);
  }

  function renderTutorial(ctx) {
    const info = tutorial(ctx.game);
    const mask = ctx.tutorialMask || document.getElementById('v2-tutorial-mask');
    const modal = ctx.tutorialModal || document.getElementById('v2-tutorial-modal');
    const cue = ctx.tutorialCue || document.getElementById('v2-tutorial-cue');
    ctx.tutorialMask = mask;
    ctx.tutorialModal = modal;
    ctx.tutorialCue = cue;
    ctx.page.classList.toggle('is-tutorial', Boolean(info && info.enabled));
    if (ctx.nodes.playerBench) {
      ctx.nodes.playerBench.setAttribute('data-count', String((info && info.character_slots) || 4));
    }
    if (ctx.nodes.opponentBench) {
      ctx.nodes.opponentBench.setAttribute('data-count', String((info && info.character_slots) || 4));
    }
    clearCues();
    if (!info || !info.enabled) {
      applyHide(ctx, []);
      ctx.page.querySelectorAll('.tutorial-spotlight').forEach(function (node) {
        node.classList.remove('tutorial-spotlight');
      });
      if (mask) {
        mask.setAttribute('hidden', '');
        mask.setAttribute('aria-hidden', 'true');
      }
      if (modal) {
        modal.setAttribute('hidden', '');
        modal.innerHTML = '';
      }
      if (cue) {
        cue.setAttribute('hidden', '');
        cue.textContent = '';
      }
      ctx.tutorialModalDismissed = null;
      return;
    }
    applyHide(ctx, info.hide || []);
    // Hand lessons must remain reachable behind the blocking tutorial mask.
    if (window.matchMedia('(pointer: coarse) and (max-width: 1200px)').matches
        && ctx.page.dataset.mobileHandTutorialStep !== info.step_id && !ctx.dragState) {
      ctx.page.dataset.mobileHandTutorialStep = info.step_id;
      if ((info.spotlights || []).some(function (token) { return String(token).indexOf('hand:') === 0; })) {
        ctx.page.classList.add('mobile-hand-open');
      } else {
        ctx.page.classList.remove('mobile-hand-open');
      }
    }
    ctx.page.querySelectorAll('.tutorial-spotlight').forEach(function (node) {
      node.classList.remove('tutorial-spotlight');
    });
    const holes = [];
    (info.spotlights || []).forEach(function (token) {
      const node = resolveNode(ctx, token);
      if (!node) {
        return;
      }
      node.classList.add('tutorial-spotlight');
      const rect = visibleRect(node);
      if (!rect) {
        return;
      }
      holes.push({
        x: Math.max(0, rect.left - 8),
        y: Math.max(0, rect.top - 8),
        w: rect.width + 16,
        h: rect.height + 16,
      });
      addFocusFrame(rect);
    });
    const dismissed = ctx.tutorialModalDismissed === info.step_id;
    if (mask) {
      const showMask = Boolean(info.mask) && (Boolean(info.ack_required) || !dismissed);
      if (showMask) {
        mask.removeAttribute('hidden');
        paintMaskHoles(mask, holes);
      } else {
        mask.setAttribute('hidden', '');
      }
      mask.setAttribute('aria-hidden', showMask ? 'false' : 'true');
    }
    if (info.drag_arrow) {
      const mobileHand = window.matchMedia('(pointer: coarse) and (max-width: 1200px)').matches
        && String(info.drag_arrow.from).indexOf('hand:') === 0;
      addDragCue(resolveNode(ctx, info.drag_arrow.from), resolveNode(ctx, mobileHand ? 'play_area' : info.drag_arrow.to));
    }
    if (info.click_cue) {
      addClickCue(resolveNode(ctx, info.click_cue));
    }
    if (modal) {
      modal.className = 'v2-tutorial-modal placement-' + String(info.placement || 'bottom-right');
      if (!info.modal || dismissed) {
        modal.setAttribute('hidden', '');
        modal.innerHTML = '';
      } else {
        modal.removeAttribute('hidden');
        modal.innerHTML =
          '<div class="v2-tutorial-dialog">' +
            '<p class="eyebrow">第 ' + V2.escapeHtml(info.level_index) + ' / ' + V2.escapeHtml(info.level_count) + ' 关</p>' +
            '<h2>' + V2.escapeHtml((info.modal && info.modal.title) || info.title || '新手教学') + '</h2>' +
            '<p>' + V2.escapeHtml((info.modal && info.modal.body) || '') + '</p>' +
            '<div class="hero-actions">' +
              '<button class="primary-btn" type="button" data-tutorial-primary="1">' +
                V2.escapeHtml((info.modal && info.modal.primary) || '继续') +
              '</button>' +
            '</div>' +
          '</div>';
      }
    }
    if (cue) {
      const hint = !info.ack_required && dismissed ? String(info.cue || '').trim() : '';
      if (hint) {
        cue.textContent = hint;
        cue.removeAttribute('hidden');
        const play = visibleRect(ctx.nodes.playArea);
        const hand = visibleRect(document.getElementById('v2-opponent-hand'));
        cue.style.left = play ? ((play.left + play.width / 2) + 'px') : '50%';
        cue.style.top = (hand ? (hand.top + Math.max(6, (hand.height - 44) / 2)) : 14) + 'px';
      } else {
        cue.textContent = '';
        cue.setAttribute('hidden', '');
        cue.style.left = '';
        cue.style.top = '';
      }
    }
  }

  function showBlocked(ctx, reason) {
    const toast = ctx.toast;
    const text = reason || (tutorial(ctx.game) && tutorial(ctx.game).blocked_reason) || '请按高亮提示操作。';
    if (!toast) {
      return;
    }
    toast.textContent = text;
    toast.setAttribute('aria-hidden', 'false');
    window.clearTimeout(ctx.tutorialToastTimer);
    ctx.tutorialToastTimer = window.setTimeout(function () {
      toast.setAttribute('aria-hidden', 'true');
    }, 1800);
  }

  function bindTutorial(ctx) {
    ['tutorialModal', 'tutorialMask', 'tutorialCue'].forEach(function (key) {
      const node = ctx[key];
      if (node && node.parentElement !== document.body) {
        document.body.appendChild(node);
      }
    });
    const mask = ctx.tutorialMask;
    const modal = ctx.tutorialModal;
    if (mask) {
      mask.addEventListener('pointerdown', function (event) {
        if (!tutorial(ctx.game)) {
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        showBlocked(ctx);
      });
    }
    if (modal) {
      modal.addEventListener('click', function (event) {
        const primary = event.target.closest('[data-tutorial-primary]');
        const info = tutorial(ctx.game);
        if (primary && info) {
          if (info.ack_required) {
            if (typeof ctx.submitEntry === 'function') {
              ctx.submitEntry({ action: { type: 'tutorial_ack' } });
            }
          } else {
            ctx.tutorialModalDismissed = info.step_id;
            renderTutorial(ctx);
          }
        }
      });
    }
    window.addEventListener('resize', function () {
      if (tutorial(ctx.game)) {
        renderTutorial(ctx);
      }
    });
  }

  TABLE.tutorialInfo = tutorial;
  TABLE.renderTutorial = renderTutorial;
  TABLE.bindTutorial = bindTutorial;
  TABLE.drawTutorialDragArrow = function (ctx) {
    renderTutorial(ctx);
  };
}(window));
