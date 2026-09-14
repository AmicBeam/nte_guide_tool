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
    if (raw === 'end_turn') {
      return ctx.nodes.endTurnBtn;
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
      return ctx.page.querySelector('[data-player-side="' + CSS.escape(side) + '"]');
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
    if (raw.indexOf('character:') === 0) {
      const parts = raw.split(':');
      return ctx.page.querySelector('[data-entity-id="' + CSS.escape(parts[1] + ':' + parts[2]) + '"]');
    }
    return ctx.page.querySelector('[data-tutorial-id="' + CSS.escape(raw) + '"]');
  }

  function centerOf(node) {
    if (!node || typeof node.getBoundingClientRect !== 'function') {
      return null;
    }
    const rect = node.getBoundingClientRect();
    const page = ctxPageRect();
    return {
      x: rect.left + rect.width / 2 - page.left,
      y: rect.top + rect.height / 2 - page.top,
    };
  }

  function ctxPageRect() {
    const page = document.getElementById('v2-table-page');
    return page ? page.getBoundingClientRect() : { left: 0, top: 0 };
  }

  function drawArrow(svg, from, to, klass) {
    if (!svg || !from || !to) {
      return;
    }
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('class', 'v2-fx-arrow' + (klass ? ' ' + klass : ''));
    const midX = (from.x + to.x) / 2;
    const midY = Math.min(from.y, to.y) - 36;
    path.setAttribute('d', 'M ' + from.x + ' ' + from.y + ' Q ' + midX + ' ' + midY + ' ' + to.x + ' ' + to.y);
    svg.appendChild(path);
  }

  function clearArrows(ctx) {
    const svg = ctx.tutorialArrows;
    if (svg) {
      svg.innerHTML = '';
    }
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
    const svg = ctx.tutorialArrows || document.getElementById('v2-tutorial-arrows');
    ctx.tutorialMask = mask;
    ctx.tutorialModal = modal;
    ctx.tutorialArrows = svg;
    ctx.page.classList.toggle('is-tutorial', Boolean(info && info.enabled));
    if (ctx.nodes.playerBench) {
      ctx.nodes.playerBench.setAttribute('data-count', String((info && info.character_slots) || 4));
    }
    if (ctx.nodes.opponentBench) {
      ctx.nodes.opponentBench.setAttribute('data-count', String((info && info.character_slots) || 4));
    }
    if (!info || !info.enabled) {
      applyHide(ctx, []);
      ctx.page.querySelectorAll('.tutorial-spotlight').forEach(function (node) {
        node.classList.remove('tutorial-spotlight');
      });
      if (mask) {
        mask.hidden = true;
        mask.setAttribute('aria-hidden', 'true');
      }
      if (modal) {
        modal.hidden = true;
        modal.innerHTML = '';
      }
      clearArrows(ctx);
      ctx.tutorialModalDismissed = null;
      return;
    }
    applyHide(ctx, info.hide || []);
    ctx.page.querySelectorAll('.tutorial-spotlight').forEach(function (node) {
      node.classList.remove('tutorial-spotlight');
    });
    (info.spotlights || []).forEach(function (token) {
      const node = resolveNode(ctx, token);
      if (node) {
        node.classList.add('tutorial-spotlight');
      }
    });
    if (mask) {
      if (info.mask) {
        mask.removeAttribute('hidden');
      } else {
        mask.setAttribute('hidden', '');
      }
      mask.setAttribute('aria-hidden', info.mask ? 'false' : 'true');
    }
    const dismissed = ctx.tutorialModalDismissed === info.step_id;
    if (modal) {
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
              (info.can_skip_step
                ? '<button class="secondary-btn" type="button" data-tutorial-skip-step="1">跳过说明</button>'
                : '') +
            '</div>' +
          '</div>';
      }
    }
    clearArrows(ctx);
    if (svg && ctx.dragState && info.drag_arrow && info.drag_arrow.when === 'drag_attack') {
      const from = centerOf(resolveNode(ctx, info.drag_arrow.from));
      const to = centerOf(resolveNode(ctx, info.drag_arrow.to));
      drawArrow(svg, from, to, 'is-guide');
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
    ['tutorialModal', 'tutorialArrows', 'tutorialMask'].forEach(function (key) {
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
        const skip = event.target.closest('[data-tutorial-skip-step]');
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
        if (skip && info) {
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
    ctx.page.addEventListener('pointerdown', function (event) {
      const info = tutorial(ctx.game);
      if (!info || !info.mask || info.ack_required) {
        return;
      }
      const spotlight = event.target.closest('.tutorial-spotlight');
      if (spotlight) {
        return;
      }
      if (event.target.closest('.v2-tutorial-modal, .v2-tutorial-dialog, .v2-menu-overlay, .v2-result-overlay')) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      showBlocked(ctx);
    }, true);
  }

  TABLE.tutorialInfo = tutorial;
  TABLE.renderTutorial = renderTutorial;
  TABLE.bindTutorial = bindTutorial;
  TABLE.drawTutorialDragArrow = function (ctx) {
    renderTutorial(ctx);
  };
}(window));
