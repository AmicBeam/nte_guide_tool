(function (root) {
  const V2 = root.NTE_V2;
  const TABLE = root.NTE_V2_TABLE = root.NTE_V2_TABLE || {};
  const jinguOverlays = new WeakMap();

  function popIfChanged(node, signature) {
    if (!node || signature == null) {
      return;
    }
    const prev = node.getAttribute('data-pop-sig');
    node.setAttribute('data-pop-sig', String(signature));
    if (prev == null || prev === '' || prev === String(signature)) {
      return;
    }
    node.classList.remove('v2-stat-pop');
    void node.offsetWidth;
    node.classList.add('v2-stat-pop');
  }

  function shortName(character) {
    const data = character || {};
    return data.name || data.id || '角色';
  }

  function fitResources(node) {
    const row = node.matches('.v2-resource-row') ? node : node.querySelector('.v2-resource-row');
    if (!row || !row.clientWidth) return;
    const labels = row.querySelectorAll('[data-resource-full]');
    labels.forEach(function (label) { label.textContent = label.dataset.resourceFull; });
    const gap = parseFloat(getComputedStyle(row).columnGap) || 0;
    const required = Array.from(row.children).reduce(function (sum, item) { return sum + item.getBoundingClientRect().width; }, 0) + gap * Math.max(0, row.children.length - 1);
    if (required > row.getBoundingClientRect().width) {
      labels.forEach(function (label) { label.textContent = label.dataset.resourceShort; });
    }
  }
  const resourceRows = new WeakMap();
  const resourceObserver = typeof ResizeObserver === 'function' ? new ResizeObserver(function (entries) {
    entries.forEach(function (entry) { fitResources(entry.target); });
  }) : null;

  function fillCharacterCard(node, character, options) {
    options = options || {};
    const data = Object.assign({}, V2.characterFallback(character && character.id) || {}, character || {});
    const side = options.side || node.getAttribute('data-side') || 'a';
    if (options.ghost) {
      node.removeAttribute('data-entity-id');
      node.removeAttribute('data-drag-kind');
      node.setAttribute('data-ghost-character', data.id || '');
    } else {
      node.setAttribute('data-entity-id', TABLE.entityId(side, data.id));
      node.setAttribute('data-drag-kind', 'character');
      node.removeAttribute('data-ghost-character');
    }
    node.setAttribute('data-character-id', data.id || '');
    node.setAttribute('data-side', side);
    node.removeAttribute('data-drop-role');
    node.classList.toggle('legal', Boolean(options.canAttack || options.canAwaken || options.canTarget));
    node.classList.toggle('down', Number(data.down_turns) > 0);
    const damageImmune = Boolean(data.damage_immune && !options.placeholder && !data.down_turns && data.hp > 0);
    const damageLimited = Boolean(typeof data.damage_limit === 'number' && data.damage_limit >= 0 &&
      !options.placeholder && !data.down_turns && data.hp > 0);
    node.classList.toggle('is-damage-immune', damageImmune);
    node.classList.toggle('is-damage-limited', damageLimited);
    node.classList.toggle('is-opponent', Boolean(options.isOpponent));
    ['v2-fx-down', 'v2-fx-lunge', 'v2-fx-enter', 'v2-fx-hurt', 'v2-fx-revive'].forEach(function (name) {
      node.classList.remove(name);
    });
    if (node.style) {
      node.style.transform = '';
      node.style.opacity = '';
    }
    node.classList.toggle('is-placeholder', Boolean(options.placeholder));
    node.classList.toggle('target-legal', Boolean(options.canTarget));
    node.classList.toggle('harmony-source', Boolean(data.harmony_source));
    node.classList.toggle('harmony-ready', Boolean(data.harmony_ready));
    const compact = !options.usePortrait;
    const energyMax = data.energy_max == null ? 5 : data.energy_max;
    const energyNow = data.energy == null ? 0 : data.energy;
    const attack = data.attack == null ? '-' : data.attack;
    const hp = (data.hp == null ? '-' : data.hp) + '/' + (data.max_hp == null ? '-' : data.max_hp);
    const shield = Number(data.shield) || 0;
    function combatStat(kind, shown, title) {
      return '<span class="v2-combat-stat is-' + kind + '" title="' + V2.escapeAttr(title) + '">' +
        V2.statIconSvg(kind) + '<em>' + V2.escapeHtml(String(shown)) + '</em></span>';
    }
    const jinguHtml = data.jingu == null ? '' : combatStat('jingu', data.jingu, '金谷 ' + data.jingu + '（可为负，倒地保留）');
    const auraHtml = data.protagonist_aura == null ? '' : combatStat('aura', data.protagonist_aura, '主角光环 ' + data.protagonist_aura + '/6（倒地保留）');
    const fangHtml = Number(data.beast_fangs) > 0 ? combatStat('fang', data.beast_fangs, '兽牙影刺：剩余 ' + data.beast_fangs + ' 个己方回合') : '';
    const vitalsHtml = options.usePortrait
      ? ('<div class="v2-portrait-vitals">' +
          (shield > 0 ? combatStat('shield', shield, '护盾 ' + shield) : '') +
          '<div class="v2-portrait-vitals-row">' +
            '<span class="v2-attack-stack">' + jinguHtml + auraHtml + fangHtml + combatStat('atk', attack, '攻击 ' + attack) + '</span>' +
            combatStat('hp', hp, '生命 ' + hp) +
          '</div></div>')
      : '';
    const page = document.getElementById('v2-table-page');
    const hideHarmony = Boolean(page && page.classList.contains('tutorial-hide-harmony'));
    const hideEnergy = Boolean(page && page.classList.contains('tutorial-hide-energy'));
    function resourceLabel(full, short, value, cls) {
      return '<span class="' + cls + '" title="' + V2.escapeAttr(full + ' ' + value) + '">' +
        '<span data-resource-full="' + full + ' " data-resource-short="' + short + '">' + full + ' </span>' + V2.escapeHtml(value) + '</span>';
    }
    const resourceHtml = (hideHarmony ? '' : resourceLabel('环合', '环', String(data.harmony || 0) + '/' + (data.harmony_max == null ? 2 : data.harmony_max), 'harmony')) +
      (hideEnergy ? '' : resourceLabel(data.resource_name || '能量', data.resource_name ? '时' : '能', energyNow + '/' + energyMax, 'energy'));
    const stats = compact ? [
      { html: '<span class="v2-attack-stack">' + jinguHtml + auraHtml + fangHtml + combatStat('atk', attack, '攻击 ' + attack) + '</span>' },
      { html: combatStat('hp', hp, '生命 ' + hp) },
      { html: combatStat('shield', shield, '护盾 ' + shield) },
    ] : [];
    if (data.truth_keys != null) stats.push({ cls: 'truth-keys', text: '真理之匙 ' + data.truth_keys });
    if (data.arid != null) stats.push({ cls: 'arid', text: '荒时 ' + data.arid });
    if (resourceHtml && !data.summoned) stats.push({ html: '<div class="v2-resource-row">' + resourceHtml + '</div>' });
    if (!hideEnergy && data.awakened) stats.push({ cls: 'energy', text: '终结 ' + (data.ultimate_expires === 'turn_end' ? '本回合' : (data.ultimate_turns || 0)) });
    if (Number(data.growth) > 0) {
      stats.push({ cls: '', text: '成长 ' + data.growth });
    }
    if (data.id !== 'zhenhong' && Number(data.gaze) > 0) {
      stats.push({ cls: '', text: '凝视 ' + (data.gaze == null ? 0 : data.gaze) });
    }
    if (Number(data.brand) > 0) {
      stats.push({ cls: '', text: '焚印 ' + data.brand });
    }
    if (data.pact && !(data.effect_markers || []).some(function (marker) { return marker.id === 'pact'; })) {
      stats.push({ cls: '', text: '枚约' });
    }
    const attackBonus = (Number(data.atk_buff) || 0) + (Number(data.form_attack) || 0);
    if (attackBonus > 0) {
      stats.push({ cls: '', text: '攻击+' + attackBonus });
    }
    if (Number(data.atk_debuff) > 0) {
      stats.push({ cls: '', text: '攻击-' + data.atk_debuff });
    }
    if (Number(data.nightmare) > 0) {
      stats.push({ cls: '', text: '噩梦 ' + data.nightmare });
    }
    const downTurns = Number(data.down_turns) > 0 ? Number(data.down_turns) : 0;
    const note = (page && page.classList.contains('is-tutorial') && !data.shape) ? '' : (data.shape || '无武备') + (data.shape_id === 'R08' ? '（威慑凝视 ' + (data.intimidation || 0) + '）' : '');
    const defaultPortrait = V2.characterAsset(data, 'portrait') || V2.characterAsset(data, 'avatar');
    const portrait = options.usePortrait ? V2.battlePortrait(data) : V2.characterAsset(data, 'avatar');
    const mobileCard = window.matchMedia('(pointer: coarse) and (max-width: 1200px)').matches;
    const actions = [];
    if (options.canAwaken) {
      actions.push('<button class="v2-awaken-btn" type="button" data-ultimate-id="' + V2.escapeAttr(data.id) + '">终结</button>');
    }
    const panelSig = [attack, hp, shield, data.harmony, energyNow, energyMax, data.awakened ? 1 : 0, data.jingu, data.protagonist_aura, data.beast_fangs].join('|');
    const prevPanel = node.getAttribute('data-pop-sig');
    node.setAttribute('data-pop-sig', panelSig);
    const protectionLabel = damageLimited ? '单次至多受到 ' + data.damage_limit + ' 点伤害' : '免疫伤害';
    node.innerHTML = (damageImmune || damageLimited ? '<svg class="v2-immunity-frame' +
      (damageLimited ? ' v2-damage-limit-frame' : '') + '" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="' + protectionLabel + '"><path d="M8 1 H92 Q99 1 99 8 V98 Q90 102 50 106 Q10 102 1 98 V8 Q1 1 8 1 Z"/></svg>' : '') +
      '<div class="v2-character-art' + (actions.length && !mobileCard ? ' has-ultimate' : '') + '">' + (data.summoned && !portrait ? '<div class="v2-summon-art" aria-label="召唤物">⌛</div>' : V2.imageMarkup(portrait, data.name, options.usePortrait ? 'v2-portrait' : 'v2-avatar')) +
      vitalsHtml + (mobileCard ? '' : actions.join('')) + '</div>' +
      '<div class="v2-character-panel">' +
        '<div class="v2-character-heading">' +
          '<strong title="' + V2.escapeAttr(data.name || '') + '">' + V2.escapeHtml(shortName(data)) + '</strong>' +
          (downTurns ? '<span class="v2-down-badge">倒地 ' + downTurns + '</span>' : '') +
          (data.attribute ? V2.attributeMarkup(data.attribute) : '') +
        '</div>' +
        '<div class="v2-character-stats">' + stats.map(function (item) {
          if (item.html) {
            return item.html;
          }
          return '<span class="' + item.cls + '">' + V2.escapeHtml(item.text) + '</span>';
        }).join('') + '</div>' +
        recordsHtml(options.records) +
        '<div class="v2-character-note"><span title="' + V2.escapeAttr(note) + '">' + V2.escapeHtml(note) + '</span></div>' +
      '</div>' + (mobileCard ? '<div class="v2-character-extras">' + actions.join('') : '') +
      effectMarkersHtml(options.placeholder ? [] : data.effect_markers, 'v2-character-effects') + (mobileCard ? '</div>' : '');
    const portraitImage = node.querySelector('.v2-character-art > img');
    if (portraitImage && options.usePortrait && portrait !== defaultPortrait) {
      portraitImage.onerror = function () {
        this.onerror = null;
        this.src = defaultPortrait;
      };
    }
    if (prevPanel && prevPanel !== panelSig) {
      const previous = prevPanel.split('|');
      const current = panelSig.split('|');
      [['.is-atk', 0], ['.is-hp', 1], ['.is-shield', 2], ['.harmony', 3], ['.energy', 4], ['.is-jingu', 7], ['.is-aura', 8], ['.is-fang', 9]].forEach(function (stat) {
        if (previous[stat[1]] === current[stat[1]]) return;
        node.querySelectorAll(stat[0]).forEach(function (badge) {
          badge.classList.add('v2-stat-pop');
        });
      });
    }
    if (resourceObserver && !options.ghost) {
      const previousRow = resourceRows.get(node);
      if (previousRow) resourceObserver.unobserve(previousRow);
      const row = node.querySelector('.v2-resource-row');
      if (row) { resourceObserver.observe(row); resourceRows.set(node, row); }
    }
    requestAnimationFrame(function () { fitResources(node); });
    return node;
  }

  function reclaimCharacters(container, pool) {
    if (!container || !pool) return;
    Array.prototype.slice.call(container.querySelectorAll('[data-entity-id]')).forEach(function (node) {
      pool.appendChild(node);
    });
  }

  function ensureCharacterNode(pool, side, characterId) {
    const id = TABLE.entityId(side, characterId);
    let node = document.querySelector('[data-entity-id="' + id + '"]');
    if (!node && pool) node = pool.querySelector('[data-entity-id="' + id + '"]');
    if (!node) {
      node = document.createElement('article');
      node.className = 'v2-character-card';
      node.setAttribute('data-entity-id', id);
      node.setAttribute('data-character-id', characterId);
      node.setAttribute('data-drag-kind', 'character');
      node.setAttribute('data-side', side);
      if (pool) pool.appendChild(node);
    }
    return node;
  }

  function effectMarkersHtml(markers, className) {
    if (!Array.isArray(markers) || !markers.length) return '';
    return '<div class="' + className + '">' + markers.map(function (marker) {
      const detail = [marker.name, marker.description, marker.timing, marker.clears_on_down ? '倒地时失去' : '倒地后保留'].filter(Boolean).join(' · ');
      if (className === 'v2-character-effects') {
        // Older replay markers predate the explicit presentation kind.
        const debuff = marker.kind ? marker.kind === 'debuff' : ['timed_attack', 'next_resistance'].includes(marker.id);
        const label = debuff ? '减益' : '增益';
        const path = debuff ? 'M12 4v16m-6-6 6 6 6-6' : 'M12 20V4m-6 6 6-6 6 6';
        return '<span class="v2-effect-icon is-' + (debuff ? 'debuff' : 'buff') + '" title="' + V2.escapeAttr(detail) + '" aria-label="' + V2.escapeAttr(label + '：' + detail) + '">' +
          '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="' + path + '"/></svg></span>';
      }
      return '<span class="v2-chip" title="' + V2.escapeAttr(detail) + '" aria-label="' + V2.escapeAttr(detail) + '">' +
        '<span class="v2-effect-name">' + V2.escapeHtml(marker.name) + '</span><small>' + V2.escapeHtml(marker.timing || '') + '</small></span>';
    }).join('') + '</div>';
  }

  function fillFrontDebuff(node, state) {
    if (!node) {
      return;
    }
    const zone = (state && state.front_debuff) || {};
    const labels = { delay: '延滞', burn: '浊燃', star: '黯星', weave: '覆纹', stain: '浸染' };
    const chips = Object.keys(labels).filter(function (key) { return Boolean(zone[key]); }).map(function (key) {
      const status = zone[key];
      let duration = status.infinite ? '无限' : (status.expires_side || key === 'weave' || key === 'stain'
        ? '本回合' : ('倒计时 ' + (status.left == null ? '—' : status.left)));
      if (key === 'burn' && status.stacks > 1) {
        duration = status.stacks + '层 · ' + duration;
      }
      return '<span class="v2-chip">' + V2.escapeHtml(labels[key]) +
        '<small>' + V2.escapeHtml(duration) + '</small></span>';
    });
    Object.keys(zone.dots || {}).forEach(function (key) {
      const status = zone.dots[key];
      if (!status) return;
      const timing = status.left == null ? (status.stacks + '层') : (status.stacks + '层 · 倒计时 ' + status.left);
      chips.push('<span class="v2-chip">' + V2.escapeHtml(status.name || key) +
        '<small>' + V2.escapeHtml(timing) + '</small></span>');
    });
    node.innerHTML = chips.join('');
    node.hidden = chips.length === 0;
  }

  function emptyFront(slot, label) {
    while (slot.firstChild) {
      slot.removeChild(slot.firstChild);
    }
    const empty = document.createElement('div');
    empty.className = 'v2-front-slot-empty subtle';
    empty.textContent = label || '战斗区空';
    slot.appendChild(empty);
  }

  function fillHarmonySourceCaption(zone, state, empty) {
    if (!zone) {
      return;
    }
    let caption = zone.querySelector('.v2-harmony-source');
    if (!caption) {
      caption = document.createElement('div');
      caption.className = 'v2-harmony-source';
      zone.appendChild(caption);
    }
    const show = Boolean(empty && state && state.harmony_available);
    const html = show ? TABLE.harmonySourceCaptionHtml(TABLE.harmonySourceCharacter(state)) : '';
    caption.innerHTML = html;
    caption.hidden = !html;
  }

  function placeFront(ctx, side, frontId, canAct) {
    const zone = side === ctx.viewer ? ctx.nodes.playerFrontZone : ctx.nodes.opponentFrontZone;
    const slot = side === ctx.viewer ? ctx.nodes.playerFront : ctx.nodes.opponentFront;
    const state = TABLE.sideState(ctx.game, side);
    const isOpponent = side !== ctx.viewer;
    reclaimCharacters(slot, ctx.pool);
    while (slot.firstChild) {
      slot.removeChild(slot.firstChild);
    }
    if (zone) {
      zone.classList.toggle('is-empty', !frontId);
      zone.classList.toggle('harmony-available', Boolean(state && state.harmony_available));
    }
    fillFrontDebuff(side === ctx.viewer ? ctx.nodes.playerFrontDebuff : ctx.nodes.opponentFrontDebuff, state);
    if (zone) {
      let effects = zone.querySelector('.v2-zone-effects');
      if (!effects) {
        effects = document.createElement('div');
        effects.className = 'v2-zone-effects';
        zone.appendChild(effects);
      }
      effects.innerHTML = effectMarkersHtml(state && state.effect_markers, 'v2-zone-effect-list');
      effects.hidden = !effects.innerHTML;
    }
    fillHarmonySourceCaption(zone, state, !frontId);
    if (!frontId) {
      emptyFront(slot, '战斗区空');
      return;
    }
    const character = TABLE.characterMap(state)[frontId] || V2.characterFallback(frontId);
    const node = ensureCharacterNode(ctx.pool, side, frontId);
    fillCharacterCard(node, character, {
      side: side,
      isOpponent: isOpponent,
      usePortrait: true,
      canAttack: canAct && !isOpponent && Boolean(ctx.findAction('attack', { character_id: frontId })),
      canAwaken: canAct && !isOpponent && Boolean(ctx.findAction('ultimate', { character_id: frontId })),
      records: frontId === 'xun' ? ((state && state.records) || []) : null,
    });
    slot.appendChild(node);
  }

  function placeBench(ctx, side, canAct) {
    const bench = side === ctx.viewer ? ctx.nodes.playerBench : ctx.nodes.opponentBench;
    const state = TABLE.sideState(ctx.game, side);
    const frontId = state && state.front;
    const isOpponent = side !== ctx.viewer;
    const chars = V2.sortedCharacters(((state && state.characters) || []).filter(function (h) { return !h.summoned; }));
    reclaimCharacters(bench, ctx.pool);
    while (bench.firstChild) {
      bench.removeChild(bench.firstChild);
    }
    chars.forEach(function (character, index) {
      const wrap = document.createElement('div');
      wrap.className = 'v2-bench-slot';
      wrap.setAttribute('data-arc', String(index));
      if (character.id === frontId) {
        wrap.classList.add('is-sortie');
        const ghost = document.createElement('article');
        ghost.className = 'v2-character-card is-placeholder';
        fillCharacterCard(ghost, character, {
          side: side,
          isOpponent: isOpponent,
          usePortrait: true,
          ghost: true,
          placeholder: true,
          records: character.id === 'xun' ? ((state && state.records) || []) : null,
        });
        wrap.appendChild(ghost);
        bench.appendChild(wrap);
        return;
      }
      const node = ensureCharacterNode(ctx.pool, side, character.id);
      fillCharacterCard(node, character, {
        side: side,
        isOpponent: isOpponent,
        usePortrait: true,
        canAttack: canAct && !isOpponent && Boolean(ctx.findAction('attack', { character_id: character.id })),
        canAwaken: canAct && !isOpponent && Boolean(ctx.findAction('ultimate', { character_id: character.id })),
        placeholder: false,
        records: character.id === 'xun' ? ((state && state.records) || []) : null,
      });
      wrap.appendChild(node);
      bench.appendChild(wrap);
    });
  }

  function handCardHtml(card, options) {
    options = options || {};
    if (!card || card.hidden) {
      return '<article class="v2-hidden-card" aria-label="隐藏手牌"></article>';
    }
    card = V2.handCardFace(card);
    const classes = ['v2-hand-card'];
    const mark = card.jingu_mark;
    const marked = Boolean(mark && [-1, 0, 1].includes(mark.delta));
    if (marked) classes.push('has-jingu-mark');
    const markText = marked ? (mark.delta > 0 ? '+' : '') + mark.delta : '';
    const markHtml = marked ? '<span class="v2-jingu-mark" title="本牌结算后，小吱金谷 ' + markText + '" aria-label="金谷 ' + markText + '">' + V2.statIconSvg('jingu') + '<em>' + markText + '</em></span>' : '';
    const inMulligan = options.phase === 'mulligan';
    const unusable = !inMulligan && !options.legal;
    const reason = unusable ? (card.unavailable_reason || '现在不能使用') : '';
    if (options.legal && !unusable) classes.push('legal');
    if (options.selected) classes.push('selected');
    if (options.raised) classes.push('raised');
    if (unusable) classes.push('is-unusable');
    if (options.groupStart) classes.push('v2-hand-group-start');
    if (card.action_point_free) classes.push('is-ap-free');
    const freeBorder = card.action_point_free ? '<svg class="v2-free-card-border" aria-hidden="true"><rect /></svg>' : '';
    const timing = V2.cardTimingClass(options.instantSpent
      ? Object.assign({}, card, { instant: false }) : card);
    if (timing) classes.push(timing);
    const portrait = V2.characterAsset(V2.characterFallback(card.character_id) || {}, 'portrait');
    const style = portrait ? ' style="--card-art:url(\'' + String(portrait).replace(/'/g, '%27') + '\')"' : '';
    const instanceAttr = card.instance_id ? ' data-instance-id="' + V2.escapeAttr(card.instance_id) + '"' : '';
    const title = reason ? ' title="' + V2.escapeAttr(reason) + '"' : '';
    const label = (card.name || '卡牌') + (reason ? '，' + reason : '');
    return (
      '<article class="' + classes.join(' ') + '" draggable="false" data-card-id="' + V2.escapeAttr(card.instance_id) + '" data-drag-kind="card" data-card-owner="' + V2.escapeAttr((card.hand_face || card).character_id || '') + '" data-card-type="' + V2.escapeAttr(card.type || '') + '"' + instanceAttr + style + title + ' aria-label="' + V2.escapeAttr(label) + '">' +
        freeBorder + markHtml + (card.copy ? '<span class="v2-chip copy">复制</span>' : '') +
        '<strong>' + V2.escapeHtml(card.name || '卡牌') + '</strong>' +
        V2.cardDescriptionHtml(card.description, '正在读取卡牌效果。', 'subtle') +
        V2.cardFrameStat(card) +
        (card.response ? '<span class="v2-response-mark">响应</span>' : '') +
      '</article>'
    );
  }

  function recordsHtml(records) {
    const list = Array.isArray(records) ? records : [];
    if (!list.length) {
      return '';
    }
    return '<div class="v2-xun-records">' + list.map(recordHtml).join('') + '</div>';
  }

  function recordHtml(record) {
    if (!record) return '';
    if (typeof record === 'string') {
      return '<span class="v2-chip copy">' + V2.escapeHtml(record) + '</span>';
    }
    const name = record.name || record.card_name || record.label || '记录';
    const owner = (V2.characterFallback(record.character_id) || {}).name || record.source || '';
    const type = record.type ? V2.cardTypeLabel(record.type, record.terminal) : '';
    return '<span class="v2-chip copy">' + V2.escapeHtml([name, owner, type].filter(Boolean).join(' · ')) + '</span>';
  }

  function eventSourceText(event) {
    const bits = [];
    if (event && event.text) bits.push(event.text);
    if (event && event.source) bits.push('来源 ' + event.source);
    if (event && event.target) bits.push('目标 ' + event.target);
    if (event && event.amount != null && event.amount !== '') bits.push('数值 ' + event.amount);
    if (event && event.type) bits.push(event.type);
    return bits.join(' · ');
  }

  function setPileLabel(node, label, count) {
    if (!node) return;
    node.innerHTML = V2.escapeHtml(label) + ' <strong>' + V2.escapeHtml(count == null ? '-' : count) + '</strong>';
  }

  function bindHudSides(ctx) {
    const viewer = ctx.viewer;
    const opponent = ctx.opponent;
    if (ctx.nodes.playerLife) {
      ctx.nodes.playerLife.setAttribute('data-player-side', viewer);
      ctx.nodes.playerLife.setAttribute('data-entity-id', viewer + ':player');
    }
    if (ctx.nodes.opponentLife) {
      ctx.nodes.opponentLife.setAttribute('data-player-side', opponent);
      ctx.nodes.opponentLife.setAttribute('data-entity-id', opponent + ':player');
    }
    if (ctx.nodes.playerHand) ctx.nodes.playerHand.setAttribute('data-hand-side', viewer);
    if (ctx.nodes.opponentHand) ctx.nodes.opponentHand.setAttribute('data-hand-side', opponent);
  }

  function syncJinguOverlay(ctx) {
    const rail = ctx.nodes.playerHand;
    if (!ctx.page || !rail || !rail.querySelectorAll) return;
    if (!jinguOverlays.has(rail)) {
      const layer = document.createElement('div');
      layer.className = 'v2-jingu-overlay';
      layer.setAttribute('aria-hidden', 'true');
      ctx.page.appendChild(layer);
      let pending = false;
      const update = function () {
        pending = false;
        layer.replaceChildren();
        const bounds = rail.getBoundingClientRect();
        if (!bounds.width || !bounds.height || getComputedStyle(rail).visibility === 'hidden') return;
        const origin = layer.getBoundingClientRect();
        const gap = 5 * (parseFloat(getComputedStyle(ctx.page).getPropertyValue('--table-layout-scale')) || 1);
        const fragment = document.createDocumentFragment();
        rail.querySelectorAll('.v2-hand-card > .v2-jingu-mark').forEach(function (mark) {
          const rect = mark.parentElement.getBoundingClientRect();
          const center = rect.left + rect.width / 2;
          // Hide marks for cards scrolled outside the visible hand viewport.
          if (center < bounds.left || center > bounds.right) return;
          const clone = mark.cloneNode(true);
          clone.style.left = (center - origin.left) + 'px';
          clone.style.top = (rect.top - origin.top - gap) + 'px';
          fragment.appendChild(clone);
        });
        layer.appendChild(fragment);
      };
      const schedule = function () {
        if (pending) return;
        pending = true;
        root.requestAnimationFrame(update);
      };
      const observer = new MutationObserver(schedule);
      observer.observe(rail, { childList: true, subtree: true, attributes: true });
      observer.observe(ctx.page, { attributes: true, attributeFilter: ['class', 'style'] });
      const resize = new ResizeObserver(schedule);
      resize.observe(rail);
      resize.observe(ctx.page);
      rail.addEventListener('scroll', schedule, { passive: true });
      root.addEventListener('resize', schedule, { passive: true });
      jinguOverlays.set(rail, { schedule: schedule, observer: observer, resize: resize });
    }
    jinguOverlays.get(rail).schedule();
  }

  function renderHands(ctx) {
    const mine = TABLE.sideState(ctx.game, ctx.viewer);
    const theirs = TABLE.sideState(ctx.game, ctx.opponent);
    const myHand = TABLE.sortHandCards(Array.isArray(mine.hand) ? mine.hand : [], mine);
    if (ctx.nodes.playerHand && ctx.nodes.playerHand.setAttribute) {
      ctx.nodes.playerHand.setAttribute('data-character-order', JSON.stringify((mine.characters || []).map(function (item) { return item.id; })));
    }
    const selected = {};
    (ctx.mulliganIds || []).forEach(function (id) { selected[id] = true; });
    const inMulligan = ctx.showMulligan != null
      ? Boolean(ctx.showMulligan)
      : Boolean(ctx.game && ctx.game.phase === 'mulligan');
    const cardsHtml = myHand.length
      ? myHand.map(function (card, index) {
          const entries = ctx.playEntries(card.instance_id);
          const previous = myHand[index - 1];
          return handCardHtml(card, {
            legal: !card.unavailable_reason && entries.length > 0,
            instantSpent: Boolean(mine.used && mine.used.instant),
            selected: Boolean(selected[card.instance_id]),
            raised: ctx.raisedCardId && String(ctx.raisedCardId) === String(card.instance_id),
            phase: ctx.game && ctx.game.phase,
            groupStart: Boolean(index && previous && (previous.hand_face || previous).character_id !== (card.hand_face || card).character_id),
          });
        }).join('')
      : '<div class="subtle">手牌空</div>';
    if (inMulligan && ctx.nodes.mulliganRail) {
      ctx.nodes.mulliganRail.innerHTML = cardsHtml;
      ctx.nodes.playerHand.innerHTML = '';
    } else {
      if (ctx.nodes.mulliganRail) ctx.nodes.mulliganRail.innerHTML = '';
      ctx.nodes.playerHand.innerHTML = cardsHtml;
    }
    if (ctx.page) {
      ctx.page.classList.toggle('is-mulligan', inMulligan);
    }
    if (ctx.nodes.mulliganStage) {
      V2.setHidden(ctx.nodes.mulliganStage, !inMulligan);
    }
    syncJinguOverlay(ctx);
    const hiddenCount = Number(theirs.hand_count != null ? theirs.hand_count : (theirs.hand || []).length || 0);
    const opponentCards = Array.isArray(theirs.hand) ? theirs.hand.slice() : [];
    while (opponentCards.length < hiddenCount) opponentCards.push({ hidden: true });
    ctx.nodes.opponentHand.innerHTML = hiddenCount
      ? opponentCards.slice(0, hiddenCount).map(function (card) {
          if (card && card.copy) {
            const instanceAttr = card.instance_id ? ' data-instance-id="' + V2.escapeAttr(card.instance_id) + '"' : '';
            return '<button class="v2-public-copy" data-public-copy="' + V2.escapeAttr(card.instance_id) + '"' + instanceAttr + '>复制 · ' + V2.escapeHtml(card.name) + '</button>';
          }
          if (card && card.revealed) {
            card = V2.handCardFace(card);
            const portrait = V2.characterAsset(V2.characterFallback(card.character_id) || {}, 'portrait');
            const style = portrait ? ' style="--card-art:url(\'' + String(portrait).replace(/'/g, '%27') + '\')"' : '';
            return '<button class="v2-revealed-card" data-public-copy="' + V2.escapeAttr(card.instance_id || '') + '" data-card-type="' + V2.escapeAttr(card.type || '') + '"' + style + ' aria-label="' + V2.escapeAttr(card.name || '公开卡牌') + '"></button>';
          }
          return '<article class="v2-hidden-card" aria-label="对方手牌"></article>';
        }).join('')
      : '<div class="subtle">对手手牌 0</div>';
  }

  function historyHeadline(event) {
    if (event && event.type === 'discard') return '';
    const text = (event && event.text) || eventSourceText(event);
    const card = event && event.card;
    if (event && event.type === 'draw' && card && card.name && text && text.indexOf('「') < 0) {
      return String(text).replace(/抽 1 张牌。?/, '抽到「' + card.name + '」。');
    }
    return text;
  }

  function renderLogs(ctx) {
    const events = Array.isArray(ctx.game && ctx.game.events) ? ctx.game.events : [];
    const logs = Array.isArray(ctx.game && ctx.game.logs) ? ctx.game.logs : [];
    let html = '';
    if (events.length) {
      html = events.filter(function (event) { return event.type !== 'discard'; }).slice(-40).map(function (event) {
        const text = historyHeadline(event);
        if (!text) {
          return '';
        }
        const desc = TABLE.historyDescription(event);
        const extra = desc && desc !== text ? V2.cardDescriptionHtml(desc, '', 'subtle') : '';
        return '<div class="v2-log-item"><div>' + V2.escapeHtml(text) + '</div>' + extra + '</div>';
      }).filter(Boolean).join('');
    } else if (logs.length) {
      html = logs.filter(function (line) { return !/^「.*」进入弃牌堆[。！!]?$/u.test(String(line)); }).slice(-40).map(function (line) {
        return '<div class="v2-log-item">' + V2.escapeHtml(line) + '</div>';
      }).join('');
    }
    ctx.nodes.logList.innerHTML = html || '<div class="v2-log-item subtle">暂无日志</div>';
    ctx.nodes.logList.scrollTop = ctx.nodes.logList.scrollHeight;
  }

  function renderRecords(ctx) {
    if (!ctx.nodes.records) {
      return;
    }
    ctx.nodes.records.innerHTML = '';
    ctx.nodes.records.hidden = true;
  }

  function renderGame(ctx) {
    const game = ctx.game;
    const mine = TABLE.sideState(game, ctx.viewer);
    const theirs = TABLE.sideState(game, ctx.opponent);
    bindHudSides(ctx);
    const escalation = ctx.page.querySelector('#v2-escalation');
    if (escalation) {
      const progress = game && game.escalation;
      escalation.hidden = !progress;
      escalation.textContent = progress ? progress.hint : '';
      escalation.title = progress ? [progress.description, (progress.active || []).length ? '已生效：' + progress.active.join('；') : '尚未生效'].filter(Boolean).join('\n') : '';
    }
    const tutorial = game && game.tutorial;
    const hide = (tutorial && tutorial.hide) || [];
    ctx.page.classList.toggle('is-tutorial', Boolean(tutorial && tutorial.enabled));
    ctx.page.classList.toggle('tutorial-hide-hand', hide.indexOf('hand') >= 0);
    ctx.page.classList.toggle('tutorial-hide-deck', hide.indexOf('deck') >= 0);
    ctx.page.classList.toggle('tutorial-hide-harmony', hide.indexOf('harmony') >= 0);
    ctx.page.classList.toggle('tutorial-hide-energy', hide.indexOf('energy') >= 0);
    ctx.page.classList.toggle('tutorial-hide-ultimate', hide.indexOf('ultimate') >= 0);
    ctx.page.classList.toggle('tutorial-hide-mulligan', hide.indexOf('mulligan') >= 0);
    const slots = tutorial && tutorial.character_slots ? String(tutorial.character_slots) : '4';
    if (ctx.nodes.playerBench) ctx.nodes.playerBench.setAttribute('data-count', slots);
    if (ctx.nodes.opponentBench) ctx.nodes.opponentBench.setAttribute('data-count', slots);
    if (ctx.nodes.playerFrontZone) ctx.nodes.playerFrontZone.setAttribute('data-zone-side', ctx.viewer);
    if (ctx.nodes.opponentFrontZone) ctx.nodes.opponentFrontZone.setAttribute('data-zone-side', ctx.opponent);
    if (ctx.nodes.playerAp) ctx.nodes.playerAp.setAttribute('data-ap-side', ctx.viewer);
    ctx.nodes.roundLabel.textContent = game && game.turn ? ('第 ' + game.turn + ' 回合') : '第 - 回合';
    ctx.nodes.opponentHp.textContent = theirs.hp == null ? '-' : theirs.hp;
    ctx.nodes.opponentShield.textContent = '护盾 ' + (theirs.shield == null ? '-' : theirs.shield);
    setPileLabel(ctx.nodes.opponentAp, '行动力', theirs.ap == null ? '-' : theirs.ap);
    popIfChanged(ctx.nodes.opponentAp, theirs.ap);
    ctx.nodes.playerHp.textContent = mine.hp == null ? '-' : mine.hp;
    ctx.nodes.playerHp.closest('.v2-player-hud').classList.toggle('is-life-locked', Boolean(mine.life_locked));
    ctx.nodes.opponentHp.closest('.v2-player-hud').classList.toggle('is-life-locked', Boolean(theirs.life_locked));
    ctx.nodes.playerShield.textContent = '护盾 ' + (mine.shield == null ? '-' : mine.shield);
    ctx.nodes.playerAp.textContent = mine.ap == null ? '-' : mine.ap;
    const playerApReadout = ctx.nodes.playerAp && ctx.nodes.playerAp.closest('.v2-ap-readout');
    const instantSpent = Boolean(mine.used && mine.used.instant);
    if (playerApReadout) {
      playerApReadout.classList.toggle('is-instant-spent', instantSpent);
      playerApReadout.title = instantSpent ? '本回合第一张瞬发已用' : '本回合第一张瞬发不耗行动力';
    }
    popIfChanged(ctx.nodes.opponentHp && ctx.nodes.opponentHp.closest('.v2-player-hud'), [theirs.hp, theirs.shield].join('|'));
    popIfChanged(ctx.nodes.playerHp && ctx.nodes.playerHp.closest('.v2-player-hud'), [mine.hp, mine.shield].join('|'));
    popIfChanged(playerApReadout, [mine.ap, instantSpent ? 1 : 0].join('|'));
    popIfChanged(ctx.nodes.normalAttack, mine.normal_attack_available ? '1' : '0');
    popIfChanged(ctx.nodes.ultimate, mine.ultimate_remaining == null ? (mine.ultimate_available ? '1' : '0') : String(mine.ultimate_remaining));
    ctx.nodes.playerName.textContent = mine.name || '我方';
    ctx.nodes.opponentName.textContent = theirs.name || '对手';
    const waiting = game.phase === 'mulligan' || mine.ap == null;
    if (ctx.nodes.normalAttack) {
      ctx.nodes.normalAttack.textContent = waiting ? '-' : (mine.normal_attack_available ? '1' : '0');
    }
    if (ctx.nodes.ultimate) {
      ctx.nodes.ultimate.textContent = waiting ? '-' : (mine.ultimate_remaining == null ? (mine.ultimate_available ? '1' : '0') : String(mine.ultimate_remaining));
    }
    setPileLabel(ctx.nodes.opponentDeck, '牌库', theirs.deck_count);
    setPileLabel(ctx.nodes.opponentDiscard, '弃牌', (theirs.discard || []).length);
    setPileLabel(ctx.nodes.opponentHandCount, '手牌', theirs.hand_count != null ? theirs.hand_count : (theirs.hand || []).length);
    setPileLabel(ctx.nodes.playerDeck, '牌库', mine.deck_count);
    setPileLabel(ctx.nodes.playerDiscard, '弃牌', (mine.discard || []).length);
    setPileLabel(ctx.nodes.playerHandCount, '手牌', mine.hand_count != null ? mine.hand_count : (mine.hand || []).length);
    const canAct = Boolean(game && game.is_my_turn && game.phase === 'playing' && !ctx.busy);
    placeFront(ctx, ctx.opponent, theirs.front, false);
    placeFront(ctx, ctx.viewer, mine.front, canAct);
    placeBench(ctx, ctx.opponent, false);
    placeBench(ctx, ctx.viewer, canAct);
    renderHands(ctx);
    renderLogs(ctx);
    renderRecords(ctx);
    if (ctx.nodes.turnCopy && !ctx.replayMode) {
      ctx.nodes.turnCopy.textContent = '';
    }
    if (ctx.nodes.waitChip) {
      ctx.nodes.waitChip.textContent = ctx.busy ? '演出中' : (game && game.is_my_turn ? '你的回合' : '对手回合');
    }
    if (typeof TABLE.renderTutorial === 'function') {
      TABLE.renderTutorial(ctx);
    }
  }

  TABLE.fillCharacterCard = fillCharacterCard;
  TABLE.ensureCharacterNode = ensureCharacterNode;
  TABLE.handCardHtml = handCardHtml;
  TABLE.recordHtml = recordHtml;
  TABLE.eventSourceText = eventSourceText;
  TABLE.setPileLabel = setPileLabel;
  TABLE.renderGame = renderGame;
  TABLE.renderHands = renderHands;
}(window));
