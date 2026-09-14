(function (root) {
  const V2 = root.NTE_V2;
  const TABLE = root.NTE_V2_TABLE = root.NTE_V2_TABLE || {};

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
    const vitalsHtml = options.usePortrait
      ? ('<div class="v2-portrait-vitals">' +
          (shield > 0 ? combatStat('shield', shield, '护盾 ' + shield) : '') +
          '<div class="v2-portrait-vitals-row">' +
            combatStat('atk', attack, '攻击 ' + attack) +
            combatStat('hp', hp, '生命 ' + hp) +
          '</div></div>')
      : '';
    const page = document.getElementById('v2-table-page');
    const hideHarmony = Boolean(page && page.classList.contains('tutorial-hide-harmony'));
    const hideEnergy = Boolean(page && page.classList.contains('tutorial-hide-energy'));
    const stats = compact
      ? [
          { html: combatStat('atk', attack, '攻击 ' + attack) },
          { html: combatStat('hp', hp, '生命 ' + hp) },
          { html: combatStat('shield', shield, '护盾 ' + shield) },
        ].concat(hideHarmony ? [] : [{ cls: 'harmony', text: '环 ' + (data.harmony == null ? 0 : data.harmony) }])
         .concat(hideEnergy ? [] : [{ cls: 'energy', text: data.awakened ? ('终 ' + (data.ultimate_turns == null ? 0 : data.ultimate_turns) + ' 能 ' + energyNow + '/' + energyMax) : ('能 ' + energyNow + '/' + energyMax) }])
      : []
        .concat(hideHarmony ? [] : [{ cls: 'harmony', text: '环合 ' + (data.harmony == null ? 0 : data.harmony) + '/2' }])
        .concat(hideEnergy ? [] : [{ cls: 'energy', text: data.awakened ? ('终结 ' + (data.ultimate_turns == null ? 0 : data.ultimate_turns) + ' · 能 ' + energyNow + '/' + energyMax) : ('能量 ' + energyNow + '/' + energyMax) }]);
    if (Number(data.growth) > 0) {
      stats.push({ cls: '', text: '成长 ' + data.growth });
    }
    if (data.id === 'zhenhong' || Number(data.gaze) > 0) {
      stats.push({ cls: '', text: '凝视 ' + (data.gaze == null ? 0 : data.gaze) });
    }
    if (Number(data.brand) > 0) {
      stats.push({ cls: '', text: '焚印 ' + data.brand });
    }
    if (data.pact) {
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
    const note = (page && page.classList.contains('is-tutorial') && !data.shape) ? '' : (data.shape || '无武备');
    const portrait = V2.characterAsset(data, options.usePortrait ? 'portrait' : 'avatar') || V2.characterAsset(data, 'avatar');
    const actions = [];
    if (options.canAwaken) {
      actions.push('<button class="v2-awaken-btn" type="button" data-ultimate-id="' + V2.escapeAttr(data.id) + '">终结</button>');
    }
    const panelSig = [attack, hp, shield, data.harmony, energyNow, energyMax, data.awakened ? 1 : 0].join('|');
    const prevPanel = node.getAttribute('data-pop-sig');
    node.setAttribute('data-pop-sig', panelSig);
    node.innerHTML =
      '<div class="v2-character-art">' + V2.imageMarkup(portrait, data.name, options.usePortrait ? 'v2-portrait' : 'v2-avatar') +
      vitalsHtml + '</div>' +
      '<div class="v2-character-panel">' +
        '<div class="v2-character-heading">' +
          '<strong title="' + V2.escapeAttr(data.name || '') + '">' + V2.escapeHtml(shortName(data)) + '</strong>' +
          (downTurns ? '<span class="v2-down-badge">倒地 ' + downTurns + '</span>' : '') +
          V2.attributeMarkup(data.attribute) +
        '</div>' +
        '<div class="v2-character-stats">' + stats.map(function (item) {
          if (item.html) {
            return item.html;
          }
          return '<span class="' + item.cls + '">' + V2.escapeHtml(item.text) + '</span>';
        }).join('') + '</div>' +
        recordsHtml(options.records) +
        '<div class="v2-character-note"><span title="' + V2.escapeAttr(note) + '">' + V2.escapeHtml(note) + '</span>' + actions.join('') + '</div>' +
      '</div>';
    if (prevPanel && prevPanel !== panelSig) {
      const statsNode = node.querySelector('.v2-character-stats');
      if (statsNode) {
        statsNode.classList.add('v2-stat-pop');
      }
    }
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

  function fillFrontDebuff(node, state) {
    if (!node) {
      return;
    }
    const zone = (state && state.front_debuff) || {};
    const chips = [];
    if (zone.delay) {
      chips.push('延滞');
    }
    if (zone.burn) {
      chips.push('浊燃');
    }
    if (zone.star) {
      chips.push('黯星');
    }
    if (zone.weave) {
      chips.push('覆纹');
    }
    if (zone.stain) {
      chips.push('浸染');
    }
    node.innerHTML = chips.map(function (name) {
      return '<span class="v2-chip">' + V2.escapeHtml(name) + '</span>';
    }).join('');
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
    const chars = V2.sortedCharacters((state && state.characters) || []);
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
    const classes = ['v2-hand-card'];
    const inMulligan = options.phase === 'mulligan';
    const unusable = !inMulligan && !options.legal;
    const reason = unusable ? (card.unavailable_reason || '现在不能使用') : '';
    if (options.legal && !unusable) classes.push('legal');
    if (options.selected) classes.push('selected');
    if (options.raised) classes.push('raised');
    if (unusable) classes.push('is-unusable');
    if (options.groupStart) classes.push('v2-hand-group-start');
    const timing = V2.cardTimingClass(card);
    if (timing) classes.push(timing);
    const portrait = V2.characterAsset(V2.characterFallback(card.character_id) || {}, 'portrait');
    const style = portrait ? ' style="--card-art:url(\'' + String(portrait).replace(/'/g, '%27') + '\')"' : '';
    const instanceAttr = card.instance_id ? ' data-instance-id="' + V2.escapeAttr(card.instance_id) + '"' : '';
    const title = reason ? ' title="' + V2.escapeAttr(reason) + '"' : '';
    const label = (card.name || '卡牌') + (reason ? '，' + reason : '');
    return (
      '<article class="' + classes.join(' ') + '" draggable="false" data-card-id="' + V2.escapeAttr(card.instance_id) + '" data-drag-kind="card" data-card-type="' + V2.escapeAttr(card.type || '') + '"' + instanceAttr + style + title + ' aria-label="' + V2.escapeAttr(label) + '">' +
        (card.copy ? '<span class="v2-chip copy">复制</span>' : '') +
        (card.derived ? '<span class="v2-chip">不可构筑</span>' : '') +
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

  function renderHands(ctx) {
    const mine = TABLE.sideState(ctx.game, ctx.viewer);
    const theirs = TABLE.sideState(ctx.game, ctx.opponent);
    const myHand = TABLE.sortHandCards(Array.isArray(mine.hand) ? mine.hand : [], mine);
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
            selected: Boolean(selected[card.instance_id]),
            raised: ctx.raisedCardId && String(ctx.raisedCardId) === String(card.instance_id),
            phase: ctx.game && ctx.game.phase,
            groupStart: Boolean(index && previous && previous.character_id !== card.character_id),
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
            const portrait = V2.characterAsset(V2.characterFallback(card.character_id) || {}, 'portrait');
            const style = portrait ? ' style="--card-art:url(\'' + String(portrait).replace(/'/g, '%27') + '\')"' : '';
            return '<button class="v2-revealed-card" data-public-copy="' + V2.escapeAttr(card.instance_id || '') + '" data-card-type="' + V2.escapeAttr(card.type || '') + '"' + style + ' aria-label="' + V2.escapeAttr(card.name || '公开卡牌') + '"></button>';
          }
          return '<article class="v2-hidden-card" aria-label="对方手牌"></article>';
        }).join('')
      : '<div class="subtle">对手手牌 0</div>';
    ctx.nodes.handCount.textContent = '手牌 ' + (mine.hand_count != null ? mine.hand_count : myHand.length);
  }

  function historyHeadline(event) {
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
      html = events.slice(-40).map(function (event) {
        const text = historyHeadline(event);
        if (!text) {
          return '';
        }
        const desc = TABLE.historyDescription(event);
        const extra = desc && desc !== text ? V2.cardDescriptionHtml(desc, '', 'subtle') : '';
        return '<div class="v2-log-item"><div>' + V2.escapeHtml(text) + '</div>' + extra + '</div>';
      }).filter(Boolean).join('');
    } else if (logs.length) {
      html = logs.slice(-40).map(function (line) {
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
    popIfChanged(ctx.nodes.ultimate, mine.ultimate_available ? '1' : '0');
    ctx.nodes.playerName.textContent = mine.name || '我方';
    ctx.nodes.opponentName.textContent = theirs.name || '对手';
    const waiting = game.phase === 'mulligan' || mine.ap == null;
    if (ctx.nodes.normalAttack) {
      ctx.nodes.normalAttack.textContent = waiting ? '-' : (mine.normal_attack_available ? '1' : '0');
    }
    if (ctx.nodes.ultimate) {
      ctx.nodes.ultimate.textContent = waiting ? '-' : (mine.ultimate_available ? '1' : '0');
    }
    setPileLabel(ctx.nodes.opponentDeck, '牌库', theirs.deck_count);
    setPileLabel(ctx.nodes.opponentDiscard, '弃牌', (theirs.discard || []).length);
    setPileLabel(ctx.nodes.playerDeck, '牌库', mine.deck_count);
    setPileLabel(ctx.nodes.playerDiscard, '弃牌', (mine.discard || []).length);
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
