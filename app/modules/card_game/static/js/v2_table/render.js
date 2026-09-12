(function (root) {
  const V2 = root.NTE_V2;
  const TABLE = root.NTE_V2_TABLE = root.NTE_V2_TABLE || {};

  function shortName(character) {
    const data = character || {};
    return data.id === 'zero' ? '主角「零」' : (data.name || data.id || '角色');
  }

  function fillCharacterCard(node, character, options) {
    options = options || {};
    const data = Object.assign({}, V2.characterFallback(character && character.id) || {}, character || {});
    const side = options.side || node.getAttribute('data-side') || 'a';
    node.setAttribute('data-entity-id', TABLE.entityId(side, data.id));
    node.setAttribute('data-character-id', data.id || '');
    node.setAttribute('data-drag-kind', 'character');
    node.setAttribute('data-side', side);
    node.setAttribute('data-drop-role', 'character');
    node.classList.toggle('legal', Boolean(options.canAttack || options.canAwaken || options.canTarget));
    node.classList.toggle('down', Number(data.down_turns) > 0);
    node.classList.toggle('is-placeholder', Boolean(options.placeholder));
    node.classList.toggle('target-legal', Boolean(options.canTarget));
    const compact = !options.usePortrait;
    const stats = compact
      ? [
          { cls: 'atk', text: '攻 ' + (data.attack == null ? '-' : data.attack) },
          { cls: 'hp', text: (data.hp == null ? '-' : data.hp) + '/' + (data.max_hp == null ? '-' : data.max_hp) },
          { cls: 'harmony', text: '环 ' + (data.harmony == null ? 0 : data.harmony) },
          { cls: 'energy', text: data.awakened ? '觉醒' : ('能 ' + (data.energy == null ? 0 : data.energy)) },
        ]
      : [
          { cls: 'atk', text: '攻 ' + (data.attack == null ? '-' : data.attack) },
          { cls: 'hp', text: '生命 ' + (data.hp == null ? '-' : data.hp) + '/' + (data.max_hp == null ? '-' : data.max_hp) },
          { cls: '', text: '盾 ' + (data.shield == null ? 0 : data.shield) },
          { cls: 'harmony', text: '环合 ' + (data.harmony == null ? 0 : data.harmony) + '/2' },
          { cls: 'energy', text: data.awakened ? '已觉醒' : ('能量 ' + (data.energy == null ? 0 : data.energy) + '/4') },
        ];
    if (data.id === 'nanali') {
      stats.push({ cls: '', text: '成长 ' + (data.growth == null ? 0 : data.growth) });
    }
    const note = Number(data.down_turns) > 0 ? ('倒地 ' + data.down_turns + ' 回合') : (data.shape || '基础形态');
    const portrait = V2.characterAsset(data, options.usePortrait ? 'portrait' : 'avatar') || V2.characterAsset(data, 'avatar');
    const actions = [];
    if (options.canAttack) {
      actions.push('<button class="v2-attack-btn" type="button" data-attack-character="' + V2.escapeAttr(data.id) + '">出击</button>');
    }
    if (options.canAwaken) {
      actions.push('<button class="v2-awaken-btn" type="button" data-awaken-id="' + V2.escapeAttr(data.id) + '">觉醒</button>');
    }
    actions.push('<button class="v2-inspect" type="button" data-inspect-character="' + V2.escapeAttr(data.id) + '" data-opponent="' + Boolean(options.isOpponent) + '">详情</button>');
    node.innerHTML =
      '<div class="v2-character-art">' + V2.imageMarkup(portrait, data.name, options.usePortrait ? 'v2-portrait' : 'v2-avatar') + '</div>' +
      '<div class="v2-character-panel">' +
        '<div class="v2-character-heading"><strong title="' + V2.escapeAttr(data.name || '') + '">' + V2.escapeHtml(shortName(data)) + '</strong>' + V2.attributeMarkup(data.attribute) + '</div>' +
        '<div class="v2-character-stats">' + stats.map(function (item) {
          return '<span class="' + item.cls + '">' + V2.escapeHtml(item.text) + '</span>';
        }).join('') + '</div>' +
        '<div class="v2-character-note"><span title="' + V2.escapeAttr(note) + '">' + V2.escapeHtml(note) + '</span>' + actions.join('') + '</div>' +
      '</div>';
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

  function emptyFront(slot, label) {
    while (slot.firstChild) {
      slot.removeChild(slot.firstChild);
    }
    const empty = document.createElement('div');
    empty.className = 'v2-front-slot-empty subtle';
    empty.textContent = label || '战斗区空';
    slot.appendChild(empty);
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
    if (zone) zone.classList.toggle('is-empty', !frontId);
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
      canAwaken: canAct && !isOpponent && Boolean(ctx.findAction('awaken', { character_id: frontId })),
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
    chars.forEach(function (character) {
      const wrap = document.createElement('div');
      wrap.className = 'v2-bench-slot';
      if (character.id === frontId) {
        wrap.innerHTML = '<div class="v2-bench-placeholder">出击中</div>';
        bench.appendChild(wrap);
        return;
      }
      const node = ensureCharacterNode(ctx.pool, side, character.id);
      fillCharacterCard(node, character, {
        side: side,
        isOpponent: isOpponent,
        usePortrait: false,
        canAttack: canAct && !isOpponent && Boolean(ctx.findAction('attack', { character_id: character.id })),
        canAwaken: canAct && !isOpponent && Boolean(ctx.findAction('awaken', { character_id: character.id })),
        placeholder: false,
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
    if (options.legal) classes.push('legal');
    if (options.selected) classes.push('selected');
    if (options.raised) classes.push('raised');
    const owner = (V2.characterFallback(card.character_id) || {}).name || '';
    const portrait = V2.characterAsset(V2.characterFallback(card.character_id) || {}, 'portrait');
    const chips = [
      '<span class="v2-chip">' + V2.escapeHtml(V2.cardTypeLabel(card.type, card.terminal)) + '</span>',
    ];
    if (card.copy) chips.push('<span class="v2-chip copy">复制</span>');
    const style = portrait ? ' style="--card-art:url(\'' + String(portrait).replace(/'/g, '%27') + '\')"' : '';
    const instanceAttr = card.instance_id ? ' data-instance-id="' + V2.escapeAttr(card.instance_id) + '"' : '';
    return (
      '<article class="' + classes.join(' ') + '" draggable="false" data-card-id="' + V2.escapeAttr(card.instance_id) + '" data-drag-kind="card" data-card-type="' + V2.escapeAttr(card.type || '') + '"' + instanceAttr + style + '>' +
        '<span class="v2-cost-badge">' + V2.escapeHtml(card.cost == null ? '-' : card.cost) + '</span>' +
        '<div class="v2-stat-row">' + chips.join('') + '</div>' +
        '<span class="v2-card-owner">' + V2.escapeHtml(owner) + '</span>' +
        '<strong>' + V2.escapeHtml(card.name || '卡牌') + '</strong>' +
        '<p class="subtle">' + V2.escapeHtml(card.description || '正在读取卡牌效果。') + '</p>' +
        (card.unavailable_reason && options.phase !== 'mulligan' ? '<small class="v2-unavailable">' + V2.escapeHtml(card.unavailable_reason) + '</small>' : '') +
      '</article>'
    );
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
    if (ctx.nodes.playerLife) ctx.nodes.playerLife.setAttribute('data-player-side', viewer);
    if (ctx.nodes.opponentLife) ctx.nodes.opponentLife.setAttribute('data-player-side', opponent);
    if (ctx.nodes.playerHand) ctx.nodes.playerHand.setAttribute('data-hand-side', viewer);
    if (ctx.nodes.opponentHand) ctx.nodes.opponentHand.setAttribute('data-hand-side', opponent);
  }

  function renderHands(ctx) {
    const mine = TABLE.sideState(ctx.game, ctx.viewer);
    const theirs = TABLE.sideState(ctx.game, ctx.opponent);
    const myHand = Array.isArray(mine.hand) ? mine.hand : [];
    const selected = {};
    (ctx.mulliganIds || []).forEach(function (id) { selected[id] = true; });
    ctx.nodes.playerHand.innerHTML = myHand.length
      ? myHand.map(function (card) {
          const entries = ctx.playEntries(card.instance_id);
          return handCardHtml(card, {
            legal: entries.length > 0 || Boolean(ctx.findAction('cycle', { card_id: card.instance_id })),
            selected: Boolean(selected[card.instance_id]),
            raised: ctx.raisedCardId && String(ctx.raisedCardId) === String(card.instance_id),
            phase: ctx.game && ctx.game.phase,
          });
        }).join('')
      : '<div class="subtle">手牌空</div>';
    const hiddenCount = Number(theirs.hand_count != null ? theirs.hand_count : (theirs.hand || []).length || 0);
    const opponentCards = Array.isArray(theirs.hand) ? theirs.hand.slice() : [];
    while (opponentCards.length < hiddenCount) opponentCards.push({ hidden: true });
    ctx.nodes.opponentHand.innerHTML = hiddenCount
      ? opponentCards.slice(0, hiddenCount).map(function (card) {
          if (card && card.copy) {
            const instanceAttr = card.instance_id ? ' data-instance-id="' + V2.escapeAttr(card.instance_id) + '"' : '';
            return '<button class="v2-public-copy" data-public-copy="' + V2.escapeAttr(card.instance_id) + '"' + instanceAttr + '>复制 · ' + V2.escapeHtml(card.name) + '</button>';
          }
          return '<article class="v2-hidden-card" aria-label="对手手牌"></article>';
        }).join('')
      : '<div class="subtle">对手手牌 0</div>';
    ctx.nodes.handCount.textContent = '手牌 ' + (mine.hand_count != null ? mine.hand_count : myHand.length);
  }

  function renderLogs(ctx) {
    const logs = Array.isArray(ctx.game && ctx.game.logs) ? ctx.game.logs : [];
    const events = Array.isArray(ctx.game && ctx.game.events) ? ctx.game.events : [];
    const lines = logs.length ? logs : events.map(eventSourceText).filter(Boolean);
    ctx.nodes.logList.innerHTML = lines.length
      ? lines.slice(-40).map(function (line) { return '<div class="v2-log-item">' + V2.escapeHtml(line) + '</div>'; }).join('')
      : '<div class="v2-log-item subtle">暂无日志</div>';
    ctx.nodes.logList.scrollTop = ctx.nodes.logList.scrollHeight;
  }

  function renderRecords(ctx) {
    const mine = TABLE.sideState(ctx.game, ctx.viewer).records || [];
    const theirs = TABLE.sideState(ctx.game, ctx.opponent).records || [];
    const all = []
      .concat(mine.map(function (item) { return { item: item, side: '我方' }; }))
      .concat(theirs.map(function (item) { return { item: item, side: '对手' }; }));
    ctx.nodes.records.innerHTML = all.length
      ? all.map(function (entry) { return '<span class="v2-record-side">' + entry.side + '</span>' + recordHtml(entry.item); }).join('')
      : '<span class="subtle">无浔记录</span>';
  }

  function playHistoryHtml(item) {
    const card = item && item.card || {};
    const name = card.name || '公开出牌';
    const owner = item.side === 'a' || item.side === 'b' ? (item.side === (item.viewer) ? '我方' : '对手') : '';
    return '<button class="v2-play-chip" type="button" data-play-seq="' + V2.escapeAttr(item.seq) + '">' +
      V2.escapeHtml((owner ? owner + ' · ' : '') + name) +
      '</button>';
  }

  function renderPlayHistory(ctx) {
    const history = Array.isArray(ctx.playHistory) ? ctx.playHistory : [];
    if (ctx.nodes.playHistory) {
      ctx.nodes.playHistory.innerHTML = history.length
        ? history.slice(-6).map(playHistoryHtml).join('')
        : '<span class="subtle">无公开出牌</span>';
    }
    if (ctx.nodes.playHistoryDetail) {
      ctx.nodes.playHistoryDetail.innerHTML = history.length
        ? history.slice().reverse().map(function (item) {
            const card = item.card || {};
            return '<div class="v2-log-item"><strong>' + V2.escapeHtml(card.name || '公开出牌') + '</strong>' +
              '<p class="subtle">' + V2.escapeHtml(V2.cardTypeLabel(card.type, card.terminal) + (card.cost != null ? ' · 费用 ' + card.cost : '')) + '</p>' +
              '<p>' + V2.escapeHtml(item.text || card.description || '') + '</p></div>';
          }).join('')
        : '<div class="v2-log-item subtle">跳过演出后仍可在此查看公开出牌。</div>';
    }
  }

  function renderGame(ctx) {
    const game = ctx.game;
    const mine = TABLE.sideState(game, ctx.viewer);
    const theirs = TABLE.sideState(game, ctx.opponent);
    bindHudSides(ctx);
    ctx.nodes.roundLabel.textContent = game && game.turn ? ('第 ' + game.turn + ' 回合') : '第 - 回合';
    ctx.nodes.opponentHp.textContent = theirs.hp == null ? '-' : theirs.hp;
    ctx.nodes.opponentShield.textContent = '护盾 ' + (theirs.shield == null ? '-' : theirs.shield);
    if (ctx.nodes.opponentAp) ctx.nodes.opponentAp.textContent = '行动力 ' + (theirs.ap == null ? '-' : theirs.ap);
    ctx.nodes.playerHp.textContent = mine.hp == null ? '-' : mine.hp;
    ctx.nodes.playerShield.textContent = '护盾 ' + (mine.shield == null ? '-' : mine.shield);
    ctx.nodes.playerAp.textContent = mine.ap == null ? '-' : mine.ap;
    ctx.nodes.playerName.textContent = mine.name || '我方';
    ctx.nodes.opponentName.textContent = theirs.name || '对手';
    ctx.nodes.normalAttack.textContent = '普通出击机会 ' + (game.phase === 'mulligan' ? '待开局' : mine.normal_attack_available ? '未用' : '已用');
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
    renderPlayHistory(ctx);
    ctx.nodes.turnCopy.textContent = V2.roomStatusLabel(ctx.room, game);
    if (ctx.nodes.waitChip) {
      ctx.nodes.waitChip.textContent = ctx.busy ? '演出中' : (game && game.is_my_turn ? '你的回合' : '对手回合');
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
  TABLE.renderPlayHistory = renderPlayHistory;
}(window));
