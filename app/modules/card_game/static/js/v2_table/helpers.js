(function (root) {
  const V2 = root.NTE_V2;
  const TABLE = root.NTE_V2_TABLE = root.NTE_V2_TABLE || {};
  const OPTIONS_KEY = 'nte_v2_table_options';
  const CHARACTER_ORDER = (V2 && V2.CHARACTER_ORDER) || ['nanali', 'zero', 'jiuyuan', 'xun'];

  function clone(value) {
    if (value == null) {
      return value;
    }
    try {
      return JSON.parse(JSON.stringify(value));
    } catch (error) {
      return value;
    }
  }

  function entityId(side, id) {
    if (!id && id !== 0) {
      return null;
    }
    const token = String(id);
    if (token.indexOf(':') >= 0) {
      return token;
    }
    if (token === 'player') {
      return String(side || 'a') + ':player';
    }
    return String(side || 'a') + ':' + token;
  }

  function parseEntityId(value) {
    const token = String(value || '');
    const index = token.indexOf(':');
    if (index < 0) {
      return { side: null, id: token };
    }
    return { side: token.slice(0, index), id: token.slice(index + 1) };
  }

  function canonicalEntityId(value) {
    const token = String(value || '');
    if (!token || token.indexOf(':') < 0) {
      return null;
    }
    return token;
  }

  function sameEntityId(left, right) {
    const a = canonicalEntityId(left);
    const b = canonicalEntityId(right);
    return Boolean(a && b && a === b);
  }

  function viewerSide(game) {
    return V2.viewerSide(game);
  }

  function opponentSide(game) {
    return V2.opponentSide(game);
  }

  function sideState(game, side) {
    return V2.sideState(game, side) || {};
  }

  function characterOrderIndex(side, characterId) {
    const list = Array.isArray(side && side.characters) ? side.characters : [];
    for (let index = 0; index < list.length; index += 1) {
      if (list[index] && list[index].id === characterId) {
        return index;
      }
    }
    return list.length;
  }

  function sortHandCards(cards, side) {
    const typeOrder = { tactic: 0, battle: 1, form: 2 };
    const list = Array.isArray(cards) ? cards.slice() : [];
    return list.sort(function (left, right) {
      left = left && (left.hand_face || left);
      right = right && (right.hand_face || right);
      const byOwner = characterOrderIndex(side, left && left.character_id) - characterOrderIndex(side, right && right.character_id);
      if (byOwner) {
        return byOwner;
      }
      return (typeOrder[left && left.type] || 9) - (typeOrder[right && right.type] || 9);
    });
  }

  function characterMap(side) {
    const map = {};
    ((side && side.characters) || []).forEach(function (item) {
      if (item && item.id) {
        map[item.id] = item;
      }
    });
    return map;
  }

  function harmonySourceCharacter(state) {
    const list = (state && state.characters) || [];
    for (let index = 0; index < list.length; index += 1) {
      if (list[index] && list[index].harmony_source) {
        return list[index];
      }
    }
    return null;
  }

  function harmonySourceCaptionHtml(character) {
    if (!character) {
      return '';
    }
    const data = Object.assign({}, V2.characterFallback(character.id) || {}, character);
    const name = data.name || data.id || '角色';
    const attribute = data.attribute || '';
    const icon = V2.elementIcon(attribute);
    const iconHtml = icon
      ? V2.imageMarkup(icon, attribute, 'v2-harmony-source-icon')
      : (attribute ? V2.escapeHtml(attribute) : '');
    return '环合来源方：' + iconHtml + V2.escapeHtml(name);
  }

  function findCharacter(game, characterId, preferredSide) {
    const parsed = parseEntityId(characterId);
    const wanted = parsed.id;
    const sides = [];
    if (parsed.side) {
      sides.push(parsed.side);
    }
    if (preferredSide) {
      sides.push(preferredSide);
    }
    if (game) {
      sides.push(viewerSide(game), opponentSide(game), 'a', 'b');
    }
    for (let index = 0; index < sides.length; index += 1) {
      const side = sides[index];
      const found = characterMap(sideState(game, side))[wanted];
      if (found) {
        return { character: found, side: side };
      }
    }
    const fallback = V2.characterFallback(wanted);
    return fallback ? { character: fallback, side: preferredSide || parsed.side || 'a' } : null;
  }

  function cardByInstance(game, instanceId) {
    const wanted = String(instanceId || '');
    if (!wanted || !game) {
      return null;
    }
    const piles = [];
    ['a', 'b'].forEach(function (side) {
      const state = sideState(game, side);
      piles.push.apply(piles, state.hand || []);
      piles.push.apply(piles, state.discard || []);
    });
    if (game.resolving_card) {
      piles.push(game.resolving_card);
    }
    return piles.find(function (card) {
      return card && String(card.instance_id) === wanted;
    }) || null;
  }

  function mergeCharacters(existing, patchList) {
    const byId = {};
    (existing || []).forEach(function (item) {
      if (item && item.id) {
        byId[item.id] = Object.assign({}, item);
      }
    });
    (patchList || []).forEach(function (item) {
      if (!item || !item.id) {
        return;
      }
      byId[item.id] = Object.assign({}, byId[item.id] || { id: item.id }, item);
    });
    const seen = {};
    const next = [];
    (existing || []).forEach(function (item) {
      if (!item || !item.id) {
        return;
      }
      next.push(byId[item.id]);
      seen[item.id] = true;
    });
    (patchList || []).forEach(function (item) {
      if (item && item.id && !seen[item.id]) {
        next.push(byId[item.id]);
      }
    });
    return next.filter(function (item) { return !item.removed; });
  }

  function hiddenCard() {
    return { hidden: true };
  }

  function isHiddenCard(card) {
    return !card || card.hidden || (!card.copy && !card.instance_id && !card.name);
  }

  function isPublicCopy(card) {
    return Boolean(card && card.copy && (card.instance_id || card.name));
  }

  function reconcileHandCount(team, count, options) {
    options = options || {};
    const next = Object.assign({}, team || {});
    const hand = Array.isArray(next.hand) ? next.hand.slice() : [];
    const target = Number(count);
    if (!Number.isFinite(target)) {
      next.hand = hand;
      return next;
    }
    next.hand_count = target;
    while (hand.length > target) {
      let index = -1;
      for (let i = hand.length - 1; i >= 0; i -= 1) {
        if (isHiddenCard(hand[i]) && !isPublicCopy(hand[i])) {
          index = i;
          break;
        }
      }
      if (index < 0) {
        if (options.allowNamed) {
          index = hand.length - 1;
        } else {
          break;
        }
      }
      hand.splice(index, 1);
    }
    while (hand.length < target) {
      if (options.allowNamed || options.inventNamed) {
        break;
      }
      hand.push(hiddenCard());
    }
    next.hand = hand;
    return next;
  }

  function copyViewerHand(display, live, viewer) {
    if (!display || !live || !viewer) {
      return display;
    }
    const from = sideState(live, viewer);
    const to = sideState(display, viewer);
    if (!from || !to) {
      return display;
    }
    to.hand = clone(from.hand || []);
    to.hand_count = from.hand_count != null ? from.hand_count : to.hand.length;
    return display;
  }

  function removePlayedCard(hand, event, isSelf) {
    const next = Array.isArray(hand) ? hand.slice() : [];
    const card = event && event.card;
    if (isSelf && card && card.instance_id) {
      const index = next.findIndex(function (item) {
        return item && String(item.instance_id) === String(card.instance_id);
      });
      if (index >= 0) {
        next.splice(index, 1);
        return next;
      }
    }
    if (!isSelf && card && card.copy && card.instance_id) {
      const copyIndex = next.findIndex(function (item) {
        return item && String(item.instance_id) === String(card.instance_id);
      });
      if (copyIndex >= 0) {
        next.splice(copyIndex, 1);
        return next;
      }
    }
    if (!isSelf) {
      const backIndex = next.findIndex(function (item) {
        return isHiddenCard(item) && !isPublicCopy(item);
      });
      if (backIndex >= 0) {
        next.splice(backIndex, 1);
      }
    }
    return next;
  }

  function applyEventToHand(game, event) {
    if (!game || !event || !event.side || !game.sides || !game.sides[event.side]) {
      return game;
    }
    const type = String(event.type || '');
    if (['play', 'discard', 'remove', 'draw', 'gain', 'mulligan'].indexOf(type) < 0) {
      return game;
    }
    const next = clone(game);
    const team = Object.assign({}, next.sides[event.side] || {});
    const isSelf = event.side === viewerSide(next);
    let hand = Array.isArray(team.hand) ? team.hand.slice() : [];
    if (type === 'mulligan' && isSelf) {
      const outgoing = (event.card_ids || []).map(String);
      hand = hand.filter(function (card) { return !outgoing.includes(String(card && card.instance_id)); });
      team.hand_count = hand.length;
    } else if (type === 'play' || type === 'discard' || type === 'remove') {
      hand = removePlayedCard(hand, event, isSelf);
      if (team.hand_count != null) {
        team.hand_count = Math.max(0, Number(team.hand_count) - 1);
      }
    } else if (type === 'draw' || type === 'gain') {
      if (isSelf && event.card && !event.card.hidden) {
        const exists = hand.some(function (item) {
          return item && event.card.instance_id && String(item.instance_id) === String(event.card.instance_id);
        });
        if (!exists) {
          hand.push(clone(event.card));
        }
      } else if (isSelf) {
        // Own draws without a visible card must not become opponent-style backs.
      } else if (!isSelf) {
        if (event.card && event.card.copy) {
          hand.push(clone(event.card));
        } else {
          hand.push(hiddenCard());
        }
      }
      if (team.hand_count != null) {
        team.hand_count = Number(team.hand_count) + 1;
      }
    }
    team.hand = hand;
    next.sides[event.side] = team;
    return next;
  }

  function applyPatch(game, patch, event) {
    if (!game) {
      return patch ? clone(patch) : game;
    }
    let next = clone(game);
    if (event) {
      next = applyEventToHand(next, event);
    }
    if (!patch || typeof patch !== 'object') {
      return next;
    }
    ['phase', 'active_side', 'turn', 'winner', 'reason', 'escalation'].forEach(function (key) {
      if (patch[key] !== undefined) {
        next[key] = patch[key];
      }
    });
    if (patch.sides && typeof patch.sides === 'object') {
      next.sides = next.sides || {};
      Object.keys(patch.sides).forEach(function (side) {
        const src = Object.assign({}, patch.sides[side] || {});
        // Older replay patches hid retained opening cards despite known outgoing IDs.
        // Recover only when every placeholder matches an already known survivor.
        if (game.phase === 'mulligan' && side === viewerSide(next) && Array.isArray(src.hand)) {
          const known = (next.sides[side].hand || []).filter(function (card) { return !isHiddenCard(card); });
          const named = new Set(src.hand.filter(function (card) { return !isHiddenCard(card); }).map(function (card) { return String(card.instance_id); }));
          const survivors = known.filter(function (card) { return !named.has(String(card.instance_id)); });
          const hiddenCount = src.hand.filter(isHiddenCard).length;
          if (hiddenCount && survivors.length === hiddenCount && known.length === src.hand.length
              && known.length === (next.sides[side].hand || []).length) {
            let index = 0;
            src.hand = src.hand.map(function (card) { return isHiddenCard(card) ? clone(survivors[index++]) : card; });
          }
        }
        let dst = Object.assign({}, next.sides[side] || {});
        Object.keys(src).forEach(function (key) {
          if (key === 'characters') {
            dst.characters = mergeCharacters(dst.characters, src.characters);
            return;
          }
          if (key === 'hand') {
            dst.hand = clone(src.hand);
            return;
          }
          if (key === 'hand_count') {
            return;
          }
          dst[key] = clone(src[key]);
        });
        if (src.hand) {
          dst.hand = clone(src.hand);
        }
        if (src.hand_count != null) {
          dst = reconcileHandCount(dst, src.hand_count, {
            allowNamed: side === viewerSide(next),
          });
        }
        next.sides[side] = dst;
      });
    }
    return next;
  }

  function historyDescription(event) {
    if (!event) {
      return '';
    }
    const card = event.card || (event.item && event.item.card) || null;
    if (card && !card.hidden && card.description) {
      return String(card.description).trim();
    }
    if (event.description) {
      return String(event.description).trim();
    }
    return '';
  }

  function publicPlayFromEvent(event) {
    if (!event || event.type !== 'play') {
      return null;
    }
    const card = event.card;
    if (!card || card.hidden || event.private_side) {
      return null;
    }
    if (!card.name && !card.card_id && !card.instance_id) {
      return null;
    }
    return {
      seq: event.seq,
      side: event.side,
      text: event.text || '',
      actor: event.actor || null,
      card: clone(card),
    };
  }

  function publicPlayOwner(side, viewer) {
    if (side !== 'a' && side !== 'b') {
      return '';
    }
    return side === viewer ? '我方' : '对手';
  }

  function appendPresentedEvent(game, event) {
    if (!game || !event) {
      return game;
    }
    const events = Array.isArray(game.events) ? game.events.slice() : [];
    const seq = Number(event.seq);
    if (Number.isFinite(seq) && events.some(function (item) {
      return Number(item && item.seq) === seq;
    })) {
      return game;
    }
    events.push(clone(event));
    game.events = events;
    game.logs = events.map(function (item) {
      return (item && item.text) || '';
    });
    return game;
  }

  function appendPublicPlay(history, event) {
    const item = publicPlayFromEvent(event);
    const list = Array.isArray(history) ? history.slice() : [];
    if (!item) {
      return list;
    }
    const exists = list.some(function (row) {
      return Number(row && row.seq) === Number(item.seq);
    });
    if (exists) {
      return list;
    }
    list.push(item);
    return list.slice(-12);
  }

  function presentationFrom(payload, game) {
    const raw = (payload && payload.presentation) || null;
    const events = Array.isArray(raw && raw.events)
      ? raw.events
      : (Array.isArray(game && game.events) ? game.events : []);
    const seqs = events.map(function (event) { return Number(event && event.seq); }).filter(function (value) {
      return Number.isFinite(value);
    });
    const cursor = raw && raw.cursor != null ? Number(raw.cursor) : (seqs.length ? Math.max.apply(null, seqs) : -1);
    const oldest = raw && raw.oldest_seq != null ? Number(raw.oldest_seq) : (seqs.length ? Math.min.apply(null, seqs) : 0);
    return {
      schema_version: raw && raw.schema_version ? raw.schema_version : 1,
      cursor: Number.isFinite(cursor) ? cursor : -1,
      oldest_seq: Number.isFinite(oldest) ? oldest : 0,
      events: events,
    };
  }

  function eventsAfter(events, cursor) {
    return (events || []).filter(function (event) {
      const seq = Number(event && event.seq);
      return Number.isFinite(seq) && seq > Number(cursor);
    });
  }

  function splitPresentationAtTurn(events) {
    const list = Array.isArray(events) ? events : [];
    let index = -1;
    for (let i = 0; i < list.length; i += 1) {
      if (list[i] && list[i].type === 'turn') {
        index = i;
        break;
      }
    }
    if (index < 0) {
      return { settle: list.slice(), follow: [] };
    }
    return { settle: list.slice(0, index), follow: list.slice(index) };
  }

  function applyEventPatches(board, events) {
    let next = board;
    (Array.isArray(events) ? events : []).forEach(function (event) {
      next = applyPatch(next, (event && event.patch) || {}, event);
    });
    return next;
  }

  function loadOptions() {
    const fallback = { muted: false, reducedMotion: false, speed: 1, dragTargeting: true, fontSize: 2, layoutScale: 1 };
    try {
      const raw = root.localStorage.getItem(OPTIONS_KEY);
      if (!raw) {
        return fallback;
      }
      const parsed = JSON.parse(raw);
      return {
        muted: Boolean(parsed && parsed.muted),
        dragTargeting: !parsed || parsed.dragTargeting !== false,
        reducedMotion: Boolean(parsed && parsed.reducedMotion),
        speed: Number(parsed && parsed.speed) === 2 ? 2 : 1,
        layoutScale: parsed && [0.8, 1, 1.5, 2].includes(parsed.layoutScale) ? parsed.layoutScale : 1,
        fontSize: parsed && Number.isInteger(parsed.fontSize) && parsed.fontSize >= 0 && parsed.fontSize <= 4 ? parsed.fontSize : 2,
      };
    } catch (error) {
      return fallback;
    }
  }

  function saveOptions(options) {
    try {
      root.localStorage.setItem(OPTIONS_KEY, JSON.stringify(options));
    } catch (error) {
      // ignore private-mode persistence failures
    }
  }

  function actionTargetId(action) {
    return canonicalEntityId(action && action.target_id);
  }

  function inferInteraction(game, entry) {
    const action = (entry && entry.action) || {};
    const side = viewerSide(game);
    if (action.type === 'attack') {
      return { kind: 'attack', actor_id: entityId(side, action.character_id), target_ids: [] };
    }
    if (action.type === 'ultimate') {
      return { kind: 'ultimate', actor_id: entityId(side, action.character_id), target_ids: [] };
    }
    if (action.type === 'play_card') {
      const card = cardByInstance(game, action.card_id);
      if (card && card.type === 'form') {
        const selfId = entityId(side, card.character_id);
        return { kind: 'form', actor_id: selfId, target_ids: [selfId] };
      }
      const target = actionTargetId(action);
      if (target) {
        return {
          kind: 'target',
          actor_id: card ? entityId(side, card.character_id) : null,
          target_ids: [target],
        };
      }
      if (action.target_id) {
        return { kind: 'choice', actor_id: card ? entityId(side, card.character_id) : null, target_ids: [] };
      }
      return { kind: 'cast', actor_id: card ? entityId(side, card.character_id) : null, target_ids: [] };
    }
    if (action.type === 'choose' || action.type === 'cycle' || action.type === 'mulligan') {
      return { kind: 'choice', actor_id: null, target_ids: [] };
    }
    return { kind: 'other', actor_id: null, target_ids: [] };
  }

  function interactionOf(game, entry) {
    const raw = entry && entry.interaction;
    if (raw && raw.kind) {
      const targets = Array.isArray(raw.target_ids)
        ? raw.target_ids.map(canonicalEntityId).filter(Boolean)
        : [];
      return {
        kind: raw.kind,
        actor_id: raw.actor_id || null,
        target_ids: targets,
      };
    }
    return inferInteraction(game, entry);
  }

  function entryMatchesTarget(entry, targetId) {
    if (!entry || !targetId) {
      return false;
    }
    const wanted = canonicalEntityId(targetId);
    if (!wanted) {
      return false;
    }
    const actionTarget = actionTargetId(entry.action);
    if (actionTarget) {
      return actionTarget === wanted;
    }
    const ids = (entry.interaction && entry.interaction.target_ids) || [];
    return ids.indexOf(wanted) >= 0;
  }

  function legalEntries(game) {
    return V2.legalEntries(game);
  }

  function playOptionAtPoint(card, entries, rect, point) {
    const options = (card && card.play_options) || [];
    if (!options.length || !rect || !point || rect.width <= 0 || point.x < rect.left
        || point.x >= rect.right || point.y < rect.top || point.y >= rect.bottom) return null;
    const index = Math.min(options.length - 1, Math.floor((point.x - rect.left) / rect.width * options.length));
    const option = options[index];
    return (entries || []).find(function (entry) { return entry.action.option_id === option.id; }) || null;
  }

  function groupPlayEntries(game, entries) {
    const groups = { cast: [], target: [], form: [], other: [] };
    (entries || []).forEach(function (entry) {
      const interaction = interactionOf(game, entry);
      entry.interaction = interaction;
      if (interaction.kind === 'form') groups.form.push(entry);
      else if (interaction.kind === 'target') groups.target.push(entry);
      else if (interaction.kind === 'cast' || interaction.kind === 'attack') groups.cast.push(entry);
      else groups.other.push(entry);
    });
    return { entries: entries || [], groups: groups };
  }

  function isPlayAreaRole(role) {
    return role === 'cast' || role === 'play-area' || role === 'player-front' || role === 'opponent-front' || role === 'combat-core';
  }

  function legalDropFor(kind, grouped, zone, attackEntry, options) {
    if (!zone) return null;
    options = options || {};
    const roleNode = zone.closest ? zone.closest('[data-drop-role]') : null;
    const role = roleNode && roleNode.getAttribute ? roleNode.getAttribute('data-drop-role') : '';
    const entityNode = zone.closest ? zone.closest('[data-entity-id]') : null;
    if (kind === 'character') {
      if (!attackEntry) {
        return null;
      }
      if (options.fromFront) {
        return role === 'opponent-front' ? attackEntry : null;
      }
      return isPlayAreaRole(role) ? attackEntry : null;
    }
    grouped = grouped || { entries: [], groups: { cast: [], target: [], form: [], other: [] } };
    if (!grouped.entries.length) {
      return null;
    }
    const targetId = canonicalEntityId(entityNode && entityNode.getAttribute && entityNode.getAttribute('data-entity-id'));
    if (targetId) {
      const match = grouped.groups.target.concat(grouped.groups.form).find(function (entry) {
        return entryMatchesTarget(entry, targetId);
      });
      if (match) {
        return match;
      }
    }
    if (isPlayAreaRole(role)) {
      return grouped.groups.cast[0] || grouped.groups.form[0] || grouped.groups.target[0] || grouped.entries[0] || null;
    }
    return null;
  }

  function resourceLine(character) {
    const data = character || {};
    const bits = [
      '攻击 ' + (data.attack == null ? '-' : data.attack),
      '生命 ' + (data.hp == null ? '-' : data.hp) + '/' + (data.max_hp == null ? '-' : data.max_hp),
      '护盾 ' + (data.shield == null ? 0 : data.shield),
      '环合 ' + (data.harmony == null ? 0 : data.harmony),
      (data.resource_name || '能量') + ' ' + (data.energy == null ? 0 : data.energy) + '/' + (data.energy_max == null ? 5 : data.energy_max),
    ];
    if (data.protagonist_aura != null) bits.push('主角光环 ' + data.protagonist_aura + '/6');
    if (data.truth_keys != null) bits.push('真理之匙 ' + data.truth_keys);
    if (data.arid != null) bits.push('荒时 ' + data.arid);
    if (data.growth != null && data.growth !== '') {
      bits.push('成长 ' + data.growth);
    }
    if (data.shape) {
      bits.push('武备 ' + data.shape);
    }
    if (data.awakened) {
      bits.push(data.ultimate_turns ? ('终结 ' + data.ultimate_turns) : '终结中');
    }
    if (data.collapse) {
      bits.push('倾陷');
    } else if (data.collapse_count) {
      bits.push('倾陷 ' + data.collapse_count + '/' + (data.collapse_max == null ? 5 : data.collapse_max));
    }
    if (Number(data.down_turns) > 0) {
      bits.push('倒地 ' + data.down_turns);
    }
    return bits.join(' · ');
  }

  function FallbackPlayer(config) {
    this.root = config && config.root;
    this.applyPatch = (config && config.applyPatch) || function () {};
    this.onBusy = (config && config.onBusy) || function () {};
    this.onPublicCard = (config && config.onPublicCard) || function () {};
    this.options = { muted: false, reducedMotion: false, speed: 1 };
    this._cancelled = false;
    this._seen = {};
  }

  FallbackPlayer.prototype.setOptions = function (options) {
    this.options = Object.assign({}, this.options, options || {});
  };

  FallbackPlayer.prototype.cancel = function () {
    this._cancelled = true;
    this.onBusy(false);
  };

  FallbackPlayer.prototype.play = function (events, playOptions) {
    const self = this;
    this._cancelled = false;
    const list = Array.isArray(events) ? events : [];
    this.onBusy(true);
    return Promise.resolve().then(function () {
      list.forEach(function (event) {
        if (self._cancelled) {
          return;
        }
        const key = String((event && event.seq) + ':' + (event && event.action_id));
        if (self._seen[key]) {
          return;
        }
        self._seen[key] = true;
        if (event && event.card) {
          self.onPublicCard(event.card, event);
        }
        self.applyPatch((event && event.patch) || {}, event);
      });
      self.onBusy(false);
      return { instant: Boolean(playOptions && playOptions.instant), cancelled: self._cancelled };
    });
  };

  function createPlayer(config) {
    const Player = root.NTEDuelPresentation && root.NTEDuelPresentation.Player;
    if (typeof Player === 'function') {
      return new Player(config);
    }
    return new FallbackPlayer(config);
  }

  function mulliganCardNode(rail, id) {
    if (!rail || typeof rail.querySelector !== 'function' || id == null || id === '') {
      return null;
    }
    const key = String(id);
    try {
      return rail.querySelector('[data-instance-id="' + key + '"]')
        || rail.querySelector('[data-card-id="' + key + '"]');
    } catch (_error) {
      return null;
    }
  }

  function animateMulliganSwap(config) {
    config = config || {};
    const rail = config.rail;
    const outgoingIds = (config.outgoingIds || []).map(String).filter(Boolean);
    const incomingIds = (config.incomingIds || []).map(String).filter(Boolean);
    const reduced = Boolean(config.reducedMotion);
    const replace = typeof config.replace === 'function' ? config.replace : function () {};
    const cancelled = typeof config.cancelled === 'function' ? config.cancelled : function () { return false; };
    const selectMs = reduced ? 0 : 380;
    const moveMs = reduced ? 0 : 420;
    const holdMs = reduced ? 0 : 1000;
    const wait = typeof config.wait === 'function'
      ? config.wait
      : function (ms) {
        if (!ms || cancelled()) return Promise.resolve();
        return new Promise(function (resolve) {
          const timer = setTimeout(resolve, ms);
          if (cancelled()) {
            clearTimeout(timer);
            resolve();
          }
        });
      };

    function mark(ids, className) {
      ids.forEach(function (id) {
        const node = mulliganCardNode(rail, id);
        if (node && node.classList) node.classList.add(className);
      });
    }

    if (typeof config.replaceOne === 'function') {
      return (async function () {
        for (let index = 0; index < outgoingIds.length; index += 1) {
          if (cancelled()) return;
          const outgoing = outgoingIds[index];
          mark([outgoing], 'selected');
          mark([outgoing], 'is-leaving');
          await wait(moveMs);
          if (cancelled()) return;
          config.replaceOne(outgoing, incomingIds[index], index);
          mark([incomingIds[index]], 'is-entering');
          await wait(moveMs);
          if (cancelled()) return;
          const node = mulliganCardNode(rail, incomingIds[index]);
          if (node && node.classList) node.classList.remove('is-entering');
        }
      })();
    }

    return Promise.resolve().then(function () {
      if (cancelled()) return;
      mark(outgoingIds, 'selected');
      return wait(outgoingIds.length ? selectMs : 0);
    }).then(function () {
      if (cancelled()) return;
      mark(outgoingIds, 'is-leaving');
      return wait(outgoingIds.length ? moveMs : 0);
    }).then(function () {
      if (cancelled()) return;
      replace();
      mark(incomingIds, 'is-entering');
      return wait(incomingIds.length ? moveMs : 0);
    }).then(function () {
      if (cancelled() || !incomingIds.length) return;
      return wait(holdMs);
    });
  }

  TABLE.OPTIONS_KEY = OPTIONS_KEY;
  TABLE.CHARACTER_ORDER = CHARACTER_ORDER;
  TABLE.clone = clone;
  TABLE.entityId = entityId;
  TABLE.parseEntityId = parseEntityId;
  TABLE.canonicalEntityId = canonicalEntityId;
  TABLE.sameEntityId = sameEntityId;
  TABLE.viewerSide = viewerSide;
  TABLE.opponentSide = opponentSide;
  TABLE.sideState = sideState;
  TABLE.characterOrderIndex = characterOrderIndex;
  TABLE.sortHandCards = sortHandCards;
  TABLE.characterMap = characterMap;
  TABLE.harmonySourceCharacter = harmonySourceCharacter;
  TABLE.harmonySourceCaptionHtml = harmonySourceCaptionHtml;
  TABLE.findCharacter = findCharacter;
  TABLE.cardByInstance = cardByInstance;
  TABLE.applyPatch = applyPatch;
  TABLE.applyEventToHand = applyEventToHand;
  TABLE.reconcileHandCount = reconcileHandCount;
  TABLE.copyViewerHand = copyViewerHand;
  TABLE.historyDescription = historyDescription;
  TABLE.publicPlayFromEvent = publicPlayFromEvent;
  TABLE.publicPlayOwner = publicPlayOwner;
  TABLE.appendPresentedEvent = appendPresentedEvent;
  TABLE.appendPublicPlay = appendPublicPlay;
  TABLE.presentationFrom = presentationFrom;
  TABLE.eventsAfter = eventsAfter;
  TABLE.splitPresentationAtTurn = splitPresentationAtTurn;
  TABLE.applyEventPatches = applyEventPatches;
  TABLE.loadOptions = loadOptions;
  TABLE.saveOptions = saveOptions;
  TABLE.interactionOf = interactionOf;
  TABLE.inferInteraction = inferInteraction;
  TABLE.entryMatchesTarget = entryMatchesTarget;
  TABLE.actionTargetId = actionTargetId;
  TABLE.legalEntries = legalEntries;
  TABLE.groupPlayEntries = groupPlayEntries;
  TABLE.playOptionAtPoint = playOptionAtPoint;
  TABLE.isPlayAreaRole = isPlayAreaRole;
  TABLE.legalDropFor = legalDropFor;
  TABLE.resourceLine = resourceLine;
  TABLE.FallbackPlayer = FallbackPlayer;
  TABLE.createPlayer = createPlayer;
  TABLE.mulliganCardNode = mulliganCardNode;
  TABLE.animateMulliganSwap = animateMulliganSwap;
}(window));
