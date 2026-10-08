const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function loadTable() {
  const helpersPath = path.resolve('app/modules/card_game/static/js/v2_table/helpers.js');
  const context = {
    window: { localStorage: { getItem() { return null; }, setItem() {} } },
    console,
    document: {
      getElementById() { return null; },
      querySelector() { return null; },
      querySelectorAll() { return []; },
    },
  };
  context.window.NTE_V2 = {
    viewerSide: (game) => (game && game.viewer_side) || 'a',
    opponentSide: (game) => ((game && game.viewer_side) === 'b' ? 'a' : 'b'),
    sideState: (game, side) => ((game && game.sides) || {})[side] || {},
    characterFallback: () => null,
    legalEntries: (game) => (game && game.legal_actions) || [],
    findLegalEntry: (game, predicate) => ((game && game.legal_actions) || []).find((entry) => predicate(entry.action, entry)) || null,
    findLegalEntries: (game, predicate) => ((game && game.legal_actions) || []).filter((entry) => predicate(entry.action, entry)),
    sameIdSet: (left, right) => JSON.stringify(left || []) === JSON.stringify(right || []),
    escapeHtml: String,
    escapeAttr: String,
    elementIcon: (attribute) => (attribute ? '/static/images/elements/' + encodeURIComponent(attribute) + '.png' : ''),
    imageMarkup: (src, alt, className) => (src ? '<img class="' + className + '" src="' + src + '" alt="' + (alt || '') + '">' : ''),
    previewMarkup: () => '',
    setHidden() {},
    cardTypeLabel: (type) => type || '',
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(helpersPath, 'utf8'), context);
  return context.window.NTE_V2_TABLE;
}

function sampleGame() {
  return {
    viewer_side: 'a',
    sides: {
      a: {
        hp: 30,
        ap: 2,
        hand: [
          { instance_id: 'a-n06', card_id: 'N06', character_id: 'nanali', name: '共赴死局', type: 'tactic', cost: 1 },
          { instance_id: 'a-n07', card_id: 'N07', character_id: 'nanali', name: '预备备', type: 'form', cost: 1 },
        ],
        characters: [{ id: 'nanali' }, { id: 'zero' }, { id: 'jiuyuan' }, { id: 'xun' }],
      },
      b: {
        hp: 30,
        ap: 2,
        hand: [{ hidden: true }, { hidden: true }, { copy: true, instance_id: 'b-copy', name: '复制牌' }],
        hand_count: 3,
        characters: [{ id: 'nanali' }, { id: 'zero' }, { id: 'jiuyuan' }, { id: 'xun' }],
      },
    },
    legal_actions: [
      {
        action: { type: 'play_card', card_id: 'a-n06', target_id: 'a:zero' },
        interaction: { kind: 'target', actor_id: 'a:nanali', target_ids: ['a:zero'] },
      },
      {
        action: { type: 'play_card', card_id: 'a-n06', target_id: 'a:jiuyuan' },
        interaction: { kind: 'target', actor_id: 'a:nanali', target_ids: ['a:jiuyuan'] },
      },
      {
        action: { type: 'play_card', card_id: 'a-n02' },
        interaction: { kind: 'form', actor_id: 'a:nanali', target_ids: ['a:nanali'] },
      },
    ],
  };
}

function zoneFor(entityId) {
  return {
    closest(sel) {
      if (sel === '[data-entity-id]') {
        return { getAttribute: () => entityId };
      }
      return { getAttribute: () => 'character' };
    },
  };
}

function zoneWith(entityId, role) {
  return {
    closest(sel) {
      if (sel === '[data-entity-id]') {
        return entityId ? { getAttribute: () => entityId } : null;
      }
      return { getAttribute: () => role };
    },
  };
}

test('legalDrop rejects enemy namesakes and unmatched ally targets', () => {
  const TABLE = loadTable();
  const game = sampleGame();
  const n06 = TABLE.groupPlayEntries(game, game.legal_actions.filter((entry) => entry.action.card_id === 'a-n06'));
  const form = TABLE.groupPlayEntries(game, game.legal_actions.filter((entry) => entry.action.card_id === 'a-n02'));
  const matched = TABLE.legalDropFor('card', n06, zoneFor('a:zero'));
  assert.equal(matched.action.target_id, 'a:zero');
  assert.equal(TABLE.legalDropFor('card', n06, zoneFor('a:xun')), null);
  assert.equal(TABLE.legalDropFor('card', n06, zoneFor('b:zero')), null);
  assert.equal(TABLE.legalDropFor('card', form, zoneFor('a:nanali')).interaction.kind, 'form');
  assert.equal(TABLE.legalDropFor('card', form, zoneFor('b:nanali')), null);
});

test('mulligan swap marks selected cards leaving then incoming cards entering', async () => {
  const TABLE = loadTable();
  const rail = {
    nodes: {},
    querySelector(selector) {
      const match = /data-(?:instance-id|card-id)="([^"]+)"/.exec(selector);
      return match ? this.nodes[match[1]] || null : null;
    },
  };
  function card(id) {
    const node = { id: id, classList: { names: new Set(), add(name) { this.names.add(name); }, contains(name) { return this.names.has(name); } } };
    rail.nodes[id] = node;
    return node;
  }
  const oldA = card('old-a');
  const oldB = card('old-b');
  const keep = card('keep');
  let replaced = false;
  const done = TABLE.animateMulliganSwap({
    rail: rail,
    outgoingIds: ['old-a', 'old-b'],
    incomingIds: ['new-1'],
    reducedMotion: true,
    replace() {
      replaced = true;
      card('new-1');
      card('keep');
    },
  });
  await done;
  assert.equal(replaced, true);
  assert.equal(oldA.classList.contains('selected'), true);
  assert.equal(oldA.classList.contains('is-leaving'), true);
  assert.equal(rail.nodes['new-1'].classList.contains('is-entering'), true);
  assert.equal(keep.classList.contains('is-leaving'), false);
});

test('cards drop on the whole center play area then keep unmatched bench rejects', () => {
  const TABLE = loadTable();
  const game = sampleGame();
  const n06 = TABLE.groupPlayEntries(game, game.legal_actions.filter((entry) => entry.action.card_id === 'a-n06'));
  const form = TABLE.groupPlayEntries(game, game.legal_actions.filter((entry) => entry.action.card_id === 'a-n02'));
  const playArea = zoneWith(null, 'play-area');
  const playerFront = zoneWith(null, 'player-front');
  const opponentFront = zoneWith(null, 'opponent-front');
  const attack = { action: { type: 'attack', character_id: 'nanali' } };
  assert.equal(TABLE.isPlayAreaRole('play-area'), true);
  assert.ok(TABLE.legalDropFor('card', n06, playArea));
  assert.ok(TABLE.legalDropFor('card', n06, playerFront));
  assert.ok(TABLE.legalDropFor('card', n06, opponentFront));
  assert.ok(TABLE.legalDropFor('card', form, playArea));
  assert.ok(TABLE.legalDropFor('card', n06, zoneWith('a:xun', 'player-front')));
  assert.equal(TABLE.legalDropFor('card', n06, zoneFor('a:xun')), null);
  assert.ok(TABLE.legalDropFor('character', null, playArea, attack));
  assert.ok(TABLE.legalDropFor('character', null, playerFront, attack));
  assert.ok(TABLE.legalDropFor('character', null, opponentFront, attack));
  assert.equal(TABLE.legalDropFor('character', null, playerFront, attack, { fromFront: true }), null);
  assert.ok(TABLE.legalDropFor('character', null, opponentFront, attack, { fromFront: true }));
});

test('unknown interaction never infers an enemy card target', () => {
  const TABLE = loadTable();
  const game = sampleGame();
  game.legal_actions = [{ action: { type: 'play_card', card_id: 'mystery' } }];
  const interaction = TABLE.interactionOf(game, game.legal_actions[0]);
  assert.equal(interaction.kind, 'cast');
  assert.equal(interaction.target_ids.length, 0);
  const grouped = TABLE.groupPlayEntries(game, game.legal_actions);
  assert.equal(TABLE.legalDropFor('card', grouped, zoneFor('b:zero')), null);
});

test('play events remove the exact self card and one opponent back', () => {
  const TABLE = loadTable();
  const game = sampleGame();
  const afterSelf = TABLE.applyPatch(game, {}, {
    type: 'play',
    side: 'a',
    card: { instance_id: 'a-n06', name: '共赴死局' },
  });
  assert.equal(afterSelf.sides.a.hand.length, 1);
  assert.equal(afterSelf.sides.a.hand[0].instance_id, 'a-n07');
  const afterOpp = TABLE.applyPatch(game, { sides: { b: { hand_count: 2 } } }, {
    type: 'play',
    side: 'b',
    card: { name: '未知牌' },
  });
  assert.equal(afterOpp.sides.b.hand_count, 2);
  assert.equal(afterOpp.sides.b.hand.filter((card) => card.copy).length, 1);
  assert.equal(afterOpp.sides.b.hand.filter((card) => card.hidden).length, 1);
});

test('hand cards sort by side character order', () => {
  const TABLE = loadTable();
  const side = {
    characters: [{ id: 'nanali' }, { id: 'zero' }, { id: 'jiuyuan' }, { id: 'xun' }],
  };
  const sorted = TABLE.sortHandCards([
    { instance_id: '3', character_id: 'xun' },
    { instance_id: '1', character_id: 'nanali' },
    { instance_id: '2', character_id: 'xun' },
    { instance_id: '4', character_id: 'zero' },
  ], side);
  assert.deepEqual(sorted.map((card) => card.instance_id), ['1', '4', '3', '2']);
});

test('presentation splits mulligan settle events from the first turn banner', () => {
  const TABLE = loadTable();
  const parts = TABLE.splitPresentationAtTurn([
    { seq: 4, type: 'discard' },
    { seq: 5, type: 'draw' },
    { seq: 6, type: 'turn', text: '对手的第 1 个回合开始。' },
    { seq: 7, type: 'play' },
  ]);
  assert.deepEqual(parts.settle.map((event) => event.type), ['discard', 'draw']);
  assert.deepEqual(parts.follow.map((event) => event.type), ['turn', 'play']);
  const none = TABLE.splitPresentationAtTurn([{ seq: 1, type: 'mulligan' }]);
  assert.equal(none.follow.length, 0);
  assert.equal(none.settle.length, 1);
});

test('public play chips label the viewer as 我方 and the other side as 对手', () => {
  const TABLE = loadTable();
  assert.equal(TABLE.publicPlayOwner('a', 'a'), '我方');
  assert.equal(TABLE.publicPlayOwner('b', 'a'), '对手');
  assert.equal(TABLE.publicPlayOwner('a', 'b'), '对手');
  assert.equal(TABLE.publicPlayOwner('b', 'b'), '我方');
  assert.equal(TABLE.publicPlayOwner('', 'a'), '');
  const mine = TABLE.appendPublicPlay([], { type: 'play', seq: 8, side: 'a', card: { name: '家族壮大' } });
  const theirs = TABLE.appendPublicPlay(mine, { type: 'play', seq: 9, side: 'b', card: { name: '要叫大姐头' } });
  assert.equal(TABLE.publicPlayOwner(theirs[0].side, 'a') + ' · ' + theirs[0].card.name, '我方 · 家族壮大');
  assert.equal(TABLE.publicPlayOwner(theirs[1].side, 'a') + ' · ' + theirs[1].card.name, '对手 · 要叫大姐头');
});

test('empty front harmony caption names the source with an attribute icon', () => {
  const TABLE = loadTable();
  const source = TABLE.harmonySourceCharacter({
    characters: [
      { id: 'nanali', name: '娜娜莉', attribute: '灵', harmony_source: false },
      { id: 'zero', name: '零', attribute: '光', harmony_source: true },
    ],
  });
  assert.equal(source.id, 'zero');
  assert.equal(TABLE.harmonySourceCharacter({ characters: [{ id: 'nanali' }] }), null);
  const html = TABLE.harmonySourceCaptionHtml(source);
  assert.equal(html.includes('环合来源方：'), true);
  assert.equal(html.includes('class="v2-harmony-source-icon"'), true);
  assert.equal(html.includes('/static/images/elements/' + encodeURIComponent('光') + '.png'), true);
  assert.equal(html.includes('alt="光"'), true);
  assert.equal(html.endsWith('零'), true);
  assert.equal(html.includes('光零'), false);
  assert.equal(TABLE.harmonySourceCaptionHtml(null), '');
});

test('presented events append to history in play order without duplicating seq', () => {
  const TABLE = loadTable();
  let game = { events: [], logs: [] };
  game = TABLE.appendPresentedEvent(game, { seq: 4, type: 'turn', text: '回合开始' });
  game = TABLE.appendPresentedEvent(game, { seq: 5, type: 'draw', text: '抽到「倾世之雨」。' });
  game = TABLE.appendPresentedEvent(game, { seq: 5, type: 'draw', text: '抽到「倾世之雨」。' });
  assert.deepEqual(game.logs, ['回合开始', '抽到「倾世之雨」。']);
  assert.equal(game.events.length, 2);
});

test('history description prefers public card text then event description', () => {
  const TABLE = loadTable();
  assert.equal(TABLE.historyDescription({
    type: 'play',
    text: '零使用「倾世之雨」。',
    card: { name: '倾世之雨', description: '回合结束时，己方前排攻击 +2。' },
  }), '回合结束时，己方前排攻击 +2。');
  assert.equal(TABLE.historyDescription({
    type: 'ultimate',
    text: '零发动终结。',
    description: '2 个己方回合内，己方武备牌变为瞬发。',
  }), '2 个己方回合内，己方武备牌变为瞬发。');
  assert.equal(TABLE.historyDescription({ type: 'draw', card: { hidden: true, description: '秘密' } }), '');
  assert.equal(TABLE.historyDescription({ type: 'turn', text: '回合开始' }), '');
});

test('public play history ignores private draws and clears across rooms conceptually', () => {
  const TABLE = loadTable();
  let history = [];
  history = TABLE.appendPublicPlay(history, { type: 'draw', card: { name: '秘密抽牌' }, private_side: 'a' });
  assert.equal(history.length, 0);
  history = TABLE.appendPublicPlay(history, { type: 'play', seq: 3, side: 'b', card: { name: '公开牌', instance_id: 'x' }, text: '对手出牌' });
  assert.equal(history.length, 1);
  history = [];
  assert.equal(history.length, 0);
});

test('opening replacements finish each slot before the next leaves', async () => {
  const TABLE = loadTable();
  const timeline = [];
  const nodes = {};
  const make = (id) => nodes[id] = {classList: {
    add(name) { timeline.push(id + ':' + name); },
    remove(name) { timeline.push(id + ':clear-' + name); },
  }};
  ['a', 'b', 'keep'].forEach(make);
  const rail = {querySelector(selector) { return nodes[selector.match(/="([^"]+)"/)[1]]; }};
  await TABLE.animateMulliganSwap({
    rail, outgoingIds: ['a', 'b'], incomingIds: ['x', 'y'],
    wait: async () => { timeline.push('wait'); },
    replaceOne(old, next) { timeline.push(old + '->' + next); make(next); },
  });
  assert.deepEqual(timeline, [
    'a:selected', 'a:is-leaving', 'wait', 'a->x', 'x:is-entering', 'wait', 'x:clear-is-entering',
    'b:selected', 'b:is-leaving', 'wait', 'b->y', 'y:is-entering', 'wait', 'y:clear-is-entering',
  ]);
});

test('cancelled opening replacement never inserts a stale card', async () => {
  const TABLE = loadTable();
  let stopped = false;
  let replacements = 0;
  await TABLE.animateMulliganSwap({
    rail: { querySelector() { return null; } }, outgoingIds: ['a'], incomingIds: ['b'],
    cancelled: () => stopped,
    wait: async () => { stopped = true; },
    replaceOne() { replacements += 1; },
  });
  assert.equal(replacements, 0);
});

test('replay mulligan removes selected identities and restores known retained cards in legacy patches', () => {
  const TABLE = loadTable();
  for (const viewer of ['a', 'b']) {
    const enemy = viewer === 'a' ? 'b' : 'a';
    let game = { phase: 'mulligan', viewer_side: viewer, sides: {
      [viewer]: { hand: ['keep-1', 'swap', 'keep-2'].map(id => ({instance_id:id, name:id})), hand_count:3 },
      [enemy]: { hand: [{hidden:true},{hidden:true}], hand_count:2 },
    }};
    const patch = {sides:{[viewer]:{hand:[{hidden:true},{hidden:true}],hand_count:2}}};
    game = TABLE.applyPatch(game, patch, {type:'mulligan', side:viewer, card_ids:['swap']});
    assert.equal(JSON.stringify(game.sides[viewer].hand.map(c=>c.instance_id)), JSON.stringify(['keep-1','keep-2']));
    assert.ok(patch.sides[viewer].hand.every(c=>c.hidden), 'do not mutate stored replay patches');
    const drawn = {instance_id:'new',name:'new'};
    game = TABLE.applyPatch(game, {sides:{[viewer]:{hand:[{hidden:true},{hidden:true},drawn],hand_count:3}}}, {type:'draw',side:viewer,card:drawn});
    game = TABLE.applyPatch(game, {phase:'playing'}, {type:'turn',side:viewer});
    assert.equal(JSON.stringify(game.sides[viewer].hand.map(c=>c.instance_id)), JSON.stringify(['keep-1','keep-2','new']));
    assert.ok(game.sides[enemy].hand.every(c=>c.hidden));
  }
});

test('replay does not guess unknown opening card identities', () => {
  const TABLE = loadTable();
  const game = {phase:'mulligan', viewer_side:'a', sides:{a:{hand:[{hidden:true},{instance_id:'known',name:'known'}],hand_count:2}}};
  const next = TABLE.applyPatch(game, {sides:{a:{hand:[{hidden:true}],hand_count:1}}}, {type:'mulligan',side:'a'});
  assert.ok(next.sides.a.hand[0].hidden);
});
