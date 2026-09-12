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
          { instance_id: 'a-n02', card_id: 'N02', character_id: 'nanali', name: '形态', type: 'form', cost: 1 },
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
  assert.equal(afterSelf.sides.a.hand[0].instance_id, 'a-n02');
  const afterOpp = TABLE.applyPatch(game, { sides: { b: { hand_count: 2 } } }, {
    type: 'play',
    side: 'b',
    card: { name: '未知牌' },
  });
  assert.equal(afterOpp.sides.b.hand_count, 2);
  assert.equal(afterOpp.sides.b.hand.filter((card) => card.copy).length, 1);
  assert.equal(afterOpp.sides.b.hand.filter((card) => card.hidden).length, 1);
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
