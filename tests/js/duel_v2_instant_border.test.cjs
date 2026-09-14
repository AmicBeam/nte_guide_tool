const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

test('hand instant borders follow the viewer opportunity and reset next turn', () => {
  const context = { window: {} };
  vm.createContext(context);
  for (const file of ['v2_api.js', 'v2_table/helpers.js', 'v2_table/render.js']) {
    vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/' + file, 'utf8'), context);
  }
  const card = { instance_id: 'instant-card', character_id: 'zero', name: '瞬发测试', type: 'tactic', instant: true };
  const mine = { hand: [card], used: { instant: false } };
  const ctx = {
    viewer: 'b', opponent: 'a', game: { phase: 'action', sides: { a: { hand: [], used: { instant: true } }, b: mine } },
    playEntries: () => [{}], nodes: { playerHand: {}, opponentHand: {}, handCount: {} },
  };
  const table = context.window.NTE_V2_TABLE;
  const classes = () => { table.renderHands(ctx); return ctx.nodes.playerHand.innerHTML.match(/class="([^"]+)"/)[1].split(' '); };
  assert.ok(classes().includes('is-instant'));
  mine.used.instant = true;
  assert.ok(!classes().includes('is-instant'));
  assert.ok(classes().includes('legal'), 'payable instant card remains usable');
  card.response = true;
  assert.ok(classes().includes('is-response'));
  assert.ok(!classes().includes('is-instant-response'));
  mine.used.instant = false;
  assert.ok(classes().includes('is-instant-response'));
  assert.equal(card.instant, true, 'presentation must not mutate the card rules');
});
