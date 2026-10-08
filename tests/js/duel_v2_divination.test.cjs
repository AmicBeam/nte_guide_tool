const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function load() {
  const context = {window: {NTE_V2: {
    findLegalEntry: (game, predicate) => (game.legal_actions || []).find(entry => predicate(entry.action, entry)) || null,
  }}};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/v2_table/divination.js', 'utf8'), context);
  return context.window.NTE_V2_TABLE;
}
const plain = value => JSON.parse(JSON.stringify(value));

test('dragging another card to the bottom swaps the old bottom into its original top slot', () => {
  const table = load();
  const before = {topIds: ['a', 'b', 'c'], bottomId: 'd'};
  assert.deepEqual(plain(table.moveDivinationCard(before, 'b', 'bottom')), {topIds: ['a', 'd', 'c'], bottomId: 'b'});
  assert.deepEqual(before, {topIds: ['a', 'b', 'c'], bottomId: 'd'});
});

test('top cards can be reordered and a bottom card can be returned at any top position', () => {
  const table = load();
  const before = {topIds: ['a', 'b', 'c'], bottomId: 'd'};
  const reordered = table.moveDivinationCard(before, 'c', 'top', 0);
  assert.deepEqual(plain(reordered), {topIds: ['c', 'a', 'b'], bottomId: 'd'});
  assert.deepEqual(plain(table.moveDivinationCard(reordered, 'd', 'top', 1)), {topIds: ['c', 'd', 'a', 'b'], bottomId: null});
});

test('a single inspected card can be bottomed or kept, without adding another card', () => {
  const table = load();
  const bottomed = table.moveDivinationCard({topIds: ['a'], bottomId: null}, 'a', 'bottom');
  assert.deepEqual(plain(bottomed), {topIds: [], bottomId: 'a'});
  assert.deepEqual(plain(table.moveDivinationCard(bottomed, 'a', 'top', 0)), {topIds: ['a'], bottomId: null});
  assert.deepEqual(plain(table.moveDivinationCard(bottomed, 'foreign', 'bottom')), {topIds: [], bottomId: 'a'});
});

test('confirmation matches server-authorized order, rather than an unordered card set', () => {
  const table = load();
  const reversed = {action: {type: 'choose_deck_order', card_ids: ['b', 'a'], bottom_card_id: 'c'}};
  const wanted = {action: {type: 'choose_deck_order', card_ids: ['a', 'b'], bottom_card_id: 'c'}};
  const game = {legal_actions: [reversed, wanted]};
  assert.equal(table.findDivinationEntry(game, {topIds: ['a', 'b'], bottomId: 'c'}), wanted);
  assert.equal(table.findDivinationEntry(game, {topIds: ['a', 'b'], bottomId: null}), null);
  assert.equal(table.findDivinationEntry({legal_actions: [reversed]}, {topIds: ['a', 'b'], bottomId: 'c'}), null);
});

test('same-named physical cards keep separate instance identities during swaps', () => {
  const table = load();
  const before = {topIds: ['P01-instance-1', 'P01-instance-2'], bottomId: 'P01-instance-3'};
  assert.deepEqual(plain(table.moveDivinationCard(before, 'P01-instance-2', 'bottom')),
    {topIds: ['P01-instance-1', 'P01-instance-3'], bottomId: 'P01-instance-2'});
});
