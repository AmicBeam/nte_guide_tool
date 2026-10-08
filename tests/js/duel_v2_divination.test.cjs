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
test('multiple bottom cards remain together rather than displacing an earlier card', () => {
  const table = load();
  const before = {topIds: ['a', 'b', 'c'], bottomIds: ['d']};
  assert.deepEqual(plain(table.moveDivinationCard(before, 'b', 'bottom')),
    {topIds: ['a', 'c'], bottomIds: ['d', 'b']});
  assert.deepEqual(before, {topIds: ['a', 'b', 'c'], bottomIds: ['d']});
});
test('both rows can be reordered and bottom cards can return to any top slot', () => {
  const table = load();
  const before = {topIds: ['a', 'b'], bottomIds: ['c', 'd']};
  const reordered = table.moveDivinationCard(before, 'd', 'bottom', 0);
  assert.deepEqual(plain(reordered), {topIds: ['a', 'b'], bottomIds: ['d', 'c']});
  assert.deepEqual(plain(table.moveDivinationCard(reordered, 'd', 'top', 1)),
    {topIds: ['a', 'd', 'b'], bottomIds: ['c']});
});
test('all cards may be bottomed without adding or losing an instance', () => {
  const table = load();
  let draft = {topIds: ['a', 'b'], bottomIds: []};
  draft = table.moveDivinationCard(draft, 'a', 'bottom');
  draft = table.moveDivinationCard(draft, 'b', 'bottom');
  assert.deepEqual(plain(draft), {topIds: [], bottomIds: ['a', 'b']});
  assert.deepEqual(plain(table.moveDivinationCard(draft, 'foreign', 'bottom')), plain(draft));
});
test('confirmation matches ordered top and ordered bottom arrays', () => {
  const table = load();
  const reversed = {action: {type: 'choose_deck_order', card_ids: ['a'], bottom_card_ids: ['c', 'b']}};
  const wanted = {action: {type: 'choose_deck_order', card_ids: ['a'], bottom_card_ids: ['b', 'c']}};
  const game = {legal_actions: [reversed, wanted]};
  assert.equal(table.findDivinationEntry(game, {topIds: ['a'], bottomIds: ['b', 'c']}), wanted);
  assert.equal(table.findDivinationEntry(game, {topIds: ['a', 'b', 'c'], bottomIds: []}), null);
});
test('same-named cards retain distinct identities in the bottom row', () => {
  const table = load();
  const draft = {topIds: ['P05-1'], bottomIds: ['P05-2']};
  assert.deepEqual(plain(table.moveDivinationCard(draft, 'P05-1', 'bottom')),
    {topIds: [], bottomIds: ['P05-2', 'P05-1']});
});
