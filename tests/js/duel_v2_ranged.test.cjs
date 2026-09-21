const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

test('ranged ultimate renders the keyword in bold without literal markdown', () => {
  const context = {window: {localStorage: {getItem: () => ''}}};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/v2_api.js', 'utf8'), context);
  const catalog = JSON.parse(fs.readFileSync('app/modules/card_game/content/duel_v2/catalog.json', 'utf8'));
  const text = catalog.characters.find(c => c.id === 'haiyue').awakened_passive;
  assert.equal(text, '本回合，海月可以在备战区远程出击。');
  assert.equal(context.window.NTE_V2.cardDescriptionMarkup(text),
    '本回合，海月可以在备战区<strong>远程</strong>出击。');
  assert.equal(context.window.NTE_V2.cardDescriptionMarkup('<远程>'), '&lt;<strong>远程</strong>&gt;');
});
