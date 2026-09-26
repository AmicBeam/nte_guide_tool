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

test('murk cards show instant, repeatable and zero-action-point keywords in bold', () => {
  const context = {window: {localStorage: {getItem: () => ''}}};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/v2_api.js', 'utf8'), context);
  const catalog = JSON.parse(fs.readFileSync('app/modules/card_game/content/duel_v2/catalog.json', 'utf8'));
  const markup = id => context.window.NTE_V2.cardDescriptionMarkup(catalog.cards.find(c => c.id === id).description);
  assert.match(markup('S04'), /^<strong>瞬发<\/strong>。<strong>可重复使用<\/strong>。/);
  assert.match(markup('A03'), /^<strong>不消耗行动力<\/strong>。/);
  for (const id of ['A02', 'A04', 'S06']) assert.match(markup(id), /^<strong>瞬发<\/strong>。/);
});

test('negative battle attack is visible while zero placeholders remain hidden', () => {
  const context = {window: {localStorage: {getItem: () => ''}}};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/v2_api.js', 'utf8'), context);
  const api = context.window.NTE_V2;
  assert.equal(api.cardFrameStatText({type:'battle',attack:-2,shield:0}), '攻击 -2');
  assert.match(api.cardFrameStat({type:'battle',attack:-2,shield:0}), /-2/);
  assert.equal(api.cardFrameStat({type:'battle',attack:0,shield:0}), '');
  assert.equal(api.cardFrameStatText({type:'battle',attack:1,shield:0}), '攻击 +1');
});
