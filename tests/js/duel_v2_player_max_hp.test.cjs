const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

test('player HUD exposes a reduced maximum and supports legacy replay frames', () => {
  const context = vm.createContext({window: {NTE_V2: {}}});
  vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/v2_table/render.js', 'utf8'), context);
  const label = {}, hud = {querySelector: () => label};
  const node = {closest: () => hud};
  const render = context.window.NTE_V2_TABLE.fillPlayerLife;
  render(node, {hp: 30, max_hp: 30});
  assert.equal(node.textContent, 30);
  assert.equal(label.hidden, true);
  render(node, JSON.parse(JSON.stringify({hp: 15, max_hp: 27})));
  assert.equal(label.textContent, '上限 27');
  assert.equal(label.hidden, false);
  assert.equal(node.title, '生命 15/27');
  render(node, {hp: 20});
  assert.equal(label.hidden, true);
  assert.equal(node.title, '生命 20/30');
});
