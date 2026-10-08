const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

test('player resistance icons render for either side, replay and expiry', () => {
  const effects = {innerHTML: ''};
  const line = {querySelector: () => effects};
  const hud = {querySelector: () => line};
  const context = {window: {}, document: {getElementById: () => null}, requestAnimationFrame() {}};
  vm.createContext(context);
  for (const file of ['v2_api.js', 'v2_table/render.js']) {
    vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/' + file, 'utf8'), context);
  }
  const marker = {id: 'effect:1', name: '光抗性 -1', kind: 'debuff',
    timing: '至灵可下个己方回合开始', clears_on_down: false, description: '光抗性 -1'};
  for (const markers of [[marker], JSON.parse(JSON.stringify([marker]))]) {
    context.window.NTE_V2_TABLE.fillPlayerEffects(hud, markers);
    assert.match(effects.innerHTML, /is-debuff/);
    assert.match(effects.innerHTML, /减益：光抗性 -1/);
    assert.match(effects.innerHTML, /至灵可下个己方回合开始/);
  }
  context.window.NTE_V2_TABLE.fillPlayerEffects(hud, []);
  assert.equal(effects.innerHTML, '');
});
