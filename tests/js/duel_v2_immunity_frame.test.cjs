const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

test('immunity frame follows public state through expiry, death and replay', () => {
  const context = {window: {matchMedia: () => ({matches: false})}, document: {getElementById: () => null}, requestAnimationFrame() {}};
  vm.createContext(context);
  for (const file of ['v2_api.js', 'v2_table/helpers.js', 'v2_table/render.js']) {
    vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/' + file, 'utf8'), context);
  }
  const classes = new Set(), attrs = {};
  const node = {style: {}, classList: {toggle(k, v) {v ? classes.add(k) : classes.delete(k)}, remove() {}},
    setAttribute(k, v) {attrs[k] = v}, getAttribute(k) {return attrs[k]}, removeAttribute(k) {delete attrs[k]},
    querySelector() {return null}, querySelectorAll() {return []}};
  const hero = {id: 'zhenhong', name: '真红', hp: 5, max_hp: 5, damage_immune: true};
  for (const [patch, options, expected] of [
    [{}, {}, true], [{damage_immune: false}, {}, false], [{}, {}, true],
    [{down_turns: 3, hp: 0}, {}, false], [{}, {placeholder: true}, false],
    [JSON.parse(JSON.stringify(hero)), {side: 'b'}, true],
  ]) {
    context.window.NTE_V2_TABLE.fillCharacterCard(node, {...hero, ...patch}, {side: 'a', usePortrait: true, ...options});
    assert.equal(classes.has('is-damage-immune'), expected);
    assert.equal(node.innerHTML.includes('v2-immunity-frame'), expected);
    assert.ok(!node.innerHTML.includes('直到下个己方回合开始'));
  }
});

test('blue damage limit frame is distinct from immunity and follows public patches', () => {
  const context = {window: {matchMedia: () => ({matches: false})}, document: {getElementById: () => null}, requestAnimationFrame() {}};
  vm.createContext(context);
  for (const file of ['v2_api.js', 'v2_table/helpers.js', 'v2_table/render.js']) {
    vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/' + file, 'utf8'), context);
  }
  const classes = new Set(), attrs = {};
  const node = {style: {}, classList: {toggle(k, v) {v ? classes.add(k) : classes.delete(k)}, remove() {}},
    setAttribute(k, v) {attrs[k] = v}, getAttribute(k) {return attrs[k]}, removeAttribute(k) {delete attrs[k]},
    querySelector() {return null}, querySelectorAll() {return []}};
  const hero = {id: 'zhenhong', name: '真红', hp: 5, max_hp: 5, damage_immune: false, damage_limit: 1};
  for (const [patch, options, expected] of [
    [{}, {}, true], [{damage_limit: null}, {}, false], [{}, {}, true],
    [{down_turns: 3, hp: 0}, {}, false], [{}, {placeholder: true}, false],
    [JSON.parse(JSON.stringify(hero)), {side: 'b'}, true],
  ]) {
    context.window.NTE_V2_TABLE.fillCharacterCard(node, {...hero, ...patch}, {side: 'a', usePortrait: true, ...options});
    assert.equal(classes.has('is-damage-limited'), expected);
    assert.equal(classes.has('is-damage-immune'), false);
    assert.equal(node.innerHTML.includes('v2-damage-limit-frame'), expected);
    assert.equal(node.innerHTML.includes('单次至多受到 1 点伤害'), expected);
    assert.ok(!node.innerHTML.includes('免疫伤害'));
  }
});
