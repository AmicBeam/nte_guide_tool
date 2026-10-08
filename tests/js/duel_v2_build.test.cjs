const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function makeEl(id) {
  const el = {
    id: id || '',
    innerHTML: '',
    textContent: '',
    value: '',
    disabled: false,
    hidden: false,
    className: '',
    attrs: {},
    listeners: [],
    classList: {
      add(name) { el.className += ' ' + name; },
      remove(name) { el.className = el.className.split(/\s+/).filter((item) => item && item !== name).join(' '); },
      toggle(name, on) { if (on) this.add(name); else this.remove(name); },
    },
    setAttribute(key, value) { el.attrs[key] = String(value); },
    getAttribute(key) { return Object.prototype.hasOwnProperty.call(el.attrs, key) ? el.attrs[key] : null; },
    addEventListener(type, fn) { el.listeners.push([type, fn]); },
    querySelectorAll() { return []; },
    click() {
      el.listeners.filter((item) => item[0] === 'click').forEach((item) => {
        item[1]({ target: el, preventDefault() {}, stopPropagation() {} });
      });
    },
  };
  return el;
}

function waitUntil(predicate) {
  return new Promise((resolve, reject) => {
    let tries = 30;
    const tick = () => {
      if (predicate()) {
        resolve();
        return;
      }
      tries -= 1;
      if (tries <= 0) {
        reject(new Error('timed out'));
        return;
      }
      setImmediate(tick);
    };
    tick();
  });
}

function loadBuildPage() {
  const nodes = {};
  [
    'v2-build-banner', 'v2-build-groups', 'v2-build-team', 'v2-build-library',
    'v2-build-roster', 'v2-build-total', 'v2-build-source',
    'v2-build-name', 'v2-save-build-btn', 'v2-build-new-btn',
    'v2-build-delete-btn', 'v2-build-detail',
    'v2-build-transfer', 'v2-build-text', 'v2-build-transfer-error',
    'v2-build-import-apply', 'v2-build-import-btn', 'v2-build-export-btn',
    'v2-build-transfer-close', 'v2-build-copy', 'v2-build-download',
    'v2-build-transfer-title',
  ].forEach((id) => { nodes[id] = makeEl(id); });
  const context = {
    window: {
      localStorage: { getItem() { return 'token'; }, setItem() {}, removeItem() {} },
    },
    console,
    document: {
      getElementById(id) { return nodes[id] || null; },
      querySelector() { return null; },
      querySelectorAll() { return []; },
    },
  };
  context.window.document = context.document;
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.resolve('app/modules/card_game/static/js/v2_api.js'), 'utf8'), context);
  context.window.NTE_V2.getCatalog = async function () {
    return {
      characters: [
        { id: 'nanali', name: '娜娜莉', attribute: '灵', attack: 3, max_hp: 7, energy_max: 5, awakened_passive: '追击' },
        { id: 'zero', name: '零', attribute: '光', attack: 2, max_hp: 9, energy_max: 5 },
        { id: 'jiuyuan', name: '九原', attribute: '灵', attack: 2, max_hp: 8, energy_max: 5 },
        { id: 'xun', name: '浔', attribute: '光', attack: 2, max_hp: 8, energy_max: 5 },
      ],
      cards: [],
      starter_deck: {
        id: 'starter',
        name: '基础预组',
        character_ids: ['nanali', 'zero', 'jiuyuan', 'xun'],
        card_ids: [],
      },
      starter_decks: [],
      saved_builds: [{
        id: 'starter-saved',
        name: '基础预组',
        character_ids: ['nanali', 'zero', 'jiuyuan', 'xun'],
        card_ids: ['N01'],
      }],
      saved_build: {
        id: 'starter-saved',
        name: '基础预组',
        character_ids: ['nanali', 'zero', 'jiuyuan', 'xun'],
        card_ids: ['N01'],
      },
      active_build_id: 'starter-saved',
    };
  };
  vm.runInContext(fs.readFileSync(path.resolve('app/modules/card_game/static/js/v2_build.js'), 'utf8'), context);
  return nodes;
}

test('new build starts with no characters selected', async () => {
  const nodes = loadBuildPage();
  await waitUntil(() => nodes['v2-build-team'].innerHTML.indexOf('娜娜莉') >= 0);
  await waitUntil(() => nodes['v2-build-groups'].innerHTML.indexOf('v2-character-stats') >= 0);
  assert.match(nodes['v2-build-groups'].innerHTML, /v2-stat-badge is-atk/);
  assert.match(nodes['v2-build-groups'].innerHTML, /v2-stat-badge is-hp/);
  assert.match(nodes['v2-build-groups'].innerHTML, /<em>3<\/em>/);
  assert.match(nodes['v2-build-groups'].innerHTML, /<em>7<\/em>/);
  assert.match(nodes['v2-build-groups'].innerHTML, /终结（5）：/);
  assert.equal(nodes['v2-build-groups'].innerHTML.includes('subtle">攻击'), false);
  assert.match(nodes['v2-build-library'].innerHTML, /基础预组/);
  assert.equal(nodes['v2-build-library'].innerHTML.includes('可拖动排序'), false);
  assert.match(nodes['v2-build-team'].innerHTML, /v2-avatar/);
  assert.match(nodes['v2-build-team'].innerHTML, /characters\/avatar/);
  assert.equal(nodes['v2-build-team'].innerHTML.includes('v2-portrait'), false);
  nodes['v2-build-new-btn'].click();
  assert.equal(nodes['v2-build-name'].value, '新的构筑');
  assert.equal((nodes['v2-build-team'].innerHTML.match(/选择角色/g) || []).length, 4);
  assert.equal(nodes['v2-build-team'].innerHTML.indexOf('娜娜莉'), -1);
  assert.match(nodes['v2-build-groups'].innerHTML, /选择出战角色/);
  assert.match(nodes['v2-build-source'].textContent, /新的构筑/);
});

test('battle cards can show attack and shield icons together', () => {
  const context = {
    window: { localStorage: { getItem() { return ''; }, setItem() {}, removeItem() {} } },
    document: { getElementById() { return null; } },
    console,
  };
  context.window.document = context.document;
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.resolve('app/modules/card_game/static/js/v2_api.js'), 'utf8'), context);
  const V2 = context.window.NTE_V2;
  const battle = V2.cardFrameStat({ type: 'battle', attack: 2, shield: 1 });
  assert.match(battle, /v2-frame-stats/);
  assert.match(battle, /is-atk/);
  assert.match(battle, /\+2/);
  assert.match(battle, /is-shield/);
  assert.match(battle, /<em>1<\/em>/);
  assert.equal(V2.cardFrameStat({ type: 'battle', attack: 2, shield: 0 }), V2.cardFrameStat({ type: 'battle', attack: 2 }));
  assert.equal(/is-atk/.test(V2.cardFrameStat({ type: 'battle', attack: 0, shield: 2 })), false);
  assert.equal(/is-shield/.test(V2.cardFrameStat({ type: 'battle', attack: 2, shield: 0 })), false);
  const form = V2.cardFrameStat({ type: 'form', attack: 2, hp: 5 });
  assert.match(form, /is-hp/);
  assert.match(form, /is-atk/);
  assert.match(form, />2</);
  assert.match(form, />5</);
  assert.equal(V2.cardFrameStat({ type: 'tactic', cost: 1 }), '');
});

test('character info uses attack/hp icons and energy-cap ultimate label', () => {
  const context = {
    window: { localStorage: { getItem() { return ''; }, setItem() {}, removeItem() {} } },
    document: { getElementById() { return null; } },
    console,
  };
  context.window.document = context.document;
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.resolve('app/modules/card_game/static/js/v2_api.js'), 'utf8'), context);
  const V2 = context.window.NTE_V2;
  const stats = V2.characterStatMarkup({ attack: 2, max_hp: 5 });
  assert.match(stats, /v2-character-stats/);
  assert.match(stats, /is-atk/);
  assert.match(stats, /is-hp/);
  assert.match(stats, /<em>2<\/em>/);
  assert.match(stats, /<em>5<\/em>/);
  assert.equal(V2.characterUltimateLabel({ attribute: '灵' }), '终结（5）');
  assert.equal(V2.characterUltimateLabel({ attribute: '暗' }), '终结（6）');
  assert.equal(V2.characterUltimateLabel({ attribute: '咒', energy_max: 5 }), '终结（5）');
  assert.equal(V2.characterUltimateLabel({ attribute: '暗', energy_max: 4 }), '终结（4）');
});

function loadCodexPage() {
  const nodes = {};
  ['v2-codex-banner', 'v2-codex-nav', 'v2-codex-body', 'v2-codex-summary',
    'v2-codex-glossary', 'v2-codex-help', 'v2-glossary-search', 'v2-glossary-close', 'v2-glossary-empty'].forEach((id) => {
    nodes[id] = makeEl(id);
  });
  const context = {
    window: {
      localStorage: { getItem() { return 'token'; }, setItem() {}, removeItem() {} },
    },
    console,
    document: {
      getElementById(id) { return nodes[id] || null; },
      querySelector() { return null; },
      querySelectorAll() { return []; },
    },
  };
  context.window.document = context.document;
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.resolve('app/modules/card_game/static/js/v2_api.js'), 'utf8'), context);
  context.window.NTE_V2.getCatalog = async function () {
    return {
      characters: [
        { id: 'nanali', name: '娜娜莉', attribute: '灵', attack: 2, max_hp: 5, energy_max: 5, passive: '家族壮大', awakened_passive: '追击' },
        { id: 'anhunqu', name: '安魂曲', attribute: '暗', attack: 2, max_hp: 5, energy_max: 6, passive: '噩梦', awakened_passive: '噩梦强化' },
      ],
      cards: [],
    };
  };
  vm.runInContext(fs.readFileSync(path.resolve('app/modules/card_game/static/js/v2_codex.js'), 'utf8'), context);
  return { nodes, context };
}

test('codex character info uses attack/hp icons and energy-cap ultimate label', async () => {
  const { nodes } = loadCodexPage();
  await waitUntil(() => nodes['v2-codex-body'].innerHTML.indexOf('v2-character-stats') >= 0);
  assert.match(nodes['v2-codex-body'].innerHTML, /v2-stat-badge is-atk/);
  assert.match(nodes['v2-codex-body'].innerHTML, /v2-stat-badge is-hp/);
  assert.match(nodes['v2-codex-body'].innerHTML, /终结（5）：/);
  assert.equal(nodes['v2-codex-body'].innerHTML.includes('subtle">攻击'), false);
  const anhunqu = nodes['v2-codex-nav'].listeners.find((item) => item[0] === 'click');
  assert.ok(anhunqu);
  anhunqu[1]({
    target: {
      closest() {
        return { getAttribute() { return 'anhunqu'; } };
      },
    },
  });
  assert.match(nodes['v2-codex-body'].innerHTML, /终结（6）：/);
  assert.match(nodes['v2-codex-body'].innerHTML, /安魂曲/);
});
