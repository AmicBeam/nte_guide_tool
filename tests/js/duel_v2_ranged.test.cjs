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
  assert.equal(text, '海月远程出击的伤害 +2，持续 2 个己方回合。');
  assert.equal(context.window.NTE_V2.cardDescriptionMarkup(text),
    '海月<strong>远程</strong>出击的伤害 +2，持续 2 个己方回合。');
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
  assert.doesNotMatch(markup('A04'), /瞬发/);
  for (const id of ['A02', 'S06']) assert.match(markup(id), /^<strong>瞬发<\/strong>。/);
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

test('Heiyu divination benefits start a separate displayed line while escaping HTML', () => {
  const context = {window: {localStorage: {getItem: () => ''}}};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/v2_api.js', 'utf8'), context);
  const markup = context.window.NTE_V2.cardDescriptionMarkup;
  const raw = JSON.parse(fs.readFileSync('app/modules/card_game/content/duel_v2/catalog.json', 'utf8'));
  for (const card of raw.cards.filter(c => c.character_id === 'heiyu' && !c.derived)) {
    const at = card.description.indexOf('【占卜效果】');
    assert.ok(at >= 0);
    if (at) {
      assert.equal(card.description[at - 1], '\n', card.id);
      assert.match(markup(card.description), /<br>【占卜效果】/);
    } else {
      assert.ok(!markup(card.description).startsWith('<br>'));
    }
  }
  assert.equal(markup('<script>\n【占卜效果】抽牌。'), '&lt;script&gt;<br>【占卜效果】抽牌。');
});
