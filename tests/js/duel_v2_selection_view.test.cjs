const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function element(id) {
  const classes = new Set(), attrs = {};
  return {id, hidden: false, inert: false, listeners: {},
    classList: {add: (...items) => items.forEach(x => classes.add(x)),
      remove: (...items) => items.forEach(x => classes.delete(x)), contains: x => classes.has(x),
      toggle(x, yes) { yes ? classes.add(x) : classes.delete(x); }},
    setAttribute(k, v) {attrs[k] = v;}, getAttribute: k => attrs[k],
    addEventListener(k, fn) {this.listeners[k] = fn;}, focus() {this.focused = true;},
    querySelector: () => ({disabled: false}), querySelectorAll: () => []};
}

function setup() {
  const window = {NTE_V2: {escapeAttr: String, escapeHtml: String,
    findLegalEntry: (game, predicate) => (game.legal_actions || []).find(e => predicate(e.action))}};
  const context = vm.createContext({window});
  for (const file of ['selection_view.js', 'divination.js'])
    vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/v2_table/' + file, 'utf8'), context);
  const ctx = {page: element('page'), nodes: {selectionToggle: element('toggle')},
    choiceOverlay: element('choice'), hideHoverPreview() {this.hiddenPreviews = (this.hiddenPreviews || 0) + 1;}};
  window.NTE_V2_TABLE.bindSelectionView(ctx);
  return {table: window.NTE_V2_TABLE, ctx};
}

test('collapsing and restoring a choice preserves drafts and survives repaint', () => {
  const {ctx} = setup(), region = ctx.choiceOverlay;
  const draft = {topIds: ['b'], bottomIds: ['a']};
  ctx.divinationDraft = draft;
  region.classList.add('open');
  const config = {key: 'divination:one', label: '占卜', region};
  ctx.setSelectionView(config);
  assert.equal(ctx.nodes.selectionToggle.textContent, '收起占卜 · 查看战场');
  ctx.nodes.selectionToggle.listeners.click();
  assert.equal(region.inert, true);
  assert.equal(region.getAttribute('aria-hidden'), 'true');
  assert.equal(ctx.nodes.selectionToggle.textContent, '返回占卜');
  ctx.setSelectionView(config);
  assert.equal(ctx.selectionView.collapsed, true);
  assert.equal(ctx.divinationDraft, draft);
  ctx.nodes.selectionToggle.listeners.click();
  assert.equal(region.inert, false);
  assert.equal(region.getAttribute('aria-hidden'), 'false');
  assert.equal(ctx.divinationDraft, draft);
  ctx.clearSelectionView(region);
  assert.equal(ctx.nodes.selectionToggle.hidden, true);
  assert.equal(ctx.page.classList.contains('selection-collapsed'), false);
});

test('mulligan uses the same control and the next choice starts expanded', () => {
  const {ctx} = setup(), region = element('mulligan');
  const selected = ['first', 'third'];
  ctx.mulliganIds = selected;
  ctx.setSelectionView({key: 'mulligan:room', label: '换牌', region});
  ctx.nodes.selectionToggle.listeners.click();
  assert.equal(region.inert, true);
  assert.equal(ctx.mulliganIds, selected);
  ctx.clearSelectionView(ctx.choiceOverlay);
  assert.equal(ctx.nodes.selectionToggle.hidden, false);
  ctx.setSelectionView({key: 'divination:new', label: '占卜', region: ctx.choiceOverlay});
  assert.equal(region.inert, false);
  assert.equal(ctx.selectionView.collapsed, false);
  assert.equal(ctx.nodes.selectionToggle.getAttribute('aria-controls'), 'choice');
});

test('divination detail shows a hover preview without selecting or submitting', () => {
  const {ctx, table} = setup();
  const card = {name: '失明症', description: '然后占卜。'};
  ctx.game = {sides: {a: {characters: []}}, legal_actions: []};
  ctx.showHoverPreview = (payload, source, options) => {ctx.preview = {payload, source, options};};
  table.bindDivination(ctx, () => '<article></article>');
  ctx.openDivinationChoice({side: 'a', kind: 'heiyu_divination', choices: [{id: 'one', card}]});
  const detail = {getAttribute: () => 'one', contains: () => false};
  const target = {closest: selector => selector === '[data-divination-detail]' ? detail : null};
  ctx.choiceOverlay.listeners.pointerover({target, pointerType: 'mouse'});
  assert.equal(ctx.preview.payload.card, card);
  assert.equal(ctx.preview.options.pointerType, 'mouse');
  assert.equal(ctx.localChoice.kind, 'server-divination');
  assert.deepEqual(Array.from(ctx.divinationDraft.topIds), ['one']);
  const before = ctx.hiddenPreviews;
  ctx.choiceOverlay.listeners.pointerout({target, relatedTarget: null});
  assert.equal(ctx.hiddenPreviews, before + 1);
  ctx.preview = null;
  ctx.choiceOverlay.listeners.pointerover({target, pointerType: 'touch'});
  assert.equal(ctx.preview, null);
});

test('only divination candidates with a benefit receive the extra glowing edge', () => {
  const {ctx, table} = setup();
  ctx.game = {sides: {}, legal_actions: []};
  table.bindDivination(ctx, card => '<article>' + card.description + '</article>');
  const choices = [{id: 'benefit', card: {name: '失明症', description: '抽牌。\n【占卜效果】抽 1 张牌。'}},
    {id: 'ordinary', card: {name: '其他牌', description: '普通效果。'}}];
  ctx.openDivinationChoice({side: 'a', kind: 'heiyu_divination', choices});
  assert.equal((ctx.choiceOverlay.innerHTML.match(/has-divination-benefit/g) || []).length, 1);
  assert.match(ctx.choiceOverlay.innerHTML, /class="v2-divination-card has-divination-benefit"[^>]+data-divination-id="benefit"/);
  ctx.openDivinationChoice({side: 'a', kind: 'hand_deck_placement', hand_placement: true, choices});
  assert.doesNotMatch(ctx.choiceOverlay.innerHTML, /has-divination-benefit/);
});
