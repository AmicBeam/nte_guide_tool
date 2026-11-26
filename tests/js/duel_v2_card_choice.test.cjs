const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function button(id) {
  return {listeners: {}, getAttribute: () => id, addEventListener(type, handler) { this.listeners[type] = handler; }};
}

function setup(cards) {
  const pickButtons = cards.map(c => button(c.id)), details = cards.map(c => button(c.id));
  const overlay = {innerHTML: '', classList: {add() {}, remove() {}}, setAttribute() {},
    querySelector: () => null,
    querySelectorAll: selector => selector === '[data-choice-id]' ? pickButtons : details};
  const context = vm.createContext({window: {localStorage: {getItem: () => ''}},
    document: {getElementById: () => button('close')}, choiceOverlay: overlay,
    ctx: {setSelectionView(config) { this.selection = config; }},
    closeOverlay() {}, markRaisedCard() {}, showHoverPreview(...args) {context.preview = args;},
    showCardDetail(...args) {context.detail = args;}, hideHoverPreview() {context.preview = null;}});
  for (const file of ['v2_api.js', 'v2_table/helpers.js', 'v2_table/render.js'])
    vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/' + file, 'utf8'), context);
  context.V2 = context.window.NTE_V2;
  context.TABLE = context.window.NTE_V2_TABLE;
  const source = fs.readFileSync('app/modules/card_game/static/js/v2_table/interaction.js', 'utf8');
  vm.runInContext(source.slice(source.indexOf('    function openChoice('), source.indexOf('    function openPile(')), context);
  return {context, overlay, pickButtons, details};
}

for (const sourceId of ['K01', 'J01']) test(sourceId + ' uses full shared card faces with separate details and one pick', () => {
  const catalog = JSON.parse(fs.readFileSync('app/modules/card_game/content/duel_v2/catalog.json'));
  const cards = ['Z03', 'Z04', 'Z08'].map((id, index) => ({id: 'instance-' + index,
    card: {...catalog.cards.find(c => c.id === id), card_id: id, instance_id: 'instance-' + index}}));
  const {context, overlay, pickButtons, details} = setup(cards);
  let picked;
  context.openChoice({title: catalog.cards.find(c => c.id === sourceId).name, choices: cards, onPick: c => { picked = c; }});
  assert.equal((overlay.innerHTML.match(/v2-selection-card/g) || []).length, 3);
  assert.equal((overlay.innerHTML.match(/data-choice-detail=/g) || []).length, 3);
  assert.match(overlay.innerHTML, /学园随想/);
  assert.match(overlay.innerHTML, /回合结束时/);
  assert.doesNotMatch(overlay.innerHTML, /v2-hover-card|v2-choice-confirm/);
  assert.equal(context.ctx.selection.label, '选择');
  details[0].listeners.pointerenter({type: 'pointerenter', pointerType: 'mouse'});
  assert.equal(context.preview[0].card, cards[0].card);
  details[0].listeners.click();
  assert.equal(context.detail[0], cards[0].card);
  assert.equal(picked, undefined, 'viewing details must not choose a card');
  pickButtons[1].listeners.click();
  assert.equal(picked.id, cards[1].id);
});
