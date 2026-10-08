const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');

function element() {
  const classes = new Set();
  return {dataset: {}, classList: {add: x => classes.add(x), remove: x => classes.delete(x),
    contains: x => classes.has(x), toggle: (x, yes) => yes ? classes.add(x) : classes.delete(x)},
    listeners: {}, children: [], setAttribute() {}, querySelectorAll: () => [],
    appendChild(x) { this.children.push(x); }, addEventListener(k, fn) {this.listeners[k] = fn;}};
}
const clone = x => JSON.parse(JSON.stringify(x));
async function spin(check) {
  for (let i = 0; i < 200; i++) {if (check()) return; await new Promise(r => setTimeout(r, 1));}
  throw Error('Expected async state was not reached');
}
function harness(advance) {
  const window = {setTimeout, clearTimeout, NTE_V2: {makeRequestId: () => 'one-request', advanceAi: advance,
    findLegalEntry: () => null, findLegalEntries: () => [], setHidden() {},
    viewerSide: game => game && game.viewer_side || 'a', opponentSide: game => game.viewer_side === 'b' ? 'a' : 'b',
    showBanner(el, text, kind) {el.textContent = text; el.dataset.kind = kind;}, roomStatusLabel: () => '对局进行中'}};
  const document = {hidden: false, createElement: element, getElementById: () => null};
  const context = vm.createContext({window, document, console});
  vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/v2_table/helpers.js', 'utf8'), context);
  const table = window.NTE_V2_TABLE;
  table.renderGame = () => {};
  table.renderHands = () => {};
  const ctx = {page: element(), nodes: new Proxy({}, {get(t,k) {return t[k] || (t[k] = element());}}),
    roomPanel: element(), emptyState: element(), tableLayout: element(), handDock: element(),
    choiceOverlay: element(), banner: element(), options: {}, mulliganIds: [], initiativePlayed: true, closeOverlay() {}};
  let play;
  table.createPlayer = options => ({cancel() {}, setOptions() {}, play(events) {return play(events, options);}});
  vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/v2_table/sync.js', 'utf8'), context);
  table.bindSync(ctx);
  return {ctx, setPlay(fn) {play = fn;}};
}
function payload(version, pending, sequence) {
  return {room: {room_code:'STEP', mode:'advanced', status:'playing', ai_pending:pending},
    presentation:{cursor:sequence,events:sequence ? [{seq:sequence,type:'effect',patch:{}}] : []},
    game: {version, phase:'playing', turn:1, viewer_side:'a', legal_actions:[],
      sides:{a:{hand:[],characters:[]},b:{hand:[],characters:[]}},
      presentation:{cursor:sequence, events:sequence ? [{seq:sequence,type:'effect',patch:{}}] : []}}};
}

test('next AI request waits for the current operation presentation', async () => {
  let calls = 0, release;
  const h = harness(async () => {calls++; return payload(calls + 1, calls === 1, calls);});
  h.setPlay(() => calls === 1 ? new Promise(resolve => {release = resolve;}) : Promise.resolve());
  await h.ctx.applyEnvelope(payload(1, true, 0));
  await spin(() => calls === 1 && release);
  await new Promise(r => setTimeout(r, 15));
  assert.equal(calls, 1, 'No next search while the first action is still presenting');
  release();
  await spin(() => calls === 2 && !h.ctx.busy);
  assert.equal(h.ctx.game.version, 3);
  assert.equal(h.ctx.room.ai_pending, false);
});

test('network retry preserves its request identity and does not loop automatically', async () => {
  const requests = [];
  const h = harness(async request => {
    requests.push(clone(request));
    if (requests.length === 1) throw {status:0,message:'offline'};
    return payload(2, false, 1);
  });
  h.setPlay(() => Promise.resolve());
  await h.ctx.applyEnvelope(payload(1, true, 0));
  await spin(() => h.ctx.aiStepFailed);
  await new Promise(r => setTimeout(r, 15));
  assert.equal(requests.length, 1);
  const retry = h.ctx.banner.children[0];
  assert.equal(retry.textContent, '重试这一步');
  retry.listeners.click();
  await spin(() => requests.length === 2 && !h.ctx.busy);
  assert.deepEqual(requests[0], requests[1]);
});
