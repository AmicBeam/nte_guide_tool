"""Exercise replay phase changes through the browser's real render/sync paths."""
import subprocess
import unittest
from pathlib import Path


class ReplayPhaseTest(unittest.TestCase):
    def test_autoplay_and_seek_follow_presented_phase(self):
        script = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
function node() {
  const classes = new Set();
  return {
    hidden: false, dataset: {}, textContent: '', innerHTML: '', listeners: {},
    classList: {add: x => classes.add(x), remove: x => classes.delete(x),
      contains: x => classes.has(x), toggle: (x, yes) => yes ? classes.add(x) : classes.delete(x)},
    addEventListener(type, fn) { this.listeners[type] = fn; },
    setAttribute() {}, replaceChildren() {}, appendChild() {}, querySelectorAll: () => [],
  };
}
async function scenario(animated) {
  const elements = new Map();
  const document = {getElementById(id) {
    if (!elements.has(id)) elements.set(id, node());
    return elements.get(id);
  }, createElement: node};
  const opening = {phase: 'mulligan', turn: 0, viewer_side: 'a',
    sides: {a: {hand: [], hand_count: 0}, b: {hand: [], hand_count: 0}}};
  const events = [
    {seq: 1, type: 'mulligan', text: '换牌完成', patch: {}},
    {seq: 2, side: 'a', type: 'turn', text: '我方 · 我方玩家的第一回合', patch: {phase: 'playing', turn: 1}},
    {seq: 3, type: 'draw', text: '抽牌', patch: {}},
  ];
  const window = {NTE_V2: {
    setHidden: (el, hidden) => { if (el) el.hidden = hidden; },
    getReplay: async () => ({room: {room_code: 'TEST', status: 'finished'},
      replay: {opening_board: opening, events}}),
    viewerSide: game => game.viewer_side || 'a',
    opponentSide: game => game.viewer_side === 'b' ? 'a' : 'b',
    sideState: (game, side) => game.sides[side],
  }};
  const sandbox = vm.createContext({window, document, console});
  for (const file of ['helpers', 'render', 'sync', 'replay']) {
    vm.runInContext(fs.readFileSync('app/modules/card_game/static/js/v2_table/' + file + '.js', 'utf8'), sandbox);
  }
  const table = window.NTE_V2_TABLE;
  const ctx = {page: node(), nodes: {}, mulliganIds: [], options: {},
    emptyState: node(), roomPanel: node(), tableLayout: node(), handDock: node()};
  for (const key of ['playerHand', 'opponentHand', 'mulliganRail', 'mulliganStage',
    'roomChip', 'phaseChip', 'connectionChip', 'endTurnBtn', 'concedeBtn', 'mulliganBtn',
    'mulliganStageCopy', 'turnCopy', 'waitChip']) ctx.nodes[key] = node();
  const frames = [];
  // Keep the actual hand renderer: its phase drives panel visibility and card placement.
  table.renderGame = renderCtx => {
    table.renderHands(renderCtx);
    frames.push({phase: renderCtx.game.phase, visible: !ctx.nodes.mulliganStage.hidden, viewer: renderCtx.viewer, opponent: renderCtx.opponent});
  };
  table.bindSync(ctx);
  if (animated) {
    table.createPlayer = options => ({cancel() {}, setOptions() {}, async play(batch) {
      for (const event of batch) options.applyPatch(event.patch, event);
    }});
  } else {
    ctx.bindPlayer = undefined;
    ctx.player = null;
  }
  await table.startReplay(ctx, 'TEST');
  for (let tries = 0; ctx.playing && tries < 30; tries++) await new Promise(setImmediate);
  assert.equal(ctx.playing, false);
  assert.ok(frames.some(frame => frame.phase === 'mulligan' && frame.visible));
  assert.ok(frames.some(frame => frame.phase === 'playing'));
  for (const frame of frames) assert.equal(frame.visible, frame.phase === 'mulligan', JSON.stringify(frame));
  assert.equal(ctx.page.classList.contains('is-mulligan'), false);
  const select = document.getElementById('v2-replay-turn');
  select.value = '0'; select.listeners.change();
  assert.equal(ctx.nodes.mulliganStage.hidden, false);
  // A single step within the opening must not hide the panel early.
  document.getElementById('v2-replay-next').listeners.click();
  await new Promise(setImmediate);
  assert.equal(ctx.nodes.mulliganStage.hidden, false);
  document.getElementById('v2-replay-next').listeners.click();
  await new Promise(setImmediate);
  assert.equal(ctx.nodes.mulliganStage.hidden, true, JSON.stringify({animated, index:ctx.index, phase:ctx.displayGame.phase, frames}));
  select.value = '0'; select.listeners.change();
  select.value = '2'; select.listeners.change();
  assert.equal(ctx.nodes.mulliganStage.hidden, true);
  const perspective = document.getElementById('v2-replay-perspective');
  const before = JSON.stringify(ctx.game.sides);
  perspective.listeners.click();
  assert.equal(ctx.index, 2);
  assert.equal(ctx.playing, false);
  assert.equal(frames.at(-1).viewer, 'b');
  assert.equal(frames.at(-1).opponent, 'a');
  assert.equal(ctx.viewer(), 'b');
  assert.equal(ctx.game.viewer_side, 'a'); // Reconstruction retains the recording's knowledge.
  assert.equal(JSON.stringify(ctx.game.sides), before);
  assert.equal(ctx.game.events[1].text, '对手 · 对手玩家的第一回合');
  document.getElementById('v2-replay-next').listeners.click();
  await new Promise(setImmediate);
  assert.equal(frames.at(-1).viewer, 'b'); // Animated patch callbacks keep the selected view.
  perspective.listeners.click();
  assert.equal(ctx.index, 3);
  assert.equal(ctx.viewer(), 'a');
  assert.equal(ctx.game.events[1].text, '我方 · 我方玩家的第一回合');
  assert.equal(events[1].text, '我方 · 我方玩家的第一回合');
}
(async () => { await scenario(true); await scenario(false); })().catch(error => { console.error(error); process.exitCode = 1; });
'''
        result = subprocess.run(['node', '-e', script], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
