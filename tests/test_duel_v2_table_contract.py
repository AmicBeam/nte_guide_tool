import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'app' / 'modules' / 'card_game' / 'templates' / 'card_game' / 'v2_table.html'
TABLE_JS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_table.js'
HELPERS_JS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_table' / 'helpers.js'
RENDER_JS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_table' / 'render.js'
INTERACTION_JS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_table' / 'interaction.js'
SYNC_JS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_table' / 'sync.js'
TABLE_CSS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'css' / 'v2_table.css'
LAYOUT_CSS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'css' / 'v2_table' / 'layout.css'
CARDS_CSS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'css' / 'v2_table' / 'cards.css'
JS_FILES = (TABLE_JS, HELPERS_JS, RENDER_JS, INTERACTION_JS, SYNC_JS)


class DuelV2TableContractTest(unittest.TestCase):
    def test_template_loads_animation_before_table_and_keeps_stable_nodes(self) -> None:
        html = TEMPLATE.read_text(encoding='utf-8')
        required_ids = [
            'v2-table-page',
            'v2-opponent-life',
            'v2-player-life',
            'v2-opponent-hp',
            'v2-player-hp',
            'v2-player-ap',
            'v2-end-turn-btn',
            'v2-opponent-front',
            'v2-player-front',
            'v2-opponent-hand',
            'v2-player-hand',
            'v2-fx-stage',
            'v2-cast-zone',
            'v2-history-drawer',
            'v2-menu-overlay',
            'v2-mute-toggle',
            'v2-motion-toggle',
            'v2-skip-btn',
            'v2-skip-inline-btn',
            'v2-speed-1',
            'v2-speed-2',
            'v2-card-preview',
            'v2-play-history',
            'v2-play-history-detail',
        ]
        for node_id in required_ids:
            self.assertIn(f'id="{node_id}"', html, node_id)

        self.assertIn('data-entity-id="a:nanali"', html)
        self.assertIn('data-entity-id="b:xun"', html)
        self.assertIn('data-player-side="a"', html)
        self.assertIn('data-player-side="b"', html)
        self.assertIn('data-hand-side="a"', html)
        self.assertIn('data-hand-side="b"', html)
        self.assertIn('data-drop-role="cast"', html)
        self.assertIn('data-drop-role="player-front"', html)
        self.assertIn('v2-hand-stack', html)

        animation_css = html.index("filename='css/v2_animation.css'")
        table_css = html.index("filename='css/v2_table.css'")
        animation_js = html.index("filename='js/v2_animation.js'")
        helpers_js = html.index("filename='js/v2_table/helpers.js'")
        table_js = html.index("filename='js/v2_table.js'")
        self.assertLess(animation_css, table_css)
        self.assertLess(animation_js, helpers_js)
        self.assertLess(helpers_js, table_js)

    def test_layout_keeps_core_hud_on_one_screen(self) -> None:
        layout = LAYOUT_CSS.read_text(encoding='utf-8')
        css = TABLE_CSS.read_text(encoding='utf-8') + layout + CARDS_CSS.read_text(encoding='utf-8')
        self.assertIn('height: 100dvh', layout)
        self.assertIn('overflow: hidden', layout)
        self.assertIn('grid-template-rows: var(--table-top) minmax(0, 1fr) var(--table-bottom)', layout)
        self.assertNotIn('.v2-arena {\n  overflow: auto', layout)
        self.assertIn('@media (max-height: 520px) and (min-width: 700px)', layout)
        self.assertIn('max-height: 430px', layout)
        self.assertIn('v2-character-art', css)
        self.assertIn('v2-cost-badge', css)
        self.assertIn('v2-history-drawer', css)
        self.assertNotIn('grid-template-columns: minmax(0,1fr) 210px', css)
        self.assertIn('v2-hand-stack', layout)
        self.assertIn('--avatar-size: 36px', layout)
        self.assertIn('--avatar-size: 28px', layout)
        self.assertIn('grid-template-columns: 156px minmax(0, 1fr) 118px', layout)
        self.assertIn('.v2-front-zone.is-empty', layout)
        self.assertNotIn('padding: 2px 18%', layout)

    def test_javascript_preserves_safety_and_classified_drops(self) -> None:
        table = TABLE_JS.read_text(encoding='utf-8')
        helpers = HELPERS_JS.read_text(encoding='utf-8')
        render = RENDER_JS.read_text(encoding='utf-8')
        interaction = INTERACTION_JS.read_text(encoding='utf-8')
        sync = SYNC_JS.read_text(encoding='utf-8')
        combined = '\n'.join([table, helpers, render, interaction, sync])

        self.assertIn('NTEDuelPresentation', helpers)
        self.assertIn('createPlayer', helpers)
        self.assertIn('applyPatch', helpers)
        self.assertIn('eventsAfter', helpers)
        self.assertIn("kind: 'cast'", helpers)
        self.assertIn("kind: 'form'", helpers)

        self.assertIn('data-entity-id', render)
        self.assertIn('出击中', render)
        self.assertIn('v2-attack-btn', render)
        self.assertIn('确认出击', interaction)
        self.assertIn("showCharacterDetail", interaction)
        self.assertIn("role === 'cast'", helpers)
        self.assertIn('legalDropFor', helpers)
        self.assertNotIn("if (attack && !awaken) {\n        submitEntry(attack)", interaction)

        self.assertIn('expectedVersion', sync)
        self.assertIn('requestId', sync)
        self.assertIn('retryingMulligan', sync)
        self.assertIn('ignoreIfBusy', sync)
        self.assertIn('nextVersion < Number(ctx.game.version)', sync)
        self.assertIn('pendingRequest', sync)
        self.assertIn('player.play', sync)
        self.assertIn('alignAuthoritative', sync)
        self.assertIn('applyDisplayPatch', sync)
        self.assertIn('player.cancel', combined)
        self.assertIn('setOptions', combined)
        self.assertIn('muted', combined)
        self.assertIn('reducedMotion', combined)
        self.assertIn('canonicalEntityId', helpers)
        self.assertIn('entryMatchesTarget', helpers)
        self.assertIn('appendPublicPlay', helpers)
        self.assertIn('data-instance-id', render)
        self.assertIn('resetRoomPresentation', sync)
        self.assertIn('playerGeneration', sync)
        self.assertNotIn("split(':').pop()", interaction)
        self.assertIn('legalDropFor', helpers)
        self.assertIn('groupPlayEntries', helpers)

    def test_javascript_syntax(self) -> None:
        for path in JS_FILES:
            completed = subprocess.run(
                ['node', '--check', str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_fake_player_applies_patches_without_final_hp_first(self) -> None:
        script = r'''
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const helpersPath = path.resolve('app/modules/card_game/static/js/v2_table/helpers.js');
const context = {
  window: {},
  console,
};
context.window.NTE_V2 = {
  viewerSide: (game) => (game && game.viewer_side) || 'a',
  opponentSide: (game) => ((game && game.viewer_side) === 'b' ? 'a' : 'b'),
  sideState: (game, side) => ((game && game.sides) || {})[side] || {},
  characterFallback: () => null,
  legalEntries: (game) => (game && game.legal_actions) || [],
};
context.window = context.window;
vm.createContext(context);
vm.runInContext(fs.readFileSync(helpersPath, 'utf8'), context);
const TABLE = context.window.NTE_V2_TABLE;
const game = {
  version: 3,
  viewer_side: 'a',
  sides: {
    a: { hp: 20, ap: 2, characters: [{ id: 'nanali', hp: 7 }] },
    b: { hp: 18, ap: 2, characters: [{ id: 'xun', hp: 8 }] },
  },
};
const patched = TABLE.applyPatch(game, {
  sides: {
    b: { hp: 15, characters: [{ id: 'xun', hp: 5 }] },
  },
});
if (patched.sides.b.hp !== 15) throw new Error('player hp patch failed');
if (patched.sides.b.characters[0].hp !== 5) throw new Error('character merge failed');
if (patched.sides.a.hp !== 20) throw new Error('unrelated side mutated');
const events = [
  { seq: 1, action_id: '1', type: 'attack', patch: { sides: { b: { hp: 17 } } } },
  { seq: 2, action_id: '1', type: 'damage', patch: { sides: { b: { hp: 15 } } } },
];
const later = TABLE.eventsAfter(events, 1);
if (later.length !== 1 || later[0].seq !== 2) throw new Error('cursor filter failed');
let hpTrace = [];
const player = new TABLE.FallbackPlayer({
  applyPatch: (patch) => {
    hpTrace.push(patch.sides.b.hp);
  },
  onBusy: () => {},
});
player.play(later, { instant: true }).then(() => {
  if (hpTrace.join(',') !== '15') throw new Error('expected patch-only playback, got ' + hpTrace);
  const interaction = TABLE.inferInteraction(game, {
    action: { type: 'play_card', card_id: 'c1', target_id: 'a:xun' },
  });
  if (interaction.kind !== 'target') throw new Error('target inference failed');
  const rejected = TABLE.legalDropFor(
    'card',
    TABLE.groupPlayEntries(game, [{ action: { type: 'play_card', card_id: 'c1', target_id: 'a:xun' }, interaction: interaction }]),
    { closest: (sel) => sel === '[data-entity-id]' ? { getAttribute: () => 'b:xun' } : { getAttribute: () => 'character' } }
  );
  if (rejected) throw new Error('enemy namesake should be rejected');
  console.log('FAKE_OK');
});
'''
        completed = subprocess.run(
            ['node', '-e', script],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        self.assertIn('FAKE_OK', completed.stdout)


if __name__ == '__main__':
    unittest.main()
