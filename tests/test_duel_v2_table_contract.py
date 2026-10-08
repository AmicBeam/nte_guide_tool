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
REPLAY_JS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_table' / 'replay.js'
HOME_JS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_home.js'
REPLAYS_JS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_replays.js'
BUILD_JS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_build.js'
CODEX_JS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_codex.js'
HOME_TEMPLATE = ROOT / 'app' / 'modules' / 'card_game' / 'templates' / 'card_game' / 'v2_index.html'
REPLAYS_TEMPLATE = ROOT / 'app' / 'modules' / 'card_game' / 'templates' / 'card_game' / 'v2_replays.html'
TABLE_CSS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'css' / 'v2_table.css'
LAYOUT_CSS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'css' / 'v2_table' / 'layout.css'
CARDS_CSS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'css' / 'v2_table' / 'cards.css'
TUTORIAL_JS = ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_table' / 'tutorial.js'
JS_FILES = (TABLE_JS, HELPERS_JS, RENDER_JS, INTERACTION_JS, TUTORIAL_JS, SYNC_JS, REPLAY_JS, HOME_JS, REPLAYS_JS)


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
            'v2-opponent-harmony-source',
            'v2-player-harmony-source',
            'v2-opponent-hand',
            'v2-player-hand',
            'v2-fx-stage',
            'v2-cast-zone',
            'v2-play-area',
            'v2-mulligan-stage',
            'v2-mulligan-rail',
            'v2-initiative-overlay',
            'v2-initiative-chip',
            'v2-target-mask',
            'v2-target-cancel',
            'v2-history-drawer',
            'v2-menu-overlay',
            'v2-mute-toggle',
            'v2-motion-toggle',
            'v2-speed-1',
            'v2-speed-2',
            'v2-card-preview',
            'v2-replay-bar',
            'v2-replay-play',
            'v2-replay-pause',
            'v2-replay-prev',
            'v2-replay-next',
            'v2-replay-step',
            'v2-replay-link',
            'v2-tutorial-mask',
            'v2-tutorial-modal',
            'v2-tutorial-cue',
        ]
        for node_id in required_ids:
            self.assertIn(f'id="{node_id}"', html, node_id)
        self.assertNotIn('id="v2-play-history"', html)
        self.assertNotIn('id="v2-play-history-detail"', html)
        self.assertNotIn('aria-label="最近公开出牌"', html)
        self.assertNotIn('aria-label="公开出牌详情"', html)

        self.assertIn('data-entity-id="a:nanali"', html)
        self.assertIn('data-entity-id="b:xun"', html)
        self.assertIn('data-player-side="a"', html)
        self.assertIn('data-player-side="b"', html)
        self.assertIn('data-hand-side="a"', html)
        self.assertIn('data-hand-side="b"', html)
        self.assertIn('data-drop-role="cast"', html)
        self.assertIn('data-drop-role="play-area"', html)
        self.assertIn('data-drop-role="player-front"', html)
        self.assertIn('v2-hand-stack', html)
        self.assertIn('v2-side-actions', html)
        self.assertIn('v2-ultimate', html)
        self.assertIn('v2-normal-attack', html)

        animation_css = html.index("filename='css/v2_animation.css'")
        table_css = html.index("filename='css/v2_table.css'")
        animation_js = html.index("filename='js/v2_animation.js'")
        helpers_js = html.index("filename='js/v2_table/helpers.js'")
        tutorial_js = html.index("filename='js/v2_table/tutorial.js'")
        replay_js = html.index("filename='js/v2_table/replay.js'")
        table_js = html.index("filename='js/v2_table.js'")
        self.assertLess(animation_css, table_css)
        self.assertLess(animation_js, helpers_js)
        self.assertLess(helpers_js, table_js)
        self.assertLess(tutorial_js, table_js)
        self.assertLess(replay_js, table_js)
        home = HOME_TEMPLATE.read_text(encoding='utf-8')
        self.assertIn('新手教学', home)
        self.assertIn('id="v2-tutorial-prev"', home)
        self.assertIn('id="v2-tutorial-start"', home)
        self.assertIn('进入', home)
        self.assertNotIn('跳过新手教学', home)
        self.assertNotIn('跳过说明', (ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_table' / 'tutorial.js').read_text(encoding='utf-8'))
        self.assertNotIn('再打这一关', home)
        self.assertNotIn('开始这一关', home)
        self.assertIn('v2-replay-bar', html)
        home = HOME_TEMPLATE.read_text(encoding='utf-8')
        self.assertIn("url_for('main.replays_page')", home)
        self.assertIn('v2-home-play', home)
        self.assertIn('id="v2-home-link"', html)
        self.assertIn('id="v2-result-overlay"', html)
        self.assertIn('收藏录像', html)
        self.assertIn('离开房间', html)
        self.assertIn('leaveRoom', TABLE_JS.read_text(encoding='utf-8'))
        self.assertNotIn('id="v2-replay-list"', home)
        replays = REPLAYS_TEMPLATE.read_text(encoding='utf-8')
        self.assertIn('id="v2-replay-list"', replays)
        self.assertIn('formatReplayTime', (ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_replays.js').read_text(encoding='utf-8'))
        self.assertIn('个回合', REPLAYS_JS.read_text(encoding='utf-8'))
        self.assertIn('id="v2-replay-turn"', html)
        self.assertNotIn('id="v2-replay-time"', html)
        self.assertIn('id="v2-replay-bar"', html)
        self.assertLess(html.index('id="v2-opponent-hand"'), html.index('id="v2-replay-bar"'))
        self.assertLess(html.index('id="v2-replay-bar"'), html.index('id="v2-phase-chip"'))
        self.assertNotIn('v2-arena', html[html.index('id="v2-replay-bar"'):html.index('id="v2-phase-chip"')])
        self.assertIn('mulliganStage', RENDER_JS.read_text(encoding='utf-8'))
        self.assertIn('showMulligan', RENDER_JS.read_text(encoding='utf-8'))
        self.assertIn('animateMulliganSwap', HELPERS_JS.read_text(encoding='utf-8'))
        self.assertIn('is-leaving', CARDS_CSS.read_text(encoding='utf-8'))
        self.assertIn('is-entering', CARDS_CSS.read_text(encoding='utf-8'))
        replay_js = REPLAY_JS.read_text(encoding='utf-8')
        self.assertIn('presentOwnPlays', replay_js)
        self.assertIn('ctx.index = 0', replay_js)
        self.assertIn('function replayViewer', replay_js)
        self.assertIn("ctx.game.phase === 'mulligan'", replay_js)
        self.assertIn('mulliganStageCopy', replay_js)
        self.assertNotIn("ctx.viewer = ctx.game.viewer_side", replay_js)
        self.assertIn('presentOwnPlays', (ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_animation.js').read_text(encoding='utf-8'))
        self.assertIn('对局回放', replays)

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
        self.assertIn('v2-portrait-vitals', css)
        self.assertIn('v2-down-badge', css)
        self.assertIn('harmony-ready', css)
        self.assertIn('harmony-source', css)
        self.assertIn('harmony-available', css)
        self.assertIn('v2-harmony-source', layout)
        self.assertIn('v2-front-frame', layout)
        self.assertIn('is-unusable', css)
        self.assertIn('v2-frame-stats', css)
        self.assertIn('v2-history-drawer', css)
        common = (ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'css' / 'v2_common.css').read_text(encoding='utf-8')
        self.assertIn('[data-card-type="battle"]', common)
        self.assertIn('.v2-hand-card p strong', common)
        self.assertIn('[data-card-type="tactic"]', common)
        self.assertIn('[data-card-type="form"]', common)
        self.assertNotIn('grid-template-columns: minmax(0,1fr) 210px', css)
        self.assertIn('v2-hand-stack', layout)
        self.assertIn('--avatar-size: calc(42px * var(--table-layout-scale, 1))', layout)
        self.assertIn('--avatar-size: calc(32px * var(--table-layout-scale, 1))', layout)
        self.assertIn('--bench-card-w', layout)
        self.assertIn('grid-template-columns: calc(156px * var(--table-layout-scale, 1)) minmax(0, 1fr)', layout)
        self.assertIn('v2-side-actions', layout)
        self.assertIn('is-mulligan', css)
        self.assertIn('v2-mulligan-stage', layout)
        self.assertIn('build-antique-interior.webp', layout)
        self.assertIn('.v2-front-zone.is-empty', layout)
        self.assertNotIn('padding: 2px 18%', layout)

    def test_javascript_preserves_safety_and_classified_drops(self) -> None:
        table = TABLE_JS.read_text(encoding='utf-8')
        helpers = HELPERS_JS.read_text(encoding='utf-8')
        render = RENDER_JS.read_text(encoding='utf-8')
        interaction = INTERACTION_JS.read_text(encoding='utf-8')
        sync = SYNC_JS.read_text(encoding='utf-8')
        replay = REPLAY_JS.read_text(encoding='utf-8')
        home = HOME_JS.read_text(encoding='utf-8')
        replays = REPLAYS_JS.read_text(encoding='utf-8')
        combined = '\n'.join([table, helpers, render, interaction, sync, replay, home, replays])

        self.assertIn('hasUnplayedPresentation', TABLE_JS.read_text(encoding='utf-8'))
        self.assertIn('NTEDuelPresentation', helpers)
        self.assertIn('createPlayer', helpers)
        self.assertIn('applyPatch', helpers)
        self.assertIn('eventsAfter', helpers)
        self.assertIn("kind: 'cast'", helpers)
        self.assertIn("kind: 'form'", helpers)

        self.assertIn('data-entity-id', render)
        self.assertIn('is-sortie', render)
        self.assertNotIn('v2-attack-btn', render)
        self.assertIn('energy_max', render)
        self.assertIn('harmony-ready', render)
        self.assertIn('harmony-source', render)
        self.assertIn('harmony_available', render)
        self.assertIn('harmonySourceCaptionHtml', helpers)
        self.assertIn('环合来源方：', helpers)
        self.assertIn('harmonySourceCaptionHtml', render)
        self.assertIn('cardDescriptionHtml', render)
        self.assertIn('historyDescription', render)
        self.assertIn('historyDescription', HELPERS_JS.read_text(encoding='utf-8'))
        self.assertIn('v2-hand-group-start', render)
        self.assertIn('copyViewerHand', helpers)
        self.assertIn('copyViewerHand', sync)
        self.assertIn('data.arid', render)
        self.assertIn('荒时 ', render)
        self.assertIn('v2-hand-group-start', CARDS_CSS.read_text(encoding='utf-8'))
        self.assertIn('ultimate_available', render)
        self.assertIn('is-instant-spent', render)
        self.assertIn('is-instant-spent', LAYOUT_CSS.read_text(encoding='utf-8'))
        self.assertIn('cardTimingClass', render)
        self.assertIn('legal.is-instant', CARDS_CSS.read_text(encoding='utf-8'))
        self.assertIn('border-style: dashed', CARDS_CSS.read_text(encoding='utf-8'))
        self.assertIn('cardDescriptionHtml', interaction)
        self.assertIn('v2-choice-card', interaction)
        self.assertIn('hoverCardMarkup(card)', interaction)
        api = (ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_api.js').read_text(encoding='utf-8')
        self.assertIn('Number(card.shield) > 0', api)
        self.assertNotIn("card.shield != null && card.shield !== ''", api)
        self.assertIn('CARD_TYPE_ORDER', api)
        self.assertIn('tactic: 0, battle: 1, form: 2', api)
        self.assertNotIn("'/4'", render)
        self.assertIn('startBoardTargetSelect', interaction)
        self.assertIn('is-targeting', interaction)
        self.assertNotIn("id === 'player'", interaction)
        self.assertIn('.v2-player-hud.target-legal', interaction)
        self.assertIn('target-dim', interaction)
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
        self.assertIn('splitPresentationAtTurn', helpers)
        self.assertIn('waitMulliganHandDeal', sync)
        self.assertIn('dealMulliganToHand', sync)
        self.assertIn('applyEventPatches', helpers)
        self.assertIn('openResultFromEvent', TABLE_JS.read_text(encoding='utf-8'))
        self.assertIn('onFinish', sync)
        self.assertIn('player.cancel', combined)
        self.assertIn('setOptions', combined)
        self.assertIn('muted', combined)
        self.assertIn('reducedMotion', combined)
        self.assertIn('canonicalEntityId', helpers)
        self.assertIn('entryMatchesTarget', helpers)
        self.assertIn('appendPublicPlay', helpers)
        self.assertIn('publicPlayOwner', helpers)
        self.assertIn('historyHeadline', render)
        self.assertIn('appendPresentedEvent', helpers)
        self.assertIn('appendPresentedEvent', sync)
        self.assertIn('#v2-history-btn', TABLE_CSS.read_text(encoding='utf-8'))
        animation_css = (ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'css' / 'v2_animation.css').read_text(encoding='utf-8')
        self.assertNotIn('-webkit-line-clamp', animation_css)
        self.assertIn('min-height: calc(430px * var(--table-layout-scale, 1))', animation_css)
        animation = (ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_animation.js').read_text(encoding='utf-8')
        self.assertIn('_presentDraw', animation)
        self.assertIn('_isSilent', animation)
        self.assertIn('draw-public', animation)
        self.assertIn('!ctx.playingEvents', SYNC_JS.read_text(encoding='utf-8'))
        self.assertIn('v2-portrait-vitals', render)
        self.assertIn('usePortrait: true', render)
        self.assertNotIn('v2-play-chip', render)
        self.assertNotIn('item.viewer', render)
        self.assertIn('data-instance-id', render)
        self.assertIn('resetRoomPresentation', sync)
        self.assertIn('playerGeneration', sync)
        self.assertNotIn("split(':').pop()", interaction)
        self.assertIn('legalDropFor', helpers)
        self.assertIn('groupPlayEntries', helpers)
        self.assertIn('startReplay', replay)
        self.assertIn('gameAt', replay)
        self.assertIn('replayMode', combined)
        self.assertIn('/table?replay=', combined)
        self.assertIn('listReplays', replays)
        self.assertIn('starReplay', replays)
        self.assertIn('v2-replay-avatar', replays)
        self.assertIn('data-star-replay', replays)
        self.assertIn("resumeBtn.textContent = finished ? '离开房间' : '返回牌桌'", home)
        self.assertNotIn('listReplays', home)

    def test_new_build_starts_without_characters(self) -> None:
        source = BUILD_JS.read_text(encoding='utf-8')
        self.assertIn("name: '新的构筑'", source)
        self.assertIn('character_ids: []', source)
        self.assertIn("nameInput.value = '新的构筑'", source)
        self.assertNotIn("applyDeck(starterDeck(), '已创建草稿", source)

    def test_keyword_markup_bolds_pierce_in_passive_copy(self) -> None:
        api = (ROOT / 'app' / 'modules' / 'card_game' / 'static' / 'js' / 'v2_api.js').read_text(encoding='utf-8')
        self.assertIn('瞬发|响应|穿透', api)
        self.assertIn("kind === 'ok'", api)
        self.assertIn('_bannerTimer', api)
        interaction = INTERACTION_JS.read_text(encoding='utf-8')
        self.assertIn("cardDescriptionMarkup(found.passive", interaction)
        self.assertIn("cardDescriptionMarkup(character.passive", CODEX_JS.read_text(encoding='utf-8'))
        self.assertIn("cardDescriptionMarkup(character.passive", BUILD_JS.read_text(encoding='utf-8'))

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
