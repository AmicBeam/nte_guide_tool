import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.modules.shaft.service import get_shaft_catalog_payload, simulate_shaft_axis
from app.modules.shaft.domain.catalog import get_record_map, load_shaft_catalog


ROOT = Path(__file__).resolve().parents[1]
SHAFT_JS = ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'js' / 'shaft.js'
SHAFT_ENGINE_JS = ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'js' / 'shaft_engine.js'
SHAFT_CSS = ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css'
SHAFT_TEMPLATE = ROOT / 'app' / 'modules' / 'shaft' / 'templates' / 'shaft' / 'index.html'


def _js_number_constant(name: str) -> int:
    source = SHAFT_JS.read_text(encoding='utf-8')
    match = re.search(rf'\bconst\s+{re.escape(name)}\s*=\s*([0-9]+)\s*;', source)
    if not match:
        raise AssertionError(f'frontend constant {name} is missing or not numeric.')
    return int(match.group(1))


class ShaftBuffStatusUiTestCase(unittest.TestCase):
    def test_rotation_header_has_no_manual_calculation_button(self) -> None:
        template = SHAFT_TEMPLATE.read_text(encoding='utf-8')
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertNotIn('id="shaft-run-btn"', template)
        self.assertNotIn("$('shaft-run-btn')", source)

    def test_shaft_source_version_is_1_0_2(self) -> None:
        catalog = get_shaft_catalog_payload()
        template = SHAFT_TEMPLATE.read_text(encoding='utf-8')

        self.assertEqual(catalog['source_meta']['version_label'], '异环云配队 1.0.2')
        self.assertIn('{% block title %}异环云配队——排轴计算{% endblock %}', template)
        self.assertIn('<h1>异环云配队——排轴计算</h1>', template)

    def test_unimplemented_awakenings_are_marked_in_tooltips(self) -> None:
        catalog = get_shaft_catalog_payload()
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertNotIn('implemented', catalog['awakenings']['浔'][2])
        self.assertNotIn('implemented', catalog['awakenings']['埃德嘉'][0])
        self.assertEqual(
            catalog['awakenings']['浔'][3]['implementation_status'],
            'out_of_scope',
        )
        self.assertNotIn('implemented', catalog['awakenings']['浔'][0])
        self.assertIn("entry.implementation_status === 'out_of_scope'", source)
        self.assertIn("? '（不实装）'", source)
        self.assertIn("entry.implemented === false ? '（未实装）' : ''", source)
        self.assertIn("`${entry.title || '未命名'}${implementationLabel}", source)

    def test_enemy_uses_one_initial_resistance_input_and_personal_resources_have_ui_slots(self) -> None:
        template = SHAFT_TEMPLATE.read_text(encoding='utf-8')
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('id="shaft-initial-resistance-input"', template)
        self.assertIn('<span>初始抗性</span>', template)
        self.assertNotIn('id="shaft-resistance-grid"', template)
        self.assertNotIn('data-resistance-element', source)
        self.assertIn("RESISTANCE_ELEMENTS.map((element) => [element, resistance])", source)
        self.assertIn("snapshot.personal_resources", source)
        self.assertIn("resource.initial_personal_resources", source)
        self.assertIn("${personalResourceLabels}", source)

    def test_timeline_always_shows_iloy_reverie_and_hides_requiem_nightmare_resource(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn("char_a01c39f576: ['臆想']", source)
        self.assertIn("char_31c5130304: ['真理之匙']", source)
        self.assertIn("new Set(['噩梦'])", source)
        self.assertIn(
            'timelinePersonalResourceEntries(character, resources.personalResources)',
            source,
        )

    def test_axis_info_keeps_fields_and_actions_on_one_row(self) -> None:
        css = (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css').read_text(encoding='utf-8')
        command_panel = re.search(r'\.shaft-command-panel\s*\{(?P<body>[^}]*)\}', css)

        self.assertIsNotNone(command_panel)
        self.assertIn('grid-template-columns: minmax(0, 1fr) auto;', command_panel.group('body'))

    def test_loop_axis_uses_settings_dialog_with_character_resources(self) -> None:
        template = SHAFT_TEMPLATE.read_text(encoding='utf-8')
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('id="shaft-loop-settings-btn"', template)
        self.assertIn('id="shaft-loop-settings-dialog"', template)
        self.assertIn('id="shaft-loop-enabled"', template)
        self.assertIn('id="shaft-loop-resource-list"', template)
        self.assertIn('data-loop-initial-energy', source)
        self.assertIn('data-loop-initial-harmony', source)
        self.assertIn('data-loop-initial-reaction', source)
        self.assertIn("'<option value=\"\">不携带</option>'", source)
        self.assertIn('.filter((option) => option && Number(option.duration_ticks) > 0)', source)
        self.assertIn('data-loop-initial-personal-resource', source)
        self.assertIn('function loopPersonalResourceDefinitions(member)', source)
        self.assertIn("'personal_resource_cost', 'personal_resource_gain', 'personal_resource_threshold'", source)
        self.assertIn('personal_resources: personalResources,', source)
        self.assertIn('reaction,', source)
        self.assertIn('function confirmLoopSettings()', source)
        self.assertNotIn("$('shaft-loop-enabled').addEventListener('change', (event) => {", source)


def _frontend_team_panel_bonus_defaults() -> dict[str, float]:
    source = SHAFT_JS.read_text(encoding='utf-8')
    match = re.search(r'const\s+DEFAULT_TEAM_PANEL_BONUS\s*=\s*\{(?P<body>.*?)\};', source, re.S)
    if not match:
        raise AssertionError('frontend team panel bonus defaults are missing.')
    return {
        key: float(value)
        for key, value in re.findall(r'([a-z_]+)\s*:\s*([0-9.]+)', match.group('body'))
    }


def _is_q_action(action: dict) -> bool:
    return str(action.get('action_type') or '') == 'Q' or str(action.get('damage_type') or '') == 'Q'


def _zero_q_actions() -> list[dict]:
    catalog = load_shaft_catalog()
    actions = []
    used_characters = set()
    for action in catalog['actions']:
        if not _is_q_action(action) or int(action.get('duration_ticks') or 0) != 0:
            continue
        character_id = str(action.get('character_id') or '')
        if character_id in used_characters:
            continue
        used_characters.add(character_id)
        actions.append(action)
        if len(actions) == 4:
            break
    if len(actions) != 4:
        raise AssertionError('expected at least four zero-duration Q actions.')
    return actions


def _frontend_timeline_layout(axis_payload: dict) -> list[dict]:
    catalog = load_shaft_catalog()
    actions_by_id = get_record_map(catalog['actions'])
    result = simulate_shaft_axis(axis_payload)['result']
    result_detail_by_step_id = {detail['step_id']: detail for detail in result['details']}

    zero_action_visual_ticks = _js_number_constant('ZERO_ACTION_VISUAL_TICKS')
    timeline_tick_px = _js_number_constant('TIMELINE_TICK_PX')
    timeline_label_px = _js_number_constant('TIMELINE_LABEL_PX')
    min_action_card_px = _js_number_constant('MIN_ACTION_CARD_PX')

    details = []
    for step in axis_payload['steps']:
        action = actions_by_id[str(step['action_id'])]
        result_detail = result_detail_by_step_id.get(step['id'], {})
        duration_ticks = max(0, int(action.get('duration_ticks') or 0))
        display_start_tick = int(result_detail.get('display_start_tick') or result_detail.get('visual_start_tick') or step.get('start_tick') or 0)
        display_end_tick = int(result_detail.get('display_end_tick') or (display_start_tick + duration_ticks))
        display_duration_ticks = int(result_detail.get('display_duration_ticks') if result_detail.get('display_duration_ticks') is not None else duration_ticks)
        nominal_display_visual_end_tick = display_start_tick + (
            duration_ticks if duration_ticks > 0 else zero_action_visual_ticks
        )
        fallback_display_visual_end_tick = (
            max(nominal_display_visual_end_tick, int(result_detail.get('visual_end_tick') or nominal_display_visual_end_tick))
            if _is_q_action(action)
            else nominal_display_visual_end_tick
        )
        display_visual_end_tick = int(
            result_detail.get('display_visual_end_tick') or fallback_display_visual_end_tick
        )
        details.append({
            **result_detail,
            'step_id': step['id'],
            'slot': int(step.get('slot') or 0),
            'action_id': step['action_id'],
            'display_start_tick': display_start_tick,
            'display_end_tick': display_end_tick,
            'display_duration_ticks': display_duration_ticks,
            'display_visual_end_tick': display_visual_end_tick,
            'nominal_display_visual_end_tick': nominal_display_visual_end_tick,
            'is_background_damage': False,
            'is_basic_background': bool(step.get('placement') == 'background'),
        })

    expansion_by_tick: dict[int, int] = {}
    for detail in details:
        action = actions_by_id[str(detail['action_id'])]
        display_duration_ticks = int(detail['display_duration_ticks'])
        tick = max(0, int(detail['display_start_tick']))
        visual_end_tick = int(detail['display_visual_end_tick'])
        blocks_track = not detail.get('is_background_damage') or detail.get('is_basic_background')
        is_zero_q = display_duration_ticks == 0 and not detail.get('is_background_damage') and _is_q_action(action)
        if not blocks_track and not is_zero_q:
            continue
        if not is_zero_q:
            continue
        display_end_tick = max(
            tick + 1,
            max(tick + zero_action_visual_ticks, int(detail['nominal_display_visual_end_tick'])),
        )
        raw_width = max(1, display_end_tick - tick) * timeline_tick_px
        extra_px = max(0, min_action_card_px - raw_width)
        if extra_px > 0:
            expansion_by_tick[tick] = max(expansion_by_tick.get(tick, 0), extra_px)

    expansion_breaks = [
        {
            'tick': tick,
            'extra_px': extra_px,
        }
        for tick, extra_px in sorted(expansion_by_tick.items())
    ]

    def visual_offset_px(tick: int) -> int:
        safe_tick = max(0, int(tick or 0))
        return safe_tick * timeline_tick_px + sum(
            item['extra_px']
            for item in expansion_breaks
            if item['tick'] < safe_tick
        )

    def left_px(tick: int) -> int:
        return timeline_label_px + visual_offset_px(tick)

    def width_px(start: int, end: int, visual_end: int, min_width: int = 1) -> int:
        fixed_end = visual_end or end or (int(start or 0) + zero_action_visual_ticks)
        if int(fixed_end or start) <= int(start or 0):
            return min_width
        return max(
            min_width,
            visual_offset_px(max(fixed_end, int(start or 0) + 1)) - visual_offset_px(start),
        )

    layout = []
    for detail in sorted(details, key=lambda item: int(item['display_start_tick'])):
        action = actions_by_id[str(detail['action_id'])]
        is_zero_q = int(detail['display_duration_ticks']) == 0 and _is_q_action(action)
        width = min_action_card_px if is_zero_q else width_px(
            int(detail['display_start_tick']),
            int(detail['display_end_tick']),
            int(detail['display_visual_end_tick']),
        )
        if is_zero_q and int(detail['display_visual_end_tick']) > int(detail['display_start_tick']) + zero_action_visual_ticks:
            width = width_px(
                int(detail['display_start_tick']),
                int(detail['display_end_tick']),
                int(detail['display_visual_end_tick']),
            )
        left = left_px(int(detail['display_start_tick']))
        layout.append({
            'step_id': detail['step_id'],
            'left_px': left,
            'width_px': width,
            'right_px': left + width,
        })
    return layout


def _calculation_tick_from_visual(visual_tick: int, q_starts: list[int]) -> int:
    safe_tick = max(0, int(visual_tick or 0))
    offset = sum(
        min(_js_number_constant('ZERO_ACTION_VISUAL_TICKS'), safe_tick - int(q_start_tick or 0))
        for q_start_tick in q_starts
        if int(q_start_tick or 0) < safe_tick
    )
    return max(0, safe_tick - offset)


def _visual_tick_label(visual_tick: int, q_starts: list[int]) -> str:
    safe_tick = max(0, int(visual_tick or 0))
    calculation_tick = _calculation_tick_from_visual(safe_tick, q_starts)
    return f'{calculation_tick / 10:.1f}s'


class ShaftFrontendTimelineLayoutTestCase(unittest.TestCase):
    def test_axis_preview_is_read_only_zoomable_and_action_only(self) -> None:
        template = SHAFT_TEMPLATE.read_text(encoding='utf-8')
        source = SHAFT_JS.read_text(encoding='utf-8')
        css = (
            ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css'
        ).read_text(encoding='utf-8')

        self.assertIn('id="shaft-preview-btn"', template)
        self.assertIn('id="shaft-axis-preview-dialog"', template)
        self.assertIn('id="shaft-axis-preview-viewport"', template)
        self.assertIn('id="shaft-axis-preview-meta"', template)
        self.assertIn('id="shaft-axis-preview-summary"', template)
        self.assertIn('id="shaft-axis-preview-cancel-btn"', template)
        self.assertIn('id="shaft-axis-preview-save-btn"', template)
        self.assertIn('function groupedPreviewActions', source)
        self.assertIn('return startsForeground(step, action);', source)
        self.assertIn(".join(' ')", source)
        self.assertIn('previous.durationTicks += detail.durationTicks', source)
        self.assertIn('function damageContributionShares', source)
        self.assertIn('function previewContributionShares', source)
        self.assertIn("label: '环合伤害'", source)
        self.assertIn("label: '倾陷伤害'", source)
        self.assertIn('percent: Math.round(Number(item.percent || 0))', source)
        self.assertIn('<span>伤害占比</span>', source)
        self.assertIn('function prioritizeAxisPreviewActionLabels', source)
        self.assertIn('function handleAxisPreviewWheel', source)
        self.assertIn('function previewMinimumTickPx', source)
        self.assertIn('function previewQVirtualIntervals', source)
        self.assertIn('const calculationEndTick = calculationTickFromVisual(endTick, previewQStarts);', source)
        self.assertIn('const visualTick = visualTickFromCalculation(tick, previewQStarts);', source)
        self.assertIn('class="shaft-axis-preview-grid-mark"', source)
        self.assertIn('.shaft-axis-preview-grid-mark', css)
        self.assertIn('async function openMarketAxisPreview(axisId, trigger)', source)
        self.assertIn("if (card.dataset.axisSource === 'market')", source)
        self.assertIn('await openMarketAxisPreview(Number(card.dataset.axisId), card)', source)
        self.assertIn('function marketAxisLocalTitle(payload)', source)
        self.assertIn('const suffix = ` - ${author}`;', source)
        self.assertIn('async function saveMarketAxisToLocal()', source)
        self.assertIn('title: marketAxisLocalTitle(payload)', source)
        self.assertIn('axis: payload.axis', source)
        self.assertIn('result: payload.result', source)
        self.assertNotIn('shaft-buff-trigger-line', source[source.index('function renderAxisPreview'):source.index('function fitAxisPreview')])
        self.assertNotIn('shaft-reaction-damage-marker', source[source.index('function renderAxisPreview'):source.index('function fitAxisPreview')])
        self.assertIn('.shaft-axis-preview-bubble', css)
        self.assertIn('.shaft-axis-preview-summary', css)
        self.assertIn('.shaft-axis-preview-contributions', css)
        self.assertIn('.shaft-axis-preview-bubble.hide-duration > em', css)
        self.assertIn('padding-inline: 3px;', css)
        self.assertIn('.shaft-axis-preview-footer', css)

    def test_timeline_length_uses_foreground_and_clips_after_one_second_padding(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn(
            'const foregroundSteps = (state.axis?.steps || []).filter((step) => startsForeground(step, actionForStep(step)));',
            source,
        )
        self.assertIn('const foregroundDetails = details.filter((detail) => !detail.is_background_damage);', source)
        self.assertIn('const displayCutoffTick = foregroundAxisEndTick + TIMELINE_END_PADDING_TICKS;', source)
        self.assertIn(
            '.filter((detail) => Number(detail.display_start_tick ?? detail.start_tick ?? 0) < displayCutoffTick)',
            source,
        )
        self.assertIn('display_visual_end_tick: Math.min(', source)
        self.assertIn('return Math.max(1, axisEnd + TIMELINE_END_PADDING_TICKS);', source)
        self.assertIn('&& Number(event.visual_tick ?? event.tick ?? 0) <= displayCutoffTick', source)

    def test_timeline_drag_and_marquee_auto_scroll_at_viewport_edges(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('const TIMELINE_AUTO_SCROLL_EDGE_PX', source)
        self.assertIn('const TIMELINE_AUTO_SCROLL_MAX_PX', source)
        self.assertIn('function runTimelineAutoScroll()', source)
        self.assertIn('function updateTimelineMarquee(clientX, clientY)', source)
        self.assertIn('function updateTimelineDrag(event)', source)
        self.assertIn('shell.scrollLeft - Number(drag.originScrollLeft || 0)', source)
        self.assertIn('shell.scrollLeft - Number(marquee.originScrollLeft || 0)', source)
        self.assertIn('stopTimelineAutoScroll();', source)

    def test_action_detail_explains_hidden_buff_effect_values(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        css = (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css').read_text(encoding='utf-8')

        self.assertIn('buff?.effects || {}', source)
        self.assertNotIn('增益解释', source)
        self.assertNotIn('const appliedBuffs =', source)
        self.assertIn('<strong class="shaft-detail-line-list">${detailLines(appliedBuffExplanations)}</strong>', source)
        self.assertIn('<div class="shaft-detail-kv"><span>通伤</span><strong>${formatNumber((panelStats.all_dmg || 0) * 100, 1)}%</strong></div>', source)
        self.assertIn('<div class="shaft-detail-kv"><span>属伤</span><strong>${formatNumber((panelStats.element_dmg || 0) * 100, 1)}%</strong></div>', source)
        self.assertIn('const finalDamageBonus = Number(panelStats.final_dmg || 0);', source)
        self.assertIn('...(finalDamageBonus ? [`<div class="shaft-detail-kv"><span>最终伤害</span><strong>${formatNumber(finalDamageBonus * 100, 1)}%</strong></div>`] : [])', source)
        self.assertIn('panelStats.other_dmg', source)
        self.assertIn('- Number(panelStats.all_dmg || 0)', source)
        self.assertIn('- Number(panelStats.element_dmg || 0)', source)
        self.assertIn('<div class="shaft-detail-kv"><span>其他增伤</span><strong>${formatNumber(otherDamageBonus * 100, 1)}%</strong></div>', source)
        self.assertIn('<span>敌人结算属抗</span>', source)
        self.assertIn('detail?.formula_parts?.settled_resistance', source)
        self.assertIn('<span>敌人结算防御</span>', source)
        self.assertIn('detail?.formula_parts?.settled_defense', source)
        self.assertNotIn('期望暴击段', source)
        self.assertNotIn('expectedCriticalHits', source)
        self.assertIn('.shaft-detail-hero .shaft-detail-line-list', css)
        self.assertIn("line_hidden_reason", (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'js' / 'shaft_engine.js').read_text(encoding='utf-8'))

    def test_action_detail_only_shows_panel_stats_used_by_selected_action(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('function actionPanelParticipation(action, detail, slot) {', source)
        self.assertIn('const usesAttack = Number(multipliers.atk || 0) !== 0;', source)
        self.assertIn('const usesHp = Number(multipliers.hp || 0) !== 0;', source)
        self.assertIn('const usesDefense = Number(multipliers.def || 0) !== 0;', source)
        self.assertIn('Number(triggeredReaction.contributor_slot) === Number(slot)', source)
        self.assertIn('action?.damage_uses_stagger_strength === true', source)
        self.assertIn('detail?.formula_parts?.uses_stagger_strength === true', source)
        self.assertNotIn('Number(action?.stagger || 0) !== 0 || Number(detail?.stagger_amount || 0) !== 0', source)
        self.assertIn('if (panelParticipation.atk) {', source)
        self.assertIn('if (panelParticipation.hp) {', source)
        self.assertIn('if (panelParticipation.def) {', source)
        self.assertIn('if (panelParticipation.harmony_strength) {', source)
        self.assertIn('if (panelParticipation.stagger_strength) {', source)
        self.assertIn('if (panelParticipation.damage) {', source)
        self.assertIn("${realtimePanelRows.join('') || '<div class=\"shaft-empty\">该动作不读取角色面板属性</div>'}", source)

    def test_action_detail_displays_resolved_action_multiplier(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('const realtimeAtk = Number(panelStats.atk || 0);', source)
        self.assertIn('const actionBase = Number(detail?.formula_parts?.base || 0) * actionMultiplier;', source)
        self.assertIn('? actionBase / realtimeAtk', source)
        self.assertNotIn('? Number(detail?.direct_damage || 0) / realtimeAtk', source)
        self.assertIn('`${formatNumber(finalActionAtkMultiplier * 100, 1)}%攻击`', source)
        self.assertIn('<span>动作倍率</span><strong>${escapeHtml(actionMultiplierText)}</strong>', source)

    def test_action_detail_hides_requiem_nightmare_from_applied_buffs(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn("'character_requiem_nightmare'", source)
        self.assertIn("'character_requiem_nightmare_stack'", source)
        self.assertIn(
            "!DETAIL_HIDDEN_APPLIED_BUFF_IDS.has(String(buff?.definition_id || ''))",
            source,
        )

    def test_action_detail_merges_same_buff_layers_and_labels_dodge_counter_damage(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn("dodge_counter_dmg: '闪反增伤'", source)
        self.assertNotIn("dodge_counter_dmg: '闪避反击伤害'", source)
        self.assertIn("if (trigger.event === 'full_stack') return '叠满层时';", source)
        self.assertIn('const mergedAppliedBuffs = Array.from(appliedBuffList.reduce((groups, buff) => {', source)
        self.assertIn("const key = String(buff?.definition_id || buff?.rule_id || buff?.name || '');", source)
        self.assertIn('current.stack_count = Number(current.stack_count || 0) + Number(buff?.stack_count || 0);', source)
        self.assertIn('current.effects[effectKey] = Number(current.effects[effectKey] || 0) + Number(value || 0);', source)
        self.assertIn("const stackSuffix = stackCount > 1 ? ` · ${formatNumber(stackCount, 0)}层` : '';", source)
        self.assertIn('if (Number(nightmareStacks) > 0) {', source)
        self.assertIn(
            "!DETAIL_HIDDEN_APPLIED_BUFF_IDS.has(String(buff?.rule_id || ''))",
            source,
        )

    def test_canhong_is_only_available_to_shaft_test_accounts_and_yiloyi_is_public(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        public_catalog = get_shaft_catalog_payload()
        test_catalog = get_shaft_catalog_payload(player=SimpleNamespace(shaft_test_whitelisted=True))
        public_canhong = next(character for character in public_catalog['characters'] if character['name'] == '残红')
        test_canhong = next(character for character in test_catalog['characters'] if character['name'] == '残红')
        public_yiloyi = next(character for character in public_catalog['characters'] if character['name'] == '伊洛伊')

        self.assertTrue(public_canhong['selection_disabled'])
        self.assertFalse(test_canhong['selection_disabled'])
        self.assertFalse(public_yiloyi['selection_disabled'])
        self.assertIn("record.selection_disabled ? 'disabled' : ''", source)
        self.assertIn(".filter((character) => !character.selection_disabled)", source)
        self.assertIn('const disabledCharacterIds = new Set(', source)
        self.assertIn('state.axis.steps = state.axis.steps.filter((step) => !restrictedSlots.has(Number(step.slot)));', source)

    def test_canhong_uses_tencent_document_panel_and_actions(self) -> None:
        catalog = get_shaft_catalog_payload(player=SimpleNamespace(shaft_test_whitelisted=True))
        characters = {character['name']: character for character in catalog['characters']}
        canhong = characters['残红']
        protagonist = characters['主角']

        self.assertTrue(canhong['test_character'])
        self.assertNotIn('bond_bonus', canhong)
        self.assertNotIn('bond_bonus', protagonist)
        self.assertEqual(canhong['element'], '咒')
        self.assertEqual(canhong['adaptation'], '气态')
        self.assertEqual(canhong['base_stats'], {'atk': 660.0, 'hp': 15513.0, 'def': 909.0})
        actions = {action['name']: action for action in catalog['actions_by_character'][canhong['id']]}
        self.assertIn('a1', actions)
        self.assertIn('e', actions)
        self.assertIn('强化e', actions)
        self.assertIn('幻e', actions)
        self.assertNotIn('离魂错', actions)
        self.assertIn('血宴', actions)
        self.assertIn('z', actions)
        self.assertIn('幻z', actions)
        self.assertNotIn('绰影', actions)
        self.assertNotIn('落月', actions)
        self.assertIn('下落', actions)
        self.assertIn('闪反', actions)
        self.assertIn('援护', actions)
        self.assertNotIn('折楼', actions)
        self.assertNotIn('折楼（最高）', actions)
        self.assertNotIn('虚徊', actions)
        self.assertEqual(actions['幻e']['multipliers']['atk'], 4.498)
        self.assertEqual(actions['幻e']['energy_gain'], 24.999)
        self.assertEqual(actions['幻e']['harmony'], 18.901)
        self.assertEqual(actions['幻e']['stagger'], 2.299)
        self.assertEqual(actions['血宴']['multipliers']['atk'], 6.999)
        self.assertEqual(actions['血宴']['energy_cost'], 120)
        self.assertEqual(actions['e']['duration_ticks'], 10)
        self.assertEqual(actions['强化e']['duration_ticks'], 10)
        self.assertEqual(actions['幻e']['duration_ticks'], 22)
        self.assertEqual(
            {actions[name]['cooldown_group'] for name in ('e', '强化e', '幻e')},
            {'canhong_e'},
        )
        self.assertEqual(actions['幻a1']['duration_ticks'], 3)
        self.assertEqual(actions['幻a4']['duration_ticks'], 11)
        self.assertEqual(actions['援护']['duration_ticks'], 15)
        self.assertEqual(actions['援护']['multipliers']['atk'], 2.0)
        self.assertEqual(actions['援护']['energy_gain'], 5.2)
        self.assertEqual(actions['援护']['harmony'], 0)
        self.assertEqual(actions['援护']['stagger'], 2.5)
        self.assertEqual(actions['援护']['hit_count'], 1)
        self.assertFalse(actions['援护']['is_background_damage'])
        self.assertIn('SkillDamageData_Zankou_Radio.xlsx', actions['援护']['source_note'])
        self.assertTrue(all(
            actions[name]['can_background_override']
            for name in ('幻a1', '幻a2', '幻a3', '幻a4')
        ))
        self.assertEqual(actions['焚天']['duration_ticks'], 0)
        self.assertEqual(actions['强化焚天']['duration_ticks'], 0)
        self.assertEqual(actions['血宴']['duration_ticks'], 0)
        self.assertTrue(all(
            action['duration_ticks'] > 0
            for action in actions.values()
            if action['action_type'] not in {'无', 'Q'}
        ))
        self.assertTrue(all(
            action.get('is_background_damage') is not True
            for action in actions.values()
        ))
        buff_ids = {buff['id'] for buff in catalog['buffs']}
        self.assertIn('character_canhong_corrosion_heart', buff_ids)
        self.assertIn('character_canhong_poison_fire_five', buff_ids)
        self.assertIn('character_canhong_harmony_strength', buff_ids)
        self.assertIn('character_canhong_a5_team_stagger_damage', buff_ids)
        buffs = {buff['id']: buff for buff in catalog['buffs']}
        self.assertEqual(buffs['character_canhong_delusion']['target']['tags'], ['DOT'])
        self.assertEqual(buffs['character_canhong_delusion']['duration']['ticks'], 160)
        self.assertEqual(buffs['character_canhong_delusion_exit_delay']['duration']['ticks'], 80)
        self.assertEqual(buffs['character_canhong_a1_delusion']['target']['tags'], ['DOT'])
        self.assertEqual(buffs['character_canhong_hunt']['target'], {'scope': 'registrar'})
        self.assertEqual(buffs['character_canhong_hunt']['effects']['all_dmg'], 0.25)
        self.assertEqual(buffs['character_canhong_a1_hunt']['effects']['all_dmg'], 0.4)
        self.assertEqual(buffs['character_canhong_b_blaze_stack']['stacking']['max_stacks'], 1)
        self.assertEqual(buffs['character_canhong_b_blaze_fentian_damage']['effects']['blaze_final_dmg'], 1.5)
        self.assertEqual(buffs['character_canhong_six_node_resonance_attack']['trigger']['event'], 'action_hit')
        self.assertTrue(buffs['character_canhong_six_node_resonance_attack']['trigger']['periodic_damage'])
        self.assertEqual(buffs['character_canhong_hunt_illusion_delay']['duration']['ticks'], 80)
        self.assertEqual(buffs['character_canhong_hunt_reality_restore']['duration']['type'], 'permanent')
        self.assertEqual(buffs['character_canhong_hunt_natural_restore']['duration']['delay_ticks'], 80)
        self.assertEqual(
            buffs['character_canhong_hunt_natural_restore']['calculation']['team_unique_key'],
            'character_canhong_hunt_effect',
        )
        self.assertTrue(
            buffs['character_canhong_illusion_dot_spread']['activation']['increase_all_active_dot_layers']
        )
        canhong_arc = next(arc for arc in catalog['arcs'] if arc['name'] == '噬心诡刃')
        self.assertEqual(catalog['arcs'][0]['id'], canhong_arc['id'])
        self.assertEqual(canhong_arc['adaptation'], '气态')
        self.assertEqual(canhong_arc['base_atk'], 570)
        self.assertEqual(catalog['arc_refinements']['arcs'][canhong_arc['id']]['levels']['5']['panel_modifiers']['crit_rate'], 0.56)
        source = SHAFT_JS.read_text(encoding='utf-8')
        self.assertIn('function characterHasBondBonus(characterId)', source)
        self.assertIn("const bondLabel = hasBondBonus ? character.bond_bonus.label : '无羁绊加成';", source)
        self.assertIn("${hasBondBonus ? '' : 'disabled'}", source)

    def test_yiloyi_bond_bonus_is_five_percent_attack(self) -> None:
        catalog = get_shaft_catalog_payload(player=SimpleNamespace(shaft_test_whitelisted=True))
        yiloyi = next(character for character in catalog['characters'] if character['name'] == '伊洛伊')

        self.assertEqual(yiloyi['bond_bonus']['label'], '5攻击')
        self.assertEqual(yiloyi['bond_bonus']['modifiers']['atk_pct'], 0.05)
        self.assertEqual(yiloyi['bond_bonus']['modifiers']['crit_rate'], 0.0)

    def test_xun_hides_energy_controls_and_labels(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        catalog = get_shaft_catalog_payload()
        xun = next(character for character in catalog['characters'] if character['name'] == '浔')

        self.assertFalse(xun['uses_energy'])
        self.assertFalse(xun['uses_cooldowns'])
        self.assertIn("const usesEnergy = character.uses_energy !== false;", source)
        self.assertIn("const showsEnergy = character.uses_energy !== false;", source)
        self.assertIn("resources.usesEnergy ? `<label>", source)
        self.assertIn("showsEnergy ? `<small>能量 ${energyLabel}</small>` : ''", source)
        self.assertIn("showsEnergy ? `<div class=\"shaft-detail-kv\"><span>回能</span>", source)
        self.assertIn("showsEnergy ? `<div class=\"shaft-detail-kv\"><span>额外回能</span>", source)

    def test_action_detail_uses_start_end_times_and_omits_dps(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('<span>开始时间</span>', source)
        self.assertIn('<span>结束时间</span>', source)
        self.assertNotIn('<span>显示开始</span>', source)
        self.assertNotIn('<span>计算开始</span>', source)
        self.assertNotIn('<span>计算结束</span>', source)
        self.assertNotIn('<div class="shaft-detail-kv"><span>DPS</span>', source)

    def test_build_page_uses_character_loadout_overview_three_column_order(self) -> None:
        template = (ROOT / 'app' / 'modules' / 'shaft' / 'templates' / 'shaft' / 'index.html').read_text(encoding='utf-8')
        css = (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css').read_text(encoding='utf-8')

        summary = template.index('class="glass-card shaft-panel shaft-build-summary"')
        loadout = template.index('class="glass-card shaft-panel shaft-team-panel"')
        overview = template.index('class="glass-card shaft-panel shaft-build-overview"')
        self.assertLess(summary, loadout)
        self.assertLess(loadout, overview)
        self.assertIn('<h2>角色面板总结</h2>', template)
        self.assertIn('<h2>全队总览</h2>', template)
        self.assertEqual(template.count('id="shaft-build-panel"'), 1)
        self.assertIn(
            'grid-template-columns: minmax(220px, 250px) minmax(700px, 1fr) minmax(270px, 310px);',
            css,
        )
        self.assertIn('@media (max-width: 1239px)', css)
        self.assertLess(template.index('id="shaft-build-panel"'), template.index('<h2>敌人</h2>'))
        self.assertLess(template.index('<h2>敌人</h2>'), loadout)
        self.assertNotIn('<h2>全队面板</h2>', template)
        self.assertIn('<h2>家具上限</h2>', template)

    def test_character_panel_distinguishes_general_and_element_damage(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        formula_constants = (
            ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'data' / 'formula_constants.json'
        ).read_text(encoding='utf-8')

        self.assertIn("{ key: 'all_dmg', label: '通伤', percent: true }", source)
        self.assertIn("['属伤', formatPercent(panelStats.element_dmg || 0)]", source)
        self.assertIn("['通伤', formatPercent(panelStats.all_dmg || 0)]", source)
        self.assertIn('"label": "通伤"', formula_constants)

        labels = ['暴击', '暴伤', '环合强度', '倾陷强度', '通伤', '属伤', '充能']
        positions = [source.index(f"['{label}',", source.index('const minorStats = [')) for label in labels]
        self.assertEqual(positions, sorted(positions))

    def test_team_furniture_bonus_displays_actual_caps(self) -> None:
        defaults = _frontend_team_panel_bonus_defaults()

        self.assertEqual(defaults['furniture_crit_dmg'], 0.04)
        self.assertEqual(defaults['furniture_flat_atk'], 20)
        self.assertEqual(defaults['furniture_flat_def'], 30)

        source = SHAFT_JS.read_text(encoding='utf-8')
        self.assertIn("kind: 'percent', max: 4, step: 0.1", source)
        self.assertIn("kind: 'flat', max: 20, step: 1", source)
        self.assertIn("kind: 'flat', max: 30, step: 1", source)
        self.assertIn('data-team-bonus="${escapeHtml(field.key)}"', source)

    def test_rotation_columns_share_axis_workspace_height_and_self_check_stays_conditional(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        css = (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css').read_text(encoding='utf-8')
        template = (ROOT / 'app' / 'modules' / 'shaft' / 'templates' / 'shaft' / 'index.html').read_text(encoding='utf-8')

        self.assertIn('--shaft-rotation-panel-height: min(720px, calc(100vh - 28px));', css)
        self.assertEqual(css.count('height: var(--shaft-rotation-panel-height);'), 3)
        self.assertIn('id="shaft-self-check" aria-live="polite" hidden', template)
        self.assertIn('node.hidden = warnings.length === 0;', source)
        self.assertIn('<strong>自检提示</strong>', source)
        self.assertNotIn('<span class="shaft-detail-muted">校验</span>', source)
        self.assertIn('.shaft-self-check[hidden]', css)
        self.assertIn('background: rgba(54, 8, 15, 0.96);', css)

    def test_workbench_summary_shows_duration_and_optional_damage_sources(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        engine = (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'js' / 'shaft_engine.js').read_text(encoding='utf-8')

        self.assertIn('<div><span>总轴长</span><strong>${formatNumber(summary.duration_seconds || 0, 1)}s</strong></div>', source)
        self.assertIn('Number(summary.direct_damage || 0) - Number(summary.harmony_damage || 0)', source)
        self.assertIn('<div><span>直伤</span><strong>${formatNumber(workbenchDirectDamage)}</strong></div>', source)
        self.assertIn('<div><span>环合</span><strong>${formatNumber(summary.harmony_damage || 0)}</strong></div>', source)
        self.assertIn('<div><span>倾陷</span><strong>${formatNumber(summary.stagger_damage || 0)}</strong></div>', source)
        self.assertIn("const HARMONY_DAMAGE_SOURCES = ['创生', '创生复制体', '覆纹', '浊燃', '黯星'];", engine)
        self.assertIn("const SPECIAL_DAMAGE_SOURCES = ['创生', '创生复制体', '覆纹', '浊燃', '黯星'];", engine)
        self.assertIn("{ source: '创生', members: ['创生', '创生复制体'] }", engine)
        self.assertIn("{ source: '覆纹', members: ['覆纹'] }", engine)
        self.assertNotIn("const SPECIAL_DAMAGE_SOURCES = ['创生', '浊燃', '黯星', '噩梦'];", engine)
        self.assertIn("return SPECIAL_DAMAGE_SOURCES.includes(explicitSource) ? explicitSource : '';", engine)
        self.assertNotIn("actionName.includes(source)", engine)
        self.assertIn("{ source: '倾陷', damage: staggerDamage }", engine)
        self.assertIn('const sourceBars = (result?.damage_by_source || []).map', source)
        self.assertIn('<span>覆纹伤害</span><strong>${formatNumber(detail.fuwen_damage)}</strong>', source)

        catalog = load_shaft_catalog()
        dark_star_action = next(action for action in catalog['actions'] if action.get('name') == '黯星扣血')
        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': dark_star_action['character_id'],
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [{
                'id': 'dark_star_damage',
                'slot': 0,
                'action_id': dark_star_action['id'],
                'start_tick': 0,
            }],
        })['result']
        damage_by_source = {item['source']: item for item in result['damage_by_source']}

        self.assertGreater(damage_by_source['黯星']['damage'], 0)
        self.assertAlmostEqual(result['summary']['harmony_damage'], damage_by_source['黯星']['damage'])
        self.assertEqual(result['summary']['character_damage'], 0)
        self.assertEqual(result['damage_by_slot'][0]['damage'], 0)
        self.assertEqual(result['damage_by_action_by_slot'][0]['actions'], [])
        self.assertAlmostEqual(
            sum(item['damage'] for item in result['harmony_contributions_by_slot']),
            result['summary']['harmony_damage'],
        )
        self.assertEqual(result['harmony_contributions_by_slot'][0]['sources'][0]['source'], '黯星')
        self.assertGreater(damage_by_source['黯星']['percent'], 0)
        self.assertNotIn('创生', damage_by_source)
        self.assertNotIn('浊燃', damage_by_source)

        jiuyuan_extra = next(action for action in catalog['actions'] if action.get('name') == '创生追加')
        haiyue_extra = next(action for action in catalog['actions'] if action.get('name') == '黯星追加')
        attached_result = simulate_shaft_axis({
            'team': [
                {
                    'slot': 0,
                    'character_id': jiuyuan_extra['character_id'],
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': haiyue_extra['character_id'],
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {
                    'id': 'jiuyuan_extra',
                    'slot': 0,
                    'action_id': jiuyuan_extra['id'],
                    'start_tick': 0,
                },
                {
                    'id': 'haiyue_extra',
                    'slot': 1,
                    'action_id': haiyue_extra['id'],
                    'start_tick': 0,
                },
            ],
        })['result']
        self.assertGreater(attached_result['summary']['character_damage'], 0)
        self.assertEqual(attached_result['summary']['harmony_damage'], 0)
        self.assertTrue(all(
            contribution['damage'] == 0
            for contribution in attached_result['harmony_contributions_by_slot']
        ))

    def test_special_damage_sources_use_distinct_semantic_colors(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        css = (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css').read_text(encoding='utf-8')

        self.assertIn("'创生': '#4fdcc8'", source)
        self.assertIn("'浊燃': '#ff665c'", source)
        self.assertIn("'黯星': '#6f8fff'", source)
        self.assertIn("'倾陷': '#eef4ff'", source)
        self.assertNotIn("'创生': '#77e36f'", source)
        self.assertIn("'创生': DAMAGE_SOURCE_COLORS['创生']", source)
        self.assertIn("'浊燃': DAMAGE_SOURCE_COLORS['浊燃']", source)
        self.assertIn("'黯星': DAMAGE_SOURCE_COLORS['黯星']", source)
        contribution_shares = source[source.index('function damageContributionShares'):source.index('function renderResults()')]
        render_results = source[source.index('function renderResults()'):source.index('function actionContributionForSlot')]
        self.assertIn("label: '环合伤害'", contribution_shares)
        self.assertIn("label: '倾陷伤害'", contribution_shares)
        self.assertIn("color: DAMAGE_SOURCE_COLORS['创生']", contribution_shares)
        self.assertIn("color: DAMAGE_SOURCE_COLORS['倾陷']", contribution_shares)
        self.assertIn("const independentRows = contributionShares.filter", render_results)
        self.assertIn('shaft-contribution-source-row', render_results)
        self.assertIn('--contribution-color:${item.color}', render_results)
        self.assertIn('.shaft-contribution-source-row .shaft-contribution-bar span', css)

    def test_substat_contribution_analysis_is_local_normalized_and_non_mutating(self) -> None:
        template = SHAFT_TEMPLATE.read_text(encoding='utf-8')
        source = SHAFT_JS.read_text(encoding='utf-8')
        engine = SHAFT_ENGINE_JS.read_text(encoding='utf-8')
        css = SHAFT_CSS.read_text(encoding='utf-8')

        self.assertIn('id="shaft-substat-contribution-btn"', template)
        self.assertIn('id="shaft-substat-contribution-dialog"', template)
        self.assertIn('id="shaft-substat-contribution-content"', template)
        self.assertIn('>副词条贡献度</button>', template)
        self.assertIn('>副词条贡献度</h2>', template)
        self.assertNotIn('>计算词条贡献度</button>', template)
        self.assertIn('以8类主要副词条各 ${formatNumber(analysis.base_count || 0)} 条', source)
        self.assertIn('纯前端计算，不改配装或对比快照', source)
        self.assertNotIn('基准临时将每位角色的通伤', source)
        self.assertIn('.shaft-substat-contribution-table', css)
        self.assertIn('max-height: calc(100vh - 28px);', css)
        self.assertIn('overflow-y: auto;', css)
        self.assertIn('function analyzeSubstatContributions(', engine)
        self.assertIn("{ key: 'all_dmg', label: '通伤' }", engine)
        self.assertIn("{ key: 'def_pct', label: '防御%' }", engine)
        self.assertIn("'flat_atk'", engine)
        self.assertIn("'flat_hp'", engine)
        self.assertIn("'flat_def'", engine)
        self.assertIn('const BASELINE_SUBSTAT_FIELDS = SUBSTAT_CONTRIBUTION_FIELDS.slice(0, 8);', engine)
        self.assertIn('num(candidateMember.substat_counts[field.key]) + incrementCount', engine)
        self.assertIn('row.increase_damage / maxIncreaseDamage * 100', engine)
        self.assertIn('const contributionByCell = new Map', source)
        self.assertIn('const topContributionRanksBySlot = new Map', source)
        self.assertIn('.slice(0, 4);', source)
        self.assertIn('contribution-rank-${rank}', source)
        self.assertIn("contribution > 0 ? `${formatNumber(contribution, 1)}%` : '—'", source)
        for rank in range(1, 5):
            self.assertIn(f'td.contribution-rank-{rank}', css)

        open_analysis = source[
            source.index('async function openSubstatContributionAnalysis'):
            source.index('function closeSubstatContributionAnalysis')
        ]
        self.assertIn('window.ShaftEngine.analyzeSubstatContributions(state.axis, state.catalog', open_analysis)
        self.assertNotIn('shaftRequest(', open_analysis)
        self.assertNotIn('state.compareSnapshot =', open_analysis)

        node_bin = shutil.which('node')
        if not node_bin:
            self.skipTest('node is required for shaft engine regression tests')
        catalog = get_shaft_catalog_payload()
        action = next(
            item
            for item in catalog['actions']
            if item.get('action_type') == '普攻'
            and float((item.get('multipliers') or {}).get('atk') or 0) > 0
        )
        character = next(item for item in catalog['characters'] if item['id'] == action['character_id'])
        axis = {
            'team': [{
                'slot': 0,
                'character_id': character['id'],
                'character_name': character['name'],
                'arc_id': '',
                'cartridge_id': '',
                'substat_counts': {'all_dmg': 30, 'flat_atk': 30},
            }],
            'steps': [{
                'id': 'substat-analysis-action',
                'slot': 0,
                'action_id': action['id'],
                'start_tick': 0,
            }],
            'enemy': {'level': 90, 'initial_resistance': 0.3},
            'initial_energy': 1000,
        }
        completed = subprocess.run(
            [node_bin, str(ROOT / 'scripts' / 'shaft_compute.js')],
            input=json.dumps({
                'operation': 'analyze_substats',
                'axis': axis,
                'catalog': catalog,
                'options': {'base_count': 15, 'increment_count': 5},
            }),
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        analysis = json.loads(completed.stdout)['result']
        self.assertEqual(analysis['base_count'], 15)
        self.assertEqual(analysis['increment_count'], 5)
        self.assertGreater(analysis['baseline_total_damage'], 0)
        self.assertTrue(analysis['rows'])
        self.assertAlmostEqual(analysis['rows'][0]['contribution_percent'], 100.0)
        self.assertTrue(all(row['increase_damage'] > 0 for row in analysis['rows']))
        self.assertEqual(
            {field['key'] for field in analysis['fields']},
            {
                'all_dmg', 'crit_rate', 'crit_dmg', 'harmony_strength',
                'stagger_strength', 'atk_pct', 'hp_pct', 'def_pct',
                'flat_atk', 'flat_hp', 'flat_def',
            },
        )
        self.assertIn('flat_atk', {row['stat_key'] for row in analysis['rows']})

    def test_character_action_contribution_dialog_aggregates_repeated_actions(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        css = (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css').read_text(encoding='utf-8')
        template = (ROOT / 'app' / 'modules' / 'shaft' / 'templates' / 'shaft' / 'index.html').read_text(encoding='utf-8')
        catalog = load_shaft_catalog()
        character = catalog['characters'][0]
        actions = [
            action
            for action in catalog['actions']
            if action['character_id'] == character['id'] and float(action.get('multipliers', {}).get('atk') or 0) > 0
        ][:2]
        self.assertEqual(len(actions), 2)

        result = simulate_shaft_axis({
            'team': [{
                'slot': 0,
                'character_id': character['id'],
                'arc_id': '',
                'cartridge_id': '',
            }],
            'steps': [
                {'id': 'first', 'slot': 0, 'action_id': actions[0]['id'], 'start_tick': 0},
                {'id': 'repeat', 'slot': 0, 'action_id': actions[0]['id'], 'start_tick': 20},
                {'id': 'second', 'slot': 0, 'action_id': actions[1]['id'], 'start_tick': 40},
            ],
        })['result']
        contribution = result['damage_by_action_by_slot'][0]
        self.assertEqual(contribution['character_id'], character['id'])
        self.assertEqual(len(contribution['actions']), 2)
        self.assertTrue(all(action['action_type'] != '倾陷' for action in contribution['actions']))
        self.assertAlmostEqual(sum(action['percent'] for action in contribution['actions']), 100, places=6)
        self.assertAlmostEqual(
            sum(action['damage'] for action in contribution['actions']),
            contribution['total_damage'],
            places=6,
        )
        self.assertIn('data-action-contribution-slot', source)
        self.assertIn('function actionContributionPie(actions)', source)
        self.assertIn('id="shaft-action-contribution-dialog"', template)
        self.assertIn('data-analysis-dimension="type"', source)
        self.assertIn('data-analysis-view="bars"', source)
        self.assertIn('shaft-action-contribution-segment', source)
        self.assertIn("const additionalTags = new Set(['追击', '附着']);", source)
        self.assertIn("additionalTags.has(damageType)", source)
        self.assertIn("? (actionType || '其他')", source)
        self.assertIn('damage_type: detail.damage_type', SHAFT_ENGINE_JS.read_text(encoding='utf-8'))
        self.assertIn("if (tags.has('追击')) total += panelMods.follow_dmg;", SHAFT_ENGINE_JS.read_text(encoding='utf-8'))
        self.assertIn("if (tags.has('附着')) total += panelMods.attach_dmg;", SHAFT_ENGINE_JS.read_text(encoding='utf-8'))
        self.assertIn('damage_by_action_by_slot: clone(result.damage_by_action_by_slot || [])', source)
        self.assertIn('function mergeActionAnalysisComparison(', source)
        self.assertIn("'被动': '#ff5aa5'", source)
        self.assertIn('const reservedColors = new Set(', source)
        self.assertIn('const assignedColors = new Set();', source)
        self.assertIn('if (color && assignedColors.has(color)) color = \'\';', source)
        self.assertIn('color = fallbackColors.find((candidate) => !assignedColors.has(candidate));', source)
        self.assertIn('function actionAnalysisRelativePercent(current, baseline)', source)
        self.assertIn('function actionAnalysisRelativePercentText(current, baseline)', source)
        self.assertIn('shaft-action-analysis-relative', source)
        self.assertIn('actionAnalysisRelativePercentText(contribution.total_damage, baselineContribution.total_damage)', source)
        self.assertIn('actionAnalysisRelativePercentText(group.damage, group.baselineDamage)', source)
        self.assertIn('对比快照', source)
        self.assertIn('排行图中的竖线表示快照占比', source)
        contribution_legend = css[
            css.index('.shaft-action-contribution-legend {'):
            css.index('.shaft-action-contribution-item {')
        ]
        self.assertIn('padding: 2px 4px 2px 2px;', contribution_legend)
        self.assertIn('overflow-x: hidden;', contribution_legend)
        self.assertIn('overflow-y: auto;', contribution_legend)

    def test_substat_input_defers_simulation_until_editing_finishes(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        input_handler = source[
            source.index('function handleSubstatInput(event)'):
            source.index('function handleSubstatCommit(event)')
        ]
        commit_handler = source[
            source.index('function handleSubstatCommit(event)'):
            source.index('function handleSubstatKeydown(event)')
        ]

        self.assertIn("setStatus('编辑词条中，退出刷新', 'dirty');", input_handler)
        self.assertNotIn('renderTeam();', input_handler)
        self.assertNotIn('renderBuildPanel();', input_handler)
        self.assertNotIn('scheduleSimulation();', input_handler)
        self.assertIn('renderBuildPanel();', commit_handler)
        self.assertIn('scheduleSimulation();', commit_handler)
        self.assertIn("addEventListener('focusout', handleSubstatCommit)", source)
        self.assertIn("addEventListener('keydown', handleSubstatKeydown)", source)

    def test_substat_total_explains_drive_and_cartridge_conversion(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        css = SHAFT_CSS.read_text(encoding='utf-8')

        self.assertIn(
            "const SUBSTAT_TOTAL_TOOLTIP = 'II / III / IV 型驱动上的每个副词条分别对应 2 / 3 / 4 个词条，卡带上的副词条对应 10 个词条。只支持金色卡带、驱动块。';",
            source,
        )
        self.assertIn('class="shaft-substat-total-tooltip"', source)
        self.assertIn('tabindex="0"', source)
        self.assertIn('.shaft-substat-total-tooltip::after', css)
        self.assertIn('.shaft-substat-total-tooltip:hover::after', css)
        self.assertIn('.shaft-substat-total-tooltip:focus::after', css)

    def test_build_panel_avatar_scrolls_matching_loadout_into_view(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        css = SHAFT_CSS.read_text(encoding='utf-8')
        scroll_handler = source[
            source.index('function scrollBuildMemberIntoView(slot)'):
            source.index('function timelineVisualOffsetWithScale(')
        ]

        self.assertIn('.shaft-member-card[data-slot="${Number(slot)}"]', scroll_handler)
        self.assertIn("behavior: reduceMotion ? 'auto' : 'smooth'", scroll_handler)
        self.assertIn("block: 'start'", scroll_handler)
        self.assertIn("inline: 'nearest'", scroll_handler)
        self.assertIn('window.requestAnimationFrame(() => scrollBuildMemberIntoView(state.buildPanelSlot));', scroll_handler)
        self.assertIn('scroll-margin-top: 14px;', css)

    def test_build_page_has_separate_stagger_contribution_dialog(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        template = (ROOT / 'app' / 'modules' / 'shaft' / 'templates' / 'shaft' / 'index.html').read_text(encoding='utf-8')
        css = (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css').read_text(encoding='utf-8')

        self.assertNotIn('id="shaft-stagger-analysis-btn"', template)
        self.assertIn('id="shaft-stagger-analysis-dialog"', template)
        self.assertIn('id="shaft-stagger-analysis-content"', template)
        self.assertIn('data-open-stagger-analysis', source)
        self.assertIn('function renderStaggerAnalysis()', source)
        self.assertIn('不计入角色自身伤害占比或动作贡献', source)
        self.assertIn('result.stagger_contributions_by_slot', source)
        self.assertIn('平均倾陷增伤', source)
        self.assertIn('item.average_stagger_damage_bonus', source)
        self.assertIn('.shaft-stagger-contribution-list', css)

    def test_build_page_has_separate_harmony_contribution_dialog(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        template = (ROOT / 'app' / 'modules' / 'shaft' / 'templates' / 'shaft' / 'index.html').read_text(encoding='utf-8')

        self.assertIn('id="shaft-harmony-damage"', template)
        self.assertNotIn('id="shaft-harmony-analysis-btn"', template)
        self.assertIn('id="shaft-harmony-analysis-dialog"', template)
        self.assertIn('id="shaft-harmony-analysis-content"', template)
        self.assertIn('data-open-harmony-analysis', source)
        self.assertIn('class="shaft-contribution-row shaft-contribution-source-row"', source)
        self.assertIn('function renderHarmonyAnalysis()', source)
        self.assertIn('result.harmony_contributions_by_slot', source)
        self.assertIn('环合伤害独立计入全队总伤和 DPS', source)
        self.assertIn('不计入角色自身伤害占比或动作贡献', source)
        self.assertIn("renderResultCard('shaft-direct-damage', '角色伤害', summary.character_damage", source)

    def test_rotation_header_exposes_shortcut_help_dialog(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        template = (ROOT / 'app' / 'modules' / 'shaft' / 'templates' / 'shaft' / 'index.html').read_text(encoding='utf-8')

        self.assertIn('id="shaft-shortcut-help-btn"', template)
        self.assertIn('id="shaft-shortcut-dialog"', template)
        self.assertIn("shortcutHelpButton.hidden = state.page !== 'rotation';", source)
        self.assertIn("event.key === 'ArrowLeft' || event.key === 'ArrowRight'", source)
        self.assertIn("movementKey === 'w' || movementKey === 'a' || movementKey === 's' || movementKey === 'd'", source)
        self.assertIn("movementKey === 'q' || movementKey === 'e'", source)
        self.assertIn('时间光标前后移动 0.1 秒', template)
        self.assertIn('所选动作前后移动 0.1 秒', template)

    def test_build_header_exposes_version_and_feedback_dialog(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        template = SHAFT_TEMPLATE.read_text(encoding='utf-8')

        self.assertIn('id="shaft-build-info-btn"', template)
        self.assertIn('id="shaft-build-info-dialog"', template)
        self.assertIn('id="shaft-build-info-version"', template)
        self.assertIn('新角色在游戏上线后需要 2–3 周才能上线网站', template)
        self.assertIn('反馈问题请加群', template)
        self.assertIn('class="shaft-known-issues"', template)
        self.assertIn('多层独立计算CD的叠层buff', template)
        self.assertIn('首次触发的那层的剩余时间，可能与预期不符', template)
        self.assertIn('暂时不支持浔复现技能', template)
        self.assertIn("config['SHAFT_SUPPORT_GROUP']", template)
        self.assertIn("buildInfoButton.hidden = state.page !== 'build';", source)
        self.assertIn("$('shaft-build-info-version').textContent = versionLabel;", source)
        self.assertIn('function openBuildInfo()', source)

    def test_timeline_tooltips_render_above_sticky_character_labels(self) -> None:
        styles = SHAFT_CSS.read_text(encoding='utf-8')
        buff_hover = re.search(
            r'\.shaft-buff-trigger-line:hover,\s*'
            r'\.shaft-buff-trigger-line:focus-visible,\s*'
            r'\.shaft-buff-trigger-line\.selected\s*\{(?P<body>[^}]*)\}',
            styles,
        )
        damage_hover = re.search(
            r'\.shaft-reaction-damage-marker:hover,\s*'
            r'\.shaft-reaction-damage-marker:focus-visible\s*\{(?P<body>[^}]*)\}',
            styles,
        )

        self.assertIsNotNone(buff_hover)
        self.assertIn('z-index: 40;', buff_hover.group('body'))
        self.assertIsNotNone(damage_hover)
        self.assertIn('z-index: 40;', damage_hover.group('body'))

    def test_six_awakening_nodes_are_selected_independently(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        css = (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css').read_text(encoding='utf-8')

        self.assertIn('function normalizeAwakeningNodes(rawNodes, legacyAwakening = 0)', source)
        self.assertIn('const checked = activeAwakeningNodes.has(level)', source)
        self.assertIn('awakeningNodes.add(level);', source)
        self.assertIn('awakeningNodes.delete(level);', source)
        self.assertNotIn('member.awakening = control.checked ? level : level - 1;', source)
        self.assertIn('awakening_nodes: awakeningNodes', source)
        self.assertNotIn('aria-label="${escapeHtml(tooltip)}" tabindex="0"', source)
        self.assertNotIn('.shaft-awakening-dot[data-tooltip]:focus-visible::after', css)

    def test_build_rotation_and_plaza_avatar_badges_show_active_awakening_count(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('function activeAwakeningCount(member) {', source)
        self.assertIn('normalizeAwakeningNodes(member?.awakening_nodes, member?.awakening).length', source)
        self.assertEqual(source.count('const awakeningCount = activeAwakeningCount('), 3)
        self.assertEqual(
            source.count('<b aria-label="已激活 ${awakeningCount} 个觉醒">${awakeningCount}</b>'),
            2,
        )
        self.assertNotIn('<b>${Number(member.slot) + 1}</b>', source)
        self.assertNotIn('<b>${Number(item.slot) + 1}</b>', source)

    def test_build_cards_show_buff_and_implemented_talent_mechanisms(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        css = (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css').read_text(encoding='utf-8')

        self.assertIn('function activeMechanismGroups(member)', source)
        self.assertIn('(state.catalog?.buffs || []).forEach((rule) => {', source)
        self.assertIn('(state.catalog?.mechanisms || [])', source)
        self.assertIn("name: `天赋${mechanism.talent === 1 ? '一' : '二'}", source)
        self.assertIn('replaces_rule_ids', source)
        self.assertIn("base_multiplier_pct: '基础倍率'", source)
        self.assertIn("base_multiplier_pct: '基础倍率提升'", source)
        self.assertIn("const selectedProviders = {", source)
        self.assertIn("character: String(member?.character_id || '')", source)
        self.assertIn("arc: String(member?.arc_id || '')", source)
        self.assertIn("cartridge: String(member?.cartridge_id || '')", source)
        self.assertIn("if (type === 'awakening_min')", source)
        self.assertIn("if (type === 'awakening_max')", source)
        self.assertIn("if (type === 'awakening_count_min')", source)
        self.assertIn("return awakeningNodes.size >= requiredCount;", source)
        self.assertIn("if (type === 'owner_character_id')", source)
        self.assertNotIn('panel_modifiers || arc.modifiers', source[source.index('function activeMechanismGroups(member)'):source.index('function activeMechanismTooltipHtml(member, character)')])
        self.assertIn('function mechanismRuleSummary(rule)', source)
        self.assertIn("arcRefinementRecord(member)?.buff_effects", source)
        self.assertIn('mechanismRuleSummary(rule)', source)
        self.assertIn('class="shaft-mechanism-item"', source)
        self.assertIn('class="shaft-mechanism-tooltip"', source)
        self.assertIn('当前激活机制 · ${total}', source)
        self.assertIn('.shaft-mechanism-popover', css)
        identity_start = source.index('<section class="shaft-member-identity">')
        identity_end = source.index('</section>', identity_start)
        character_select_start = source.index('<div class="shaft-character-select shaft-build-character-picker"', identity_end)
        character_select_end = source.index('</div>', character_select_start)
        self.assertIn('${mechanismTooltip}', source[identity_start:identity_end])
        self.assertNotIn('${mechanismTooltip}', source[character_select_start:character_select_end])
        self.assertIn('.shaft-build-layout .shaft-member-identity .shaft-mechanism-tooltip', css)

    def test_catalog_exposes_implemented_talents_for_active_mechanism_tooltips(self) -> None:
        catalog = get_shaft_catalog_payload()
        mechanisms = {mechanism['id']: mechanism for mechanism in catalog['mechanisms']}

        self.assertIn('talent_1_nanali_genesis', mechanisms)
        self.assertIn('talent_2_nanali_fair_duel', mechanisms)
        self.assertIn('talent_1_iloy_mirror', mechanisms)
        self.assertIn('talent_2_iloy_sympathetic_nervous_system', mechanisms)
        descriptions = [mechanism['description'] for mechanism in mechanisms.values()]
        self.assertFalse(any('已实装' in description for description in descriptions))
        self.assertFalse(any('部分实装' in description for description in descriptions))
        self.assertFalse(any('尚未实装' in description for description in descriptions))
        self.assertIn('每2秒最多触发1次', mechanisms['talent_2_nanali_fair_duel']['description'])

    def test_build_character_picker_reuses_market_cards_as_single_select(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        css = (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css').read_text(encoding='utf-8')

        self.assertNotIn('data-field="character_id"', source)
        self.assertIn('data-build-character-trigger', source)
        self.assertIn('data-build-character-popover', source)
        self.assertIn('data-build-character-id="${escapeHtml(candidate.id)}"', source)
        self.assertIn('class="shaft-character-filter-option ${candidate.id === member.character_id', source)
        self.assertIn('function applyMemberCharacterSelection(member, characterId)', source)
        self.assertIn('setBuildCharacterPickerOpen(picker, Boolean(popover?.hidden));', source)
        self.assertIn("$('shaft-team-slots').addEventListener('click', handleTeamClick);", source)
        self.assertIn('.shaft-build-character-popover', css)

    def test_character_switch_protects_unsaved_axis_before_mutating_or_clearing_steps(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        selection_start = source.index('function applyMemberCharacterSelection(member, characterId)')
        selection_end = source.index('function setBuildCharacterPickerOpen', selection_start)
        selection_body = source[selection_start:selection_end]
        confirmation = (
            "window.confirm('当前动作轴有未保存的修改，"
            "切换角色会清空该角色的动作，确定继续吗？')"
        )

        self.assertIn('hasUnsavedAxisChanges()', selection_body)
        self.assertIn(confirmation, selection_body)
        self.assertLess(selection_body.index(confirmation), selection_body.index('rememberMemberBuild(member);'))
        self.assertLess(selection_body.index(confirmation), selection_body.index('member.character_id = characterId;'))
        self.assertLess(selection_body.index(confirmation), selection_body.index('removeInvalidStepsForSlot(member.slot);'))
        self.assertIn('!state.characterSwitchUnsavedConfirmed', selection_body)
        self.assertIn('state.characterSwitchUnsavedConfirmed = true;', selection_body)
        self.assertIn('state.characterSwitchUnsavedConfirmed = false;', source)

    def test_history_paste_and_delete_reveal_the_affected_timeline_position(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        restore_start = source.index('function restoreEditorSnapshot(snapshot)')
        restore_end = source.index('function pushUndoSnapshot()', restore_start)
        restore_body = source[restore_start:restore_end]
        self.assertIn('revealTimelineTick(state.cursorTick);', restore_body)

        remove_start = source.index('function removeSteps(stepIds)')
        remove_end = source.index('function removeSelectedSteps()', remove_start)
        remove_body = source[remove_start:remove_end]
        self.assertIn('const removedAtTick =', remove_body)
        self.assertIn('Math.abs(Number(left.start_tick || 0) - removedAtTick)', remove_body)
        self.assertIn('revealTimelineTick(removedAtTick);', remove_body)

        paste_start = source.index('function pasteStepsAtCursor()')
        paste_end = source.index('function addBuffRule()', paste_start)
        paste_body = source[paste_start:paste_end]
        self.assertIn('revealTimelineTick(state.cursorTick);', paste_body)
        self.assertIn('const clipboardSpanTicks = Math.max(1,', paste_body)
        self.assertIn('step.start_tick = Number(step.start_tick || 0) + clipboardSpanTicks;', paste_body)
        self.assertIn('normalizeEditedSteps(new Set(newIds));', paste_body)

        self.assertIn('function compactSpaceReleasedByDeletion(', source)
        self.assertIn('function compactReleasedTimelineIntervals(intervals) {', source)
        self.assertIn('beforeDeleteResult?.time_axis?.frozen_intervals', remove_body)
        self.assertIn('compactSpaceReleasedByDeletion(', remove_body)
        self.assertIn('oldEnd > newEnd', source)
        self.assertIn('originalTick - releasedBeforeStep', source)
        self.assertIn('compactReleasedTimelineIntervals(releasedIntervals);', source)

    def test_modifier_backspace_removes_the_empty_interval_before_current_action(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        template = SHAFT_TEMPLATE.read_text(encoding='utf-8')

        shortcut_start = source.index('function removeGapBeforeCurrentStep()')
        shortcut_end = source.index('function pasteStepsAtCursor()', shortcut_start)
        shortcut_body = source[shortcut_start:shortcut_end]
        keydown = source[source.index('function handleKeydown(event) {'):source.index('function handleClipboardCopy(event) {')]

        self.assertIn('state.selectedStepId', shortcut_body)
        self.assertIn('detail?.display_visual_end_tick', shortcut_body)
        self.assertIn('start < currentStart && end > currentStart', shortcut_body)
        self.assertIn('compactReleasedTimelineIntervals([{ start: gapStart, end: currentStart }]);', shortcut_body)
        self.assertIn('normalizeEditedSteps(new Set([currentStep.id]));', shortcut_body)
        self.assertIn("(event.ctrlKey || event.metaKey) && !event.altKey && event.key === 'Backspace'", keydown)
        self.assertIn('removeGapBeforeCurrentStep();', keydown)
        self.assertIn('<kbd>Ctrl / ⌘</kbd><span>+</span><kbd>Backspace</kbd>', template)

    def test_batch_drag_preserves_each_selected_action_placement(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        drag_start = source.index('function updateTimelineDrag(event)')
        drag_end = source.index('function updateTimelineInteraction', drag_start)
        drag_body = source[drag_start:drag_end]
        self.assertIn('const canChangePlacement = dragStepIds.size === 1;', drag_body)
        self.assertIn('const placementChanged = canChangePlacement && dragPlacementWouldChange', drag_body)
        self.assertIn('if (canChangePlacement) {', drag_body)
        self.assertIn('applyDraggedPlacement(item, drag, deltaY);', drag_body)

    def test_native_background_action_multiplier_uses_confirmation_dialog_and_is_visible_on_bar(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        engine = SHAFT_ENGINE_JS.read_text(encoding='utf-8')
        template = (ROOT / 'app' / 'modules' / 'shaft' / 'templates' / 'shaft' / 'index.html').read_text(encoding='utf-8')
        css = (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css').read_text(encoding='utf-8')

        self.assertIn('const MAX_BACKGROUND_ACTION_MULTIPLIER = 999;', source)
        self.assertIn('const MAX_BACKGROUND_ACTION_MULTIPLIER = 999;', engine)
        self.assertIn('function backgroundActionMultiplier(step, action = actionForStep(step)) {', source)
        self.assertIn('${isBackgroundAction(action) ? `', source)
        self.assertIn('data-open-background-multiplier', source)
        self.assertIn('aria-haspopup="dialog">设置</button>', source)
        self.assertNotIn('aria-haspopup="dialog">×${actionMultiplier} · 设置</button>', source)
        self.assertNotIn('data-background-action-multiplier', source)
        self.assertIn('function openBackgroundActionMultiplier(stepId, trigger = null) {', source)
        self.assertIn('function confirmBackgroundActionMultiplier() {', source)
        self.assertIn('pushUndoSnapshot();', source)
        self.assertIn("step.repeat = nextMultiplier;", source)
        self.assertIn("addEventListener('click', handleStepDetailClick);", source)
        self.assertNotIn("addEventListener('input', handleStepDetailChange);", source)
        self.assertIn('id="shaft-background-multiplier-dialog"', template)
        self.assertIn('id="shaft-background-multiplier-input"', template)
        self.assertIn('min="1" max="999"', template)
        self.assertIn('id="shaft-background-multiplier-confirm"', template)
        self.assertIn('.shaft-background-multiplier-dialog {', css)
        self.assertIn('.shaft-background-multiplier-modal {', css)
        self.assertIn('.shaft-detail-multiplier button {\n    width: 100%;\n    min-width: 0;', css)
        self.assertNotIn('.shaft-detail-multiplier button {\n    width: 92px;', css)
        self.assertNotIn('Math.min(12, Math.round(Number(step?.repeat || 1)))', source)
        self.assertNotIn('Math.min(12, int(step?.repeat, 1))', engine)
        self.assertIn("actionMultiplier > 1 ? `×${actionMultiplier}` : ''", source)
        self.assertIn('step.repeat = backgroundActionMultiplier(step, action);', source)

    def test_native_background_actions_display_as_zero_seconds_with_five_tick_footprint(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        engine = SHAFT_ENGINE_JS.read_text(encoding='utf-8')

        self.assertIn('function isInstantNativeBackgroundAction(step, action = actionForStep(step)) {', source)
        self.assertIn('function actionCalculationDurationTicks(action, step = null) {', source)
        self.assertIn('ticksToSeconds(actionCalculationDurationTicks(action))', source)
        self.assertIn('function isInstantNativeBackgroundAction(step, action) {', engine)
        self.assertIn('return isBackgroundAction(action) && !isSupportAction(action) && !Boolean(action?.pre_input_node);', engine)
        self.assertIn('calculation_at_start_only: calculationAtStartOnly,', engine)
        self.assertIn('if (scheduled.calculation_at_start_only) {', engine)

    def test_buff_lines_are_single_state_line_with_multiline_tooltip(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        css = (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css').read_text(encoding='utf-8')

        self.assertIn('function mergedBuffLineSegments', source)
        self.assertIn('const buffSegments = mergedBuffLineSegments', source)
        self.assertNotIn('mergedBuffLineRows', source)
        self.assertNotIn('const maxBuffLineCount = buffRows.length', source)
        self.assertIn("tooltipLines.join('\\n')", source)
        tooltip_match = re.search(r'\.shaft-buff-trigger-line::after\s*{(?P<body>.*?)\n}', css, re.S)
        if not tooltip_match:
            raise AssertionError('buff line tooltip style is missing.')
        self.assertIn('white-space: pre-line;', tooltip_match.group('body'))

    def test_reaction_lines_use_contributor_row_and_damage_markers_do_not_change_track_height(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        css = (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css').read_text(encoding='utf-8')

        self.assertIn('Number(effect.trigger_slot) === Number(member.slot)', source)
        self.assertIn('Number(event.contributor_slot) === Number(member.slot)', source)
        self.assertIn('function reactionBuffLineSegments(effect, axisEndTick, loopEnabled) {', source)
        self.assertIn("if (!loopEnabled || effect?.loop_primed || endTick <= Number(axisEndTick || 0)) {", source)
        self.assertIn('return segments.filter((segment) => segment.startTick > 0);', source)
        self.assertIn('reactionBuffLineSegments(effect, buffAxisEndTick, loopEnabled).forEach((segment) => {', source)
        self.assertIn('const timelineDamageEvents = [...reactionDamageEvents, ...periodicDamageEvents];', source)
        self.assertIn('function groupedTimelineDamageEvents(events) {', source)
        self.assertIn('const groupedDamageEvents = groupedTimelineDamageEvents(timelineDamageEvents);', source)
        self.assertIn('const reactionDamageMarkers = groupedDamageEvents', source)
        self.assertIn("return event.events.map((item) => damageMarkerTooltip(item)).join('\\n');", source)
        self.assertIn('class="shaft-reaction-damage-marker ${isPeriodicDamage ?', source)
        self.assertIn('const opensUpward = isPeriodicDamage && Number(member.slot) === 3;', source)
        self.assertIn("${opensUpward ? 'shaft-damage-tooltip-above' : ''}", source)
        self.assertIn('function damageMarkerTooltip(event) {', source)
        self.assertIn("addZone('基础区', isPeriodicDamage ? formula.unscaled_base : formula.base);", source)
        self.assertNotIn("['噩梦', '蚀心'].includes", source)
        self.assertNotIn('攻击力（基础区）', source)
        self.assertIn("addZone('频率乘区', formula.frequency_multiplier);", source)
        self.assertNotIn("addZone('九原频率乘区', formula.frequency_multiplier);", source)
        self.assertNotIn('const frequencyText = Number(effect.frequency_multiplier || 1) > 1', source)
        self.assertIn('tooltip: `${effect.reaction} · ${ticksToSeconds(effect.duration_ticks || 1)}s`', source)
        self.assertNotIn("addZone('技能等级', formula.skill_multiplier);", source)
        self.assertIn("addZone('最终倍率区', formula.final_multiplier);", source)
        self.assertEqual(source.count("addZone('最终倍率区', formula.final_multiplier);"), 2)
        self.assertIn("if (String(event?.reaction || '') !== '黯星') addZone('防御', formula.defense);", source)
        self.assertIn("addZone('双暴区', formula.critical);", source)
        self.assertEqual(source.count("addZone('双暴区', formula.critical);"), 2)
        self.assertIn("if (String(event?.reaction || '') === '浊燃') addZone('双暴区', formula.critical);", source)
        self.assertIn('data-tooltip="${escapeHtml(tooltip)}"', source)
        self.assertIn('tabindex="0"', source)
        self.assertIn('function positionDamageMarkerTooltip(event) {', source)
        self.assertIn("marker.style.setProperty('--damage-tooltip-shift-x'", source)
        self.assertIn("$('shaft-timeline').addEventListener('focusin', positionDamageMarkerTooltip);", source)
        self.assertIn('style="left:${leftPx(Number(event.visual_tick ?? event.tick ?? 0))}px;', source)
        self.assertNotIn('`触发者 ${effect.trigger_character_name || memberName(effect.trigger_slot)}`', source)
        self.assertNotIn('`援护角色 ${effect.support_character_name || memberName(effect.support_slot)}`', source)
        marker_match = re.search(r'\.shaft-reaction-damage-marker\s*{(?P<body>.*?)\n}', css, re.S)
        if not marker_match:
            raise AssertionError('reaction damage marker style is missing.')
        marker_css = marker_match.group('body')
        self.assertIn('position: absolute;', marker_css)
        self.assertIn('width: 0;', marker_css)
        self.assertIn('height: 0;', marker_css)
        self.assertIn('border-top: 5px solid', marker_css)
        track_label_blocks = re.findall(r'\.shaft-track-label\s*{(?P<body>.*?)\n}', css, re.S)
        track_label_css = next(
            (body for body in track_label_blocks if 'position: sticky;' in body),
            '',
        )
        if not track_label_css:
            raise AssertionError('sticky track label style is missing.')
        self.assertIn('position: sticky;', track_label_css)
        self.assertIn('z-index: 24;', track_label_css)
        self.assertIn('background: rgba(13, 20, 31, 0.98);', track_label_css)
        marker_tooltip_match = re.search(r'\.shaft-reaction-damage-marker::after\s*{(?P<body>.*?)\n}', css, re.S)
        if not marker_tooltip_match:
            raise AssertionError('reaction damage marker tooltip style is missing.')
        self.assertIn('white-space: pre-line;', marker_tooltip_match.group('body'))
        self.assertIn('--damage-tooltip-shift-x', marker_tooltip_match.group('body'))
        self.assertIn('.shaft-reaction-damage-marker:focus-visible {\n    z-index: 40;', css)
        periodic_marker_match = re.search(r'\.shaft-periodic-damage-marker\s*{(?P<body>.*?)\n}', css, re.S)
        if not periodic_marker_match:
            raise AssertionError('periodic damage marker style is missing.')
        periodic_marker_css = periodic_marker_match.group('body')
        self.assertIn('border-top: 0;', periodic_marker_css)
        self.assertIn('border-bottom: 5px solid', periodic_marker_css)
        self.assertIn('filter: none;', periodic_marker_css)
        self.assertIn('.shaft-reaction-damage-marker.shaft-damage-tooltip-above::after {', css)
        self.assertIn('bottom: 9px;', css)
        self.assertIn('.shaft-reaction-damage-marker.shaft-damage-tooltip-above:hover::after,', css)
        self.assertIn("isPeriodicDamage ? trackHeight - 5 : buffLineTop + 9", source)
        self.assertIn('const warnings = Array.from(new Set([...axisWarnings, ...simulationWarnings]));', source)
        self.assertIn('inspectAxis(state.axis, state.catalog, resultDetails)', source)

    def test_buff_line_uses_only_the_latest_stack_snapshot(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('function latestBuffStates', source)
        self.assertIn('const current = latestById.get(buff.id);', source)
        self.assertIn('const startsLater = Number(buff.startTick) > Number(current?.startTick ?? -1);', source)
        self.assertIn('const activeBuffs = latestBuffStates(', source)
        self.assertIn('entries.filter((entry) => entry.startTick <= startTick && entry.endTick >= endTick)', source)
        self.assertIn("String(buff?.stackingMode || '') === 'independent'", source)
        self.assertIn('current.stackCount = Math.min(', source)
        self.assertIn('buffStackValue(current.stackCount) + buffStackValue(buff.stackCount)', source)
        self.assertIn('current.endTick = Math.max(Number(current.endTick || 0), Number(buff.endTick || 0));', source)
        self.assertNotIn('endTick: Number.isFinite(Number(segmentEndTick)) ? Number(segmentEndTick) : buff.endTick,', source)
        self.assertIn("id: String(buff?.definition_id || buff?.rule_id ||", source)
        self.assertIn('stackingMode: String(buff?.stacking_mode || \'\')', source)
        self.assertIn("stacking_mode: String(stacking.mode || 'refresh')", SHAFT_ENGINE_JS.read_text(encoding='utf-8'))

    def test_regular_and_reaction_buff_lines_share_one_merged_render_segment(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        css = (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css').read_text(encoding='utf-8')

        self.assertIn('function mergedRenderedBuffSegments(segments = [])', source)
        self.assertIn('const allBuffSegments = mergedRenderedBuffSegments([', source)
        self.assertIn(".flatMap((segment) => String(segment.tooltip || '').split('\\n'))", source)
        self.assertIn("tooltip: tooltipLines.join('\\n')", source)
        self.assertIn("buffIds: buffIds.join(' ')", source)
        self.assertIn('tooltipItems: [{', source)
        self.assertIn('name: effect.reaction,', source)
        self.assertIn('endTick: segment.endTick,', source)
        self.assertIn('durationTicks: effect.duration_ticks,', source)
        self.assertIn('min-width: 1px;', css)
        self.assertNotIn('layoutWidthPx(segment.startTick, segment.endTick, 22)', source)

    def test_buff_line_merges_duration_only_refreshes(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn(
            '.map((buff) => `${buff.id}:${buff.name}`)',
            source,
        )
        self.assertNotIn(
            '.map((buff) => `${buff.id}:${buff.name}:${buffStackKey(buff.stackCount)}:${Number(buff.endTick || 0)}`)',
            source,
        )

    def test_buff_line_merges_stack_only_changes_in_loop_and_non_loop_axes(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('.map((buff) => `${buff.id}:${buff.name}`)', source)
        self.assertNotIn(
            '.map((buff) => `${buff.id}:${buff.name}:${buffStackKey(buff.stackCount)}`)',
            source,
        )
        self.assertIn('previous.tooltipItems.push(...tooltipItems);', source)
        self.assertIn('visualStartTick: startTick,', source)
        self.assertIn('visualEndTick: endTick,', source)
        self.assertIn('Number(item.visualStartTick) <= pointerVisualTick', source)
        self.assertIn('pointerVisualTick < Number(item.visualEndTick)', source)

    def test_buff_lines_follow_buff_owner_and_hide_very_long_durations(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('const BUFF_DURATION_LABEL_LIMIT_TICKS = 9000;', source)
        self.assertIn('ownerSlot === Number(trackSlot)', source)
        self.assertIn(
            'const buffSegments = mergedBuffLineSegments(buffTimelineDetails, buffAxisEndTick, loopEnabled, member.slot);',
            source,
        )
        self.assertIn('durationTicks > BUFF_DURATION_LABEL_LIMIT_TICKS', source)
        self.assertIn('buffRemainingTicksAtCalculationTick(', source)

    def test_damage_triggered_refreshes_are_included_in_buff_timeline(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('const periodicDamageEvents = result?.periodic_damage_events || [];', source)
        self.assertIn('const damageTriggeredBuffDetails = [...reactionDamageEvents, ...periodicDamageEvents]', source)
        self.assertIn('.filter((event) => Array.isArray(event?.triggered_buffs) && event.triggered_buffs.length)', source)
        self.assertIn('const buffTimelineDetails = [...details, ...damageTriggeredBuffDetails];', source)
        self.assertIn('if (!buff || buff.cancelled || isUnlinedBuff(buff))', source)
        self.assertIn(
            'const buffSegments = mergedBuffLineSegments(buffTimelineDetails, buffAxisEndTick, loopEnabled, member.slot);',
            source,
        )

    def test_buff_line_tooltip_follows_pointer_position(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        css = (ROOT / 'app' / 'modules' / 'shaft' / 'static' / 'css' / 'shaft-page.css').read_text(encoding='utf-8')

        self.assertIn('function positionBuffLineTooltip(event)', source)
        self.assertIn('function buffRemainingTicksAtCalculationTick(buff, pointerCalculationTick)', source)
        self.assertIn(
            'buff?.calculationEndTick ?? calculationTickFromVisual(Number(buff?.endTick || 0))',
            source,
        )
        self.assertIn('const cycles = Math.ceil((calculationStartTick - effectivePointerTick) / loopDurationTicks);', source)
        self.assertIn('buff?.end_tick ?? calculationTickFromVisual(endTick)', source)
        self.assertIn('calculationEndTick: buff.calculationEndTick,', source)
        self.assertIn('calculationEndTick: Number(effect.end_tick || 0),', source)
        self.assertIn('buffRemainingTicksAtCalculationTick(item, pointerCalculationTick)', source)
        self.assertNotIn(
            'calculationTickFromVisual(Number(item.endTick || 0)) - pointerCalculationTick',
            source,
        )
        self.assertIn('`${item.name} · 剩余${ticksToSeconds(remainingTicks)}s${stackText}`', source)
        self.assertIn("buffLine.style.setProperty('--buff-tooltip-x', `${offsetX}px`);", source)
        self.assertIn("$('shaft-timeline').addEventListener('pointermove', positionBuffLineTooltip);", source)
        self.assertIn('left: var(--buff-tooltip-x, 50%);', css)
        self.assertIn('transform: translate(-50%, 4px);', css)

    def test_buff_line_uses_specific_passive_name_instead_of_character_name(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        function_match = re.search(
            r'function\s+triggeredBuffLines\s*\([^)]*\)\s*{(?P<body>.*?)\n  }',
            source,
            re.S,
        )
        if not function_match:
            raise AssertionError('triggeredBuffLines function is missing.')
        body = function_match.group('body')
        self.assertIn('const name = buffRuleDisplayName(buff);', body)
        self.assertNotIn('const name = buffDisplayName(buff);', body)
        self.assertIn("const passiveName = ruleId ? `被动 ${ruleId}` : '未命名被动';", source)

    def test_unlined_buff_tooltip_hides_duration_and_deduplicates_buffs(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('const uniqueBuffs = new Map();', source)
        self.assertIn('const key = buff.name;', source)
        self.assertIn('.map((buff) => `${buff.name}${buffStackText(buff.stackCount)}`);', source)
        self.assertNotIn('parts.push(buff.tooltip);', source)
        self.assertNotIn('tooltip: `${buffRuleDisplayName(buff)} · ${ticksToSeconds', source)

    def test_action_detail_deduplicates_triggered_buff_names(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('const triggeredBuffs = Array.from(new Set(', source)
        self.assertIn(".map((buff) => String(buff?.name || ''))", source)
        self.assertIn('.filter(Boolean),', source)

    def test_axis_edit_schedules_real_simulation_after_local_render(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        schedule_match = re.search(
            r'function\s+scheduleSimulation\s*\([^)]*\)\s*{\s*(?P<body>.*?)\n  }',
            source,
            re.S,
        )
        if not schedule_match:
            raise AssertionError('scheduleSimulation function is missing.')

        schedule_body = schedule_match.group('body')
        self.assertIn('renderLocalCalculationState(statusText);', schedule_body)
        self.assertIn('window.setTimeout(runSimulation, SIMULATION_DEBOUNCE_MS)', schedule_body)

    def test_frontend_uses_engine_display_ticks_for_timeline(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn("const displayStartTick = resultTick('display_start_tick', resultDetail.visual_start_tick ?? startTick);", source)
        self.assertIn("resultTick('display_visual_end_tick', fallbackDisplayVisualEndTick)", source)
        self.assertIn('triggered_buffs: projectedTriggeredBuffs', source)
        self.assertNotIn('const displayStartTick = startTick;', source)
        self.assertNotIn('function rebaseTriggeredBuffsForDisplay', source)

    def test_stale_timeline_projection_keeps_buffs_and_q_layout_until_recalculation(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('state.dragState?.resultSnapshot || state.result || null', source)
        self.assertIn('const usesStaleResult = Boolean(state.isResultStale && state.result && result === state.result);', source)
        self.assertIn("['visual_start_tick', 'visual_end_tick', 'display_start_tick', 'display_end_tick'].forEach", source)
        self.assertIn('Number(resultDetail.nominal_display_visual_end_tick) + projectionShiftTicks', source)

    def test_timeline_blank_click_updates_track_resources_at_cursor(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('function slotResourcesAtCursor(slot) {', source)
        self.assertIn('const cursorTick = Number(state.cursorTick || 0);', source)
        self.assertIn('detailTick > cursorTick', source)
        self.assertIn('latestDetail?.resources_after_by_slot', source)
        self.assertIn('const calculationCursorTick = calculationTickFromVisual(cursorTick);', source)
        self.assertIn('(result?.energy_events || []).reduce', source)
        self.assertIn('Number(event.tick || 0) > calculationCursorTick', source)
        self.assertIn('latestEnergyEvent?.energy_after ?? resource.initial_energy ?? snapshot.energy', source)
        self.assertIn('if (state.axis.initial_energy === 100) {', source)
        self.assertNotIn('const energyCapacity = Number(resource.energy_capacity ?? 0);', source)
        self.assertIn('const energyLabel = formatNumber(resources.energy || 0, 1);', source)
        self.assertIn('const harmony = Number(snapshot.harmony ?? resource.initial_harmony ?? 0);', source)
        self.assertNotIn('function slotResourcesForAxis', source)
        self.assertNotIn('primarySelectedLayoutForCursor', source)
        self.assertIn("node.style.left = `${leftPx(state.cursorTick)}px`;", source)
        self.assertIn('state.cursorTick = timelineTickFromEvent(event);', source)
        self.assertIn('renderTimeline();\n        renderStepDetail();\n        renderEditorActions();', source)

    def test_selection_and_initial_shortcuts_use_display_timeline_coordinates(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        catalog = get_shaft_catalog_payload()
        result = simulate_shaft_axis(catalog['starter_axis'])['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(details['step_017']['display_end_tick'], 106)
        self.assertEqual(details['step_018']['display_start_tick'], 106)
        self.assertIn('state.cursorTick = timelineDisplayTickForStep(step);', source)
        self.assertIn('detail?.display_start_tick ??', source)
        self.assertIn('timeline.focus({ preventScroll: true });', source)
        self.assertIn('focusTimelineForShortcuts();', source)

    def test_action_insertion_prioritizes_new_step_and_reveals_its_end(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        add_match = re.search(
            r'function\s+addActionAt\s*\([^)]*\)\s*{\s*(?P<body>.*?)\n  }\n\n  function addStep',
            source,
            re.S,
        )
        if not add_match:
            raise AssertionError('addActionAt function is missing.')

        add_body = add_match.group('body')
        self.assertNotIn('shiftStepsFromTick', source)
        self.assertIn('const insertTick = prepareInsertionTick(startTick, action, slot);', add_body)
        self.assertIn('startsForeground(candidateStep, action || {})', source)
        self.assertIn('!actionIntervalAtTick(target, slot)', source)
        self.assertIn('!startsForeground(candidateStep, action || {}) && !blocksSlotOverlap(candidateStep, action || {})', source)
        self.assertIn('normalizeEditedSteps(new Set([step.id]));', add_body)
        self.assertIn('revealTimelineTick(state.cursorTick);', add_body)
        self.assertLess(add_body.index('selectStep(step.id, false);'), add_body.index('state.cursorTick = Number(step.start_tick || 0) + actionVisualDurationTicks(action, step);'))
        self.assertNotIn('renderAll();', add_body)
        self.assertLess(add_body.index('scheduleSimulation();'), add_body.index('revealTimelineTick(state.cursorTick);'))
        self.assertIn("const shell = timeline?.closest('.shaft-timeline-shell');", source)

    def test_drag_preview_keeps_timeline_and_buff_snapshots_until_drop(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('function timelineResult() {', source)
        self.assertIn('return freshResult() || state.dragState?.previewResult || state.dragState?.resultSnapshot || state.result || null;', source)
        self.assertIn('originAxisTicks[item.id] = Number(item.start_tick || 0);', source)
        self.assertIn('const originalProjectionTick = Number(drag?.originAxisTicks?.[step.id] ?? resultDetail.raw_start_tick ?? startTick);', source)
        self.assertLess(
            source.index('originAxisTicks[item.id] = Number(item.start_tick || 0);'),
            source.index('if (dragStepIds.includes(item.id)) {'),
        )
        self.assertIn('expansionBreaks = clone(drag.timelineScale.expansionBreaks || []);', source)
        self.assertIn('buffLinesBySlot[slot] = Array.from(track.querySelectorAll(\'.shaft-buff-trigger-line\'))', source)
        self.assertIn('const heldBuffLines = heldBuffPreview?.buffLinesBySlot?.[String(member.slot)];', source)
        self.assertIn('Array.isArray(heldBuffLines) ? heldBuffLines.join(\'\') : allBuffSegments.map', source)
        self.assertIn('resultSnapshot: freshResult(),', source)
        self.assertIn('const usesDragPreviewResult = Boolean(drag?.previewResult && result === drag.previewResult);', source)
        self.assertIn('const projectionShiftTicks = (drag && !usesDragPreviewResult) || usesStaleResult', source)
        self.assertIn('if (drag?.timelineScale && !usesDragPreviewResult) {', source)
        self.assertIn("drag.previewResult = window.ShaftEngine.simulateAxis(state.axis, state.catalog);", source)
        self.assertNotIn('animateActionLayout(previousRects, { skipStepIds: dragStepIds, durationMs: 80 });', source)
        mouse_up_match = re.search(
            r'function\s+handleTimelineMouseUp\s*\([^)]*\)\s*{\s*(?P<body>.*?)\n  }\n\n  function handleTeamDockClick',
            source,
            re.S,
        )
        if not mouse_up_match:
            raise AssertionError('handleTimelineMouseUp function is missing.')
        mouse_up_body = mouse_up_match.group('body')
        self.assertIn('runSimulation();', mouse_up_body)
        self.assertNotIn('scheduleSimulation();', mouse_up_body)

    def test_dragging_left_across_a_visual_action_start_snaps_to_its_tick(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('function snapTickAfterCrossingVisualStart(drag, clientX, fallbackTick) {', source)
        self.assertIn("activeBar.closest('.action-track')?.querySelectorAll('.shaft-action-bar[data-step-id]')", source)
        self.assertIn('dragDisplayDetail?.display_start_tick ?? dragDisplayDetail?.visual_start_tick ?? step.start_tick', source)
        self.assertIn('originVisualOffset: timelineVisualOffset(originDisplayTick),', source)
        self.assertIn('originBarLeft: barRect.left,', source)
        self.assertIn('anchorOffsetX: event.clientX - barRect.left,', source)
        self.assertIn('const pressedBarRect = bar.getBoundingClientRect();', source)
        self.assertIn(".find((node) => node.dataset.stepId === step.id) || bar;", source)
        self.assertIn('activeBar.isConnected ? activeBar.getBoundingClientRect() : pressedBarRect;', source)
        self.assertIn('startTick: Math.max(0, Number(targetStep.start_tick || 0)),', source)
        self.assertIn('const nextTick = snapTickAfterCrossingVisualStart(drag, event.clientX, mappedTick);', source)

    def test_drag_mapping_preserves_raw_tick_when_display_tick_has_internal_q_offset(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('function axisTickFromDragDisplayTick(drag, displayTick) {', source)
        self.assertIn('const originAxisTick = Number(drag?.originTick || 0);', source)
        self.assertIn('const originDisplayTick = Number(drag?.originDisplayTick ?? originAxisTick);', source)
        self.assertIn('return Math.max(0, originAxisTick + Number(displayTick || 0) - originDisplayTick);', source)
        self.assertIn('const mappedDisplayTick = Math.max(0, tickFromTimelineXWithScale(', source)
        self.assertIn('const mappedTick = axisTickFromDragDisplayTick(drag, mappedDisplayTick);', source)

    def test_drag_priority_does_not_reverse_when_crossing_yi_e_to_support_end(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('function preserveGapsAfterAutomaticShift(originalStartTicks, priorityStepIds = new Set()) {', source)
        self.assertIn('const priorityIds = new Set(priorityStepIds || []);', source)
        self.assertIn('const carriedShiftBySlot = new Map();', source)
        self.assertIn('const slot = Number(step.slot || 0);', source)
        self.assertIn('if (priorityIds.has(step.id)) {', source)
        self.assertIn('step.start_tick = resolvedStartTick;', source)
        self.assertIn('carriedShiftBySlot.set(slot, 0);', source)
        self.assertIn('const maxPasses = Math.max(2, (state.axis?.steps || []).length + 1);', source)

    def test_foreground_conflicts_keep_q_and_support_locks_separate_from_start_gap(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('const slotBlockingEnd = new Map();', source)
        self.assertIn('previousForegroundStartTick + MIN_FOREGROUND_START_GAP_TICKS', source)
        self.assertIn('function locksForegroundSwitch(step, action = actionForStep(step)) {', source)
        self.assertIn('function foregroundLockEndTick(step, action, startTick) {', source)
        self.assertIn('detail.display_visual_end_tick ?? detail.visual_end_tick', source)
        self.assertIn('const foregroundLocks = [];', source)
        self.assertIn('.filter((lock) => startTick < Number(lock.endTick))', source)
        self.assertNotIn('const supportConflict = foregroundStarts.find', source)
        self.assertNotIn('const supportProtectsItsInterval', source)
        self.assertNotIn('const qConflict =', source)

    def test_loading_saved_axis_recomputes_stale_stored_result(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        load_match = re.search(
            r'async function loadAxis\([^)]*\)\s*{(?P<body>.*?)\n  }\n\n  async function deleteAxis',
            source,
            re.S,
        )
        if not load_match:
            raise AssertionError('loadAxis function is missing.')
        body = load_match.group('body')

        self.assertNotIn('acceptSimulationResult(payload.result);', body)
        self.assertIn('state.result = null;', body)
        self.assertIn('markSimulationStale();', body)
        self.assertIn('await runSimulation();', body)
        self.assertLess(body.index('await runSimulation();'), body.index('markAxisDocumentClean();'))

    def test_my_axis_cards_can_backup_and_open_the_copy(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('data-backup-axis="${axis.id}"', source)
        self.assertIn('async function backupAxis(axisId, button) {', source)
        self.assertIn('`/api/shaft/axes/${axisId}/backup`', source)
        self.assertIn("await loadAxis(Number(backup.id), 'mine');", source)

    def test_my_axis_share_link_copies_and_opening_protects_dirty_workspace(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        template = SHAFT_TEMPLATE.read_text(encoding='utf-8')

        self.assertIn('data-share-axis="${axis.id}"', source)
        self.assertIn('`/api/shaft/axes/${axisId}/share`', source)
        self.assertIn('navigator.clipboard?.writeText', source)
        self.assertIn("showToast('分享链接已复制到剪切板');", source)
        self.assertIn('id="shaft-share-open-dialog"', template)
        for value in ('cancel', 'discard', 'save_as', 'save'):
            self.assertIn(f'value="{value}"', template)
        self.assertIn('async function openSharedAxisFromUrl()', source)
        self.assertIn('if (hasUnsavedAxisChanges()) {', source)
        self.assertIn("action === 'save' && !await saveAxis()", source)
        self.assertIn("action === 'save_as' && !await saveAxisAsCopy()", source)
        self.assertIn('state.savedAxisId = null;', source)
        self.assertIn('sharedReadOnly: false,', source)
        self.assertIn('state.sharedReadOnly = Boolean(payload.read_only);', source)
        self.assertIn('id="shaft-shared-readonly-notice"', template)
        self.assertIn('if (state.sharedReadOnly) {', source)
        self.assertIn('input.readOnly = readOnly;', source)
        base_template = (ROOT / 'app' / 'templates' / 'common' / 'base.html').read_text(encoding='utf-8')
        self.assertIn('not (allow_anonymous_shaft_share | default(false))', base_template)
        self.assertIn("setPage('rotation');", source)
        self.assertIn('await openSharedAxisFromUrl();', source)

    def test_plaza_uses_equal_columns_snapshot_upload_and_my_axis_filters(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        template = SHAFT_TEMPLATE.read_text(encoding='utf-8')
        css = SHAFT_CSS.read_text(encoding='utf-8')

        self.assertIn('grid-template-columns: repeat(2, minmax(0, 1fr));', css)
        self.assertNotIn('id="shaft-publish-btn"', template)
        self.assertIn('data-publish-axis="${axis.id}"', source)
        self.assertIn('`/api/shaft/axes/${axisId}/publish`', source)
        self.assertIn('id="shaft-my-axis-scope"', template)
        self.assertIn('id="shaft-my-character-filter-trigger"', template)
        self.assertIn('id="shaft-my-axis-sort"', template)
        self.assertIn('id="shaft-my-axis-query"', template)
        self.assertIn('id="shaft-market-query"', template)
        self.assertIn(
            '我的排轴 <span class="shaft-axis-total" id="shaft-my-axis-total">0</span>',
            template,
        )
        self.assertIn('state.myAxisCharacterIds.forEach', source)
        market_card = source[source.index('function marketCardHtml'):source.index('function renderMarketList')]
        self.assertIn("${axis.loop_enabled ? '循环' : '单轮'}", market_card)
        self.assertIn('shaft-axis-mode-badge', market_card)
        self.assertIn('<span>轴长 ${formatNumber(axis.duration_seconds || 0, 1)}s</span>', market_card)
        self.assertIn('<span>环合 ${formatNumber(axis.harmony_damage || 0)}</span>', market_card)
        self.assertIn('data-like-axis="${axis.id}"', market_card)
        self.assertIn('data-dislike-axis="${axis.id}"', market_card)
        self.assertIn('data-unpublish-axis="${axis.id}"', market_card)
        self.assertIn('member.character_avatar', market_card)
        self.assertIn('shaft-market-character', market_card)
        self.assertIn('const awakeningCount = activeAwakeningCount(member);', market_card)
        self.assertIn('shaft-market-character-avatar', market_card)
        self.assertIn('shaft-market-character-awakening', market_card)
        self.assertIn('aria-label="已激活 ${awakeningCount} 个觉醒"', market_card)
        self.assertIn('.shaft-market-character-awakening {', css)
        self.assertIn('font-size: 8px;', css)
        self.assertIn('descriptionCharacters.slice(0, 64)', market_card)
        self.assertIn("descriptionPreview || '暂无备注'", market_card)
        self.assertIn('min-height: 1.4em;', css)
        self.assertIn('text-overflow: ellipsis;', css)
        self.assertIn('white-space: nowrap;', css)
        self.assertIn('data-publish-mode="update"', market_card)
        self.assertIn('publishedSnapshotId', market_card)
        self.assertIn('踩 ${axis.dislike_count || 0}', market_card)
        self.assertNotIn('<span>倾陷 ', market_card)

        self.assertIn('async function unpublishAxis(axisId, title, button)', source)
        self.assertIn("const unpublishButton = event.target.closest('[data-unpublish-axis]');", source)
        self.assertIn('公开快照将从广场移除', source)
        self.assertIn('私有原轴仍保留', source)
        self.assertIn("params.set('q', state.myAxisQuery);", source)
        self.assertIn("params.set('q', state.marketQuery);", source)
        self.assertIn('const MARKET_SEARCH_INTERVAL_MS = 3000;', source)
        self.assertIn('const MARKET_SEARCH_EDIT_DELAY_MS = 600;', source)
        self.assertIn('function renderMarketSearchLoading()', source)
        self.assertIn("list.setAttribute('aria-busy', 'true');", source)
        self.assertIn('shaft_market_search_rate_limited', source)
        self.assertIn('error.payload.retry_after_ms', source)
        self.assertIn("$('shaft-my-axis-total').textContent", source)
        self.assertIn('MARKET_SEARCH_EDIT_DELAY_MS,', source)
        market_query_listener = source[source.index("$('shaft-market-query').addEventListener('input'"):]
        market_query_listener = market_query_listener[:market_query_listener.index("\n    });") + 8]
        self.assertNotIn('renderMarketSearchLoading()', market_query_listener)
        self.assertIn('.shaft-search-loading', css)

    def test_axis_save_submits_fresh_local_result_without_server_simulation(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        save_start = source.index("async function saveAxis(conflictAction = '')")
        save_end = source.index('async function loadMarket', save_start)
        save_body = source[save_start:save_end]
        self.assertIn('if (!freshResult()) {', save_body)
        self.assertIn('await runSimulation();', save_body)
        self.assertIn('const result = freshResult();', save_body)
        self.assertIn('result,', save_body)
        self.assertLess(save_body.index('await runSimulation();'), save_body.index('const payload = {'))

    def test_axis_save_as_requires_a_changed_title(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        template = SHAFT_TEMPLATE.read_text(encoding='utf-8')

        self.assertIn('id="shaft-save-as-btn"', template)
        self.assertIn('function canSaveAxisAs() {', source)
        self.assertIn('currentTitle !== normalizedAxisTitle(state.savedAxisTitle)', source)
        self.assertIn('saveAsButton.disabled = state.sharedReadOnly || !state.savedAxisId;', source)
        self.assertIn('async function saveAxisAsNamedCopy() {', source)
        self.assertIn("showToast('请先填写新的轴名，再点击另存', 'warning');", source)
        self.assertIn("$('shaft-title-input');", source)
        self.assertIn('titleInput.focus();', source)
        self.assertIn('titleInput.select();', source)
        self.assertIn("$('shaft-save-as-btn').addEventListener('click', saveAxisAsNamedCopy);", source)

    def test_iloy_lucid_dream_can_move_from_foreground_to_background(self) -> None:
        catalog = load_shaft_catalog()
        lucid_dream = next(
            action for action in catalog['actions']
            if action['id'] == 'action_iloy_lucid_dream'
        )

        self.assertEqual(lucid_dream['name'], '清明梦')
        self.assertEqual(lucid_dream['action_type'], '普攻')
        self.assertTrue(lucid_dream['can_background_override'])

    def test_market_dislike_uses_authenticated_toggle_and_refreshes_market(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('async function toggleDislike(axisId, active)', source)
        self.assertIn('`/api/shaft/axes/${axisId}/dislike`', source)
        self.assertIn("const dislikeButton = event.target.closest('[data-dislike-axis]');", source)
        self.assertIn('await toggleDislike(', source)

    def test_publish_button_is_restored_after_success_or_failure(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        publish = source[source.index('async function publishAxis'):source.index('async function toggleLike')]

        self.assertIn('} finally {', publish)
        self.assertIn('button.disabled = false;', publish)
        self.assertIn('button.textContent = originalLabel;', publish)

    def test_same_name_save_offers_rename_overwrite_or_cancel(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        template = SHAFT_TEMPLATE.read_text(encoding='utf-8')

        self.assertIn('id="shaft-name-conflict-dialog"', template)
        self.assertIn('value="cancel"', template)
        self.assertIn('value="overwrite"', template)
        self.assertIn('value="rename"', template)
        self.assertIn("error.payload?.code === 'axis_name_conflict'", source)
        self.assertIn("await saveAxis('overwrite');", source)
        self.assertIn("$('shaft-title-input').value = resolution.title;", source)

    def test_keyboard_moves_cursor_and_selected_actions_without_hijacking_inputs(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('if (isEditableTarget(event.target)) {', source)
        keydown = source[source.index('function handleKeydown(event) {'):source.index('function handleClipboardCopy(event) {')]
        self.assertIn("if (state.page !== 'rotation') {", keydown)
        self.assertLess(keydown.index("if (state.page !== 'rotation') {"), keydown.index("event.key.toLowerCase() === 'z'"))
        clipboard_copy = source[source.index('function handleClipboardCopy(event) {'):source.index('function handleClipboardPaste(event) {')]
        clipboard_paste = source[source.index('function handleClipboardPaste(event) {'):source.index('function timelineShell() {')]
        self.assertIn("state.page !== 'rotation' || state.sharedReadOnly", clipboard_copy)
        self.assertIn("state.page !== 'rotation' || state.sharedReadOnly", clipboard_paste)
        self.assertIn('function selectAllTimelineSteps() {', source)
        self.assertIn("event.key.toLowerCase() === 'a' &&", source)
        self.assertIn("state.page === 'rotation'", source)
        self.assertIn('selectAllTimelineSteps();', source)
        self.assertLess(source.index('if (isEditableTarget(event.target)) {'), source.index("event.key.toLowerCase() === 'a' &&"))
        self.assertIn("moveTimelineCursor(event.key === 'ArrowLeft' ? -1 : 1);", source)
        self.assertIn('function selectStepByKeyboard(key) {', source)
        self.assertIn('selectStepByKeyboard(movementKey);', source)
        self.assertIn("} else if (key === 'd') {", source)
        self.assertIn('moveTimelineCursorTo(stepVisualEndTickForCursor(primary));', source)
        self.assertIn('detail?.display_visual_end_tick ?? detail?.visual_end_tick ?? fallbackEndTick', source)
        self.assertIn("moveSelectedStepsByKeyboard(movementKey === 'q' ? -1 : 1);", source)
        self.assertNotIn('moveSelectedStepsToPlacement', source)
        self.assertIn("finishKeyboardStepEdit(stepIds, primary.id, swapped ? '已交换动作位置' : '已移动动作');", source)
        self.assertIn('const spanEnd = stepStart + stepDuration;', source)
        self.assertIn('const spanEnd = neighborStart + neighborDuration;', source)

    def test_axis_editing_preserves_player_authored_gaps_without_reflow(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertNotIn('function compactIdleGaps()', source)
        self.assertNotIn('compactIdleGaps();', source)
        self.assertIn('function preserveGapsAfterAutomaticShift(originalStartTicks, priorityStepIds = new Set()) {', source)
        self.assertIn('const currentOriginalTick = Math.max(0, Number(', source)
        self.assertIn('groupSteps.forEach((step) => {', source)
        self.assertIn('const carriedShiftBySlot = new Map();', source)
        self.assertIn('const carriedShiftTicks = Math.max(0, Number(carriedShiftBySlot.get(slot) || 0));', source)
        self.assertIn('currentOriginalTick + carriedShiftTicks', source)
        self.assertIn('carriedShiftBySlot.set(slot, Math.max(carriedShiftTicks, step.start_tick - currentOriginalTick));', source)
        self.assertNotIn('function reflowSteps()', source)
        self.assertNotIn("$('shaft-reflow-btn')", source)

    def test_new_axis_has_a_clear_entry_and_protects_unsaved_changes(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        template = (ROOT / 'app' / 'modules' / 'shaft' / 'templates' / 'shaft' / 'index.html').read_text(encoding='utf-8')

        self.assertIn('id="shaft-new-btn"', template)
        self.assertIn('>新建动作轴</button>', template)
        self.assertNotIn('id="shaft-reset-btn"', template)
        self.assertIn('function hasUnsavedAxisChanges() {', source)
        self.assertIn("window.confirm('当前动作轴有未保存的修改，确定新建并放弃这些修改吗？')", source)
        self.assertIn("$('shaft-new-btn').addEventListener('click', newAxis);", source)
        self.assertNotIn("$('shaft-reset-btn')", source)

    def test_axis_info_bar_is_shared_by_build_and_rotation_but_hidden_on_plaza(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        template = (ROOT / 'app' / 'modules' / 'shaft' / 'templates' / 'shaft' / 'index.html').read_text(encoding='utf-8')
        styles = SHAFT_CSS.read_text(encoding='utf-8')

        self.assertEqual(template.count('data-shaft-axis-info'), 1)
        self.assertLess(
            template.index('data-shaft-axis-info'),
            template.index('data-shaft-view="build"'),
        )
        self.assertEqual(template.count('id="shaft-title-input"'), 1)
        self.assertEqual(template.count('id="shaft-description-input"'), 1)
        self.assertEqual(template.count('id="shaft-new-btn"'), 1)
        self.assertEqual(template.count('id="shaft-save-btn"'), 1)
        self.assertIn("axisInfoBar.hidden = state.page === 'plaza';", source)
        self.assertIn('.shaft-command-panel[hidden]', styles)

    def test_header_page_tabs_use_equal_outer_columns_for_true_centering(self) -> None:
        styles = SHAFT_CSS.read_text(encoding='utf-8')

        self.assertIn(
            'grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);',
            styles,
        )
        self.assertIn('.shaft-header .shaft-page-tabs', styles)
        self.assertIn('justify-self: center;', styles)
        self.assertIn('.shaft-header .header-actions', styles)
        self.assertIn('justify-self: end;', styles)

    def test_action_library_shows_energy_cost_instead_of_zero_energy_gain(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('Number(action.energy_cost || 0) > 0', source)
        self.assertIn('`耗能 ${formatNumber(action.energy_cost, 0)}`', source)

    def test_zero_q_visual_slot_label_hides_internal_sequence(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        zero_action_visual_ticks = _js_number_constant('ZERO_ACTION_VISUAL_TICKS')
        timeline_tick_px = _js_number_constant('TIMELINE_TICK_PX')
        min_action_card_px = _js_number_constant('MIN_ACTION_CARD_PX')
        q_starts = [index * zero_action_visual_ticks for index in range(4)]

        self.assertNotIn('Q时序', source)
        self.assertGreaterEqual(zero_action_visual_ticks * timeline_tick_px, min_action_card_px)
        self.assertLess(zero_action_visual_ticks * timeline_tick_px - min_action_card_px, timeline_tick_px)
        for index, visual_tick in enumerate(q_starts, start=1):
            self.assertEqual(_calculation_tick_from_visual(visual_tick, q_starts), 0)
            self.assertEqual(_visual_tick_label(visual_tick, q_starts), '0.0s')

        right_side_of_first_q = zero_action_visual_ticks
        self.assertEqual(_calculation_tick_from_visual(right_side_of_first_q, q_starts), 0)
        self.assertEqual(_visual_tick_label(right_side_of_first_q, q_starts), '0.0s')

    def test_q_timeline_abilities_require_zero_duration_foreground_q(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('function isZeroForegroundQStep(step, action = actionForStep(step)) {', source)
        self.assertIn('startsForeground(step, action) && isQAction(action) && actionDurationTicks(action) === 0', source)
        self.assertIn('const actionIsQ = isZeroForegroundQStep(step, action);', source)
        self.assertIn('.filter((item) => isZeroForegroundQStep(item.step, item.action))', source)
        self.assertIn('isZeroForegroundQStep(candidateStep, action || {}) ||', source)
        self.assertIn('tickHasForegroundQ(target) ||', source)
        self.assertNotIn('const currentIsQ = foregroundStart && isQAction(action);', source)

    def test_noop_switch_has_independent_timeline_semantics(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')
        engine = SHAFT_ENGINE_JS.read_text(encoding='utf-8')

        self.assertIn('function isInstantSwitchAction(action) {', source)
        self.assertIn("function tickHasInstantSwitchAction(tick, ignoreStepId = '') {", source)
        self.assertIn('!isInstantSwitchAction(action)', source)
        self.assertIn('!isInstantSwitchAction(action)', engine)
        self.assertIn('!isSupportAction(scheduled.action)', engine)

    def test_four_connected_zero_q_actions_keep_min_width_without_overlap(self) -> None:
        zero_q_actions = _zero_q_actions()
        zero_action_visual_ticks = _js_number_constant('ZERO_ACTION_VISUAL_TICKS')
        min_action_card_px = _js_number_constant('MIN_ACTION_CARD_PX')
        payload = {
            'team': [
                {
                    'slot': index,
                    'character_id': action['character_id'],
                    'arc_id': '',
                    'cartridge_id': '',
                }
                for index, action in enumerate(zero_q_actions)
            ],
            'steps': [
                {
                    'id': f'connected_q_{index}',
                    'slot': index,
                    'action_id': action['id'],
                    'start_tick': index * zero_action_visual_ticks,
                }
                for index, action in enumerate(zero_q_actions)
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }

        layout = _frontend_timeline_layout(payload)
        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}

        self.assertEqual(len(layout), 4)
        for entry in layout:
            self.assertEqual(entry['width_px'], min_action_card_px)
        for previous, current in zip(layout, layout[1:]):
            self.assertEqual(current['left_px'], previous['right_px'])
            self.assertGreaterEqual(current['left_px'], previous['right_px'])
        self.assertEqual(result['summary']['duration_ticks'], 0)
        self.assertEqual(result['summary']['duration_seconds'], 0)
        for index in range(4):
            detail = details[f'connected_q_{index}']
            self.assertEqual(detail['raw_start_tick'], index * zero_action_visual_ticks)
            self.assertEqual(detail['start_tick'], 0)
            self.assertEqual(detail['end_tick'], 0)
            self.assertEqual(detail['duration_ticks'], 0)

    def test_yi_q_e_reordering_preserves_total_visual_span(self) -> None:
        team = [{
            'slot': 0,
            'character_id': 'char_7578b18979',
            'arc_id': '',
            'cartridge_id': '',
        }]
        common = {
            'team': team,
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }
        q_then_e = _frontend_timeline_layout({
            **common,
            'steps': [
                {'id': 'yi_q', 'slot': 0, 'action_id': 'action_6ece34aff8', 'start_tick': 0},
                {'id': 'yi_e', 'slot': 0, 'action_id': 'action_2e072f2b0b', 'start_tick': 5},
            ],
        })
        e_then_q = _frontend_timeline_layout({
            **common,
            'steps': [
                {'id': 'yi_e', 'slot': 0, 'action_id': 'action_2e072f2b0b', 'start_tick': 0},
                {'id': 'yi_q', 'slot': 0, 'action_id': 'action_6ece34aff8', 'start_tick': 15},
            ],
        })

        def visual_span(layout: list[dict]) -> int:
            return max(entry['right_px'] for entry in layout) - min(entry['left_px'] for entry in layout)

        self.assertEqual(q_then_e[0]['right_px'], q_then_e[1]['left_px'])
        self.assertEqual(e_then_q[0]['right_px'], e_then_q[1]['left_px'])
        self.assertEqual(visual_span(q_then_e), visual_span(e_then_q))
        self.assertEqual(visual_span(q_then_e), 20 * _js_number_constant('TIMELINE_TICK_PX'))

    def test_short_nonzero_action_after_q_keeps_true_timeline_position(self) -> None:
        min_action_card_px = _js_number_constant('MIN_ACTION_CARD_PX')
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'zhenhong_q', 'slot': 0, 'action_id': 'action_c32b4b9417', 'start_tick': 0},
                {'id': 'dragon_a1', 'slot': 0, 'action_id': 'action_40e3b63106', 'start_tick': 2},
                {'id': 'dragon_a2', 'slot': 0, 'action_id': 'action_c2c019771f', 'start_tick': 5},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }

        layout = _frontend_timeline_layout(payload)
        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}
        layout_by_step = {entry['step_id']: entry for entry in layout}

        self.assertEqual(layout_by_step['zhenhong_q']['width_px'], min_action_card_px)
        self.assertEqual(layout_by_step['dragon_a1']['width_px'], 3 * _js_number_constant('TIMELINE_TICK_PX'))
        self.assertEqual(layout_by_step['dragon_a1']['left_px'], layout_by_step['zhenhong_q']['right_px'])
        self.assertEqual(layout_by_step['dragon_a2']['left_px'], layout_by_step['dragon_a1']['right_px'])
        self.assertEqual(details['dragon_a1']['duration_ticks'], 3)
        self.assertEqual(details['dragon_a1']['end_tick'], details['dragon_a1']['start_tick'] + 3)
        self.assertEqual(details['dragon_a1']['visual_end_tick'] - details['dragon_a1']['visual_start_tick'], 3)
        self.assertFalse(details['dragon_a1']['q_instant_release'])
        source = SHAFT_JS.read_text(encoding='utf-8')
        self.assertNotIn('const foregroundLaneEnds = [];', source)
        self.assertIn('let laneIndex = 0;\n        if (detail.is_background_damage)', source)
        self.assertIn('laneIndex = backgroundLaneIndex + 1;', source)

    def test_background_actions_only_use_third_row_for_visual_overlap(self) -> None:
        source = SHAFT_JS.read_text(encoding='utf-8')

        self.assertIn('backgroundLaneEnds.findIndex((endPx) => startPx >= endPx)', source)
        self.assertNotIn('backgroundLaneEnds.findIndex((endPx) => startPx >= endPx + 6)', source)

    def test_q_cover_expands_over_chain_without_overlapping_return_q(self) -> None:
        min_action_card_px = _js_number_constant('MIN_ACTION_CARD_PX')
        payload = {
            'team': [
                {
                    'slot': 0,
                    'character_id': 'char_b52cc8f160',
                    'arc_id': '',
                    'cartridge_id': '',
                },
                {
                    'slot': 1,
                    'character_id': 'char_dd034941ef',
                    'arc_id': '',
                    'cartridge_id': '',
                },
            ],
            'steps': [
                {'id': 'zhenhong_a4', 'slot': 0, 'action_id': 'action_5cd7ad2380', 'start_tick': 0},
                {'id': 'main_q', 'slot': 1, 'action_id': 'action_b356b46da2', 'start_tick': 2},
                {'id': 'zhenhong_a5', 'slot': 0, 'action_id': 'action_cf3b21dac1', 'start_tick': 8, 'placement': 'background'},
                {'id': 'main_e', 'slot': 1, 'action_id': 'action_982c67944f', 'start_tick': 18},
                {'id': 'zhenhong_q', 'slot': 0, 'action_id': 'action_c32b4b9417', 'start_tick': 20},
            ],
            'options': {'switch_loss_ticks': 0},
            'initial_energy': 200,
        }

        layout = _frontend_timeline_layout(payload)
        result = simulate_shaft_axis(payload)['result']
        details = {detail['step_id']: detail for detail in result['details']}
        layout_by_step = {entry['step_id']: entry for entry in layout}

        self.assertGreater(layout_by_step['main_q']['width_px'], min_action_card_px)
        self.assertEqual(details['main_q']['visual_end_tick'], details['zhenhong_a5']['visual_end_tick'])
        self.assertEqual(layout_by_step['main_q']['right_px'], layout_by_step['zhenhong_a5']['right_px'])
        self.assertEqual(layout_by_step['main_e']['left_px'], layout_by_step['main_q']['right_px'])
        self.assertGreaterEqual(layout_by_step['zhenhong_q']['left_px'], layout_by_step['zhenhong_a5']['right_px'])
        self.assertEqual(details['main_e']['start_tick'], 2)
        self.assertEqual(details['zhenhong_q']['start_tick'], 4)


if __name__ == '__main__':
    unittest.main()
