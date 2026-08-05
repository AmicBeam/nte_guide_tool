(function () {
  const ELEMENTS = ['光', '灵', '咒', '暗', '魂', '相'];
  const RESISTANCE_ELEMENTS = [...ELEMENTS, '心灵'];
  const SLOT_COLORS = ['#58d8d2', '#ffbf57', '#ff5aa5', '#77e36f'];
  const ACTION_CONTRIBUTION_COLORS = ['#58d8d2', '#ffbf57', '#ff5aa5', '#77e36f', '#b28cff', '#62b7ff', '#ff7a63', '#d8e26a'];
  const ACTION_TYPE_COLORS = {
    '普攻': '#58d8d2',
    'E': '#62b7ff',
    'Q': '#b28cff',
    '援护': '#ffbf57',
    '被动': '#ff5aa5',
    '无': '#b9c6d8',
    '闪反': '#77e36f',
    '下落': '#ff7a63',
    '环合': '#ff5aa5',
    '其他': '#b9c6d8',
  };
  const DAMAGE_SOURCE_COLORS = {
    '创生': '#4fdcc8',
    '创生复制体': '#82eadb',
    '浊燃': '#ff665c',
    '黯星': '#6f8fff',
    '倾陷': '#eef4ff',
  };
  const BUFF_LINE_COLORS = ['#62b7ff', '#f9ca62', '#95ec87', '#ff7ab8', '#b28cff', '#55dacd'];
  const ACTION_TYPES = ['', '普攻', 'E', 'Q', '援护', '无'];
  const SUBSTAT_ORDER = [
    'all_dmg',
    'crit_rate',
    'crit_dmg',
    'harmony_strength',
    'stagger_strength',
    'atk_pct',
    'flat_atk',
    'hp_pct',
    'flat_hp',
    'def_pct',
    'flat_def',
  ];
  const BUFF_MODIFIERS = [
    { key: 'all_dmg', label: '通伤', percent: true },
    { key: 'crit_rate', label: '暴击', percent: true },
    { key: 'crit_dmg', label: '暴伤', percent: true },
    { key: 'atk_pct', label: '攻击%', percent: true },
    { key: 'def_down', label: '减防', percent: true },
    { key: 'res_down', label: '抗性降低', percent: true },
    { key: 'skill_dmg', label: '战技伤害', percent: true },
    { key: 'ultimate_dmg', label: '终结伤害', percent: true },
    { key: 'final_dmg', label: '最终伤害', percent: true },
    { key: 'harmony_strength', label: '环合强度', percent: false },
    { key: 'stagger_strength', label: '倾陷强度', percent: false },
  ];
  const ZERO_ACTION_VISUAL_TICKS = 5;
  const MIN_FOREGROUND_START_GAP_TICKS = 2;
  const MAX_BACKGROUND_ACTION_MULTIPLIER = 999;
  const TIMELINE_END_PADDING_TICKS = 10;
  const SUBSTAT_TOTAL_TOOLTIP = 'II / III / IV 型驱动上的每个副词条分别对应 2 / 3 / 4 个词条，卡带上的副词条对应 10 个词条。只支持金色卡带、驱动块。';
  const TIMELINE_TICK_PX = 12;
  const TIMELINE_LABEL_PX = 128;
  const PREVIEW_LABEL_PX = 116;
  const PREVIEW_MAX_TICK_PX = 24;
  const PREVIEW_MIN_BODY_PX = 1;
  const MIN_ACTION_CARD_PX = 60;
  const BUFF_DURATION_LABEL_LIMIT_TICKS = 9000;
  const MAX_VISUAL_LANES_PER_SLOT = 3;
  const BACKGROUND_DRAG_THRESHOLD_PX = 24;
  const TIMELINE_AUTO_SCROLL_EDGE_PX = 52;
  const TIMELINE_AUTO_SCROLL_MAX_PX = 24;
  const SIMULATION_DEBOUNCE_MS = 320;
  const MARKET_SEARCH_INTERVAL_MS = 3000;
  const MARKET_SEARCH_EDIT_DELAY_MS = 600;
  const DRAFT_STORAGE_KEY = 'shaft_axis_draft_v1';
  const TIMELINE_FIXED_PERSONAL_RESOURCES = {
    char_31c5130304: ['真理之匙'],
    char_a01c39f576: ['臆想'],
  };
  const TIMELINE_HIDDEN_PERSONAL_RESOURCES = new Set(['噩梦']);
  const DETAIL_HIDDEN_APPLIED_BUFF_IDS = new Set([
    'character_requiem_nightmare',
    'character_requiem_nightmare_stack',
  ]);
  const DEFAULT_TEAM_PANEL_BONUS = {
    version: 3,
    furniture_crit_dmg: 0.04,
    furniture_flat_atk: 20,
    furniture_flat_def: 30,
    small_flat_atk: 420,
    small_flat_hp: 5200,
  };
  const TEAM_PANEL_BONUS_FIELDS = [
    { key: 'furniture_crit_dmg', label: '暴伤家具', kind: 'percent', max: 4, step: 0.1 },
    { key: 'furniture_flat_atk', label: '攻击家具', kind: 'flat', max: 20, step: 1 },
    { key: 'furniture_flat_def', label: '防御家具', kind: 'flat', max: 30, step: 1 },
  ];
  const SKILL_LEVEL_FIELDS = [
    { key: 'basic', label: '普攻' },
    { key: 'skill', label: 'E' },
    { key: 'ultimate', label: 'Q' },
    { key: 'support', label: '援护' },
  ];
  const AWAKENING_LABELS = ['A', 'B', 'C', 'D', 'E', 'F'];
  const DEFAULT_SKILL_LEVELS = {
    basic: 10,
    skill: 10,
    ultimate: 10,
    support: 10,
  };
  const actionAnalysisState = {
    slot: null,
    dimension: 'action',
    view: 'donut',
  };
  const CURTAIN_PASSIVE_TYPES = [
    { key: 'type2', label: '2型被动' },
    { key: 'type3', label: '3型被动' },
    { key: 'type4', label: '4型被动' },
  ];

  const state = {
    page: 'rotation',
    catalog: null,
    axis: null,
    result: null,
    marketItems: [],
    marketPage: 1,
    marketHasMore: false,
    marketCharacterIds: [],
    marketQuery: '',
    marketSearchTimer: 0,
    marketLastSearchAt: 0,
    marketRequestId: 0,
    myAxisFilter: 'mine',
    myAxisCharacterIds: [],
    myAxisSort: 'new',
    myAxisQuery: '',
    myAxisSearchTimer: 0,
    toastTimer: 0,
    savedAxisId: null,
    savedAxisTitle: '',
    sharedReadOnly: false,
    axisDocumentBaseline: '',
    characterSwitchUnsavedConfirmed: false,
    simulationTimer: 0,
    simulationInFlight: false,
    resultFingerprint: '',
    isResultStale: false,
    selectedStepId: '',
    selectedStepIds: [],
    compareSnapshot: null,
    buildPanelSlot: 0,
    cursorTick: 0,
    insertMode: 'cursor',
    librarySlot: 0,
    commandTypeFilter: '',
    commandSearch: '',
    timelineDurationTicks: TIMELINE_END_PADDING_TICKS,
    timelineScale: { expansionBreaks: [] },
    timelineDisplayDetails: [],
    dragState: null,
    marqueeState: null,
    timelineAutoScroll: null,
    clipboardSteps: [],
    suppressClipboardPasteUntil: 0,
    undoStack: [],
    redoStack: [],
    suppressTimelineClickUntil: 0,
    buffDraft: {
      triggerSlot: 0,
      modifierKey: 'all_dmg',
    },
    previewTickPx: 0,
    axisPreviewPayload: null,
    axisPreviewSaving: false,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function axisDocumentFingerprint() {
    const axis = clone(state.axis || {});
    delete axis.duration_ticks;
    return JSON.stringify({
      title: $('shaft-title-input')?.value || '未命名排轴',
      description: $('shaft-description-input')?.value || '',
      axis,
      compareSnapshot: state.compareSnapshot || null,
    });
  }

  function markAxisDocumentClean() {
    state.axisDocumentBaseline = axisDocumentFingerprint();
    state.characterSwitchUnsavedConfirmed = false;
  }

  function hasUnsavedAxisChanges() {
    if (!state.axisDocumentBaseline) {
      return Boolean(
        (state.axis?.steps || []).length ||
        (state.axis?.buff_rules || []).length ||
        state.compareSnapshot ||
        ($('shaft-title-input')?.value && $('shaft-title-input').value !== '未命名排轴') ||
        $('shaft-description-input')?.value
      );
    }
    return axisDocumentFingerprint() !== state.axisDocumentBaseline;
  }

  function normalizedAxisTitle(value) {
    return String(value || '').trim();
  }

  function canSaveAxisAs() {
    const currentTitle = normalizedAxisTitle($('shaft-title-input')?.value);
    return Boolean(
      state.savedAxisId &&
      currentTitle &&
      currentTitle !== normalizedAxisTitle(state.savedAxisTitle)
    );
  }

  function renderSaveActions() {
    const saveAsButton = $('shaft-save-as-btn');
    if (saveAsButton) {
      saveAsButton.disabled = state.sharedReadOnly || !state.savedAxisId;
    }
  }

  function emptyAxisFromCatalog() {
    const starter = clone(state.catalog?.starter_axis || {});
    starter.steps = [];
    starter.buff_rules = [];
    starter.duration_ticks = 1;
    return starter;
  }

  function displayText(value) {
    return String(value ?? '').replace(/精通/g, '环合强度');
  }

  function escapeHtml(value) {
    return displayText(value).replace(/[&<>"']/g, (char) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    }[char]));
  }

  function formatNumber(value, digits = 0) {
    const number = Number(value || 0);
    return number.toLocaleString('zh-CN', {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits,
    });
  }

  function formatPercent(value, digits = 1) {
    return `${formatNumber(Number(value || 0) * 100, digits)}%`;
  }

  function secondsToTicks(value) {
    return Math.max(0, Math.round(Number(value || 0) * 10));
  }

  function ticksToSeconds(value) {
    return (Number(value || 0) / 10).toFixed(1);
  }

  function getVisitorKey() {
    let key = window.localStorage.getItem('shaft_visitor_key') || '';
    if (!key) {
      key = `visitor_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
      window.localStorage.setItem('shaft_visitor_key', key);
    }
    return key;
  }

  function readAxisDraft() {
    try {
      const raw = window.localStorage.getItem(DRAFT_STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (error) {
      return null;
    }
  }

  function axisDraftPayload() {
    if (!state.axis) {
      return null;
    }
    return {
      savedAxisId: state.savedAxisId || null,
      savedAxisTitle: state.savedAxisTitle || '',
      page: state.page || 'rotation',
      title: $('shaft-title-input')?.value || '未命名排轴',
      description: $('shaft-description-input')?.value || '',
      axis: state.axis,
      compareSnapshot: state.compareSnapshot,
      axisDocumentBaseline: state.axisDocumentBaseline,
      updatedAt: Date.now(),
    };
  }

  function persistAxisDraft() {
    if (state.sharedReadOnly) {
      return;
    }
    const payload = axisDraftPayload();
    if (!payload) {
      return;
    }
    try {
      window.localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(payload));
    } catch (error) {
      // 本地草稿只是体验增强，写入失败不阻塞编辑。
    }
  }

  function axisCalculationFingerprint(axis = state.axis) {
    try {
      return JSON.stringify(axis || null);
    } catch (error) {
      return '';
    }
  }

  function freshResult() {
    if (!state.result || state.isResultStale) {
      return null;
    }
    const currentFingerprint = axisCalculationFingerprint();
    return state.resultFingerprint === currentFingerprint ? state.result : null;
  }

  function timelineResult() {
    return freshResult() || state.dragState?.previewResult || state.dragState?.resultSnapshot || state.result || null;
  }

  function acceptSimulationResult(result) {
    state.result = result;
    state.resultFingerprint = axisCalculationFingerprint();
    state.isResultStale = false;
  }

  function markSimulationStale() {
    state.isResultStale = true;
    state.resultFingerprint = '';
  }

  function restoreAxisDraft(activePage) {
    const draft = readAxisDraft();
    if (!draft || !draft.axis || typeof draft.axis !== 'object') {
      return false;
    }
    state.axis = clone(draft.axis);
    state.compareSnapshot = draft.compareSnapshot ? clone(draft.compareSnapshot) : null;
    state.savedAxisId = Number(draft.savedAxisId || 0) || null;
    state.savedAxisTitle = String(draft.savedAxisTitle || (state.savedAxisId ? draft.title : '') || '');
    state.axisDocumentBaseline = String(draft.axisDocumentBaseline || '');
    state.page = ['build', 'rotation', 'plaza'].includes(activePage)
      ? activePage
      : (['build', 'rotation', 'plaza'].includes(draft.page) ? draft.page : 'rotation');
    $('shaft-title-input').value = draft.title || '未命名排轴';
    $('shaft-description-input').value = draft.description || '';
    return true;
  }

  async function shaftRequest(url, options = {}, meta = {}) {
    const headers = Object.assign({}, options.headers || {});
    const body = options.body;
    if (body && !headers['Content-Type']) {
      headers['Content-Type'] = 'application/json';
    }
    const token = getToken();
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    headers['X-Shaft-Visitor-Key'] = getVisitorKey();
    headers['X-Log-Id'] = typeof nextLogId === 'function' ? nextLogId() : `shaft-${Date.now()}`;
    const response = await fetch(url, Object.assign({}, options, { headers }));
    let payload = {};
    try {
      payload = await response.json();
    } catch (error) {
      payload = {};
    }
    if (response.status === 401 && meta.authRequired) {
      redirectToLogin();
      throw new Error('未登录或登录态已失效。');
    }
    if (!response.ok || payload.error) {
      const logId = response.headers.get('X-Log-Id') || headers['X-Log-Id'];
      const requestError = new Error(`${payload.error || '请求失败'}（logId: ${logId}）`);
      requestError.payload = payload;
      requestError.status = response.status;
      throw requestError;
    }
    return payload;
  }

  function getCharacterMap() {
    return new Map((state.catalog?.characters || []).map((item) => [item.id, item]));
  }

  function normalizeCharacterNameForAwakening(name) {
    const value = String(name || '').replace(/[「」]/g, '').trim();
    if (value === '零' || value === '鉴定师') {
      return '主角';
    }
    return value;
  }

  function awakeningsForCharacter(character, member) {
    const awakenings = state.catalog?.awakenings || {};
    const names = [
      character?.name,
      member?.character_name,
      character?.avatar?.includes('鉴定师') ? '主角' : '',
    ];
    for (const name of names) {
      const key = normalizeCharacterNameForAwakening(name);
      if (key && Array.isArray(awakenings[key])) {
        return awakenings[key];
      }
    }
    return [];
  }

  function awakeningTooltipText(entry, fallback = '暂无觉醒内容') {
    if (!entry) {
      return fallback;
    }
    const implementationLabel = entry.implementation_status === 'out_of_scope'
      ? '（不实装）'
      : (entry.implemented === false ? '（未实装）' : '');
    return `${entry.title || '未命名'}${implementationLabel}\n${entry.description || '暂无描述'}`;
  }

  function mechanismConfigurationConditionMatches(condition, member, awakeningNodes) {
    const type = String(condition?.type || '');
    if (type === 'awakening_min') {
      const level = Math.round(numberValue(condition.min, condition.value));
      return awakeningNodes.has(level);
    }
    if (type === 'awakening_max') {
      const nextLevel = Math.round(numberValue(condition.max, condition.value)) + 1;
      return !awakeningNodes.has(nextLevel);
    }
    if (type === 'awakening_count_min') {
      const requiredCount = Math.round(numberValue(condition.min, condition.value));
      return awakeningNodes.size >= requiredCount;
    }
    if (type === 'awakening_count_max') {
      const maxCount = Math.round(numberValue(condition.max, condition.value));
      return awakeningNodes.size <= maxCount;
    }
    if (type === 'owner_character_id') {
      return (condition.ids || []).map(String).includes(String(member?.character_id || ''));
    }
    return type !== 'unsupported';
  }

  function mechanismEffectLabel(key) {
    const labels = {
      atk_pct: '攻击',
      flat_atk: '固定攻击',
      hp_pct: '生命',
      flat_hp: '固定生命',
      def_pct: '防御',
      flat_def: '固定防御',
      crit_rate: '暴击率',
      crit_dmg: '暴击伤害',
      def_ignore: '无视防御',
      def_down: '防御降低',
      res_down: '抗性降低',
      energy_recharge: '充能效率',
      harmony_strength: '环合强度',
      stagger_strength: '倾陷强度',
      basic_dmg: '普攻伤害',
      dodge_counter_dmg: '闪反增伤',
      skill_dmg: '变轨技伤害',
      ultimate_dmg: '终结技伤害',
      follow_dmg: '追击伤害',
      mind_dmg: '心灵伤害',
      attach_dmg: '附着伤害',
      element_dmg: '属性伤害',
      all_dmg: '伤害',
      final_dmg: '最终伤害',
      base_multiplier_pct: '基础倍率',
    };
    return labels[key] || (String(key).startsWith('res_down_') ? `${String(key).slice('res_down_'.length)}属性抗性降低` : '');
  }

  function mechanismEffectValue(key, value) {
    const flatKeys = new Set(['flat_atk', 'flat_hp', 'flat_def', 'harmony_strength', 'stagger_strength']);
    return flatKeys.has(key)
      ? formatNumber(value, Number.isInteger(Number(value)) ? 0 : 1)
      : `${formatNumber(Number(value) * 100, 1)}%`;
  }

  function mechanismTriggerText(rule) {
    const trigger = rule?.trigger || {};
    const source = trigger.source || {};
    const actionTypeLabels = { E: '变轨技', Q: '终结技', '普攻': '普攻', '援护': '援护技' };
    const actionTypes = (source.action_types || []).map((type) => actionTypeLabels[type] || type);
    const actionNames = source.action_names || [];
    const actionLabel = [...actionTypes, ...actionNames].join('或');
    if (trigger.event === 'passive') return '常驻';
    if (trigger.event === 'front_time') return '前台驻场时';
    if (trigger.event === 'reaction_trigger') {
      const reactions = (trigger.conditions || [])
        .filter((condition) => condition?.type === 'reaction_type')
        .flatMap((condition) => condition.reactions || []);
      return reactions.length ? `触发${reactions.join('或')}时` : '触发环合时';
    }
    if (trigger.event === 'action_hit') return actionLabel ? `${actionLabel}命中后` : '攻击命中后';
    if (trigger.event === 'action_end') return actionLabel ? `${actionLabel}结束后` : '动作结束后';
    if (trigger.event === 'action_start') return actionLabel ? `施放${actionLabel}时` : '满足动作条件时';
    if (trigger.event === 'full_stack') return '叠满层时';
    return '';
  }

  function mechanismTargetText(rule) {
    const target = rule?.target || {};
    const scopeLabels = {
      team: '全队',
      other_team: '其他队友',
      front: '前台角色',
      front_registrar: '自身在前台时',
      front_non_registrar: '其他前台角色',
    };
    const parts = [];
    if (scopeLabels[target.scope]) parts.push(scopeLabels[target.scope]);
    if ((target.elements || []).length) parts.push(`${target.elements.join('或')}属性`);
    if ((target.action_types || []).length) parts.push(target.action_types.join('或'));
    if ((target.action_names || []).length) parts.push(target.action_names.join('或'));
    if ((target.tags || []).length) parts.push(target.tags.join('或'));
    return parts.join('');
  }

  function mechanismRuleSummary(rule) {
    if (rule?.description) {
      return String(rule.description);
    }
    const stacking = rule?.stacking || {};
    const maxStacks = Math.max(1, Math.round(numberValue(stacking.max_stacks, 1)));
    const effects = Object.entries(rule?.effects || {})
      .filter(([key, value]) => mechanismEffectLabel(key) && Number(value))
      .map(([key, value]) => {
        const amount = mechanismEffectValue(key, value);
        return stacking.mode === 'add_stack' || stacking.mode === 'independent'
          ? `${mechanismEffectLabel(key)}每层${Number(value) >= 0 ? '+' : ''}${amount}`
          : `${mechanismEffectLabel(key)}${Number(value) >= 0 ? '+' : ''}${amount}`;
      });
    const dynamicEffects = Object.values(rule?.dynamic_effects || {}).map((dynamic) => {
      const key = dynamic?.effect_key || '';
      const perValue = numberValue(dynamic?.per_count, dynamic?.per_interval);
      const maximum = numberValue(dynamic?.max_count);
      if (!mechanismEffectLabel(key) || !perValue) return '';
      return `${mechanismEffectLabel(key)}每层${perValue >= 0 ? '+' : ''}${mechanismEffectValue(key, perValue)}${maximum > 1 ? `，最多${formatNumber(maximum, 0)}层` : ''}`;
    }).filter(Boolean);
    const parts = [mechanismTriggerText(rule), mechanismTargetText(rule), [...effects, ...dynamicEffects].join('、')].filter(Boolean);
    if (maxStacks > 1 && effects.length) {
      parts.push(`最多${maxStacks}层`);
    }
    const duration = rule?.duration || {};
    const durationTicks = Math.max(0, numberValue(duration.ticks));
    if (duration.type === 'time' && durationTicks > 1 && durationTicks < BUFF_DURATION_LABEL_LIMIT_TICKS) {
      parts.push(`持续${formatNumber(durationTicks / 10, durationTicks % 10 ? 1 : 0)}秒`);
    }
    return parts.join('，') || '按该机制规则触发';
  }

  function activeMechanismGroups(member) {
    const selectedProviders = {
      character: String(member?.character_id || ''),
      arc: String(member?.arc_id || ''),
      cartridge: String(member?.cartridge_id || ''),
    };
    const awakeningNodes = new Set(normalizeAwakeningNodes(member?.awakening_nodes, member?.awakening));
    const rulesByKind = { character: [], arc: [], cartridge: [] };
    const characterMechanisms = (state.catalog?.mechanisms || [])
      .filter((mechanism) => String(mechanism?.character_id || '') === selectedProviders.character)
      .filter((mechanism) => (mechanism?.conditions || []).every(
        (condition) => mechanismConfigurationConditionMatches(condition, member, awakeningNodes)
      ));
    const replacedRuleIds = new Set(characterMechanisms.flatMap(
      (mechanism) => mechanism?.replaces_rule_ids || []
    ).map(String));
    characterMechanisms.forEach((mechanism) => {
      rulesByKind.character.push({
        ...clone(mechanism),
        name: `天赋${mechanism.talent === 1 ? '一' : '二'} · ${mechanism.name || '未命名'}`,
      });
    });
    (state.catalog?.buffs || []).forEach((rule) => {
      if (replacedRuleIds.has(String(rule?.id || ''))) {
        return;
      }
      const provider = (rule?.providers || []).find((item) => {
        const kind = String(item?.kind || '');
        return Object.prototype.hasOwnProperty.call(selectedProviders, kind) &&
          String(item?.id || '') === selectedProviders[kind];
      });
      if (!provider) {
        return;
      }
      const conditions = [
        ...(rule?.trigger?.conditions || []),
        ...(rule?.target?.conditions || []),
      ];
      if (!conditions.every((condition) => mechanismConfigurationConditionMatches(condition, member, awakeningNodes))) {
        return;
      }
      const activeRule = clone(rule);
      if (String(provider.kind) === 'arc') {
        const refinedEffects = arcRefinementRecord(member)?.buff_effects?.[String(rule.id || '')];
        if (refinedEffects && typeof refinedEffects === 'object') {
          activeRule.effects = clone(refinedEffects);
        }
      }
      rulesByKind[String(provider.kind)].push(activeRule);
    });

    const sourceNames = {
      character: getCharacterMap().get(member?.character_id)?.name || member?.character_name || '角色',
      arc: getArcMap().get(member?.arc_id)?.name || '弧盘',
      cartridge: getCartridgeMap().get(member?.cartridge_id)?.name || '卡带',
    };
    const sourceLabels = { character: '角色', arc: '弧盘', cartridge: '卡带' };
    return ['character', 'arc', 'cartridge']
      .map((kind) => ({
        kind,
        label: sourceLabels[kind],
        sourceName: sourceNames[kind],
        rules: rulesByKind[kind],
      }))
      .filter((group) => group.rules.length);
  }

  function activeMechanismTooltipHtml(member, character) {
    const groups = activeMechanismGroups(member);
    const total = groups.reduce((sum, group) => sum + group.rules.length, 0);
    const groupHtml = groups.map((group) => `
      <span class="shaft-mechanism-group">
        <strong>${escapeHtml(group.label)} · ${escapeHtml(group.sourceName)}</strong>
        ${group.rules.map((rule) => `
          <span class="shaft-mechanism-item">
            <b>• ${escapeHtml(rule.name || rule.id || '未命名机制')}</b>
            <small>${escapeHtml(mechanismRuleSummary(rule))}</small>
          </span>
        `).join('')}
      </span>
    `).join('');
    const name = character?.name || member?.character_name || '该角色';
    return `
      <button class="shaft-mechanism-tooltip" type="button"
        aria-label="查看${escapeHtml(name)}当前激活机制，共${total}项"
        aria-describedby="shaft-mechanisms-${Number(member?.slot || 0)}">
        <span class="shaft-mechanism-icon" aria-hidden="true">!</span>
        <span class="shaft-mechanism-popover" id="shaft-mechanisms-${Number(member?.slot || 0)}" role="tooltip">
          <strong class="shaft-mechanism-title">当前激活机制 · ${total}</strong>
          ${groupHtml || '<span class="shaft-mechanism-empty">当前配装没有启用 buff 机制</span>'}
        </span>
      </button>
    `;
  }

  function getActionMap() {
    return new Map((state.catalog?.actions || []).map((item) => [item.id, item]));
  }

  function getArcMap() {
    return new Map((state.catalog?.arcs || []).map((item) => [item.id, item]));
  }

  function getCartridgeMap() {
    return new Map((state.catalog?.cartridges || []).map((item) => [item.id, item]));
  }

  function numberValue(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function emptyPanelMods() {
    return {
      crit_rate: 0,
      crit_dmg: 0,
      atk_pct: 0,
      flat_atk: 0,
      hp_pct: 0,
      flat_hp: 0,
      def_pct: 0,
      flat_def: 0,
      def_ignore: 0,
      def_down: 0,
      res_down: 0,
      energy_recharge: 0,
      harmony_strength: 0,
      stagger_strength: 0,
      basic_dmg: 0,
      element_dmg: 0,
      follow_dmg: 0,
      mind_dmg: 0,
      attach_dmg: 0,
      skill_dmg: 0,
      ultimate_dmg: 0,
      all_dmg: 0,
      final_dmg: 0,
    };
  }

  function mergePanelMods(base, delta, factor = 1) {
    Object.entries(delta || {}).forEach(([key, value]) => {
      if (Object.prototype.hasOwnProperty.call(base, key)) {
        base[key] += numberValue(value) * factor;
      }
    });
    return base;
  }

  function substatPanelMods(counts) {
    const mods = emptyPanelMods();
    const units = state.catalog?.formula_constants?.substat_units || {};
    SUBSTAT_ORDER.forEach((key) => {
      const unit = units[key] || {};
      if (Object.prototype.hasOwnProperty.call(mods, key)) {
        mods[key] += Math.max(0, numberValue(counts?.[key])) * numberValue(unit.unit_value);
      }
    });
    return mods;
  }

  function normalizeTeamPanelBonus(raw) {
    const source = raw && typeof raw === 'object' ? raw : {};
    const version = Math.max(0, Math.round(numberValue(source.version)));
    const legacyFurniture = version < DEFAULT_TEAM_PANEL_BONUS.version;
    const furnitureCritDmg = legacyFurniture
      ? DEFAULT_TEAM_PANEL_BONUS.furniture_crit_dmg
      : Math.round(Math.max(0, Math.min(DEFAULT_TEAM_PANEL_BONUS.furniture_crit_dmg, numberValue(source.furniture_crit_dmg, DEFAULT_TEAM_PANEL_BONUS.furniture_crit_dmg))) * 1000) / 1000;
    const furnitureFlatAtk = legacyFurniture
      ? DEFAULT_TEAM_PANEL_BONUS.furniture_flat_atk
      : Math.round(Math.max(0, Math.min(DEFAULT_TEAM_PANEL_BONUS.furniture_flat_atk, numberValue(source.furniture_flat_atk, DEFAULT_TEAM_PANEL_BONUS.furniture_flat_atk))));
    const furnitureFlatDef = legacyFurniture
      ? DEFAULT_TEAM_PANEL_BONUS.furniture_flat_def
      : Math.round(Math.max(0, Math.min(DEFAULT_TEAM_PANEL_BONUS.furniture_flat_def, numberValue(source.furniture_flat_def, DEFAULT_TEAM_PANEL_BONUS.furniture_flat_def))));
    let smallFlatAtk = Math.max(0, Math.min(5000, numberValue(source.small_flat_atk, DEFAULT_TEAM_PANEL_BONUS.small_flat_atk)));
    if (version < 2 && furnitureFlatAtk === 20 && smallFlatAtk === 440) {
      smallFlatAtk = 420;
    }
    return {
      version: DEFAULT_TEAM_PANEL_BONUS.version,
      furniture_crit_dmg: furnitureCritDmg,
      furniture_flat_atk: furnitureFlatAtk,
      furniture_flat_def: furnitureFlatDef,
      small_flat_atk: smallFlatAtk,
      small_flat_hp: Math.max(0, Math.min(100000, numberValue(source.small_flat_hp, DEFAULT_TEAM_PANEL_BONUS.small_flat_hp))),
    };
  }

  function teamPanelBonusMods(raw = state.axis?.team_panel_bonus) {
    const bonus = normalizeTeamPanelBonus(raw);
    const mods = emptyPanelMods();
    mods.crit_dmg += bonus.furniture_crit_dmg;
    mods.flat_atk += bonus.furniture_flat_atk + bonus.small_flat_atk;
    mods.flat_def += bonus.furniture_flat_def;
    mods.flat_hp += bonus.small_flat_hp;
    return mods;
  }

  function buildOptionDefaults(characterId) {
    const defaults = state.catalog?.formula_constants?.default_build_options || {};
    return defaults[characterId] && typeof defaults[characterId] === 'object' ? defaults[characterId] : {};
  }

  function normalizeStatName(value) {
    const text = String(value || '').trim();
    return text === '环合强度' ? '精通' : text;
  }

  function mainStatOptions() {
    return state.catalog?.formula_constants?.cartridge_main_stat_options || {};
  }

  function curtainStatOptions() {
    return state.catalog?.formula_constants?.curtain_bonus_stat_options || {};
  }

  function normalizeCartridgeMainStat(raw, characterId = '') {
    const options = mainStatOptions();
    const defaultValue = normalizeStatName(buildOptionDefaults(characterId).cartridge_main_stat) || Object.keys(options)[0] || '';
    const selected = normalizeStatName(raw) || defaultValue;
    return Object.prototype.hasOwnProperty.call(options, selected) ? selected : defaultValue;
  }

  function normalizeCurtainBonus(raw, characterId = '') {
    const defaults = buildOptionDefaults(characterId).curtain_bonus || {};
    const source = raw && typeof raw === 'object' ? raw : {};
    const options = curtainStatOptions();
    const defaultStat = normalizeStatName(defaults.stat) || Object.keys(options)[0] || '';
    const stat = normalizeStatName(source.stat) || defaultStat;
    const passiveType = CURTAIN_PASSIVE_TYPES.some((item) => item.key === source.passive_type)
      ? source.passive_type
      : (CURTAIN_PASSIVE_TYPES.some((item) => item.key === defaults.passive_type) ? defaults.passive_type : 'type3');
    return {
      value: Math.max(0, Math.min(100, numberValue(defaults.value))),
      stat: Object.prototype.hasOwnProperty.call(options, stat) ? stat : defaultStat,
      passive_type: passiveType,
    };
  }

  function curtainPassiveLayers(member) {
    const cartridge = getCartridgeMap().get(member?.cartridge_id || '');
    const bonus = normalizeCurtainBonus(member?.curtain_bonus, member?.character_id || '');
    return Math.max(0, Math.round(numberValue(cartridge?.passive_counts?.[bonus.passive_type])));
  }

  function mainStatPanelMods(mainStat) {
    const mods = emptyPanelMods();
    const option = mainStatOptions()[mainStat] || {};
    const key = option.modifier_key || '';
    if (Object.prototype.hasOwnProperty.call(mods, key)) {
      mods[key] += numberValue(option.unit_value);
    }
    return mods;
  }

  function curtainBonusPanelMods(member) {
    const mods = emptyPanelMods();
    const bonus = normalizeCurtainBonus(member?.curtain_bonus, member?.character_id || '');
    const option = curtainStatOptions()[bonus.stat] || {};
    const key = option.modifier_key || '';
    if (Object.prototype.hasOwnProperty.call(mods, key)) {
      mods[key] += numberValue(bonus.value) / 100 * curtainPassiveLayers(member);
    }
    return mods;
  }

  function curtainBonusResultText(member) {
    const bonus = normalizeCurtainBonus(member?.curtain_bonus, member?.character_id || '');
    const option = curtainStatOptions()[bonus.stat] || {};
    const layers = curtainPassiveLayers(member);
    const value = numberValue(bonus.value) / 100 * layers;
    const key = String(option.modifier_key || '');
    const isPercent = option.kind === 'percent' || key.endsWith('_pct') || key.includes('dmg') || key.includes('crit');
    return {
      stat: option.label || bonus.stat || '空幕',
      passive: CURTAIN_PASSIVE_TYPES.find((item) => item.key === bonus.passive_type)?.label || bonus.passive_type,
      layers,
      display: isPercent ? `+${formatNumber(value * 100, 1)}%` : `+${formatNumber(value, 0)}`,
    };
  }

  function computeBuildPanelForMember(member) {
    if (!member || !state.catalog) {
      return null;
    }
    const character = getCharacterMap().get(member.character_id);
    if (!character) {
      return null;
    }
    const arc = getArcMap().get(member.arc_id);
    const arcRefinement = arcRefinementRecord(member);
    const cartridge = getCartridgeMap().get(member.cartridge_id);
    const mods = emptyPanelMods();
    // Character-sheet bonuses are intentionally excluded. Passive and awakening
    // effects come from the buff registry; every character starts at 5% / 50%.
    mods.crit_rate = 0.05;
    mods.crit_dmg = 0.5;
    if (member.bond_full || numberValue(member.bond_level) > 0) {
      mergePanelMods(mods, character.bond_bonus?.modifiers);
    }
    if (arc) {
      mergePanelMods(mods, arcRefinement?.panel_modifiers || arc.modifiers);
    }
    if (cartridge) {
      const cartridgeModifiers = { ...(cartridge.modifiers || {}) };
      const requiredElement = String(cartridge.required_element || '');
      if (requiredElement && requiredElement !== String(character.element || '')) {
        cartridgeModifiers.element_dmg = 0;
      }
      mergePanelMods(mods, cartridgeModifiers);
    }
    mergePanelMods(mods, mainStatPanelMods(normalizeCartridgeMainStat(member.cartridge_main_stat, member.character_id)));
    mergePanelMods(mods, curtainBonusPanelMods(member));
    mergePanelMods(mods, substatPanelMods(member.substat_counts));
    mergePanelMods(mods, teamPanelBonusMods());
    const element = character.element || '';
    mods.element_dmg += numberValue((arcRefinement?.element_dmg || arc?.element_dmg)?.[element]);

    const baseStats = character.base_stats || {};
    const baseAtk = numberValue(baseStats.atk) + numberValue(arc?.base_atk);
    const baseHp = numberValue(baseStats.hp);
    const baseDef = numberValue(baseStats.def);
    const panel = {
      atk: baseAtk * (1 + mods.atk_pct) + mods.flat_atk,
      hp: baseHp * (1 + mods.hp_pct) + mods.flat_hp,
      def: baseDef * (1 + mods.def_pct) + mods.flat_def,
      crit_rate: mods.crit_rate,
      crit_dmg: mods.crit_dmg,
      element_dmg: mods.element_dmg,
      energy_recharge: mods.energy_recharge,
      harmony_strength: mods.harmony_strength,
      stagger_strength: mods.stagger_strength,
      all_dmg: mods.all_dmg,
    };
    return {
      slot: Number(member.slot || 0),
      character_id: character.id || '',
      character_name: character.name || member.character_name || '',
      base_stats: { atk: baseAtk, hp: baseHp, def: baseDef },
      zones: {
        atk_pct: mods.atk_pct,
        flat_atk: mods.flat_atk,
        hp_pct: mods.hp_pct,
        flat_hp: mods.flat_hp,
        def_pct: mods.def_pct,
        flat_def: mods.flat_def,
      },
      panel,
      build_options: {
        cartridge_main_stat: normalizeCartridgeMainStat(member.cartridge_main_stat, member.character_id),
        curtain_bonus: Object.assign({}, normalizeCurtainBonus(member.curtain_bonus, member.character_id), {
          layers: curtainPassiveLayers(member),
        }),
      },
    };
  }

  function currentBuildPanels() {
    const team = state.axis?.team || [];
    const computed = team.map(computeBuildPanelForMember).filter(Boolean);
    if (computed.length) {
      return computed;
    }
    return freshResult()?.build_panels_by_slot || [];
  }

  function memberBySlot(slot) {
    return (state.axis?.team || []).find((item) => Number(item.slot) === Number(slot));
  }

  function validStepIdSet() {
    return new Set((state.axis?.steps || []).map((step) => step.id));
  }

  function selectedStepIds() {
    const validIds = validStepIdSet();
    const ids = Array.isArray(state.selectedStepIds) ? state.selectedStepIds : [];
    return ids.filter((id, index) => validIds.has(id) && ids.indexOf(id) === index);
  }

  function selectedSteps() {
    const ids = new Set(selectedStepIds());
    return (state.axis?.steps || []).filter((step) => ids.has(step.id));
  }

  function syncSelection(fallbackToFirst = false) {
    const ids = selectedStepIds();
    if (state.selectedStepId && validStepIdSet().has(state.selectedStepId) && !ids.includes(state.selectedStepId)) {
      ids.push(state.selectedStepId);
    }
    state.selectedStepIds = ids;
    if (!ids.includes(state.selectedStepId)) {
      state.selectedStepId = ids[ids.length - 1] || (fallbackToFirst ? state.axis?.steps?.[0]?.id || '' : '');
    }
    if (state.selectedStepId && !state.selectedStepIds.includes(state.selectedStepId)) {
      state.selectedStepIds.push(state.selectedStepId);
    }
  }

  function setSelectedStepIds(ids, primaryId = '', render = true) {
    const validIds = validStepIdSet();
    state.selectedStepIds = (ids || []).filter((id, index) => validIds.has(id) && (ids || []).indexOf(id) === index);
    state.selectedStepId = validIds.has(primaryId) ? primaryId : state.selectedStepIds[state.selectedStepIds.length - 1] || '';
    if (state.selectedStepId && !state.selectedStepIds.includes(state.selectedStepId)) {
      state.selectedStepIds.push(state.selectedStepId);
    }
    const step = selectedStep();
    if (step) {
      state.cursorTick = timelineDisplayTickForStep(step);
      syncAddTimeInput(state.cursorTick);
    }
    if (render) {
      renderTeamDock();
      renderSteps();
      renderTimeline();
      renderStepDetail();
      renderEditorActions();
    } else {
      updateSelectionClasses();
    }
  }

  function selectedStep() {
    return (state.axis?.steps || []).find((step) => step.id === state.selectedStepId) || null;
  }

  function timelineDisplayTickForStep(step) {
    const detail = (state.timelineDisplayDetails || []).find((item) => item.step_id === step?.id) ||
      (timelineResult()?.details || []).find((item) => item.step_id === step?.id);
    return Math.max(0, Number(
      detail?.display_start_tick ??
      detail?.visual_start_tick ??
      step?.start_tick ??
      0
    ));
  }

  function detailByStepId(stepId) {
    return (freshResult()?.details || []).find((detail) => detail.step_id === stepId) || null;
  }

  function slotResourcesAtCursor(slot) {
    const slotNumber = Number(slot);
    const result = timelineResult();
    const resource = (result?.resources_by_slot || []).find((item) => Number(item.slot) === slotNumber) || {};
    const cursorTick = Number(state.cursorTick || 0);
    const details = state.timelineDisplayDetails?.length ? state.timelineDisplayDetails : (result?.details || []);
    const latestDetail = details.reduce((latest, detail) => {
      const detailTick = Number(detail.display_start_tick ?? detail.visual_start_tick ?? detail.start_tick ?? 0);
      if (detailTick > cursorTick) {
        return latest;
      }
      if (!latest) {
        return detail;
      }
      const latestTick = Number(latest.display_start_tick ?? latest.visual_start_tick ?? latest.start_tick ?? 0);
      if (detailTick > latestTick) {
        return detail;
      }
      if (detailTick === latestTick && Number(detail.resource_sequence ?? -1) > Number(latest.resource_sequence ?? -1)) {
        return detail;
      }
      return latest;
    }, null);
    const snapshot = (latestDetail?.resources_after_by_slot || [])
      .find((item) => Number(item.slot) === slotNumber) || {};
    const calculationCursorTick = calculationTickFromVisual(cursorTick);
    const latestEnergyEvent = (result?.energy_events || []).reduce((latest, event) => {
      if (Number(event.slot) !== slotNumber || Number(event.tick || 0) > calculationCursorTick) {
        return latest;
      }
      return event;
    }, null);
    const energy = Number(latestEnergyEvent?.energy_after ?? resource.initial_energy ?? snapshot.energy ?? state.axis?.initial_energy ?? 1000);
    const harmony = Number(snapshot.harmony ?? resource.initial_harmony ?? 0);
    const personalResources = snapshot.personal_resources && typeof snapshot.personal_resources === 'object'
      ? snapshot.personal_resources
      : resource.initial_personal_resources && typeof resource.initial_personal_resources === 'object'
        ? resource.initial_personal_resources
      : {};
    return { energy, harmony, personalResources };
  }

  function timelinePersonalResourceEntries(character, personalResources) {
    const resources = personalResources && typeof personalResources === 'object'
      ? personalResources
      : {};
    const fixedNames = TIMELINE_FIXED_PERSONAL_RESOURCES[String(character?.id || '')] || [];
    const entries = fixedNames.map((name) => [name, Number(resources[name] || 0)]);
    const fixedNameSet = new Set(fixedNames);
    Object.entries(resources).forEach(([name, value]) => {
      if (
        fixedNameSet.has(name) ||
        TIMELINE_HIDDEN_PERSONAL_RESOURCES.has(name) ||
        !Number(value)
      ) {
        return;
      }
      entries.push([name, Number(value)]);
    });
    return entries;
  }

  function memberName(slot) {
    const member = memberBySlot(slot);
    return member?.character_name || getCharacterMap().get(member?.character_id)?.name || `角色 ${Number(slot) + 1}`;
  }

  function actionsForSlot(slot) {
    const member = memberBySlot(slot);
    if (!member) {
      return [];
    }
    return state.catalog?.actions_by_character?.[member.character_id] || [];
  }

  function isBackgroundAction(action) {
    const marker = `${action?.name || ''} ${action?.extra_tag || ''}`;
    return Boolean(action?.is_background_damage) || marker.includes('后台');
  }

  function backgroundActionMultiplier(step, action = actionForStep(step)) {
    return isBackgroundAction(action)
      ? Math.max(1, Math.min(MAX_BACKGROUND_ACTION_MULTIPLIER, Math.round(Number(step?.repeat || 1))))
      : 1;
  }

  function isBasicAction(action) {
    return String(action?.action_type || '') === '普攻' || String(action?.damage_type || '') === '普攻';
  }

  function canBackgroundOverride(action) {
    return Boolean(action?.can_background_override) && isBasicAction(action);
  }

  function isBasicBackgroundOverride(step, action = actionForStep(step)) {
    return !isBackgroundAction(action) && canBackgroundOverride(action) && step?.placement === 'background';
  }

  function isStepBackground(step, action = actionForStep(step)) {
    return isBackgroundAction(action) || isBasicBackgroundOverride(step, action);
  }

  function startsForeground(step, action = actionForStep(step)) {
    return !isStepBackground(step, action);
  }

  function blocksSlotOverlap(step, action = actionForStep(step)) {
    return startsForeground(step, action) || isBasicBackgroundOverride(step, action);
  }

  function sanitizeStepPlacement(step) {
    if (!step) {
      return;
    }
    const action = actionForStep(step);
    if (!isBasicBackgroundOverride(step, action)) {
      delete step.placement;
    }
    step.repeat = backgroundActionMultiplier(step, action);
  }

  function isSupportAction(action) {
    return String(action?.action_type || '') === '援护';
  }

  function isInstantNativeBackgroundAction(step, action = actionForStep(step)) {
    return isBackgroundAction(action) && !isSupportAction(action) && !Boolean(action?.pre_input_node);
  }

  function isQAction(action) {
    return String(action?.action_type || '') === 'Q' || String(action?.damage_type || '') === 'Q';
  }

  function isInstantSwitchAction(action) {
    return Boolean(action?.is_instant_switch);
  }

  function tickHasInstantSwitchAction(tick, ignoreStepId = '') {
    const target = Number(tick || 0);
    return (state.axis?.steps || []).some((step) => (
      step.id !== ignoreStepId &&
      Number(step.start_tick || 0) === target &&
      isInstantSwitchAction(actionForStep(step))
    ));
  }

  function tickHasForegroundQ(tick, ignoreStepId = '') {
    const target = Number(tick || 0);
    return (state.axis?.steps || []).some((step) => {
      if (step.id === ignoreStepId || Number(step.start_tick || 0) !== target) {
        return false;
      }
      const action = actionForStep(step);
      return isZeroForegroundQStep(step, action);
    });
  }

  function actionDurationTicks(action) {
    return Math.max(0, Number(action?.duration_ticks || 0));
  }

  function actionCalculationDurationTicks(action, step = null) {
    return isInstantNativeBackgroundAction(step, action) ? 0 : actionDurationTicks(action);
  }

  function actionEditorDurationTicks(action, step = null) {
    if (isInstantNativeBackgroundAction(step, action)) {
      return ZERO_ACTION_VISUAL_TICKS;
    }
    return Math.max(1, actionDurationTicks(action));
  }

  function qVisualDurationTicks(action) {
    const durationTicks = actionDurationTicks(action);
    return durationTicks > 0 ? durationTicks : ZERO_ACTION_VISUAL_TICKS;
  }

  function actionVisualDurationTicks(action, step = null) {
    if (step && isZeroForegroundQStep(step, action)) {
      return qVisualDurationTicks(action);
    }
    return actionEditorDurationTicks(action, step);
  }

  function isZeroForegroundQStep(step, action = actionForStep(step)) {
    return startsForeground(step, action) && isQAction(action) && actionDurationTicks(action) === 0;
  }

  function locksForegroundSwitch(step, action = actionForStep(step)) {
    return startsForeground(step, action) && (
      isSupportAction(action) ||
      isZeroForegroundQStep(step, action)
    );
  }

  function foregroundLockEndTick(step, action, startTick) {
    if (isSupportAction(action)) {
      return startTick + actionVisualDurationTicks(action, step);
    }
    if (!isZeroForegroundQStep(step, action)) {
      return startTick;
    }
    const detail = (timelineResult()?.details || []).find((item) => item.step_id === step?.id);
    const qSpanTicks = resultDetailMatchesStep(detail, step)
      ? Math.max(
        ZERO_ACTION_VISUAL_TICKS,
        Number(detail.display_visual_end_tick ?? detail.visual_end_tick ?? startTick + ZERO_ACTION_VISUAL_TICKS) -
          Number(detail.display_start_tick ?? detail.visual_start_tick ?? startTick),
      )
      : ZERO_ACTION_VISUAL_TICKS;
    return startTick + qSpanTicks;
  }

  function qVirtualStartTicks(steps = state.axis?.steps || []) {
    const frozenIntervals = timelineResult()?.time_axis?.frozen_intervals;
    if (Array.isArray(frozenIntervals)) {
      return frozenIntervals.map((interval) => ({
        start_tick: Math.max(0, Number(interval?.start_tick || 0)),
        end_tick: Math.max(0, Number(interval?.end_tick || 0)),
      }));
    }
    const resultDetails = timelineResult()?.details || [];
    const resultDetailByStepId = new Map(resultDetails.map((detail) => [detail.step_id, detail]));
    return (steps || [])
      .map((step, order) => ({ step, order, action: actionForStep(step) }))
      .filter((item) => isZeroForegroundQStep(item.step, item.action))
      .sort((a, b) => Number(a.step.start_tick || 0) - Number(b.step.start_tick || 0) || a.order - b.order)
      .map((item) => {
        const detail = resultDetailByStepId.get(item.step.id);
        const startTick = resultDetailMatchesStep(detail, item.step)
          ? Math.max(0, Number(detail.display_start_tick ?? detail.visual_start_tick ?? item.step.start_tick ?? 0))
          : Math.max(0, Number(item.step.start_tick || 0));
        const visualEndTick = resultDetailMatchesStep(detail, item.step)
          ? Math.max(startTick + ZERO_ACTION_VISUAL_TICKS, Number(detail.display_visual_end_tick ?? detail.visual_end_tick ?? startTick + ZERO_ACTION_VISUAL_TICKS))
          : startTick + ZERO_ACTION_VISUAL_TICKS;
        return {
          start_tick: startTick,
          end_tick: visualEndTick,
        };
      });
  }

  function qVirtualStartTick(qInterval) {
    return typeof qInterval === 'number' ? Number(qInterval || 0) : Number(qInterval?.start_tick || 0);
  }

  function qVirtualEndTick(qInterval) {
    if (typeof qInterval === 'number') {
      return Number(qInterval || 0) + ZERO_ACTION_VISUAL_TICKS;
    }
    return Math.max(qVirtualStartTick(qInterval), Number(qInterval?.end_tick ?? qVirtualStartTick(qInterval) + ZERO_ACTION_VISUAL_TICKS));
  }

  function calculationTickFromVisual(visualTick, qStarts = qVirtualStartTicks()) {
    const safeTick = Math.max(0, Number(visualTick || 0));
    const offset = qStarts
      .filter((qStartTick) => qVirtualStartTick(qStartTick) < safeTick)
      .reduce((sum, qStartTick) => {
        const startTick = qVirtualStartTick(qStartTick);
        const endTick = qVirtualEndTick(qStartTick);
        return sum + Math.min(Math.max(0, endTick - startTick), safeTick - startTick);
      }, 0);
    return Math.max(0, safeTick - offset);
  }

  function visualTickParts(visualTick, qStarts = qVirtualStartTicks()) {
    const safeTick = Math.max(0, Number(visualTick || 0));
    const calculationTick = calculationTickFromVisual(safeTick, qStarts);
    const qStartsBefore = qStarts.filter((qStartTick) => (
      qVirtualStartTick(qStartTick) < safeTick &&
      calculationTickFromVisual(qVirtualStartTick(qStartTick), qStarts) === calculationTick
    )).length;
    const qStartsAtTick = qStarts.filter((qStartTick) => (
      qVirtualStartTick(qStartTick) === safeTick &&
      calculationTickFromVisual(qVirtualStartTick(qStartTick), qStarts) === calculationTick
    )).length;
    return {
      visualTick: safeTick,
      calculationTick,
      qSequence: qStartsBefore + 1,
      hasQSequence: qStartsBefore > 0 || qStartsAtTick > 0,
    };
  }

  function visualTickLabel(visualTick, qStarts = qVirtualStartTicks()) {
    const parts = visualTickParts(visualTick, qStarts);
    return `${ticksToSeconds(parts.calculationTick)}s`;
  }

  function syncAddTimeInput(tick) {
    const addTime = $('shaft-add-time');
    if (!addTime) {
      return;
    }
    addTime.dataset.visualTick = String(Math.max(0, Number(tick || 0)));
    addTime.dataset.calculationTick = String(calculationTickFromVisual(tick));
    addTime.title = visualTickLabel(tick);
    if (addTime.type === 'number') {
      addTime.value = ticksToSeconds(calculationTickFromVisual(tick));
    } else {
      addTime.value = visualTickLabel(tick);
    }
    addTime.dataset.syncedValue = String(addTime.value || '');
  }

  function tickFromTimeInput(input) {
    if (!input) {
      return 0;
    }
    const rawValue = String(input.value || '').trim();
    if (
      input.dataset.visualTick !== undefined &&
      (
        rawValue === String(input.dataset.syncedValue || '') ||
        rawValue === ticksToSeconds(Number(input.dataset.calculationTick || 0))
      )
    ) {
      return Math.max(0, Number(input.dataset.visualTick || 0));
    }
    return secondsToTicks(rawValue);
  }

  function actionForStep(step) {
    return getActionMap().get(step?.action_id || '') || {};
  }

  function stepStartTick(step, visual = false, qStarts = qVirtualStartTicks()) {
    const startTick = Number(step?.start_tick || 0);
    return visual ? startTick : calculationTickFromVisual(startTick, qStarts);
  }

  function stepEndTick(step, visual = false, qStarts = qVirtualStartTicks()) {
    const action = actionForStep(step);
    const duration = actionDurationTicks(action);
    if (visual) {
      return Number(step?.start_tick || 0) + actionVisualDurationTicks(action, step);
    }
    return stepStartTick(step, false, qStarts) + duration;
  }

  function axisEndTick(visual = false) {
    const qStarts = qVirtualStartTicks();
    const foregroundSteps = (state.axis?.steps || []).filter((step) => startsForeground(step, actionForStep(step)));
    return Math.max(
      0,
      ...foregroundSteps.map((step) => stepEndTick(step, visual, qStarts)),
      ...foregroundSteps.map((step) => stepStartTick(step, visual, qStarts)),
    );
  }

  function updateAxisDuration() {
    if (state.axis) {
      state.axis.duration_ticks = Math.max(0, axisEndTick(false));
    }
  }

  function editorSnapshot() {
    return {
      steps: clone(state.axis?.steps || []),
      selectedStepId: state.selectedStepId,
      selectedStepIds: clone(selectedStepIds()),
      cursorTick: state.cursorTick,
    };
  }

  function restoreEditorSnapshot(snapshot) {
    if (!snapshot || !state.axis) {
      return;
    }
    state.axis.steps = clone(snapshot.steps || []);
    state.selectedStepIds = clone(snapshot.selectedStepIds || (snapshot.selectedStepId ? [snapshot.selectedStepId] : []));
    state.selectedStepId = snapshot.selectedStepId || state.selectedStepIds[state.selectedStepIds.length - 1] || '';
    syncSelection(true);
    state.cursorTick = Number(snapshot.cursorTick || 0);
    syncAddTimeInput(state.cursorTick);
    updateAxisDuration();
    closeContextMenu();
    renderAll();
    scheduleSimulation();
    revealTimelineTick(state.cursorTick);
  }

  function pushUndoSnapshot() {
    if (!state.axis) {
      return;
    }
    state.undoStack.push(editorSnapshot());
    if (state.undoStack.length > 80) {
      state.undoStack.shift();
    }
    state.redoStack = [];
  }

  function undoLastEdit() {
    const snapshot = state.undoStack.pop();
    if (!snapshot) {
      return;
    }
    state.redoStack.push(editorSnapshot());
    restoreEditorSnapshot(snapshot);
  }

  function redoLastEdit() {
    const snapshot = state.redoStack.pop();
    if (!snapshot) {
      return;
    }
    state.undoStack.push(editorSnapshot());
    restoreEditorSnapshot(snapshot);
  }

  function renderEditorActions() {
    const undoButton = $('shaft-undo-btn');
    const redoButton = $('shaft-redo-btn');
    const deleteButton = $('shaft-delete-step-btn');
    const copyButton = $('shaft-copy-step-btn');
    const pasteButton = $('shaft-paste-step-btn');
    const loopSettingsButton = $('shaft-loop-settings-btn');
    const unlinedIndicator = $('shaft-unlined-buff-indicator');
    const selectionCount = selectedStepIds().length;
    if (undoButton) {
      undoButton.disabled = state.undoStack.length === 0;
    }
    if (redoButton) {
      redoButton.disabled = state.redoStack.length === 0;
    }
    if (deleteButton) {
      deleteButton.disabled = selectionCount === 0;
    }
    if (copyButton) {
      copyButton.disabled = selectionCount === 0;
    }
    if (pasteButton) {
      pasteButton.disabled = !state.clipboardSteps.length;
    }
    if (loopSettingsButton) {
      const enabled = Boolean(state.axis?.options?.loop_enabled);
      loopSettingsButton.classList.toggle('active', enabled);
      loopSettingsButton.textContent = enabled ? '循环轴：已开启' : '循环轴';
    }
    if (unlinedIndicator) {
      const tooltip = unlinedBuffToolbarTooltip();
      unlinedIndicator.hidden = !tooltip;
      unlinedIndicator.textContent = tooltip ? '!' : '';
      unlinedIndicator.setAttribute('data-tooltip', tooltip);
      unlinedIndicator.setAttribute('aria-label', tooltip);
    }
    applySharedReadOnlyUi();
  }

  function loopInitialResourceForMember(member) {
    const character = getCharacterMap().get(member?.character_id) || {};
    const configured = state.axis?.options?.loop_initial_resources?.[member?.character_id];
    const usesEnergy = character.uses_energy !== false;
    const energyCapacity = Math.max(0, Number(character.energy_capacity || 0));
    const fallbackEnergy = Math.max(0, Number(state.axis?.initial_energy ?? 1000));
    const configuredPersonal = configured?.personal_resources && typeof configured.personal_resources === 'object'
      ? configured.personal_resources
      : {};
    return {
      energy: !usesEnergy ? 0 : configured && typeof configured === 'object'
        ? Math.max(0, Number(configured.energy || 0))
        : (energyCapacity > 0 ? Math.min(fallbackEnergy, energyCapacity) : fallbackEnergy),
      harmony: configured && typeof configured === 'object'
        ? Math.max(0, Math.min(100, Number(configured.harmony || 0)))
        : 0,
      reaction: configured && typeof configured === 'object'
        ? String(configured.reaction || '')
        : '',
      energyCapacity,
      usesEnergy,
      personalResources: configuredPersonal,
    };
  }

  function loopPersonalResourceDefinitions(member) {
    const characterId = String(member?.character_id || '');
    const hidden = new Set([
      ...TIMELINE_HIDDEN_PERSONAL_RESOURCES,
      ...(state.catalog?.formula_constants?.hidden_personal_resources || []).map(String),
    ]);
    const names = new Set();
    (state.catalog?.actions || [])
      .filter((action) => String(action?.character_id || '') === characterId)
      .forEach((action) => {
        ['personal_resource_cost', 'personal_resource_gain', 'personal_resource_threshold'].forEach((field) => {
          Object.keys(action?.[field] || {}).forEach((name) => {
            if (!hidden.has(String(name))) {
              names.add(String(name));
            }
          });
        });
      });
    const caps = state.catalog?.formula_constants?.personal_resource_caps?.[characterId] || {};
    return Array.from(names).sort((left, right) => left.localeCompare(right, 'zh-CN')).map((name) => ({
      name,
      max: Number.isFinite(Number(caps[name])) ? Math.max(0, Number(caps[name])) : null,
    }));
  }

  function renderLoopResourceRows() {
    const list = $('shaft-loop-resource-list');
    if (!list) {
      return;
    }
    list.innerHTML = (state.axis?.team || []).map((member) => {
      const character = getCharacterMap().get(member.character_id) || {};
      const resources = loopInitialResourceForMember(member);
      const reactionOptions = (state.catalog?.formula_constants?.loop_initial_reaction_options || [])
        .filter((option) => option && Number(option.duration_ticks) > 0)
        .map((option) => String(option.id || ''))
        .filter(Boolean);
      const reactionFields = [
        '<option value="">不携带</option>',
        ...reactionOptions.map((reaction) => `<option value="${escapeHtml(reaction)}"${resources.reaction === reaction ? ' selected' : ''}>${escapeHtml(reaction)}</option>`),
      ].join('');
      const personalResourceFields = loopPersonalResourceDefinitions(member).map((definition) => {
        const maximum = definition.max == null ? '' : ` max="${definition.max}"`;
        const value = Math.max(0, Number(resources.personalResources?.[definition.name] || 0));
        return `<label>
          <span>起始${escapeHtml(definition.name)}</span>
          <input data-loop-initial-personal-resource="${escapeHtml(definition.name)}" type="number" min="0"${maximum} step="1" value="${value}">
        </label>`;
      }).join('');
      return `
        <div class="shaft-loop-resource-row" data-loop-resource-character="${escapeHtml(member.character_id)}">
          <span class="shaft-loop-resource-character">
            <img src="${escapeHtml(character.avatar || member.character_avatar || '')}" alt="">
            <strong>${escapeHtml(character.name || member.character_name || '未选择')}</strong>
          </span>
          <div class="shaft-loop-resource-fields">
            ${resources.usesEnergy ? `<label>
              <span>起始能量</span>
              <input data-loop-initial-energy type="number" min="0" max="${resources.energyCapacity || 1000}" step="1" value="${resources.energy}">
            </label>` : ''}
            <label>
              <span>起始环合</span>
              <input data-loop-initial-harmony type="number" min="0" max="100" step="1" value="${resources.harmony}">
            </label>
            <label>
              <span>携带环合</span>
              <select data-loop-initial-reaction>${reactionFields}</select>
            </label>
            ${personalResourceFields}
          </div>
        </div>
      `;
    }).join('') || '<div class="shaft-empty">当前队伍没有可配置角色</div>';
  }

  function syncLoopResourceInputsDisabled() {
    const enabled = Boolean($('shaft-loop-enabled')?.checked);
    const list = $('shaft-loop-resource-list');
    list?.classList.toggle('is-disabled', !enabled);
    list?.querySelectorAll('input, select').forEach((control) => {
      control.disabled = !enabled;
    });
  }

  function openLoopSettings(trigger = null) {
    const dialog = $('shaft-loop-settings-dialog');
    if (!dialog || dialog.open) {
      return;
    }
    dialog._returnFocus = trigger;
    $('shaft-loop-enabled').checked = Boolean(state.axis?.options?.loop_enabled);
    renderLoopResourceRows();
    syncLoopResourceInputsDisabled();
    dialog.showModal();
  }

  function closeLoopSettings() {
    const dialog = $('shaft-loop-settings-dialog');
    if (!dialog?.open) {
      return;
    }
    const returnFocus = dialog._returnFocus;
    dialog.close();
    if (returnFocus?.isConnected) {
      returnFocus.focus();
    }
  }

  function confirmLoopSettings() {
    const resources = {};
    $('shaft-loop-resource-list')?.querySelectorAll('[data-loop-resource-character]').forEach((row) => {
      const characterId = row.dataset.loopResourceCharacter;
      const character = getCharacterMap().get(characterId) || {};
      const energyCapacity = Math.max(0, Number(character.energy_capacity || 0));
      const energy = Math.max(0, Number(row.querySelector('[data-loop-initial-energy]')?.value || 0));
      const harmony = Math.max(0, Math.min(100, Number(row.querySelector('[data-loop-initial-harmony]')?.value || 0)));
      const reaction = String(row.querySelector('[data-loop-initial-reaction]')?.value || '');
      const personalResources = {};
      row.querySelectorAll('[data-loop-initial-personal-resource]').forEach((input) => {
        const name = String(input.dataset.loopInitialPersonalResource || '');
        const maximum = input.max === '' ? Number.POSITIVE_INFINITY : Math.max(0, Number(input.max));
        if (name) {
          personalResources[name] = Math.max(0, Math.min(maximum, Number(input.value || 0)));
        }
      });
      resources[characterId] = {
        energy: energyCapacity > 0 ? Math.min(energyCapacity, energy) : energy,
        harmony,
        reaction,
        personal_resources: personalResources,
      };
    });
    pushUndoSnapshot();
    state.axis.options.loop_enabled = Boolean($('shaft-loop-enabled')?.checked);
    state.axis.options.loop_initial_resources = resources;
    closeLoopSettings();
    renderEditorActions();
    markSimulationStale();
    renderTimeline();
    scheduleSimulation();
    persistAxisDraft();
  }

  function actionLabel(action) {
    if (!action) {
      return '';
    }
    return `${action.action_type || '动作'} · ${action.name}`;
  }

  function actionTypeClass(type) {
    const normalized = String(type || 'other').toLowerCase();
    if (normalized === 'e') {
      return 'skill';
    }
    if (normalized === 'q') {
      return 'ultimate';
    }
    if (String(type || '') === '普攻') {
      return 'basic';
    }
    if (String(type || '') === '援护') {
      return 'support';
    }
    if (String(type || '') === '无') {
      return 'utility';
    }
    return 'other';
  }

  function optionHtml(records, selected, getLabel) {
    return (records || []).map((record) => {
      const label = getLabel ? getLabel(record) : record.name;
      return `<option value="${escapeHtml(record.id)}" ${record.id === selected ? 'selected' : ''} ${record.selection_disabled ? 'disabled' : ''}>${escapeHtml(label)}</option>`;
    }).join('');
  }

  function arcsForCharacter(characterId) {
    const character = getCharacterMap().get(characterId);
    const adaptation = String(character?.adaptation || '');
    if (!adaptation) {
      return [];
    }
    return (state.catalog?.arcs || []).filter((arc) => String(arc.adaptation || '') === adaptation);
  }

  function ensureMemberCompatibleArc(member) {
    const compatibleArcs = arcsForCharacter(member?.character_id);
    if (!compatibleArcs.some((arc) => arc.id === member?.arc_id)) {
      member.arc_id = compatibleArcs[0]?.id || '';
    }
    updateMemberNames(member);
  }

  function cartridgesForCharacter() {
    return state.catalog?.cartridges || [];
  }

  function ensureMemberCompatibleCartridge(member) {
    const compatibleCartridges = cartridgesForCharacter(member?.character_id);
    const selectedCartridge = compatibleCartridges.find((cartridge) => cartridge.id === member?.cartridge_id);
    if (!selectedCartridge) {
      member.cartridge_id = compatibleCartridges[0]?.id || '';
    }
    updateMemberNames(member);
  }

  function slotOptions(selected) {
    return (state.axis?.team || []).map((member) => `
      <option value="${member.slot}" ${Number(member.slot) === Number(selected) ? 'selected' : ''}>
        ${Number(member.slot) + 1} · ${escapeHtml(member.character_name)}
      </option>
    `).join('');
  }

  function emptySubstatCounts() {
    const out = {};
    SUBSTAT_ORDER.forEach((key) => { out[key] = 0; });
    return out;
  }

  function clampSubstatCount(value) {
    return Math.max(0, Math.min(30, Math.round(Number(value || 0))));
  }

  function substatTotal(counts) {
    return SUBSTAT_ORDER.reduce((sum, key) => sum + clampSubstatCount(counts?.[key]), 0);
  }

  function substatBonusText(key, count) {
    const unit = state.catalog?.formula_constants?.substat_units?.[key] || {};
    const value = clampSubstatCount(count) * Number(unit.unit_value || 0);
    if (!value) {
      return '+0';
    }
    return unit.kind === 'percent' ? `+${formatNumber(value * 100, 1)}%` : `+${formatNumber(value, 0)}`;
  }

  function normalizeSubstatCounts(raw) {
    const out = Object.assign({}, emptySubstatCounts(), raw || {});
    SUBSTAT_ORDER.forEach((key) => {
      out[key] = clampSubstatCount(out[key]);
    });
    return out;
  }

  function clampSkillLevel(value) {
    return Math.max(1, Math.min(10, Math.round(Number(value || 10))));
  }

  function defaultArcRefinement(arcId) {
    const level = Number(state.catalog?.arc_refinements?.arcs?.[arcId || '']?.default_level || 1);
    return Math.max(1, Math.min(5, Math.round(level)));
  }

  function clampArcRefinement(value, arcId = '') {
    const level = Math.round(Number(value));
    return level >= 1 && level <= 5 ? level : defaultArcRefinement(arcId);
  }

  function arcRefinementRecord(member) {
    const arcs = state.catalog?.arc_refinements?.arcs || {};
    const arc = arcs[member?.arc_id || ''] || {};
    const level = clampArcRefinement(member?.arc_refinement, member?.arc_id);
    return arc.levels?.[String(level)] || null;
  }

  function arcRefinementSummary(member) {
    const level = clampArcRefinement(member?.arc_refinement, member?.arc_id);
    const source = state.catalog?.arc_refinements?.arcs?.[member?.arc_id || ''] || {};
    const record = arcRefinementRecord(member);
    if (!record) {
      return `精炼 ${level} · 暂无 Nanoka 数据`;
    }
    const values = (record.effect_values || []).filter((value) => value != null && value !== '');
    return `精炼 ${level} · ${source.effect_name || '弧盘效果'}${values.length ? `：${values.join(' / ')}` : ''}`;
  }

  function normalizeSkillLevels(raw) {
    const out = Object.assign({}, DEFAULT_SKILL_LEVELS, raw || {});
    SKILL_LEVEL_FIELDS.forEach((field) => {
      out[field.key] = clampSkillLevel(out[field.key]);
    });
    return out;
  }

  function normalizeAwakeningNodes(rawNodes, legacyAwakening = 0) {
    if (Array.isArray(rawNodes)) {
      return Array.from(new Set(rawNodes
        .map((value) => Math.round(Number(value)))
        .filter((value) => value >= 1 && value <= 6)))
        .sort((left, right) => left - right);
    }
    const legacyLevel = Math.max(0, Math.min(6, Math.round(Number(legacyAwakening || 0))));
    return Array.from({ length: legacyLevel }, (_, index) => index + 1);
  }

  function activeAwakeningCount(member) {
    return normalizeAwakeningNodes(member?.awakening_nodes, member?.awakening).length;
  }

  function characterHasBondBonus(characterId) {
    const character = getCharacterMap().get(String(characterId || ''));
    const modifiers = character?.bond_bonus?.modifiers;
    return Boolean(modifiers && typeof modifiers === 'object' && Object.keys(modifiers).length);
  }

  function buildSnapshotFromMember(member) {
    const awakeningNodes = normalizeAwakeningNodes(member.awakening_nodes, member.awakening);
    const hasBondBonus = characterHasBondBonus(member.character_id);
    return {
      character_id: member.character_id || '',
      character_name: member.character_name || '',
      arc_id: member.arc_id || '',
      arc_name: member.arc_name || '',
      arc_refinement: clampArcRefinement(member.arc_refinement, member.arc_id),
      cartridge_id: member.cartridge_id || '',
      cartridge_name: member.cartridge_name || '',
      awakening: awakeningNodes.length,
      awakening_nodes: awakeningNodes,
      bond_level: hasBondBonus ? Math.max(0, Math.min(1, Number(member.bond_level || (member.bond_full ? 1 : 0)))) : 0,
      bond_full: hasBondBonus ? Boolean(member.bond_full) || Number(member.bond_level || 0) > 0 : false,
      skill_levels: normalizeSkillLevels(member.skill_levels),
      cartridge_main_stat: normalizeCartridgeMainStat(member.cartridge_main_stat, member.character_id),
      curtain_bonus: normalizeCurtainBonus(member.curtain_bonus, member.character_id),
      substat_counts: normalizeSubstatCounts(member.substat_counts),
    };
  }

  function applyBuildToMember(member, build) {
    if (!member || !build) {
      return;
    }
    member.arc_id = build.arc_id || member.arc_id || '';
    member.arc_refinement = clampArcRefinement(build.arc_refinement, member.arc_id);
    member.cartridge_id = build.cartridge_id || member.cartridge_id || '';
    member.awakening_nodes = normalizeAwakeningNodes(build.awakening_nodes, build.awakening);
    member.awakening = member.awakening_nodes.length;
    const hasBondBonus = characterHasBondBonus(member.character_id);
    member.bond_level = hasBondBonus ? Math.max(0, Math.min(1, Number(build.bond_level || (build.bond_full ? 1 : 0)))) : 0;
    member.bond_full = hasBondBonus ? Boolean(build.bond_full) || member.bond_level > 0 : false;
    member.skill_levels = normalizeSkillLevels(build.skill_levels);
    member.cartridge_main_stat = normalizeCartridgeMainStat(build.cartridge_main_stat, member.character_id);
    member.curtain_bonus = normalizeCurtainBonus(build.curtain_bonus, member.character_id);
    member.substat_counts = normalizeSubstatCounts(build.substat_counts);
    updateMemberNames(member);
  }

  function normalizeCharacterBuild(raw) {
    const build = raw && typeof raw === 'object' ? raw : {};
    const awakeningNodes = normalizeAwakeningNodes(build.awakening_nodes, build.awakening);
    const hasBondBonus = characterHasBondBonus(build.character_id);
    return {
      character_id: build.character_id || '',
      character_name: build.character_name || '',
      arc_id: build.arc_id || '',
      arc_name: build.arc_name || '',
      arc_refinement: clampArcRefinement(build.arc_refinement, build.arc_id),
      cartridge_id: build.cartridge_id || '',
      cartridge_name: build.cartridge_name || '',
      awakening: awakeningNodes.length,
      awakening_nodes: awakeningNodes,
      bond_level: hasBondBonus ? Math.max(0, Math.min(1, Number(build.bond_level || (build.bond_full ? 1 : 0)))) : 0,
      bond_full: hasBondBonus ? Boolean(build.bond_full) || Number(build.bond_level || 0) > 0 : false,
      skill_levels: normalizeSkillLevels(build.skill_levels),
      cartridge_main_stat: normalizeCartridgeMainStat(build.cartridge_main_stat, build.character_id || ''),
      curtain_bonus: normalizeCurtainBonus(build.curtain_bonus, build.character_id || ''),
      substat_counts: normalizeSubstatCounts(build.substat_counts),
    };
  }

  function seedCharacterBuilds(rawBuilds) {
    const characterIds = new Set((state.catalog?.characters || []).map((character) => character.id));
    const seeded = {};
    const starterBuilds = state.catalog?.starter_axis?.character_builds || {};
    Object.entries(starterBuilds).forEach(([characterId, build]) => {
      if (characterIds.has(characterId)) {
        seeded[characterId] = normalizeCharacterBuild(Object.assign({}, Array.isArray(build) ? build[0] : build, { character_id: characterId }));
      }
    });
    Object.entries(rawBuilds && typeof rawBuilds === 'object' ? rawBuilds : {}).forEach(([characterId, build]) => {
      if (characterIds.has(characterId)) {
        const rawBuild = Array.isArray(build) ? build[0] : (build?.variants && Array.isArray(build.variants) ? build.variants[0] : build);
        seeded[characterId] = normalizeCharacterBuild(Object.assign({}, rawBuild, { character_id: characterId }));
      }
    });
    return seeded;
  }

  function rememberMemberBuild(member) {
    if (!member?.character_id) {
      return;
    }
    state.axis.character_builds = seedCharacterBuilds(state.axis.character_builds);
    state.axis.character_builds[member.character_id] = buildSnapshotFromMember(member);
  }

  function ensureAxisShape() {
    if (!state.axis) {
      state.axis = emptyAxisFromCatalog();
    }
    const hadCharacterBuilds = Boolean(
      state.axis.character_builds &&
      typeof state.axis.character_builds === 'object' &&
      Object.keys(state.axis.character_builds).length,
    );
    state.axis.character_builds = seedCharacterBuilds(state.axis.character_builds);
    state.axis.team = Array.isArray(state.axis.team) && state.axis.team.length
      ? state.axis.team.slice(0, 4)
      : clone(state.catalog.starter_axis.team);
    state.axis.steps = Array.isArray(state.axis.steps) ? state.axis.steps : [];
    const disabledCharacterIds = new Set(
      (state.catalog?.characters || [])
        .filter((character) => character.selection_disabled)
        .map((character) => character.id),
    );
    const restrictedSlots = new Set();
    state.axis.team = state.axis.team.map((member, index) => {
      if (!disabledCharacterIds.has(member?.character_id)) {
        return member;
      }
      const slot = Number(member?.slot ?? index);
      restrictedSlots.add(slot);
      return clone(
        (state.catalog?.starter_axis?.team || []).find((candidate) => Number(candidate.slot) === slot)
        || (state.catalog?.starter_axis?.team || [])[index]
        || member,
      );
    });
    if (restrictedSlots.size) {
      state.axis.steps = state.axis.steps.filter((step) => !restrictedSlots.has(Number(step.slot)));
    }
    state.axis.steps.forEach(sanitizeStepPlacement);
    state.axis.enemy = Object.assign(
      {},
      state.catalog.formula_constants.default_enemy || { level: 90, track_outside: false, weakness_elements: [] },
      state.axis.enemy || {},
    );
    const initialResistance = Math.max(-1, Math.min(1, Number(
      state.axis.enemy.initial_resistance
      ?? state.axis.enemy.resistances?.光
      ?? 0.3,
    )));
    state.axis.enemy.initial_resistance = initialResistance;
    state.axis.enemy.resistances = Object.fromEntries(
      RESISTANCE_ELEMENTS.map((element) => [element, initialResistance]),
    );
    state.axis.options = Object.assign(
      {
        switch_gap_ticks: state.catalog.formula_constants.switch_loss_ticks || 2,
        switch_loss_ticks: state.catalog.formula_constants.switch_loss_ticks || 2,
        loop_enabled: false,
      },
      state.axis.options || {},
    );
    state.axis.options.switch_gap_ticks = Number(
      state.axis.options.switch_gap_ticks ?? state.axis.options.switch_loss_ticks ?? 2,
    );
    state.axis.options.switch_loss_ticks = state.axis.options.switch_gap_ticks;
    state.axis.options.loop_enabled = Boolean(state.axis.options.loop_enabled);
    state.axis.options.loop_initial_resources = state.axis.options.loop_initial_resources &&
      typeof state.axis.options.loop_initial_resources === 'object'
      ? state.axis.options.loop_initial_resources
      : {};
    state.axis.team_panel_bonus = normalizeTeamPanelBonus(state.axis.team_panel_bonus);
    state.axis.initial_energy = Number(state.axis.initial_energy ?? 1000);
    if (state.axis.initial_energy === 100) {
      state.axis.initial_energy = 1000;
    }
    state.axis.buff_rules = Array.isArray(state.axis.buff_rules) ? state.axis.buff_rules : [];
    delete state.axis.active_build_variant;
    delete state.axis.compare_settings;
    state.axis.team.forEach((member, index) => {
      member.slot = Number(member.slot ?? index);
      member.awakening_nodes = normalizeAwakeningNodes(member.awakening_nodes, member.awakening);
      member.awakening = member.awakening_nodes.length;
      const hasBondBonus = characterHasBondBonus(member.character_id);
      member.bond_level = hasBondBonus ? Math.max(0, Math.min(1, Number(member.bond_level || (member.bond_full ? 1 : 0)))) : 0;
      member.bond_full = hasBondBonus ? Boolean(member.bond_full) || member.bond_level > 0 : false;
      member.skill_levels = normalizeSkillLevels(member.skill_levels);
      member.cartridge_main_stat = normalizeCartridgeMainStat(member.cartridge_main_stat, member.character_id);
      member.curtain_bonus = normalizeCurtainBonus(member.curtain_bonus, member.character_id);
      member.substat_counts = normalizeSubstatCounts(member.substat_counts);
      updateMemberNames(member);
      if (hadCharacterBuilds && state.axis.character_builds[member.character_id]) {
        applyBuildToMember(member, state.axis.character_builds[member.character_id]);
      }
      ensureMemberCompatibleArc(member);
      ensureMemberCompatibleCartridge(member);
      rememberMemberBuild(member);
    });
    syncSelection(true);
    updateAxisDuration();
  }

  function saveCompareSnapshot() {
    const result = freshResult();
    if (!result) {
      showCompareSnapshotToast('请先完成计算后再保存快照');
      runSimulation().then(() => {
        if (freshResult()) {
          saveCompareSnapshot();
        }
      });
      return;
    }
    state.compareSnapshot = {
      saved_at: Date.now(),
      summary: clone(result.summary || {}),
      damage_by_slot: clone(result.damage_by_slot || []),
      damage_by_action_by_slot: clone(result.damage_by_action_by_slot || []),
      harmony_contributions_by_slot: clone(result.harmony_contributions_by_slot || []),
    };
    renderResults();
    if ($('shaft-action-contribution-dialog')?.open) {
      renderActionContributionAnalysis();
    }
    persistAxisDraft();
    setStatus('已保存对比快照');
  }

  function clearCompareSnapshot() {
    state.compareSnapshot = null;
    renderResults();
    persistAxisDraft();
    setStatus('已清除对比快照');
  }

  function showCompareSnapshotToast(message) {
    setStatus(message, 'dirty');
  }

  function setStatus(text, tone = '') {
    const node = $('shaft-status');
    if (!node) {
      return;
    }
    node.textContent = text;
    node.dataset.tone = tone;
  }

  function showToast(message, tone = 'success') {
    const node = $('shaft-toast');
    if (!node) {
      return;
    }
    window.clearTimeout(state.toastTimer);
    node.textContent = message;
    node.dataset.tone = tone;
    node.hidden = false;
    requestAnimationFrame(() => node.classList.add('visible'));
    state.toastTimer = window.setTimeout(() => {
      node.classList.remove('visible');
      window.setTimeout(() => { node.hidden = true; }, 180);
    }, 1800);
  }

  async function copyTextToClipboard(text) {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const input = document.createElement('textarea');
    input.value = text;
    input.setAttribute('readonly', '');
    input.style.position = 'fixed';
    input.style.opacity = '0';
    document.body.appendChild(input);
    input.select();
    const copied = document.execCommand('copy');
    input.remove();
    if (!copied) {
      throw new Error('浏览器未允许自动复制，请手动复制地址栏中的链接。');
    }
  }

  function setSimulationBusy(busy) {
    state.simulationInFlight = Boolean(busy);
  }

  function setPage(page, push = true) {
    const normalizedPage = page === 'market' ? 'plaza' : page;
    state.page = ['build', 'rotation', 'plaza'].includes(normalizedPage) ? normalizedPage : 'rotation';
    document.querySelectorAll('[data-shaft-view]').forEach((node) => {
      node.classList.toggle('active', node.dataset.shaftView === state.page);
    });
    document.querySelectorAll('[data-shaft-page-link]').forEach((node) => {
      node.classList.toggle('active', node.dataset.shaftPageLink === state.page);
    });
    const axisInfoBar = document.querySelector('[data-shaft-axis-info]');
    if (axisInfoBar) {
      axisInfoBar.hidden = state.page === 'plaza';
    }
    const shortcutHelpButton = $('shaft-shortcut-help-btn');
    if (shortcutHelpButton) {
      shortcutHelpButton.hidden = state.page !== 'rotation';
    }
    const buildInfoButton = $('shaft-build-info-btn');
    if (buildInfoButton) {
      buildInfoButton.hidden = state.page !== 'build';
    }
    if (push) {
      window.history.replaceState({}, '', `/shaft/${state.page}`);
      persistAxisDraft();
    }
  }

  function renderLoginState() {
    const link = $('shaft-login-link');
    if (!link) {
      return;
    }
    if (getToken()) {
      link.href = '/profile';
      link.textContent = '账号';
    } else {
      link.href = typeof loginUrlForCurrentPage === 'function' ? loginUrlForCurrentPage() : '/login';
      link.textContent = '登录';
    }
  }

  function renderTeam() {
    const container = $('shaft-team-slots');
    const characters = state.catalog.characters || [];
    const substatUnits = state.catalog.formula_constants.substat_units || {};
    const characterMap = getCharacterMap();
    container.innerHTML = (state.axis.team || []).map((member) => {
      const character = characterMap.get(member.character_id) || {};
      const compatibleArcs = arcsForCharacter(member.character_id);
      const compatibleCartridges = cartridgesForCharacter(member.character_id);
      const arcRefinement = clampArcRefinement(member.arc_refinement, member.arc_id);
      const hasBondBonus = characterHasBondBonus(member.character_id);
      const bondLabel = hasBondBonus ? character.bond_bonus.label : '无羁绊加成';
      const activeAwakeningNodes = new Set(normalizeAwakeningNodes(member.awakening_nodes, member.awakening));
      const activeAwakeningCount = activeAwakeningNodes.size;
      const slotColor = SLOT_COLORS[Number(member.slot) % SLOT_COLORS.length];
      const characterAwakenings = awakeningsForCharacter(character, member);
      const awakeningToggles = Array.from({ length: 6 }, (_, index) => {
        const level = index + 1;
        const label = AWAKENING_LABELS[index] || String(level);
        const checked = activeAwakeningNodes.has(level) ? 'checked' : '';
        const tooltip = awakeningTooltipText(characterAwakenings[index]);
        return `
          <label class="shaft-awakening-dot" data-tooltip="${escapeHtml(tooltip)}" aria-label="${escapeHtml(tooltip)}">
            <input data-slot="${member.slot}" data-field="awakening" data-awakening-level="${level}" type="checkbox" ${checked}>
            <span>${escapeHtml(label)}</span>
          </label>
        `;
      }).join('');
      const awakeningInfos = [6, 7].map((index, infoIndex) => {
        const tooltip = awakeningTooltipText(characterAwakenings[index]);
        const active = infoIndex === 0 ? activeAwakeningCount >= 3 : activeAwakeningCount >= 6;
        return `
          <span class="shaft-awakening-dot shaft-awakening-info ${active ? 'is-active' : ''}" data-tooltip="${escapeHtml(tooltip)}" aria-label="${escapeHtml(tooltip)}">
            <span>!</span>
          </span>
        `;
      }).join('');
      const skillLevels = normalizeSkillLevels(member.skill_levels);
      const skillLevelControls = SKILL_LEVEL_FIELDS.map((field) => `
        <label class="shaft-skill-level-tile">
          <span>${escapeHtml(field.label)}</span>
          <input data-slot="${member.slot}" data-skill-level="${field.key}" type="number" min="1" max="10" step="1" inputmode="numeric" value="${skillLevels[field.key]}" aria-label="${escapeHtml(field.label)}技能等级">
        </label>
      `).join('');
      const mainStat = normalizeCartridgeMainStat(member.cartridge_main_stat, member.character_id);
      const curtainBonus = normalizeCurtainBonus(member.curtain_bonus, member.character_id);
      const curtainResult = curtainBonusResultText(member);
      const mechanismTooltip = activeMechanismTooltipHtml(member, character);
      const selectableCharacters = characters.filter((candidate) => !candidate.selection_disabled);
      const characterOptions = selectableCharacters.map((candidate) => `
        <button class="shaft-character-filter-option ${candidate.id === member.character_id ? 'selected' : ''}"
                data-build-character-id="${escapeHtml(candidate.id)}"
                data-slot="${member.slot}"
                type="button"
                role="option"
                aria-selected="${candidate.id === member.character_id ? 'true' : 'false'}">
          <img src="${escapeHtml(candidate.avatar || candidate.portrait || '')}" alt="" loading="lazy" decoding="async">
          <span>${escapeHtml(candidate.name)}</span>
          <small>${escapeHtml(candidate.element || '')}</small>
        </button>
      `).join('');
      const mainStatSelect = Object.entries(mainStatOptions()).map(([key, meta]) => `
        <option value="${escapeHtml(key)}" ${key === mainStat ? 'selected' : ''}>${escapeHtml(meta.label || key)}</option>
      `).join('');
      const substatUsed = substatTotal(member.substat_counts);
      const substats = SUBSTAT_ORDER.map((key) => {
        const unit = substatUnits[key] || {};
        const unitValue = Number(unit.unit_value || 0);
        const unitLabel = unit.kind === 'percent'
          ? `${formatNumber(unitValue * 100, 2)}%/B`
          : `${formatNumber(unitValue, 0)}/B`;
        const count = clampSubstatCount(member.substat_counts?.[key]);
        const countPct = Math.max(0, Math.min(100, (count / 30) * 100));
        return `
          <label class="shaft-substat-tile" style="--count-pct:${countPct}%">
            <span class="shaft-substat-name">${escapeHtml(unit.label || key)}</span>
            <small>${escapeHtml(substatBonusText(key, count))}</small>
            <span class="shaft-substat-entry">
              <input data-slot="${member.slot}" data-substat="${key}" type="number" min="0" max="30" step="1" inputmode="numeric" value="${count}" aria-label="${escapeHtml(unit.label || key)}词条数，最多总计120">
              <b>B</b>
            </span>
            <span class="shaft-substat-meter"><i></i></span>
          </label>
        `;
      }).join('');
      return `
        <article class="shaft-member-card" data-slot="${member.slot}" style="--slot-color:${slotColor}">
          <section class="shaft-member-identity">
            <span class="shaft-member-slot">0${Number(member.slot) + 1}</span>
            <img class="shaft-member-avatar" src="${escapeHtml(character.avatar || member.character_avatar || '')}" alt="">
            <strong>${escapeHtml(character.name || member.character_name || '未选择')}</strong>
            <small>角色80 · 弧盘80 · 精炼${arcRefinement}</small>
            ${mechanismTooltip}
          </section>
          <section class="shaft-member-editor">
            <div class="shaft-loadout-grid">
              <div class="shaft-character-select shaft-build-character-picker" data-build-character-picker>
                <span>角色</span>
                <button class="shaft-character-filter-trigger shaft-build-character-trigger"
                        data-build-character-trigger
                        type="button"
                        aria-haspopup="listbox"
                        aria-expanded="false"
                        aria-label="选择${Number(member.slot) + 1}号位角色，当前${escapeHtml(character.name || member.character_name || '未选择')}">
                  <span class="shaft-character-filter-selected">
                    <img src="${escapeHtml(character.avatar || member.character_avatar || '')}" alt="">
                    <b>${escapeHtml(character.name || member.character_name || '未选择')}</b>
                  </span>
                  <small>单选</small>
                </button>
                <div class="shaft-character-filter-popover shaft-build-character-popover"
                     data-build-character-popover
                     role="listbox"
                     aria-label="${Number(member.slot) + 1}号位角色"
                     hidden>
                  <div class="shaft-character-filter-grid">${characterOptions}</div>
                </div>
              </div>
              <label>
                <span>弧盘</span>
                <select data-slot="${member.slot}" data-field="arc_id">
                  ${optionHtml(compatibleArcs, member.arc_id)}
                </select>
              </label>
              <label>
                <span>弧盘精炼</span>
                <select data-slot="${member.slot}" data-field="arc_refinement" aria-label="弧盘精炼等级">
                  ${Array.from({ length: 5 }, (_, index) => {
                    const level = index + 1;
                    return `<option value="${level}" ${arcRefinement === level ? 'selected' : ''}>精炼 ${level}</option>`;
                  }).join('')}
                </select>
              </label>
            </div>
            <small class="shaft-arc-refinement-summary">${escapeHtml(arcRefinementSummary(member))}</small>
            <div class="shaft-member-switches">
              <div class="shaft-awakening-control">
                <span>觉醒</span>
                <div class="shaft-awakening-dots">${awakeningToggles}${awakeningInfos}</div>
              </div>
              <label class="shaft-bond-toggle">
                <input data-slot="${member.slot}" data-field="bond_full" type="checkbox" ${hasBondBonus && member.bond_full ? 'checked' : ''} ${hasBondBonus ? '' : 'disabled'}>
                <span>羁绊加成 · ${escapeHtml(bondLabel)}</span>
              </label>
            </div>
            <div class="shaft-skill-levels">
              <div>
                <span>技能等级</span>
                <strong>基础等级 1-10</strong>
              </div>
              <div class="shaft-skill-level-grid">${skillLevelControls}</div>
            </div>
            <div class="shaft-cartridge-tuning">
              <div class="shaft-tuning-title">
                <span>卡带调校</span>
                <strong>主词条与空幕</strong>
              </div>
              <label class="shaft-cartridge-select">
                <span>卡带</span>
                <select data-slot="${member.slot}" data-field="cartridge_id">
                  ${optionHtml(compatibleCartridges, member.cartridge_id)}
                </select>
              </label>
              <label class="shaft-main-stat-select">
                <span>主词条</span>
                <select data-slot="${member.slot}" data-field="cartridge_main_stat">
                  ${mainStatSelect}
                </select>
              </label>
              <output class="shaft-readonly-output shaft-curtain-passive-output">
                <span>空幕被动</span>
                <strong>${escapeHtml(curtainResult.passive.replace('被动', '驱动'))}：${formatNumber(curtainBonus.value || 0, 0)}${escapeHtml(curtainResult.stat)}</strong>
              </output>
              <output class="shaft-readonly-output shaft-curtain-result-output">
                <span>层数 / 结果</span>
                <strong>${formatNumber(curtainResult.layers, 0)} 层 · ${escapeHtml(curtainResult.display)}</strong>
              </output>
            </div>
            <div class="shaft-substat-head">
              <div>
                <span>副词条</span>
                <span
                  class="shaft-substat-total-tooltip"
                  data-tooltip="${escapeHtml(SUBSTAT_TOTAL_TOOLTIP)}"
                  tabindex="0"
                  aria-label="副词条 ${formatNumber(substatUsed, 0)}/120。${escapeHtml(SUBSTAT_TOTAL_TOOLTIP)}"
                >
                  <strong>${formatNumber(substatUsed, 0)}/120</strong>
                </span>
              </div>
              <em>单项 0-30 B</em>
            </div>
            <div class="shaft-substat-grid">${substats}</div>
          </section>
        </article>
      `;
    }).join('');
  }

  function renderEnemy() {
    $('shaft-enemy-level').value = state.axis.enemy.level || 90;
    $('shaft-enemy-track-outside').checked = Boolean(state.axis.enemy.track_outside);
    const active = new Set(state.axis.enemy.weakness_elements || []);
    $('shaft-weakness-row').innerHTML = ELEMENTS.map((element) => `
      <label class="shaft-chip">
        <input type="checkbox" value="${element}" ${active.has(element) ? 'checked' : ''}>
        <span>${element}</span>
      </label>
    `).join('');
    $('shaft-initial-resistance-input').value = Math.round(
      Number(state.axis.enemy.initial_resistance ?? 0.3) * 100,
    );
  }

  function renderActionAdder() {
    const addSlot = $('shaft-add-slot');
    const librarySlot = $('shaft-library-slot');
    if (!addSlot) {
      if (librarySlot) {
        librarySlot.innerHTML = slotOptions(state.librarySlot);
      }
      return;
    }
    const selectedSlot = Number(addSlot.value || state.librarySlot || 0);
    addSlot.innerHTML = slotOptions(selectedSlot);
    const slot = Number(addSlot.value || selectedSlot);
    const addAction = $('shaft-add-action');
    addAction.innerHTML = optionHtml(actionsForSlot(slot), addAction.value, actionLabel);
    if (librarySlot) {
      librarySlot.innerHTML = slotOptions(state.librarySlot);
    }
  }

  function renderTeamDock() {
    const dock = $('shaft-team-dock');
    if (!dock) {
      return;
    }
    dock.innerHTML = (state.axis.team || []).map((member) => {
      const selected = Number(member.slot) === Number(state.librarySlot) ? 'active' : '';
      const color = SLOT_COLORS[Number(member.slot) % SLOT_COLORS.length];
      const awakeningCount = activeAwakeningCount(member);
      return `
        <button class="shaft-operator-card ${selected}" data-dock-slot="${member.slot}" type="button" style="--slot-color:${color}" title="${escapeHtml(member.character_name)} · 已激活 ${awakeningCount} 个觉醒">
          <img src="${escapeHtml(member.character_avatar || '')}" alt="">
          <b aria-label="已激活 ${awakeningCount} 个觉醒">${awakeningCount}</b>
        </button>
      `;
    }).join('');
  }

  function renderTeamBonus() {
    const node = $('shaft-team-bonus-grid');
    if (!node) {
      return;
    }
    state.axis.team_panel_bonus = normalizeTeamPanelBonus(state.axis.team_panel_bonus);
    const bonus = state.axis.team_panel_bonus;
    node.innerHTML = TEAM_PANEL_BONUS_FIELDS.map((field) => {
      const rawValue = bonus[field.key];
      const value = field.kind === 'percent' ? formatNumber(rawValue * 100, 1) : formatNumber(rawValue, 0);
      const suffix = field.kind === 'percent' ? '%' : '';
      return `
        <label class="shaft-team-bonus-tile">
          <span>${escapeHtml(field.label)}</span>
          <span class="shaft-team-bonus-input">
            <input data-team-bonus="${escapeHtml(field.key)}" type="number" min="0" max="${field.max}" step="${field.step}" value="${escapeHtml(value)}" inputmode="decimal">
            ${suffix ? `<b>${escapeHtml(suffix)}</b>` : ''}
          </span>
          <small>范围 0–${escapeHtml(formatNumber(field.max, field.kind === 'percent' ? 1 : 0))}${escapeHtml(suffix)}</small>
        </label>
      `;
    }).join('');
  }

  function renderCommandTypes() {
    $('shaft-command-type-tabs').innerHTML = ACTION_TYPES.map((type) => `
      <button class="shaft-type-chip ${state.commandTypeFilter === type ? 'active' : ''}" data-command-type="${escapeHtml(type)}" type="button">
        ${escapeHtml(type || '全部')}
      </button>
    `).join('');
  }

  function renderCommandLibrary() {
    const search = state.commandSearch.trim().toLowerCase();
    const libraryCharacter = getCharacterMap().get(memberBySlot(state.librarySlot)?.character_id) || {};
    const showsEnergy = libraryCharacter.uses_energy !== false;
    const actions = actionsForSlot(state.librarySlot)
      .filter((action) => !state.commandTypeFilter || action.action_type === state.commandTypeFilter)
      .filter((action) => !search || `${action.name} ${action.action_type} ${action.extra_tag || ''}`.toLowerCase().includes(search));
    $('shaft-command-list').innerHTML = actions.map((action) => `
      <article class="shaft-command-card ${actionTypeClass(action.action_type)} ${isBackgroundAction(action) ? 'background-damage' : ''}" data-action-id="${escapeHtml(action.id)}">
        <div class="shaft-command-card-body">
          <div class="shaft-command-title">
            <span class="shaft-command-type">${escapeHtml(action.action_type || '动作')}</span>
            <strong class="shaft-command-name">${escapeHtml(action.name)}</strong>
          </div>
          <span class="shaft-command-meta">${ticksToSeconds(actionCalculationDurationTicks(action))}s · ${formatNumber(action.hit_count || 0, 0)}段${showsEnergy ? ` · ${Number(action.energy_cost || 0) > 0 ? `耗能 ${formatNumber(action.energy_cost, 0)}` : `回能 ${formatNumber(action.energy_gain || 0, 1)}`}` : ''}${isBackgroundAction(action) ? ' · 后台' : ''}</span>
        </div>
        <button class="secondary-btn" data-library-action="${escapeHtml(action.id)}" data-library-slot="${state.librarySlot}" type="button">加入</button>
      </article>
    `).join('') || '<div class="shaft-empty">没有动作</div>';
  }

  function renderSteps() {
    const stepList = $('shaft-step-list');
    if (!stepList) {
      return;
    }
    const selectedIds = new Set(selectedStepIds());
    const rows = (state.axis.steps || []).map((step, index) => {
      const detail = detailByStepId(step.id) || {};
      const actions = actionsForSlot(step.slot);
      const warnings = (detail.warnings || []).length ? ` · ${escapeHtml(detail.warnings.join('；'))}` : '';
      return `
        <div class="shaft-step-row ${selectedIds.has(step.id) ? 'selected' : ''}" data-step-id="${escapeHtml(step.id)}">
          <output class="shaft-step-index"><span>#${index + 1}</span><strong>${escapeHtml(visualTickLabel(step.start_tick))}</strong></output>
          <label>
            <span>角色</span>
            <select data-step-id="${escapeHtml(step.id)}" data-step-field="slot">${slotOptions(step.slot)}</select>
          </label>
          <label>
            <span>动作</span>
            <select data-step-id="${escapeHtml(step.id)}" data-step-field="action_id">
              ${optionHtml(actions, step.action_id, actionLabel)}
            </select>
          </label>
          <label>
            <span>开始</span>
            <input data-step-id="${escapeHtml(step.id)}" data-step-field="start_tick" type="text" value="${escapeHtml(visualTickLabel(step.start_tick))}" data-visual-tick="${Number(step.start_tick || 0)}" data-calculation-tick="${calculationTickFromVisual(step.start_tick)}" data-synced-value="${escapeHtml(visualTickLabel(step.start_tick))}">
          </label>
          <output class="shaft-step-damage"><span>直伤${warnings}</span><strong>${formatNumber(detail.direct_damage || 0)}</strong></output>
          <button class="secondary-btn shaft-remove-btn" data-remove-step="${escapeHtml(step.id)}" type="button" title="删除">×</button>
        </div>
      `;
    }).join('');
    stepList.innerHTML = rows || '<div class="shaft-empty">暂无动作</div>';
  }

  function renderBuffEditor() {
    const triggerSlotSelect = $('shaft-buff-trigger-slot');
    const triggerActionSelect = $('shaft-buff-trigger-action');
    const targetSlots = $('shaft-buff-target-slots');
    const modifierKeySelect = $('shaft-buff-modifier-key');
    if (!triggerSlotSelect || !triggerActionSelect || !targetSlots || !modifierKeySelect) {
      return;
    }
    triggerSlotSelect.innerHTML = slotOptions(state.buffDraft.triggerSlot);
    const triggerSlot = Number(triggerSlotSelect.value || state.buffDraft.triggerSlot || 0);
    triggerActionSelect.innerHTML = optionHtml(actionsForSlot(triggerSlot), triggerActionSelect.value, actionLabel);
    targetSlots.innerHTML = (state.axis.team || []).map((member) => `
      <label class="shaft-chip">
        <input type="checkbox" value="${member.slot}" checked>
        <span>${Number(member.slot) + 1} · ${escapeHtml(member.character_name)}</span>
      </label>
    `).join('');
    modifierKeySelect.innerHTML = BUFF_MODIFIERS.map((item) => `
      <option value="${item.key}" ${item.key === state.buffDraft.modifierKey ? 'selected' : ''}>${escapeHtml(item.label)}</option>
    `).join('');
    renderBuffList();
  }

  function modifierMeta(key) {
    return BUFF_MODIFIERS.find((item) => item.key === key) || BUFF_MODIFIERS[0];
  }

  function modifierDisplay(key, value) {
    const meta = modifierMeta(key);
    return meta.percent ? `${formatNumber(Number(value || 0) * 100, 1)}%` : formatNumber(value, 0);
  }

  function renderBuffList() {
    const buffList = $('shaft-buff-list');
    if (!buffList) {
      return;
    }
    const items = (state.axis.buff_rules || []).map((rule) => {
      const modifierLine = Object.entries(rule.modifiers || {})
        .map(([key, value]) => `${modifierMeta(key).label} ${modifierDisplay(key, value)}`)
        .join('，');
      const triggerAction = getActionMap().get(rule.trigger?.action_id);
      const targetSlots = (rule.targets?.slots || []).map((slot) => memberName(slot)).join('、') || '全部角色';
      const targetTypes = (rule.targets?.action_types || []).join('、') || '全部动作';
      return `
        <div class="shaft-buff-item">
          <div>
            <strong>${escapeHtml(rule.name)}</strong>
            <span>${escapeHtml(memberName(rule.trigger?.slot))} · ${escapeHtml(triggerAction?.name || '动作')} → ${escapeHtml(targetSlots)} · ${escapeHtml(targetTypes)} · ${ticksToSeconds(rule.duration_ticks)}s · ${escapeHtml(modifierLine)}</span>
          </div>
          <button class="secondary-btn shaft-remove-btn" data-remove-buff="${escapeHtml(rule.id)}" type="button" title="删除">×</button>
        </div>
      `;
    }).join('');
    buffList.innerHTML = items || '<div class="shaft-empty">暂无触发增益</div>';
  }

  function renderResults() {
    renderSelfCheck();
    const result = freshResult();
    const summary = result?.summary || {};
    const compareSummary = state.compareSnapshot?.summary || null;
    const compareControls = $('shaft-compare-controls');
    if (compareControls) {
      const savedText = state.compareSnapshot?.saved_at
        ? `快照 ${new Date(state.compareSnapshot.saved_at).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}`
        : '暂无快照';
      compareControls.innerHTML = `
        <div class="shaft-compare-row">
          <button class="secondary-btn" data-save-compare-snapshot type="button">保存为对比快照</button>
          <button class="secondary-btn" data-clear-compare-snapshot type="button" ${state.compareSnapshot ? '' : 'disabled'}>清除快照</button>
          <span class="shaft-compare-snapshot-status">${escapeHtml(savedText)}</span>
        </div>
      `;
    }
    renderResultCard('shaft-direct-damage', '角色伤害', summary.character_damage || 0, compareSummary?.character_damage, summary.total_damage || 0);
    renderResultCard('shaft-harmony-damage', '环合伤害', summary.harmony_damage || 0, compareSummary?.harmony_damage, summary.total_damage || 0);
    renderResultCard('shaft-stagger-damage', '倾陷伤害', summary.stagger_damage || 0, compareSummary?.stagger_damage, summary.total_damage || 0);
    renderResultCard('shaft-total-damage', '总伤', summary.total_damage || 0, compareSummary?.total_damage, summary.total_damage || 0);
    renderResultCard('shaft-dps', 'DPS', summary.dps || 0, compareSummary?.dps, null);
    const contribution = result?.damage_by_slot || [];
    const characterRows = contribution.map((item) => `
      <div class="shaft-contribution-row">
        <span>${escapeHtml(item.character_name)}</span>
        <div class="shaft-contribution-bar"><span style="width: ${Math.max(0, Math.min(100, Number(item.percent || 0)))}%; --contribution-color:${SLOT_COLORS[Number(item.slot) % SLOT_COLORS.length]}"></span></div>
        <span>${formatNumber(item.percent || 0, 1)}%</span>
        <button class="secondary-btn shaft-action-contribution-btn" data-action-contribution-slot="${Number(item.slot)}" type="button" ${Number(item.damage || 0) > 0 ? '' : 'disabled'}>分析详情</button>
      </div>
    `).join('');
    const independentRows = [
      {
        label: '环合伤害',
        damage: Number(summary.harmony_damage || 0),
        percent: Number(summary.total_damage || 0) > 0
          ? Number(summary.harmony_damage || 0) / Number(summary.total_damage) * 100
          : 0,
        color: DAMAGE_SOURCE_COLORS['创生'],
        trigger: 'data-open-harmony-analysis',
      },
      {
        label: '倾陷伤害',
        damage: Number(summary.stagger_damage || 0),
        percent: Number(summary.total_damage || 0) > 0
          ? Number(summary.stagger_damage || 0) / Number(summary.total_damage) * 100
          : 0,
        color: DAMAGE_SOURCE_COLORS['倾陷'],
        trigger: 'data-open-stagger-analysis',
      },
    ].map((item) => `
      <div class="shaft-contribution-row shaft-contribution-source-row">
        <span>${escapeHtml(item.label)}</span>
        <div class="shaft-contribution-bar"><span style="width: ${Math.max(0, Math.min(100, item.percent))}%; --contribution-color:${item.color}"></span></div>
        <span>${formatNumber(item.percent || 0, 1)}%</span>
        <button class="secondary-btn shaft-action-contribution-btn" ${item.trigger} type="button" aria-haspopup="dialog" ${item.damage > 0 ? '' : 'disabled'}>分析详情</button>
      </div>
    `).join('');
    $('shaft-contribution-list').innerHTML = (characterRows || independentRows)
      ? `${characterRows}${independentRows}`
      : '<div class="shaft-empty">点击计算刷新结果</div>';
    renderBuildPanel();
    renderWorkbenchSummary();
  }

  function actionContributionForSlot(slot, result = freshResult()) {
    const projected = (result?.damage_by_action_by_slot || []).find((item) => Number(item.slot) === Number(slot));
    if (projected) {
      const actionMap = getActionMap();
      return {
        ...projected,
        actions: (projected.actions || []).map((item) => {
          const action = actionMap.get(item.action_id) || {};
          return {
            ...item,
            action_type: item.action_type || action.action_type || '其他',
            damage_type: item.damage_type || action.damage_type || item.action_type || action.action_type || '其他',
            damage_element: item.damage_element || action.damage_element || '',
          };
        }),
      };
    }
    const details = (result?.details || []).filter((detail) => Number(detail.slot) === Number(slot));
    const actions = new Map();
    details.forEach((detail) => {
      const key = detail.action_id || detail.action_name;
      const current = actions.get(key) || {
        action_id: detail.action_id,
        action_name: detail.action_name,
        action_type: detail.action_type || '其他',
        damage_type: detail.damage_type || detail.action_type || '其他',
        damage_element: detail.damage_element || '',
        damage: 0,
      };
      current.damage += Number(detail.direct_damage || 0);
      actions.set(key, current);
    });
    const totalDamage = Array.from(actions.values()).reduce((sum, item) => sum + item.damage, 0);
    return {
      slot,
      character_name: memberName(slot),
      total_damage: totalDamage,
      actions: Array.from(actions.values())
        .filter((item) => item.damage > 0)
        .sort((left, right) => right.damage - left.damage)
        .map((item) => ({ ...item, percent: totalDamage > 0 ? item.damage / totalDamage * 100 : 0 })),
    };
  }

  function actionContributionGroups(contribution, dimension) {
    const actions = contribution?.actions || [];
    if (dimension === 'action') {
      return actions.map((action, index) => ({
        key: action.action_id || action.action_name || String(index),
        label: action.action_name || '未命名动作',
        detail: action.detail || action.damage_type || action.action_type || '其他',
        damage: Number(action.damage || 0),
        percent: Number(action.percent || 0),
        color: ACTION_CONTRIBUTION_COLORS[index % ACTION_CONTRIBUTION_COLORS.length],
      }));
    }
    const groups = new Map();
    const additionalTags = new Set(['追击', '附着']);
    actions.forEach((action) => {
      const damageType = String(action.damage_type || '');
      const actionType = String(action.action_type || '');
      const label = additionalTags.has(damageType)
        ? (actionType || '其他')
        : (damageType || actionType || '其他');
      const current = groups.get(label) || {
        key: label,
        label,
        detail: '合并同类型动作',
        damage: 0,
        actionCount: 0,
      };
      current.damage += Number(action.damage || 0);
      current.actionCount += 1;
      groups.set(label, current);
    });
    const total = Number(contribution?.total_damage || 0);
    const sortedGroups = Array.from(groups.values())
      .filter((item) => item.damage > 0)
      .sort((left, right) => right.damage - left.damage || left.label.localeCompare(right.label, 'zh-CN'));
    const reservedColors = new Set(
      sortedGroups
        .map((item) => ACTION_TYPE_COLORS[item.label])
        .filter(Boolean),
    );
    const fallbackColors = [
      ...ACTION_CONTRIBUTION_COLORS.filter((color) => !reservedColors.has(color)),
      ...ACTION_CONTRIBUTION_COLORS.filter((color) => reservedColors.has(color)),
    ];
    const assignedColors = new Set();
    return sortedGroups.map((item) => {
      let color = ACTION_TYPE_COLORS[item.label];
      if (color && assignedColors.has(color)) color = '';
      if (!color) {
        color = fallbackColors.find((candidate) => !assignedColors.has(candidate));
      }
      color ||= ACTION_CONTRIBUTION_COLORS[assignedColors.size % ACTION_CONTRIBUTION_COLORS.length];
      assignedColors.add(color);
      return {
        ...item,
        detail: `${item.actionCount} 个动作`,
        percent: total > 0 ? item.damage / total * 100 : 0,
        color,
      };
    });
  }

  function mergeActionAnalysisComparison(currentGroups, baselineGroups) {
    const baselineMap = new Map((baselineGroups || []).map((item) => [String(item.key), item]));
    const currentKeys = new Set(currentGroups.map((item) => String(item.key)));
    const merged = currentGroups.map((item) => {
      const baseline = baselineMap.get(String(item.key));
      return {
        ...item,
        baselineDamage: Number(baseline?.damage || 0),
        baselinePercent: Number(baseline?.percent || 0),
        damageDelta: Number(item.damage || 0) - Number(baseline?.damage || 0),
        percentDelta: Number(item.percent || 0) - Number(baseline?.percent || 0),
        isNew: !baseline,
      };
    });
    (baselineGroups || []).forEach((item) => {
      if (currentKeys.has(String(item.key))) return;
      merged.push({
        ...item,
        damage: 0,
        percent: 0,
        baselineDamage: Number(item.damage || 0),
        baselinePercent: Number(item.percent || 0),
        damageDelta: -Number(item.damage || 0),
        percentDelta: -Number(item.percent || 0),
        isRemoved: true,
      });
    });
    return merged;
  }

  function actionAnalysisDeltaText(value, digits = 0, suffix = '') {
    const raw = Number(value || 0);
    const numeric = Math.abs(raw) < (0.5 * (10 ** -digits)) ? 0 : raw;
    const sign = numeric > 0 ? '+' : '';
    return `${sign}${formatNumber(numeric, digits)}${suffix}`;
  }

  function actionAnalysisDeltaTone(value) {
    const numeric = Number(value || 0);
    if (Math.abs(numeric) < 0.005) return '';
    return numeric > 0 ? 'is-up' : 'is-down';
  }

  function actionAnalysisRelativePercent(current, baseline) {
    const currentDamage = Number(current || 0);
    const baselineDamage = Number(baseline || 0);
    if (baselineDamage <= 0) return currentDamage > 0 ? null : 0;
    return (currentDamage - baselineDamage) / baselineDamage * 100;
  }

  function actionAnalysisRelativePercentText(current, baseline) {
    const relativePercent = actionAnalysisRelativePercent(current, baseline);
    return relativePercent === null
      ? '新增'
      : actionAnalysisDeltaText(relativePercent, 1, '%');
  }

  function polarPoint(radius, angle) {
    const radians = (angle - 90) * Math.PI / 180;
    return {
      x: 50 + radius * Math.cos(radians),
      y: 50 + radius * Math.sin(radians),
    };
  }

  function donutSegmentPath(startAngle, endAngle) {
    const safeEnd = Math.min(endAngle, startAngle + 359.999);
    const outerStart = polarPoint(44, startAngle);
    const outerEnd = polarPoint(44, safeEnd);
    const innerEnd = polarPoint(25, safeEnd);
    const innerStart = polarPoint(25, startAngle);
    const largeArc = safeEnd - startAngle > 180 ? 1 : 0;
    return [
      `M ${outerStart.x} ${outerStart.y}`,
      `A 44 44 0 ${largeArc} 1 ${outerEnd.x} ${outerEnd.y}`,
      `L ${innerEnd.x} ${innerEnd.y}`,
      `A 25 25 0 ${largeArc} 0 ${innerStart.x} ${innerStart.y}`,
      'Z',
    ].join(' ');
  }

  function actionContributionPie(actions) {
    const groups = actions;
    let angle = 0;
    const segments = groups.map((group, index) => {
      const sweep = Math.max(0, Math.min(100, Number(group.percent || 0))) * 3.6;
      const startAngle = angle;
      const endAngle = angle + sweep;
      const middle = (startAngle + endAngle) / 2;
      angle = endAngle;
      return `
        <path
          class="shaft-action-contribution-segment"
          data-analysis-index="${index}"
          d="${donutSegmentPath(startAngle, endAngle)}"
          fill="${group.color}"
          style="--explode-x:${(Math.sin(middle * Math.PI / 180) * 2.4).toFixed(2)}px;--explode-y:${(-Math.cos(middle * Math.PI / 180) * 2.4).toFixed(2)}px"
          tabindex="0"
          role="button"
          aria-label="${escapeHtml(group.label)}，${formatNumber(group.percent || 0, 1)}%"
        >
          <title>${escapeHtml(group.label)} ${formatNumber(group.percent || 0, 1)}%</title>
        </path>
      `;
    }).join('');
    return `
      <svg class="shaft-action-contribution-pie" viewBox="0 0 100 100" role="img" aria-label="角色各动作伤害贡献饼图">
        ${segments}
      </svg>
    `;
  }

  function actionContributionBars(groups, hasComparison = false) {
    return `
      <div class="shaft-action-analysis-bars" role="img" aria-label="伤害贡献排行图">
        ${groups.map((group, index) => `
          <div class="shaft-action-analysis-bar" data-analysis-index="${index}" tabindex="0">
            <div>
              <span>${escapeHtml(group.label)}</span>
              <b>${formatNumber(group.percent || 0, 1)}%</b>
            </div>
            <i>
              ${hasComparison ? `<em style="left:${Math.max(0, Math.min(100, Number(group.baselinePercent || 0)))}%" title="快照占比 ${formatNumber(group.baselinePercent || 0, 1)}%"></em>` : ''}
              <span style="width:${Math.max(0, Number(group.percent || 0))}%;--action-color:${group.color}"></span>
            </i>
          </div>
        `).join('')}
      </div>
    `;
  }

  function renderActionContributionAnalysis() {
    const content = $('shaft-action-contribution-content');
    const contribution = actionContributionForSlot(actionAnalysisState.slot);
    if (!content || !contribution) {
      return;
    }
    const currentGroups = actionContributionGroups(contribution, actionAnalysisState.dimension);
    if (!currentGroups.length) {
      content.innerHTML = '<div class="shaft-empty">该角色当前没有造成动作伤害</div>';
      return;
    }
    const snapshotHasDetails = Array.isArray(state.compareSnapshot?.damage_by_action_by_slot);
    const baselineContribution = snapshotHasDetails
      ? actionContributionForSlot(actionAnalysisState.slot, {
        damage_by_action_by_slot: state.compareSnapshot.damage_by_action_by_slot,
      })
      : null;
    const baselineGroups = baselineContribution
      ? actionContributionGroups(baselineContribution, actionAnalysisState.dimension)
      : [];
    const hasComparison = Boolean(baselineContribution);
    const groups = hasComparison
      ? mergeActionAnalysisComparison(currentGroups, baselineGroups)
      : currentGroups;
    const top = currentGroups[0];
    const topThreePercent = currentGroups.slice(0, 3).reduce((sum, item) => sum + Number(item.percent || 0), 0);
    const baselineTopThreePercent = baselineGroups.slice(0, 3).reduce((sum, item) => sum + Number(item.percent || 0), 0);
    const baselineTopMap = new Map(baselineGroups.map((item) => [String(item.key), item]));
    const topBaseline = baselineTopMap.get(String(top.key));
    const totalDelta = Number(contribution.total_damage || 0) - Number(baselineContribution?.total_damage || 0);
    const countDelta = currentGroups.length - baselineGroups.length;
    const chart = actionAnalysisState.view === 'bars'
      ? actionContributionBars(groups, hasComparison)
      : actionContributionPie(currentGroups);
    const snapshotTime = state.compareSnapshot?.saved_at
      ? new Date(state.compareSnapshot.saved_at).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
      : '';
    content.innerHTML = `
      <div class="shaft-action-analysis-toolbar">
        <div class="shaft-action-analysis-switch" role="tablist" aria-label="分析维度">
          <button class="${actionAnalysisState.dimension === 'action' ? 'is-active' : ''}" data-analysis-dimension="action" type="button" role="tab" aria-selected="${actionAnalysisState.dimension === 'action'}">动作明细</button>
          <button class="${actionAnalysisState.dimension === 'type' ? 'is-active' : ''}" data-analysis-dimension="type" type="button" role="tab" aria-selected="${actionAnalysisState.dimension === 'type'}">合并伤害类型</button>
        </div>
        <div class="shaft-action-analysis-switch shaft-action-analysis-view-switch" aria-label="图表类型">
          <button class="${actionAnalysisState.view === 'donut' ? 'is-active' : ''}" data-analysis-view="donut" type="button">环形占比</button>
          <button class="${actionAnalysisState.view === 'bars' ? 'is-active' : ''}" data-analysis-view="bars" type="button">排行条形</button>
        </div>
      </div>
      ${hasComparison ? `
        <div class="shaft-action-analysis-snapshot">
          <div>
            <span>对比快照 · ${escapeHtml(snapshotTime)}</span>
            <strong>
              ${formatNumber(baselineContribution.total_damage || 0)} → ${formatNumber(contribution.total_damage || 0)}
              <small class="shaft-action-analysis-relative ${actionAnalysisDeltaTone(totalDelta)}">${actionAnalysisRelativePercentText(contribution.total_damage, baselineContribution.total_damage)}</small>
            </strong>
          </div>
          <b class="${actionAnalysisDeltaTone(totalDelta)}">${actionAnalysisDeltaText(totalDelta)}</b>
          <small>排行图中的竖线表示快照占比</small>
        </div>
      ` : `
        <div class="shaft-action-analysis-snapshot shaft-action-analysis-snapshot-empty">
          <span>${state.compareSnapshot ? '当前快照仅包含总览数据，请重新保存以分析动作变化' : '保存当前结果为快照后，可分析每个动作和伤害类型的变化'}</span>
          <button class="secondary-btn" data-save-action-snapshot type="button">${state.compareSnapshot ? '更新快照' : '保存当前快照'}</button>
        </div>
      `}
      <div class="shaft-action-analysis-kpis">
        <div><span>角色总伤</span><strong>${formatNumber(contribution.total_damage || 0)}</strong>${hasComparison ? `<small class="${actionAnalysisDeltaTone(totalDelta)}">较快照 ${actionAnalysisDeltaText(totalDelta)}</small>` : ''}</div>
        <div><span>${actionAnalysisState.dimension === 'action' ? '有效动作' : '伤害类型'}</span><strong>${currentGroups.length}</strong>${hasComparison ? `<small class="${actionAnalysisDeltaTone(countDelta)}">较快照 ${actionAnalysisDeltaText(countDelta)}</small>` : ''}</div>
        <div><span>最高贡献</span><strong>${formatNumber(top.percent || 0, 1)}%</strong><small>${escapeHtml(top.label)}${hasComparison ? ` · ${actionAnalysisDeltaText(Number(top.percent || 0) - Number(topBaseline?.percent || 0), 1, 'pp')}` : ''}</small></div>
        <div><span>前三集中度</span><strong>${formatNumber(topThreePercent, 1)}%</strong>${hasComparison ? `<small class="${actionAnalysisDeltaTone(topThreePercent - baselineTopThreePercent)}">较快照 ${actionAnalysisDeltaText(topThreePercent - baselineTopThreePercent, 1, 'pp')}</small>` : ''}</div>
      </div>
      <div class="shaft-action-contribution-body" data-analysis-chart-root>
        <div class="shaft-action-contribution-chart shaft-action-contribution-chart-${actionAnalysisState.view}">
          ${chart}
          ${actionAnalysisState.view === 'donut' ? `
            <div class="shaft-action-chart-center">
              <strong data-analysis-center-value>${formatNumber(contribution.total_damage || 0)}</strong>
              <span data-analysis-center-label>角色总伤</span>
            </div>
          ` : ''}
        </div>
        <div class="shaft-action-contribution-legend">
          ${groups.map((group, index) => `
            <button class="shaft-action-contribution-item ${group.isRemoved ? 'is-removed' : ''}" data-analysis-index="${index}" style="--action-color:${group.color}" type="button">
              <i style="--action-color:${group.color}"></i>
              <span>
                <b>${escapeHtml(group.label)}${group.isNew ? ' · 新增' : ''}${group.isRemoved ? ' · 已移除' : ''}</b>
                <small>${escapeHtml(group.detail || '')}${hasComparison ? ` · 快照 ${formatNumber(group.baselineDamage || 0)}` : ''}</small>
              </span>
              <strong>
                ${formatNumber(group.damage || 0)}
                ${hasComparison ? `<small class="shaft-action-analysis-relative ${actionAnalysisDeltaTone(group.damageDelta)}">${actionAnalysisRelativePercentText(group.damage, group.baselineDamage)}</small>` : ''}
              </strong>
              <em>
                ${formatNumber(group.percent || 0, 1)}%
                ${hasComparison ? `<small class="${actionAnalysisDeltaTone(group.percentDelta)}">${actionAnalysisDeltaText(group.percentDelta, 1, 'pp')}</small>` : ''}
              </em>
            </button>
          `).join('')}
        </div>
      </div>
    `;
    content._analysisGroups = groups;
    content._analysisTotal = Number(contribution.total_damage || 0);
  }

  function openActionContribution(slot, trigger = null) {
    const dialog = $('shaft-action-contribution-dialog');
    const contribution = actionContributionForSlot(slot);
    if (!dialog || !contribution) {
      return;
    }
    $('shaft-action-contribution-title').textContent = `${contribution.character_name || memberName(slot)} · 动作贡献`;
    actionAnalysisState.slot = Number(slot);
    actionAnalysisState.dimension = 'action';
    actionAnalysisState.view = 'donut';
    renderActionContributionAnalysis();
    dialog._returnFocus = trigger;
    dialog.showModal();
  }

  function setActionAnalysisHighlight(index, active) {
    const content = $('shaft-action-contribution-content');
    if (!content) return;
    content.querySelectorAll('[data-analysis-index]').forEach((node) => {
      node.classList.toggle('is-active', active && Number(node.dataset.analysisIndex) === Number(index));
      node.classList.toggle('is-muted', active && Number(node.dataset.analysisIndex) !== Number(index));
    });
    const group = content._analysisGroups?.[Number(index)];
    const value = content.querySelector('[data-analysis-center-value]');
    const label = content.querySelector('[data-analysis-center-label]');
    if (value && label) {
      value.textContent = active && group
        ? `${formatNumber(group.percent || 0, 1)}%`
        : formatNumber(content._analysisTotal || 0);
      label.textContent = active && group ? group.label : '角色总伤';
    }
  }

  function closeActionContribution() {
    const dialog = $('shaft-action-contribution-dialog');
    if (!dialog?.open) {
      return;
    }
    const returnFocus = dialog._returnFocus;
    dialog.close();
    if (returnFocus?.isConnected) {
      returnFocus.focus();
    }
  }

  function renderStaggerAnalysis() {
    const content = $('shaft-stagger-analysis-content');
    const result = freshResult();
    if (!content || !result) {
      return;
    }
    const summary = result.summary || {};
    const contributions = (result.stagger_contributions_by_slot || [])
      .slice()
      .sort((left, right) => Number(right.damage || 0) - Number(left.damage || 0));
    if (!contributions.length || Number(summary.stagger_damage || 0) <= 0) {
      content.innerHTML = '<div class="shaft-empty">当前轴没有产生倾陷伤害</div>';
      return;
    }
    content.innerHTML = `
      <div class="shaft-stagger-analysis-note">倾陷伤害独立计入全队总伤和 DPS，不计入角色自身伤害占比或动作贡献。</div>
      <div class="shaft-action-analysis-kpis shaft-stagger-analysis-kpis">
        <div><span>倾陷总伤</span><strong>${formatNumber(summary.stagger_damage || 0)}</strong></div>
        <div><span>单次倾陷</span><strong>${formatNumber(summary.stagger_damage_per_trigger || 0)}</strong></div>
        <div><span>倾陷频率</span><strong>${formatNumber(summary.stagger_frequency || 0, 3)}</strong><small>本轴期望触发次数</small></div>
        <div><span>恢复时间</span><strong>${formatNumber(summary.stagger_recovery_seconds || 0, 1)}s</strong><small>全轴倾陷 ${formatNumber(summary.total_stagger || 0, 1)}</small></div>
      </div>
      <div class="shaft-stagger-contribution-list">
        ${contributions.map((item) => `
          <article class="shaft-stagger-contribution-card" style="--stagger-color:${SLOT_COLORS[Number(item.slot) % SLOT_COLORS.length]}">
            <div class="shaft-stagger-contribution-head">
              <div>
                <span>${escapeHtml(item.character_name || memberName(item.slot))}</span>
                <strong>${formatNumber(item.damage || 0)}</strong>
              </div>
              <b>${formatNumber(item.percent || 0, 1)}%</b>
            </div>
            <div class="shaft-stagger-contribution-meter"><span style="width:${Math.max(0, Math.min(100, Number(item.percent || 0)))}%"></span></div>
            <dl>
              <div><dt>单次贡献</dt><dd>${formatNumber(item.damage_per_trigger || 0)}</dd></div>
              <div><dt>平均倾陷增伤</dt><dd>${formatNumber(Number(item.average_stagger_damage_bonus || 0) * 100, 1)}%</dd></div>
              <div><dt>平均穿防</dt><dd>${formatNumber(Number(item.average_def_ignore || 0) * 100, 1)}%</dd></div>
              <div><dt>平均减防 / 减抗</dt><dd>${formatNumber(Number(item.average_def_down || 0) * 100, 1)}% / ${formatNumber(Number(item.average_res_down || 0) * 100, 1)}%</dd></div>
            </dl>
          </article>
        `).join('')}
      </div>
    `;
  }

  function renderHarmonyAnalysis() {
    const content = $('shaft-harmony-analysis-content');
    const result = freshResult();
    if (!content || !result) return;
    const summary = result.summary || {};
    const contributions = (result.harmony_contributions_by_slot || [])
      .filter((item) => Number(item.damage || 0) > 0)
      .slice()
      .sort((left, right) => Number(right.damage || 0) - Number(left.damage || 0));
    if (!contributions.length || Number(summary.harmony_damage || 0) <= 0) {
      content.innerHTML = '<div class="shaft-empty">当前轴没有产生环合伤害</div>';
      return;
    }
    content.innerHTML = `
      <div class="shaft-stagger-analysis-note">环合伤害独立计入全队总伤和 DPS，不计入角色自身伤害占比或动作贡献。</div>
      <div class="shaft-action-analysis-kpis shaft-stagger-analysis-kpis">
        <div><span>环合总伤</span><strong>${formatNumber(summary.harmony_damage || 0)}</strong></div>
        <div><span>环合类型</span><strong>${new Set(contributions.flatMap((item) => (item.sources || []).map((source) => source.source))).size}</strong></div>
        <div><span>贡献角色</span><strong>${contributions.length}</strong></div>
        <div><span>占全队总伤</span><strong>${formatNumber(Number(summary.harmony_damage || 0) / Math.max(Number(summary.total_damage || 0), 1) * 100, 1)}%</strong></div>
      </div>
      <div class="shaft-stagger-contribution-list">
        ${contributions.map((item) => `
          <article class="shaft-stagger-contribution-card" style="--stagger-color:${SLOT_COLORS[Number(item.slot) % SLOT_COLORS.length]}">
            <div class="shaft-stagger-contribution-head">
              <div><span>${escapeHtml(item.character_name || memberName(item.slot))}</span><strong>${formatNumber(item.damage || 0)}</strong></div>
              <b>${formatNumber(item.percent || 0, 1)}%</b>
            </div>
            <div class="shaft-stagger-contribution-meter"><span style="width:${Math.max(0, Math.min(100, Number(item.percent || 0)))}%"></span></div>
            <dl class="shaft-harmony-source-list">
              ${(item.sources || []).map((source) => `
                <div><dt>${escapeHtml(source.source)}</dt><dd>${formatNumber(source.damage || 0)} · ${formatNumber(source.percent || 0, 1)}%</dd></div>
              `).join('')}
            </dl>
          </article>
        `).join('')}
      </div>
    `;
  }

  function openHarmonyAnalysis(trigger = null) {
    const dialog = $('shaft-harmony-analysis-dialog');
    if (!dialog || Number(freshResult()?.summary?.harmony_damage || 0) <= 0) return;
    renderHarmonyAnalysis();
    dialog._returnFocus = trigger;
    dialog.showModal();
  }

  function closeHarmonyAnalysis() {
    const dialog = $('shaft-harmony-analysis-dialog');
    if (!dialog?.open) return;
    const returnFocus = dialog._returnFocus;
    dialog.close();
    if (returnFocus?.isConnected) returnFocus.focus();
  }

  function openStaggerAnalysis(trigger = null) {
    const dialog = $('shaft-stagger-analysis-dialog');
    if (!dialog || Number(freshResult()?.summary?.stagger_damage || 0) <= 0) {
      return;
    }
    renderStaggerAnalysis();
    dialog._returnFocus = trigger;
    dialog.showModal();
  }

  function closeStaggerAnalysis() {
    const dialog = $('shaft-stagger-analysis-dialog');
    if (!dialog?.open) {
      return;
    }
    const returnFocus = dialog._returnFocus;
    dialog.close();
    if (returnFocus?.isConnected) {
      returnFocus.focus();
    }
  }

  function openShortcutHelp() {
    const dialog = $('shaft-shortcut-dialog');
    if (dialog && !dialog.open) {
      dialog.showModal();
    }
  }

  function closeShortcutHelp() {
    const dialog = $('shaft-shortcut-dialog');
    if (dialog?.open) {
      dialog.close();
      $('shaft-shortcut-help-btn')?.focus();
    }
  }

  function openBuildInfo() {
    const dialog = $('shaft-build-info-dialog');
    if (dialog && !dialog.open) {
      dialog.showModal();
    }
  }

  function closeBuildInfo() {
    const dialog = $('shaft-build-info-dialog');
    if (dialog?.open) {
      dialog.close();
      $('shaft-build-info-btn')?.focus();
    }
  }

  function previewDetails() {
    const previewPayload = state.axisPreviewPayload;
    if (!previewPayload && !state.timelineDisplayDetails.length) {
      renderTimeline();
    }
    const previewAxis = previewPayload?.axis || state.axis;
    const sourceDetails = previewPayload?.result?.details || state.timelineDisplayDetails;
    const stepById = new Map((previewAxis?.steps || []).map((step) => [step.id, step]));
    return sourceDetails
      .filter((detail) => {
        const step = stepById.get(detail.step_id) || { action_id: detail.action_id };
        const action = actionForStep(step);
        return startsForeground(step, action);
      })
      .map((detail) => {
        const startTick = Math.max(0, Number(detail.display_start_tick ?? detail.start_tick ?? 0));
        const durationTicks = Math.max(0, Number(detail.display_duration_ticks ?? detail.duration_ticks ?? 0));
        const visualEndTick = Math.max(
          startTick + (durationTicks > 0 ? durationTicks : ZERO_ACTION_VISUAL_TICKS),
          Number(detail.display_visual_end_tick ?? detail.visual_end_tick ?? detail.end_tick ?? startTick),
        );
        return {
          slot: Number(detail.slot || 0),
          name: detail.action_name || actionForStep({ action_id: detail.action_id }).name || '',
          startTick,
          endTick: Math.max(startTick, startTick + durationTicks),
          visualEndTick,
          durationTicks,
        };
      });
  }

  function groupedPreviewActions(details, slot) {
    const groups = [];
    details
      .filter((detail) => Number(detail.slot) === Number(slot))
      .sort((left, right) => left.startTick - right.startTick || left.visualEndTick - right.visualEndTick)
      .forEach((detail) => {
        const previous = groups[groups.length - 1];
        if (previous && detail.startTick <= previous.visualEndTick) {
          previous.names.push(detail.name);
          previous.endTick = Math.max(previous.endTick, detail.endTick);
          previous.visualEndTick = Math.max(previous.visualEndTick, detail.visualEndTick);
          previous.durationTicks += detail.durationTicks;
          return;
        }
        groups.push({
          names: [detail.name],
          startTick: detail.startTick,
          endTick: detail.endTick,
          visualEndTick: detail.visualEndTick,
          durationTicks: detail.durationTicks,
        });
      });
    return groups;
  }

  function previewEndTick(details = previewDetails()) {
    return Math.max(
      1,
      ...details.map((detail) => Math.max(detail.visualEndTick, detail.endTick)),
    );
  }

  function previewResult() {
    return state.axisPreviewPayload?.result || freshResult() || state.result || null;
  }

  function previewDamageTypeShares(result = previewResult()) {
    const damageByType = new Map();
    const additionalTags = new Set(['追击', '附着']);
    (result?.damage_by_action_by_slot || []).forEach((contribution) => {
      (contribution.actions || []).forEach((action) => {
        const damageType = String(action.damage_type || '');
        const actionType = String(action.action_type || '');
        const label = additionalTags.has(damageType)
          ? (actionType || '其他')
          : (damageType || actionType || '其他');
        damageByType.set(label, (damageByType.get(label) || 0) + Math.max(0, Number(action.damage || 0)));
      });
    });
    const summary = result?.summary || {};
    const harmonyDamage = Math.max(0, Number(summary.harmony_damage || 0));
    const staggerDamage = Math.max(0, Number(summary.stagger_damage || 0));
    if (harmonyDamage > 0) damageByType.set('环合', (damageByType.get('环合') || 0) + harmonyDamage);
    if (staggerDamage > 0) damageByType.set('倾陷', (damageByType.get('倾陷') || 0) + staggerDamage);
    const totalDamage = Math.max(
      0,
      Number(summary.total_damage || 0),
      Array.from(damageByType.values()).reduce((sum, damage) => sum + damage, 0),
    );
    return Array.from(damageByType.entries())
      .filter(([, damage]) => damage > 0)
      .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0], 'zh-CN'))
      .map(([label, damage]) => ({
        label,
        percent: totalDamage > 0 ? Math.round(damage / totalDamage * 100) : 0,
      }));
  }

  function renderAxisPreviewSummary() {
    const node = $('shaft-axis-preview-summary');
    if (!node) return;
    const result = previewResult();
    const summary = result?.summary || {};
    const damageTypes = previewDamageTypeShares(result);
    node.innerHTML = `
      <div class="shaft-axis-preview-stat"><span>DPS</span><strong>${formatNumber(summary.dps || 0)}</strong></div>
      <div class="shaft-axis-preview-stat"><span>轴长</span><strong>${formatNumber(summary.duration_seconds || 0, 1)}s</strong></div>
      <div class="shaft-axis-preview-damage-types">
        <span>伤害类型</span>
        <div>${damageTypes.length
          ? damageTypes.map((item) => `<b>${escapeHtml(item.label)} ${item.percent}%</b>`).join('')
          : '<b>暂无伤害</b>'}</div>
      </div>
    `;
  }

  function prioritizeAxisPreviewActionLabels() {
    $('shaft-axis-preview-canvas')?.querySelectorAll('.shaft-axis-preview-bubble').forEach((bubble) => {
      const name = bubble.querySelector(':scope > span');
      const duration = bubble.querySelector(':scope > em');
      if (!name || !duration) return;
      bubble.classList.remove('hide-duration');
      const styles = window.getComputedStyle(bubble);
      const availableWidth = bubble.clientWidth
        - Number.parseFloat(styles.paddingLeft || 0)
        - Number.parseFloat(styles.paddingRight || 0);
      if (name.scrollWidth + duration.scrollWidth + 6 > availableWidth) {
        bubble.classList.add('hide-duration');
      }
    });
  }

  function previewMinimumTickPx(details = previewDetails()) {
    const viewport = $('shaft-axis-preview-viewport');
    const availableWidth = Math.max(PREVIEW_MIN_BODY_PX, Number(viewport?.clientWidth || 0) - PREVIEW_LABEL_PX - 20);
    return Math.min(PREVIEW_MAX_TICK_PX, availableWidth / previewEndTick(details));
  }

  function renderAxisPreview() {
    const canvas = $('shaft-axis-preview-canvas');
    if (!canvas || !$('shaft-axis-preview-dialog')?.open) {
      return;
    }
    const details = previewDetails();
    const endTick = previewEndTick(details);
    const minTickPx = previewMinimumTickPx(details);
    state.previewTickPx = Math.max(minTickPx, Math.min(PREVIEW_MAX_TICK_PX, state.previewTickPx || minTickPx));
    const tickPx = state.previewTickPx;
    const bodyWidth = Math.max(1, endTick * tickPx);
    const totalWidth = PREVIEW_LABEL_PX + bodyWidth;
    const leftPx = (tick) => PREVIEW_LABEL_PX + Math.max(0, Number(tick || 0)) * tickPx;
    const widthPx = (start, end) => Math.max(4, (Math.max(Number(end || start), Number(start || 0) + 0.1) - Number(start || 0)) * tickPx);
    const rulerStepTicks = tickPx >= 8 ? 10 : (tickPx >= 3 ? 20 : 50);
    const rulerMarks = [];
    for (let tick = 0; tick <= endTick; tick += rulerStepTicks) {
      rulerMarks.push(`
        <span class="shaft-axis-preview-mark" style="left:${leftPx(tick)}px">
          <span>${escapeHtml(ticksToSeconds(tick))}s</span>
        </span>
      `);
    }
    const previewAxis = state.axisPreviewPayload?.axis || state.axis;
    const previewTeam = state.axisPreviewPayload?.team || previewAxis?.team || [];
    const tracks = previewTeam.slice(0, 4).map((member) => {
      const color = SLOT_COLORS[Number(member.slot) % SLOT_COLORS.length];
      const bubbles = groupedPreviewActions(details, member.slot).map((group) => {
        const joinedName = group.names.filter(Boolean).join(' ');
        const visibleEndTick = Math.max(group.visualEndTick, group.startTick + group.durationTicks);
        const duration = group.durationTicks > 0 ? `<em>${escapeHtml(ticksToSeconds(group.durationTicks))}s</em>` : '';
        return `
          <span
            class="shaft-axis-preview-bubble"
            style="left:${leftPx(group.startTick)}px; width:${widthPx(group.startTick, visibleEndTick)}px; --slot-color:${color}"
            title="${escapeHtml(joinedName)}${group.durationTicks > 0 ? ` · ${escapeHtml(ticksToSeconds(group.durationTicks))}s` : ''}"
          >
            <span>${escapeHtml(joinedName)}</span>
            ${duration}
          </span>
        `;
      }).join('');
      return `
        <div class="shaft-axis-preview-track" style="width:${totalWidth}px">
          <span class="shaft-axis-preview-member">
            <img src="${escapeHtml(member.character_avatar || '')}" alt="">
            <span>${escapeHtml(member.character_name || '')}</span>
          </span>
          <span class="shaft-axis-preview-grid" style="background-size:${Math.max(1, tickPx * 10)}px 100%"></span>
          ${bubbles}
        </div>
      `;
    }).join('');
    canvas.style.width = `${totalWidth}px`;
    canvas.style.setProperty('--preview-label-width', `${PREVIEW_LABEL_PX}px`);
    canvas.innerHTML = `
      <div class="shaft-axis-preview-ruler" style="width:${totalWidth}px">
        <span class="shaft-axis-preview-ruler-label">时间</span>
        ${rulerMarks.join('')}
      </div>
      ${tracks || '<div class="shaft-empty">当前动作轴没有角色</div>'}
    `;
    renderAxisPreviewSummary();
    prioritizeAxisPreviewActionLabels();
  }

  function fitAxisPreview() {
    state.previewTickPx = previewMinimumTickPx();
    renderAxisPreview();
    $('shaft-axis-preview-viewport').scrollLeft = 0;
  }

  function openAxisPreview(trigger = null, payload = null) {
    const dialog = $('shaft-axis-preview-dialog');
    if (!dialog || dialog.open) {
      return;
    }
    state.axisPreviewPayload = payload;
    const marketPreview = Boolean(payload);
    const ownerName = String(payload?.owner?.nickname || payload?.owner?.player_uid || '').trim();
    $('shaft-axis-preview-title').textContent = marketPreview ? String(payload.title || '未命名排轴') : '动作轴预览';
    $('shaft-axis-preview-meta').hidden = !marketPreview;
    $('shaft-axis-preview-meta').textContent = marketPreview
      ? `${ownerName || '未知作者'} · ${String(payload.description || '').trim() || '暂无备注'}`
      : '';
    $('shaft-axis-preview-footer').hidden = !marketPreview;
    dialog._returnFocus = trigger;
    dialog.showModal();
    state.previewTickPx = 0;
    requestAnimationFrame(fitAxisPreview);
  }

  function closeAxisPreview() {
    const dialog = $('shaft-axis-preview-dialog');
    if (!dialog?.open) {
      return;
    }
    const returnFocus = dialog._returnFocus;
    dialog.close();
    state.axisPreviewPayload = null;
    state.axisPreviewSaving = false;
    $('shaft-axis-preview-save-btn').disabled = false;
    $('shaft-axis-preview-save-btn').textContent = '保存到本地';
    if (returnFocus?.isConnected) {
      returnFocus.focus();
    }
  }

  function marketAxisLocalTitle(payload) {
    const author = String(payload?.owner?.nickname || payload?.owner?.player_uid || '作者').trim() || '作者';
    const suffix = ` - ${author}`;
    const maxBaseLength = Math.max(1, 80 - Array.from(suffix).length);
    const base = Array.from(String(payload?.title || '未命名排轴').trim() || '未命名排轴')
      .slice(0, maxBaseLength)
      .join('');
    return `${base}${suffix}`;
  }

  async function openMarketAxisPreview(axisId, trigger) {
    try {
      setStatus('读取广场排轴');
      const payload = await shaftRequest(`/api/shaft/axes/${axisId}`);
      openAxisPreview(trigger, payload);
      setStatus('只读预览');
    } catch (error) {
      setStatus(error.message, 'error');
    }
  }

  async function saveMarketAxisToLocal() {
    const payload = state.axisPreviewPayload;
    if (!payload || state.axisPreviewSaving) {
      return;
    }
    if (!getToken()) {
      persistAxisDraft();
      redirectToLogin();
      return;
    }
    state.axisPreviewSaving = true;
    const button = $('shaft-axis-preview-save-btn');
    button.disabled = true;
    button.textContent = '保存中';
    try {
      const saved = await shaftRequest('/api/shaft/axes', {
        method: 'POST',
        body: JSON.stringify({
          title: marketAxisLocalTitle(payload),
          description: String(payload.description || ''),
          axis: payload.axis,
          result: payload.result,
        }),
      }, { authRequired: true });
      await loadMyAxes();
      closeAxisPreview();
      setStatus('已保存到本地');
      showToast(`已保存为「${saved.title}」`);
    } catch (error) {
      state.axisPreviewSaving = false;
      button.disabled = false;
      button.textContent = '保存到本地';
      setStatus(error.message, 'error');
    }
  }

  function handleAxisPreviewWheel(event) {
    const viewport = $('shaft-axis-preview-viewport');
    if (!viewport || !$('shaft-axis-preview-dialog')?.open) {
      return;
    }
    event.preventDefault();
    const details = previewDetails();
    const minTickPx = previewMinimumTickPx(details);
    const previousTickPx = Math.max(minTickPx, state.previewTickPx || minTickPx);
    const nextTickPx = Math.max(
      minTickPx,
      Math.min(PREVIEW_MAX_TICK_PX, previousTickPx * Math.exp(-event.deltaY * 0.0015)),
    );
    if (Math.abs(nextTickPx - previousTickPx) < 0.001) {
      return;
    }
    const bounds = viewport.getBoundingClientRect();
    const pointerX = Math.max(PREVIEW_LABEL_PX, event.clientX - bounds.left + viewport.scrollLeft);
    const anchorTick = Math.max(0, (pointerX - PREVIEW_LABEL_PX) / previousTickPx);
    const localPointerX = event.clientX - bounds.left;
    state.previewTickPx = nextTickPx;
    renderAxisPreview();
    viewport.scrollLeft = Math.max(0, PREVIEW_LABEL_PX + anchorTick * nextTickPx - localPointerX);
  }

  function handleContributionClick(event) {
    const actionButton = event.target.closest('[data-action-contribution-slot]');
    if (actionButton) {
      openActionContribution(Number(actionButton.dataset.actionContributionSlot), actionButton);
      return;
    }
    const harmonyButton = event.target.closest('[data-open-harmony-analysis]');
    if (harmonyButton) {
      openHarmonyAnalysis(harmonyButton);
      return;
    }
    const staggerButton = event.target.closest('[data-open-stagger-analysis]');
    if (staggerButton) openStaggerAnalysis(staggerButton);
  }

  function renderSelfCheck() {
    const node = $('shaft-self-check');
    if (!node) {
      return;
    }
    const axisWarnings = window.ShaftSelfCheck?.inspectAxis(state.axis, state.catalog) || [];
    const simulationWarnings = (freshResult()?.details || []).flatMap((detail) =>
      (detail.warnings || []).map((warning) =>
        `${detail.character_name || memberName(detail.slot)}「${detail.action_name || '动作'}」：${warning}`
      )
    );
    const warnings = Array.from(new Set([...axisWarnings, ...simulationWarnings]));
    node.hidden = warnings.length === 0;
    node.innerHTML = warnings.length ? `
      <strong>自检提示</strong>
      <ul>${warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join('')}</ul>
    ` : '';
  }

  function comparePercent(current, baseline) {
    if (baseline == null || !Number(baseline)) {
      return null;
    }
    return (Number(current || 0) / Number(baseline)) * 100;
  }

  function valueToneClass(percent) {
    if (percent == null || Math.abs(percent - 100) < 0.05) {
      return '';
    }
    return percent > 100 ? ' compare-up' : ' compare-down';
  }

  function damagePercentText(value, total) {
    if (total == null || !Number(total)) {
      return '';
    }
    return `${formatNumber((Number(value || 0) / Number(total)) * 100, 1)}%`;
  }

  function renderResultCard(nodeId, label, currentValue, baselineValue, currentTotal) {
    const node = $(nodeId);
    if (!node) {
      return;
    }
    const percent = comparePercent(currentValue, baselineValue);
    const showComparePercent = percent != null && Math.abs(percent - 100) >= 0.05;
    const currentPercent = damagePercentText(currentValue, currentTotal);
    const totalLine = currentPercent && currentPercent !== '100.0%' ? `当前占总伤 ${currentPercent}` : '';
    node.className = `shaft-stat-value${valueToneClass(percent)}`;
    node.innerHTML = `
      <span>${formatNumber(currentValue || 0)}</span>
      ${showComparePercent ? `<b>${formatNumber(percent, 1)}%</b>` : ''}
      ${totalLine ? `<small>${escapeHtml(totalLine)}</small>` : ''}
    `;
    node.setAttribute('aria-label', `${label} ${node.textContent}`);
  }

  function renderBuildPanel() {
    const node = $('shaft-build-panel');
    if (!node) {
      return;
    }
    const team = state.axis?.team || [];
    if (!team.length) {
      node.innerHTML = '<div class="shaft-empty">先选择队伍</div>';
      return;
    }
    if (!team.some((member) => Number(member.slot) === Number(state.buildPanelSlot))) {
      state.buildPanelSlot = Number(team[0]?.slot || 0);
    }
    const panels = currentBuildPanels();
    const panel = panels.find((item) => Number(item.slot) === Number(state.buildPanelSlot));
    const member = memberBySlot(state.buildPanelSlot) || team[0] || {};
    const panelStats = panel?.panel || {};
    const baseStats = panel?.base_stats || {};
    const zones = panel?.zones || {};
    const buildOptions = panel?.build_options || {};
    const curtain = buildOptions.curtain_bonus || {};
    const avatarButtons = team.map((item) => {
      const active = Number(item.slot) === Number(state.buildPanelSlot) ? 'active' : '';
      const color = SLOT_COLORS[Number(item.slot) % SLOT_COLORS.length];
      const awakeningCount = activeAwakeningCount(item);
      return `
        <button class="shaft-panel-avatar ${active}" data-build-panel-slot="${item.slot}" type="button" style="--slot-color:${color}" title="${escapeHtml(item.character_name)} · 已激活 ${awakeningCount} 个觉醒">
          <img src="${escapeHtml(item.character_avatar || '')}" alt="">
          <b aria-label="已激活 ${awakeningCount} 个觉醒">${awakeningCount}</b>
        </button>
      `;
    }).join('');
    if (!panel) {
      node.innerHTML = `
        <div class="shaft-panel-avatar-row">${avatarButtons}</div>
        <div class="shaft-empty">等待计算面板</div>
      `;
      return;
    }
    const mainStats = [
      { label: '攻击', key: 'atk', pct: 'atk_pct', flat: 'flat_atk' },
      { label: '生命', key: 'hp', pct: 'hp_pct', flat: 'flat_hp' },
      { label: '防御', key: 'def', pct: 'def_pct', flat: 'flat_def' },
    ].map((item) => `
      <div class="shaft-panel-major">
        <span>${item.label}</span>
        <strong>${formatNumber(panelStats[item.key] || 0)}</strong>
        <small>基础 ${formatNumber(baseStats[item.key] || 0)} · ${formatPercent(zones[item.pct] || 0)} · 固定 ${formatNumber(zones[item.flat] || 0)}</small>
      </div>
    `).join('');
    const minorStats = [
      ['暴击', formatPercent(panelStats.crit_rate || 0)],
      ['暴伤', formatPercent(panelStats.crit_dmg || 0)],
      ['环合强度', formatNumber(panelStats.harmony_strength || 0)],
      ['倾陷强度', formatNumber(panelStats.stagger_strength || 0)],
      ['通伤', formatPercent(panelStats.all_dmg || 0)],
      ['属伤', formatPercent(panelStats.element_dmg || 0)],
      ['充能', formatPercent(panelStats.energy_recharge || 0)],
    ].map(([label, value]) => `
      <div class="shaft-panel-minor">
        <span>${label}</span>
        <strong>${value}</strong>
      </div>
    `).join('');
    node.innerHTML = `
      <div class="shaft-panel-avatar-row">${avatarButtons}</div>
      <div class="shaft-panel-profile">
        <div>
          <span>当前查看</span>
          <strong>${escapeHtml(member.character_name || panel.character_name || '角色')}</strong>
        </div>
        <small>${escapeHtml(member.arc_name || '')} · ${escapeHtml(member.cartridge_name || '')} · 主词条 ${escapeHtml(buildOptions.cartridge_main_stat || '')} · 空幕 ${escapeHtml(curtain.value || 0)}${escapeHtml(curtain.stat || '')} x ${formatNumber(curtain.layers || 0, 0)}</small>
      </div>
      <div class="shaft-panel-major-grid">${mainStats}</div>
      <div class="shaft-panel-minor-grid">${minorStats}</div>
    `;
  }

  function renderWorkbenchSummary() {
    const node = $('shaft-workbench-summary');
    if (!node) {
      return;
    }
    const result = freshResult();
    const summary = result?.summary || {};
    const workbenchDirectDamage = Math.max(
      0,
      Number(summary.direct_damage || 0) - Number(summary.harmony_damage || 0),
    );
    const contribution = result?.damage_by_slot || [];
    const characterBars = contribution.map((item, index) => `
      <div class="shaft-mini-contribution">
        <span>${escapeHtml(item.character_name)}</span>
        <b>${formatNumber(item.percent || 0, 1)}%</b>
        <i style="--bar:${Math.max(0, Math.min(100, Number(item.percent || 0)))}%; --slot-color:${SLOT_COLORS[index % SLOT_COLORS.length]}"></i>
      </div>
    `).join('');
    const sourceBars = (result?.damage_by_source || []).map((item) => `
      <div class="shaft-mini-contribution shaft-mini-contribution-source">
        <span>${escapeHtml(item.source)}</span>
        <b>${formatNumber(item.percent || 0, 1)}%</b>
        <i style="--bar:${Math.max(0, Math.min(100, Number(item.percent || 0)))}%; --slot-color:${DAMAGE_SOURCE_COLORS[item.source] || '#b9c6d8'}"></i>
      </div>
    `).join('');
    node.innerHTML = `
      <div class="shaft-summary-grid">
        <div><span>总伤害</span><strong>${formatNumber(summary.total_damage || 0)}</strong></div>
        <div><span>总轴长</span><strong>${formatNumber(summary.duration_seconds || 0, 1)}s</strong></div>
        <div><span>DPS</span><strong>${formatNumber(summary.dps || 0)}</strong></div>
        <div><span>直伤</span><strong>${formatNumber(workbenchDirectDamage)}</strong></div>
        <div><span>环合</span><strong>${formatNumber(summary.harmony_damage || 0)}</strong></div>
        <div><span>倾陷</span><strong>${formatNumber(summary.stagger_damage || 0)}</strong></div>
      </div>
      <div class="shaft-mini-contribution-list">${characterBars}${sourceBars}</div>
    `;
  }

  function renderedAxisEndTick(details = []) {
    const foregroundDetails = details.filter((detail) => !detail.is_background_damage);
    return Math.max(
      0,
      ...foregroundDetails.map((detail) => Math.max(
        Number(detail.display_visual_end_tick ?? detail.visual_end_tick ?? detail.end_tick ?? detail.start_tick ?? 0),
        Number(detail.display_start_tick ?? detail.start_tick ?? 0) + 1,
      )),
    );
  }

  function timelineMaxTick(renderDetails = null) {
    const details = renderDetails || timelineResult()?.details || [];
    const axisEnd = details.length ? renderedAxisEndTick(details) : axisEndTick(true);
    return Math.max(1, axisEnd + TIMELINE_END_PADDING_TICKS);
  }

  function buffDisplayName(buff) {
    const provider = String(buff?.provider_name || '').trim();
    if (provider) {
      return provider;
    }
    const name = String(buff?.name || '').trim();
    return name.split(/[：:]/)[0] || name;
  }

  function buffRuleDisplayName(buff) {
    const provider = String(buff?.provider_name || '').trim();
    const name = String(buff?.name || '').trim();
    if (name && name !== provider) {
      return name;
    }
    const ruleId = String(buff?.rule_id || buff?.definition_id || '').trim();
    const passiveName = ruleId ? `被动 ${ruleId}` : '未命名被动';
    return provider ? `${provider}·${passiveName}` : passiveName;
  }

  function isUnlinedBuff(buff) {
    const provider = buffDisplayName(buff);
    const name = buffRuleDisplayName(buff);
    return provider === '引爆全场' || name.startsWith('引爆全场');
  }

  function shouldDrawBuffLine(buff) {
    if (!buff || buff.cancelled || isUnlinedBuff(buff)) {
      return false;
    }
    return buff.display_as_line !== false;
  }

  function wrappedTick(tick, axisEndTick) {
    if (axisEndTick <= 0) {
      return 0;
    }
    return ((Number(tick || 0) % axisEndTick) + axisEndTick) % axisEndTick;
  }

  function buffLineSegments(startTick, endTick, axisEndTick, loopEnabled) {
    const axisEnd = Math.max(1, Number(axisEndTick || 1));
    const start = Number(startTick || 0);
    const end = Math.max(start, Number(endTick || start));
    const duration = end - start;
    if (duration <= 0) {
      return [];
    }
    if (!loopEnabled) {
      const clippedStart = Math.max(0, Math.min(axisEnd, start));
      const clippedEnd = Math.max(clippedStart, Math.min(axisEnd, end));
      return clippedEnd > clippedStart ? [{ startTick: clippedStart, endTick: clippedEnd }] : [];
    }
    if (duration >= axisEnd) {
      return [{ startTick: 0, endTick: axisEnd }];
    }
    const startMod = wrappedTick(start, axisEnd);
    const endMod = startMod + duration;
    if (endMod <= axisEnd) {
      return [{ startTick: startMod, endTick: endMod }];
    }
    return [
      { startTick: startMod, endTick: axisEnd },
      { startTick: 0, endTick: endMod - axisEnd },
    ].filter((segment) => segment.endTick > segment.startTick);
  }

  function reactionBuffLineSegments(effect, axisEndTick, loopEnabled) {
    const startTick = Number(effect?.visual_start_tick ?? effect?.start_tick ?? 0);
    const endTick = Number(effect?.visual_end_tick ?? effect?.end_tick ?? startTick);
    const segments = buffLineSegments(startTick, endTick, axisEndTick, loopEnabled);
    if (!loopEnabled || effect?.loop_primed || endTick <= Number(axisEndTick || 0)) {
      return segments;
    }
    return segments.filter((segment) => segment.startTick > 0);
  }

  function triggeredBuffLines(detail, axisEndTick = 0, loopEnabled = false, trackSlot = null) {
    const seen = new Set();
    return (detail?.triggered_buffs || [])
      .filter((buff) => shouldDrawBuffLine(buff))
      .filter((buff) => {
        if (trackSlot === null) {
          return true;
        }
        const ownerSlot = Number(buff?.owner_slot ?? -1);
        return ownerSlot >= 0
          ? ownerSlot === Number(trackSlot)
          : Number(detail?.slot) === Number(trackSlot);
      })
      .map((buff) => {
        const startTick = Math.max(
          Number(detail?.display_start_tick ?? detail?.visual_start_tick ?? detail?.start_tick ?? 0),
          Number(buff?.visual_start_tick ?? buff?.trigger_tick ?? buff?.start_tick ?? detail?.display_start_tick ?? 0),
        );
        const endTick = Math.max(
          startTick + 1,
          Number(buff?.visual_end_tick ?? buff?.display_end_tick ?? buff?.end_tick ?? startTick + 1),
        );
        const name = buffRuleDisplayName(buff);
        const segments = buffLineSegments(startTick, endTick, axisEndTick, loopEnabled);
        return {
          id: String(buff?.definition_id || buff?.rule_id || `${name}_${startTick}_${endTick}`),
          name,
          startTick,
          endTick,
          calculationStartTick: Number(
            buff?.start_tick ?? calculationTickFromVisual(startTick),
          ),
          calculationEndTick: Number(
            buff?.end_tick ?? calculationTickFromVisual(endTick),
          ),
          loopDurationTicks: loopEnabled
            ? Math.max(1, calculationTickFromVisual(Number(axisEndTick || 0)))
            : 0,
          durationTicks: Math.max(1, Number(buff?.duration_ticks ?? endTick - startTick)),
          segments,
          stackCount: buff?.stack_count ?? 1,
          stackingMode: String(buff?.stacking_mode || ''),
          maxStacks: Math.max(1, Number(buff?.max_stacks || 1)),
        };
      })
      .filter((buff) => buff.name && Number.isFinite(buff.startTick) && Number.isFinite(buff.endTick) && buff.endTick > buff.startTick && buff.segments.length)
      .filter((buff) => {
        const key = `${buff.id}:${buff.startTick}:${buff.endTick}`;
        if (seen.has(key)) {
          return false;
        }
        seen.add(key);
        return true;
      });
  }

  function resultDetailMatchesStep(detail, step) {
    return detail &&
      String(detail.action_id || '') === String(step?.action_id || '') &&
      Number(detail.slot) === Number(step?.slot);
  }

  function buffStackValue(stackCount) {
    const value = Number(stackCount ?? 1);
    return Number.isFinite(value) ? value : 1;
  }

  function buffStackKey(stackCount) {
    const value = buffStackValue(stackCount);
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }

  function buffStackText(stackCount) {
    const value = buffStackValue(stackCount);
    return value > 1 ? ` · ${formatNumber(value, Number.isInteger(value) ? 0 : 1)}层` : '';
  }

  function buffRemainingTicksAtCalculationTick(buff, pointerCalculationTick) {
    const calculationStartTick = Number(
      buff?.calculationStartTick ?? calculationTickFromVisual(Number(buff?.startTick || 0)),
    );
    const calculationEndTick = Number(
      buff?.calculationEndTick ?? calculationTickFromVisual(Number(buff?.endTick || 0)),
    );
    const loopDurationTicks = Math.max(0, Number(buff?.loopDurationTicks || 0));
    let effectivePointerTick = Number(pointerCalculationTick || 0);
    if (loopDurationTicks > 0 && effectivePointerTick < calculationStartTick) {
      const cycles = Math.ceil((calculationStartTick - effectivePointerTick) / loopDurationTicks);
      effectivePointerTick += cycles * loopDurationTicks;
    }
    return Math.max(0, calculationEndTick - effectivePointerTick);
  }

  function buffTooltipText(buff, atVisualTick = buff.startTick) {
    const stackText = buffStackText(buff.stackCount);
    const durationTicks = Math.max(1, Number(buff.durationTicks || 1));
    const durationText = durationTicks > BUFF_DURATION_LABEL_LIMIT_TICKS
      ? ''
      : ` · 剩余${ticksToSeconds(buffRemainingTicksAtCalculationTick(
        buff,
        calculationTickFromVisual(Number(atVisualTick || 0)),
      ))}s`;
    return `${buff.name}${durationText}${stackText}`;
  }

  function activeBuffSegmentKey(buffs = []) {
    return buffs
      .map((buff) => `${buff.id}:${buff.name}`)
      .sort()
      .join('|');
  }

  function latestBuffStates(buffs = []) {
    const latestById = new Map();
    buffs.forEach((buff) => {
      if (String(buff?.stackingMode || '') === 'independent') {
        const current = latestById.get(buff.id);
        if (!current) {
          latestById.set(buff.id, {
            ...buff,
            stackCount: buffStackValue(buff.stackCount),
          });
          return;
        }
        current.stackCount = Math.min(
          Math.max(1, Number(current.maxStacks || buff.maxStacks || 1)),
          buffStackValue(current.stackCount) + buffStackValue(buff.stackCount),
        );
        current.startTick = Math.max(Number(current.startTick || 0), Number(buff.startTick || 0));
        current.endTick = Math.max(Number(current.endTick || 0), Number(buff.endTick || 0));
        current.calculationStartTick = Math.max(
          Number(current.calculationStartTick || 0),
          Number(buff.calculationStartTick || 0),
        );
        current.calculationEndTick = Math.max(
          Number(current.calculationEndTick || 0),
          Number(buff.calculationEndTick || 0),
        );
        current.durationTicks = Math.max(1, Math.max(Number(current.durationTicks || 1), Number(buff.durationTicks || 1)));
        return;
      }
      const current = latestById.get(buff.id);
      const startsLater = Number(buff.startTick) > Number(current?.startTick ?? -1);
      const startsTogetherWithMoreStacks = Number(buff.startTick) === Number(current?.startTick) &&
        buffStackValue(buff.stackCount) > buffStackValue(current?.stackCount);
      if (!current || startsLater || startsTogetherWithMoreStacks) {
        latestById.set(buff.id, buff);
      }
    });
    return Array.from(latestById.values());
  }

  function mergedBuffLineSegments(details = [], axisEndTick = 0, loopEnabled = false, trackSlot = null) {
    const entries = [];
    const clearTicksByBuffId = new Map();
    if (loopEnabled) {
      (details || []).forEach((detail) => {
        (detail?.triggered_buffs || []).forEach((buff) => {
          const ownerSlot = Number(buff?.owner_slot ?? detail?.slot ?? -1);
          if (trackSlot !== null && ownerSlot >= 0 && ownerSlot !== Number(trackSlot)) {
            return;
          }
          const clearTick = Number(
            buff?.visual_start_tick ??
            buff?.trigger_tick ??
            detail?.display_start_tick ??
            detail?.visual_start_tick ??
            detail?.start_tick ??
            0
          );
          (buff?.cleared_buff_keys || []).forEach((buffId) => {
            const key = String(buffId || '');
            if (!key || !Number.isFinite(clearTick)) {
              return;
            }
            const ticks = clearTicksByBuffId.get(key) || [];
            ticks.push(clearTick);
            clearTicksByBuffId.set(key, ticks);
          });
        });
      });
      clearTicksByBuffId.forEach((ticks) => ticks.sort((left, right) => left - right));
    }
    (details || [])
      .slice()
      .sort((a, b) => Number(a.start_tick || 0) - Number(b.start_tick || 0))
      .forEach((detail) => {
        triggeredBuffLines(detail, axisEndTick, loopEnabled, trackSlot).forEach((buff) => {
          buff.segments.forEach((segment) => {
            const nextClearTick = (clearTicksByBuffId.get(buff.id) || [])
              .find((tick) => tick > Number(segment.startTick) && tick < Number(segment.endTick));
            entries.push({
              id: buff.id,
              name: buff.name,
              startTick: Number(segment.startTick || 0),
              endTick: nextClearTick == null ? Number(segment.endTick || 0) : nextClearTick,
              calculationStartTick: Number(buff.calculationStartTick || 0),
              calculationEndTick: Number(buff.calculationEndTick || 0),
              loopDurationTicks: Math.max(0, Number(buff.loopDurationTicks || 0)),
              durationTicks: Math.max(1, Number(buff.durationTicks || 1)),
              stackCount: buffStackValue(buff.stackCount),
              stackingMode: String(buff.stackingMode || ''),
              maxStacks: Math.max(1, Number(buff.maxStacks || 1)),
            });
          });
        });
      });
    const boundaries = Array.from(new Set(entries
      .flatMap((entry) => [entry.startTick, entry.endTick])
      .filter((tick) => Number.isFinite(tick))))
      .sort((a, b) => a - b);
    const segments = [];
    for (let index = 0; index < boundaries.length - 1; index += 1) {
      const startTick = boundaries[index];
      const endTick = boundaries[index + 1];
      if (endTick <= startTick) {
        continue;
      }
      const activeBuffs = latestBuffStates(
        entries.filter((entry) => entry.startTick <= startTick && entry.endTick >= endTick),
      )
        .sort((a, b) => a.name.localeCompare(b.name, 'zh-Hans-CN') || a.id.localeCompare(b.id));
      if (!activeBuffs.length) {
        continue;
      }
      const key = activeBuffSegmentKey(activeBuffs);
      const previous = segments[segments.length - 1];
      const tooltipItems = activeBuffs.map((buff) => ({
        id: buff.id,
        name: buff.name,
        endTick: buff.endTick,
        calculationStartTick: buff.calculationStartTick,
        calculationEndTick: buff.calculationEndTick,
        loopDurationTicks: buff.loopDurationTicks,
        durationTicks: buff.durationTicks,
        stackCount: buff.stackCount,
        stackingMode: buff.stackingMode,
        maxStacks: buff.maxStacks,
        visualStartTick: startTick,
        visualEndTick: endTick,
      }));
      if (previous && previous.key === key && previous.endTick === startTick) {
        previous.endTick = endTick;
        previous.tooltip = activeBuffs.map((buff) => buffTooltipText(buff, startTick)).join('\n');
        previous.tooltipItems.push(...tooltipItems);
        continue;
      }
      const seen = new Set();
      const tooltipLines = activeBuffs
        .filter((buff) => {
          const buffKey = `${buff.id}:${buff.name}:${buffStackKey(buff.stackCount)}:${buff.durationTicks}`;
          if (seen.has(buffKey)) {
            return false;
          }
          seen.add(buffKey);
          return true;
        })
        .map((buff) => buffTooltipText(buff, startTick));
      segments.push({
        key,
        startTick,
        endTick,
        tooltip: tooltipLines.join('\n'),
        tooltipItems,
        buffIds: activeBuffs.map((buff) => buff.id).join(' '),
      });
    }
    return segments;
  }

  function mergedRenderedBuffSegments(segments = []) {
    const normalized = segments
      .map((segment) => ({
        ...segment,
        startTick: Number(segment.startTick || 0),
        endTick: Number(segment.endTick || 0),
      }))
      .filter((segment) => Number.isFinite(segment.startTick) && Number.isFinite(segment.endTick) && segment.endTick > segment.startTick);
    const boundaries = Array.from(new Set(normalized
      .flatMap((segment) => [segment.startTick, segment.endTick])))
      .sort((left, right) => left - right);
    const merged = [];
    for (let index = 0; index < boundaries.length - 1; index += 1) {
      const startTick = boundaries[index];
      const endTick = boundaries[index + 1];
      const active = normalized.filter((segment) => segment.startTick <= startTick && segment.endTick >= endTick);
      if (!active.length) {
        continue;
      }
      const tooltipLines = Array.from(new Set(active
        .flatMap((segment) => String(segment.tooltip || '').split('\n'))
        .map((line) => line.trim())
        .filter(Boolean)));
      const buffIds = Array.from(new Set(active
        .flatMap((segment) => String(segment.buffIds || '').split(/\s+/))
        .filter(Boolean)));
      const colors = Array.from(new Set(active.map((segment) => String(segment.color || '')).filter(Boolean)));
      const tooltipItems = Array.from(new Map(active
        .flatMap((segment) => Array.isArray(segment.tooltipItems) ? segment.tooltipItems : [])
        .map((item) => [
          `${item.id}:${item.calculationStartTick}:${item.calculationEndTick}:${buffStackKey(item.stackCount)}:${item.visualStartTick}:${item.visualEndTick}`,
          item,
        ]))
        .values());
      const key = JSON.stringify({ buffIds: buffIds.slice().sort(), colors: colors.slice().sort() });
      const previous = merged[merged.length - 1];
      if (previous && previous.key === key && previous.endTick === startTick) {
        previous.endTick = endTick;
        previous.tooltip = tooltipLines.join('\n');
        previous.tooltipItems = Array.from(new Map([
          ...previous.tooltipItems,
          ...tooltipItems,
        ].map((item) => [
          `${item.id}:${item.calculationStartTick}:${item.calculationEndTick}:${buffStackKey(item.stackCount)}:${item.visualStartTick}:${item.visualEndTick}`,
          item,
        ])).values());
        continue;
      }
      merged.push({
        key,
        startTick,
        endTick,
        tooltip: tooltipLines.join('\n'),
        tooltipItems,
        buffIds: buffIds.join(' '),
        color: colors[0] || '#ffffff',
      });
    }
    return merged;
  }

  function triggeredUnlinedBuffs(detail) {
    const seen = new Set();
    return (detail?.triggered_buffs || [])
      .filter((buff) => isUnlinedBuff(buff))
      .map((buff) => {
        const startTick = Math.max(
          Number(detail?.display_start_tick ?? detail?.visual_start_tick ?? detail?.start_tick ?? 0),
          Number(buff?.visual_start_tick ?? buff?.trigger_tick ?? buff?.start_tick ?? detail?.display_start_tick ?? 0),
        );
        const endTick = Math.max(
          startTick + 1,
          Number(buff?.visual_end_tick ?? buff?.display_end_tick ?? buff?.end_tick ?? startTick + 1),
        );
        return {
          id: String(buff?.rule_id || `${buffRuleDisplayName(buff)}_${startTick}_${endTick}`),
          name: buffRuleDisplayName(buff),
          startTick,
          endTick,
          durationTicks: Math.max(1, endTick - startTick),
          stackCount: buff?.stack_count ?? 1,
        };
      })
      .filter((buff) => buff.name && Number.isFinite(buff.startTick) && Number.isFinite(buff.endTick) && buff.endTick > buff.startTick)
      .filter((buff) => {
        const key = `${buff.id}:${buff.startTick}:${buff.endTick}`;
        if (seen.has(key)) {
          return false;
        }
        seen.add(key);
        return true;
      });
  }

  function unlinedBuffWarnings(details = []) {
    const groups = new Map();
    details.forEach((detail) => {
      const buffs = triggeredUnlinedBuffs(detail);
      if (!buffs.length) {
        return;
      }
      buffs.forEach((buff) => {
        const tick = Math.max(0, Number(buff.startTick || 0));
        const key = String(tick);
        const group = groups.get(key) || { tick, buffs: [], seen: new Set() };
        const buffKey = `${buff.id}:${buff.startTick}:${buff.endTick}`;
        if (group.seen.has(buffKey)) {
          return;
        }
        group.seen.add(buffKey);
        group.buffs.push(buff);
        groups.set(key, group);
      });
    });
    return Array.from(groups.values()).sort((a, b) => a.tick - b.tick);
  }

  function unlinedBuffToolbarTooltip() {
    const uniqueBuffs = new Map();
    unlinedBuffWarnings(state.timelineDisplayDetails || freshResult()?.details || []).forEach((warning) => {
      warning.buffs.forEach((buff) => {
        const key = buff.name;
        const current = uniqueBuffs.get(key);
        if (!current || buffStackValue(buff.stackCount) > buffStackValue(current.stackCount)) {
          uniqueBuffs.set(key, buff);
        }
      });
    });
    const parts = Array.from(uniqueBuffs.values())
      .map((buff) => `${buff.name}${buffStackText(buff.stackCount)}`);
    return parts.length ? `未划线的 buffs：${parts.join('；')}` : '';
  }

  function buffLineColor(index) {
    return BUFF_LINE_COLORS[index % BUFF_LINE_COLORS.length];
  }

  function reactionLineColor(reaction) {
    return {
      '创生': DAMAGE_SOURCE_COLORS['创生'],
      '创生复制体': DAMAGE_SOURCE_COLORS['创生复制体'],
      '延滞': '#f4d66f',
      '覆纹': '#ff7f9d',
      '浊燃': DAMAGE_SOURCE_COLORS['浊燃'],
      '黯星': DAMAGE_SOURCE_COLORS['黯星'],
      '浸染': '#65c8ff',
    }[reaction] || '#ffffff';
  }

  function damageMarkerTooltip(event) {
    if (Array.isArray(event?.events)) {
      return event.events.map((item) => damageMarkerTooltip(item)).join('\n');
    }
    const formula = event?.formula_parts || {};
    const isPeriodicDamage = Boolean(event?.kind);
    const zones = [];
    const addZone = (label, value) => {
      const multiplier = Number(value);
      if (Number.isFinite(multiplier)) {
        zones.push(`${label} ×${formatNumber(multiplier, 3)}`);
      }
    };
    addZone('基础区', isPeriodicDamage ? formula.unscaled_base : formula.base);
    if (isPeriodicDamage) {
      addZone('周期系数', formula.periodic_scale);
      addZone('增伤', 1 + Number(formula.damage_bonus || 0));
      addZone('双暴区', formula.critical);
      addZone('防御', formula.defense);
      addZone('抗性', formula.resistance);
      addZone('最终倍率区', formula.final_multiplier);
    } else {
      addZone('环合强度', formula.strength);
      if (Number(formula.damage_scale) !== 1) addZone('复制倍率', formula.damage_scale);
      if (Number(formula.frequency_multiplier || 1) > 1) {
        addZone('频率乘区', formula.frequency_multiplier);
      }
      if (String(event?.reaction || '') !== '黯星') addZone('防御', formula.defense);
      if (String(event?.reaction || '') === '浊燃') addZone('双暴区', formula.critical);
      addZone('抗性', formula.resistance);
      addZone('最终倍率区', formula.final_multiplier);
    }
    const heading = `${event?.reaction || '伤害'} · ${event?.contributor_character_name || memberName(event?.contributor_slot)} · ${formatNumber(event?.damage || 0)} 伤害`;
    return zones.length ? `${heading}\n乘区　${zones.join('　')}` : heading;
  }

  function groupedTimelineDamageEvents(events) {
    const groups = new Map();
    (events || []).forEach((event) => {
      const tick = Number(event?.visual_tick ?? event?.tick ?? 0);
      const key = `${Number(event?.contributor_slot)}:${tick}`;
      const group = groups.get(key) || {
        contributor_slot: Number(event?.contributor_slot),
        tick: Number(event?.tick ?? 0),
        visual_tick: tick,
        events: [],
      };
      group.events.push(event);
      groups.set(key, group);
    });
    return Array.from(groups.values()).map((group) => {
      const primary = group.events.find((event) => !event?.kind) || group.events[0] || {};
      return {
        ...group,
        reaction: primary.reaction,
        kind: group.events.every((event) => Boolean(event?.kind)) ? 'periodic_group' : '',
      };
    });
  }

  function renderTimeline() {
    const drag = state.dragState || null;
    const result = timelineResult();
    const reactionEffects = result?.reaction_effects || [];
    const periodicDamageEvents = result?.periodic_damage_events || [];
    const reactionDamageEvents = result?.reaction_damage_events || [];
    const timelineDamageEvents = [...reactionDamageEvents, ...periodicDamageEvents];
    const groupedDamageEvents = groupedTimelineDamageEvents(timelineDamageEvents);
    const usesDragPreviewResult = Boolean(drag?.previewResult && result === drag.previewResult);
    const usesStaleResult = Boolean(state.isResultStale && state.result && result === state.result);
    const resultDetails = result?.details || [];
    const resultDetailByStepId = new Map(resultDetails.map((detail) => [detail.step_id, detail]));
    const resultDetailMatchesTimelineStep = (step, resultDetail) => {
      if (!resultDetail) {
        return false;
      }
      const matchesIdentity =
        String(resultDetail.action_id || '') === String(step.action_id || '') &&
        Number(resultDetail.slot) === Number(step.slot);
      if (!matchesIdentity) {
        return false;
      }
      if ((drag && !usesDragPreviewResult) || usesStaleResult) {
        return true;
      }
      return Number(resultDetail.raw_start_tick ?? resultDetail.start_tick ?? -1) === Number(step.start_tick || 0);
    };
    const projectedDetails = (state.axis.steps || []).map((step) => {
      const action = actionForStep(step);
      const durationTicks = actionDurationTicks(action);
      const startTick = Number(step.start_tick || 0);
      const matchedResultDetail = resultDetailByStepId.get(step.id) || null;
      const canUseResultTiming = resultDetailMatchesTimelineStep(step, matchedResultDetail);
      const resultDetail = canUseResultTiming ? matchedResultDetail : {};
      const originalProjectionTick = Number(drag?.originAxisTicks?.[step.id] ?? resultDetail.raw_start_tick ?? startTick);
      const projectionShiftTicks = (drag && !usesDragPreviewResult) || usesStaleResult
        ? startTick - originalProjectionTick
        : 0;
      const resultTick = (key, fallback) => {
        const value = resultDetail[key];
        return value === undefined || value === null ? Number(fallback || 0) : Number(value) + projectionShiftTicks;
      };
      const projectedTriggeredBuffs = (resultDetail.triggered_buffs || []).map((buff) => {
        if (!projectionShiftTicks) {
          return buff;
        }
        const projectedBuff = Object.assign({}, buff);
        ['visual_start_tick', 'visual_end_tick', 'display_start_tick', 'display_end_tick'].forEach((key) => {
          if (projectedBuff[key] !== undefined && projectedBuff[key] !== null) {
            projectedBuff[key] = Number(projectedBuff[key]) + projectionShiftTicks;
          }
        });
        return projectedBuff;
      });
      const renderStartTick = resultTick('start_tick', startTick);
      const renderEndTick = resultTick('end_tick', renderStartTick + durationTicks);
      const renderDurationTicks = Number(resultDetail.duration_ticks ?? Math.max(0, renderEndTick - renderStartTick));
      const renderVisualStartTick = resultTick('visual_start_tick', renderStartTick);
      const renderVisualEndTick = resultTick('visual_end_tick', renderVisualStartTick + (renderDurationTicks > 0 ? renderDurationTicks : ZERO_ACTION_VISUAL_TICKS));
      const actionIsQ = isZeroForegroundQStep(step, action);
      const displayStartTick = resultTick('display_start_tick', resultDetail.visual_start_tick ?? startTick);
      const displayDurationTicks = Number(resultDetail.display_duration_ticks ?? durationTicks);
      const displayEndTick = resultTick('display_end_tick', displayStartTick + durationTicks);
      const nominalDisplayVisualEndTick = resultDetail.nominal_display_visual_end_tick === undefined
        ? displayStartTick + (durationTicks > 0 ? durationTicks : ZERO_ACTION_VISUAL_TICKS)
        : Number(resultDetail.nominal_display_visual_end_tick) + projectionShiftTicks;
      const fallbackDisplayVisualEndTick = actionIsQ
        ? Math.max(nominalDisplayVisualEndTick, resultTick('visual_end_tick', nominalDisplayVisualEndTick))
        : nominalDisplayVisualEndTick;
      const displayVisualEndTick = actionIsQ
        ? Math.max(nominalDisplayVisualEndTick, resultTick('display_visual_end_tick', fallbackDisplayVisualEndTick))
        : resultTick('display_visual_end_tick', fallbackDisplayVisualEndTick);
      return Object.assign({}, resultDetail, {
        step_id: step.id,
        slot: Number(step.slot || 0),
        action_id: step.action_id,
        action_name: action.name || step.action_name || '',
        action_type: action.action_type || '',
        raw_start_tick: startTick,
        start_tick: renderStartTick,
        end_tick: renderEndTick,
        duration_ticks: renderDurationTicks,
        visual_start_tick: renderVisualStartTick,
        visual_end_tick: renderVisualEndTick,
        triggered_buffs: projectedTriggeredBuffs,
        display_start_tick: displayStartTick,
        display_end_tick: displayEndTick,
        display_duration_ticks: displayDurationTicks,
        display_visual_end_tick: displayVisualEndTick,
        nominal_display_visual_end_tick: nominalDisplayVisualEndTick,
        is_background_damage: isStepBackground(step, action),
        is_basic_background: isBasicBackgroundOverride(step, action),
        placement: step.placement || 'foreground',
      });
    });
    const foregroundAxisEndTick = renderedAxisEndTick(projectedDetails);
    const displayCutoffTick = foregroundAxisEndTick + TIMELINE_END_PADDING_TICKS;
    const details = projectedDetails
      .filter((detail) => Number(detail.display_start_tick ?? detail.start_tick ?? 0) < displayCutoffTick)
      .map((detail) => ({
        ...detail,
        display_end_tick: Math.min(
          displayCutoffTick,
          Number(detail.display_end_tick ?? detail.end_tick ?? displayCutoffTick),
        ),
        display_visual_end_tick: Math.min(
          displayCutoffTick,
          Number(detail.display_visual_end_tick ?? detail.visual_end_tick ?? displayCutoffTick),
        ),
      }));
    state.timelineDisplayDetails = details;
    const damageTriggeredBuffDetails = [...reactionDamageEvents, ...periodicDamageEvents]
      .filter((event) => Number(event.visual_tick ?? event.tick ?? 0) <= displayCutoffTick)
      .filter((event) => Array.isArray(event?.triggered_buffs) && event.triggered_buffs.length)
      .map((event) => {
        const visualTick = Number(event.visual_tick ?? event.tick ?? 0);
        return {
          slot: Number(event.contributor_slot ?? -1),
          start_tick: visualTick,
          visual_start_tick: visualTick,
          display_start_tick: visualTick,
          triggered_buffs: event.triggered_buffs,
        };
      });
    const buffTimelineDetails = [...details, ...damageTriggeredBuffDetails];
    const loopEnabled = Boolean(state.axis?.options?.loop_enabled);
    const actionAxisEndTick = foregroundAxisEndTick;
    const buffAxisEndTick = loopEnabled
      ? actionAxisEndTick
      : displayCutoffTick;
    const heldBuffPreview = state.dragState?.buffPreview || null;
    const durationTicks = timelineMaxTick(details);
    state.timelineDurationTicks = durationTicks;
    const expansionByTick = new Map();
    details.forEach((detail) => {
      const displayDurationTicks = Number(detail.display_duration_ticks ?? detail.duration_ticks ?? 0);
      const tick = Math.max(0, Number(detail.display_start_tick ?? detail.start_tick ?? 0));
      const visualEndTick = Number(detail.display_visual_end_tick ?? detail.visual_end_tick ?? detail.end_tick ?? tick);
      const blocksTrack = !detail.is_background_damage || detail.is_basic_background;
      const isZeroForegroundQ = displayDurationTicks === 0 &&
        !detail.is_background_damage &&
        isQAction(actionForStep({ action_id: detail.action_id }));
      if (!blocksTrack && !isZeroForegroundQ) {
        return;
      }
      if (!isZeroForegroundQ) {
        return;
      }
      const displayEndTick = Math.max(
        tick + 1,
        Math.max(tick + ZERO_ACTION_VISUAL_TICKS, Number(detail.nominal_display_visual_end_tick ?? visualEndTick ?? tick + 1)),
      );
      const rawWidth = Math.max(1, displayEndTick - tick) * TIMELINE_TICK_PX;
      const extraPx = Math.max(0, MIN_ACTION_CARD_PX - rawWidth);
      if (extraPx > 0) {
        expansionByTick.set(tick, Math.max(Number(expansionByTick.get(tick) || 0), extraPx));
      }
    });
    let expansionBreaks = Array.from(expansionByTick.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([tick, extraPx]) => ({
        tick,
        extraPx,
      }));
    const visualOffsetPx = (tick) => {
      const safeTick = Math.max(0, Number(tick || 0));
      return safeTick * TIMELINE_TICK_PX + expansionBreaks
        .filter((breakItem) => breakItem.tick < safeTick)
        .reduce((sum, breakItem) => sum + breakItem.extraPx, 0);
    };
    let timelineBodyPx = Math.max(900, visualOffsetPx(durationTicks));
    if (drag?.timelineScale && !usesDragPreviewResult) {
      expansionBreaks = clone(drag.timelineScale.expansionBreaks || []);
      timelineBodyPx = Math.max(
        900,
        Number(drag.timelineScale.bodyPx || 0),
        timelineVisualOffsetWithScale(durationTicks, { expansionBreaks }),
      );
    }
    state.timelineScale = { expansionBreaks, bodyPx: timelineBodyPx };
    const timelineTotalPx = TIMELINE_LABEL_PX + timelineBodyPx;
    const leftPx = (tick) => TIMELINE_LABEL_PX + visualOffsetPx(tick);
    const widthPx = (start, end, visualEnd, minWidth = 1) => {
      const fixedEnd = visualEnd || end || (Number(start || 0) + ZERO_ACTION_VISUAL_TICKS);
      if (Number(fixedEnd || start) <= Number(start || 0)) {
        return minWidth;
      }
      return Math.max(minWidth, visualOffsetPx(Math.max(fixedEnd, Number(start || 0) + 1)) - visualOffsetPx(start));
    };
    const bandWidthPx = (start, end) => Math.max(1, visualOffsetPx(Math.max(Number(end || start), Number(start || 0) + 1)) - visualOffsetPx(start));
    const rulerMarks = [];
    const timelineQStarts = details
      .filter((detail) => Number(detail.display_duration_ticks ?? detail.duration_ticks ?? 0) === 0 && !detail.is_background_damage && isQAction(actionForStep({ action_id: detail.action_id })))
      .map((detail) => ({
        start_tick: Number(detail.display_start_tick ?? detail.start_tick ?? 0),
        end_tick: Math.max(
          Number(detail.display_start_tick ?? detail.start_tick ?? 0) + ZERO_ACTION_VISUAL_TICKS,
          Number(detail.display_visual_end_tick ?? detail.visual_end_tick ?? detail.end_tick ?? detail.start_tick ?? 0),
        ),
      }));
    const cursor = (labeled) => `<span class="shaft-cursor-line" style="left:${leftPx(state.cursorTick)}px">${labeled ? `<b>${escapeHtml(visualTickLabel(state.cursorTick, timelineQStarts))}</b>` : ''}</span>`;
    const rulerQStarts = timelineQStarts;
    for (let tick = 0; tick <= durationTicks; tick += 10) {
      const rulerParts = visualTickParts(tick, rulerQStarts);
      const isMajor = rulerParts.calculationTick % 50 === 0;
      const isSecond = rulerParts.calculationTick % 10 === 0;
      rulerMarks.push(`<span class="shaft-ruler-mark ${isMajor ? 'major' : ''} ${isSecond ? 'second' : ''}" style="left:${leftPx(tick)}px"><span>${escapeHtml(visualTickLabel(tick, rulerQStarts))}</span></span>`);
    }
    $('shaft-time-ruler').style.width = `${timelineTotalPx}px`;
    $('shaft-timeline').style.width = `${timelineTotalPx}px`;
    $('shaft-time-ruler').style.setProperty('--label-width', `${TIMELINE_LABEL_PX}px`);
    $('shaft-timeline').style.setProperty('--label-width', `${TIMELINE_LABEL_PX}px`);
    $('shaft-time-ruler').innerHTML = rulerMarks.join('') + cursor(true);
    const foregroundEvents = details
      .filter((detail) => !detail.is_background_damage)
      .sort((a, b) => Number(a.display_start_tick ?? a.start_tick ?? 0) - Number(b.display_start_tick ?? b.start_tick ?? 0))
      .map((detail) => ({
        slot: Number(detail.slot || 0),
        start_tick: Number(detail.display_start_tick ?? detail.start_tick ?? 0),
        end_tick: Number(detail.display_visual_end_tick ?? detail.visual_end_tick ?? detail.end_tick ?? detail.start_tick ?? 0),
      }));
    const frontWindowItems = [];
    foregroundEvents.forEach((eventItem, index) => {
      const endTick = index + 1 < foregroundEvents.length
        ? Math.max(eventItem.start_tick + 1, foregroundEvents[index + 1].start_tick)
        : Math.max(eventItem.start_tick + 1, eventItem.end_tick);
      const lastWindow = frontWindowItems[frontWindowItems.length - 1];
      if (lastWindow && Number(lastWindow.slot) === Number(eventItem.slot)) {
        lastWindow.end_tick = Math.max(lastWindow.end_tick, endTick);
        return;
      }
      frontWindowItems.push({
        slot: eventItem.slot,
        start_tick: eventItem.start_tick,
        end_tick: endTick,
      });
    });
    const actionTracks = (state.axis.team || []).map((member) => {
      const selectedIds = new Set(selectedStepIds());
      const color = SLOT_COLORS[Number(member.slot) % SLOT_COLORS.length];
      const character = getCharacterMap().get(member.character_id) || {};
      const showsEnergy = character.uses_energy !== false;
      const resources = slotResourcesAtCursor(member.slot);
      const energyLabel = formatNumber(resources.energy || 0, 1);
      const harmonyLabel = formatNumber(resources.harmony || 0, 1);
      const personalResourceLabels = timelinePersonalResourceEntries(character, resources.personalResources)
        .map(([name, value]) => `<small>${escapeHtml(name)} ${formatNumber(value, 1)}</small>`)
        .join('');
      const slotDetails = details
        .filter((detail) => Number(detail.slot) === Number(member.slot))
        .sort((a, b) => Number(a.display_start_tick ?? a.start_tick ?? 0) - Number(b.display_start_tick ?? b.start_tick ?? 0));
      const backgroundLaneEnds = [];
      const laidOut = slotDetails.map((detail) => {
        const displayStartTick = Number(detail.display_start_tick ?? detail.start_tick ?? 0);
        const displayEndTick = Number(detail.display_end_tick ?? detail.end_tick ?? displayStartTick);
        const displayVisualEndTick = Number(detail.display_visual_end_tick ?? detail.visual_end_tick ?? displayEndTick);
        let startPx = leftPx(displayStartTick);
        let cardWidth = widthPx(displayStartTick, displayEndTick, displayVisualEndTick);
        if (
          Number(detail.display_duration_ticks ?? detail.duration_ticks ?? 0) === 0 &&
          !detail.is_background_damage &&
          isQAction(actionForStep({ action_id: detail.action_id }))
        ) {
          cardWidth = MIN_ACTION_CARD_PX;
          if (displayVisualEndTick > displayStartTick + ZERO_ACTION_VISUAL_TICKS) {
            cardWidth = widthPx(displayStartTick, displayEndTick, displayVisualEndTick);
          }
        }
        let laneIndex = 0;
        if (detail.is_background_damage) {
          const backgroundLaneLimit = Math.max(1, MAX_VISUAL_LANES_PER_SLOT - 1);
          let backgroundLaneIndex = backgroundLaneEnds.findIndex((endPx) => startPx >= endPx);
          if (backgroundLaneIndex < 0) {
            if (backgroundLaneEnds.length < backgroundLaneLimit) {
              backgroundLaneIndex = backgroundLaneEnds.length;
              backgroundLaneEnds.push(0);
            } else {
              backgroundLaneIndex = backgroundLaneEnds.indexOf(Math.min(...backgroundLaneEnds));
            }
          }
          backgroundLaneEnds[backgroundLaneIndex] = startPx + cardWidth;
          laneIndex = backgroundLaneIndex + 1;
        }
        return { detail, laneIndex, startPx, cardWidth };
      });
      const layoutPxAtTick = (tick) => leftPx(Math.max(0, Number(tick || 0)));
      const layoutWidthPx = (startTick, endTick, minWidth = 1) => Math.max(
        minWidth,
        layoutPxAtTick(Math.max(Number(endTick || startTick), Number(startTick || 0) + 1)) - layoutPxAtTick(startTick),
      );
      const frontBands = frontWindowItems
        .filter((windowItem) => Number(windowItem.slot) === Number(member.slot))
        .map((windowItem) => `
          <span
            class="shaft-lane-front-band"
            style="left:${layoutPxAtTick(windowItem.start_tick)}px; width:${layoutWidthPx(windowItem.start_tick, windowItem.end_tick)}px; --slot-color:${color}"
            title="${escapeHtml(memberName(windowItem.slot))} 前台 ${escapeHtml(visualTickLabel(windowItem.start_tick))}"
          ></span>
        `).join('');
      const buffSegments = mergedBuffLineSegments(buffTimelineDetails, buffAxisEndTick, loopEnabled, member.slot);
      const reactionSegmentMap = new Map();
      reactionEffects
        .filter((effect) => Number(effect.trigger_slot) === Number(member.slot))
        .forEach((effect) => {
          reactionBuffLineSegments(effect, buffAxisEndTick, loopEnabled).forEach((segment) => {
            const key = `${effect.reaction}:${segment.startTick}:${segment.endTick}`;
            reactionSegmentMap.set(key, {
              startTick: segment.startTick,
              endTick: segment.endTick,
              reaction: effect.reaction,
              buffIds: effect.id,
              tooltip: `${effect.reaction} · ${ticksToSeconds(effect.duration_ticks || 1)}s`,
              tooltipItems: [{
                id: effect.id,
                name: effect.reaction,
                endTick: segment.endTick,
                calculationStartTick: Number(effect.start_tick || 0),
                calculationEndTick: Number(effect.end_tick || 0),
                loopDurationTicks: loopEnabled
                  ? Math.max(1, calculationTickFromVisual(Number(buffAxisEndTick || 0)))
                  : 0,
              durationTicks: effect.duration_ticks,
              stackCount: 1,
              visualStartTick: segment.startTick,
              visualEndTick: segment.endTick,
            }],
          });
          });
        });
      const reactionSegments = Array.from(reactionSegmentMap.values());
      const maxLaneIndex = laidOut.reduce((maxIndex, entry) => Math.max(maxIndex, entry.laneIndex), 0);
      let actionTopBase = 18;
      actionTopBase = Math.max(actionTopBase, Number(heldBuffPreview?.actionTopBaseBySlot?.[member.slot] || 0));
      let trackHeight = Math.max(132, actionTopBase + (maxLaneIndex + 1) * 38 + 20);
      trackHeight = Math.max(trackHeight, Number(heldBuffPreview?.trackHeightBySlot?.[member.slot] || 0));
      const buffLineTop = actionTopBase - 12;
      const heldBuffLines = heldBuffPreview?.buffLinesBySlot?.[String(member.slot)];
      const allBuffSegments = mergedRenderedBuffSegments([
        ...buffSegments.map((segment, index) => ({ ...segment, color: buffLineColor(index) })),
        ...reactionSegments.map((segment) => ({ ...segment, color: reactionLineColor(segment.reaction) })),
      ]);
      const buffLines = Array.isArray(heldBuffLines) ? heldBuffLines.join('') : allBuffSegments.map((segment) => {
        const tooltip = segment.tooltip || '';
        return `
          <span
            class="shaft-buff-trigger-line"
            data-buff-id="${escapeHtml(segment.buffIds || '')}"
            data-tooltip="${escapeHtml(tooltip)}"
            data-tooltip-items="${escapeHtml(JSON.stringify(segment.tooltipItems || []))}"
            tabindex="0"
            style="left:${layoutPxAtTick(segment.startTick)}px; top:${buffLineTop}px; width:${layoutWidthPx(segment.startTick, segment.endTick)}px; --slot-color:${color}; --buff-color:${segment.color}"
            aria-label="${escapeHtml(tooltip)}"
          ></span>
        `;
      }).join('');
      const reactionDamageMarkers = groupedDamageEvents
        .filter((event) => (
          Number(event.contributor_slot) === Number(member.slot)
          && Number(event.visual_tick ?? event.tick ?? 0) <= displayCutoffTick
        ))
        .map((event) => {
          const isPeriodicDamage = Boolean(event.kind);
          const opensUpward = isPeriodicDamage && Number(member.slot) === 3;
          const tooltip = damageMarkerTooltip(event);
          return `
          <span
            class="shaft-reaction-damage-marker ${isPeriodicDamage ? 'shaft-periodic-damage-marker' : ''} ${opensUpward ? 'shaft-damage-tooltip-above' : ''}"
            style="left:${leftPx(Number(event.visual_tick ?? event.tick ?? 0))}px; top:${isPeriodicDamage ? trackHeight - 5 : buffLineTop + 9}px; --reaction-color:${reactionLineColor(event.reaction)}"
            data-tooltip="${escapeHtml(tooltip)}"
            tabindex="0"
            aria-label="${escapeHtml(tooltip)}"
          ></span>
        `;
        }).join('');
      const bars = laidOut
        .map((entry) => {
          const detail = entry.detail;
          const isSelected = selectedIds.has(detail.step_id);
          const selected = isSelected ? 'selected' : '';
          const dragging = state.dragState?.stepIds?.includes(detail.step_id) ? 'dragging-preview' : '';
          const frontStart = !detail.is_background_damage ? 'front-start' : '';
          const basicBackground = detail.is_basic_background ? 'basic-background' : '';
          const qInstant = detail.q_instant_release ? `q-instant-release q-instant-${detail.q_instant_release_kind || 'release'}` : '';
          const top = actionTopBase + entry.laneIndex * 38;
          const actionMultiplier = backgroundActionMultiplier(
            (state.axis.steps || []).find((step) => step.id === detail.step_id),
            actionForStep({ action_id: detail.action_id }),
          );
          const actionMeta = [
            actionMultiplier > 1 ? `×${actionMultiplier}` : '',
            Number(detail.duration_ticks || 0) > 0 ? `${ticksToSeconds(detail.duration_ticks)}s` : '',
          ].filter(Boolean).join(' · ');
          const durationLabel = actionMeta ? `<em>${actionMeta}</em>` : '';
          return `
            <span class="shaft-action-bar ${actionTypeClass(detail.action_type)} ${detail.is_background_damage ? 'background-damage' : ''} ${basicBackground} ${frontStart} ${qInstant} ${selected} ${dragging}" data-step-id="${escapeHtml(detail.step_id)}" style="left:${entry.startPx}px; top:${top}px; width:${entry.cardWidth}px; --slot-color:${color}" title="${escapeHtml(detail.action_name)}${detail.q_instant_release ? ' · Q即时释放' : ''}">
              <span>${escapeHtml(detail.action_name)}</span>
              ${durationLabel}
            </span>
          `;
        }).join('');
      return `
        <div class="shaft-time-track action-track" data-slot="${member.slot}" style="width:${timelineTotalPx}px; height:${trackHeight}px">
          <span class="shaft-track-label">
            <span class="shaft-track-portrait">
              <img src="${escapeHtml(member.character_avatar || '')}" alt="">
              ${showsEnergy ? `<small>能量 ${energyLabel}</small>` : ''}
              <small>环合 ${harmonyLabel}</small>
              ${personalResourceLabels}
            </span>
            <span class="shaft-track-name">${escapeHtml(member.character_name)}</span>
          </span>
          <span class="shaft-track-surface"></span>
          ${frontBands}${buffLines}${reactionDamageMarkers}${bars}${cursor(false)}
        </div>
      `;
    }).join('');
    $('shaft-timeline').innerHTML = actionTracks;
    document.querySelectorAll('.shaft-cursor-line').forEach((node) => {
      node.style.left = `${leftPx(state.cursorTick)}px`;
    });
  }

  function actionPanelParticipation(action, detail, slot) {
    const multipliers = action?.multipliers || {};
    const usesAttack = Number(multipliers.atk || 0) !== 0;
    const usesHp = Number(multipliers.hp || 0) !== 0;
    const usesDefense = Number(multipliers.def || 0) !== 0;
    const hasScalingBase = usesAttack
      || usesHp
      || usesDefense
      || Number(multipliers.flat || 0) !== 0
      || Number(detail?.formula_parts?.base || 0) !== 0;
    const damageType = String(action?.damage_type || '');
    const hasDirectDamage = !['', '无'].includes(damageType) && hasScalingBase;
    const triggeredReaction = detail?.triggered_reaction;
    const contributesToTriggeredReaction = Boolean(
      triggeredReaction
      && Number(triggeredReaction.contributor_slot) === Number(slot)
    );
    const usesReactionAmplification = Number(detail?.formula_parts?.reaction_amplification ?? 1) !== 1;
    return {
      atk: usesAttack,
      hp: usesHp,
      def: usesDefense,
      harmony_strength: contributesToTriggeredReaction || usesReactionAmplification,
      stagger_strength: action?.damage_uses_stagger_strength === true
        || detail?.formula_parts?.uses_stagger_strength === true,
      damage: hasDirectDamage,
    };
  }

  function renderStepDetail() {
    const selectionCount = selectedStepIds().length;
    const step = selectedStep();
    const detail = step ? detailByStepId(step.id) : null;
    const action = step ? actionForStep(step) : null;
    const badge = $('shaft-selected-badge');
    const panel = $('shaft-step-detail');
    if (selectionCount > 1) {
      badge.textContent = `${selectionCount} 个动作`;
      panel.innerHTML = `
        <div class="shaft-detail-hero">
          <span class="shaft-detail-muted">当前框选</span>
          <strong>${selectionCount} 个动作</strong>
        </div>
      `;
      return;
    }
    if (!step || !action) {
      badge.textContent = '未选择';
      panel.innerHTML = '<div class="shaft-empty">选择时间轴上的动作</div>';
      return;
    }
    badge.textContent = visualTickLabel(step.start_tick);
    const panelStats = detail?.panel || {};
    const finalDamageBonus = Number(panelStats.final_dmg || 0);
    const otherDamageBonus = Number(
      panelStats.other_dmg
      ?? (
        Number(detail?.formula_parts?.dmg_bonus || 0)
        - Number(panelStats.all_dmg || 0)
        - Number(panelStats.element_dmg || 0)
      )
    );
    const panelParticipation = actionPanelParticipation(action, detail, step.slot);
    const realtimePanelRows = [];
    if (panelParticipation.atk) {
      realtimePanelRows.push(`<div class="shaft-detail-kv"><span>攻击</span><strong>${formatNumber(panelStats.atk || 0)}</strong></div>`);
    }
    if (panelParticipation.hp) {
      realtimePanelRows.push(`<div class="shaft-detail-kv"><span>生命</span><strong>${formatNumber(panelStats.hp || 0)}</strong></div>`);
    }
    if (panelParticipation.def) {
      realtimePanelRows.push(`<div class="shaft-detail-kv"><span>防御</span><strong>${formatNumber(panelStats.def || 0)}</strong></div>`);
    }
    if (panelParticipation.harmony_strength) {
      realtimePanelRows.push(`<div class="shaft-detail-kv"><span>环合强度</span><strong>${formatNumber(panelStats.harmony_strength || 0)}</strong></div>`);
    }
    if (panelParticipation.stagger_strength) {
      realtimePanelRows.push(`<div class="shaft-detail-kv"><span>倾陷强度</span><strong>${formatNumber(panelStats.stagger_strength || 0)}</strong></div>`);
    }
    if (panelParticipation.damage) {
      realtimePanelRows.push(
        `<div class="shaft-detail-kv"><span>暴击</span><strong>${formatNumber((panelStats.crit_rate || 0) * 100, 1)}%</strong></div>`,
        `<div class="shaft-detail-kv"><span>暴伤</span><strong>${formatNumber((panelStats.crit_dmg || 0) * 100, 1)}%</strong></div>`,
        `<div class="shaft-detail-kv"><span>通伤</span><strong>${formatNumber((panelStats.all_dmg || 0) * 100, 1)}%</strong></div>`,
        `<div class="shaft-detail-kv"><span>属伤</span><strong>${formatNumber((panelStats.element_dmg || 0) * 100, 1)}%</strong></div>`,
        `<div class="shaft-detail-kv"><span>其他增伤</span><strong>${formatNumber(otherDamageBonus * 100, 1)}%</strong></div>`,
        ...(finalDamageBonus ? [`<div class="shaft-detail-kv"><span>最终伤害</span><strong>${formatNumber(finalDamageBonus * 100, 1)}%</strong></div>`] : []),
        `<div class="shaft-detail-kv"><span>敌人结算属抗</span><strong>${formatNumber(Number(detail?.formula_parts?.settled_resistance || 0) * 100, 1)}%</strong></div>`,
        `<div class="shaft-detail-kv"><span>敌人结算防御</span><strong>${formatNumber(detail?.formula_parts?.settled_defense || 0, 1)}</strong></div>`,
      );
    }
    const durationTicks = Number(detail?.duration_ticks ?? action.duration_ticks ?? 0);
    const durationSeconds = durationTicks / 10;
    const character = getCharacterMap().get(memberBySlot(step.slot)?.character_id) || {};
    const showsEnergy = character.uses_energy !== false;
    const totalEnergyGain = Number(detail?.energy_gain ?? (Number(action.energy_gain || 0) + Number(action.energy_return || 0)));
    const extraEnergyGain = Number(detail?.energy_return ?? action.energy_return ?? 0);
    const energyGain = Number(detail?.base_energy_gain ?? Math.max(0, totalEnergyGain - extraEnergyGain));
    const harmonyGain = Number(detail?.harmony ?? action.harmony ?? 0);
    const buffEffectLabels = {
      atk_pct: '攻击', flat_atk: '固定攻击', hp_pct: '生命', flat_hp: '固定生命',
      def_pct: '防御', flat_def: '固定防御', crit_rate: '暴击', crit_dmg: '暴伤',
      def_ignore: '无视防御', res_down: '抗性降低', energy_recharge: '充能',
      harmony_strength: '环合强度', stagger_strength: '倾陷强度', basic_dmg: '普攻增伤',
      stagger_damage_bonus: '倾陷伤害增伤',
      dodge_counter_dmg: '闪反增伤',
      skill_dmg: '变轨增伤', ultimate_dmg: '终结增伤', follow_dmg: '追击增伤',
      mind_dmg: '心灵增伤', attach_dmg: '附着增伤', element_dmg: '属性增伤',
      all_dmg: '全伤增伤', final_dmg: '最终增伤', base_multiplier_pct: '基础倍率提升',
    };
    const flatBuffEffects = new Set(['flat_atk', 'flat_hp', 'flat_def', 'harmony_strength', 'stagger_strength']);
    const describeBuff = (buff) => {
      const effects = Object.entries(buff?.effects || {}).filter(([, value]) => Number(value));
      const effectText = effects.map(([key, value]) => {
        const label = buffEffectLabels[key] || (key.startsWith('res_down_') ? `${key.slice('res_down_'.length)}抗降低` : key);
        return `${label} ${Number(value) >= 0 ? '+' : ''}${flatBuffEffects.has(key) ? formatNumber(value, 1) : `${formatNumber(Number(value) * 100, 1)}%`}`;
      }).join('、');
      return `${buff?.name || '未命名增益'}${effectText ? `：${effectText}` : ''}`;
    };
    const detailLines = (items) => {
      const normalized = (items || []).map(String).filter(Boolean);
      if (!normalized.length) {
        return '<span>无</span>';
      }
      return normalized.map((item) => `<span>${escapeHtml(item)}</span>`).join('');
    };
    const appliedBuffList = (detail?.applied_buffs || []).filter((buff) => (
      !DETAIL_HIDDEN_APPLIED_BUFF_IDS.has(String(buff?.definition_id || '')) &&
      !DETAIL_HIDDEN_APPLIED_BUFF_IDS.has(String(buff?.rule_id || ''))
    ));
    const mergedAppliedBuffs = Array.from(appliedBuffList.reduce((groups, buff) => {
      const key = String(buff?.definition_id || buff?.rule_id || buff?.name || '');
      const current = groups.get(key);
      if (!current) {
        groups.set(key, {
          ...buff,
          effects: { ...(buff?.effects || {}) },
        });
        return groups;
      }
      current.stack_count = Number(current.stack_count || 0) + Number(buff?.stack_count || 0);
      Object.entries(buff?.effects || {}).forEach(([effectKey, value]) => {
        current.effects[effectKey] = Number(current.effects[effectKey] || 0) + Number(value || 0);
      });
      return groups;
    }, new Map()).values());
    const appliedBuffExplanations = mergedAppliedBuffs.map((buff) => {
      const stackCount = Number(buff?.stack_count || 0);
      const stackSuffix = stackCount > 1 ? ` · ${formatNumber(stackCount, 0)}层` : '';
      return `${describeBuff(buff)}${stackSuffix}`;
    });
    const triggeredBuffs = Array.from(new Set(
      (detail?.triggered_buffs || [])
        .map((buff) => String(buff?.name || ''))
        .filter(Boolean),
    ));
    const specialStats = [];
    const nightmareStacks = detail?.nightmare_stacks ?? action.nightmare_stacks;
    const sinRecovery = detail?.sin_recovery ?? action.sin_recovery;
    const hitCount = detail?.hit_count ?? action.hit_count;
    const appliedEnemyDebuffs = (detail?.applied_enemy_debuffs || []).join('、');
    const actionMultiplier = backgroundActionMultiplier(step, action);
    const realtimeAtk = Number(panelStats.atk || 0);
    const actionBase = Number(detail?.formula_parts?.base || 0) * actionMultiplier;
    const finalActionAtkMultiplier = realtimeAtk > 0
      ? actionBase / realtimeAtk
      : 0;
    const actionMultiplierText = `${formatNumber(finalActionAtkMultiplier * 100, 1)}%攻击`;
    specialStats.push(`<div class="shaft-detail-kv"><span>动作倍率</span><strong>${escapeHtml(actionMultiplierText)}</strong></div>`);
    if (hitCount !== undefined && hitCount !== null) {
      specialStats.push(`<div class="shaft-detail-kv"><span>伤害段数</span><strong>${formatNumber(hitCount, 0)}</strong></div>`);
    }
    const actionSelfAllDmg = Number(action.self_modifiers?.all_dmg || 0);
    if (actionSelfAllDmg) {
      specialStats.push(`<div class="shaft-detail-kv"><span>动作自增伤</span><strong>${formatNumber(actionSelfAllDmg * 100, 1)}%</strong></div>`);
    }
    if (appliedEnemyDebuffs) {
      specialStats.push(`<div class="shaft-detail-kv"><span>挂载状态</span><strong>${escapeHtml(appliedEnemyDebuffs)}</strong></div>`);
    }
    if (Number(nightmareStacks) > 0) {
      specialStats.push(`<div class="shaft-detail-kv"><span>噩梦层数</span><strong>${formatNumber(nightmareStacks, 0)}</strong></div>`);
    }
    if (sinRecovery !== undefined && sinRecovery !== null) {
      specialStats.push(`<div class="shaft-detail-kv"><span>回复罪状</span><strong>${formatNumber(sinRecovery, 0)}</strong></div>`);
    }
    panel.innerHTML = `
      <div class="shaft-detail-hero">
        <strong>${escapeHtml(action.name)}</strong>
        <span class="shaft-detail-muted">${escapeHtml(memberName(step.slot))} · ${escapeHtml(action.action_type || '动作')} · ${isStepBackground(step, action) ? (isBasicBackgroundOverride(step, action) ? '后台普攻' : '后台伤害') : '前台动作'}</span>
      </div>
      ${isBackgroundAction(action) ? `
        <div class="shaft-detail-multiplier">
          <span>动作倍数</span>
          <button class="secondary-btn" data-open-background-multiplier data-step-id="${escapeHtml(step.id)}" type="button" aria-haspopup="dialog">设置</button>
          <small>等效为同一时刻重叠 ${actionMultiplier} 个该后台动作</small>
        </div>
      ` : ''}
      <div class="shaft-detail-grid">
        <div class="shaft-detail-kv"><span>开始时间</span><strong>${ticksToSeconds(detail?.start_tick ?? calculationTickFromVisual(step.start_tick))}s</strong></div>
        <div class="shaft-detail-kv"><span>结束时间</span><strong>${ticksToSeconds(detail?.end_tick ?? calculationTickFromVisual(step.start_tick))}s</strong></div>
        <div class="shaft-detail-kv"><span>直伤</span><strong>${formatNumber(detail?.direct_damage || 0)}</strong></div>
        ${Number(detail?.fuwen_damage || 0) > 0 ? `<div class="shaft-detail-kv"><span>覆纹伤害</span><strong>${formatNumber(detail.fuwen_damage)}</strong></div>` : ''}
        <div class="shaft-detail-kv"><span>倾陷</span><strong>${formatNumber(detail?.stagger_amount || 0, 2)}</strong></div>
        <div class="shaft-detail-kv"><span>耗时</span><strong>${formatNumber(durationSeconds, 1)}s</strong></div>
        ${showsEnergy ? `<div class="shaft-detail-kv"><span>回能</span><strong>${formatNumber(energyGain, 1)}</strong></div>` : ''}
        ${showsEnergy ? `<div class="shaft-detail-kv"><span>额外回能</span><strong>${formatNumber(extraEnergyGain, 1)}</strong></div>` : ''}
        <div class="shaft-detail-kv"><span>环合</span><strong>${formatNumber(harmonyGain, 1)}</strong></div>
        ${specialStats.join('')}
      </div>
      <div class="shaft-detail-hero">
        <span class="shaft-detail-muted">实时面板</span>
        <strong>${escapeHtml(memberName(step.slot))}</strong>
      </div>
      <div class="shaft-detail-grid">
        ${realtimePanelRows.join('') || '<div class="shaft-empty">该动作不读取角色面板属性</div>'}
      </div>
      <div class="shaft-detail-hero">
        <span class="shaft-detail-muted">生效增益</span>
        <strong class="shaft-detail-line-list">${detailLines(appliedBuffExplanations)}</strong>
      </div>
      <div class="shaft-detail-hero">
        <span class="shaft-detail-muted">触发增益</span>
        <strong class="shaft-detail-line-list">${detailLines(triggeredBuffs)}</strong>
      </div>
    `;
  }

  function renderCharacterFilter(selectedIds, config) {
    const selected = new Set(selectedIds);
    const characters = (state.catalog?.characters || []).filter((character) => !character.selection_disabled);
    $(config.gridId).innerHTML = characters.map((character) => `
      <button class="shaft-character-filter-option ${selected.has(character.id) ? 'selected' : ''}"
              ${config.dataAttribute}="${escapeHtml(character.id)}"
              type="button"
              aria-pressed="${selected.has(character.id) ? 'true' : 'false'}">
        <img src="${escapeHtml(character.avatar || character.portrait || '')}" alt="" loading="lazy" decoding="async">
        <span>${escapeHtml(character.name)}</span>
        <small>${escapeHtml(character.element || '')}</small>
      </button>
    `).join('');
    const selectedCharacters = selectedIds
      .map((id) => characters.find((character) => character.id === id))
      .filter(Boolean);
    $(config.summaryId).innerHTML = selectedCharacters.length
      ? `<span class="shaft-character-filter-selected">${selectedCharacters.map((character) => `<img src="${escapeHtml(character.avatar || character.portrait || '')}" alt=""><b>${escapeHtml(character.name)}</b>`).join('')}</span>`
      : '全部角色';
    $(config.clearId).disabled = selectedCharacters.length === 0;
  }

  function renderMarketFilters() {
    renderCharacterFilter(state.marketCharacterIds, {
      gridId: 'shaft-character-filter-grid',
      summaryId: 'shaft-character-filter-summary',
      clearId: 'shaft-character-filter-clear',
      dataAttribute: 'data-market-character-id',
    });
    renderCharacterFilter(state.myAxisCharacterIds, {
      gridId: 'shaft-my-character-filter-grid',
      summaryId: 'shaft-my-character-filter-summary',
      clearId: 'shaft-my-character-filter-clear',
      dataAttribute: 'data-my-character-id',
    });
    $('shaft-my-axis-scope').value = state.myAxisFilter;
    $('shaft-my-axis-sort').value = state.myAxisSort;
  }

  function marketCardHtml(axis, compact, source = compact ? 'mine' : 'market') {
    const team = axis.team || [];
    const mine = source === 'mine';
    const publishedSnapshotId = Number(axis.published_snapshot_id || 0);
    const description = String(axis.description || '').trim();
    const descriptionCharacters = Array.from(description);
    const descriptionPreview = descriptionCharacters.length > 64
      ? `${descriptionCharacters.slice(0, 64).join('')}…`
      : description;
    return `
      <article class="shaft-market-card" data-axis-id="${axis.id}" data-axis-source="${source}">
        <div class="shaft-market-card-title">
          <strong>${escapeHtml(axis.title)}</strong>
          <div class="shaft-market-card-badges">
            <span class="shaft-axis-mode-badge">${axis.loop_enabled ? '循环' : '单轮'}</span>
            ${mine ? '<span>我的轴</span>' : ''}
          </div>
        </div>
        <div class="shaft-market-team" aria-label="队伍角色">
          ${team.map((member) => {
            const awakeningCount = activeAwakeningCount(member);
            return `
              <span class="shaft-market-character">
                <span class="shaft-market-character-avatar" title="${escapeHtml(member.character_name || '')} · 已激活 ${awakeningCount} 个觉醒">
                  <img src="${escapeHtml(member.character_avatar || '')}" alt="${escapeHtml(member.character_name || '')}" loading="lazy" decoding="async">
                  <span class="shaft-market-character-awakening" aria-label="已激活 ${awakeningCount} 个觉醒">${awakeningCount}</span>
                </span>
                <b>${escapeHtml(member.character_name || '')}</b>
              </span>
            `;
          }).join('')}
        </div>
        <p class="shaft-market-description ${description ? '' : 'is-empty'}" title="${escapeHtml(description)}">${escapeHtml(descriptionPreview || '暂无备注')}</p>
        <div class="shaft-market-stats">
          <span>DPS ${formatNumber(axis.dps || 0)}</span>
          <span>轴长 ${formatNumber(axis.duration_seconds || 0, 1)}s</span>
          <span>直伤 ${formatNumber(axis.direct_damage || 0)}</span>
          <span>环合 ${formatNumber(axis.harmony_damage || 0)}</span>
        </div>
        <div class="shaft-market-meta">${escapeHtml(axis.owner?.nickname || '')} · ${escapeHtml(axis.source_version || '')}</div>
        ${compact ? '' : `
          <div class="shaft-market-actions">
            <button class="secondary-btn ${axis.liked ? 'active' : ''}" data-like-axis="${axis.id}" type="button">赞 ${axis.like_count || 0}</button>
            <button class="secondary-btn shaft-dislike-btn ${axis.disliked ? 'active' : ''}" data-dislike-axis="${axis.id}" type="button">踩 ${axis.dislike_count || 0}</button>
            ${axis.is_owner
              ? `<span class="shaft-own-axis-note">自己的排轴</span><button class="secondary-btn danger-btn" data-unpublish-axis="${axis.id}" data-axis-title="${escapeHtml(axis.title)}" type="button">下架</button>`
              : `<button class="secondary-btn ${axis.favorited ? 'active' : ''}" data-favorite-axis="${axis.id}" type="button">藏 ${axis.favorite_count || 0}</button>`}
          </div>
        `}
        ${compact && mine ? `
          <div class="shaft-market-actions">
            <button class="secondary-btn" data-backup-axis="${axis.id}" type="button">备份</button>
            <button class="secondary-btn" data-share-axis="${axis.id}" type="button">分享</button>
            <button class="secondary-btn danger-btn" data-delete-axis="${axis.id}" data-axis-title="${escapeHtml(axis.title)}" type="button">删除</button>
            ${publishedSnapshotId
              ? `<button class="primary-btn shaft-publish-axis-btn" data-publish-axis="${axis.id}" data-publish-mode="update" type="button">更新</button><button class="secondary-btn danger-btn" data-unpublish-axis="${publishedSnapshotId}" data-axis-title="${escapeHtml(axis.title)}" type="button">下架</button>`
              : `<button class="primary-btn shaft-publish-axis-btn" data-publish-axis="${axis.id}" data-publish-mode="create" type="button">上传</button>`}
          </div>
        ` : ''}
        ${compact && !mine ? `
          <div class="shaft-market-actions">
            <button class="secondary-btn active" data-favorite-axis="${axis.id}" type="button">取消收藏</button>
          </div>
        ` : ''}
      </article>
    `;
  }

  function renderMarketList() {
    $('shaft-market-list').innerHTML = state.marketItems.map((axis) => marketCardHtml(axis, false)).join('') || '<div class="shaft-empty">暂无公开排轴</div>';
    $('shaft-market-more-btn').hidden = !state.marketHasMore;
  }

  function renderInsertMode() {
    document.querySelectorAll('[data-shaft-insert-mode]').forEach((button) => {
      button.classList.toggle('active', button.dataset.shaftInsertMode === state.insertMode);
    });
  }

  function applySharedReadOnlyUi() {
    const readOnly = Boolean(state.sharedReadOnly);
    document.querySelector('.shaft-page')?.classList.toggle('is-shared-readonly', readOnly);
    const notice = $('shaft-shared-readonly-notice');
    if (notice) {
      notice.hidden = !readOnly;
    }
    ['shaft-title-input', 'shaft-description-input'].forEach((id) => {
      const input = $(id);
      if (input) {
        input.readOnly = readOnly;
      }
    });
    ['shaft-new-btn', 'shaft-save-btn'].forEach((id) => {
      const button = $(id);
      if (button) {
        button.disabled = readOnly;
      }
    });
    const buildView = document.querySelector('[data-shaft-view="build"]');
    if (buildView) {
      buildView.inert = readOnly;
    }
    const commandLibrary = document.querySelector('.shaft-command-library');
    if (commandLibrary) {
      commandLibrary.inert = readOnly;
    }
    const detailPanel = document.querySelector('.shaft-detail-panel');
    if (detailPanel) {
      detailPanel.querySelectorAll('input, select, textarea, button').forEach((control) => {
        control.disabled = readOnly;
      });
    }
    document.querySelectorAll('.shaft-editor-actions button:not(#shaft-preview-btn)').forEach((button) => {
      if (readOnly) {
        button.disabled = true;
      }
    });
    renderSaveActions();
  }

  function renderAll() {
    setPage(state.page, false);
    renderLoginState();
    renderTeam();
    renderTeamDock();
    renderTeamBonus();
    renderEnemy();
    renderActionAdder();
    renderCommandTypes();
    renderCommandLibrary();
    renderSteps();
    renderBuffEditor();
    renderResults();
    renderTimeline();
    renderStepDetail();
    renderMarketFilters();
    renderEditorActions();
    applySharedReadOnlyUi();
  }

  function updateMemberNames(member) {
    if (!state.catalog) {
      return;
    }
    const character = getCharacterMap().get(member.character_id) || {};
    const arc = getArcMap().get(member.arc_id) || {};
    const cartridge = getCartridgeMap().get(member.cartridge_id) || {};
    member.character_name = character.name || member.character_name || '';
    member.character_avatar = character.avatar || '';
    member.character_element = character.element || '';
    member.arc_name = arc.name || '';
    member.cartridge_name = cartridge.name || '';
  }

  function removeInvalidStepsForSlot(slot) {
    const validActions = new Set(actionsForSlot(slot).map((action) => action.id));
    state.axis.steps = state.axis.steps.filter((step) => Number(step.slot) !== Number(slot) || validActions.has(step.action_id));
  }

  function sortSteps() {
    state.axis.steps.sort((a, b) => Number(a.start_tick) - Number(b.start_tick) || Number(a.slot) - Number(b.slot));
  }

  function applyForegroundConflictOrder(orderedSteps) {
    const slotBlockingEnd = new Map();
    let previousForegroundSlot = null;
    let previousForegroundStartTick = null;
    const foregroundLocks = [];
    for (const step of orderedSteps) {
      const action = actionForStep(step);
      const foregroundStart = startsForeground(step, action);
      const slotBlocking = blocksSlotOverlap(step, action) && !isInstantSwitchAction(action);
      if (!foregroundStart && !slotBlocking) {
        continue;
      }
      let startTick = Math.max(0, Number(step.start_tick || 0));
      if (slotBlocking) {
        startTick = Math.max(startTick, Number(slotBlockingEnd.get(Number(step.slot)) || 0));
      }
      if (foregroundStart) {
        let lockEndTick = Math.max(
          0,
          ...foregroundLocks
            .filter((lock) => startTick < Number(lock.endTick))
            .map((lock) => Number(lock.endTick)),
        );
        while (lockEndTick > startTick) {
          startTick = lockEndTick;
          lockEndTick = Math.max(
            0,
            ...foregroundLocks
              .filter((lock) => startTick < Number(lock.endTick))
              .map((lock) => Number(lock.endTick)),
          );
        }
      }
      if (
        foregroundStart &&
        previousForegroundSlot !== null &&
        previousForegroundSlot !== Number(step.slot) &&
        previousForegroundStartTick !== null &&
        !isInstantSwitchAction(action)
      ) {
        startTick = Math.max(startTick, previousForegroundStartTick + MIN_FOREGROUND_START_GAP_TICKS);
      }
      step.start_tick = startTick;
      const endTick = startTick + actionVisualDurationTicks(action, step);
      if (foregroundStart) {
        previousForegroundSlot = Number(step.slot);
        previousForegroundStartTick = startTick;
      }
      if (slotBlocking) {
        slotBlockingEnd.set(Number(step.slot), endTick);
      }
      if (locksForegroundSwitch(step, action)) {
        foregroundLocks.push({
          slot: Number(step.slot),
          endTick: foregroundLockEndTick(step, action, startTick),
        });
      }
    }
  }

  function foregroundConflictStepOrder(a, b) {
    const startDelta = Number(a.start_tick || 0) - Number(b.start_tick || 0);
    if (startDelta) {
      return startDelta;
    }
    const instantSwitchDelta = Number(isInstantSwitchAction(actionForStep(b))) -
      Number(isInstantSwitchAction(actionForStep(a)));
    if (instantSwitchDelta) {
      return instantSwitchDelta;
    }
    const lockDelta = Number(locksForegroundSwitch(b)) - Number(locksForegroundSwitch(a));
    return lockDelta || Number(a.slot) - Number(b.slot);
  }

  function resolveForegroundConflicts() {
    sortSteps();
    applyForegroundConflictOrder(state.axis.steps.slice().sort(foregroundConflictStepOrder));
  }

  function resolveForegroundConflictsWithPriority(priorityStepIds) {
    const priorityIds = new Set(priorityStepIds || []);
    if (!priorityIds.size) {
      resolveForegroundConflicts();
      return;
    }
    sortSteps();
    const prioritySteps = state.axis.steps
      .filter((step) => priorityIds.has(step.id) && (
        startsForeground(step, actionForStep(step)) || blocksSlotOverlap(step, actionForStep(step))
      ))
      .sort(foregroundConflictStepOrder);
    if (!prioritySteps.length) {
      resolveForegroundConflicts();
      return;
    }
    const priorityMinTick = Math.min(...prioritySteps.map((step) => Number(step.start_tick || 0)));
    const preSteps = [];
    const postSteps = [];
    state.axis.steps.forEach((step) => {
      const action = actionForStep(step);
      if (priorityIds.has(step.id) || (!startsForeground(step, action) && !blocksSlotOverlap(step, action))) {
        return;
      }
      const startTick = Number(step.start_tick || 0);
      const mustStayAhead = startTick < priorityMinTick;
      if (mustStayAhead) {
        preSteps.push(step);
      } else {
        postSteps.push(step);
      }
    });
    preSteps.sort(foregroundConflictStepOrder);
    postSteps.sort(foregroundConflictStepOrder);
    applyForegroundConflictOrder(preSteps.concat(prioritySteps, postSteps));
  }

  function preserveGapsAfterAutomaticShift(originalStartTicks, priorityStepIds = new Set()) {
    const priorityIds = new Set(priorityStepIds || []);
    const orderedSteps = (state.axis?.steps || [])
      .slice()
      .sort((a, b) => (
        Number(originalStartTicks.get(a.id) ?? a.start_tick ?? 0) - Number(originalStartTicks.get(b.id) ?? b.start_tick ?? 0) ||
        Number(a.slot || 0) - Number(b.slot || 0)
      ));
    const carriedShiftBySlot = new Map();
    let groupStart = 0;
    while (groupStart < orderedSteps.length) {
      const currentOriginalTick = Math.max(0, Number(
        originalStartTicks.get(orderedSteps[groupStart].id) ?? orderedSteps[groupStart].start_tick ?? 0,
      ));
      let groupEnd = groupStart + 1;
      while (
        groupEnd < orderedSteps.length &&
        Math.max(0, Number(originalStartTicks.get(orderedSteps[groupEnd].id) ?? orderedSteps[groupEnd].start_tick ?? 0)) === currentOriginalTick
      ) {
        groupEnd += 1;
      }
      const groupSteps = orderedSteps.slice(groupStart, groupEnd);
      groupSteps.forEach((step) => {
        const slot = Number(step.slot || 0);
        const carriedShiftTicks = Math.max(0, Number(carriedShiftBySlot.get(slot) || 0));
        const resolvedStartTick = Math.max(0, Number(step.start_tick || 0));
        if (priorityIds.has(step.id)) {
          step.start_tick = resolvedStartTick;
          carriedShiftBySlot.set(slot, 0);
          return;
        }
        step.start_tick = Math.max(resolvedStartTick, currentOriginalTick + carriedShiftTicks);
        carriedShiftBySlot.set(slot, Math.max(carriedShiftTicks, step.start_tick - currentOriginalTick));
      });
      groupStart = groupEnd;
    }
  }

  function normalizeEditedSteps(priorityStepIds = null) {
    const priorityIds = new Set(priorityStepIds || []);
    const originalStartTicks = new Map((state.axis?.steps || []).map((step) => [
      step.id,
      Math.max(0, Number(step.start_tick || 0)),
    ]));
    const resolveOnce = priorityIds.size
      ? () => resolveForegroundConflictsWithPriority(priorityIds)
      : () => resolveForegroundConflicts();
    const preserveOnce = priorityIds.size
      ? () => preserveGapsAfterAutomaticShift(originalStartTicks, priorityIds)
      : () => preserveGapsAfterAutomaticShift(originalStartTicks);
    const maxPasses = Math.max(2, (state.axis?.steps || []).length + 1);
    for (let pass = 0; pass < maxPasses; pass += 1) {
      const before = (state.axis?.steps || []).map((step) => `${step.id}:${Number(step.start_tick || 0)}`).join('|');
      resolveOnce();
      preserveOnce();
      const after = (state.axis?.steps || []).map((step) => `${step.id}:${Number(step.start_tick || 0)}`).join('|');
      if (after === before) {
        break;
      }
    }
    sortSteps();
    updateAxisDuration();
  }

  function actionIntervalAtTick(tick, slot = null) {
    const target = Math.max(0, Number(tick || 0));
    return (state.axis.steps || [])
      .map((step) => ({
        step,
        action: actionForStep(step),
        start: Number(step.start_tick || 0),
        end: Number(step.start_tick || 0) + actionVisualDurationTicks(actionForStep(step), step),
      }))
      .filter((item) => (
        startsForeground(item.step, item.action) &&
        !isInstantSwitchAction(item.action) &&
        item.start < target &&
        target < item.end &&
        (slot === null || Number(item.step.slot) === Number(slot))
      ))
      .sort((a, b) => b.end - a.end)[0] || null;
  }

  function prepareInsertionTick(tick, action = null, slot = null) {
    const target = Math.max(0, Number(tick || 0));
    const candidateStep = { action_id: action?.id || '' };
    if (
      isZeroForegroundQStep(candidateStep, action || {}) ||
      isInstantSwitchAction(action || {}) ||
      tickHasForegroundQ(target) ||
      tickHasInstantSwitchAction(target)
    ) {
      return target;
    }
    if (!startsForeground(candidateStep, action || {}) && !blocksSlotOverlap(candidateStep, action || {})) {
      return target;
    }
    if (
      startsForeground(candidateStep, action || {}) &&
      !actionIntervalAtTick(target, slot)
    ) {
      return target;
    }
    const interval = actionIntervalAtTick(tick);
    return interval ? interval.end : target;
  }

  function duplicateStartTick(tick, ignoreStepId = '', actionId = '') {
    const candidateAction = getActionMap().get(actionId) || {};
    const candidateStep = { action_id: actionId };
    if (!startsForeground(candidateStep, candidateAction)) {
      return false;
    }
    if (
      isZeroForegroundQStep(candidateStep, candidateAction) ||
      isInstantSwitchAction(candidateAction) ||
      tickHasForegroundQ(tick, ignoreStepId) ||
      tickHasInstantSwitchAction(tick, ignoreStepId)
    ) {
      return false;
    }
    return state.axis.steps.some((step) => (
      Number(step.start_tick) === Number(tick) &&
      step.id !== ignoreStepId &&
      startsForeground(step, actionForStep(step))
    ));
  }

  function selectStep(stepId, render = true, additive = false) {
    if (!stepId) {
      setSelectedStepIds([], '', render);
      return;
    }
    if (additive) {
      const ids = selectedStepIds();
      const nextIds = ids.includes(stepId) ? ids.filter((id) => id !== stepId) : ids.concat(stepId);
      setSelectedStepIds(nextIds, nextIds.includes(stepId) ? stepId : nextIds[nextIds.length - 1] || '', render);
      return;
    }
    setSelectedStepIds([stepId], stepId, render);
  }

  function insertionTickForAction() {
    if (state.insertMode === 'after-selected') {
      const step = selectedStep();
      const detail = step ? detailByStepId(step.id) : null;
      if (detail) {
        const action = actionForStep(step);
        return Number(step.start_tick || 0) + actionVisualDurationTicks(action, step);
      }
      if (step) {
        const action = actionForStep(step);
        return Number(step.start_tick || 0) + actionVisualDurationTicks(action, step);
      }
    }
    return state.cursorTick;
  }

  function addActionAt(slot, actionId, startTick) {
    const action = getActionMap().get(actionId) || {};
    if (!actionId) {
      setStatus('请选择动作', 'error');
      return;
    }
    pushUndoSnapshot();
    const insertTick = prepareInsertionTick(startTick, action, slot);
    const step = {
      id: `step_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`,
      slot,
      action_id: actionId,
      action_name: action.name || '',
      start_tick: insertTick,
      repeat: 1,
      tags: [],
    };
    state.axis.steps.push(step);
    normalizeEditedSteps(new Set([step.id]));
    selectStep(step.id, false);
    state.cursorTick = Number(step.start_tick || 0) + actionVisualDurationTicks(action, step);
    syncAddTimeInput(state.cursorTick);
    closeContextMenu();
    scheduleSimulation();
    revealTimelineTick(state.cursorTick);
  }

  function addStep() {
    const addSlot = $('shaft-add-slot');
    const addAction = $('shaft-add-action');
    const addTime = $('shaft-add-time');
    if (!addSlot || !addAction || !addTime) {
      return;
    }
    addActionAt(Number(addSlot.value || 0), addAction.value, tickFromTimeInput(addTime));
  }

  function removeStep(stepId) {
    if (!stepId || !state.axis.steps.some((step) => step.id === stepId)) {
      return;
    }
    removeSteps([stepId]);
  }

  function removeSteps(stepIds) {
    const ids = new Set(stepIds || []);
    if (!ids.size || !(state.axis?.steps || []).some((step) => ids.has(step.id))) {
      return;
    }
    const removedSteps = state.axis.steps.filter((step) => ids.has(step.id));
    const beforeDeleteResult = freshResult();
    const beforeDeleteDetails = clone(state.timelineDisplayDetails?.length
      ? state.timelineDisplayDetails
      : beforeDeleteResult?.details || []);
    const beforeDeleteFrozenIntervals = clone(beforeDeleteResult?.time_axis?.frozen_intervals || []);
    const primaryRemovedStep = removedSteps.find((step) => step.id === state.selectedStepId)
      || removedSteps.slice().sort((left, right) => Number(left.start_tick || 0) - Number(right.start_tick || 0))[0];
    const removedAtTick = Number(primaryRemovedStep?.start_tick ?? state.cursorTick ?? 0);
    pushUndoSnapshot();
    state.axis.steps = state.axis.steps.filter((step) => !ids.has(step.id));
    compactSpaceReleasedByDeletion(
      removedSteps,
      beforeDeleteDetails,
      beforeDeleteFrozenIntervals,
    );
    normalizeEditedSteps();
    state.selectedStepIds = selectedStepIds().filter((id) => !ids.has(id));
    if (!state.selectedStepIds.length) {
      const nearestStep = state.axis.steps.slice().sort((left, right) => (
        Math.abs(Number(left.start_tick || 0) - removedAtTick) - Math.abs(Number(right.start_tick || 0) - removedAtTick) ||
        Number(Number(left.start_tick || 0) < removedAtTick) - Number(Number(right.start_tick || 0) < removedAtTick) ||
        Number(left.start_tick || 0) - Number(right.start_tick || 0)
      ))[0];
      state.selectedStepIds = nearestStep ? [nearestStep.id] : [];
    }
    state.selectedStepId = state.selectedStepIds[state.selectedStepIds.length - 1] || '';
    syncSelection(Boolean(state.selectedStepId));
    state.cursorTick = removedAtTick;
    syncAddTimeInput(state.cursorTick);
    closeContextMenu();
    renderAll();
    scheduleSimulation();
    revealTimelineTick(removedAtTick);
  }

  function removeSelectedSteps() {
    removeSteps(selectedStepIds());
  }

  function copySelectedSteps() {
    const steps = selectedSteps()
      .sort((a, b) => Number(a.start_tick || 0) - Number(b.start_tick || 0) || Number(a.slot) - Number(b.slot));
    if (!steps.length) {
      return;
    }
    const minTick = Math.min(...steps.map((step) => Number(step.start_tick || 0)));
    state.clipboardSteps = steps.map((step) => Object.assign({}, clone(step), {
      relative_start_tick: Number(step.start_tick || 0) - minTick,
    }));
    renderEditorActions();
    setStatus(`已复制 ${steps.length} 个动作`);
  }

  function mergeReleasedTimelineIntervals(intervals) {
    return (intervals || [])
      .map((interval) => ({
        start: Math.max(0, Number(interval?.start || 0)),
        end: Math.max(0, Number(interval?.end || 0)),
      }))
      .filter((interval) => interval.end > interval.start)
      .sort((left, right) => left.start - right.start || left.end - right.end)
      .reduce((merged, interval) => {
        const previous = merged[merged.length - 1];
        if (previous && interval.start <= previous.end) {
          previous.end = Math.max(previous.end, interval.end);
        } else {
          merged.push(interval);
        }
        return merged;
      }, []);
  }

  function compactSpaceReleasedByDeletion(removedSteps, beforeDetails, beforeFrozenIntervals) {
    if (!state.axis || !removedSteps?.length) {
      return;
    }
    let afterDeleteResult = null;
    if (window.ShaftEngine && typeof window.ShaftEngine.simulateAxis === 'function') {
      try {
        afterDeleteResult = window.ShaftEngine.simulateAxis(state.axis, state.catalog);
      } catch (_error) {
        afterDeleteResult = null;
      }
    }
    const detailById = new Map((beforeDetails || []).map((detail) => [detail.step_id, detail]));
    const releasedIntervals = [];
    removedSteps.forEach((step) => {
      const action = actionForStep(step);
      if (!blocksSlotOverlap(step, action) || isZeroForegroundQStep(step, action)) {
        return;
      }
      const detail = detailById.get(step.id);
      const start = Math.max(0, Number(
        detail?.display_start_tick ?? detail?.visual_start_tick ?? step.start_tick ?? 0,
      ));
      const end = Math.max(start, Number(
        detail?.display_visual_end_tick ??
        detail?.visual_end_tick ??
        start + actionVisualDurationTicks(action, step),
      ));
      releasedIntervals.push({ start, end });
    });
    const afterFrozenIntervals = afterDeleteResult?.time_axis?.frozen_intervals || [];
    (beforeFrozenIntervals || []).forEach((beforeInterval) => {
      const start = Math.max(0, Number(beforeInterval?.start_tick || 0));
      const oldEnd = Math.max(start, Number(beforeInterval?.end_tick || start));
      const matchingAfter = afterFrozenIntervals.find((afterInterval) => (
        Number(afterInterval?.start_tick || 0) === start
      ));
      const newEnd = matchingAfter
        ? Math.max(start, Number(matchingAfter.end_tick || start))
        : start;
      if (oldEnd > newEnd) {
        releasedIntervals.push({ start: newEnd, end: oldEnd });
      }
    });
    const mergedIntervals = mergeReleasedTimelineIntervals(releasedIntervals);
    if (!mergedIntervals.length) {
      return;
    }
    state.axis.steps.forEach((step) => {
      const originalTick = Math.max(0, Number(step.start_tick || 0));
      const releasedBeforeStep = mergedIntervals.reduce((sum, interval) => (
        sum + Math.max(0, Math.min(originalTick, interval.end) - interval.start)
      ), 0);
      step.start_tick = Math.max(0, originalTick - releasedBeforeStep);
    });
  }

  function pasteStepsAtCursor() {
    if (!state.clipboardSteps.length || !state.axis) {
      return;
    }
    pushUndoSnapshot();
    const baseTick = prepareInsertionTick(state.cursorTick);
    const newIds = [];
    const clipboardSpanTicks = Math.max(1, ...state.clipboardSteps.map((source) => {
      const action = getActionMap().get(source.action_id) || {};
      return Number(source.relative_start_tick || 0) + actionVisualDurationTicks(action, source);
    }));
    state.axis.steps.forEach((step) => {
      if (Number(step.start_tick || 0) >= baseTick) {
        step.start_tick = Number(step.start_tick || 0) + clipboardSpanTicks;
      }
    });
    state.clipboardSteps.forEach((source) => {
      const action = getActionMap().get(source.action_id) || {};
      const id = `step_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
      newIds.push(id);
      state.axis.steps.push({
        id,
        slot: Number(source.slot || 0),
        action_id: source.action_id,
        action_name: action.name || source.action_name || '',
        start_tick: baseTick + Number(source.relative_start_tick || 0),
        repeat: Math.max(1, Number(source.repeat || 1)),
        placement: source.placement === 'background' ? 'background' : undefined,
        tags: clone(source.tags || []),
      });
    });
    state.axis.steps.forEach(sanitizeStepPlacement);
    normalizeEditedSteps(new Set(newIds));
    setSelectedStepIds(newIds.filter((id) => state.axis.steps.some((step) => step.id === id)), newIds[newIds.length - 1] || '', false);
    closeContextMenu();
    renderAll();
    scheduleSimulation();
    revealTimelineTick(state.cursorTick);
  }

  function addBuffRule() {
    const triggerSlotSelect = $('shaft-buff-trigger-slot');
    const triggerActionSelect = $('shaft-buff-trigger-action');
    const targetSlotList = $('shaft-buff-target-slots');
    const targetTypeSelect = $('shaft-buff-target-type');
    const modifierKeySelect = $('shaft-buff-modifier-key');
    const modifierValueInput = $('shaft-buff-modifier-value');
    const durationInput = $('shaft-buff-duration');
    if (!triggerSlotSelect || !triggerActionSelect || !targetSlotList || !targetTypeSelect || !modifierKeySelect || !modifierValueInput || !durationInput) {
      return;
    }
    const triggerSlot = Number(triggerSlotSelect.value || 0);
    const triggerActionId = triggerActionSelect.value;
    const triggerAction = getActionMap().get(triggerActionId);
    const targetSlots = Array.from(targetSlotList.querySelectorAll('input:checked')).map((input) => Number(input.value));
    const targetType = targetTypeSelect.value;
    const modifierKey = modifierKeySelect.value;
    const modifierValueRaw = Number(modifierValueInput.value || 0);
    const meta = modifierMeta(modifierKey);
    const modifierValue = meta.percent ? modifierValueRaw / 100 : modifierValueRaw;
    if (!triggerActionId || !modifierValue || targetSlots.length === 0) {
      setStatus('增益参数不完整', 'error');
      return;
    }
    state.axis.buff_rules.push({
      id: `buff_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`,
      name: `${memberName(triggerSlot)} ${triggerAction?.name || ''}`,
      trigger: { slot: triggerSlot, action_id: triggerActionId, action_type: '' },
      targets: { slots: targetSlots, action_ids: [], action_types: targetType ? [targetType] : [] },
      delay_ticks: 0,
      duration_ticks: secondsToTicks(durationInput.value || 10),
      modifiers: { [modifierKey]: modifierValue },
    });
    renderBuffList();
    scheduleSimulation();
  }

  function handleTeamChange(event) {
    const control = event.target.closest('[data-field]');
    if (!control) {
      return;
    }
    const slot = Number(control.dataset.slot);
    const member = memberBySlot(slot);
    if (!member) {
      return;
    }
    if (control.dataset.field === 'awakening') {
      const level = Math.max(1, Math.min(6, Number(control.dataset.awakeningLevel || 1)));
      const awakeningNodes = new Set(normalizeAwakeningNodes(member.awakening_nodes, member.awakening));
      if (control.checked) {
        awakeningNodes.add(level);
      } else {
        awakeningNodes.delete(level);
      }
      member.awakening_nodes = Array.from(awakeningNodes).sort((left, right) => left - right);
      member.awakening = member.awakening_nodes.length;
      rememberMemberBuild(member);
    } else if (control.dataset.field === 'bond_full') {
      const hasBondBonus = characterHasBondBonus(member.character_id);
      member.bond_full = hasBondBonus && control.checked;
      member.bond_level = member.bond_full ? 1 : 0;
      control.checked = member.bond_full;
      rememberMemberBuild(member);
    } else if (control.dataset.field === 'character_id') {
      applyMemberCharacterSelection(member, control.value);
    } else {
      if (control.dataset.field === 'arc_id') {
        member.arc_id = control.value;
        member.arc_refinement = defaultArcRefinement(member.arc_id);
      } else if (control.dataset.field === 'arc_refinement') {
        member.arc_refinement = clampArcRefinement(control.value, member.arc_id);
      } else {
        member[control.dataset.field] = control.value;
      }
      updateMemberNames(member);
      rememberMemberBuild(member);
    }
    renderAll();
    scheduleSimulation();
  }

  function applyMemberCharacterSelection(member, characterId) {
    if (!member || !characterId || member.character_id === characterId) {
      return false;
    }
    if (
      hasUnsavedAxisChanges() &&
      !state.characterSwitchUnsavedConfirmed &&
      !window.confirm('当前动作轴有未保存的修改，切换角色会清空该角色的动作，确定继续吗？')
    ) {
      return false;
    }
    if (hasUnsavedAxisChanges()) {
      state.characterSwitchUnsavedConfirmed = true;
    }
    rememberMemberBuild(member);
    member.character_id = characterId;
    const storedBuild = state.axis.character_builds?.[member.character_id];
    if (storedBuild) {
      applyBuildToMember(member, storedBuild);
    } else {
      updateMemberNames(member);
    }
    ensureMemberCompatibleArc(member);
    ensureMemberCompatibleCartridge(member);
    rememberMemberBuild(member);
    removeInvalidStepsForSlot(member.slot);
    return true;
  }

  function setBuildCharacterPickerOpen(targetPicker = null, open = false) {
    document.querySelectorAll('[data-build-character-picker]').forEach((picker) => {
      const shouldOpen = Boolean(open && picker === targetPicker);
      const popover = picker.querySelector('[data-build-character-popover]');
      const trigger = picker.querySelector('[data-build-character-trigger]');
      if (popover) {
        popover.hidden = !shouldOpen;
      }
      if (trigger) {
        trigger.setAttribute('aria-expanded', shouldOpen ? 'true' : 'false');
      }
    });
  }

  function handleTeamClick(event) {
    const characterOption = event.target.closest('[data-build-character-id]');
    if (characterOption) {
      const member = memberBySlot(Number(characterOption.dataset.slot));
      if (applyMemberCharacterSelection(member, characterOption.dataset.buildCharacterId)) {
        renderAll();
        scheduleSimulation();
      } else {
        setBuildCharacterPickerOpen();
      }
      return;
    }
    const trigger = event.target.closest('[data-build-character-trigger]');
    if (!trigger) {
      return;
    }
    const picker = trigger.closest('[data-build-character-picker]');
    const popover = picker?.querySelector('[data-build-character-popover]');
    setBuildCharacterPickerOpen(picker, Boolean(popover?.hidden));
  }

  function updateSubstatDraft(input) {
    const member = memberBySlot(Number(input.dataset.slot));
    if (!member) {
      return false;
    }
    const key = input.dataset.substat;
    const previous = clampSubstatCount(member.substat_counts?.[key]);
    const usedWithoutCurrent = Math.max(0, substatTotal(member.substat_counts) - previous);
    const count = Math.min(clampSubstatCount(input.value), Math.max(0, 120 - usedWithoutCurrent));
    input.value = count;
    member.substat_counts[key] = count;
    rememberMemberBuild(member);
    const tile = input.closest('.shaft-substat-tile');
    if (tile) {
      tile.style.setProperty('--count-pct', `${Math.max(0, Math.min(100, (count / 30) * 100))}%`);
      const unit = state.catalog?.formula_constants?.substat_units?.[key] || {};
      const small = tile.querySelector('small');
      if (small) {
        const unitValue = Number(unit.unit_value || 0);
        const unitLabel = unit.kind === 'percent'
          ? `${formatNumber(unitValue * 100, 2)}%/B`
          : `${formatNumber(unitValue, 0)}/B`;
        small.textContent = substatBonusText(key, count);
      }
    }
    const total = input.closest('.shaft-member-card')?.querySelector('.shaft-substat-head strong');
    if (total) total.textContent = `${formatNumber(substatTotal(member.substat_counts), 0)}/120`;
    return true;
  }

  function handleSubstatInput(event) {
    const input = event.target.closest('input[data-substat]');
    if (!input || !updateSubstatDraft(input)) {
      return;
    }
    window.clearTimeout(state.simulationTimer);
    state.simulationTimer = 0;
    markSimulationStale();
    persistAxisDraft();
    setStatus('编辑词条中，退出刷新', 'dirty');
  }

  function handleSubstatCommit(event) {
    const input = event.target.closest('input[data-substat]');
    if (!input || !updateSubstatDraft(input)) {
      return;
    }
    renderBuildPanel();
    scheduleSimulation();
  }

  function handleSubstatKeydown(event) {
    const input = event.target.closest('input[data-substat]');
    if (input && event.key === 'Enter') {
      event.preventDefault();
      input.blur();
    }
  }

  function handleSkillLevelInput(event) {
    const input = event.target.closest('input[data-skill-level]');
    if (!input) {
      return;
    }
    const member = memberBySlot(Number(input.dataset.slot));
    if (!member) {
      return;
    }
    const key = input.dataset.skillLevel;
    if (!SKILL_LEVEL_FIELDS.some((field) => field.key === key)) {
      return;
    }
    const level = clampSkillLevel(input.value);
    input.value = level;
    member.skill_levels = normalizeSkillLevels(Object.assign({}, member.skill_levels || {}, { [key]: level }));
    rememberMemberBuild(member);
    scheduleSimulation();
  }

  function handleCurtainInput(event) {
    const control = event.target.closest('[data-curtain-field]');
    if (!control) {
      return;
    }
    const member = memberBySlot(Number(control.dataset.slot));
    if (!member) {
      return;
    }
    member.curtain_bonus = normalizeCurtainBonus(member.curtain_bonus, member.character_id);
    rememberMemberBuild(member);
    renderTeam();
    renderBuildPanel();
    scheduleSimulation();
  }

  function handleTeamBonusInput(event) {
    const input = event.target.closest('input[data-team-bonus]');
    if (!input) {
      return;
    }
    const field = TEAM_PANEL_BONUS_FIELDS.find((item) => item.key === input.dataset.teamBonus);
    if (!field) {
      return;
    }
    const rawValue = numberValue(input.value);
    const storedValue = field.kind === 'percent' ? rawValue / 100 : rawValue;
    state.axis.team_panel_bonus = normalizeTeamPanelBonus(Object.assign(
      {},
      state.axis.team_panel_bonus || {},
      { [field.key]: storedValue },
    ));
    renderBuildPanel();
    scheduleSimulation();
  }

  function handleCompareControls(event) {
    if (event.target.closest('[data-save-compare-snapshot]')) {
      saveCompareSnapshot();
      return;
    }
    if (event.target.closest('[data-clear-compare-snapshot]')) {
      clearCompareSnapshot();
    }
  }

  function handleCompareControlChange(event) {
    void event;
  }

  function handleStepChange(event) {
    const target = event.target.closest('[data-step-field]');
    if (!target) {
      return;
    }
    const step = state.axis.steps.find((item) => item.id === target.dataset.stepId);
    if (!step) {
      return;
    }
    pushUndoSnapshot();
    if (target.dataset.stepField === 'slot') {
      step.slot = Number(target.value || 0);
      const actions = actionsForSlot(step.slot);
      if (!actions.some((action) => action.id === step.action_id)) {
        step.action_id = actions[0]?.id || '';
        step.action_name = getActionMap().get(step.action_id)?.name || '';
      }
      sanitizeStepPlacement(step);
    } else if (target.dataset.stepField === 'action_id') {
      step.action_id = target.value;
      step.action_name = getActionMap().get(step.action_id)?.name || '';
      sanitizeStepPlacement(step);
    } else if (target.dataset.stepField === 'start_tick') {
      const nextTick = tickFromTimeInput(target);
      step.start_tick = nextTick;
      state.cursorTick = nextTick;
    }
    normalizeEditedSteps();
    renderSteps();
    renderTimeline();
    renderStepDetail();
    scheduleSimulation();
  }

  function openBackgroundActionMultiplier(stepId, trigger = null) {
    const dialog = $('shaft-background-multiplier-dialog');
    const input = $('shaft-background-multiplier-input');
    const step = state.axis.steps.find((item) => item.id === stepId);
    const action = step ? actionForStep(step) : null;
    if (!dialog || !input || !step || !isBackgroundAction(action) || dialog.open) {
      return;
    }
    const actionMultiplier = backgroundActionMultiplier(step, action);
    dialog.dataset.stepId = step.id;
    dialog._returnFocus = trigger;
    input.value = String(actionMultiplier);
    $('shaft-background-multiplier-hint').textContent = `等效为同一时刻重叠 ${actionMultiplier} 个“${action.name || '后台动作'}”。`;
    dialog.showModal();
    window.requestAnimationFrame(() => {
      input.focus();
      input.select();
    });
  }

  function closeBackgroundActionMultiplier({ restoreFocus = true } = {}) {
    const dialog = $('shaft-background-multiplier-dialog');
    if (!dialog?.open) {
      return;
    }
    const returnFocus = dialog._returnFocus;
    dialog.close();
    dialog.removeAttribute('data-step-id');
    dialog._returnFocus = null;
    if (restoreFocus && returnFocus?.isConnected) {
      returnFocus.focus();
    }
  }

  function confirmBackgroundActionMultiplier() {
    const dialog = $('shaft-background-multiplier-dialog');
    const input = $('shaft-background-multiplier-input');
    const stepId = String(dialog?.dataset.stepId || '');
    const step = state.axis.steps.find((item) => item.id === stepId);
    const action = step ? actionForStep(step) : null;
    if (!dialog?.open || !input || !step || !isBackgroundAction(action)) {
      closeBackgroundActionMultiplier();
      return;
    }
    const nextMultiplier = Math.max(
      1,
      Math.min(MAX_BACKGROUND_ACTION_MULTIPLIER, Math.round(Number(input.value || 1))),
    );
    input.value = String(nextMultiplier);
    if (backgroundActionMultiplier(step, action) === nextMultiplier) {
      closeBackgroundActionMultiplier();
      return;
    }
    pushUndoSnapshot();
    step.repeat = nextMultiplier;
    closeBackgroundActionMultiplier({ restoreFocus: false });
    renderTimeline();
    window.clearTimeout(state.simulationTimer);
    markSimulationStale();
    persistAxisDraft();
    renderResults();
    renderSteps();
    renderEditorActions();
    setStatus(`后台动作倍数已设为 ×${nextMultiplier}`, 'dirty');
    state.simulationTimer = window.setTimeout(runSimulation, SIMULATION_DEBOUNCE_MS);
    window.requestAnimationFrame(() => {
      document.querySelector(`[data-open-background-multiplier][data-step-id="${CSS.escape(stepId)}"]`)?.focus();
    });
  }

  function handleStepDetailClick(event) {
    const button = event.target.closest('[data-open-background-multiplier]');
    if (!button) {
      return;
    }
    openBackgroundActionMultiplier(button.dataset.stepId, button);
  }

  function handleStepClick(event) {
    const removeButton = event.target.closest('[data-remove-step]');
    if (removeButton) {
      removeStep(removeButton.dataset.removeStep);
      return;
    }
    const row = event.target.closest('[data-step-id]');
    if (row) {
      selectStep(row.dataset.stepId, true, event.ctrlKey || event.metaKey);
    }
  }

  function handleBuffClick(event) {
    const button = event.target.closest('[data-remove-buff]');
    if (!button) {
      return;
    }
    state.axis.buff_rules = state.axis.buff_rules.filter((rule) => rule.id !== button.dataset.removeBuff);
    renderBuffList();
    scheduleSimulation();
  }

  function scrollBuildMemberIntoView(slot) {
    const card = document.querySelector(`#shaft-team-slots .shaft-member-card[data-slot="${Number(slot)}"]`);
    if (!card) {
      return;
    }
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    card.scrollIntoView({
      behavior: reduceMotion ? 'auto' : 'smooth',
      block: 'start',
      inline: 'nearest',
    });
  }

  function handleBuildPanelClick(event) {
    const button = event.target.closest('[data-build-panel-slot]');
    if (!button) {
      return;
    }
    state.buildPanelSlot = Number(button.dataset.buildPanelSlot || 0);
    renderBuildPanel();
    window.requestAnimationFrame(() => scrollBuildMemberIntoView(state.buildPanelSlot));
  }

  function timelineVisualOffsetWithScale(tick, scale = state.timelineScale) {
    const safeTick = Math.max(0, Number(tick || 0));
    return safeTick * TIMELINE_TICK_PX + (scale?.expansionBreaks || [])
      .filter((breakItem) => Number(breakItem.tick) < safeTick)
      .reduce((sum, breakItem) => sum + Number(breakItem.extraPx || 0), 0);
  }

  function timelineVisualOffset(tick) {
    return timelineVisualOffsetWithScale(tick, state.timelineScale);
  }

  function snapTickAfterCrossingVisualStart(drag, clientX, fallbackTick) {
    const originLeft = Number(drag?.originBarLeft);
    const anchorOffsetX = Number(drag?.anchorOffsetX);
    if (!Number.isFinite(originLeft) || !Number.isFinite(anchorOffsetX)) {
      return fallbackTick;
    }
    const nextLeft = Number(clientX) - anchorOffsetX;
    if (!Number.isFinite(nextLeft) || nextLeft >= originLeft) {
      return fallbackTick;
    }
    const crossedTarget = (drag?.visualStartSnapTargets || [])
      .filter((target) => (
        Number.isFinite(Number(target.left)) &&
        Number(target.left) >= nextLeft &&
        Number(target.left) < originLeft
      ))
      .sort((left, right) => Number(left.left) - Number(right.left))[0];
    return crossedTarget ? Math.max(0, Number(crossedTarget.startTick || 0)) : fallbackTick;
  }

  function axisTickFromDragDisplayTick(drag, displayTick) {
    const originAxisTick = Number(drag?.originTick || 0);
    const originDisplayTick = Number(drag?.originDisplayTick ?? originAxisTick);
    return Math.max(0, originAxisTick + Number(displayTick || 0) - originDisplayTick);
  }

  function revealTimelineTick(tick) {
    const timeline = $('shaft-timeline');
    const shell = timeline?.closest('.shaft-timeline-shell');
    if (!shell) {
      return;
    }
    const targetLeft = TIMELINE_LABEL_PX + timelineVisualOffset(tick);
    const paddingPx = 32;
    const visibleLeft = shell.scrollLeft + TIMELINE_LABEL_PX + paddingPx;
    const visibleRight = shell.scrollLeft + shell.clientWidth - paddingPx;
    if (targetLeft < visibleLeft) {
      shell.scrollLeft = Math.max(0, targetLeft - TIMELINE_LABEL_PX - paddingPx);
    } else if (targetLeft > visibleRight) {
      shell.scrollLeft = Math.max(0, targetLeft - shell.clientWidth + paddingPx);
    }
  }

  function tickFromTimelineXWithScale(x, scale = state.timelineScale, durationTicks = state.timelineDurationTicks) {
    const target = Math.max(0, Number(x || 0));
    const maxTick = Math.max(
      0,
      Number(durationTicks || TIMELINE_END_PADDING_TICKS),
      Math.ceil(target / TIMELINE_TICK_PX) + TIMELINE_END_PADDING_TICKS,
    );
    let low = 0;
    let high = maxTick;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (timelineVisualOffsetWithScale(middle, scale) < target) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }
    if (low <= 0) {
      return 0;
    }
    const previous = low - 1;
    return Math.abs(timelineVisualOffsetWithScale(low, scale) - target) < Math.abs(target - timelineVisualOffsetWithScale(previous, scale)) ? low : previous;
  }

  function tickFromTimelineX(x) {
    return tickFromTimelineXWithScale(x, state.timelineScale, state.timelineDurationTicks);
  }

  function clearActionLayoutAnimation(node) {
    if (!node) {
      return;
    }
    node.style.transition = '';
    node.style.transform = '';
    delete node.dataset.layoutAnimationToken;
  }

  function captureActionRects(options = {}) {
    if (options.flushAnimations) {
      document.querySelectorAll('.shaft-action-bar[data-layout-animation-token]').forEach(clearActionLayoutAnimation);
    }
    const rects = new Map();
    document.querySelectorAll('.shaft-action-bar[data-step-id]').forEach((node) => {
      const rect = node.getBoundingClientRect();
      rects.set(node.dataset.stepId, {
        left: rect.left,
        top: rect.top,
      });
    });
    return rects;
  }

  function captureBuffDragPreview() {
    const timeline = $('shaft-timeline');
    if (!timeline) {
      return null;
    }
    const actionTopBaseBySlot = {};
    const trackHeightBySlot = {};
    const buffLinesBySlot = {};
    timeline.querySelectorAll('.action-track[data-slot]').forEach((track) => {
      const slot = String(track.dataset.slot || 0);
      const trackRect = track.getBoundingClientRect();
      trackHeightBySlot[slot] = Math.max(trackRect.height, Number.parseFloat(track.style.height) || 0);
      buffLinesBySlot[slot] = Array.from(track.querySelectorAll('.shaft-buff-trigger-line'))
        .map((node) => node.outerHTML);
      const actionTops = Array.from(track.querySelectorAll('.shaft-action-bar'))
        .map((node) => Number.parseFloat(node.style.top || ''))
        .filter((value) => Number.isFinite(value));
      if (actionTops.length) {
        actionTopBaseBySlot[slot] = Math.min(...actionTops);
      }
    });
    if (!Object.keys(trackHeightBySlot).length) {
      return null;
    }
    return {
      actionTopBaseBySlot,
      trackHeightBySlot,
      buffLinesBySlot,
    };
  }

  function animateActionLayout(previousRects, options = {}) {
    if (!previousRects || previousRects.size === 0) {
      return;
    }
    const skipStepIds = new Set(options.skipStepIds || []);
    const durationMs = Math.max(0, Number(options.durationMs || 150));
    document.querySelectorAll('.shaft-action-bar[data-step-id]').forEach((node) => {
      const stepId = node.dataset.stepId || '';
      if (skipStepIds.has(stepId)) {
        clearActionLayoutAnimation(node);
        return;
      }
      const previous = previousRects.get(stepId);
      if (!previous) {
        return;
      }
      const rect = node.getBoundingClientRect();
      const dx = previous.left - rect.left;
      const dy = previous.top - rect.top;
      if (Math.abs(dx) < 1 && Math.abs(dy) < 1) {
        return;
      }
      const token = `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
      node.dataset.layoutAnimationToken = token;
      node.style.transition = 'none';
      node.style.transform = `translate(${dx}px, ${dy}px)`;
      node.getBoundingClientRect();
      window.requestAnimationFrame(() => {
        if (node.dataset.layoutAnimationToken !== token) {
          return;
        }
        node.style.transition = `transform ${durationMs}ms cubic-bezier(.2,.8,.2,1), box-shadow 150ms ease, opacity 150ms ease`;
        node.style.transform = 'translate(0, 0)';
        window.setTimeout(() => {
          if (node.dataset.layoutAnimationToken === token) {
            clearActionLayoutAnimation(node);
          }
        }, durationMs + 30);
      });
      });
  }

  function selectionBoxFromPoints(startX, startY, currentX, currentY) {
    return {
      left: Math.min(startX, currentX),
      top: Math.min(startY, currentY),
      right: Math.max(startX, currentX),
      bottom: Math.max(startY, currentY),
      width: Math.abs(currentX - startX),
      height: Math.abs(currentY - startY),
    };
  }

  function rectsIntersect(a, b) {
    return a.left <= b.right && a.right >= b.left && a.top <= b.bottom && a.bottom >= b.top;
  }

  function marqueeElement() {
    let node = document.querySelector('.shaft-selection-marquee');
    if (!node) {
      node = document.createElement('div');
      node.className = 'shaft-selection-marquee';
      document.body.appendChild(node);
    }
    return node;
  }

  function updateMarqueeElement(rect) {
    const node = marqueeElement();
    node.style.left = `${rect.left}px`;
    node.style.top = `${rect.top}px`;
    node.style.width = `${rect.width}px`;
    node.style.height = `${rect.height}px`;
  }

  function removeMarqueeElement() {
    document.querySelector('.shaft-selection-marquee')?.remove();
  }

  function selectActionsInMarquee(rect, additive, baseIds) {
    const ids = additive ? (baseIds || []).slice() : [];
    document.querySelectorAll('.shaft-action-bar[data-step-id]').forEach((node) => {
      const barRect = node.getBoundingClientRect();
      if (rectsIntersect(rect, barRect) && !ids.includes(node.dataset.stepId)) {
        ids.push(node.dataset.stepId);
      }
    });
    setSelectedStepIds(ids, ids[ids.length - 1] || '', false);
    renderStepDetail();
    renderEditorActions();
  }

  function placementForDrag(step, action, originalPlacement, deltaY) {
    if (!canBackgroundOverride(action) || isBackgroundAction(action)) {
      return 'foreground';
    }
    if (deltaY > BACKGROUND_DRAG_THRESHOLD_PX) {
      return 'background';
    }
    if (deltaY < -BACKGROUND_DRAG_THRESHOLD_PX) {
      return 'foreground';
    }
    return originalPlacement === 'background' ? 'background' : 'foreground';
  }

  function applyDraggedPlacement(step, drag, deltaY) {
    const action = actionForStep(step);
    const originalPlacement = drag.originPlacements?.[step.id] || 'foreground';
    const nextPlacement = placementForDrag(step, action, originalPlacement, deltaY);
    if (nextPlacement === 'background') {
      step.placement = 'background';
    } else {
      delete step.placement;
    }
    return nextPlacement !== originalPlacement;
  }

  function dragPlacementWouldChange(drag, deltaY) {
    return (drag.stepIds || [drag.stepId]).some((stepId) => {
      const step = (drag.snapshot.steps || []).find((item) => item.id === stepId);
      if (!step) {
        return false;
      }
      const action = getActionMap().get(step.action_id) || {};
      const originalPlacement = drag.originPlacements?.[stepId] || 'foreground';
      return placementForDrag(step, action, originalPlacement, deltaY) !== originalPlacement;
    });
  }

  function updateSelectionClasses() {
    const selectedIds = new Set(selectedStepIds());
    document.querySelectorAll('.shaft-action-bar[data-step-id]').forEach((node) => {
      node.classList.toggle('selected', selectedIds.has(node.dataset.stepId));
    });
    document.querySelectorAll('.shaft-step-row[data-step-id]').forEach((node) => {
      node.classList.toggle('selected', selectedIds.has(node.dataset.stepId));
    });
  }

  function timelineTickFromEvent(event) {
    const timeline = $('shaft-timeline');
    const rect = timeline.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left - TIMELINE_LABEL_PX));
    return tickFromTimelineX(x);
  }

  function positionBuffLineTooltip(event) {
    const buffLine = event.target.closest('.shaft-buff-trigger-line');
    if (!buffLine) {
      return;
    }
    const rect = buffLine.getBoundingClientRect();
    const offsetX = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
    buffLine.style.setProperty('--buff-tooltip-x', `${offsetX}px`);
    let tooltipItems = [];
    try {
      tooltipItems = JSON.parse(buffLine.dataset.tooltipItems || '[]');
    } catch (_error) {
      tooltipItems = [];
    }
    if (!tooltipItems.length) {
      return;
    }
    const pointerVisualTick = timelineTickFromEvent(event);
    const pointerCalculationTick = calculationTickFromVisual(pointerVisualTick);
    buffLine.dataset.tooltipCalculationTick = String(pointerCalculationTick);
    const activeTooltipItems = Array.from(new Map(tooltipItems
      .filter((item) => (
        item.visualStartTick == null || item.visualEndTick == null || (
          Number(item.visualStartTick) <= pointerVisualTick && pointerVisualTick < Number(item.visualEndTick)
        )
      ))
      .map((item) => [String(item.id || item.name || ''), item]))
      .values());
    const tooltip = activeTooltipItems.map((item) => {
      const stackText = buffStackText(item.stackCount);
      const durationTicks = Math.max(1, Number(item.durationTicks || 1));
      if (durationTicks > BUFF_DURATION_LABEL_LIMIT_TICKS) {
        return `${item.name}${stackText}`;
      }
      const remainingTicks = buffRemainingTicksAtCalculationTick(item, pointerCalculationTick);
      return `${item.name} · 剩余${ticksToSeconds(remainingTicks)}s${stackText}`;
    }).join('\n');
    buffLine.dataset.tooltip = tooltip;
    buffLine.setAttribute('aria-label', tooltip);
  }

  function positionDamageMarkerTooltip(event) {
    const marker = event.target.closest('.shaft-reaction-damage-marker');
    const shell = marker?.closest('.shaft-timeline-shell');
    if (!marker || !shell) {
      return;
    }
    const markerRect = marker.getBoundingClientRect();
    const shellRect = shell.getBoundingClientRect();
    const tooltipHalfWidth = Math.min(280, Math.max(0, (shellRect.width - 16) / 2));
    const markerCenter = markerRect.left + markerRect.width / 2;
    const tooltipCenter = Math.max(
      shellRect.left + 8 + tooltipHalfWidth,
      Math.min(shellRect.right - 8 - tooltipHalfWidth, markerCenter),
    );
    marker.style.setProperty('--damage-tooltip-shift-x', `${tooltipCenter - markerCenter}px`);
  }

  function slotFromTimelineEvent(event) {
    const track = event.target.closest('.action-track[data-slot]');
    return track ? Number(track.dataset.slot || 0) : Number(state.librarySlot || 0);
  }

  function closeContextMenu() {
    const menu = $('shaft-context-menu');
    if (!menu) {
      return;
    }
    menu.hidden = true;
    menu.innerHTML = '';
  }

  function showContextMenu(left, top, html) {
    const menu = $('shaft-context-menu');
    if (!menu) {
      return;
    }
    menu.innerHTML = html;
    menu.hidden = false;
    const rect = menu.getBoundingClientRect();
    menu.style.left = `${Math.min(left, window.innerWidth - rect.width - 12)}px`;
    menu.style.top = `${Math.min(top, window.innerHeight - rect.height - 12)}px`;
  }

  function actionContextMenuHtml(slot, tick) {
    const actions = actionsForSlot(slot);
    return `
      <div class="shaft-context-head">
        <strong>${escapeHtml(memberName(slot))}</strong>
        <span>${escapeHtml(visualTickLabel(tick))}</span>
      </div>
      <div class="shaft-context-actions">
        ${actions.map((action) => `
          <button type="button" data-context-add-action="${escapeHtml(action.id)}" data-context-slot="${slot}" data-context-tick="${tick}">
            <b>${escapeHtml(action.action_type || '动作')}</b>
            <span>${escapeHtml(action.name)}</span>
            <em>${ticksToSeconds(actionCalculationDurationTicks(action))}s</em>
          </button>
        `).join('') || '<div class="shaft-empty">没有动作</div>'}
      </div>
    `;
  }

  function removeContextMenuHtml(stepId) {
    const step = state.axis.steps.find((item) => item.id === stepId);
    const action = actionForStep(step);
    return `
      <div class="shaft-context-head">
        <strong>${escapeHtml(action.name || '动作')}</strong>
        <span>${escapeHtml(visualTickLabel(step?.start_tick || 0))}</span>
      </div>
      <button class="shaft-context-danger" type="button" data-context-remove-step="${escapeHtml(stepId)}">移除动作</button>
    `;
  }

  function handleTimelineClick(event) {
    if (Date.now() < state.suppressTimelineClickUntil) {
      return;
    }
    closeContextMenu();
    const bar = event.target.closest('[data-step-id]');
    if (bar) {
      selectStep(bar.dataset.stepId, true, event.ctrlKey || event.metaKey);
      return;
    }
    state.cursorTick = timelineTickFromEvent(event);
    syncAddTimeInput(state.cursorTick);
    renderTimeline();
    renderStepDetail();
    renderEditorActions();
  }

  function handleTimelineContextMenu(event) {
    event.preventDefault();
    if (state.sharedReadOnly) {
      return;
    }
    const bar = event.target.closest('.shaft-action-bar[data-step-id]');
    if (bar) {
      selectStep(bar.dataset.stepId);
      showContextMenu(event.clientX, event.clientY, removeContextMenuHtml(bar.dataset.stepId));
      return;
    }
    const tick = timelineTickFromEvent(event);
    const slot = slotFromTimelineEvent(event);
    state.cursorTick = tick;
    state.librarySlot = slot;
    const addSlot = $('shaft-add-slot');
    if (addSlot) {
      addSlot.value = String(slot);
    }
    syncAddTimeInput(tick);
    renderTeamDock();
    renderActionAdder();
    renderCommandLibrary();
    renderTimeline();
    showContextMenu(event.clientX, event.clientY, actionContextMenuHtml(slot, tick));
  }

  function handleContextMenuClick(event) {
    if (state.sharedReadOnly) {
      return;
    }
    const addButton = event.target.closest('[data-context-add-action]');
    if (addButton) {
      addActionAt(
        Number(addButton.dataset.contextSlot || state.librarySlot || 0),
        addButton.dataset.contextAddAction,
        Number(addButton.dataset.contextTick || state.cursorTick || 0),
      );
      return;
    }
    const removeButton = event.target.closest('[data-context-remove-step]');
    if (removeButton) {
      removeStep(removeButton.dataset.contextRemoveStep);
    }
  }

  function isEditableTarget(target) {
    return Boolean(target?.closest?.('input, textarea, select, [contenteditable="true"]'));
  }

  function focusTimelineForShortcuts() {
    const timeline = $('shaft-timeline');
    if (!timeline) {
      return;
    }
    timeline.tabIndex = -1;
    timeline.focus({ preventScroll: true });
  }

  function moveTimelineCursorTo(tick) {
    const nextTick = Math.max(0, Number(tick || 0));
    if (nextTick === Number(state.cursorTick || 0)) {
      return;
    }
    state.cursorTick = nextTick;
    syncAddTimeInput(state.cursorTick);
    renderTimeline();
    renderStepDetail();
    renderEditorActions();
    revealTimelineTick(state.cursorTick);
  }

  function moveTimelineCursor(deltaTicks) {
    moveTimelineCursorTo(Number(state.cursorTick || 0) + Number(deltaTicks || 0));
  }

  function stepVisualEndTickForCursor(step) {
    const detail = (state.timelineDisplayDetails || []).find((item) => item.step_id === step?.id);
    const fallbackEndTick = stepEndTick(step, true);
    return Math.max(
      Number(detail?.display_start_tick ?? detail?.visual_start_tick ?? step?.start_tick ?? 0),
      Number(detail?.display_visual_end_tick ?? detail?.visual_end_tick ?? fallbackEndTick),
    );
  }

  function foregroundNeighbor(step, direction, ignoredStepIds = new Set()) {
    if (!step) {
      return null;
    }
    const startTick = Number(step.start_tick || 0);
    const candidates = (state.axis?.steps || [])
      .filter((candidate) => (
        !ignoredStepIds.has(candidate.id) &&
        Number(candidate.slot) === Number(step.slot) &&
        startsForeground(candidate, actionForStep(candidate)) &&
        (direction < 0
          ? Number(candidate.start_tick || 0) < startTick
          : Number(candidate.start_tick || 0) > startTick)
      ))
      .sort((left, right) => Number(left.start_tick || 0) - Number(right.start_tick || 0));
    return direction < 0 ? candidates[candidates.length - 1] || null : candidates[0] || null;
  }

  function swapStepWithNeighbor(step, neighbor, direction, requestedStartTick) {
    if (!step || !neighbor) {
      return false;
    }
    const stepStart = Number(step.start_tick || 0);
    const neighborStart = Number(neighbor.start_tick || 0);
    const stepDuration = actionVisualDurationTicks(actionForStep(step), step);
    const neighborDuration = actionVisualDurationTicks(actionForStep(neighbor), neighbor);
    if (direction < 0) {
      if (requestedStartTick >= neighborStart + neighborDuration) {
        return false;
      }
      const spanEnd = stepStart + stepDuration;
      step.start_tick = neighborStart;
      neighbor.start_tick = Math.max(step.start_tick + stepDuration, spanEnd - neighborDuration);
      return true;
    }
    if (requestedStartTick + stepDuration <= neighborStart) {
      return false;
    }
    const spanEnd = neighborStart + neighborDuration;
    neighbor.start_tick = stepStart;
    step.start_tick = Math.max(neighbor.start_tick + neighborDuration, spanEnd - stepDuration);
    return true;
  }

  function finishKeyboardStepEdit(stepIds, primaryId, statusText) {
    const selectedIds = new Set(stepIds || []);
    normalizeEditedSteps(selectedIds);
    setSelectedStepIds(Array.from(selectedIds), primaryId, false);
    const primary = state.axis.steps.find((step) => step.id === primaryId);
    state.cursorTick = Number(primary?.start_tick || 0);
    syncAddTimeInput(state.cursorTick);
    closeContextMenu();
    renderAll();
    revealTimelineTick(state.cursorTick);
    scheduleSimulation(statusText);
  }

  function moveSelectedStepsByKeyboard(direction) {
    const stepIds = selectedStepIds();
    const steps = selectedSteps();
    const primary = selectedStep();
    if (!primary || !steps.length) {
      return;
    }
    const deltaTicks = direction < 0 ? -1 : 1;
    const selectedIds = new Set(stepIds);
    const minStartTick = Math.min(...steps.map((step) => Number(step.start_tick || 0)));
    const safeDeltaTicks = Math.max(deltaTicks, -minStartTick);
    if (safeDeltaTicks === 0) {
      return;
    }
    pushUndoSnapshot();
    let swapped = false;
    if (steps.length === 1 && startsForeground(primary, actionForStep(primary))) {
      const neighbor = foregroundNeighbor(primary, direction, selectedIds);
      const requestedStartTick = Number(primary.start_tick || 0) + safeDeltaTicks;
      swapped = swapStepWithNeighbor(primary, neighbor, direction, requestedStartTick);
    }
    if (!swapped) {
      steps.forEach((step) => {
        step.start_tick = Math.max(0, Number(step.start_tick || 0) + safeDeltaTicks);
      });
    }
    finishKeyboardStepEdit(stepIds, primary.id, swapped ? '已交换动作位置' : '已移动动作');
  }

  function selectStepByKeyboard(key) {
    const primary = selectedStep();
    if (!primary) {
      return;
    }

    const steps = state.axis?.steps || [];
    let target = null;
    if (key === 'a' || key === 'd') {
      const trackSteps = steps
        .filter((step) => Number(step.slot) === Number(primary.slot))
        .sort((left, right) => (
          Number(left.start_tick || 0) - Number(right.start_tick || 0) ||
          steps.indexOf(left) - steps.indexOf(right)
        ));
      const currentIndex = trackSteps.findIndex((step) => step.id === primary.id);
      target = trackSteps[currentIndex + (key === 'a' ? -1 : 1)] || null;
    } else if (key === 'w' || key === 's') {
      const slots = Array.from(new Set([
        ...(state.axis?.team || []).map((member) => Number(member.slot)),
        ...steps.map((step) => Number(step.slot)),
      ])).sort((left, right) => left - right);
      const slotDirection = key === 'w' ? -1 : 1;
      let slotIndex = slots.indexOf(Number(primary.slot)) + slotDirection;
      while (slotIndex >= 0 && slotIndex < slots.length && !target) {
        const candidates = steps
          .filter((step) => Number(step.slot) === slots[slotIndex])
          .sort((left, right) => (
            Math.abs(Number(left.start_tick || 0) - Number(primary.start_tick || 0)) -
              Math.abs(Number(right.start_tick || 0) - Number(primary.start_tick || 0)) ||
            Number(left.start_tick || 0) - Number(right.start_tick || 0) ||
            steps.indexOf(left) - steps.indexOf(right)
          ));
        target = candidates[0] || null;
        slotIndex += slotDirection;
      }
    }

    if (target) {
      selectStep(target.id);
      revealTimelineTick(Number(target.start_tick || 0));
    } else if (key === 'd') {
      moveTimelineCursorTo(stepVisualEndTickForCursor(primary));
    }
  }

  function selectAllTimelineSteps() {
    const stepIds = (state.axis?.steps || []).map((step) => step.id).filter(Boolean);
    setSelectedStepIds(stepIds, stepIds[stepIds.length - 1] || '');
  }

  function handleKeydown(event) {
    if (event.key === 'Escape') {
      if (document.querySelector('[data-build-character-popover]:not([hidden])')) {
        setBuildCharacterPickerOpen();
        return;
      }
      if ($('shaft-background-multiplier-dialog')?.open) {
        closeBackgroundActionMultiplier();
        return;
      }
      if ($('shaft-loop-settings-dialog')?.open) {
        closeLoopSettings();
        return;
      }
      if ($('shaft-axis-preview-dialog')?.open) {
        closeAxisPreview();
        return;
      }
      if ($('shaft-shortcut-dialog')?.open) {
        closeShortcutHelp();
        return;
      }
      if ($('shaft-build-info-dialog')?.open) {
        closeBuildInfo();
        return;
      }
      if ($('shaft-action-contribution-dialog')?.open) {
        closeActionContribution();
        return;
      }
      closeContextMenu();
      return;
    }
    if (state.sharedReadOnly) {
      return;
    }
    if (isEditableTarget(event.target)) {
      return;
    }
    if (
      (event.ctrlKey || event.metaKey) &&
      event.key.toLowerCase() === 'a' &&
      state.page === 'rotation'
    ) {
      event.preventDefault();
      selectAllTimelineSteps();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
      event.preventDefault();
      if (event.shiftKey) {
        redoLastEdit();
      } else {
        undoLastEdit();
      }
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') {
      event.preventDefault();
      redoLastEdit();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'c') {
      event.preventDefault();
      copySelectedSteps();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'v') {
      event.preventDefault();
      state.suppressClipboardPasteUntil = Date.now() + 250;
      pasteStepsAtCursor();
      return;
    }
    if ((event.key === 'Delete' || event.key === 'Backspace') && selectedStepIds().length) {
      event.preventDefault();
      removeSelectedSteps();
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey) {
      return;
    }
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      moveTimelineCursor(event.key === 'ArrowLeft' ? -1 : 1);
      return;
    }
    const movementKey = event.key.toLowerCase();
    if (movementKey === 'w' || movementKey === 'a' || movementKey === 's' || movementKey === 'd') {
      if (!selectedStepIds().length) {
        return;
      }
      event.preventDefault();
      selectStepByKeyboard(movementKey);
      return;
    }
    if (movementKey === 'q' || movementKey === 'e') {
      if (!selectedStepIds().length) {
        return;
      }
      event.preventDefault();
      moveSelectedStepsByKeyboard(movementKey === 'q' ? -1 : 1);
    }
  }

  function handleClipboardCopy(event) {
    if (state.sharedReadOnly) {
      return;
    }
    if (isEditableTarget(event.target) || !selectedStepIds().length) {
      return;
    }
    event.preventDefault();
    copySelectedSteps();
    try {
      event.clipboardData?.setData('text/plain', `异环动作轴 ${selectedStepIds().length} 个动作`);
    } catch (error) {
      // The in-page clipboard is only a convenience; internal copy state is enough for pasting.
    }
  }

  function handleClipboardPaste(event) {
    if (state.sharedReadOnly) {
      return;
    }
    if (isEditableTarget(event.target) || !state.clipboardSteps.length) {
      return;
    }
    event.preventDefault();
    if (Date.now() < state.suppressClipboardPasteUntil) {
      return;
    }
    pasteStepsAtCursor();
  }

  function timelineShell() {
    return $('shaft-timeline')?.closest('.shaft-timeline-shell') || null;
  }

  function timelineAutoScrollVelocity(pointer, start, end) {
    const edge = TIMELINE_AUTO_SCROLL_EDGE_PX;
    if (pointer < start + edge) {
      return -TIMELINE_AUTO_SCROLL_MAX_PX * Math.min(1, Math.max(0, (start + edge - pointer) / edge));
    }
    if (pointer > end - edge) {
      return TIMELINE_AUTO_SCROLL_MAX_PX * Math.min(1, Math.max(0, (pointer - end + edge) / edge));
    }
    return 0;
  }

  function stopTimelineAutoScroll() {
    const autoScroll = state.timelineAutoScroll;
    if (autoScroll?.frameId) {
      window.cancelAnimationFrame(autoScroll.frameId);
    }
    state.timelineAutoScroll = null;
  }

  function updateTimelineMarquee(clientX, clientY) {
    const marquee = state.marqueeState;
    const shell = timelineShell();
    if (!marquee || !shell) {
      return;
    }
    const scrollDeltaX = shell.scrollLeft - Number(marquee.originScrollLeft || 0);
    const scrollDeltaY = shell.scrollTop - Number(marquee.originScrollTop || 0);
    const originX = marquee.originX - scrollDeltaX;
    const originY = marquee.originY - scrollDeltaY;
    const rect = selectionBoxFromPoints(originX, originY, clientX, clientY);
    if (rect.width <= 3 && rect.height <= 3) {
      return;
    }
    marquee.moved = true;
    marquee.currentX = clientX;
    marquee.currentY = clientY;
    updateMarqueeElement(rect);
    selectActionsInMarquee(rect, marquee.additive, marquee.baseIds);
  }

  function updateTimelineDrag(event) {
    const drag = state.dragState;
    const shell = timelineShell();
    if (!drag || !shell) {
      return;
    }
    const dragStepIds = new Set(drag.stepIds || [drag.stepId]);
    const originOffset = Number.isFinite(Number(drag.originVisualOffset))
      ? Number(drag.originVisualOffset)
      : timelineVisualOffsetWithScale(drag.originTick, drag.timelineScale || state.timelineScale);
    const scrollDeltaX = shell.scrollLeft - Number(drag.originScrollLeft || 0);
    const scrollDeltaY = shell.scrollTop - Number(drag.originScrollTop || 0);
    const mappedDisplayTick = Math.max(0, tickFromTimelineXWithScale(
      originOffset + event.clientX - drag.originX + scrollDeltaX,
      drag.timelineScale || state.timelineScale,
      drag.timelineDurationTicks || state.timelineDurationTicks,
    ));
    const mappedTick = axisTickFromDragDisplayTick(drag, mappedDisplayTick);
    const nextTick = snapTickAfterCrossingVisualStart(drag, event.clientX, mappedTick);
    const deltaY = event.clientY - Number(drag.originY || event.clientY) + scrollDeltaY;
    const canChangePlacement = dragStepIds.size === 1;
    const placementChanged = canChangePlacement && dragPlacementWouldChange(drag, deltaY);
    if (nextTick === Number(drag.previewTick || 0) && !placementChanged) {
      return;
    }
    state.axis.steps = clone(drag.snapshot.steps || []);
    state.selectedStepIds = clone(Array.from(dragStepIds));
    state.selectedStepId = drag.stepId;
    const movedSteps = state.axis.steps.filter((item) => dragStepIds.has(item.id));
    if (!movedSteps.length) {
      return;
    }
    drag.moved = true;
    drag.previewTick = nextTick;
    const deltaTicks = nextTick - Number(drag.originTick || 0);
    const minOriginTick = Math.min(...Object.values(drag.originTicks || { [drag.stepId]: drag.originTick }).map(Number));
    const safeDeltaTicks = Math.max(deltaTicks, -minOriginTick);
    movedSteps.forEach((item) => {
      item.start_tick = Math.max(0, Number(drag.originTicks?.[item.id] ?? item.start_tick ?? 0) + safeDeltaTicks);
      if (canChangePlacement) {
        applyDraggedPlacement(item, drag, deltaY);
      }
    });
    normalizeEditedSteps(dragStepIds);
    if (window.ShaftEngine && typeof window.ShaftEngine.simulateAxis === 'function') {
      try {
        drag.previewResult = window.ShaftEngine.simulateAxis(state.axis, state.catalog);
      } catch (error) {
        drag.previewResult = null;
      }
    }
    syncSelection(false);
    const normalizedStep = state.axis.steps.find((item) => item.id === drag.stepId);
    state.cursorTick = Number(normalizedStep?.start_tick ?? nextTick);
    syncAddTimeInput(state.cursorTick);
    renderSteps();
    renderTimeline();
    renderStepDetail();
    renderEditorActions();
  }

  function updateTimelineInteraction(clientX, clientY) {
    if (state.marqueeState) {
      updateTimelineMarquee(clientX, clientY);
    } else if (state.dragState) {
      updateTimelineDrag({ clientX, clientY });
    }
  }

  function runTimelineAutoScroll() {
    const autoScroll = state.timelineAutoScroll;
    const shell = timelineShell();
    if (!autoScroll || !shell || (!state.dragState && !state.marqueeState)) {
      stopTimelineAutoScroll();
      return;
    }
    const bounds = shell.getBoundingClientRect();
    const horizontalStart = Math.min(bounds.right, bounds.left + TIMELINE_LABEL_PX);
    const velocityX = timelineAutoScrollVelocity(autoScroll.clientX, horizontalStart, bounds.right);
    const velocityY = timelineAutoScrollVelocity(autoScroll.clientY, bounds.top, bounds.bottom);
    const previousLeft = shell.scrollLeft;
    const previousTop = shell.scrollTop;
    shell.scrollLeft += velocityX;
    shell.scrollTop += velocityY;
    if (shell.scrollLeft !== previousLeft || shell.scrollTop !== previousTop) {
      updateTimelineInteraction(autoScroll.clientX, autoScroll.clientY);
    }
    autoScroll.frameId = window.requestAnimationFrame(runTimelineAutoScroll);
  }

  function trackTimelineAutoScroll(clientX, clientY) {
    if (!state.dragState && !state.marqueeState) {
      stopTimelineAutoScroll();
      return;
    }
    if (!state.timelineAutoScroll) {
      state.timelineAutoScroll = {
        clientX,
        clientY,
        frameId: 0,
      };
    } else {
      state.timelineAutoScroll.clientX = clientX;
      state.timelineAutoScroll.clientY = clientY;
    }
    if (!state.timelineAutoScroll.frameId) {
      state.timelineAutoScroll.frameId = window.requestAnimationFrame(runTimelineAutoScroll);
    }
  }

  function handleTimelineMouseDown(event) {
    if (state.sharedReadOnly) {
      return;
    }
    const bar = event.target.closest('.shaft-action-bar[data-step-id]');
    if (event.button !== 0) {
      return;
    }
    if (!bar) {
      if (!event.target.closest('#shaft-timeline')) {
        return;
      }
      state.marqueeState = {
        originX: event.clientX,
        originY: event.clientY,
        originScrollLeft: timelineShell()?.scrollLeft || 0,
        originScrollTop: timelineShell()?.scrollTop || 0,
        currentX: event.clientX,
        currentY: event.clientY,
        baseIds: selectedStepIds(),
        additive: event.ctrlKey || event.metaKey,
        moved: false,
      };
      event.preventDefault();
      return;
    }
    const step = state.axis.steps.find((item) => item.id === bar.dataset.stepId);
    if (!step) {
      return;
    }
    if (event.ctrlKey || event.metaKey) {
      selectStep(step.id, true, true);
      state.suppressTimelineClickUntil = Date.now() + 250;
      event.preventDefault();
      return;
    }
    const pressedBarRect = bar.getBoundingClientRect();
    const domSelectedIds = Array.from(document.querySelectorAll('.shaft-action-bar.selected[data-step-id]'))
      .map((node) => node.dataset.stepId)
      .filter(Boolean);
    const activeIds = Array.from(new Set(selectedStepIds().concat(domSelectedIds)));
    if (!activeIds.includes(step.id)) {
      selectStep(step.id);
    } else {
      setSelectedStepIds(activeIds, step.id, false);
      renderStepDetail();
      renderEditorActions();
    }
    const dragStepIds = activeIds.includes(step.id) ? activeIds : [step.id];
    const originTicks = {};
    const originAxisTicks = {};
    const originPlacements = {};
    (state.axis.steps || []).forEach((item) => {
      originAxisTicks[item.id] = Number(item.start_tick || 0);
      if (dragStepIds.includes(item.id)) {
        originTicks[item.id] = Number(item.start_tick || 0);
        originPlacements[item.id] = item.placement === 'background' ? 'background' : 'foreground';
      }
    });
    const activeBar = Array.from(document.querySelectorAll('.shaft-action-bar[data-step-id]'))
      .find((node) => node.dataset.stepId === step.id) || bar;
    const barRect = activeBar.isConnected ? activeBar.getBoundingClientRect() : pressedBarRect;
    const dragDisplayDetail = (state.timelineDisplayDetails || []).find((detail) => detail.step_id === step.id);
    const originDisplayTick = Math.max(0, Number(
      dragDisplayDetail?.display_start_tick ?? dragDisplayDetail?.visual_start_tick ?? step.start_tick ?? 0,
    ));
    const visualStartSnapTargets = Array.from(
      activeBar.closest('.action-track')?.querySelectorAll('.shaft-action-bar[data-step-id]') || [],
    ).flatMap((node) => {
      const targetStep = state.axis.steps.find((item) => item.id === node.dataset.stepId);
      if (!targetStep || dragStepIds.includes(targetStep.id) || !startsForeground(targetStep, actionForStep(targetStep))) {
        return [];
      }
      return [{
        left: node.getBoundingClientRect().left,
        startTick: Math.max(0, Number(targetStep.start_tick || 0)),
      }];
    });
    state.dragState = {
      stepId: step.id,
      stepIds: dragStepIds,
      originX: event.clientX,
      originY: event.clientY,
      originScrollLeft: timelineShell()?.scrollLeft || 0,
      originScrollTop: timelineShell()?.scrollTop || 0,
      originTick: Number(step.start_tick || 0),
      originDisplayTick,
      originBarLeft: barRect.left,
      anchorOffsetX: event.clientX - barRect.left,
      visualStartSnapTargets,
      originVisualOffset: timelineVisualOffset(originDisplayTick),
      timelineScale: clone(state.timelineScale || { expansionBreaks: [] }),
      timelineDurationTicks: Number(state.timelineDurationTicks || TIMELINE_END_PADDING_TICKS),
      resultSnapshot: freshResult(),
      originTicks,
      originAxisTicks,
      originPlacements,
      snapshot: editorSnapshot(),
      buffPreview: captureBuffDragPreview(),
      previewTick: Number(step.start_tick || 0),
      moved: false,
    };
    document.body.classList.add('shaft-dragging');
    event.preventDefault();
  }

  function handleTimelineMouseMove(event) {
    if (!state.marqueeState && !state.dragState) {
      return;
    }
    trackTimelineAutoScroll(event.clientX, event.clientY);
    updateTimelineInteraction(event.clientX, event.clientY);
  }

  function handleTimelineMouseUp(event) {
    const marquee = state.marqueeState;
    if (marquee) {
      stopTimelineAutoScroll();
      state.marqueeState = null;
      removeMarqueeElement();
      if (marquee.moved) {
        state.suppressTimelineClickUntil = Date.now() + 250;
      } else if (event) {
        state.cursorTick = timelineTickFromEvent(event);
        syncAddTimeInput(state.cursorTick);
        renderTimeline();
        renderStepDetail();
        renderEditorActions();
      }
      return;
    }
    const drag = state.dragState;
    if (!drag) {
      return;
    }
    stopTimelineAutoScroll();
    state.dragState = null;
    document.body.classList.remove('shaft-dragging');
    if (!drag.moved) {
      return;
    }
    const stepIds = drag.stepIds || [drag.stepId];
    const movedSteps = state.axis.steps.filter((item) => stepIds.includes(item.id));
    if (movedSteps.length) {
      state.undoStack.push(drag.snapshot);
      if (state.undoStack.length > 80) {
        state.undoStack.shift();
      }
      state.redoStack = [];
      normalizeEditedSteps(new Set(stepIds));
      syncSelection(false);
      const primary = state.axis.steps.find((item) => item.id === drag.stepId) || movedSteps[0];
      state.cursorTick = Number(primary?.start_tick || 0);
      state.suppressTimelineClickUntil = Date.now() + 250;
      renderSteps();
      renderStepDetail();
      renderEditorActions();
      runSimulation();
    }
  }

  function handleTeamDockClick(event) {
    const card = event.target.closest('[data-dock-slot]');
    if (!card) {
      return;
    }
    state.librarySlot = Number(card.dataset.dockSlot || 0);
    renderTeamDock();
    renderActionAdder();
    renderCommandLibrary();
  }

  function handleLibraryClick(event) {
    if (state.sharedReadOnly) {
      return;
    }
    const button = event.target.closest('[data-library-action]');
    if (!button) {
      return;
    }
    addActionAt(Number(button.dataset.librarySlot || state.librarySlot), button.dataset.libraryAction, insertionTickForAction());
  }

  function emptySimulationResult() {
    return {
      ok: true,
      summary: {
        duration_ticks: 0,
        duration_seconds: 0,
        direct_damage: 0,
        harmony_damage: 0,
        stagger_damage: 0,
        total_damage: 0,
        dps: 0,
        team_energy: 0,
        total_harmony: 0,
      },
      damage_by_slot: [],
      damage_by_action_by_slot: [],
      damage_by_source: [],
      resources_by_slot: [],
      build_panels_by_slot: currentBuildPanels(),
      details: [],
      reaction_effects: [],
      reaction_damage_events: [],
      front_windows: [],
      time_axis: {
        tick_seconds: 0.1,
        timeline_ticks: 0,
        real_duration_ticks: 0,
        frozen_intervals: [],
      },
      enemy: state.axis?.enemy || {},
    };
  }

  function renderLocalCalculationState(statusText = '未重新计算') {
    window.clearTimeout(state.simulationTimer);
    state.simulationTimer = 0;
    updateAxisDuration();
    markSimulationStale();
    persistAxisDraft();
    renderResults();
    renderSteps();
    renderTimeline();
    renderStepDetail();
    renderEditorActions();
    setStatus(statusText, 'dirty');
  }

  async function runSimulation() {
    window.clearTimeout(state.simulationTimer);
    state.simulationTimer = 0;
    if (!Array.isArray(state.axis?.steps) || !state.axis.steps.length) {
      acceptSimulationResult(emptySimulationResult());
      renderResults();
      renderSteps();
      renderTimeline();
      renderStepDetail();
      renderEditorActions();
      persistAxisDraft();
      setStatus('待排轴');
      return;
    }
    if (freshResult()) {
      setStatus('已计算');
      return;
    }
    if (state.simulationInFlight) {
      setStatus('计算中');
      return;
    }
    setSimulationBusy(true);
    try {
      setStatus('计算中');
      if (!window.ShaftEngine || typeof window.ShaftEngine.simulateAxis !== 'function') {
        throw new Error('本地计算引擎未加载。');
      }
      const result = window.ShaftEngine.simulateAxis(state.axis, state.catalog);
      state.axis.enemy = result.enemy || state.axis.enemy;
      state.axis.duration_ticks = Number(result.summary?.duration_ticks || state.axis.duration_ticks || 1);
      ensureAxisShape();
      acceptSimulationResult(result);
      renderTeamDock();
      renderResults();
      renderSteps();
      renderTimeline();
      renderStepDetail();
      renderEditorActions();
      persistAxisDraft();
      setStatus('已计算');
    } catch (error) {
      setStatus(error.message, 'error');
    } finally {
      setSimulationBusy(false);
    }
  }

  function scheduleSimulation(statusText = '未重新计算') {
    renderLocalCalculationState(statusText);
    state.simulationTimer = window.setTimeout(runSimulation, SIMULATION_DEBOUNCE_MS);
  }

  function newAxis() {
    if (hasUnsavedAxisChanges() && !window.confirm('当前动作轴有未保存的修改，确定新建并放弃这些修改吗？')) {
      return;
    }
    state.axis = emptyAxisFromCatalog();
    state.result = null;
    state.resultFingerprint = '';
    state.isResultStale = false;
    state.compareSnapshot = null;
    state.savedAxisId = null;
    state.savedAxisTitle = '';
    state.cursorTick = 0;
    state.selectedStepId = '';
    state.undoStack = [];
    state.redoStack = [];
    $('shaft-title-input').value = '未命名排轴';
    $('shaft-description-input').value = '';
    ensureAxisShape();
    markAxisDocumentClean();
    renderAll();
    persistAxisDraft();
    runSimulation();
  }

  function suggestedConflictTitle(title) {
    const suffix = '-副本';
    const base = String(title || '未命名排轴').trim() || '未命名排轴';
    return `${base.slice(0, Math.max(0, 80 - suffix.length))}${suffix}`;
  }

  async function resolveAxisNameConflict(conflict) {
    const dialog = $('shaft-name-conflict-dialog');
    $('shaft-name-conflict-message').textContent = `“${conflict.title || '未命名排轴'}”已经存在。你可以换一个名称，或用当前内容覆盖原排轴。`;
    $('shaft-name-conflict-input').value = suggestedConflictTitle(conflict.title);
    dialog.returnValue = '';
    dialog.showModal();
    const action = await new Promise((resolve) => {
      dialog.addEventListener('close', () => resolve(dialog.returnValue || 'cancel'), { once: true });
    });
    return {
      action,
      title: $('shaft-name-conflict-input').value.trim(),
    };
  }

  async function saveAxis(conflictAction = '') {
    if (!getToken()) {
      persistAxisDraft();
      redirectToLogin();
      return false;
    }
    if (!freshResult()) {
      await runSimulation();
    }
    const result = freshResult();
    if (!result) {
      setStatus('请等待本地计算完成后再保存', 'error');
      return false;
    }
    const payload = {
      title: $('shaft-title-input').value || '未命名排轴',
      description: $('shaft-description-input').value || '',
      axis: state.axis,
      result,
    };
    if (conflictAction) {
      payload.conflict_action = conflictAction;
    }
    try {
      setStatus('保存中');
      const url = state.savedAxisId ? `/api/shaft/axes/${state.savedAxisId}` : '/api/shaft/axes';
      const method = state.savedAxisId ? 'PUT' : 'POST';
      const saved = await shaftRequest(url, { method, body: JSON.stringify(payload) }, { authRequired: true });
      state.savedAxisId = saved.id;
      state.savedAxisTitle = String(saved.title || payload.title || '');
      state.axis = saved.axis;
      ensureAxisShape();
      acceptSimulationResult(saved.result);
      markAxisDocumentClean();
      renderAll();
      persistAxisDraft();
      setStatus('已保存');
      showToast('排轴保存成功');
      await loadMyAxes();
      return true;
    } catch (error) {
      if (error.payload?.code === 'axis_name_conflict') {
        const resolution = await resolveAxisNameConflict(error.payload);
        if (resolution.action === 'overwrite') {
          const savedAfterOverwrite = await saveAxis('overwrite');
          return savedAfterOverwrite;
        }
        if (resolution.action === 'rename' && resolution.title) {
          $('shaft-title-input').value = resolution.title;
          persistAxisDraft();
          return saveAxis();
        }
        setStatus('已取消保存');
        return false;
      }
      setStatus(error.message, 'error');
      return false;
    }
  }

  function sharedCopyTitle() {
    const now = new Date();
    const stamp = [
      now.getFullYear(),
      String(now.getMonth() + 1).padStart(2, '0'),
      String(now.getDate()).padStart(2, '0'),
      '-',
      String(now.getHours()).padStart(2, '0'),
      String(now.getMinutes()).padStart(2, '0'),
      String(now.getSeconds()).padStart(2, '0'),
    ].join('');
    const suffix = `-副本-${stamp}`;
    const title = String($('shaft-title-input').value || '未命名排轴').trim() || '未命名排轴';
    return `${title.slice(0, Math.max(0, 80 - suffix.length))}${suffix}`;
  }

  async function saveAxisAsCopy() {
    const previousAxisId = state.savedAxisId;
    const previousSavedAxisTitle = state.savedAxisTitle;
    const previousTitle = $('shaft-title-input').value;
    state.savedAxisId = null;
    state.savedAxisTitle = '';
    $('shaft-title-input').value = sharedCopyTitle();
    persistAxisDraft();
    const saved = await saveAxis();
    if (!saved) {
      state.savedAxisId = previousAxisId;
      state.savedAxisTitle = previousSavedAxisTitle;
      $('shaft-title-input').value = previousTitle;
      persistAxisDraft();
    }
    return saved;
  }

  async function saveAxisAsNamedCopy() {
    if (!canSaveAxisAs()) {
      showToast('请先填写新的轴名，再点击另存', 'warning');
      setStatus('另存需要使用新的轴名', 'error');
      const titleInput = $('shaft-title-input');
      titleInput.focus();
      titleInput.select();
      return false;
    }
    const previousAxisId = state.savedAxisId;
    const previousSavedAxisTitle = state.savedAxisTitle;
    state.savedAxisId = null;
    state.savedAxisTitle = '';
    const saved = await saveAxis();
    if (!saved) {
      state.savedAxisId = previousAxisId;
      state.savedAxisTitle = previousSavedAxisTitle;
      renderSaveActions();
      persistAxisDraft();
    }
    return saved;
  }

  function renderMarketSearchLoading() {
    const list = $('shaft-market-list');
    list.setAttribute('aria-busy', 'true');
    list.innerHTML = '<div class="shaft-empty shaft-search-loading">搜索中</div>';
    $('shaft-market-more-btn').hidden = true;
  }

  async function loadMarket(reset) {
    window.clearTimeout(state.marketSearchTimer);
    if (reset) {
      state.marketPage = 1;
      state.marketItems = [];
    }
    if (state.marketQuery) {
      const retryAfter = MARKET_SEARCH_INTERVAL_MS - (Date.now() - state.marketLastSearchAt);
      if (retryAfter > 0) {
        renderMarketSearchLoading();
        state.marketSearchTimer = window.setTimeout(() => loadMarket(true), retryAfter);
        return;
      }
      state.marketLastSearchAt = Date.now();
      renderMarketSearchLoading();
    }
    const params = new URLSearchParams({
      sort: $('shaft-market-sort').value || 'dps',
      page: String(state.marketPage),
      page_size: '12',
    });
    if (state.marketQuery) {
      params.set('q', state.marketQuery);
    }
    state.marketCharacterIds.forEach((characterId) => params.append('character_id', characterId));
    const requestId = state.marketRequestId + 1;
    state.marketRequestId = requestId;
    try {
      const payload = await shaftRequest(`/api/shaft/market?${params.toString()}`);
      if (requestId !== state.marketRequestId) {
        return;
      }
      state.marketItems = reset ? payload.items : state.marketItems.concat(payload.items || []);
      state.marketHasMore = Boolean(payload.has_more);
      if (state.marketHasMore) {
        state.marketPage = Number(payload.page || state.marketPage) + 1;
      }
      renderMarketList();
    } catch (error) {
      if (requestId !== state.marketRequestId) {
        return;
      }
      if (error.status === 429 && error.payload?.code === 'shaft_market_search_rate_limited') {
        const retryAfter = Math.max(1, Number(error.payload.retry_after_ms || MARKET_SEARCH_INTERVAL_MS));
        state.marketLastSearchAt = Date.now() - (MARKET_SEARCH_INTERVAL_MS - retryAfter);
        renderMarketSearchLoading();
        state.marketSearchTimer = window.setTimeout(() => loadMarket(true), retryAfter);
        return;
      }
      $('shaft-market-list').innerHTML = `<div class="shaft-empty">${escapeHtml(error.message)}</div>`;
    } finally {
      if (requestId === state.marketRequestId && !$('shaft-market-list').querySelector('.shaft-search-loading')) {
        $('shaft-market-list').removeAttribute('aria-busy');
      }
    }
  }

  async function loadMyAxes() {
    if (!getToken()) {
      $('shaft-my-axis-total').textContent = '0';
      $('shaft-my-axis-list').innerHTML = '<div class="shaft-empty">登录后显示</div>';
      return;
    }
    try {
      const favorites = state.myAxisFilter === 'favorites';
      const params = new URLSearchParams({ sort: state.myAxisSort });
      if (state.myAxisQuery) {
        params.set('q', state.myAxisQuery);
      }
      state.myAxisCharacterIds.forEach((characterId) => params.append('character_id', characterId));
      const endpoint = favorites ? '/api/shaft/me/favorites' : '/api/shaft/me/axes';
      const payload = await shaftRequest(`${endpoint}?${params.toString()}`);
      $('shaft-my-axis-total').textContent = String(Math.max(0, Number(payload.total || 0)));
      $('shaft-my-axis-list').innerHTML = (payload.items || [])
        .map((axis) => marketCardHtml(axis, true, favorites ? 'favorites' : 'mine'))
        .join('') || `<div class="shaft-empty">${favorites ? '暂无收藏排轴' : '暂无保存排轴'}</div>`;
    } catch (error) {
      if (String(error.message || '').includes('未登录') || String(error.message || '').includes('token')) {
        clearToken();
        renderLoginState();
        $('shaft-my-axis-total').textContent = '0';
        $('shaft-my-axis-list').innerHTML = '<div class="shaft-empty">登录后显示</div>';
        return;
      }
      $('shaft-my-axis-list').innerHTML = `<div class="shaft-empty">${escapeHtml(error.message)}</div>`;
    }
  }

  async function loadAxis(axisId, source) {
    try {
      setStatus('读取排轴');
      const payload = await shaftRequest(`/api/shaft/axes/${axisId}`);
      state.axis = payload.axis;
      state.compareSnapshot = null;
      state.savedAxisId = source === 'mine' ? payload.id : null;
      state.savedAxisTitle = source === 'mine' ? String(payload.title || '') : '';
      state.selectedStepId = '';
      state.undoStack = [];
      state.redoStack = [];
      $('shaft-title-input').value = payload.title || '未命名排轴';
      $('shaft-description-input').value = payload.description || '';
      ensureAxisShape();
      state.result = null;
      markSimulationStale();
      renderAll();
      setPage('rotation');
      await runSimulation();
      markAxisDocumentClean();
      persistAxisDraft();
      setStatus('已读取');
    } catch (error) {
      setStatus(error.message, 'error');
    }
  }

  async function shareAxis(axisId, button) {
    if (!getToken()) {
      persistAxisDraft();
      redirectToLogin();
      return;
    }
    const originalLabel = button?.textContent || '分享';
    if (button) {
      button.disabled = true;
      button.textContent = '生成中';
    }
    try {
      const payload = await shaftRequest(
        `/api/shaft/axes/${axisId}/share`,
        { method: 'POST' },
        { authRequired: true },
      );
      const shareUrl = new URL(payload.share_path, window.location.origin).href;
      await copyTextToClipboard(shareUrl);
      showToast('分享链接已复制到剪切板');
      setStatus('分享链接已复制');
    } catch (error) {
      setStatus(error.message, 'error');
    } finally {
      if (button?.isConnected) {
        button.disabled = false;
        button.textContent = originalLabel;
      }
    }
  }

  function sharedAxisTokenFromUrl() {
    return String(new URLSearchParams(window.location.search).get('share') || '').trim();
  }

  function clearSharedAxisUrl() {
    window.history.replaceState({}, '', '/shaft/rotation');
  }

  async function resolveSharedAxisWorkspace() {
    const dialog = $('shaft-share-open-dialog');
    dialog.returnValue = '';
    dialog.showModal();
    return new Promise((resolve) => {
      dialog.addEventListener('close', () => resolve(dialog.returnValue || 'cancel'), { once: true });
    });
  }

  async function replaceWorkspaceWithSharedAxis(payload) {
    state.sharedReadOnly = Boolean(payload.read_only);
    if (state.sharedReadOnly && getToken()) {
      clearToken();
      renderLoginState();
    }
    state.axis = payload.axis;
    state.compareSnapshot = null;
    state.savedAxisId = null;
    state.savedAxisTitle = '';
    state.selectedStepId = '';
    state.undoStack = [];
    state.redoStack = [];
    $('shaft-title-input').value = payload.title || '未命名排轴';
    $('shaft-description-input').value = payload.description || '';
    ensureAxisShape();
    state.result = null;
    markSimulationStale();
    renderAll();
    setPage('rotation');
    await runSimulation();
    markAxisDocumentClean();
    persistAxisDraft();
    setStatus(state.sharedReadOnly ? '分享排轴 · 只读' : '已打开分享排轴');
    showToast(`已打开「${payload.title || '未命名排轴'}」${state.sharedReadOnly ? '（只读）' : ''}`);
  }

  async function openSharedAxisFromUrl() {
    const shareToken = sharedAxisTokenFromUrl();
    if (!shareToken) {
      return;
    }
    try {
      setStatus('读取分享排轴');
      const payload = await shaftRequest(`/api/shaft/shared/${encodeURIComponent(shareToken)}`);
      if (hasUnsavedAxisChanges()) {
        const action = await resolveSharedAxisWorkspace();
        if (action === 'cancel') {
          clearSharedAxisUrl();
          setStatus('未打开分享排轴');
          return;
        }
        if (action === 'save' && !await saveAxis()) {
          return;
        }
        if (action === 'save_as' && !await saveAxisAsCopy()) {
          return;
        }
      }
      await replaceWorkspaceWithSharedAxis(payload);
    } catch (error) {
      setStatus(error.message, 'error');
    }
  }

  async function deleteAxis(axisId, title) {
    if (!getToken()) {
      persistAxisDraft();
      redirectToLogin();
      return;
    }
    const label = title ? `「${title}」` : `#${axisId}`;
    if (!window.confirm(`确定删除排轴 ${label} 吗？删除后无法恢复。`)) {
      return;
    }
    try {
      setStatus('删除中');
      await shaftRequest(`/api/shaft/axes/${axisId}`, { method: 'DELETE' }, { authRequired: true });
      if (Number(state.savedAxisId || 0) === Number(axisId)) {
        state.savedAxisId = null;
        state.savedAxisTitle = '';
        renderSaveActions();
        persistAxisDraft();
      }
      await loadMyAxes();
      await loadMarket(true);
      setStatus('已删除');
    } catch (error) {
      setStatus(error.message, 'error');
    }
  }

  async function backupAxis(axisId, button) {
    if (!getToken()) {
      persistAxisDraft();
      redirectToLogin();
      return;
    }
    const originalLabel = button?.textContent || '备份';
    if (button) {
      button.disabled = true;
      button.textContent = '备份中';
    }
    try {
      setStatus('备份中');
      const backup = await shaftRequest(
        `/api/shaft/axes/${axisId}/backup`,
        { method: 'POST' },
        { authRequired: true },
      );
      showToast(`已创建「${backup.title}」`);
      await loadMyAxes();
      await loadAxis(Number(backup.id), 'mine');
    } catch (error) {
      setStatus(error.message, 'error');
    } finally {
      if (button?.isConnected) {
        button.disabled = false;
        button.textContent = originalLabel;
      }
    }
  }

  async function publishAxis(axisId, button) {
    if (!getToken()) {
      persistAxisDraft();
      redirectToLogin();
      return;
    }
    const updating = button?.dataset.publishMode === 'update';
    const originalLabel = button?.textContent || (updating ? '更新' : '上传');
    if (button) {
      button.disabled = true;
      button.textContent = updating ? '更新中' : '上传中';
    }
    try {
      setStatus(updating ? '更新公开快照中' : '上传快照中');
      const snapshot = await shaftRequest(
        `/api/shaft/axes/${axisId}/publish`,
        { method: 'POST' },
        { authRequired: true },
      );
      setStatus(updating ? '已更新' : '已上传');
      showToast(`「${snapshot.title}」的公开快照已${updating ? '更新' : '上传'}`);
      await loadMarket(true);
      await loadMyAxes();
    } catch (error) {
      setStatus(error.message, 'error');
    } finally {
      if (button?.isConnected) {
        button.disabled = false;
        button.textContent = originalLabel;
      }
    }
  }

  async function unpublishAxis(axisId, title, button) {
    if (!getToken()) {
      persistAxisDraft();
      redirectToLogin();
      return;
    }
    const label = title ? `「${title}」` : `#${axisId}`;
    if (!window.confirm(`确定下架 ${label} 吗？公开快照将从广场移除，“我的排轴”中的原轴会保留。`)) {
      return;
    }
    const originalLabel = button?.textContent || '下架';
    if (button) {
      button.disabled = true;
      button.textContent = '下架中';
    }
    try {
      setStatus('下架中');
      await shaftRequest(`/api/shaft/axes/${axisId}`, { method: 'DELETE' }, { authRequired: true });
      await loadMarket(true);
      await loadMyAxes();
      setStatus('已下架');
      showToast(`${label}已从广场下架，私有原轴仍保留`);
    } catch (error) {
      setStatus(error.message, 'error');
      if (button?.isConnected) {
        button.disabled = false;
        button.textContent = originalLabel;
      }
    }
  }

  async function toggleLike(axisId, active) {
    try {
      await shaftRequest(`/api/shaft/axes/${axisId}/like`, { method: active ? 'DELETE' : 'POST' }, { authRequired: true });
      await loadMarket(true);
    } catch (error) {
      setStatus(error.message, 'error');
    }
  }

  async function toggleDislike(axisId, active) {
    try {
      await shaftRequest(`/api/shaft/axes/${axisId}/dislike`, { method: active ? 'DELETE' : 'POST' }, { authRequired: true });
      await loadMarket(true);
    } catch (error) {
      setStatus(error.message, 'error');
    }
  }

  async function toggleFavorite(axisId, active) {
    try {
      await shaftRequest(`/api/shaft/axes/${axisId}/favorite`, { method: active ? 'DELETE' : 'POST' });
      await loadMarket(true);
      if (state.myAxisFilter === 'favorites') {
        await loadMyAxes();
      }
    } catch (error) {
      setStatus(error.message, 'error');
    }
  }

  function setCharacterFilterOpen(open) {
    $('shaft-character-filter-popover').hidden = !open;
    $('shaft-character-filter-trigger').setAttribute('aria-expanded', open ? 'true' : 'false');
  }

  function setMyCharacterFilterOpen(open) {
    $('shaft-my-character-filter-popover').hidden = !open;
    $('shaft-my-character-filter-trigger').setAttribute('aria-expanded', open ? 'true' : 'false');
  }

  function toggleCharacterFilter(selectedIds, characterId, onChange) {
    const index = selectedIds.indexOf(characterId);
    if (index >= 0) {
      selectedIds.splice(index, 1);
    } else if (selectedIds.length >= 4) {
      showToast('最多选择 4 名角色', 'warning');
      return;
    } else {
      selectedIds.push(characterId);
    }
    renderMarketFilters();
    onChange();
  }

  function toggleMarketCharacter(characterId) {
    toggleCharacterFilter(state.marketCharacterIds, characterId, () => loadMarket(true));
  }

  function toggleMyAxisCharacter(characterId) {
    toggleCharacterFilter(state.myAxisCharacterIds, characterId, loadMyAxes);
  }

  function setMyAxisFilter(filter) {
    state.myAxisFilter = filter === 'favorites' ? 'favorites' : 'mine';
    $('shaft-my-axis-scope').value = state.myAxisFilter;
    loadMyAxes();
  }

  async function handleMarketClick(event) {
    const shareButton = event.target.closest('[data-share-axis]');
    if (shareButton) {
      event.stopPropagation();
      await shareAxis(Number(shareButton.dataset.shareAxis), shareButton);
      return;
    }
    const publishButton = event.target.closest('[data-publish-axis]');
    if (publishButton) {
      event.stopPropagation();
      await publishAxis(Number(publishButton.dataset.publishAxis), publishButton);
      return;
    }
    const unpublishButton = event.target.closest('[data-unpublish-axis]');
    if (unpublishButton) {
      event.stopPropagation();
      await unpublishAxis(
        Number(unpublishButton.dataset.unpublishAxis),
        unpublishButton.dataset.axisTitle || '',
        unpublishButton,
      );
      return;
    }
    const backupButton = event.target.closest('[data-backup-axis]');
    if (backupButton) {
      event.stopPropagation();
      await backupAxis(Number(backupButton.dataset.backupAxis), backupButton);
      return;
    }
    const deleteButton = event.target.closest('[data-delete-axis]');
    if (deleteButton) {
      event.stopPropagation();
      await deleteAxis(Number(deleteButton.dataset.deleteAxis), deleteButton.dataset.axisTitle || '');
      return;
    }
    const likeButton = event.target.closest('[data-like-axis]');
    if (likeButton) {
      event.stopPropagation();
      if (!getToken()) {
        persistAxisDraft();
        redirectToLogin();
        return;
      }
      await toggleLike(Number(likeButton.dataset.likeAxis), likeButton.classList.contains('active'));
      return;
    }
    const dislikeButton = event.target.closest('[data-dislike-axis]');
    if (dislikeButton) {
      event.stopPropagation();
      if (!getToken()) {
        persistAxisDraft();
        redirectToLogin();
        return;
      }
      await toggleDislike(Number(dislikeButton.dataset.dislikeAxis), dislikeButton.classList.contains('active'));
      return;
    }
    const favoriteButton = event.target.closest('[data-favorite-axis]');
    if (favoriteButton) {
      event.stopPropagation();
      await toggleFavorite(Number(favoriteButton.dataset.favoriteAxis), favoriteButton.classList.contains('active'));
      return;
    }
    const card = event.target.closest('[data-axis-id]');
    if (card) {
      if (card.dataset.axisSource === 'market') {
        await openMarketAxisPreview(Number(card.dataset.axisId), card);
      } else {
        await loadAxis(Number(card.dataset.axisId), card.dataset.axisSource);
      }
    }
  }

  function bindEvents() {
    document.querySelectorAll('[data-shaft-page-link]').forEach((link) => {
      link.addEventListener('click', (event) => {
        event.preventDefault();
        setPage(link.dataset.shaftPageLink);
      });
    });
    $('shaft-shortcut-help-btn').addEventListener('click', openShortcutHelp);
    $('shaft-shortcut-dialog').addEventListener('click', (event) => {
      if (event.target === event.currentTarget || event.target.closest('[data-close-shortcut-help]')) {
        closeShortcutHelp();
      }
    });
    $('shaft-build-info-btn').addEventListener('click', openBuildInfo);
    $('shaft-build-info-dialog').addEventListener('click', (event) => {
      if (event.target === event.currentTarget || event.target.closest('[data-close-build-info]')) {
        closeBuildInfo();
      }
    });
    $('shaft-new-btn').addEventListener('click', newAxis);
    $('shaft-save-btn').addEventListener('click', () => saveAxis());
    $('shaft-save-as-btn').addEventListener('click', saveAxisAsNamedCopy);
    $('shaft-title-input').addEventListener('input', () => {
      persistAxisDraft();
      renderSaveActions();
    });
    $('shaft-description-input').addEventListener('input', persistAxisDraft);
    $('shaft-undo-btn').addEventListener('click', undoLastEdit);
    $('shaft-redo-btn').addEventListener('click', redoLastEdit);
    $('shaft-copy-step-btn').addEventListener('click', copySelectedSteps);
    $('shaft-paste-step-btn').addEventListener('click', pasteStepsAtCursor);
    $('shaft-delete-step-btn').addEventListener('click', removeSelectedSteps);
    $('shaft-preview-btn').addEventListener('click', (event) => openAxisPreview(event.currentTarget));
    $('shaft-axis-preview-fit-btn').addEventListener('click', fitAxisPreview);
    $('shaft-axis-preview-cancel-btn').addEventListener('click', closeAxisPreview);
    $('shaft-axis-preview-save-btn').addEventListener('click', saveMarketAxisToLocal);
    $('shaft-axis-preview-dialog').addEventListener('click', (event) => {
      if (event.target === event.currentTarget || event.target.closest('[data-close-axis-preview]')) {
        closeAxisPreview();
      }
    });
    $('shaft-background-multiplier-dialog').addEventListener('click', (event) => {
      if (event.target === event.currentTarget || event.target.closest('[data-close-background-multiplier]')) {
        closeBackgroundActionMultiplier();
      }
    });
    $('shaft-background-multiplier-confirm').addEventListener('click', confirmBackgroundActionMultiplier);
    $('shaft-background-multiplier-input').addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        confirmBackgroundActionMultiplier();
      }
    });
    $('shaft-loop-settings-btn').addEventListener('click', (event) => openLoopSettings(event.currentTarget));
    $('shaft-loop-settings-dialog').addEventListener('click', (event) => {
      if (event.target === event.currentTarget || event.target.closest('[data-close-loop-settings]')) {
        closeLoopSettings();
      }
    });
    $('shaft-loop-enabled').addEventListener('change', syncLoopResourceInputsDisabled);
    $('shaft-loop-settings-confirm').addEventListener('click', confirmLoopSettings);
    $('shaft-axis-preview-viewport').addEventListener('wheel', handleAxisPreviewWheel, { passive: false });
    $('shaft-library-slot').addEventListener('change', (event) => {
      state.librarySlot = Number(event.target.value || 0);
      renderTeamDock();
      renderActionAdder();
      renderCommandLibrary();
    });
    $('shaft-command-search').addEventListener('input', (event) => {
      state.commandSearch = event.target.value || '';
      renderCommandLibrary();
    });
    $('shaft-command-type-tabs').addEventListener('click', (event) => {
      const button = event.target.closest('[data-command-type]');
      if (!button) {
        return;
      }
      state.commandTypeFilter = button.dataset.commandType || '';
      renderCommandTypes();
      renderCommandLibrary();
    });
    $('shaft-team-dock').addEventListener('click', handleTeamDockClick);
    $('shaft-command-list').addEventListener('click', handleLibraryClick);
    $('shaft-market-refresh-btn').addEventListener('click', () => loadMarket(true));
    $('shaft-market-more-btn').addEventListener('click', () => loadMarket(false));
    $('shaft-my-refresh-btn').addEventListener('click', loadMyAxes);
    $('shaft-character-filter-trigger').addEventListener('click', () => {
      setCharacterFilterOpen($('shaft-character-filter-popover').hidden);
    });
    $('shaft-my-character-filter-trigger').addEventListener('click', () => {
      setMyCharacterFilterOpen($('shaft-my-character-filter-popover').hidden);
    });
    $('shaft-character-filter-grid').addEventListener('click', (event) => {
      const button = event.target.closest('[data-market-character-id]');
      if (button) {
        event.stopPropagation();
        toggleMarketCharacter(button.dataset.marketCharacterId);
      }
    });
    $('shaft-character-filter-clear').addEventListener('click', () => {
      state.marketCharacterIds = [];
      renderMarketFilters();
      loadMarket(true);
    });
    $('shaft-my-character-filter-grid').addEventListener('click', (event) => {
      const button = event.target.closest('[data-my-character-id]');
      if (button) {
        event.stopPropagation();
        toggleMyAxisCharacter(button.dataset.myCharacterId);
      }
    });
    $('shaft-my-character-filter-clear').addEventListener('click', () => {
      state.myAxisCharacterIds = [];
      renderMarketFilters();
      loadMyAxes();
    });
    $('shaft-my-axis-scope').addEventListener('change', (event) => {
      setMyAxisFilter(event.target.value);
    });
    $('shaft-my-axis-sort').addEventListener('change', (event) => {
      state.myAxisSort = event.target.value || 'new';
      loadMyAxes();
    });
    $('shaft-my-axis-query').addEventListener('input', (event) => {
      state.myAxisQuery = String(event.target.value || '').trim();
      window.clearTimeout(state.myAxisSearchTimer);
      state.myAxisSearchTimer = window.setTimeout(loadMyAxes, 180);
    });
    $('shaft-market-query').addEventListener('input', (event) => {
      state.marketQuery = String(event.target.value || '').trim();
      window.clearTimeout(state.marketSearchTimer);
      state.marketRequestId += 1;
      state.marketSearchTimer = window.setTimeout(
        () => loadMarket(true),
        MARKET_SEARCH_EDIT_DELAY_MS,
      );
    });
    $('shaft-market-sort').addEventListener('change', () => loadMarket(true));
    const addBuffButton = $('shaft-add-buff-btn');
    if (addBuffButton) {
      addBuffButton.addEventListener('click', addBuffRule);
    }
    const buffTriggerSlot = $('shaft-buff-trigger-slot');
    if (buffTriggerSlot) {
      buffTriggerSlot.addEventListener('change', (event) => {
        state.buffDraft.triggerSlot = Number(event.target.value || 0);
        renderBuffEditor();
      });
    }
    const buffModifierKey = $('shaft-buff-modifier-key');
    if (buffModifierKey) {
      buffModifierKey.addEventListener('change', (event) => {
        state.buffDraft.modifierKey = event.target.value;
      });
    }
    $('shaft-enemy-level').addEventListener('input', (event) => {
      state.axis.enemy.level = Number(event.target.value || 90);
      scheduleSimulation();
    });
    $('shaft-enemy-track-outside').addEventListener('change', (event) => {
      state.axis.enemy.track_outside = event.target.checked;
      scheduleSimulation();
    });
    $('shaft-team-bonus-grid').addEventListener('input', handleTeamBonusInput);
    $('shaft-team-bonus-grid').addEventListener('change', (event) => {
      handleTeamBonusInput(event);
      renderTeamBonus();
    });
    $('shaft-weakness-row').addEventListener('change', () => {
      state.axis.enemy.weakness_elements = Array.from($('shaft-weakness-row').querySelectorAll('input:checked')).map((input) => input.value);
      scheduleSimulation();
    });
    $('shaft-initial-resistance-input').addEventListener('input', (event) => {
      const resistance = Math.max(-1, Math.min(1, Number(event.target.value || 0) / 100));
      state.axis.enemy.initial_resistance = resistance;
      state.axis.enemy.resistances = Object.fromEntries(
        RESISTANCE_ELEMENTS.map((element) => [element, resistance]),
      );
      scheduleSimulation();
    });
    $('shaft-compare-controls').addEventListener('click', handleCompareControls);
    $('shaft-compare-controls').addEventListener('change', handleCompareControlChange);
    $('shaft-contribution-list').addEventListener('click', handleContributionClick);
    $('shaft-stagger-analysis-dialog').addEventListener('click', (event) => {
      if (event.target === event.currentTarget || event.target.closest('[data-close-stagger-analysis]')) {
        closeStaggerAnalysis();
      }
    });
    $('shaft-harmony-analysis-dialog').addEventListener('click', (event) => {
      if (event.target === event.currentTarget || event.target.closest('[data-close-harmony-analysis]')) {
        closeHarmonyAnalysis();
      }
    });
    $('shaft-action-contribution-dialog').addEventListener('click', (event) => {
      if (event.target === event.currentTarget || event.target.closest('[data-close-action-contribution]')) {
        closeActionContribution();
        return;
      }
      if (event.target.closest('[data-save-action-snapshot]')) {
        saveCompareSnapshot();
        return;
      }
      const dimensionButton = event.target.closest('[data-analysis-dimension]');
      if (dimensionButton) {
        actionAnalysisState.dimension = dimensionButton.dataset.analysisDimension === 'type' ? 'type' : 'action';
        renderActionContributionAnalysis();
        return;
      }
      const viewButton = event.target.closest('[data-analysis-view]');
      if (viewButton) {
        actionAnalysisState.view = viewButton.dataset.analysisView === 'bars' ? 'bars' : 'donut';
        renderActionContributionAnalysis();
      }
    });
    $('shaft-action-contribution-content').addEventListener('pointerover', (event) => {
      const target = event.target.closest('[data-analysis-index]');
      if (target) setActionAnalysisHighlight(target.dataset.analysisIndex, true);
    });
    $('shaft-action-contribution-content').addEventListener('pointerout', (event) => {
      const target = event.target.closest('[data-analysis-index]');
      if (target && !event.relatedTarget?.closest?.(`[data-analysis-index="${target.dataset.analysisIndex}"]`)) {
        setActionAnalysisHighlight(target.dataset.analysisIndex, false);
      }
    });
    $('shaft-action-contribution-content').addEventListener('focusin', (event) => {
      const target = event.target.closest('[data-analysis-index]');
      if (target) setActionAnalysisHighlight(target.dataset.analysisIndex, true);
    });
    $('shaft-action-contribution-content').addEventListener('focusout', (event) => {
      const target = event.target.closest('[data-analysis-index]');
      if (target) setActionAnalysisHighlight(target.dataset.analysisIndex, false);
    });
    $('shaft-team-slots').addEventListener('change', handleTeamChange);
    $('shaft-team-slots').addEventListener('click', handleTeamClick);
    $('shaft-team-slots').addEventListener('change', handleCurtainInput);
    $('shaft-team-slots').addEventListener('focusout', handleSubstatCommit);
    $('shaft-team-slots').addEventListener('input', handleSubstatInput);
    $('shaft-team-slots').addEventListener('keydown', handleSubstatKeydown);
    $('shaft-team-slots').addEventListener('input', handleSkillLevelInput);
    $('shaft-team-slots').addEventListener('input', handleCurtainInput);
    const buildPanel = $('shaft-build-panel');
    if (buildPanel) {
      buildPanel.addEventListener('click', handleBuildPanelClick);
    }
    if ($('shaft-step-list')) {
      $('shaft-step-list').addEventListener('change', handleStepChange);
      $('shaft-step-list').addEventListener('input', handleStepChange);
      $('shaft-step-list').addEventListener('click', handleStepClick);
    }
    $('shaft-step-detail').addEventListener('click', handleStepDetailClick);
    $('shaft-timeline').addEventListener('click', handleTimelineClick);
    $('shaft-timeline').addEventListener('contextmenu', handleTimelineContextMenu);
    $('shaft-timeline').addEventListener('mousedown', handleTimelineMouseDown);
    $('shaft-timeline').addEventListener('pointermove', positionBuffLineTooltip);
    $('shaft-timeline').addEventListener('pointermove', positionDamageMarkerTooltip);
    $('shaft-timeline').addEventListener('focusin', positionDamageMarkerTooltip);
    $('shaft-context-menu').addEventListener('click', handleContextMenuClick);
    document.addEventListener('click', (event) => {
      if (!event.target.closest('#shaft-context-menu')) {
        closeContextMenu();
      }
      if (!event.target.closest('#shaft-market-character-filter')) {
        setCharacterFilterOpen(false);
      }
      if (!event.target.closest('#shaft-my-character-filter')) {
        setMyCharacterFilterOpen(false);
      }
      if (!event.target.closest('[data-build-character-picker]')) {
        setBuildCharacterPickerOpen();
      }
    });
    document.addEventListener('keydown', handleKeydown);
    document.addEventListener('copy', handleClipboardCopy);
    document.addEventListener('paste', handleClipboardPaste);
    window.addEventListener('mousemove', handleTimelineMouseMove);
    window.addEventListener('mouseup', handleTimelineMouseUp);
    const buffList = $('shaft-buff-list');
    if (buffList) {
      buffList.addEventListener('click', handleBuffClick);
    }
    $('shaft-market-list').addEventListener('click', handleMarketClick);
    $('shaft-my-axis-list').addEventListener('click', handleMarketClick);
  }

  async function init() {
    const active = document.querySelector('.shaft-page')?.dataset.activePage || 'rotation';
    state.page = active;
    window.NTE_BEFORE_LOGIN_REDIRECT = persistAxisDraft;
    bindEvents();
    focusTimelineForShortcuts();
    try {
      state.catalog = await shaftRequest('/api/shaft/catalog');
      const restoredDraft = restoreAxisDraft(active);
      if (!restoredDraft) {
        state.axis = emptyAxisFromCatalog();
      }
      ensureAxisShape();
      if (!restoredDraft) {
        markAxisDocumentClean();
      }
      const versionLabel = displayText(state.catalog.source_meta.version_label || '');
      $('shaft-data-version').textContent = versionLabel;
      $('shaft-build-info-version').textContent = versionLabel;
      renderAll();
      await runSimulation();
      await openSharedAxisFromUrl();
      await loadMarket(true);
      await loadMyAxes();
    } catch (error) {
      setStatus(error.message, 'error');
    }
  }

  document.addEventListener('DOMContentLoaded', init);
}());
