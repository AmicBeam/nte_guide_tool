(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.ShaftEngine = factory();
  }
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const ELEMENTS = ['光', '灵', '咒', '暗', '魂', '相'];
  const ZERO_ACTION_VISUAL_TICKS = 5;
  const MIN_FOREGROUND_START_GAP_TICKS = 2;
  const MAX_BACKGROUND_ACTION_MULTIPLIER = 999;
  const ENEMY_DEBUFF_DURATIONS = {
    '延滞': 50,
    '黯星': 50,
    '浸染': 120,
    '覆纹': 120,
    '浊燃': 150,
  };
  const TEAM_PANEL_BONUS_DEFAULTS = {
    version: 3,
    furniture_crit_dmg: 0.04,
    furniture_flat_atk: 20,
    furniture_flat_def: 30,
    small_flat_atk: 420,
    small_flat_hp: 5200,
  };
  const DEFAULT_PERSONAL_RESOURCE_CAPS = {
    char_31c5130304: { '真理之匙': 3 },
    char_701295143d: { '言灵字': 4 },
    char_912dbfe17c: { '闪送之力': 6 },
    char_d38b672525: { '罪状': 1000 },
    char_a01c39f576: { '臆想': 100 },
  };
  const SKILL_LEVEL_DEFAULTS = {
    basic: 10,
    skill: 10,
    ultimate: 10,
    support: 10,
  };
  const SUBSTAT_EFFECT_KEYS = {
    all_dmg: 'all_dmg',
    crit_rate: 'crit_rate',
    crit_dmg: 'crit_dmg',
    harmony_strength: 'harmony_strength',
    stagger_strength: 'stagger_strength',
    atk_pct: 'atk_pct',
    flat_atk: 'flat_atk',
    hp_pct: 'hp_pct',
    flat_hp: 'flat_hp',
    def_pct: 'def_pct',
    flat_def: 'flat_def',
  };
  const CURTAIN_PASSIVE_TYPES = ['type2', 'type3', 'type4'];
  const SUPPORTED_TRIGGER_EVENTS = new Set(['passive', 'action_start', 'action_hit', 'action_end', 'foreground_enter', 'foreground_leave', 'loop_start', 'reaction_trigger', 'periodic_damage', 'full_stack']);
  const PERMANENT_BUFF_END_TICK = 1000000000;
  const HARMONY_DAMAGE_SOURCES = ['创生', '创生复制体', '覆纹', '浊燃', '黯星'];
  const SPECIAL_DAMAGE_SOURCES = ['创生', '创生复制体', '覆纹', '浊燃', '黯星'];
  const DAMAGE_SHARE_SOURCE_GROUPS = [
    { source: '创生', members: ['创生', '创生复制体'] },
    { source: '覆纹', members: ['覆纹'] },
    { source: '浊燃', members: ['浊燃'] },
    { source: '黯星', members: ['黯星'] },
  ];
  const REACTION_BY_ELEMENT_PAIR = new Map([
    ['光|灵', '创生'],
    ['光|相', '延滞'],
    ['灵|咒', '覆纹'],
    ['咒|暗', '浊燃'],
    ['暗|魂', '黯星'],
    ['魂|相', '浸染'],
  ]);
  const REACTION_DURATIONS = {
    '创生': 100,
    '延滞': 50,
    '覆纹': 120,
    '浊燃': 150,
    '黯星': 50,
    '浸染': 120,
  };
  const REACTION_BASE_DAMAGE = {
    5: { '创生': 80, '浊燃': 20, '黯星': 400 },
    10: { '创生': 120, '浊燃': 35, '黯星': 600 },
    15: { '创生': 200, '浊燃': 60, '黯星': 1000 },
    20: { '创生': 300, '浊燃': 90, '黯星': 1500 },
    25: { '创生': 400, '浊燃': 120, '黯星': 2000 },
    30: { '创生': 600, '浊燃': 180, '黯星': 3000 },
    35: { '创生': 800, '浊燃': 240, '黯星': 4000 },
    40: { '创生': 1000, '浊燃': 300, '黯星': 5000 },
    45: { '创生': 1700, '浊燃': 510, '黯星': 8500 },
    50: { '创生': 2200, '浊燃': 660, '黯星': 11000 },
    55: { '创生': 3600, '浊燃': 1080, '黯星': 18000 },
    60: { '创生': 5000, '浊燃': 1500, '黯星': 25000 },
    65: { '创生': 6000, '浊燃': 1800, '黯星': 30000 },
    70: { '创生': 7000, '浊燃': 2100, '黯星': 35000 },
    75: { '创生': 8000, '浊燃': 2400, '黯星': 40000 },
    80: { '创生': 9000, '浊燃': 2700, '黯星': 45000 },
  };
  const STAGGER_DAMAGE_BASE = {
    5: 101,
    10: 114,
    15: 127,
    20: 140,
    25: 193,
    30: 257,
    35: 345,
    40: 462,
    45: 621,
    50: 854,
    55: 1106,
    60: 1440,
    65: 1896,
    70: 2384,
    75: 2984,
    80: 3603,
  };
  const STAGGER_LIMIT = 50;
  const STAGGER_RECOVERY_SECONDS = 10;
  const LAST_ROSE_ARC_ID = 'arc_dcd5900afc';
  const LAST_ROSE_STAGGER_EXTENSION_SECONDS = 3;
  const ILOY_CHARACTER_ID = 'char_a01c39f576';
  const NANALI_CHARACTER_ID = 'char_bdc43f82c6';
  const JIUYUAN_CHARACTER_ID = 'char_b2e3b2bf7a';
  const ZHENHONG_CHARACTER_ID = 'char_b52cc8f160';
  const ZHENHONG_ASCENDANT_ENTRY_ACTION_ID = 'action_c32b4b9417';
  const ZHENHONG_ASCENDANT_EXIT_ACTION_ID = 'action_e3711f0cf5';
  const ZHENHONG_ASCENDANT_ENERGY_CAPACITY = 12;
  const CANHONG_CHARACTER_ID = 'char_076a1f4e53';
  const CANHONG_FENTIAN_ACTION_ID = 'action_canhong_q';
  const CANHONG_BLOOD_BANQUET_ACTION_ID = 'action_canhong_q_blood_banquet';
  const CANHONG_BLOOD_BANQUET_BUFF_ID = 'character_canhong_blood_banquet_active';
  const HARMONY_CAPACITY = 100;
  const TEAMMATE_ENERGY_SHARE_RATIO = 0.6;

  function clone(value) {
    return JSON.parse(JSON.stringify(value ?? null));
  }

  function num(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function int(value, fallback = 0) {
    return Math.round(num(value, fallback));
  }

  function awakeningNodes(source) {
    if (Array.isArray(source?.awakening_nodes)) {
      return new Set(source.awakening_nodes.map((value) => int(value)).filter((value) => value >= 1 && value <= 6));
    }
    return new Set(Array.from({ length: Math.max(0, Math.min(6, int(source?.awakening))) }, (_, index) => index + 1));
  }

  function hasAwakeningNode(source, level) {
    return awakeningNodes(source).has(int(level));
  }

  function ruleHasAwakeningNode(rule, level) {
    if (Array.isArray(rule?.owner_awakening_nodes)) {
      return rule.owner_awakening_nodes.map(int).includes(int(level));
    }
    return int(rule?.owner_awakening) >= int(level);
  }

  function durationTicksForRule(rule) {
    const duration = rule?.duration && typeof rule.duration === 'object' ? rule.duration : {};
    let ticks = Math.max(0, int(duration.ticks));
    Object.entries(duration.ticks_by_awakening_node || {}).forEach(([level, value]) => {
      if (ruleHasAwakeningNode(rule, level)) ticks = Math.max(0, int(value));
    });
    return ticks;
  }

  function periodicAtkMultiplier(periodic, rule) {
    let multiplier = num(periodic?.atk_multiplier);
    Object.entries(periodic?.awakening_node_multipliers || {}).forEach(([level, factor]) => {
      if (ruleHasAwakeningNode(rule, level)) multiplier *= num(factor, 1);
    });
    return multiplier;
  }

  function maxStacksForRule(rule) {
    const stacking = rule?.stacking && typeof rule.stacking === 'object' ? rule.stacking : {};
    let maxStacks = Math.max(1, num(stacking.max_stacks, 1));
    Object.entries(stacking.max_stacks_by_awakening_node || {}).forEach(([level, value]) => {
      if (ruleHasAwakeningNode(rule, level)) maxStacks = Math.max(1, num(value, maxStacks));
    });
    return maxStacks;
  }

  function actionValueForAwakeningNodes(action, snapshot, baseKey, nodeMapKey) {
    let value = num(action?.[baseKey]);
    Object.entries(action?.[nodeMapKey] || {}).forEach(([level, nodeValue]) => {
      if (hasAwakeningNode(snapshot, level)) value = num(nodeValue, value);
    });
    return value;
  }

  function asList(value) {
    return Array.isArray(value) ? value : [];
  }

  function strSet(value) {
    return new Set(asList(value).map((item) => String(item)).filter(Boolean));
  }

  function recordMap(records) {
    return new Map(asList(records).map((record) => [String(record?.id || ''), record]));
  }

  function mods() {
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
      res_down_光: 0,
      res_down_灵: 0,
      res_down_咒: 0,
      res_down_暗: 0,
      res_down_魂: 0,
      res_down_相: 0,
      res_down_心灵: 0,
      energy_recharge: 0,
      harmony_strength: 0,
      stagger_strength: 0,
      stagger_multiplier: 0,
      stagger_damage_bonus: 0,
      interrupt_resistance: 0,
      basic_dmg: 0,
      dodge_counter_dmg: 0,
      element_dmg: 0,
      follow_dmg: 0,
      mind_dmg: 0,
      attach_dmg: 0,
      skill_dmg: 0,
      ultimate_dmg: 0,
      all_dmg: 0,
      final_dmg: 0,
      base_multiplier_pct: 0,
    };
  }

  function mergeMods(base, delta, factor = 1) {
    if (!delta || typeof delta !== 'object') {
      return base;
    }
    Object.entries(delta).forEach(([key, value]) => {
      if (Object.prototype.hasOwnProperty.call(base, key)) {
        base[key] += num(value) * factor;
      }
    });
    return base;
  }

  function normalizeStatName(value) {
    const text = String(value || '').trim();
    return text === '环合强度' ? '精通' : text;
  }

  function defaultBuildOptions(characterId, constants) {
    const defaults = constants?.default_build_options && typeof constants.default_build_options === 'object'
      ? constants.default_build_options
      : {};
    return defaults[characterId] && typeof defaults[characterId] === 'object' ? defaults[characterId] : {};
  }

  function normalizeTeamPanelBonus(raw, catalog) {
    const starterBonus = catalog?.starter_axis?.team_panel_bonus || {};
    const defaults = Object.assign({}, TEAM_PANEL_BONUS_DEFAULTS, starterBonus);
    const source = raw && typeof raw === 'object' ? raw : {};
    const bonus = {};
    Object.keys(defaults).forEach((key) => {
      bonus[key] = num(source[key], num(defaults[key]));
    });
    const version = int(source.version);
    if (version < 2 && bonus.furniture_flat_atk === 20 && bonus.small_flat_atk === 440) {
      bonus.small_flat_atk = 420;
    }
    if (version < TEAM_PANEL_BONUS_DEFAULTS.version) {
      bonus.furniture_crit_dmg = TEAM_PANEL_BONUS_DEFAULTS.furniture_crit_dmg;
      bonus.furniture_flat_atk = TEAM_PANEL_BONUS_DEFAULTS.furniture_flat_atk;
      bonus.furniture_flat_def = TEAM_PANEL_BONUS_DEFAULTS.furniture_flat_def;
    } else {
      bonus.furniture_crit_dmg = Math.round(Math.max(0, Math.min(TEAM_PANEL_BONUS_DEFAULTS.furniture_crit_dmg, bonus.furniture_crit_dmg)) * 1000) / 1000;
      bonus.furniture_flat_atk = Math.round(Math.max(0, Math.min(TEAM_PANEL_BONUS_DEFAULTS.furniture_flat_atk, bonus.furniture_flat_atk)));
      bonus.furniture_flat_def = Math.round(Math.max(0, Math.min(TEAM_PANEL_BONUS_DEFAULTS.furniture_flat_def, bonus.furniture_flat_def)));
    }
    bonus.version = TEAM_PANEL_BONUS_DEFAULTS.version;
    return bonus;
  }

  function teamPanelBonusMods(raw, catalog) {
    const bonus = normalizeTeamPanelBonus(raw, catalog);
    const out = mods();
    out.crit_dmg += bonus.furniture_crit_dmg;
    out.flat_atk += bonus.furniture_flat_atk + bonus.small_flat_atk;
    out.flat_def += bonus.furniture_flat_def;
    out.flat_hp += bonus.small_flat_hp;
    return out;
  }

  function substatMods(counts, constants) {
    const out = mods();
    const units = constants?.substat_units || {};
    Object.entries(SUBSTAT_EFFECT_KEYS).forEach(([key, effectKey]) => {
      const unit = units[key] || {};
      out[effectKey] += Math.max(0, num(counts?.[key])) * num(unit.unit_value);
    });
    return out;
  }

  function cartridgeMainStat(member, constants) {
    const characterId = String(member?.character_id || '');
    const options = constants?.cartridge_main_stat_options || {};
    const defaults = defaultBuildOptions(characterId, constants);
    const fallback = normalizeStatName(defaults.cartridge_main_stat) || Object.keys(options)[0] || '';
    const selected = normalizeStatName(member?.cartridge_main_stat) || fallback;
    return Object.prototype.hasOwnProperty.call(options, selected) ? selected : fallback;
  }

  function curtainBonus(member, constants) {
    const characterId = String(member?.character_id || '');
    const defaults = defaultBuildOptions(characterId, constants).curtain_bonus || {};
    const source = member?.curtain_bonus && typeof member.curtain_bonus === 'object' ? member.curtain_bonus : {};
    const options = constants?.curtain_bonus_stat_options || {};
    const fallback = normalizeStatName(defaults.stat) || Object.keys(options)[0] || '';
    const stat = normalizeStatName(source.stat) || fallback;
    const passiveType = CURTAIN_PASSIVE_TYPES.includes(String(source.passive_type || ''))
      ? String(source.passive_type)
      : (CURTAIN_PASSIVE_TYPES.includes(String(defaults.passive_type || '')) ? String(defaults.passive_type) : 'type3');
    return {
      value: Math.max(0, Math.min(100, num(source.value, num(defaults.value)))),
      stat: Object.prototype.hasOwnProperty.call(options, stat) ? stat : fallback,
      passive_type: passiveType,
    };
  }

  function cartridgePassiveLayers(cartridge, passiveType) {
    const type = CURTAIN_PASSIVE_TYPES.includes(passiveType) ? passiveType : 'type3';
    return Math.max(0, int(cartridge?.passive_counts?.[type]));
  }

  function mainStatMods(mainStat, constants) {
    const out = mods();
    const option = constants?.cartridge_main_stat_options?.[mainStat] || {};
    const key = String(option.modifier_key || '');
    if (Object.prototype.hasOwnProperty.call(out, key)) {
      out[key] += num(option.unit_value);
    }
    return out;
  }

  function curtainBonusMods(bonus, cartridge, constants) {
    const out = mods();
    const option = constants?.curtain_bonus_stat_options?.[normalizeStatName(bonus.stat)] || {};
    const key = String(option.modifier_key || '');
    const layers = cartridgePassiveLayers(cartridge, String(bonus.passive_type || 'type3'));
    if (Object.prototype.hasOwnProperty.call(out, key)) {
      out[key] += num(bonus.value) / 100 * layers;
    }
    return { modifiers: out, layers };
  }

  function skillLevels(raw) {
    const source = raw && typeof raw === 'object' ? raw : {};
    return Object.fromEntries(Object.entries(SKILL_LEVEL_DEFAULTS).map(([key, value]) => [
      key,
      Math.max(1, Math.min(10, int(source[key], value))),
    ]));
  }

  function skillLevelBonus(member) {
    return awakeningNodes(member).size >= 3 ? 1 : 0;
  }

  function skillLevelCategory(action) {
    const actionType = String(action?.action_type || '');
    const damageType = String(action?.damage_type || '');
    if (damageType === '无' || damageType === '' || actionType === '无') {
      return '';
    }
    if (actionType === '普攻' || damageType === '普攻' || actionType === '闪反' || actionType === '下落' || damageType === '闪反' || damageType === '下落') {
      return 'basic';
    }
    if (actionType === 'E' || damageType === 'E') {
      return 'skill';
    }
    if (actionType === 'Q' || damageType === 'Q') {
      return 'ultimate';
    }
    if (actionType === '援护' || damageType === '援护') {
      return 'support';
    }
    return '';
  }

  function skillLevelMultiplier(snapshot, action) {
    const category = skillLevelCategory(action);
    if (!category) {
      return { category: '', level: 0, multiplier: 1 };
    }
    const levels = snapshot.skill_levels || {};
    const baseLevel = Math.max(1, int(levels[category], SKILL_LEVEL_DEFAULTS[category]));
    const resonanceBonus = action?.resonance_skill_level_bonus === false
      ? 0
      : int(snapshot.skill_level_bonus);
    const effectiveLevel = Math.max(1, baseLevel + resonanceBonus);
    return {
      category,
      level: effectiveLevel,
      multiplier: Math.pow(1.08, effectiveLevel - 1),
    };
  }

  function actionTypeBonus(action, panelMods) {
    const actionType = String(action?.action_type || '');
    const damageType = String(action?.damage_type || '');
    const tags = actionTags(action);
    let total = panelMods.all_dmg;
    if (actionType === '普攻' || damageType === '普攻') total += panelMods.basic_dmg;
    if (actionType === '闪反' || damageType === '闪反') total += panelMods.dodge_counter_dmg;
    if (actionType === 'E' || damageType === 'E') total += panelMods.skill_dmg;
    if (actionType === 'Q' || damageType === 'Q') total += panelMods.ultimate_dmg;
    if (tags.has('追击')) total += panelMods.follow_dmg;
    if (tags.has('心灵')) total += panelMods.mind_dmg;
    if (tags.has('附着')) total += panelMods.attach_dmg;
    return total;
  }

  function buildSnapshot(member, catalog, teamPanelBonus) {
    const characters = recordMap(catalog.characters);
    const arcs = recordMap(catalog.arcs);
    const cartridges = recordMap(catalog.cartridges);
    const constants = catalog.formula_constants || {};
    const character = characters.get(String(member?.character_id || ''));
    if (!character) {
      throw new Error('队伍中存在未知角色。');
    }
    const arc = arcs.get(String(member?.arc_id || '')) || null;
    const arcRefinementRecord = catalog?.arc_refinements?.arcs?.[String(member?.arc_id || '')] || {};
    const requestedArcRefinement = int(member?.arc_refinement);
    const arcRefinementLevel = requestedArcRefinement >= 1 && requestedArcRefinement <= 5
      ? requestedArcRefinement
      : Math.max(1, Math.min(5, int(arcRefinementRecord.default_level, 1)));
    const arcRefinement = arcRefinementRecord.levels?.[String(arcRefinementLevel)] || null;
    const cartridge = cartridges.get(String(member?.cartridge_id || '')) || null;
    const panelMods = mods();
    const mainStat = cartridgeMainStat(member, constants);
    const bonus = curtainBonus(member, constants);
    const curtain = curtainBonusMods(bonus, cartridge, constants);
    // Character-sheet modifiers historically contained baked passive/awakening
    // bonuses. They are deliberately ignored; character buffs are registry rules.
    panelMods.crit_rate = 0.05;
    panelMods.crit_dmg = 0.5;
    if (member?.bond_full || int(member?.bond_level) > 0) {
      mergeMods(panelMods, character.bond_bonus?.modifiers);
    }
    if (arc) mergeMods(panelMods, arcRefinement?.panel_modifiers || arc.modifiers);
    if (cartridge) {
      const cartridgeModifiers = { ...(cartridge.modifiers || {}) };
      const requiredElement = String(cartridge.required_element || '');
      if (requiredElement && requiredElement !== String(character.element || '')) {
        cartridgeModifiers.element_dmg = 0;
      }
      mergeMods(panelMods, cartridgeModifiers);
    }
    mergeMods(panelMods, mainStatMods(mainStat, constants));
    mergeMods(panelMods, curtain.modifiers);
    mergeMods(panelMods, substatMods(member?.substat_counts || {}, constants));
    mergeMods(panelMods, teamPanelBonusMods(teamPanelBonus, catalog));
    const element = String(character.element || '');
    panelMods.element_dmg += num((arcRefinement?.element_dmg || arc?.element_dmg)?.[element]);
    const baseStats = character.base_stats || {};
    const baseAtk = num(baseStats.atk) + num(arc?.base_atk);
    const baseHp = num(baseStats.hp);
    const baseDef = num(baseStats.def);
    const stats = {
      atk: baseAtk * (1 + panelMods.atk_pct) + panelMods.flat_atk,
      hp: baseHp * (1 + panelMods.hp_pct) + panelMods.flat_hp,
      def: baseDef * (1 + panelMods.def_pct) + panelMods.flat_def,
      harmony_strength: panelMods.harmony_strength,
      stagger_strength: panelMods.stagger_strength,
      crit_rate: panelMods.crit_rate,
      crit_dmg: panelMods.crit_dmg,
    };
    return {
      slot: int(member?.slot),
      awakening: awakeningNodes(member).size,
      awakening_nodes: Array.from(awakeningNodes(member)).sort((left, right) => left - right),
      character,
      arc,
      arc_refinement: arcRefinementLevel,
      cartridge,
      mods: panelMods,
      base_stats: { atk: baseAtk, hp: baseHp, def: baseDef },
      stats,
      skill_levels: skillLevels(member?.skill_levels),
      skill_level_bonus: skillLevelBonus(member),
      build_options: {
        cartridge_main_stat: mainStat,
        curtain_bonus: {
          value: num(bonus.value),
          stat: bonus.stat || '',
          passive_type: bonus.passive_type || 'type3',
          layers: curtain.layers,
        },
      },
      personal_resources: {},
    };
  }

  function buildPanelProjection(snapshot) {
    const character = snapshot.character || {};
    const panelMods = snapshot.mods || {};
    const baseStats = snapshot.base_stats || {};
    const stats = snapshot.stats || {};
    return {
      slot: int(snapshot.slot),
      character_id: character.id || '',
      character_name: character.name || '',
      base_stats: {
        atk: num(baseStats.atk),
        hp: num(baseStats.hp),
        def: num(baseStats.def),
      },
      zones: {
        atk_pct: num(panelMods.atk_pct),
        flat_atk: num(panelMods.flat_atk),
        hp_pct: num(panelMods.hp_pct),
        flat_hp: num(panelMods.flat_hp),
        def_pct: num(panelMods.def_pct),
        flat_def: num(panelMods.flat_def),
      },
      panel: {
        atk: num(stats.atk),
        hp: num(stats.hp),
        def: num(stats.def),
        crit_rate: num(stats.crit_rate),
        crit_dmg: num(stats.crit_dmg),
        element_dmg: num(panelMods.element_dmg),
        energy_recharge: num(panelMods.energy_recharge),
        harmony_strength: num(stats.harmony_strength),
        stagger_strength: num(stats.stagger_strength),
        all_dmg: num(panelMods.all_dmg),
      },
      build_options: snapshot.build_options || {},
    };
  }

  function normalizeEnemy(raw) {
    const enemy = raw && typeof raw === 'object' ? raw : {};
    const weakness = Array.isArray(enemy.weakness_elements) ? enemy.weakness_elements : [];
    const debuffs = enemy.debuffs && typeof enemy.debuffs === 'object' ? enemy.debuffs : {};
    const hpRatio = enemy.hp_ratio == null && enemy.hp_percent != null ? num(enemy.hp_percent, 100) / 100 : num(enemy.hp_ratio, 1);
    const resistances = enemy.resistances && typeof enemy.resistances === 'object' ? enemy.resistances : {};
    return {
      level: Math.max(1, Math.min(120, int(enemy.level, 90))),
      track_outside: Boolean(enemy.track_outside),
      weakness_elements: weakness.map(String).filter((item) => ELEMENTS.includes(item)),
      debuffs: Object.fromEntries(Object.entries(debuffs)
        .filter(([name]) => Object.prototype.hasOwnProperty.call(ENEMY_DEBUFF_DURATIONS, name))
        .map(([name, endTick]) => [name, Math.max(0, int(endTick))])),
      hp_ratio: Math.max(0, Math.min(1, hpRatio)),
      resistances: Object.fromEntries(
        [...ELEMENTS, '心灵'].map((element) => [element, Math.max(-1, Math.min(1, num(resistances[element], 0.3)))]),
      ),
    };
  }

  function settledResistance(character, enemy, panelMods) {
    const element = String(character?.element || '');
    const damageElement = element || '心灵';
    const baseRes = num(enemy.resistances?.[damageElement], 0.3);
    const weaknessDown = new Set(enemy.weakness_elements || []).has(element) ? 0.2 : 0;
    const elementResDown = num(panelMods[`res_down_${damageElement}`]);
    return baseRes - weaknessDown - num(panelMods.res_down) - elementResDown;
  }

  function resistanceMultiplier(character, enemy, panelMods) {
    const value = 1 - settledResistance(character, enemy, panelMods);
    if (value < 1) {
      return Math.max(0.05, value);
    }
    return Math.max(0.05, 2 - 1 / Math.max(value, 0.01));
  }

  function settledDefense(enemy, panelMods) {
    const enemyFactor = 6 * int(enemy.level, 90) + 600 - (enemy.track_outside ? 60 : 0);
    return enemyFactor
      * Math.max(0, 1 - Math.min(1, panelMods.def_ignore))
      * Math.max(0, 1 - Math.min(1, panelMods.def_down));
  }

  function defenseMultiplier(enemy, panelMods) {
    const actorLevelFactor = 6 * 80 + 600;
    const defenseLeft = settledDefense(enemy, panelMods);
    return actorLevelFactor / Math.max(actorLevelFactor + defenseLeft, 1);
  }

  function staggerBaseDamage(level) {
    const levelKey = Math.max(5, Math.min(80, Math.floor(num(level, 80) / 5) * 5));
    return STAGGER_LIMIT * num(STAGGER_DAMAGE_BASE[levelKey]) / 3;
  }

  function staggerProfile(panelMods, character) {
    const damageElement = String(character?.element || '');
    return {
      stagger_strength: num(panelMods.stagger_strength),
      stagger_damage_bonus: num(panelMods.stagger_damage_bonus),
      def_ignore: num(panelMods.def_ignore),
      def_down: num(panelMods.def_down),
      res_down: num(panelMods.res_down) + num(panelMods[`res_down_${damageElement}`]),
    };
  }

  function averageStaggerProfile(snapshot, samples) {
    const totalWeight = asList(samples).reduce((sum, sample) => sum + Math.max(0, num(sample.weight)), 0);
    if (totalWeight <= 0) {
      return staggerProfile(snapshot.mods || {}, snapshot.character);
    }
    return asList(samples).reduce((average, sample) => {
      Object.keys(average).forEach((key) => {
        average[key] += num(sample.profile?.[key]) * Math.max(0, num(sample.weight)) / totalWeight;
      });
      return average;
    }, {
      stagger_strength: 0,
      stagger_damage_bonus: 0,
      def_ignore: 0,
      def_down: 0,
      res_down: 0,
    });
  }

  function staggerContribution(snapshot, averageProfile, enemy) {
    const staggerMods = mods();
    staggerMods.def_ignore = averageProfile.def_ignore;
    staggerMods.def_down = averageProfile.def_down;
    staggerMods.res_down = averageProfile.res_down;
    const baseDamage = staggerBaseDamage(snapshot.character?.level);
    const staggerDamageBonus = averageStaggerDamageBonus(snapshot, averageProfile);
    const characterMultiplier = String(snapshot.character?.name || '') === '达芙蒂尔'
      ? (hasAwakeningNode(snapshot, 5) ? 3 : 2)
      : 1;
    return Math.max(
      0,
      baseDamage
        * (1 + staggerDamageBonus)
        * defenseMultiplier(enemy, staggerMods)
        * resistanceMultiplier(snapshot.character, enemy, staggerMods)
        * characterMultiplier,
    );
  }

  function averageStaggerDamageBonus(snapshot, averageProfile) {
    return averageProfile.stagger_strength / 300
      + averageProfile.stagger_damage_bonus;
  }

  function critMultiplier(action, panelMods) {
    const rate = String(action?.extra_tag || '') === 'DOT' ? 0.5 : Math.min(1, Math.max(0, panelMods.crit_rate));
    return Math.max(1, 1 + rate * Math.max(0, panelMods.crit_dmg));
  }

  function calculateActionDamage(snapshot, action, enemy, extraModifiers) {
    const panelMods = mergeMods(clone(snapshot.mods), extraModifiers);
    mergeMods(panelMods, action.self_modifiers);
    const baseStats = snapshot.base_stats || {};
    const stats = {
      atk: num(baseStats.atk) * (1 + panelMods.atk_pct) + panelMods.flat_atk,
      hp: num(baseStats.hp) * (1 + panelMods.hp_pct) + panelMods.flat_hp,
      def: num(baseStats.def) * (1 + panelMods.def_pct) + panelMods.flat_def,
      harmony_strength: panelMods.harmony_strength,
      stagger_strength: panelMods.stagger_strength,
      crit_rate: panelMods.crit_rate,
      crit_dmg: panelMods.crit_dmg,
    };
    const multipliers = Object.assign({}, action.multipliers || {});
    Object.entries(action.multipliers_add_by_awakening_node || {}).forEach(([level, additions]) => {
      if (!hasAwakeningNode(snapshot, level) || !additions || typeof additions !== 'object') return;
      Object.entries(additions).forEach(([key, value]) => {
        multipliers[key] = num(multipliers[key]) + num(value);
      });
    });
    const rawScalingBase = stats.atk * num(multipliers.atk) + stats.hp * num(multipliers.hp) + stats.def * num(multipliers.def);
    const baseMultiplierFactor = Math.max(0, 1 + panelMods.base_multiplier_pct);
    const scalingBase = rawScalingBase * baseMultiplierFactor;
    const skill = skillLevelMultiplier(snapshot, action);
    let base = scalingBase * skill.multiplier + num(multipliers.flat);
    let skillCategory = skill.category;
    let skillLevel = skill.level;
    let skillMult = skill.multiplier;
    if (['无', ''].includes(String(action.damage_type || ''))) {
      base = 0;
      skillCategory = '';
      skillLevel = 0;
      skillMult = 1;
    }
    const actionDamageBonus = actionTypeBonus(action, panelMods);
    const otherDamageBonus = actionDamageBonus - panelMods.all_dmg;
    const dmgBonus = actionDamageBonus + panelMods.element_dmg;
    const crit = critMultiplier(action, panelMods);
    const resistanceCharacter = actionTags(action).has('心灵') ? { element: '' } : snapshot.character;
    const effectiveResistance = settledResistance(resistanceCharacter, enemy, panelMods);
    const effectiveDefense = settledDefense(enemy, panelMods);
    const resistance = resistanceMultiplier(resistanceCharacter, enemy, panelMods);
    const defense = defenseMultiplier(enemy, panelMods);
    const direct = Math.max(0, base * (1 + dmgBonus) * crit * resistance * defense * (1 + panelMods.final_dmg));
    return {
      direct_damage: direct,
      stagger_amount: Math.max(
        0,
        actionValueForAwakeningNodes(
          action,
          snapshot,
          'stagger',
          'stagger_by_awakening_node',
        ) * (1 + panelMods.stagger_strength / 300) * (1 + panelMods.stagger_multiplier),
      ),
      harmony: num(action.harmony),
      energy_gain: num(action.energy_gain) * (1 + panelMods.energy_recharge),
      panel: {
        atk: stats.atk,
        hp: stats.hp,
        def: stats.def,
        harmony_strength: stats.harmony_strength,
      stagger_strength: stats.stagger_strength,
      interrupt_resistance: num(panelMods.interrupt_resistance),
        crit_rate: stats.crit_rate,
        crit_dmg: stats.crit_dmg,
        all_dmg: panelMods.all_dmg,
        element_dmg: panelMods.element_dmg,
        other_dmg: otherDamageBonus,
        final_dmg: panelMods.final_dmg,
      },
      formula_parts: {
        base,
        raw_base: rawScalingBase + num(multipliers.flat),
        base_multiplier_factor: baseMultiplierFactor,
        skill_level_category: skillCategory,
        skill_level: skillLevel,
        skill_level_multiplier: skillMult,
        dmg_bonus: dmgBonus,
        crit,
        settled_resistance: effectiveResistance,
        settled_defense: effectiveDefense,
        resistance,
        defense,
        final_multiplier: 1 + panelMods.final_dmg,
      },
      stagger_profile: staggerProfile(panelMods, snapshot.character),
    };
  }

  function actionTags(action) {
    const tags = new Set(asList(action?.tags).map(String).filter(Boolean));
    const extraTag = String(action?.extra_tag || '');
    if (extraTag) tags.add(extraTag);
    return tags;
  }

  function actionTagsForSnapshot(action, snapshot) {
    const tags = actionTags(action);
    Object.entries(action?.tags_by_awakening_node || {}).forEach(([level, values]) => {
      if (!hasAwakeningNode(snapshot, level)) return;
      asList(values).map(String).filter(Boolean).forEach((tag) => tags.add(tag));
    });
    return tags;
  }

  function specialDamageSource(action) {
    const explicitSource = String(action?.damage_source || '').trim();
    return SPECIAL_DAMAGE_SOURCES.includes(explicitSource) ? explicitSource : '';
  }

  function reactionForElements(firstElement, secondElement) {
    const pair = [String(firstElement || ''), String(secondElement || '')]
      .sort((a, b) => ELEMENTS.indexOf(a) - ELEMENTS.indexOf(b))
      .join('|');
    return REACTION_BY_ELEMENT_PAIR.get(pair) || '';
  }

  function reactionBaseDamage(level, reaction) {
    const bracket = Math.max(5, Math.min(80, Math.floor(Math.max(5, num(level, 80)) / 5) * 5));
    return num(REACTION_BASE_DAMAGE[bracket]?.[reaction]);
  }

  function reactionStrengthMultiplier(harmonyStrength) {
    return 1 + Math.max(0, num(harmonyStrength)) / 600;
  }

  function reactionAmplificationMultiplierForStrength(harmonyStrength) {
    const strength = Math.max(0, num(harmonyStrength));
    const strengthBonus = strength > 0 ? 0.2 * strength / (strength + 180) : 0;
    return 1.2 * (1 + strengthBonus);
  }

  function isBackgroundAction(action) {
    return Boolean(action?.is_background_damage) || `${action?.name || ''} ${action?.extra_tag || ''}`.includes('后台');
  }

  function backgroundActionMultiplier(step, action) {
    return isBackgroundAction(action)
      ? Math.max(1, Math.min(MAX_BACKGROUND_ACTION_MULTIPLIER, int(step?.repeat, 1)))
      : 1;
  }

  function isBasicAction(action) {
    return String(action?.action_type || '') === '普攻' || String(action?.damage_type || '') === '普攻';
  }

  function canBackgroundOverride(action) {
    return Boolean(action?.can_background_override) && isBasicAction(action);
  }

  function isBasicBackgroundOverride(step, action) {
    return !isBackgroundAction(action) && canBackgroundOverride(action) && String(step?.placement || '') === 'background';
  }

  function isStepBackground(step, action) {
    return isBackgroundAction(action) || isBasicBackgroundOverride(step, action);
  }

  function startsForeground(step, action) {
    return !isStepBackground(step, action);
  }

  function blocksSlotOverlap(step, action) {
    return startsForeground(step, action) || isBasicBackgroundOverride(step, action);
  }

  function isSupportAction(action) {
    return String(action?.action_type || '') === '援护';
  }

  function isInstantNativeBackgroundAction(step, action) {
    return isBackgroundAction(action) && !isSupportAction(action) && !Boolean(action?.pre_input_node);
  }

  function actionCalculationDurationTicks(step, action) {
    return isInstantNativeBackgroundAction(step, action) ? 0 : Math.max(0, int(action?.duration_ticks));
  }

  function actionVisualDurationTicks(step, action) {
    if (isInstantNativeBackgroundAction(step, action)) {
      return ZERO_ACTION_VISUAL_TICKS;
    }
    const durationTicks = Math.max(0, int(action?.duration_ticks));
    return durationTicks > 0 ? durationTicks : ZERO_ACTION_VISUAL_TICKS;
  }

  function isQAction(action) {
    return String(action?.action_type || '') === 'Q' || String(action?.damage_type || '') === 'Q';
  }

  function isInstantSwitchAction(action) {
    return Boolean(action?.is_instant_switch);
  }

  function locksForegroundSwitch(step, action) {
    return startsForeground(step, action) && (
      isSupportAction(action) ||
      isZeroForegroundQStep(step, action)
    );
  }

  function tickScheduledStepEntries(steps, actionsById, switchGapTicks = MIN_FOREGROUND_START_GAP_TICKS) {
    const ordered = steps.slice().sort((a, b) => {
      const startDelta = int(a.start_tick) - int(b.start_tick);
      if (startDelta) return startDelta;
      const aAction = actionsById.get(String(a.action_id || '')) || {};
      const bAction = actionsById.get(String(b.action_id || '')) || {};
      const instantSwitchDelta = Number(isInstantSwitchAction(bAction)) - Number(isInstantSwitchAction(aAction));
      if (instantSwitchDelta) return instantSwitchDelta;
      const lockDelta = Number(locksForegroundSwitch(b, bAction)) - Number(locksForegroundSwitch(a, aAction));
      return lockDelta || int(a.slot) - int(b.slot);
    });
    let previousForegroundSlot = null;
    let previousForegroundStartTick = null;
    const foregroundLocks = [];
    const carriedShiftBySlot = new Map();
    return ordered.map((step) => {
      const action = actionsById.get(String(step.action_id || '')) || {};
      const isBackground = isStepBackground(step, action);
      const slot = int(step.slot);
      let visualStartTick = int(step.start_tick) + Math.max(0, int(carriedShiftBySlot.get(slot)));
      const carriedStartTick = visualStartTick;
      let switchLossTicks = 0;
      let foregroundLockTicks = 0;
      if (!isBackground) {
        let lockEndTick = Math.max(
          0,
          ...foregroundLocks
            .filter((lock) => visualStartTick < int(lock.end_tick))
            .map((lock) => int(lock.end_tick)),
        );
        while (lockEndTick > visualStartTick) {
          foregroundLockTicks += lockEndTick - visualStartTick;
          visualStartTick = lockEndTick;
          lockEndTick = Math.max(
            0,
            ...foregroundLocks
              .filter((lock) => visualStartTick < int(lock.end_tick))
              .map((lock) => int(lock.end_tick)),
          );
        }
      }
      if (
        !isBackground &&
        previousForegroundSlot !== null &&
        previousForegroundSlot !== int(step.slot) &&
        previousForegroundStartTick !== null &&
        !isInstantSwitchAction(action)
      ) {
        const earliestStartTick = previousForegroundStartTick + Math.max(0, int(switchGapTicks));
        if (visualStartTick < earliestStartTick) {
          switchLossTicks = earliestStartTick - visualStartTick;
          visualStartTick = earliestStartTick;
        }
      }
      if (!isBackground) {
        previousForegroundSlot = int(step.slot);
        previousForegroundStartTick = visualStartTick;
      }
      if (locksForegroundSwitch(step, action)) {
        foregroundLocks.push({
          slot,
          end_tick: visualStartTick + qVisualDurationTicks(action),
        });
      }
      const addedShiftTicks = Math.max(0, visualStartTick - carriedStartTick);
      if (addedShiftTicks > 0) {
        carriedShiftBySlot.set(slot, Math.max(0, int(carriedShiftBySlot.get(slot))) + addedShiftTicks);
      }
      return { step, visualStartTick, switchLossTicks, foregroundLockTicks };
    });
  }

  function qVisualDurationTicks(action) {
    const durationTicks = Math.max(0, int(action?.duration_ticks));
    return durationTicks > 0 ? durationTicks : ZERO_ACTION_VISUAL_TICKS;
  }

  function isZeroForegroundQStep(step, action) {
    return startsForeground(step, action) && isQAction(action) && Math.max(0, int(action?.duration_ticks)) === 0;
  }

  function isQCoverImmuneScheduled(scheduled) {
    return isSupportAction(scheduled?.action || {}) ||
      isZeroForegroundQStep(scheduled?.step || {}, scheduled?.action || {});
  }

  function calculationTickFromVisualIntervals(visualTick, qIntervals) {
    const safeTick = Math.max(0, int(visualTick));
    let offset = 0;
    asList(qIntervals).forEach((interval) => {
      const startTick = Math.max(0, int(interval?.start_tick));
      const endTick = Math.max(startTick, int(interval?.end_tick, startTick + ZERO_ACTION_VISUAL_TICKS));
      if (startTick < safeTick) {
        offset += Math.min(endTick - startTick, safeTick - startTick);
      }
    });
    return Math.max(0, safeTick - offset);
  }

  function normalizeFrozenIntervals(qIntervals) {
    const normalized = asList(qIntervals)
      .map((interval) => {
        const startTick = Math.max(0, int(interval?.start_tick));
        return {
          start_tick: startTick,
          end_tick: Math.max(startTick, int(interval?.end_tick, startTick)),
        };
      })
      .filter((interval) => interval.end_tick > interval.start_tick)
      .sort((left, right) => left.start_tick - right.start_tick || left.end_tick - right.end_tick);
    const merged = [];
    normalized.forEach((interval) => {
      const previous = merged[merged.length - 1];
      if (previous && interval.start_tick <= previous.end_tick) {
        previous.end_tick = Math.max(previous.end_tick, interval.end_tick);
      } else {
        merged.push({ ...interval });
      }
    });
    return merged;
  }

  function visualTickFromCalculationTick(calculationTick, qIntervals) {
    const safeCalculationTick = Math.max(0, int(calculationTick));
    let visualTick = safeCalculationTick;
    asList(qIntervals)
      .slice()
      .sort((left, right) => int(left?.start_tick) - int(right?.start_tick))
      .forEach((interval) => {
        const startTick = Math.max(0, int(interval?.start_tick));
        const endTick = Math.max(startTick, int(interval?.end_tick, startTick + ZERO_ACTION_VISUAL_TICKS));
        const intervalCalculationTick = calculationTickFromVisualIntervals(startTick, qIntervals);
        if (intervalCalculationTick < safeCalculationTick) {
          visualTick += endTick - startTick;
        }
      });
    return visualTick;
  }

  function recalculateTimingsFromFrozenIntervals(scheduledSteps, qIntervals) {
    scheduledSteps.forEach((scheduled) => {
      const visualStartTick = int(scheduled.visual_start_tick, int(scheduled.step?.start_tick));
      const visualEndTick = Math.max(visualStartTick, int(scheduled.visual_end_tick, visualStartTick));
      const startTick = calculationTickFromVisualIntervals(visualStartTick, qIntervals);
      const endTick = calculationTickFromVisualIntervals(visualEndTick, qIntervals);
      scheduled.start_tick = startTick;
      if (scheduled.calculation_at_start_only) {
        scheduled.end_tick = startTick;
        scheduled.duration_ticks = 0;
      } else {
        scheduled.end_tick = Math.max(startTick, endTick);
        scheduled.duration_ticks = Math.max(0, scheduled.end_tick - scheduled.start_tick);
      }
      if (!scheduled.q_instant_release) {
        scheduled.calculation_start_sequence = 0;
        scheduled.calculation_end_sequence = 0;
      }
    });
  }

  function calculateAxisDurationTicks(steps, actionsById) {
    return Math.max(0, ...asList(steps).map((step) => {
      const action = actionsById.get(String(step?.action_id || '')) || {};
      const startTick = Math.max(0, int(step?.start_tick));
      const durationTicks = isZeroForegroundQStep(step, action)
        ? ZERO_ACTION_VISUAL_TICKS
        : actionVisualDurationTicks(step, action);
      return Math.max(startTick, startTick + durationTicks);
    }));
  }

  function actionHitCount(action) {
    return Math.max(0, int(action?.hit_count));
  }

  function actionEnemyDebuffs(action) {
    const out = new Set();
    actionTags(action).forEach((tag) => {
      if (tag.startsWith('enemy_debuff:')) {
        const name = tag.split(':')[1];
        if (Object.prototype.hasOwnProperty.call(ENEMY_DEBUFF_DURATIONS, name)) out.add(name);
      } else if (Object.prototype.hasOwnProperty.call(ENEMY_DEBUFF_DURATIONS, tag)) {
        out.add(tag);
      }
    });
    return out;
  }

  function activeEnemyDebuffs(enemyDebuffs, tick) {
    return Object.fromEntries(Object.entries(enemyDebuffs || {}).filter(([, endTick]) => tick < int(endTick)));
  }

  function applyEnemyDebuffs(enemyDebuffs, action, tick) {
    const applied = [];
    Array.from(actionEnemyDebuffs(action)).sort().forEach((name) => {
      enemyDebuffs[name] = Math.max(int(enemyDebuffs[name]), tick + ENEMY_DEBUFF_DURATIONS[name]);
      applied.push(name);
    });
    return applied;
  }

  function expectedCriticalHits(action, calc) {
    const hitCount = actionHitCount(action);
    if (hitCount <= 0) return 0;
    const panel = calc.panel || {};
    const rate = actionTags(action).has('DOT') ? 0.5 : Math.min(1, Math.max(0, num(panel.crit_rate)));
    return hitCount * rate;
  }

  function validateSteps(steps, actionsById) {
    asList(steps).forEach((step) => {
      const action = actionsById.get(String(step.action_id || ''));
      if (!action) throw new Error('轴中存在未知动作。');
    });
  }

  function placement(isBackground) {
    return isBackground ? 'background' : 'foreground';
  }

  function sourceMatches(source, rule, step, action, snapshot, isBackground) {
    const scope = String(source?.scope || 'registrar');
    if (scope === 'registrar' && int(step.slot) !== int(rule.owner_slot)) return false;
    if (scope === 'non_registrar' && int(step.slot) === int(rule.owner_slot)) return false;
    const actionTypes = strSet(source?.action_types);
    if (actionTypes.size && !actionTypes.has(String(action.action_type || ''))) return false;
    const damageTypes = strSet(source?.damage_types);
    if (damageTypes.size && !damageTypes.has(String(action.damage_type || ''))) return false;
    const actionNames = strSet(source?.action_names);
    if (actionNames.size && !actionNames.has(String(action.name || ''))) return false;
    const actionIds = strSet(source?.action_ids);
    if (actionIds.size && !actionIds.has(String(action.id || ''))) return false;
    const tags = strSet(source?.tags);
    if (tags.size && !Array.from(actionTags(action)).some((tag) => tags.has(tag))) return false;
    const placements = strSet(source?.placements);
    if (placements.size && !placements.has(placement(isBackground))) return false;
    const elements = strSet(source?.elements);
    if (elements.size && !elements.has(String(snapshot.character?.element || ''))) return false;
    return true;
  }

  function conditionsMatch(conditions, context) {
    return asList(conditions).every((condition) => {
      if (!condition || typeof condition !== 'object') return false;
      const type = String(condition.type || '');
      if (type === '' || type === 'always') return true;
      const tags = new Set(asList(context.action_tags).map(String));
      if (type === 'unsupported') return false;
      if (type === 'action_tag') return Array.from(strSet(condition.tags)).some((tag) => tags.has(tag));
      if (type === 'self_hp_loss') return ['self_hp_loss', 'hp_loss', '扣血', '降低生命'].some((tag) => tags.has(tag));
      if (type === 'heal') return ['heal', '治疗'].some((tag) => tags.has(tag));
      if (type === 'fons_full') return context.fons_full !== false;
      if (type === 'non_loop_axis') return context.loop_enabled !== true;
      if (type === 'awakening_min') {
        const level = int(condition.min, int(condition.value));
        return Array.isArray(context.owner_awakening_nodes)
          ? context.owner_awakening_nodes.map(int).includes(level)
          : int(context.owner_awakening) >= level;
      }
      if (type === 'awakening_max') {
        const nextLevel = int(condition.max, int(condition.value)) + 1;
        return Array.isArray(context.owner_awakening_nodes)
          ? !context.owner_awakening_nodes.map(int).includes(nextLevel)
          : int(context.owner_awakening) < nextLevel;
      }
      if (type === 'awakening_count_min') {
        const requiredCount = int(condition.min, int(condition.value));
        const activeCount = Array.isArray(context.owner_awakening_nodes)
          ? new Set(context.owner_awakening_nodes.map(int).filter((value) => value >= 1 && value <= 6)).size
          : int(context.owner_awakening);
        return activeCount >= requiredCount;
      }
      if (type === 'awakening_count_max') {
        const maxCount = int(condition.max, int(condition.value));
        const activeCount = Array.isArray(context.owner_awakening_nodes)
          ? new Set(context.owner_awakening_nodes.map(int).filter((value) => value >= 1 && value <= 6)).size
          : int(context.owner_awakening);
        return activeCount <= maxCount;
      }
      if (type === 'expected_critical_hit') return num(context.expected_critical_hits) > 0;
      if (type === 'hit_count_positive') return num(context.hit_count) > 0;
      if (type === 'enemy_debuff_active') {
        const active = Object.entries(context.enemy_debuffs || {}).filter(([, value]) => num(value) > num(context.tick)).map(([key]) => key);
        return Array.from(strSet(condition.debuffs)).some((name) => active.includes(name));
      }
      if (type === 'enemy_debuff_applied') {
        const applied = new Set(asList(context.applied_enemy_debuffs).map(String));
        return Array.from(strSet(condition.debuffs)).some((name) => applied.has(name));
      }
      if (type === 'shield_state') return Boolean(context.shield_active);
      if (type === 'owner_character_id') {
        return new Set(asList(condition.ids).map(String)).has(String(context.owner_character_id || ''));
      }
      if (type === 'take_damage') return Boolean(context.take_damage);
      if (type === 'enemy_hp_below') return num(context.enemy?.hp_ratio, num(context.enemy?.hp_percent, 100) / 100) < num(condition.threshold, 0.5);
      if (type === 'enemy_weak_to_owner_element') {
        const element = String(context.snapshot?.character?.element || '');
        return new Set(context.enemy?.weakness_elements || []).has(element);
      }
      if (type === 'active_buff_key') {
        return new Set(asList(context.active_buff_keys).map(String)).has(String(condition.key || ''));
      }
      if (type === 'active_buff_any') {
        const active = new Set(asList(context.active_buff_keys).map(String));
        return asList(condition.keys).map(String).some((key) => active.has(key));
      }
      if (type === 'active_buff_none') {
        const active = new Set(asList(context.active_buff_keys).map(String));
        return !asList(condition.keys).map(String).some((key) => active.has(key));
      }
      if (type === 'existing_buff_none') {
        const existing = new Set(asList(context.existing_buff_keys).map(String));
        return !asList(condition.keys).map(String).some((key) => existing.has(key));
      }
      if (type === 'personal_resource_min') {
        return num(context.personal_resources?.[String(condition.resource || '')]) >= num(condition.min, condition.value);
      }
      if (type === 'reaction_owner_involved') {
        const ownerSlot = int(context.owner_slot, -1);
        return ownerSlot === int(context.reaction?.previous_slot, -2) || ownerSlot === int(context.reaction?.support_slot, -2);
      }
      if (type === 'reaction_type') {
        return new Set(asList(condition.reactions).map(String)).has(String(context.reaction?.reaction || ''));
      }
      return false;
    });
  }

  function registeredBuffRules(team, catalog) {
    const buffs = asList(catalog.buffs);
    const characters = recordMap(catalog.characters);
    const arcs = recordMap(catalog.arcs);
    const cartridges = recordMap(catalog.cartridges);
    const rules = [];
    buffs.forEach((buff) => {
      asList(buff?.providers).forEach((provider) => {
        asList(team).forEach((member) => {
          const kind = String(provider?.kind || '');
          const providerId = String(provider?.id || '');
          const selectedId = kind === 'arc'
            ? String(member.arc_id || '')
            : (kind === 'cartridge' ? String(member.cartridge_id || '') : (kind === 'character' ? String(member.character_id || '') : ''));
          if (!providerId || providerId !== selectedId) return;
          const rule = clone(buff);
          rule.owner_slot = int(member.slot);
          rule.owner_character_id = String(member.character_id || '');
          rule.owner_character_name = String(member.character_name || '');
          rule.owner_awakening = awakeningNodes(member).size;
          rule.owner_awakening_nodes = Array.from(awakeningNodes(member)).sort((left, right) => left - right);
          const arcRefinementRecord = catalog?.arc_refinements?.arcs?.[providerId] || {};
          const requestedArcRefinement = int(member.arc_refinement);
          rule.owner_arc_refinement = requestedArcRefinement >= 1 && requestedArcRefinement <= 5
            ? requestedArcRefinement
            : Math.max(1, Math.min(5, int(arcRefinementRecord.default_level, 1)));
          if (kind === 'arc') {
            const refinement = arcRefinementRecord.levels?.[String(rule.owner_arc_refinement)];
            const refinementEffects = refinement?.buff_effects?.[String(rule.id || '')];
            if (refinementEffects && typeof refinementEffects === 'object') {
              rule.effects = clone(refinementEffects);
            }
          }
          if (kind === 'arc') rule.provider_name = arcs.get(providerId)?.name || '';
          if (kind === 'cartridge') rule.provider_name = cartridges.get(providerId)?.name || '';
          if (kind === 'character') rule.provider_name = characters.get(providerId)?.name || rule.owner_character_name;
          rule.provider_kind = kind;
          rules.push(rule);
        });
      });
    });
    return rules.sort((a, b) => int(a.priority, 100) - int(b.priority, 100) || String(a.id || '').localeCompare(String(b.id || '')) || int(a.owner_slot) - int(b.owner_slot));
  }

  function legacyBuffRules(rawRules) {
    return asList(rawRules).map((rawRule, index) => {
      if (!rawRule || typeof rawRule !== 'object') return null;
      const trigger = rawRule.trigger && typeof rawRule.trigger === 'object' ? rawRule.trigger : {};
      const targets = rawRule.targets && typeof rawRule.targets === 'object' ? rawRule.targets : {};
      const modifiers = rawRule.modifiers && typeof rawRule.modifiers === 'object' ? rawRule.modifiers : {};
      const durationTicks = Math.max(0, int(rawRule.duration_ticks));
      if (!Object.keys(modifiers).length || durationTicks <= 0) return null;
      const triggerSlot = trigger.slot;
      const hasTriggerSlot = triggerSlot != null && triggerSlot !== '';
      const targetSlots = asList(targets.slots).map(int);
      const effects = Object.fromEntries(Object.entries(modifiers).filter(([, value]) => num(value) !== 0).map(([key, value]) => [key, num(value)]));
      if (!Object.keys(effects).length) return null;
      return {
        id: String(rawRule.id || `legacy_buff_${String(index + 1).padStart(3, '0')}`),
        name: String(rawRule.name || `增益 ${index + 1}`),
        provider_kind: 'legacy',
        provider_name: '手动增益',
        owner_slot: hasTriggerSlot ? int(triggerSlot) : -1,
        priority: 10000 + index,
        trigger: {
          event: 'action_hit',
          source: {
            scope: hasTriggerSlot ? 'registrar' : 'team',
            action_ids: trigger.action_id ? [String(trigger.action_id)] : [],
            action_types: trigger.action_type ? [String(trigger.action_type)] : [],
          },
        },
        target: {
          scope: targetSlots.length ? 'slots' : 'team',
          slots: targetSlots,
          action_ids: asList(targets.action_ids).map(String),
          action_types: asList(targets.action_types).map(String),
        },
        duration: { type: 'time', ticks: durationTicks, delay_ticks: Math.max(0, int(rawRule.delay_ticks)), loop_carry: false },
        stacking: { mode: 'refresh', max_stacks: 1 },
        effects,
      };
    }).filter(Boolean);
  }

  function eventMatchesRule(rule, event, step, action, snapshot, isBackground, context = {}) {
    const trigger = rule.trigger && typeof rule.trigger === 'object' ? rule.trigger : {};
    const matchesPeriodicDamage = event === 'periodic_damage' && trigger.periodic_damage === true;
    if ((String(trigger.event || '') !== event && !matchesPeriodicDamage) || !SUPPORTED_TRIGGER_EVENTS.has(event)) return false;
    const source = trigger.source && typeof trigger.source === 'object' ? trigger.source : {};
    if (!sourceMatches(source, rule, step, action, snapshot, isBackground)) return false;
    const eventContext = Object.assign({}, context, {
      snapshot,
      owner_awakening: rule.owner_awakening,
      owner_awakening_nodes: rule.owner_awakening_nodes,
    });
    return conditionsMatch(trigger.conditions, eventContext);
  }

  function targetMatches(target, instance, step, action, snapshot, isBackground, context = {}) {
    const scope = String(target?.scope || 'registrar');
    const ownerSlot = int(instance.owner_slot);
    const stepSlot = int(step.slot);
    if (scope === 'registrar' && stepSlot !== ownerSlot) return false;
    if (scope === 'team') {
      // fall through
    } else if (scope === 'other_team' && stepSlot === ownerSlot) return false;
    else if (scope === 'front' && isBackground) return false;
    else if (scope === 'front_non_registrar' && (isBackground || stepSlot === ownerSlot)) return false;
    else if (scope === 'front_registrar' && (isBackground || stepSlot !== ownerSlot)) return false;
    else if (scope === 'slots') {
      const slots = new Set(asList(target.slots).map(int));
      if (slots.size && !slots.has(stepSlot)) return false;
    }
    const actionTypes = strSet(target?.action_types);
    if (actionTypes.size && !actionTypes.has(String(action.action_type || ''))) return false;
    const actionIds = strSet(target?.action_ids);
    if (actionIds.size && !actionIds.has(String(action.id || ''))) return false;
    const damageTypes = strSet(target?.damage_types);
    if (damageTypes.size && !damageTypes.has(String(action.damage_type || ''))) return false;
    const actionNames = strSet(target?.action_names);
    if (actionNames.size && !actionNames.has(String(action.name || ''))) return false;
    const tags = strSet(target?.tags);
    if (tags.size && !Array.from(actionTags(action)).some((tag) => tags.has(tag))) return false;
    const placements = strSet(target?.placements);
    if (placements.size && !placements.has(placement(isBackground))) return false;
    const elements = strSet(target?.elements);
    if (elements.size && !elements.has(String(snapshot.character?.element || ''))) return false;
    if (!conditionsMatch(target?.conditions, Object.assign({}, context, { snapshot }))) return false;
    return true;
  }

  function truncateBuffTimelineAt(instance, tick, visualTick = tick) {
    const resetTick = Math.max(int(instance.start_tick), int(tick));
    const resetVisualTick = Math.max(0, int(visualTick, resetTick));
    instance.end_tick = Math.min(int(instance.end_tick), resetTick);
    asList(instance.timeline_summaries).forEach((summary) => {
      if (int(summary.end_tick) <= resetTick) return;
      if (int(tick) <= int(summary.start_tick)) {
        summary.cancelled = true;
      }
      summary.end_tick = resetTick;
      summary.duration_ticks = Math.max(0, resetTick - int(summary.start_tick));
      summary.visual_end_tick = Math.min(int(summary.visual_end_tick, resetVisualTick), resetVisualTick);
      summary.display_end_tick = Math.min(int(summary.display_end_tick, resetVisualTick), resetVisualTick);
    });
  }

  function resetActiveBuffAt(instance, tick, visualTick = tick) {
    truncateBuffTimelineAt(instance, tick, visualTick);
    return true;
  }

  function activeBuffResetsOnActionStart(
    instance,
    step,
    action,
    isBackground,
    tick,
    previousForegroundSlot = null,
    visualTick = tick,
  ) {
    const rule = instance.rule && typeof instance.rule === 'object' ? instance.rule : {};
    const reset = rule.reset && typeof rule.reset === 'object' ? rule.reset : {};
    if (!reset || isBackground) return false;
    const ownerSlot = int(instance.owner_slot);
    const stepSlot = int(step.slot);
    if (reset.owner_foreground && stepSlot === ownerSlot) return resetActiveBuffAt(instance, tick, visualTick);
    if (reset.owner_leaves_foreground && stepSlot !== ownerSlot) return resetActiveBuffAt(instance, tick, visualTick);
    if (reset.owner_leaves_foreground_after_start && stepSlot !== ownerSlot) {
      if (int(tick) < int(instance.start_tick)) return false;
      if (int(previousForegroundSlot, -1) !== ownerSlot) return false;
      return resetActiveBuffAt(instance, tick, visualTick);
    }
    if (strSet(reset.action_ids).has(String(action.id || ''))) return resetActiveBuffAt(instance, tick, visualTick);
    if (strSet(reset.action_names).has(String(action.name || ''))) return resetActiveBuffAt(instance, tick, visualTick);
    if (strSet(reset.action_types).has(String(action.action_type || ''))) return resetActiveBuffAt(instance, tick, visualTick);
    return false;
  }

  function trackBuffTimelineSummary(instance, summary) {
    if (!instance || !summary) return summary;
    if (!Array.isArray(instance.timeline_summaries)) instance.timeline_summaries = [];
    instance.timeline_summaries.push(summary);
    return summary;
  }

  function triggerCooldownTicks(rule) {
    return Math.max(0, int(rule?.trigger?.cooldown_ticks));
  }

  function actionCooldownKey(slot, action) {
    const cooldownIdentity = String(action?.cooldown_group || action?.id || '');
    return `${int(slot)}:${cooldownIdentity}`;
  }

  function stackGainForRule(rule, context = {}) {
    const stacking = rule.stacking && typeof rule.stacking === 'object' ? rule.stacking : {};
    if (stacking.stack_gain != null && stacking.stack_gain !== '') return Math.max(0, num(stacking.stack_gain, 1));
    if (String(stacking.stack_gain_from || '') === 'hit_count') return Math.max(0, num(context.hit_count));
    for (const condition of asList(rule.trigger?.conditions)) {
      if (condition && typeof condition === 'object' && String(condition.type || '') === 'expected_critical_hit') {
        return Math.max(0, num(context.expected_critical_hits));
      }
    }
    return 1;
  }

  function activateBuff(activeBuffs, rule, triggerTick, stackGain = 1, context = {}) {
    const duration = rule.duration && typeof rule.duration === 'object' ? rule.duration : {};
    const isPermanent = String(duration.type || '') === 'permanent';
    const durationTicks = isPermanent ? PERMANENT_BUFF_END_TICK : durationTicksForRule(rule);
    if (durationTicks <= 0) return null;
    const startTick = triggerTick + Math.max(0, int(duration.delay_ticks));
    let endTick = startTick + durationTicks;
    const pauseWhileBuffKey = String(duration.pause_while_buff_key || '');
    if (pauseWhileBuffKey) {
      const pauseUntilTick = activeBuffs
        .filter((instance) => (
          int(instance.owner_slot) === int(rule.owner_slot)
          && String(instance.definition_id || '') === pauseWhileBuffKey
          && startTick >= int(instance.start_tick)
          && startTick < int(instance.end_tick)
        ))
        .reduce((latest, instance) => Math.max(latest, int(instance.end_tick)), startTick);
      endTick += Math.max(0, pauseUntilTick - startTick);
    }
    const stacking = rule.stacking && typeof rule.stacking === 'object' ? rule.stacking : {};
    const mode = String(stacking.mode || 'refresh');
    const maxStacks = maxStacksForRule(rule);
    const gain = Math.max(0, stackGain);
    if (gain <= 0) return null;
    const definitionId = String(stacking.key || rule.id || '');
    const ownerSlot = int(rule.owner_slot);
    for (const instance of activeBuffs) {
      const existingRule = instance.rule && typeof instance.rule === 'object' ? instance.rule : {};
      const existingStacking = existingRule.stacking && typeof existingRule.stacking === 'object' ? existingRule.stacking : {};
      const existingDefinitionId = String(instance.definition_id || existingStacking.key || existingRule.id || '');
      if (existingDefinitionId !== definitionId || int(instance.owner_slot) !== ownerSlot) continue;
      if (mode === 'independent') continue;
      if (mode === 'add_stack') {
        if (stacking.unique_source_slots) {
          const sourceSlot = int(context.source_slot, -1);
          const sourceSlots = new Set(asList(instance.source_slots).map(int));
          if (sourceSlots.has(sourceSlot)) return null;
          sourceSlots.add(sourceSlot);
          instance.source_slots = Array.from(sourceSlots);
        }
        instance.stack_count = Math.min(maxStacks, Math.max(0, num(instance.stack_count, 1)) + gain);
        instance.start_tick = Math.min(int(instance.start_tick, startTick), startTick);
        instance.end_tick = endTick;
        instance.rule = rule;
        instance.name = rule.name || instance.name || '';
        return instance;
      }
      if (mode === 'extend') {
        instance.end_tick = Math.max(int(instance.end_tick), endTick);
        return instance;
      }
      if (mode === 'extend_duration') {
        const extendedEndTick = Math.max(int(instance.end_tick), triggerTick) + durationTicks;
        const maxDurationTicks = Math.max(0, int(stacking.max_duration_ticks));
        instance.end_tick = maxDurationTicks > 0
          ? Math.min(extendedEndTick, triggerTick + maxDurationTicks)
          : extendedEndTick;
        return instance;
      }
      instance.start_tick = startTick;
      instance.end_tick = endTick;
      instance.stack_count = Math.min(maxStacks, Math.max(1, gain));
      instance.excluded_step_id = context.exclude_trigger_action ? String(context.source_step_id || '') : '';
      return instance;
    }
    if (mode === 'independent') {
      const copyCount = Math.max(1, Math.floor(gain));
      let latestInstance = null;
      for (let copyIndex = 0; copyIndex < copyCount; copyIndex += 1) {
        const siblings = activeBuffs
          .filter((instance) => String(instance.definition_id || '') === definitionId && int(instance.owner_slot) === ownerSlot)
          .sort((left, right) => int(left.end_tick) - int(right.end_tick) || int(left.start_tick) - int(right.start_tick));
        while (siblings.length >= Math.floor(maxStacks)) {
          const expiringFirst = siblings.shift();
          const index = activeBuffs.indexOf(expiringFirst);
          if (index >= 0) activeBuffs.splice(index, 1);
        }
        latestInstance = {
          rule,
          definition_id: definitionId,
          name: rule.name || '',
          owner_slot: ownerSlot,
          start_tick: startTick,
          end_tick: endTick,
          stack_count: 1,
          excluded_step_id: context.exclude_trigger_action ? String(context.source_step_id || '') : '',
        };
        if (stacking.unique_source_slots) latestInstance.source_slots = [int(context.source_slot, -1)];
        activeBuffs.push(latestInstance);
      }
      return latestInstance;
    }
    const instance = {
      rule,
      definition_id: definitionId,
      name: rule.name || '',
      owner_slot: ownerSlot,
      start_tick: startTick,
      end_tick: endTick,
      stack_count: Math.min(maxStacks, mode === 'add_stack' ? gain : Math.max(1, gain)),
      excluded_step_id: context.exclude_trigger_action ? String(context.source_step_id || '') : '',
    };
    if (stacking.unique_source_slots) instance.source_slots = [int(context.source_slot, -1)];
    activeBuffs.push(instance);
    return instance;
  }

  function buffEffects(instance, context = {}) {
    const effects = instance.rule?.effects && typeof instance.rule.effects === 'object' ? instance.rule.effects : {};
    const factor = Math.max(0, num(instance.stack_count, 1));
    const resolved = Object.fromEntries(Object.entries(effects).filter(([, value]) => num(value) !== 0).map(([key, value]) => [key, num(value) * factor]));
    const dynamic = instance.rule?.dynamic_effects && typeof instance.rule.dynamic_effects === 'object'
      ? instance.rule.dynamic_effects
      : {};
    const negative = dynamic.negative_effect_count && typeof dynamic.negative_effect_count === 'object'
      ? dynamic.negative_effect_count
      : {};
    if (negative.effect_key) {
      const enemyDebuffs = new Set(Object.keys(context.enemy_debuffs || {}));
      const activeKeys = new Set(asList(context.active_buff_keys).map(String));
      const activePeriodicActionIds = new Set(asList(context.active_periodic_action_ids).map(String));
      const damageTags = new Set(asList(negative.damage_tags).map(String));
      const taggedDamageTypeCount = damageTags.size
        ? new Set(asList(context.active_damage_sources)
          .filter((source) => asList(source?.tags).map(String).some((tag) => damageTags.has(tag)))
          .map((source) => String(source?.type_id || ''))
          .filter(Boolean)).size
        : 0;
      const requiredEnemyDebuffs = asList(negative.requires_enemy_debuffs).map(String);
      const enabled = requiredEnemyDebuffs.every((key) => enemyDebuffs.has(key));
      const count = enabled
        ? Math.min(
          Math.max(0, int(negative.max_count, 1)),
          asList(negative.enemy_debuffs).filter((key) => enemyDebuffs.has(String(key))).length
            + asList(negative.buff_keys).filter((key) => activeKeys.has(String(key))).length
            + asList(negative.periodic_action_ids).filter((key) => activePeriodicActionIds.has(String(key))).length
            + taggedDamageTypeCount,
        )
        : 0;
      resolved[String(negative.effect_key)] = num(resolved[String(negative.effect_key)]) + count * num(negative.per_count);
    }
    const activeStack = dynamic.active_stack_count && typeof dynamic.active_stack_count === 'object'
      ? dynamic.active_stack_count
      : {};
    if (activeStack.effect_key && activeStack.key) {
      const count = Math.min(
        Math.max(0, int(activeStack.max_count, 1)),
        asList(context.active_buffs)
          .filter((buff) => String(buff.definition_id || '') === String(activeStack.key))
          .reduce((sum, buff) => sum + Math.max(0, num(buff.stack_count, 1)), 0),
      );
      resolved[String(activeStack.effect_key)] = num(resolved[String(activeStack.effect_key)]) + count * num(activeStack.per_count);
    }
    const elapsed = dynamic.elapsed_ticks && typeof dynamic.elapsed_ticks === 'object' ? dynamic.elapsed_ticks : {};
    if (elapsed.effect_key) {
      const intervals = Math.max(0, Math.floor((num(context.tick) - int(instance.start_tick)) / Math.max(1, int(elapsed.interval_ticks, 10))));
      resolved[String(elapsed.effect_key)] = num(resolved[String(elapsed.effect_key)])
        + Math.min(num(elapsed.max_value, Number.POSITIVE_INFINITY), intervals * num(elapsed.per_interval));
    }
    const activeDotLayers = dynamic.active_dot_layer_count && typeof dynamic.active_dot_layer_count === 'object'
      ? dynamic.active_dot_layer_count
      : {};
    if (activeDotLayers.effect_key) {
      const count = Math.min(
        Math.max(0, int(activeDotLayers.max_count, 20)),
        Math.max(0, int(context.active_dot_layer_count)),
      );
      let perCount = num(activeDotLayers.per_count);
      if (activeDotLayers.action_last_hit_ratio === true) {
        const action = context.action && typeof context.action === 'object' ? context.action : {};
        const totalAtkMultiplier = Math.max(0, num(action.multipliers?.atk));
        const lastHitAtkMultiplier = Math.max(0, num(action.last_hit_atk_multiplier));
        perCount *= totalAtkMultiplier > 0 ? lastHitAtkMultiplier / totalAtkMultiplier : 0;
      }
      resolved[String(activeDotLayers.effect_key)] = num(resolved[String(activeDotLayers.effect_key)]) + count * perCount;
    }
    return Object.fromEntries(Object.entries(resolved).filter(([, value]) => num(value) !== 0));
  }

  function buffDisplaysAsLine(rule) {
    const display = rule.display && typeof rule.display === 'object' ? rule.display : {};
    if (display.line === false) return false;
    if (display.line === true) return true;
    const duration = rule.duration && typeof rule.duration === 'object' ? rule.duration : {};
    const target = rule.target && typeof rule.target === 'object' ? rule.target : {};
    const scope = String(target.scope || 'registrar');
    if ((scope === 'registrar' || scope === 'front_registrar') && Math.max(0, int(duration.ticks)) <= 1 && Math.max(0, int(duration.delay_ticks)) <= 0) {
      return false;
    }
    return true;
  }

  function buffSummary(instance, context = {}) {
    const rule = instance.rule && typeof instance.rule === 'object' ? instance.rule : {};
    const stacking = rule.stacking && typeof rule.stacking === 'object' ? rule.stacking : {};
    const stackCount = Math.max(0, num(instance.stack_count, 1));
    const displayAsLine = buffDisplaysAsLine(rule);
    return {
      rule_id: rule.id || '',
      definition_id: String(instance.definition_id || stacking.key || rule.id || ''),
      name: rule.name || '',
      provider_name: rule.provider_name || '',
      owner_slot: int(instance.owner_slot),
      start_tick: int(instance.start_tick),
      end_tick: int(instance.end_tick),
      duration_ticks: String(rule.duration?.type || '') === 'permanent'
        ? PERMANENT_BUFF_END_TICK
        : Math.max(0, int(instance.end_tick) - int(instance.start_tick)),
      stacking_mode: String(stacking.mode || 'refresh'),
      max_stacks: maxStacksForRule(rule),
      stack_count: Number.isInteger(stackCount) ? Math.trunc(stackCount) : stackCount,
      effects: buffEffects(instance, context),
      display_as_line: displayAsLine,
      line_hidden_reason: displayAsLine ? '' : (String(rule.duration?.type || '') === 'permanent' ? 'passive' : 'self_action'),
    };
  }

  function activeBuffApplies(instance, step, action, snapshot, isBackground, context = {}) {
    if (instance.excluded_step_id && String(instance.excluded_step_id) === String(step?.id || '')) {
      return false;
    }
    const target = instance.rule?.target && typeof instance.rule.target === 'object' ? instance.rule.target : {};
    return targetMatches(target, instance, step, action, snapshot, isBackground, context);
  }

  function applicableBuffContributions(activeBuffs, step, action, snapshot, isBackground, context = {}) {
    const contributions = activeBuffs
      .filter((buff) => (
        int(context.tick) >= int(buff.start_tick)
        && int(context.tick) < int(buff.end_tick)
        && activeBuffApplies(buff, step, action, snapshot, isBackground, context)
      ))
      .map((buff) => {
        const effects = buffEffects(buff, context);
        return {
          buff,
          effects,
          uniqueKey: String(buff.rule?.calculation?.team_unique_key || ''),
          magnitude: Object.values(effects).reduce((sum, value) => sum + Math.max(0, num(value)), 0),
        };
      });
    const selected = [];
    const highestByUniqueKey = new Map();
    contributions.forEach((contribution) => {
      if (!contribution.uniqueKey) {
        selected.push(contribution);
        return;
      }
      const current = highestByUniqueKey.get(contribution.uniqueKey);
      if (!current || contribution.magnitude > current.magnitude) {
        highestByUniqueKey.set(contribution.uniqueKey, contribution);
      }
    });
    selected.push(...highestByUniqueKey.values());
    return selected;
  }

  function markQInstantReleaseTarget(scheduled, qVisualStartTick, qCalculationStartTick, qStepId, releaseKind, startSequence, endSequence, collapseToQTick = false) {
    const startTick = int(scheduled.start_tick);
    const durationTicks = Math.max(0, int(scheduled.duration_ticks));
    const endTick = int(scheduled.end_tick);
    const visualEndTick = int(scheduled.visual_end_tick, endTick);
    scheduled.original_duration_ticks ??= durationTicks;
    scheduled.original_end_tick ??= endTick;
    scheduled.original_visual_end_tick ??= visualEndTick;
    scheduled.original_start_tick ??= startTick;
    scheduled.original_calculation_start_sequence ??= int(scheduled.calculation_start_sequence);
    scheduled.original_calculation_end_sequence ??= int(scheduled.calculation_end_sequence);
    const visualStartTick = int(scheduled.visual_start_tick, startTick);
    const originalVisualEndTick = Math.max(visualStartTick, int(scheduled.original_visual_end_tick, visualEndTick));
    scheduled.calculation_start_sequence = Math.max(0, int(startSequence));
    scheduled.calculation_end_sequence = Math.max(scheduled.calculation_start_sequence + 1, int(endSequence));
    scheduled.q_instant_release = true;
    scheduled.q_instant_release_kind = releaseKind;
    scheduled.q_instant_release_tick = qVisualStartTick;
    scheduled.q_instant_release_anchor_tick = qVisualStartTick;
    scheduled.q_instant_release_calculation_tick = qCalculationStartTick;
    scheduled.q_instant_release_anchor_step_id = qStepId;
    scheduled.q_instant_release_start_sequence = scheduled.calculation_start_sequence;
    scheduled.q_instant_release_end_sequence = scheduled.calculation_end_sequence;
    return originalVisualEndTick;
  }

  function applyQInstantRelease(scheduledSteps) {
    const qEvents = scheduledSteps
      .filter((scheduled) => isZeroForegroundQStep(scheduled.step || {}, scheduled.action || {}))
      .sort((a, b) => int(a.visual_start_tick, int(a.start_tick)) - int(b.visual_start_tick, int(b.start_tick)) || int(a.slot) - int(b.slot));
    let qVirtualIntervals = [];
    qEvents.forEach((qEvent) => {
      const qStartTick = int(qEvent.visual_start_tick, int(qEvent.start_tick));
      const qCalculationStartTick = calculationTickFromVisualIntervals(qStartTick, qVirtualIntervals);
      const qStepId = String(qEvent.step?.id || '');
      const qSlot = int(qEvent.slot);
      const releaseSlotVisualEnds = new Map();
      const releaseSlotSequences = new Map();
      const coverTargetStepIds = [];
      scheduledSteps.forEach((scheduled) => {
        if (scheduled === qEvent || scheduled.q_instant_release) return;
        if (isQCoverImmuneScheduled(scheduled)) return;
        const startTick = int(scheduled.visual_start_tick, int(scheduled.start_tick));
        const endTick = int(scheduled.visual_end_tick, int(scheduled.end_tick));
        const inQColumn = int(scheduled.slot) !== qSlot && startTick < qStartTick && qStartTick < endTick;
        const ongoingForeground = !scheduled.is_background && startTick < qStartTick && qStartTick < endTick;
        if (!inQColumn && !ongoingForeground) return;
        const targetSlot = int(scheduled.slot);
        const startSequence = releaseSlotSequences.get(targetSlot) || 0;
        const endSequence = startSequence + 1;
        const visualEnd = markQInstantReleaseTarget(
          scheduled,
          qStartTick,
          qCalculationStartTick,
          qStepId,
          inQColumn ? 'column' : 'foreground',
          startSequence,
          endSequence,
        );
        releaseSlotVisualEnds.set(targetSlot, Math.max(releaseSlotVisualEnds.get(targetSlot) || 0, visualEnd));
        releaseSlotSequences.set(targetSlot, endSequence);
        coverTargetStepIds.push(String(scheduled.step?.id || ''));
      });
      let changed = true;
      while (changed) {
        changed = false;
        scheduledSteps.forEach((scheduled) => {
          if (scheduled === qEvent || scheduled.q_instant_release || !scheduled.can_background_override) return;
          const slot = int(scheduled.slot);
          const coveredUntil = releaseSlotVisualEnds.get(slot);
          const visualStartTick = int(scheduled.visual_start_tick, int(scheduled.start_tick));
          if (coveredUntil == null || visualStartTick < qStartTick || visualStartTick > coveredUntil) return;
          const startSequence = releaseSlotSequences.get(slot) || 0;
          const endSequence = startSequence + 1;
          const visualEnd = markQInstantReleaseTarget(
            scheduled,
            qStartTick,
            qCalculationStartTick,
            qStepId,
            'basic-background',
            startSequence,
            endSequence,
            true,
          );
          releaseSlotVisualEnds.set(slot, Math.max(releaseSlotVisualEnds.get(slot) || 0, visualEnd));
          releaseSlotSequences.set(slot, endSequence);
          coverTargetStepIds.push(String(scheduled.step?.id || ''));
          changed = true;
        });
      }
      const coverVisualEndTick = Math.max(
        int(qEvent.visual_end_tick, qStartTick + qVisualDurationTicks(qEvent.action || {})),
        ...Array.from(releaseSlotVisualEnds.values()).map((tick) => int(tick)),
      );
      if (coverVisualEndTick > int(qEvent.visual_end_tick, qStartTick)) {
        qEvent.original_visual_end_tick ??= int(qEvent.visual_end_tick, qStartTick + qVisualDurationTicks(qEvent.action || {}));
        qEvent.visual_end_tick = coverVisualEndTick;
        qEvent.q_cover_visual_end_tick = coverVisualEndTick;
        qEvent.q_cover_target_step_ids = Array.from(new Set(coverTargetStepIds.filter(Boolean)));
      }
      qVirtualIntervals = normalizeFrozenIntervals(qVirtualIntervals.concat([{
        start_tick: qStartTick,
        end_tick: Math.max(qStartTick + ZERO_ACTION_VISUAL_TICKS, int(qEvent.visual_end_tick, qStartTick + ZERO_ACTION_VISUAL_TICKS)),
      }]));
    });
    recalculateTimingsFromFrozenIntervals(scheduledSteps, qVirtualIntervals);
    scheduledSteps.forEach((scheduled) => {
      if (!scheduled.q_instant_release) return;
      const qCalculationTick = calculationTickFromVisualIntervals(
        int(scheduled.q_instant_release_anchor_tick),
        qVirtualIntervals,
      );
      scheduled.q_instant_release_calculation_tick = qCalculationTick;
    });
    return qVirtualIntervals;
  }

  function clearQInstantReleaseState(scheduledSteps) {
    scheduledSteps.forEach((scheduled) => {
      [
        'q_instant_release_kind',
        'q_instant_release_tick',
        'q_instant_release_anchor_tick',
        'q_instant_release_calculation_tick',
        'q_instant_release_anchor_step_id',
        'q_instant_release_start_sequence',
        'q_instant_release_end_sequence',
        'q_cover_visual_end_tick',
        'q_cover_target_step_ids',
      ].forEach((key) => delete scheduled[key]);
      scheduled.q_instant_release = false;
      scheduled.calculation_start_sequence = 0;
      scheduled.calculation_end_sequence = 0;
      if (isZeroForegroundQStep(scheduled.step || {}, scheduled.action || {})) {
        scheduled.visual_end_tick = int(scheduled.visual_start_tick) + qVisualDurationTicks(scheduled.action || {});
      }
    });
  }

  function enforceExpandedForegroundLocks(scheduledSteps, switchGapTicks) {
    let shiftedAny = false;
    const foregroundLocks = [];
    let previousForegroundSlot = null;
    let previousForegroundStartTick = null;
    const carriedShiftBySlot = new Map();
    const ordered = scheduledSteps.slice().sort((left, right) => {
      const startDelta = int(left.visual_start_tick) - int(right.visual_start_tick);
      if (startDelta) return startDelta;
      const instantSwitchDelta = Number(isInstantSwitchAction(right.action || {})) -
        Number(isInstantSwitchAction(left.action || {}));
      if (instantSwitchDelta) return instantSwitchDelta;
      const lockDelta = Number(locksForegroundSwitch(right.step || {}, right.action || {})) -
        Number(locksForegroundSwitch(left.step || {}, left.action || {}));
      return lockDelta || int(left.slot) - int(right.slot);
    });
    ordered.forEach((scheduled) => {
      const slot = int(scheduled.slot);
      const carriedShiftTicks = Math.max(0, int(carriedShiftBySlot.get(slot)));
      if (carriedShiftTicks > 0) {
        scheduled.visual_start_tick = int(scheduled.visual_start_tick) + carriedShiftTicks;
        scheduled.visual_end_tick = int(scheduled.visual_end_tick) + carriedShiftTicks;
        shiftedAny = true;
      }
      if (scheduled.is_background || scheduled.q_instant_release) return;
      const originalStartTick = int(scheduled.visual_start_tick);
      let visualStartTick = originalStartTick;
      let lockEndTick = Math.max(
        0,
        ...foregroundLocks
          .filter((lock) => visualStartTick < int(lock.end_tick))
          .map((lock) => int(lock.end_tick)),
      );
      while (lockEndTick > visualStartTick) {
        visualStartTick = lockEndTick;
        lockEndTick = Math.max(
          0,
          ...foregroundLocks
            .filter((lock) => visualStartTick < int(lock.end_tick))
            .map((lock) => int(lock.end_tick)),
        );
      }
      if (
        previousForegroundSlot !== null &&
        previousForegroundSlot !== int(scheduled.slot) &&
        previousForegroundStartTick !== null &&
        !isInstantSwitchAction(scheduled.action || {})
      ) {
        visualStartTick = Math.max(
          visualStartTick,
          previousForegroundStartTick + Math.max(0, int(switchGapTicks)),
        );
      }
      if (visualStartTick > originalStartTick) {
        const shiftTicks = visualStartTick - originalStartTick;
        scheduled.visual_start_tick = visualStartTick;
        scheduled.visual_end_tick = int(scheduled.visual_end_tick) + shiftTicks;
        scheduled.foreground_lock_ticks = Math.max(0, int(scheduled.foreground_lock_ticks)) + shiftTicks;
        carriedShiftBySlot.set(slot, Math.max(0, int(carriedShiftBySlot.get(slot))) + shiftTicks);
        shiftedAny = true;
      }
      previousForegroundSlot = int(scheduled.slot);
      previousForegroundStartTick = int(scheduled.visual_start_tick);
      if (locksForegroundSwitch(scheduled.step || {}, scheduled.action || {})) {
        foregroundLocks.push({
          slot: int(scheduled.slot),
          end_tick: isZeroForegroundQStep(scheduled.step || {}, scheduled.action || {})
            ? int(scheduled.visual_end_tick)
            : int(scheduled.visual_start_tick) + qVisualDurationTicks(scheduled.action || {}),
        });
      }
    });
    return shiftedAny;
  }

  function resourceMap(raw) {
    if (!raw || typeof raw !== 'object') return {};
    return Object.fromEntries(Object.entries(raw).filter(([, value]) => num(value) !== 0).map(([key, value]) => [String(key), num(value)]));
  }

  function simulateAxis(axisPayload, catalog) {
    if (!catalog || typeof catalog !== 'object') {
      throw new Error('缺少排轴数据目录。');
    }
    const actionsById = recordMap(catalog.actions);
    const teamPayload = asList(axisPayload?.team).length ? asList(axisPayload.team) : asList(catalog.starter_axis?.team);
    const steps = asList(axisPayload?.steps).length ? asList(axisPayload.steps) : asList(catalog.starter_axis?.steps);
    validateSteps(steps, actionsById);

    const teamPanelBonus = normalizeTeamPanelBonus(axisPayload?.team_panel_bonus, catalog);
    const snapshots = new Map(teamPayload.map((member) => {
      const snapshot = buildSnapshot(member, catalog, teamPanelBonus);
      return [snapshot.slot, snapshot];
    }));
    const enemy = normalizeEnemy(axisPayload?.enemy);
    let enemyDebuffs = Object.assign({}, enemy.debuffs || {});
    const options = axisPayload?.options && typeof axisPayload.options === 'object' ? axisPayload.options : {};
    const personalResourceCaps = catalog.formula_constants?.personal_resource_caps
      && typeof catalog.formula_constants.personal_resource_caps === 'object'
      ? catalog.formula_constants.personal_resource_caps
      : DEFAULT_PERSONAL_RESOURCE_CAPS;
    const fonsFull = options.fons_full == null ? true : Boolean(options.fons_full);
    const switchLossTicks = Math.max(
      0,
      int(options.switch_gap_ticks, int(options.switch_loss_ticks, int(catalog.formula_constants?.switch_loss_ticks, 2))),
    );
    const details = [];
    const frontEvents = [];
    const healingEvents = [];
    let directDamage = 0;
    let staggerDamage = 0;
    let totalStagger = 0;
    const staggerProfileSamplesBySlot = new Map(Array.from(snapshots.keys()).map((slot) => [slot, []]));
    const specialDamageBySource = new Map(SPECIAL_DAMAGE_SOURCES.map((source) => [source, 0]));
    const requestedInitialEnergy = Math.max(0, num(axisPayload?.initial_energy, 1000));
    const loopInitialResources = options.loop_enabled && options.loop_initial_resources && typeof options.loop_initial_resources === 'object'
      ? options.loop_initial_resources
      : {};
    const energyCapacityBySlot = new Map(Array.from(snapshots.entries()).map(([slot, snapshot]) => {
      if (snapshot.character?.uses_energy === false) return [slot, 0];
      const configuredCapacity = num(snapshot.character?.energy_capacity);
      const inferredCapacity = Math.max(0, ...Array.from(actionsById.values())
        .filter((action) => String(action?.character_id || '') === String(snapshot.character?.id || '') && isQAction(action))
        .map((action) => num(action?.energy_cost)));
      return [slot, Math.max(0, configuredCapacity || inferredCapacity)];
    }));
    const initialEnergyBySlot = new Map(Array.from(snapshots.entries()).map(([slot, snapshot]) => {
      if (snapshot.character?.uses_energy === false) return [slot, 0];
      const capacity = energyCapacityBySlot.get(slot) || 0;
      const characterResources = loopInitialResources[String(snapshot.character?.id || '')];
      const configuredEnergy = characterResources && typeof characterResources === 'object'
        ? Math.max(0, num(characterResources.energy, requestedInitialEnergy))
        : requestedInitialEnergy;
      return [slot, capacity > 0 ? Math.min(configuredEnergy, capacity) : configuredEnergy];
    }));
    const energyBySlot = new Map(initialEnergyBySlot);
    const energyEvents = [];
    let pendingActionEnergy = [];
    let lastEnergySettlementTick = 0;
    const zhenhongAscendantBySlot = new Map(Array.from(snapshots.keys()).map((slot) => [slot, false]));
    function energyCapacityForSlot(slot) {
      const snapshot = snapshots.get(slot);
      if (
        String(snapshot?.character?.id || '') === ZHENHONG_CHARACTER_ID
        && zhenhongAscendantBySlot.get(slot)
      ) {
        return ZHENHONG_ASCENDANT_ENERGY_CAPACITY;
      }
      return energyCapacityBySlot.get(slot) || 0;
    }
    const initialHarmonyBySlot = new Map(Array.from(snapshots.entries()).map(([slot, snapshot]) => {
      const characterResources = loopInitialResources[String(snapshot.character?.id || '')];
      const harmony = options.loop_enabled
        ? (characterResources && typeof characterResources === 'object' ? num(characterResources.harmony) : 0)
        : num(snapshot.character?.initial_harmony_non_loop);
      return [slot, Math.max(0, Math.min(HARMONY_CAPACITY, harmony))];
    }));
    const harmonyBySlot = new Map(initialHarmonyBySlot);
    const cooldownUntil = new Map();
    const forcedStaggerTargets = new Set();
    const periodicHealingStates = new Map();
    const personalResources = new Map(Array.from(snapshots.keys()).map((slot) => [slot, {}]));
    const initialPersonalResourcesBySlot = new Map();
    const initialPersonalResources = axisPayload?.initial_personal_resources && typeof axisPayload.initial_personal_resources === 'object'
      ? axisPayload.initial_personal_resources
      : {};
    personalResources.forEach((resources, slot) => {
      Object.assign(resources, resourceMap(initialPersonalResources[String(slot)] || initialPersonalResources[slot] || {}));
      const snapshot = snapshots.get(slot);
      const characterId = String(snapshot?.character?.id || '');
      const loopResources = loopInitialResources[characterId];
      const configuredPersonal = loopResources?.personal_resources && typeof loopResources.personal_resources === 'object'
        ? resourceMap(loopResources.personal_resources)
        : {};
      Object.entries(configuredPersonal).forEach(([name, value]) => {
        const cap = num(personalResourceCaps[characterId]?.[name], 1000000);
        resources[name] = Math.max(0, Math.min(cap, value));
      });
      initialPersonalResourcesBySlot.set(slot, Object.assign({}, resources));
    });
    const buffRules = registeredBuffRules(teamPayload, catalog).concat(legacyBuffRules(axisPayload?.buff_rules));
    function syncBuffLayerResources(tick) {
      const bindings = new Map();
      buffRules.forEach((rule) => {
        const resource = String(rule.periodic_damage?.layer_resource || '');
        if (!resource) return;
        const ownerSlot = int(rule.owner_slot);
        const definitionId = String(rule.stacking?.key || rule.id || '');
        bindings.set(`${ownerSlot}:${definitionId}:${resource}`, {ownerSlot, definitionId, resource});
      });
      bindings.forEach(({ownerSlot, definitionId, resource}) => {
        const stackCount = activeBuffs
          .filter((instance) => (
            int(instance.owner_slot) === ownerSlot
            && String(instance.definition_id || '') === definitionId
            && int(tick) >= int(instance.start_tick)
            && int(tick) < int(instance.end_tick)
          ))
          .reduce((sum, instance) => sum + Math.max(0, num(instance.stack_count, 1)), 0);
        const resources = personalResources.get(ownerSlot) || {};
        if (stackCount > 0) resources[resource] = stackCount;
        else delete resources[resource];
      });
    }
    const orderedStepEntries = tickScheduledStepEntries(steps, actionsById, switchLossTicks);
    const scheduledSteps = [];
    const loopOpeningFrontSlot = options.loop_enabled
      ? orderedStepEntries
        .filter(({ step }) => startsForeground(step, actionsById.get(String(step.action_id || '')) || {}))
        .map(({ step }) => int(step.slot))
        .at(-1) ?? null
      : null;
    let scheduleFrontSlot = loopOpeningFrontSlot;
    orderedStepEntries.forEach(({
      step,
      visualStartTick: scheduledVisualStartTick,
      switchLossTicks: switchLossDelta,
      foregroundLockTicks,
    }) => {
      const slot = int(step.slot);
      const snapshot = snapshots.get(slot);
      if (!snapshot) return;
      const action = actionsById.get(String(step.action_id || ''));
      const visualStartTick = scheduledVisualStartTick;
      const isBackground = isStepBackground(step, action);
      const calculationAtStartOnly = isInstantNativeBackgroundAction(step, action) || isInstantSwitchAction(action);
      const configuredDurationTicks = actionCalculationDurationTicks(step, action);
      const visualDurationTicks = actionVisualDurationTicks(step, action);
      const visualEndTick = visualStartTick + visualDurationTicks;
      scheduledSteps.push({
        step,
        slot,
        action,
        previous_front_slot: scheduleFrontSlot,
        start_tick: visualStartTick,
        calculation_start_sequence: 0,
        visual_start_tick: visualStartTick,
        switch_loss_ticks: switchLossDelta,
        foreground_lock_ticks: Math.max(0, int(foregroundLockTicks)),
        is_background: isBackground,
        is_basic_background: isBasicBackgroundOverride(step, action),
        can_background_override: canBackgroundOverride(action),
        calculation_at_start_only: calculationAtStartOnly,
        duration_ticks: configuredDurationTicks,
        end_tick: visualEndTick,
        calculation_end_sequence: 0,
        visual_end_tick: visualEndTick,
        original_start_tick: visualStartTick,
        original_calculation_start_sequence: 0,
        original_duration_ticks: configuredDurationTicks,
        original_end_tick: visualStartTick + configuredDurationTicks,
        original_calculation_end_sequence: 0,
        original_visual_end_tick: visualEndTick,
      });
      if (!isBackground) {
        scheduleFrontSlot = slot;
      }
    });
    let qVirtualIntervals = applyQInstantRelease(scheduledSteps);
    for (let pass = 0; pass < scheduledSteps.length; pass += 1) {
      if (!enforceExpandedForegroundLocks(scheduledSteps, switchLossTicks)) break;
      clearQInstantReleaseState(scheduledSteps);
      qVirtualIntervals = applyQInstantRelease(scheduledSteps);
    }
    scheduledSteps.sort((left, right) => (
      int(left.visual_start_tick) - int(right.visual_start_tick) ||
      int(left.slot) - int(right.slot)
    ));
    scheduleFrontSlot = loopOpeningFrontSlot;
    scheduledSteps.forEach((scheduled) => {
      scheduled.previous_front_slot = scheduleFrontSlot;
      if (!scheduled.is_background) scheduleFrontSlot = int(scheduled.slot);
    });
    const scheduledLastTick = Math.max(
      0,
      ...scheduledSteps
        .filter((scheduled) => !scheduled.is_background)
        .map((scheduled) => Math.max(int(scheduled.end_tick), int(scheduled.start_tick))),
    );
    const loopDurationTicks = Math.max(scheduledLastTick, 1);
    const loopPrimedReactionStepIds = new Set();

    function supportBypassesHarmony(scheduled) {
      return Boolean(
        scheduled?.step?.ignore_harmony_requirement ||
        scheduled?.action?.ignore_harmony_requirement ||
        scheduled?.action?.reaction_without_harmony ||
        requiemFreeSupportSource(scheduled)
      );
    }

    function requiemFreeSupportSource(scheduled) {
      if (!isSupportAction(scheduled?.action)) return null;
      const snapshot = snapshots.get(int(scheduled?.slot));
      if (String(snapshot?.character?.id || '') !== 'char_c78f7a08d5' || !hasAwakeningNode(snapshot, 6)) return null;
      const supportTick = int(scheduled?.start_tick);
      return scheduledSteps
        .filter((candidate) => (
          int(candidate?.slot) === int(scheduled?.slot)
          && ['action_2745f804a5', 'action_7af75245df', 'action_0b958faf88'].includes(String(candidate?.action?.id || ''))
          && int(candidate?.start_tick) <= supportTick
          && supportTick - int(candidate?.start_tick) <= 50
        ))
        .sort((left, right) => int(right?.start_tick) - int(left?.start_tick))[0] || null;
    }

    function reactionPreviousSlot(scheduled) {
      return int(requiemFreeSupportSource(scheduled)?.previous_front_slot, int(scheduled?.previous_front_slot, -1));
    }

    function reactionTriggerTick(scheduled, offsetTicks = 0) {
      const visualStartTick = int(scheduled?.visual_start_tick, int(scheduled?.start_tick));
      const visualEndTick = int(scheduled?.visual_end_tick, int(scheduled?.end_tick, visualStartTick));
      return Math.max(visualStartTick, visualEndTick - 2) + int(offsetTicks);
    }

    if (options.loop_enabled) {
      const warmHarmony = new Map(initialHarmonyBySlot);
      scheduledSteps.forEach((scheduled) => {
        const slot = int(scheduled.slot);
        warmHarmony.set(
          slot,
          Math.min(
            HARMONY_CAPACITY,
            (warmHarmony.get(slot) || 0) + num(scheduled.action?.harmony) * backgroundActionMultiplier(scheduled.step, scheduled.action),
          ),
        );
        if (!isSupportAction(scheduled.action)) return;
        const previousSnapshot = snapshots.get(reactionPreviousSlot(scheduled));
        const supportSnapshot = snapshots.get(slot);
        if (!previousSnapshot || !supportSnapshot || previousSnapshot.slot === supportSnapshot.slot) return;
        if (!reactionForElements(previousSnapshot.character?.element, supportSnapshot.character?.element)) return;
        const previousHarmony = warmHarmony.get(previousSnapshot.slot) || 0;
        if (previousHarmony < 100 && !supportBypassesHarmony(scheduled)) return;
        loopPrimedReactionStepIds.add(String(scheduled.step?.id || ''));
        if (!scheduled.action?.preserve_harmony && !scheduled.step?.preserve_harmony) {
          warmHarmony.set(previousSnapshot.slot, Math.max(0, previousHarmony - 100));
        }
      });
    }
    let activeBuffs = [];
    const buffTriggerCooldowns = new Map();
    const reactionEffects = [];
    const reactionDamageEvents = [];
    const sharedPeriodicDamageStates = new Map();
    let nextSharedPeriodicDamageGeneration = 1;
    const reactionDamageBySlot = new Map(Array.from(snapshots.keys()).map((slot) => [slot, 0]));
    const periodicDamageWindows = [];
    let nextReactionEffectId = 1;
    let nextPeriodicDamageWindowId = 1;

    scheduledSteps.forEach((scheduled) => {
      const periodic = scheduled.action?.periodic_damage && typeof scheduled.action.periodic_damage === 'object'
        ? scheduled.action.periodic_damage
        : {};
      const intervalTicks = Math.max(1, int(periodic.interval_ticks));
      const tickCount = Math.max(0, int(periodic.tick_count));
      if (!tickCount || !Object.keys(periodic.multipliers || {}).length) return;
      const startTick = int(scheduled.start_tick);
      const endTick = startTick + intervalTicks * tickCount;
      const periodicWindow = {
        id: `action_periodic_${nextPeriodicDamageWindowId}`,
        step_id: String(scheduled.step?.id || ''),
        action_id: String(scheduled.action.id || ''),
        source: String(scheduled.action.name || '持续伤害'),
        contributor_slot: int(scheduled.slot),
        start_tick: startTick,
        end_tick: endTick,
        duration_ticks: endTick - startTick,
        interval_ticks: intervalTicks,
        tick_count: tickCount,
        stack_count: 1,
        max_stacks: Math.max(1, int(periodic.max_stacks, 10)),
        damage_tags: Array.from(actionTags(scheduled.action)).sort(),
        action_type: String(scheduled.action.action_type || '周期伤害'),
        damage_type: String(scheduled.action.damage_type || scheduled.action.action_type || '周期伤害'),
        multipliers: clone(periodic.multipliers || {}),
        activated: false,
      };
      nextPeriodicDamageWindowId += 1;
      periodicDamageWindows.push(periodicWindow);
      for (let index = 0; index < tickCount; index += 1) {
        reactionDamageEvents.push({
          kind: 'action_periodic',
          reaction: String(scheduled.action.name || '持续伤害'),
          tick: startTick + intervalTicks * (index + 1),
          sequence: index + 1,
          contributor_slot: int(scheduled.slot),
          contributor_character_id: String(scheduled.action.character_id || ''),
          contributor_character_name: String(scheduled.action.character_name || ''),
          action_id: String(scheduled.action.id || ''),
          action_type: String(scheduled.action.action_type || '周期伤害'),
          damage_type: String(scheduled.action.damage_type || scheduled.action.action_type || '周期伤害'),
          multipliers: clone(periodic.multipliers || {}),
          _periodic_window_id: periodicWindow.id,
          damage: null,
        });
      }
    });

    function activePeriodicActionIds(tick) {
      return periodicDamageWindows
        .filter((window) => window.activated && tick >= int(window.start_tick) && tick <= int(window.end_tick))
        .map((window) => String(window.action_id || ''));
    }

    function activeDamageSources(tick) {
      const sources = new Map();
      activeBuffs.forEach((instance) => {
        if (tick < int(instance.start_tick) || tick >= int(instance.end_tick)) return;
        const periodic = instance.rule?.periodic_damage && typeof instance.rule.periodic_damage === 'object'
          ? instance.rule.periodic_damage
          : null;
        if (!periodic) return;
        const tags = Array.from(actionTags({extra_tag: periodic.extra_tag, tags: periodic.tags})).sort();
        const typeId = String(periodic.source || instance.name || instance.definition_id || '');
        if (typeId) sources.set(`buff:${typeId}`, {type_id: `buff:${typeId}`, tags});
      });
      periodicDamageWindows.forEach((window) => {
        if (!window.activated || tick < int(window.start_tick) || tick > int(window.end_tick)) return;
        const typeId = String(window.action_id || '');
        if (typeId) sources.set(`action:${typeId}`, {type_id: `action:${typeId}`, tags: asList(window.damage_tags).map(String)});
      });
      reactionEffects.forEach((effect) => {
        if (tick < int(effect.start_tick) || tick >= int(effect.end_tick)) return;
        const taggedEvent = reactionDamageEvents.find((event) => String(event.effect_id || '') === String(effect.id || ''));
        const tags = Array.from(actionTags(taggedEvent || {})).sort();
        const typeId = String(effect.reaction || '');
        if (typeId) sources.set(`reaction:${typeId}`, {type_id: `reaction:${typeId}`, tags});
      });
      return Array.from(sources.values());
    }

    function effectiveReactionPanel(snapshot, tick, damageAction = null) {
      const panelMods = clone(snapshot.mods);
      const syntheticStep = { slot: snapshot.slot };
      const syntheticAction = damageAction || {
        name: '异能环合',
        action_type: '环合',
        damage_type: '环合',
        tags: ['环合'],
      };
      const activeAtTick = activeBuffs.filter((candidate) => tick >= int(candidate.start_tick) && tick < int(candidate.end_tick));
      const context = {
        enemy,
        tick,
        enemy_debuffs: activeEnemyDebuffs(enemyDebuffs, tick),
        active_buffs: activeAtTick,
        active_buff_keys: activeAtTick.map((candidate) => String(candidate.definition_id || '')),
        active_periodic_action_ids: activePeriodicActionIds(tick),
        active_damage_sources: activeDamageSources(tick),
      };
      applicableBuffContributions(activeBuffs, syntheticStep, syntheticAction, snapshot, false, context)
        .forEach((contribution) => mergeMods(panelMods, contribution.effects));
      return panelMods;
    }

    function reactionContributor(reaction, previousSnapshot, supportSnapshot, tick) {
      const candidates = [previousSnapshot, supportSnapshot].filter(Boolean);
      if (!candidates.length) return null;
      let selected = candidates[candidates.length - 1];
      let selectedScore = -1;
      candidates.forEach((snapshot) => {
        const panelMods = effectiveReactionPanel(snapshot, tick);
        const score = reactionBaseDamage(snapshot.character?.level, reaction) *
          reactionStrengthMultiplier(panelMods.harmony_strength);
        if (score > selectedScore || (score === selectedScore && snapshot.slot === supportSnapshot?.slot)) {
          selected = snapshot;
          selectedScore = score;
        }
      });
      return selected;
    }

    function teamHasNanali() {
      return Array.from(snapshots.values()).some((snapshot) => String(snapshot?.character?.id || '') === NANALI_CHARACTER_ID);
    }

    function teamHasJiuyuan() {
      return Array.from(snapshots.values()).some((snapshot) => String(snapshot?.character?.id || '') === JIUYUAN_CHARACTER_ID);
    }

    function evenlySpacedDamageTicks(startTick, durationTicks, count) {
      const safeCount = Math.max(0, int(count));
      const safeDuration = Math.max(0, int(durationTicks));
      return Array.from(
        { length: safeCount },
        (_, index) => startTick + Math.round((index + 1) * safeDuration / safeCount),
      );
    }

    function iloySnapshot() {
      return Array.from(snapshots.values()).find(
        (snapshot) => String(snapshot?.character?.id || '') === ILOY_CHARACTER_ID,
      ) || null;
    }

    function reactionDamageTicks(reaction, startTick) {
      if (reaction === '创生') {
        const baseFlowerCount = teamHasNanali() ? 10 : 5;
        return evenlySpacedDamageTicks(
          startTick,
          REACTION_DURATIONS['创生'],
          baseFlowerCount,
        );
      }
      if (reaction === '浊燃') return Array.from({ length: 15 }, (_, index) => startTick + (index + 1) * 10);
      if (reaction === '黯星') return [startTick + 50];
      return [];
    }

    function carriedReactionEffectAtTick(reaction, tick) {
      return reactionEffects.find((effect) => (
        String(effect.reaction || '') === String(reaction || '')
        && !effect.source_reaction
        && effect.disabled !== true
        && (effect.loop_primed === true || effect.looped === true)
        && int(effect.start_tick) <= int(tick)
        && int(tick) < int(effect.end_tick)
      )) || null;
    }

    function refreshCarriedReactionEffect(effect, reaction, tick, source, primeLoop = false) {
      const durationTicks = int(REACTION_DURATIONS[reaction]);
      const retainedDamageTicks = asList(effect.damage_ticks)
        .map((damageTick) => int(damageTick))
        .filter((damageTick) => damageTick <= int(tick));
      const refreshedDamageTicks = reactionDamageTicks(reaction, tick);
      for (let index = reactionDamageEvents.length - 1; index >= 0; index -= 1) {
        const event = reactionDamageEvents[index];
        if (
          String(event.effect_id || '') === String(effect.id || '')
          && int(event.tick) > int(tick)
        ) {
          reactionDamageEvents.splice(index, 1);
        }
      }
      Object.assign(effect, source, {
        end_tick: int(tick) + durationTicks,
        duration_ticks: int(tick) + durationTicks - int(effect.start_tick),
        damage_ticks: retainedDamageTicks.concat(refreshedDamageTicks),
        refreshed_in_loop: true,
      });
      refreshedDamageTicks.forEach((damageTick, index) => {
        const isDot = reaction === '浊燃';
        reactionDamageEvents.push({
          effect_id: effect.id,
          reaction,
          tick: damageTick,
          sequence: retainedDamageTicks.length + index + 1,
          trigger_slot: effect.trigger_slot,
          contributor_slot: effect.contributor_slot,
          contributor_character_id: effect.contributor_character_id,
          contributor_character_name: effect.contributor_character_name,
          extra_tag: isDot ? 'DOT' : '',
          tags: isDot ? ['DOT'] : [],
          frequency_multiplier: effect.frequency_multiplier,
          loop_primed: primeLoop,
          damage: null,
        });
      });
      enemyDebuffs[reaction] = Math.max(int(enemyDebuffs[reaction]), int(effect.end_tick) + 1);
      return effect;
    }

    function canReceiveEnergy(snapshot, currentFrontSlot) {
      if (String(snapshot?.character?.id || '') !== ZHENHONG_CHARACTER_ID) return true;
      return int(snapshot?.slot) === int(currentFrontSlot);
    }

    function energyRechargeForRecipient(snapshot, tick, currentFrontSlot) {
      const panelMods = clone(snapshot.mods);
      const syntheticStep = { slot: snapshot.slot };
      const syntheticAction = {
        name: '能量获得',
        action_type: '能量',
        damage_type: '无',
      };
      const activeAtTick = activeBuffs.filter((candidate) => tick >= int(candidate.start_tick) && tick < int(candidate.end_tick));
      const context = {
        enemy,
        tick,
        enemy_debuffs: activeEnemyDebuffs(enemyDebuffs, tick),
        active_buffs: activeAtTick,
        active_buff_keys: activeAtTick.map((candidate) => String(candidate.definition_id || '')),
      };
      applicableBuffContributions(
        activeBuffs,
        syntheticStep,
        syntheticAction,
        snapshot,
        int(snapshot?.slot) !== int(currentFrontSlot),
        context,
      ).forEach((contribution) => mergeMods(panelMods, contribution.effects));
      return num(panelMods.energy_recharge);
    }

    function plannedActionEnergy({ actorSlot, baseActionEnergy, actorEnergyGain, extraEnergyReturn, tick, currentFrontSlot }) {
      const plannedBySlot = new Map();
      snapshots.forEach((recipientSnapshot, recipientSlot) => {
        if (recipientSnapshot.character?.uses_energy === false) return;
        if (!canReceiveEnergy(recipientSnapshot, currentFrontSlot)) return;
        const amount = int(recipientSlot) === int(actorSlot)
          ? actorEnergyGain + extraEnergyReturn
          : baseActionEnergy * TEAMMATE_ENERGY_SHARE_RATIO * (1 + energyRechargeForRecipient(recipientSnapshot, tick, currentFrontSlot));
        if (amount <= 0) return;
        plannedBySlot.set(recipientSlot, amount);
      });
      return plannedBySlot;
    }

    function applyEnergyDelta(slot, amount, tick, source = {}) {
      if (amount <= 0) return 0;
      const currentEnergy = energyBySlot.get(slot) ?? initialEnergyBySlot.get(slot) ?? 0;
      const capacity = energyCapacityForSlot(slot);
      const nextEnergy = capacity > 0 ? Math.min(capacity, currentEnergy + amount) : currentEnergy + amount;
      const appliedAmount = Math.max(0, nextEnergy - currentEnergy);
      energyBySlot.set(slot, nextEnergy);
      if (appliedAmount > 0) {
        energyEvents.push({
          tick: int(tick),
          slot: int(slot),
          amount: appliedAmount,
          energy_after: nextEnergy,
          kind: 'action_gain',
          source_step_id: String(source.step_id || ''),
          source_action_id: String(source.action_id || ''),
        });
      }
      return appliedAmount;
    }

    function recordEnergyState(slot, previousEnergy, tick, source = {}) {
      const nextEnergy = energyBySlot.get(slot) ?? initialEnergyBySlot.get(slot) ?? 0;
      if (Math.abs(nextEnergy - previousEnergy) <= 1e-9) return;
      energyEvents.push({
        tick: int(tick),
        slot: int(slot),
        amount: nextEnergy - previousEnergy,
        energy_after: nextEnergy,
        kind: String(source.kind || 'action_cost'),
        source_step_id: String(source.step_id || ''),
        source_action_id: String(source.action_id || ''),
      });
    }

    function registerPeriodicHealing(instance) {
      const periodic = instance?.rule?.periodic_heal && typeof instance.rule.periodic_heal === 'object'
        ? instance.rule.periodic_heal
        : null;
      if (!periodic) return;
      const intervalTicks = Math.max(1, int(periodic.interval_ticks, 10));
      const key = `${int(instance.owner_slot)}:${String(instance.definition_id || instance.rule?.id || '')}`;
      const current = periodicHealingStates.get(key);
      if (current) {
        current.instance = instance;
        current.periodic = periodic;
        current.end_tick = int(instance.end_tick);
        current.next_tick = Math.max(int(current.next_tick), int(instance.start_tick) + intervalTicks);
        return;
      }
      periodicHealingStates.set(key, {
        key,
        instance,
        periodic,
        end_tick: int(instance.end_tick),
        next_tick: int(instance.start_tick) + intervalTicks,
      });
    }

    function settlePeriodicHealing(targetTick) {
      const throughTick = Math.max(0, int(targetTick));
      periodicHealingStates.forEach((state, key) => {
        const intervalTicks = Math.max(1, int(state.periodic?.interval_ticks, 10));
        const endTick = Math.min(throughTick, int(state.end_tick));
        const snapshot = snapshots.get(int(state.instance?.owner_slot));
        while (int(state.next_tick) <= endTick) {
          const tick = int(state.next_tick);
          const maxHpRatio = Math.max(0, num(state.periodic?.max_hp_ratio));
          healingEvents.push({
            kind: 'buff_periodic_heal',
            source: String(state.periodic?.source || state.instance?.name || '周期回复'),
            tick,
            slot: int(state.instance?.owner_slot),
            character_id: String(snapshot?.character?.id || ''),
            character_name: String(snapshot?.character?.name || ''),
            max_hp_ratio: maxHpRatio,
            healing: num(snapshot?.base_stats?.hp) * maxHpRatio,
          });
          state.next_tick = tick + intervalTicks;
        }
        if (int(state.next_tick) > int(state.end_tick) && throughTick >= int(state.end_tick)) {
          periodicHealingStates.delete(key);
        }
      });
    }

    function settleActionEnergy(targetTick) {
      const settledThroughTick = Math.max(lastEnergySettlementTick, int(targetTick));
      for (let tick = lastEnergySettlementTick + 1; tick <= settledThroughTick; tick += 1) {
        pendingActionEnergy.forEach((stream) => {
          if (tick <= stream.start_tick || tick > stream.end_tick) return;
          stream.planned_by_slot.forEach((totalAmount, recipientSlot) => {
            applyEnergyDelta(
              recipientSlot,
              totalAmount / stream.duration_ticks,
              tick,
              {
                step_id: stream.step_id,
                action_id: stream.action_id,
              },
            );
          });
        });
      }
      lastEnergySettlementTick = settledThroughTick;
      pendingActionEnergy = pendingActionEnergy.filter((stream) => stream.end_tick > settledThroughTick);
    }

    function scheduleActionEnergy({
      actorSlot,
      baseActionEnergy,
      actorEnergyGain,
      extraEnergyReturn,
      startTick,
      durationTicks,
      currentFrontSlot,
      stepId,
      actionId,
    }) {
      const plannedBySlot = plannedActionEnergy({
        actorSlot,
        baseActionEnergy,
        actorEnergyGain,
        extraEnergyReturn,
        tick: startTick,
        currentFrontSlot,
      });
      if (durationTicks <= 0) {
        plannedBySlot.forEach((amount, recipientSlot) => {
          applyEnergyDelta(recipientSlot, amount, startTick, {
            step_id: stepId,
            action_id: actionId,
          });
        });
      } else if (plannedBySlot.size) {
        pendingActionEnergy.push({
          step_id: String(stepId || ''),
          action_id: String(actionId || ''),
          start_tick: startTick,
          end_tick: startTick + durationTicks,
          duration_ticks: Math.max(1, durationTicks),
          planned_by_slot: plannedBySlot,
        });
      }
      return plannedBySlot;
    }

    function triggerReaction(scheduled, tick, { primeLoop = false } = {}) {
      if (!isSupportAction(scheduled.action)) return { effect: null, warning: '' };
      const supportSnapshot = snapshots.get(int(scheduled.slot));
      const previousSnapshot = snapshots.get(reactionPreviousSlot(scheduled));
      if (!supportSnapshot || !previousSnapshot) {
        return {
          effect: null,
          warning: primeLoop ? '' : '援护技前没有可用于触发环合的前台角色。',
        };
      }
      if (supportSnapshot.slot === previousSnapshot.slot) {
        return {
          effect: null,
          warning: primeLoop ? '' : '援护技没有切换角色，无法触发异能环合。',
        };
      }
      const reaction = reactionForElements(previousSnapshot.character?.element, supportSnapshot.character?.element);
      if (!reaction) {
        return {
          effect: null,
          warning: primeLoop
            ? ''
            : `${previousSnapshot.character?.element || '无属性'}与${supportSnapshot.character?.element || '无属性'}无法产生异能环合。`,
        };
      }
      if (primeLoop && !loopPrimedReactionStepIds.has(String(scheduled.step?.id || ''))) {
        return { effect: null, warning: '' };
      }
      const previousCurrentHarmony = harmonyBySlot.get(previousSnapshot.slot) || 0;
      const previousHarmony = Math.min(HARMONY_CAPACITY, previousCurrentHarmony);
      if (!primeLoop && previousHarmony < 100 && !supportBypassesHarmony(scheduled)) {
        return {
          effect: null,
          warning: `${previousSnapshot.character?.name || '上一前台角色'}环合值不足：需要 100，当前 ${Math.max(0, previousHarmony).toFixed(1)}。`,
        };
      }
      const contributor = reactionContributor(reaction, previousSnapshot, supportSnapshot, tick);
      if (!contributor) return { effect: null, warning: '' };
      if (!primeLoop && !scheduled.action?.preserve_harmony && !scheduled.step?.preserve_harmony) {
        const currentConsumption = Math.min(previousCurrentHarmony, 100);
        harmonyBySlot.set(previousSnapshot.slot, Math.max(0, previousCurrentHarmony - currentConsumption));
      }
      const durationTicks = int(REACTION_DURATIONS[reaction]);
      const frequencyMultiplier = reaction === '创生' && teamHasJiuyuan() ? 2 : 1;
      const carriedEffect = carriedReactionEffectAtTick(reaction, tick);
      if (carriedEffect) {
        return {
          effect: refreshCarriedReactionEffect(carriedEffect, reaction, tick, {
            support_slot: supportSnapshot.slot,
            support_character_id: supportSnapshot.character?.id || '',
            support_character_name: supportSnapshot.character?.name || '',
            previous_slot: previousSnapshot.slot,
            previous_character_id: previousSnapshot.character?.id || '',
            previous_character_name: previousSnapshot.character?.name || '',
            trigger_slot: contributor.slot,
            trigger_character_id: contributor.character?.id || '',
            trigger_character_name: contributor.character?.name || '',
            contributor_slot: contributor.slot,
            contributor_character_id: contributor.character?.id || '',
            contributor_character_name: contributor.character?.name || '',
            frequency_multiplier: Math.max(
              frequencyMultiplier,
              num(carriedEffect.frequency_multiplier, num(carriedEffect.stack_count, 1)),
            ),
          }, primeLoop),
          warning: '',
        };
      }
      const effect = {
        id: `reaction_${nextReactionEffectId}`,
        reaction,
        support_slot: supportSnapshot.slot,
        support_character_id: supportSnapshot.character?.id || '',
        support_character_name: supportSnapshot.character?.name || '',
        previous_slot: previousSnapshot.slot,
        previous_character_id: previousSnapshot.character?.id || '',
        previous_character_name: previousSnapshot.character?.name || '',
        trigger_slot: contributor.slot,
        trigger_character_id: contributor.character?.id || '',
        trigger_character_name: contributor.character?.name || '',
        contributor_slot: contributor.slot,
        contributor_character_id: contributor.character?.id || '',
        contributor_character_name: contributor.character?.name || '',
        start_tick: tick,
        end_tick: tick + durationTicks,
        duration_ticks: durationTicks,
        frequency_multiplier: frequencyMultiplier,
        damage_ticks: reactionDamageTicks(reaction, tick),
        loop_primed: primeLoop,
      };
      nextReactionEffectId += 1;
      reactionEffects.push(effect);
      enemyDebuffs[reaction] = Math.max(int(enemyDebuffs[reaction]), int(effect.end_tick) + 1);
      effect.damage_ticks.forEach((damageTick, index) => {
        const isDot = reaction === '浊燃';
        reactionDamageEvents.push({
          effect_id: effect.id,
          reaction,
          tick: damageTick,
          sequence: index + 1,
          trigger_slot: effect.trigger_slot,
          contributor_slot: effect.contributor_slot,
          contributor_character_id: effect.contributor_character_id,
          contributor_character_name: effect.contributor_character_name,
          extra_tag: isDot ? 'DOT' : '',
          tags: isDot ? ['DOT'] : [],
          frequency_multiplier: effect.frequency_multiplier,
          loop_primed: primeLoop,
          damage: null,
        });
      });
      if (reaction === '创生') {
        const iloy = iloySnapshot();
        if (iloy) {
          const baseFlowerCount = 20;
          const cloneFrequencyMultiplier = teamHasJiuyuan() ? 2 : 1;
          const flowerCount = baseFlowerCount;
          const cloneDurationTicks = baseFlowerCount * 5;
          const cloneDamageTicks = evenlySpacedDamageTicks(
            tick + 30,
            cloneDurationTicks,
            flowerCount,
          );
          const cloneEffect = {
            id: `reaction_${nextReactionEffectId}`,
            reaction: '创生复制体',
            support_slot: effect.support_slot,
            support_character_id: effect.support_character_id,
            support_character_name: effect.support_character_name,
            previous_slot: effect.previous_slot,
            previous_character_id: effect.previous_character_id,
            previous_character_name: effect.previous_character_name,
            trigger_slot: effect.trigger_slot,
            trigger_character_id: effect.trigger_character_id,
            trigger_character_name: effect.trigger_character_name,
            contributor_slot: effect.contributor_slot,
            contributor_character_id: effect.contributor_character_id,
            contributor_character_name: effect.contributor_character_name,
            source_reaction: '创生',
            damage_scale: 0.375,
            base_flower_count: baseFlowerCount,
            flower_count: flowerCount,
            interval_ticks: 5,
            frequency_multiplier: cloneFrequencyMultiplier,
            generation_delay_ticks: 30,
            start_tick: tick + 30,
            end_tick: cloneDamageTicks.at(-1) || tick,
            duration_ticks: cloneDurationTicks,
            damage_ticks: cloneDamageTicks,
            loop_primed: primeLoop,
          };
          nextReactionEffectId += 1;
          reactionEffects.push(cloneEffect);
          cloneDamageTicks.forEach((damageTick, index) => {
            reactionDamageEvents.push({
              effect_id: cloneEffect.id,
              reaction: cloneEffect.reaction,
              source_reaction: cloneEffect.source_reaction,
              tick: damageTick,
              sequence: index + 1,
              trigger_slot: cloneEffect.trigger_slot,
              contributor_slot: cloneEffect.contributor_slot,
              contributor_character_id: cloneEffect.contributor_character_id,
              contributor_character_name: cloneEffect.contributor_character_name,
              damage_scale: cloneEffect.damage_scale,
              frequency_multiplier: cloneEffect.frequency_multiplier,
              extra_tag: '',
              tags: [],
              loop_primed: primeLoop,
              damage: null,
            });
          });
        }
      }
      return { effect, warning: '' };
    }

    function seedLoopInitialReaction(reaction, snapshot, startTick = 0) {
      const durationTicks = Math.max(0, int(REACTION_DURATIONS[reaction]));
      if (!durationTicks || !snapshot) return null;
      const frequencyMultiplier = reaction === '创生' && teamHasJiuyuan() ? 2 : 1;
      const damageTicks = reactionDamageTicks(reaction, startTick);
      const effect = {
        id: `reaction_${nextReactionEffectId}`,
        reaction,
        support_slot: -1,
        support_character_id: '',
        support_character_name: '',
        previous_slot: -1,
        previous_character_id: '',
        previous_character_name: '',
        trigger_slot: snapshot.slot,
        trigger_character_id: snapshot.character?.id || '',
        trigger_character_name: snapshot.character?.name || '',
        contributor_slot: snapshot.slot,
        contributor_character_id: snapshot.character?.id || '',
        contributor_character_name: snapshot.character?.name || '',
        carrier_slot: snapshot.slot,
        start_tick: startTick,
        end_tick: startTick + durationTicks,
        duration_ticks: durationTicks,
        frequency_multiplier: frequencyMultiplier,
        damage_ticks: damageTicks,
        loop_primed: false,
        loop_initial: true,
      };
      nextReactionEffectId += 1;
      reactionEffects.push(effect);
      enemyDebuffs[reaction] = Math.max(int(enemyDebuffs[reaction]), int(effect.end_tick) + 1);
      damageTicks.forEach((damageTick, index) => {
        const isDot = reaction === '浊燃';
        reactionDamageEvents.push({
          effect_id: effect.id,
          reaction,
          tick: damageTick,
          sequence: index + 1,
          trigger_slot: effect.trigger_slot,
          contributor_slot: effect.contributor_slot,
          contributor_character_id: effect.contributor_character_id,
          contributor_character_name: effect.contributor_character_name,
          extra_tag: isDot ? 'DOT' : '',
          tags: isDot ? ['DOT'] : [],
          frequency_multiplier: effect.frequency_multiplier,
          loop_primed: false,
          loop_initial: true,
          damage: null,
        });
      });
      if (reaction === '创生' && iloySnapshot()) {
        const baseFlowerCount = 20;
        const cloneStartTick = startTick + 30;
        const cloneDamageTicks = evenlySpacedDamageTicks(cloneStartTick, baseFlowerCount * 5, baseFlowerCount);
        const cloneEffect = {
          ...effect,
          id: `reaction_${nextReactionEffectId}`,
          reaction: '创生复制体',
          source_reaction: '创生',
          damage_scale: 0.375,
          base_flower_count: baseFlowerCount,
          flower_count: baseFlowerCount,
          interval_ticks: 5,
          generation_delay_ticks: 30,
          start_tick: cloneStartTick,
          end_tick: cloneDamageTicks.at(-1) || cloneStartTick,
          duration_ticks: baseFlowerCount * 5,
          damage_ticks: cloneDamageTicks,
        };
        nextReactionEffectId += 1;
        reactionEffects.push(cloneEffect);
        cloneDamageTicks.forEach((damageTick, index) => {
          reactionDamageEvents.push({
            effect_id: cloneEffect.id,
            reaction: cloneEffect.reaction,
            source_reaction: cloneEffect.source_reaction,
            tick: damageTick,
            sequence: index + 1,
            trigger_slot: cloneEffect.trigger_slot,
            contributor_slot: cloneEffect.contributor_slot,
            contributor_character_id: cloneEffect.contributor_character_id,
            contributor_character_name: cloneEffect.contributor_character_name,
            damage_scale: cloneEffect.damage_scale,
            frequency_multiplier: cloneEffect.frequency_multiplier,
            extra_tag: '',
            tags: [],
            loop_primed: false,
            loop_initial: true,
            damage: null,
          });
        });
      }
      return effect;
    }

    function reactionDamageAtTick(event) {
      const snapshot = snapshots.get(int(event.contributor_slot));
      if (!snapshot) return 0;
      if (event.effect_id) {
        const effect = reactionEffects.find((candidate) => String(candidate.id || '') === String(event.effect_id));
        if (
          !effect
          || effect.disabled === true
          || int(event.tick) < int(effect.start_tick)
          || int(event.tick) > int(effect.end_tick)
        ) return 0;
      }
      if (event.kind === 'buff_periodic' || event.kind === 'buff_periodic_settlement' || event.kind === 'action_periodic') {
        if (event.kind === 'buff_periodic') {
          if (event._buff_instance && !activeBuffs.includes(event._buff_instance)) return 0;
          if (event._shared_buff_definition_id) {
            const sharedKey = `${int(event._shared_owner_slot)}:${String(event._shared_buff_definition_id)}`;
            const sharedState = sharedPeriodicDamageStates.get(sharedKey);
            if (!sharedState || int(sharedState.generation) !== int(event._shared_generation)) return 0;
            const activeLayers = activeBuffs.filter((instance) => (
              int(instance.owner_slot) === int(event._shared_owner_slot)
              && String(instance.definition_id || '') === String(event._shared_buff_definition_id)
              && int(event.tick) >= int(instance.start_tick)
              && int(event.tick) <= int(instance.end_tick)
            ));
            if (!activeLayers.length) return 0;
            event.stack_count = activeLayers.length;
            event.periodic_scale = activeLayers.length;
          }
        }
        const periodicAction = {
          id: String(event.action_id || `periodic:${String(event.reaction || '')}`),
          name: String(event.reaction || '周期伤害'),
          action_type: String(event.action_type || '周期伤害'),
          damage_type: String(event.damage_type || event.action_type || '周期伤害'),
          extra_tag: 'DOT',
          tags: ['DOT'],
        };
        const panelMods = effectiveReactionPanel(snapshot, int(event.tick), periodicAction);
        const baseStats = snapshot.base_stats || {};
        const atk = num(baseStats.atk) * (1 + panelMods.atk_pct) + panelMods.flat_atk;
        const hp = num(baseStats.hp) * (1 + panelMods.hp_pct) + panelMods.flat_hp;
        const def = num(baseStats.def) * (1 + panelMods.def_pct) + panelMods.flat_def;
        const multipliers = event.multipliers && typeof event.multipliers === 'object'
          ? event.multipliers
          : {atk: num(event.atk_multiplier), hp: 0, def: 0, flat: 0};
        const skill = skillLevelMultiplier(snapshot, periodicAction);
        const unscaledBase = (
          atk * num(multipliers.atk)
          + hp * num(multipliers.hp)
          + def * num(multipliers.def)
        ) * skill.multiplier + num(multipliers.flat);
        const actionPeriodicWindow = event.kind === 'action_periodic' && event._periodic_window_id
          ? periodicDamageWindows.find((window) => String(window.id || '') === String(event._periodic_window_id))
          : null;
        if (event.kind === 'action_periodic' && (
          !actionPeriodicWindow
          || actionPeriodicWindow.activated !== true
          || int(event.tick) < int(actionPeriodicWindow.start_tick)
          || int(event.tick) > int(actionPeriodicWindow.end_tick)
        )) return 0;
        const periodicScale = (event.periodic_scale == null ? 1 : Math.max(0, num(event.periodic_scale)))
          * Math.max(1, num(actionPeriodicWindow?.stack_count, 1));
        const base = unscaledBase * periodicScale;
        const damageBonus = actionTypeBonus(periodicAction, panelMods) + panelMods.element_dmg;
        const critRate = 0.5;
        const critical = 1 + critRate * Math.max(0, num(panelMods.crit_dmg));
        const defense = defenseMultiplier(enemy, panelMods);
        const resistance = resistanceMultiplier(snapshot.character, enemy, panelMods);
        const finalMultiplier = 1 + panelMods.final_dmg;
        event.formula_parts = {
          base,
          unscaled_base: unscaledBase,
          periodic_scale: periodicScale,
          skill_category: skill.category,
          skill_level: skill.level,
          skill_multiplier: skill.multiplier,
          damage_bonus: damageBonus,
          crit_rate: critRate,
          critical,
          defense,
          resistance,
          final_multiplier: finalMultiplier,
        };
        return Math.max(0, base * (1 + damageBonus) * critical * defense * resistance * finalMultiplier);
      }
      const sourceReaction = String(event.source_reaction || event.reaction || '');
      const reactionAction = sourceReaction === '浊燃'
        ? {
          name: '浊燃',
          action_type: '环合',
          damage_type: '环合',
          extra_tag: 'DOT',
          tags: ['DOT'],
        }
        : null;
      const panelMods = effectiveReactionPanel(snapshot, int(event.tick), reactionAction);
      const base = reactionBaseDamage(snapshot.character?.level, sourceReaction);
      const strength = reactionStrengthMultiplier(panelMods.harmony_strength);
      const defense = sourceReaction === '黯星' ? 1 : defenseMultiplier(enemy, panelMods);
      const resistanceCharacter = sourceReaction === '黯星' ? { element: '' } : snapshot.character;
      const resistance = resistanceMultiplier(resistanceCharacter, enemy, panelMods);
      const critRate = sourceReaction === '浊燃' ? 0.5 : 0;
      const critical = 1 + critRate * Math.max(0, num(panelMods.crit_dmg));
      const finalMultiplier = 1 + panelMods.final_dmg;
      const damageScale = event.damage_scale == null ? 1 : Math.max(0, num(event.damage_scale));
      const frequencyMultiplier = Math.max(1, num(event.frequency_multiplier, 1));
      event.formula_parts = {
        base,
        strength,
        frequency_multiplier: frequencyMultiplier,
        defense,
        crit_rate: critRate,
        critical,
        resistance,
        final_multiplier: finalMultiplier,
        damage_scale: damageScale,
      };
      return Math.max(
        0,
        base
          * strength
          * frequencyMultiplier
          * defense
          * critical
          * resistance
          * finalMultiplier
          * damageScale,
      );
    }

    function settleReactionDamage(untilTick) {
      reactionDamageEvents
        .filter((event) => event.damage == null && int(event.tick) >= 0 && int(event.tick) <= untilTick)
        .sort((a, b) => int(a.tick) - int(b.tick) || int(a.sequence) - int(b.sequence))
        .forEach((event) => {
          event.damage = reactionDamageAtTick(event);
          directDamage += event.damage;
          specialDamageBySource.set(event.reaction, (specialDamageBySource.get(event.reaction) || 0) + event.damage);
          reactionDamageBySlot.set(
            int(event.contributor_slot),
            (reactionDamageBySlot.get(int(event.contributor_slot)) || 0) + event.damage,
          );
          const reactionName = String(event.reaction || '');
          const triggersPeriodicDamageBuffs = Boolean(event.kind)
            || ['浊燃', '创生', '创生复制体'].includes(reactionName);
          if (triggersPeriodicDamageBuffs && event.damage > 0) {
            const contributorSlot = int(event.contributor_slot);
            const snapshot = snapshots.get(contributorSlot);
            if (snapshot) {
              const isDotDamage = Boolean(event.kind) || reactionName === '浊燃';
              const periodicAction = {
                id: `periodic:${reactionName}`,
                name: reactionName || '周期伤害',
                action_type: String(event.action_type || '周期伤害'),
                damage_type: String(event.damage_type || event.action_type || '周期伤害'),
                extra_tag: isDotDamage ? 'DOT' : '',
                tags: isDotDamage ? ['DOT'] : [],
                hit_count: 1,
              };
              event.triggered_buffs = triggerBuffsForEvent(
                'periodic_damage',
                int(event.tick),
                {id: periodicAction.id, slot: contributorSlot},
                periodicAction,
                snapshot,
                true,
              );
            }
          }
        });
    }

    function schedulePeriodicBuffDamage(instance, rule) {
      const periodic = rule.periodic_damage && typeof rule.periodic_damage === 'object' ? rule.periodic_damage : {};
      const intervalTicks = Math.max(1, int(periodic.interval_ticks));
      const source = String(periodic.source || rule.name || '周期伤害');
      if (!periodic.atk_multiplier || !intervalTicks) return;
      if (periodic.shared_cadence === true) {
        const definitionId = String(instance.definition_id || rule.stacking?.key || rule.id || '');
        const ownerSlot = int(instance.owner_slot);
        const sharedKey = `${ownerSlot}:${definitionId}`;
        const startTick = int(instance.start_tick);
        const activeEndTick = Math.max(
          int(instance.end_tick),
          ...activeBuffs
            .filter((candidate) => (
              int(candidate.owner_slot) === ownerSlot
              && String(candidate.definition_id || '') === definitionId
              && int(candidate.start_tick) <= startTick
              && int(candidate.end_tick) > startTick
            ))
            .map((candidate) => int(candidate.end_tick)),
        );
        let state = sharedPeriodicDamageStates.get(sharedKey);
        if (!state || startTick >= int(state.coverage_end_tick)) {
          state = {
            generation: nextSharedPeriodicDamageGeneration,
            last_scheduled_tick: startTick,
            coverage_end_tick: activeEndTick,
          };
          nextSharedPeriodicDamageGeneration += 1;
          sharedPeriodicDamageStates.set(sharedKey, state);
        } else {
          state.coverage_end_tick = Math.max(int(state.coverage_end_tick), activeEndTick);
        }
        let sequence = 1;
        for (
          let damageTick = int(state.last_scheduled_tick) + intervalTicks;
          damageTick <= int(state.coverage_end_tick);
          damageTick += intervalTicks
        ) {
          reactionDamageEvents.push({
            kind: 'buff_periodic',
            reaction: source,
            tick: damageTick,
            sequence,
            contributor_slot: ownerSlot,
            contributor_character_id: String(rule.owner_character_id || ''),
            contributor_character_name: String(rule.owner_character_name || ''),
            atk_multiplier: periodicAtkMultiplier(periodic, rule),
            action_type: String(periodic.action_type || '周期伤害'),
            damage_type: String(periodic.damage_type || periodic.action_type || '周期伤害'),
            _shared_buff_definition_id: definitionId,
            _shared_owner_slot: ownerSlot,
            _shared_generation: int(state.generation),
            damage: null,
          });
          state.last_scheduled_tick = damageTick;
          sequence += 1;
        }
        return;
      }
      const immediate = periodic.immediate === true;
      let sequence = 1;
      for (
        let damageTick = int(instance.start_tick) + (immediate ? 0 : intervalTicks);
        immediate ? damageTick < int(instance.end_tick) : damageTick <= int(instance.end_tick);
        damageTick += intervalTicks
      ) {
        reactionDamageEvents.push({
          kind: 'buff_periodic',
          reaction: source,
          tick: damageTick,
          sequence,
          contributor_slot: int(instance.owner_slot),
          contributor_character_id: String(rule.owner_character_id || ''),
          contributor_character_name: String(rule.owner_character_name || ''),
          atk_multiplier: periodicAtkMultiplier(periodic, rule),
          action_type: String(periodic.action_type || '周期伤害'),
          damage_type: String(periodic.damage_type || periodic.action_type || '周期伤害'),
          _buff_instance: instance,
          damage: null,
        });
        sequence += 1;
      }
    }

    function duplicateActionPeriodicWindow(sourceWindow, triggerTick) {
      const intervalTicks = Math.max(1, int(sourceWindow.interval_ticks));
      const tickCount = Math.max(1, int(sourceWindow.tick_count));
      const durationTicks = Math.max(intervalTicks, int(sourceWindow.duration_ticks, intervalTicks * tickCount));
      const duplicate = {
        ...sourceWindow,
        id: `action_periodic_${nextPeriodicDamageWindowId}`,
        step_id: '',
        start_tick: triggerTick,
        end_tick: triggerTick + durationTicks,
        duration_ticks: durationTicks,
        stack_count: 1,
        activated: true,
        duplicated_by_dot_spread: true,
      };
      nextPeriodicDamageWindowId += 1;
      periodicDamageWindows.push(duplicate);
      for (let index = 0; index < tickCount; index += 1) {
        reactionDamageEvents.push({
          kind: 'action_periodic',
          reaction: String(duplicate.source || '持续伤害'),
          tick: triggerTick + intervalTicks * (index + 1),
          sequence: index + 1,
          contributor_slot: int(duplicate.contributor_slot),
          contributor_character_id: String(snapshots.get(int(duplicate.contributor_slot))?.character?.id || ''),
          contributor_character_name: String(snapshots.get(int(duplicate.contributor_slot))?.character?.name || ''),
          action_id: String(duplicate.action_id || ''),
          action_type: String(duplicate.action_type || '周期伤害'),
          damage_type: String(duplicate.damage_type || duplicate.action_type || '周期伤害'),
          multipliers: clone(duplicate.multipliers || {}),
          _periodic_window_id: duplicate.id,
          damage: null,
        });
      }
      return duplicate;
    }

    function duplicateReactionDotEffect(sourceEffect, triggerTick) {
      const durationTicks = Math.max(1, int(sourceEffect.duration_ticks, int(sourceEffect.end_tick) - int(sourceEffect.start_tick)));
      const damageTicks = reactionDamageTicks(String(sourceEffect.reaction || ''), triggerTick);
      const duplicate = {
        ...sourceEffect,
        id: `reaction_${nextReactionEffectId}`,
        start_tick: triggerTick,
        end_tick: triggerTick + durationTicks,
        duration_ticks: durationTicks,
        damage_ticks: damageTicks,
        stack_count: 1,
        frequency_multiplier: 1,
        loop_primed: false,
        loop_initial: false,
        duplicated_by_dot_spread: true,
        disabled: false,
      };
      nextReactionEffectId += 1;
      reactionEffects.push(duplicate);
      enemyDebuffs[String(duplicate.reaction || '')] = Math.max(
        int(enemyDebuffs[String(duplicate.reaction || '')]),
        int(duplicate.end_tick) + 1,
      );
      const template = reactionDamageEvents.find(
        (event) => String(event.effect_id || '') === String(sourceEffect.id || ''),
      ) || {};
      damageTicks.forEach((damageTick, index) => {
        reactionDamageEvents.push({
          ...template,
          effect_id: duplicate.id,
          reaction: String(duplicate.reaction || ''),
          tick: damageTick,
          sequence: index + 1,
          trigger_slot: duplicate.trigger_slot,
          contributor_slot: duplicate.contributor_slot,
          contributor_character_id: duplicate.contributor_character_id,
          contributor_character_name: duplicate.contributor_character_name,
          frequency_multiplier: 1,
          loop_primed: false,
          loop_initial: false,
          damage: null,
        });
      });
      return duplicate;
    }

    function increaseTurbidBurnLayer(effect, triggerTick) {
      const maxStacks = 3;
      const currentFrequency = Math.max(
        1,
        num(effect.frequency_multiplier, num(effect.stack_count, 1)),
      );
      const nextFrequency = Math.min(maxStacks, currentFrequency + 1);
      const refreshedEndTick = triggerTick + int(REACTION_DURATIONS['浊燃']);
      effect.end_tick = refreshedEndTick;
      effect.duration_ticks = Math.max(0, refreshedEndTick - int(effect.start_tick));
      effect.stack_count = nextFrequency;
      effect.frequency_multiplier = nextFrequency;
      enemyDebuffs['浊燃'] = Math.max(int(enemyDebuffs['浊燃']), refreshedEndTick + 1);

      const effectEvents = reactionDamageEvents.filter(
        (event) => String(event.effect_id || '') === String(effect.id || ''),
      );
      effectEvents.forEach((event) => {
        if (event.damage == null && int(event.tick) > triggerTick) {
          event.frequency_multiplier = nextFrequency;
        }
      });
      const template = effectEvents[0] || {};
      const damageTicks = asList(effect.damage_ticks).map((tick) => int(tick));
      let damageTick = damageTicks.length
        ? Math.max(...damageTicks) + 10
        : int(effect.start_tick) + 10;
      while (damageTick <= refreshedEndTick) {
        damageTicks.push(damageTick);
        reactionDamageEvents.push({
          ...template,
          effect_id: effect.id,
          reaction: '浊燃',
          tick: damageTick,
          sequence: damageTicks.length,
          trigger_slot: effect.trigger_slot,
          contributor_slot: effect.contributor_slot,
          contributor_character_id: effect.contributor_character_id,
          contributor_character_name: effect.contributor_character_name,
          frequency_multiplier: nextFrequency,
          loop_primed: false,
          damage: null,
        });
        damageTick += 10;
      }
      effect.damage_ticks = damageTicks;
      return nextFrequency;
    }

    function increaseAllActiveDotLayers(triggerTick) {
      const additions = [];
      const addedBuffs = [];
      const activePeriodicBuffGroups = new Map();
      activeBuffs.forEach((instance) => {
        if (triggerTick < int(instance.start_tick) || triggerTick >= int(instance.end_tick)) return;
        if (!instance.rule?.periodic_damage) return;
        const periodic = instance.rule.periodic_damage;
        if (!actionTags({extra_tag: periodic.extra_tag, tags: periodic.tags}).has('DOT')) return;
        const key = `${int(instance.owner_slot)}:${String(instance.definition_id || '')}`;
        if (!activePeriodicBuffGroups.has(key)) activePeriodicBuffGroups.set(key, []);
        activePeriodicBuffGroups.get(key).push(instance);
      });
      activePeriodicBuffGroups.forEach((instances) => {
        const sourceInstance = instances.at(-1);
        const sourceRule = sourceInstance?.rule;
        if (!sourceRule) return;
        const currentStacks = instances.reduce(
          (sum, instance) => sum + Math.max(0, num(instance.stack_count, 1)),
          0,
        );
        const maxStacks = maxStacksForRule(sourceRule);
        const added = activateBuff(activeBuffs, sourceRule, triggerTick, 1, {
          source_slot: int(sourceInstance.owner_slot),
          source_step_id: '',
        });
        if (!added) return;
        schedulePeriodicBuffDamage(added, added.rule);
        addedBuffs.push(added);
        additions.push({
          kind: 'buff_periodic',
          name: String(sourceRule.periodic_damage?.source || sourceRule.name || ''),
          owner_slot: int(sourceInstance.owner_slot),
          stack_count: Math.min(maxStacks, currentStacks + 1),
        });
      });

      const activeReactionDotGroups = new Map();
      reactionEffects.forEach((effect) => {
        if (effect.disabled === true || triggerTick < int(effect.start_tick) || triggerTick >= int(effect.end_tick)) return;
        const taggedEvent = reactionDamageEvents.find(
          (event) => String(event.effect_id || '') === String(effect.id || ''),
        );
        if (!actionTags(taggedEvent || {}).has('DOT')) return;
        const key = String(effect.reaction || '');
        if (!activeReactionDotGroups.has(key)) activeReactionDotGroups.set(key, []);
        activeReactionDotGroups.get(key).push(effect);
      });
      activeReactionDotGroups.forEach((effects, reaction) => {
        if (reaction === '浊燃') {
          const sourceEffect = effects.slice().sort(
            (left, right) => int(right.end_tick) - int(left.end_tick) || int(right.start_tick) - int(left.start_tick),
          )[0];
          const stackCount = increaseTurbidBurnLayer(sourceEffect, triggerTick);
          additions.push({kind: 'reaction', name: reaction, stack_count: stackCount});
          return;
        }
        const maxStacks = 3;
        const activeLayers = effects.slice().sort(
          (left, right) => int(left.end_tick) - int(right.end_tick) || int(left.start_tick) - int(right.start_tick),
        );
        while (activeLayers.length >= maxStacks) {
          const expiringFirst = activeLayers.shift();
          expiringFirst.end_tick = triggerTick;
          expiringFirst.duration_ticks = Math.max(0, triggerTick - int(expiringFirst.start_tick));
          expiringFirst.damage_ticks = asList(expiringFirst.damage_ticks)
            .filter((damageTick) => int(damageTick) <= triggerTick);
        }
        duplicateReactionDotEffect(effects.at(-1), triggerTick);
        additions.push({kind: 'reaction', name: reaction, stack_count: Math.min(maxStacks, effects.length + 1)});
      });

      const activeActionPeriodicGroups = new Map();
      periodicDamageWindows.forEach((window) => {
        if (!window.activated) return;
        if (triggerTick < int(window.start_tick) || triggerTick >= int(window.end_tick)) return;
        if (!asList(window.damage_tags).map(String).includes('DOT')) return;
        const key = `${int(window.contributor_slot)}:${String(window.action_id || '')}`;
        if (!activeActionPeriodicGroups.has(key)) activeActionPeriodicGroups.set(key, []);
        activeActionPeriodicGroups.get(key).push(window);
      });
      activeActionPeriodicGroups.forEach((windows) => {
        const sourceWindow = windows.at(-1);
        const maxStacks = Math.max(1, int(sourceWindow.max_stacks, 10));
        const expiringFirst = windows.slice().sort(
          (left, right) => int(left.end_tick) - int(right.end_tick) || int(left.start_tick) - int(right.start_tick),
        );
        while (expiringFirst.length >= maxStacks) {
          expiringFirst.shift().activated = false;
        }
        duplicateActionPeriodicWindow(sourceWindow, triggerTick);
        additions.push({
          kind: 'action_periodic',
          name: String(sourceWindow.action_id || ''),
          owner_slot: int(sourceWindow.contributor_slot),
          stack_count: Math.min(maxStacks, windows.length + 1),
        });
      });
      return {additions, addedBuffs};
    }

    function activeDotLayerCount(triggerTick, action = null) {
      let count = 0;
      let corrosionHeartCount = 0;
      activeBuffs.forEach((instance) => {
        if (triggerTick < int(instance.start_tick) || triggerTick >= int(instance.end_tick)) return;
        if (!instance.rule?.periodic_damage) return;
        const layers = Math.max(0, num(instance.stack_count, 1));
        count += layers;
        if (String(instance.definition_id || '') === 'character_canhong_corrosion_heart') {
          corrosionHeartCount += layers;
        }
      });
      reactionEffects.forEach((effect) => {
        if (String(effect.reaction || '') !== '浊燃') return;
        if (triggerTick < int(effect.start_tick) || triggerTick >= int(effect.end_tick)) return;
        count += Math.max(1, num(effect.stack_count, 1));
      });
      periodicDamageWindows.forEach((window) => {
        if (!window.activated) return;
        if (triggerTick < int(window.start_tick) || triggerTick >= int(window.end_tick)) return;
        count += Math.max(1, num(window.stack_count, 1));
      });
      const appliedBeforeLastHit = Math.max(0, int(action?.dot_layers_applied_before_last_hit));
      if (appliedBeforeLastHit > 0) {
        count += Math.min(appliedBeforeLastHit, Math.max(0, 10 - corrosionHeartCount));
      }
      return Math.max(0, count);
    }

    function reactionAmplificationMultiplier(snapshot, calcPanel, tick, reactionFilter = '') {
      let multiplier = 1;
      reactionEffects.forEach((effect) => {
        if (tick < int(effect.start_tick) || tick >= int(effect.end_tick)) return;
        if (reactionFilter && String(effect.reaction || '') !== reactionFilter) return;
        const element = String(snapshot.character?.element || '');
        if (effect.reaction === '浸染' && ['魂', '相'].includes(element)) {
          multiplier *= reactionAmplificationMultiplierForStrength(calcPanel?.harmony_strength);
        }
        if (effect.reaction === '覆纹' && ['灵', '咒'].includes(element)) {
          multiplier *= reactionAmplificationMultiplierForStrength(calcPanel?.harmony_strength);
        }
      });
      return multiplier;
    }

    buffRules.forEach((rule) => {
      if (String(rule.trigger?.event || '') !== 'passive') return;
      const snapshot = snapshots.get(int(rule.owner_slot));
      if (!snapshot) return;
      const context = {
        enemy,
        snapshot,
        tick: 0,
        owner_awakening: rule.owner_awakening,
        owner_awakening_nodes: rule.owner_awakening_nodes,
        enemy_debuffs: activeEnemyDebuffs(enemyDebuffs, 0),
        fons_full: fonsFull,
      };
      if (!conditionsMatch(rule.trigger?.conditions, context)) return;
      const instance = activateBuff(activeBuffs, rule, 0, stackGainForRule(rule, context), context);
      if (instance) registerPeriodicHealing(instance);
    });

    function triggerTickForRule(rule, startTick, endTick) {
      return String(rule.trigger?.event || '') === 'action_end' ? endTick : startTick;
    }

    function triggerBuffsForEvent(event, triggerTick, step, action, snapshot, isBackground, extraContext = {}) {
      const triggered = [];
      const decorateTriggeredSummary = (summary, runtimeRule, triggerEvent = event) => {
        summary.trigger_event = triggerEvent;
        summary.trigger_tick = triggerTick;
        const visualTriggerTick = int(extraContext.visual_trigger_tick, visualTickFromCalculationTick(triggerTick, qVirtualIntervals));
        summary.visual_start_tick = int(summary.start_tick) === triggerTick
          ? visualTriggerTick
          : visualTickFromCalculationTick(int(summary.start_tick), qVirtualIntervals);
        summary.visual_end_tick = String(runtimeRule.duration?.type || '') === 'permanent'
          ? PERMANENT_BUFF_END_TICK
          : visualTickFromCalculationTick(int(summary.end_tick), qVirtualIntervals);
        summary.display_start_tick = summary.visual_start_tick;
        summary.display_end_tick = summary.visual_end_tick;
        return summary;
      };
      const baseContext = Object.assign({
        enemy,
        snapshot,
        tick: triggerTick,
        action_tags: Array.from(actionTagsForSnapshot(action, snapshot)).sort(),
        hit_count: actionHitCount(action),
        enemy_debuffs: activeEnemyDebuffs(enemyDebuffs, triggerTick),
        applied_enemy_debuffs: [],
        fons_full: fonsFull,
        loop_enabled: options.loop_enabled,
      }, extraContext);
      buffRules.forEach((rule) => {
        const ownerSlot = int(rule.owner_slot);
        const ownerInstances = activeBuffs.filter((instance) => (
          int(instance.owner_slot) === ownerSlot
          && triggerTick < int(instance.end_tick)
        ));
        const ownerBuffs = ownerInstances.filter((instance) => triggerTick >= int(instance.start_tick));
        const context = Object.assign({}, baseContext, {
          owner_slot: ownerSlot,
          owner_character_id: String(rule.owner_character_id || ''),
          personal_resources: personalResources.get(ownerSlot) || {},
          active_buff_keys: ownerBuffs.map((instance) => String(instance.definition_id || '')),
          existing_buff_keys: ownerInstances.map((instance) => String(instance.definition_id || '')),
          shield_active: activeBuffs.some((instance) => (
            asList(instance.rule?.tags).map(String).includes('护盾')
            && triggerTick >= int(instance.start_tick)
            && triggerTick < int(instance.end_tick)
            && activeBuffApplies(instance, step, action, snapshot, isBackground, baseContext)
          )),
          source_slot: int(step.slot),
          source_step_id: String(step.id || ''),
          exclude_trigger_action: Boolean(rule.activation?.exclude_trigger_action),
        });
        if (!eventMatchesRule(rule, event, step, action, snapshot, isBackground, context)) return;
        const cooldownTicks = triggerCooldownTicks(rule);
        const cooldownKey = `${int(rule.owner_slot)}:${String(rule.id || '')}`;
        const cooldownUntilTick = buffTriggerCooldowns.has(cooldownKey)
          ? buffTriggerCooldowns.get(cooldownKey)
          : Number.NEGATIVE_INFINITY;
        if (cooldownTicks > 0 && triggerTick < cooldownUntilTick) return;
        const activation = rule.activation && typeof rule.activation === 'object' ? rule.activation : {};
        const runtimeRule = clone(rule);
        const resourceDuration = activation.duration_from_personal_resource
          && typeof activation.duration_from_personal_resource === 'object'
          ? activation.duration_from_personal_resource
          : {};
        if (resourceDuration.resource && num(resourceDuration.per_second) > 0) {
          const resourceAmount = num(personalResources.get(ownerSlot)?.[String(resourceDuration.resource)]);
          runtimeRule.duration = Object.assign({}, runtimeRule.duration, {
            type: 'time',
            ticks: Math.max(
              0,
              int(resourceDuration.min_ticks),
              Math.round(resourceAmount / num(resourceDuration.per_second) * 10),
            ),
          });
        }
        const stackSourceKey = String(activation.effects_from_stack?.key || '');
        const stackSource = ownerBuffs.find((instance) => String(instance.definition_id || '') === stackSourceKey);
        if (stackSourceKey) {
          const sourceStacks = Math.max(0, num(stackSource?.stack_count));
          const sourceEffects = runtimeRule.effects && typeof runtimeRule.effects === 'object' ? runtimeRule.effects : {};
          const fullEffectKeys = new Set(asList(activation.effects_from_stack?.full_effect_keys).map(String));
          const perStackEntries = Object.entries(activation.effects_from_stack?.per_stack_from_effect || {});
          const helperEffectKeys = new Set(perStackEntries
            .map(([targetKey, config]) => String(config?.source || targetKey))
            .filter((sourceKey, index) => sourceKey !== String(perStackEntries[index]?.[0] || '')));
          const resolvedEffects = Object.fromEntries(Object.entries(sourceEffects)
            .filter(([key]) => !fullEffectKeys.has(key) && !helperEffectKeys.has(key)));
          perStackEntries.forEach(([targetKey, config]) => {
            const sourceKey = String(config?.source || targetKey);
            resolvedEffects[targetKey] = num(resolvedEffects[targetKey]) + num(sourceEffects[sourceKey]) * num(config?.factor, 1) * sourceStacks;
          });
          if (sourceStacks >= Math.max(1, int(activation.effects_from_stack?.full_stacks, 1))) {
            fullEffectKeys.forEach((key) => {
              resolvedEffects[key] = num(sourceEffects[key]);
            });
          }
          runtimeRule.effects = resolvedEffects;
          runtimeRule.consumed_stacks = sourceStacks;
        }
        const ownerBase = activation.effects_from_owner_base && typeof activation.effects_from_owner_base === 'object'
          ? activation.effects_from_owner_base
          : {};
        if (ownerBase.effect_key && ownerBase.stat) {
          const ownerSnapshot = snapshots.get(ownerSlot);
          runtimeRule.effects = Object.assign({}, runtimeRule.effects, {
            [String(ownerBase.effect_key)]: num(ownerSnapshot?.base_stats?.[String(ownerBase.stat)]) * num(ownerBase.factor, 1),
          });
        }
        const settlementKey = String(activation.settle_periodic_key || '');
        if (settlementKey) {
          const settlementLayers = ownerBuffs
            .filter((instance) => String(instance.definition_id || '') === settlementKey);
          const remainingSeconds = settlementLayers.reduce(
            (sum, instance) => sum + Math.max(0, int(instance.end_tick) - triggerTick) / 10,
            0,
          );
          const firstLayer = settlementLayers[0];
          if (firstLayer && remainingSeconds > 0) {
            const periodic = firstLayer.rule?.periodic_damage && typeof firstLayer.rule.periodic_damage === 'object'
              ? firstLayer.rule.periodic_damage
              : {};
            reactionDamageEvents.push({
              kind: 'buff_periodic_settlement',
              reaction: String(periodic.source || firstLayer.name || '周期伤害'),
              tick: triggerTick,
              sequence: 1,
              contributor_slot: ownerSlot,
              contributor_character_id: String(rule.owner_character_id || ''),
              contributor_character_name: String(rule.owner_character_name || ''),
              atk_multiplier: periodicAtkMultiplier(periodic, firstLayer.rule),
              action_type: String(periodic.action_type || '周期伤害'),
              damage_type: String(periodic.damage_type || periodic.action_type || '周期伤害'),
              stack_count: settlementLayers.length,
              remaining_seconds: remainingSeconds,
              periodic_scale: remainingSeconds,
              damage: null,
            });
          }
        }
        const clearedBuffKeys = asList(activation.clear_keys).map(String);
        clearedBuffKeys.forEach((key) => {
          activeBuffs = activeBuffs.filter((instance) => {
            if (int(instance.owner_slot) !== ownerSlot || String(instance.definition_id || '') !== key) {
              return true;
            }
            truncateBuffTimelineAt(
              instance,
              triggerTick,
              int(extraContext.visual_trigger_tick, visualTickFromCalculationTick(triggerTick, qVirtualIntervals)),
            );
            return false;
          });
          sharedPeriodicDamageStates.delete(`${ownerSlot}:${key}`);
        });
        const activationResources = personalResources.get(ownerSlot) || {};
        Object.entries(resourceMap(activation.personal_resource_gain)).forEach(([key, gain]) => {
          const cap = num(
            personalResourceCaps[String(rule.owner_character_id || '')]?.[key],
            Number.POSITIVE_INFINITY,
          );
          activationResources[key] = Math.min(cap, (activationResources[key] || 0) + gain);
        });
        asList(activation.personal_resource_set_to_cap).map(String).filter(Boolean).forEach((key) => {
          const cap = num(
            personalResourceCaps[String(rule.owner_character_id || '')]?.[key],
            Number.POSITIVE_INFINITY,
          );
          if (Number.isFinite(cap)) activationResources[key] = cap;
        });
        const activationEnergyGain = num(activation.energy_gain);
        if (activationEnergyGain !== 0) {
          const energyBeforeActivation = energyBySlot.get(ownerSlot) ?? initialEnergyBySlot.get(ownerSlot) ?? 0;
          energyBySlot.set(
            ownerSlot,
            Math.max(0, Math.min(
              energyCapacityForSlot(ownerSlot),
              energyBeforeActivation + activationEnergyGain,
            )),
          );
          recordEnergyState(ownerSlot, energyBeforeActivation, triggerTick, {
            kind: 'buff_activation',
            step_id: String(step.id || ''),
            action_id: String(action.id || ''),
            buff_id: String(rule.id || ''),
          });
        }
        const activationHarmonyGain = num(activation.harmony_gain);
        if (activationHarmonyGain !== 0) {
          harmonyBySlot.set(
            ownerSlot,
            Math.max(0, Math.min(
              HARMONY_CAPACITY,
              (harmonyBySlot.get(ownerSlot) || 0) + activationHarmonyGain,
            )),
          );
        }
        const dotLayerExpansion = activation.increase_all_active_dot_layers === true
          ? increaseAllActiveDotLayers(triggerTick)
          : {additions: [], addedBuffs: []};
        asList(activation.reset_action_cooldowns).map(String).filter(Boolean).forEach((actionId) => {
          const resetAction = actionsById.get(actionId);
          cooldownUntil.set(
            resetAction ? actionCooldownKey(ownerSlot, resetAction) : `${ownerSlot}:${actionId}`,
            triggerTick,
          );
        });
        const previousInstances = new Set(activeBuffs);
        const stackGain = stackGainForRule(runtimeRule, context);
        const stackingDefinitionId = String(runtimeRule.stacking?.key || runtimeRule.id || '');
        const previousStackCount = activeBuffs
          .filter((candidate) => (
            int(candidate.owner_slot) === ownerSlot
            && String(candidate.definition_id || '') === stackingDefinitionId
          ))
          .reduce((sum, candidate) => sum + Math.max(0, num(candidate.stack_count, 1)), 0);
        const instance = activateBuff(activeBuffs, runtimeRule, triggerTick, stackGain, context);
        if (!instance) return;
        activeBuffs
          .filter((candidate) => !previousInstances.has(candidate) && candidate.rule?.periodic_damage)
          .forEach((candidate) => schedulePeriodicBuffDamage(candidate, candidate.rule));
        if (instance.rule?.periodic_heal) registerPeriodicHealing(instance);
        if (cooldownTicks > 0) buffTriggerCooldowns.set(cooldownKey, triggerTick + cooldownTicks);
        const summary = buffSummary(instance);
        if (dotLayerExpansion.additions.length) summary.dot_layer_additions = dotLayerExpansion.additions;
        if (clearedBuffKeys.length) {
          summary.cleared_buff_keys = clearedBuffKeys;
        }
        if (String(runtimeRule.stacking?.mode || '') === 'independent') {
          summary.stack_count = Math.min(
            Math.max(1, num(runtimeRule.stacking?.max_stacks, 1)),
            Math.max(1, Math.floor(stackGain)),
          );
        }
        triggered.push(trackBuffTimelineSummary(
          instance,
          decorateTriggeredSummary(summary, runtimeRule),
        ));
        dotLayerExpansion.addedBuffs.forEach((addedBuff) => {
          triggered.push(trackBuffTimelineSummary(
            addedBuff,
            decorateTriggeredSummary(buffSummary(addedBuff), addedBuff.rule),
          ));
        });
        const fullStackBuffId = String(activation.full_stack_buff_id || '');
        const fullStackThreshold = Math.max(1, num(
          activation.full_stack_threshold,
          maxStacksForRule(runtimeRule),
        ));
        const currentStackCount = String(runtimeRule.stacking?.mode || '') === 'independent'
          ? activeBuffs
            .filter((candidate) => (
              int(candidate.owner_slot) === ownerSlot
              && String(candidate.definition_id || '') === stackingDefinitionId
            ))
            .reduce((sum, candidate) => sum + Math.max(0, num(candidate.stack_count, 1)), 0)
          : Math.max(0, num(instance.stack_count, 1));
        if (fullStackBuffId && previousStackCount < fullStackThreshold && currentStackCount >= fullStackThreshold) {
          const fullStackRule = buffRules.find((candidate) => (
            int(candidate.owner_slot) === ownerSlot
            && String(candidate.id || '') === fullStackBuffId
          ));
          if (fullStackRule) {
            const fullStackInstance = activateBuff(activeBuffs, fullStackRule, triggerTick, 1, context);
            if (fullStackInstance) {
              triggered.push(trackBuffTimelineSummary(
                fullStackInstance,
                decorateTriggeredSummary(
                  buffSummary(fullStackInstance),
                  fullStackRule,
                  'full_stack',
                ),
              ));
            }
          }
        }
      });
      return triggered;
    }

    if (options.loop_enabled && loopDurationTicks > 0) {
      const configuredLoopInitialReactions = [];
      snapshots.forEach((snapshot) => {
        const configured = loopInitialResources[String(snapshot.character?.id || '')];
        const reaction = configured && typeof configured === 'object' ? String(configured.reaction || '') : '';
        if (!reaction) return;
        configuredLoopInitialReactions.push({reaction, snapshot});
        seedLoopInitialReaction(reaction, snapshot, -loopDurationTicks);
      });
      scheduledSteps.forEach((scheduled) => {
        const scheduledStartTick = int(scheduled.start_tick);
        if (scheduledStartTick > loopDurationTicks) return;
        if (
          scheduledStartTick === loopDurationTicks
          && String(scheduled.action?.action_type || '') === '无'
        ) return;
        const step = scheduled.step;
        const action = scheduled.action;
        const snapshot = snapshots.get(int(scheduled.slot));
        if (!snapshot) return;
        const startTick = int(scheduled.start_tick) - loopDurationTicks;
        const endTick = int(scheduled.end_tick) - loopDurationTicks;
        activeBuffs = activeBuffs.filter((instance) => startTick < int(instance.end_tick));
        const actionMultiplier = backgroundActionMultiplier(step, action);
        for (let copyIndex = 0; copyIndex < actionMultiplier; copyIndex += 1) {
          triggerBuffsForEvent(
            'action_start',
            startTick,
            step,
            action,
            snapshot,
            scheduled.is_background,
            {
              visual_trigger_tick: int(scheduled.visual_start_tick) - loopDurationTicks,
              loop_prime_only: true,
            },
          );
          triggerBuffsForEvent(
            'action_hit',
            startTick,
            step,
            action,
            snapshot,
            scheduled.is_background,
            {
              visual_trigger_tick: int(scheduled.visual_start_tick) - loopDurationTicks,
              loop_prime_only: true,
              expected_critical_hits: expectedCriticalHits(action, calculateActionDamage(snapshot, action, enemy, mods())),
            },
          );
          triggerBuffsForEvent(
            'action_end',
            endTick,
            step,
            action,
            snapshot,
            scheduled.is_background,
            {
              visual_trigger_tick: int(scheduled.visual_end_tick) - loopDurationTicks,
              loop_prime_only: true,
            },
          );
        }
        const reactionTick = calculationTickFromVisualIntervals(reactionTriggerTick(scheduled), qVirtualIntervals);
        const previousInstances = new Set(activeBuffs);
        const reactionTrigger = triggerReaction(
          scheduled,
          reactionTick - loopDurationTicks,
          { primeLoop: true },
        );
        if (!reactionTrigger.effect) return;
        triggerBuffsForEvent(
          'reaction_trigger',
          reactionTick - loopDurationTicks,
          scheduled.step,
          scheduled.action,
          snapshot,
          scheduled.is_background,
          {
            reaction: reactionTrigger.effect,
            visual_trigger_tick: reactionTriggerTick(scheduled),
            loop_prime_only: true,
          },
        );
        activeBuffs
          .filter((instance) => !previousInstances.has(instance))
          .forEach((instance) => {
            instance.looped = true;
          });
      });
      activeBuffs = activeBuffs.filter((instance) => (
        String(instance.rule?.trigger?.event || '') === 'passive'
        || (instance.rule?.duration?.loop_carry === true && int(instance.end_tick) > 0)
      ));
      activeBuffs.forEach((instance) => {
        instance.start_tick = Math.max(0, int(instance.start_tick));
        instance.looped = true;
      });
      energyBySlot.clear();
      initialEnergyBySlot.forEach((value, slot) => energyBySlot.set(slot, value));
      harmonyBySlot.clear();
      initialHarmonyBySlot.forEach((value, slot) => harmonyBySlot.set(slot, value));
      personalResources.forEach((resources, slot) => {
        Object.keys(resources).forEach((key) => delete resources[key]);
        Object.assign(resources, initialPersonalResourcesBySlot.get(slot) || {});
      });
      const expiredPrimedReactionIds = new Set(
        reactionEffects
          .filter((effect) => int(effect.end_tick) <= 0)
          .map((effect) => String(effect.id || '')),
      );
      for (let index = reactionEffects.length - 1; index >= 0; index -= 1) {
        if (expiredPrimedReactionIds.has(String(reactionEffects[index].id || ''))) reactionEffects.splice(index, 1);
      }
      for (let index = reactionDamageEvents.length - 1; index >= 0; index -= 1) {
        if (expiredPrimedReactionIds.has(String(reactionDamageEvents[index].effect_id || ''))) reactionDamageEvents.splice(index, 1);
      }
      configuredLoopInitialReactions.forEach(({reaction, snapshot}) => {
        const carriedEffects = reactionEffects.filter((effect) => (
          String(effect.reaction || '') === reaction
          && int(effect.carrier_slot, int(effect.contributor_slot, -1)) === int(snapshot.slot)
          && int(effect.start_tick) <= 0
          && int(effect.end_tick) > 0
        ));
        const carriedFrequency = carriedEffects.reduce(
          (maximum, effect) => Math.max(
            maximum,
            num(effect.frequency_multiplier, num(effect.stack_count, 1)),
          ),
          1,
        );
        const carriedIds = new Set(carriedEffects.map((effect) => String(effect.id || '')));
        for (let index = reactionEffects.length - 1; index >= 0; index -= 1) {
          if (carriedIds.has(String(reactionEffects[index].id || ''))) reactionEffects.splice(index, 1);
        }
        for (let index = reactionDamageEvents.length - 1; index >= 0; index -= 1) {
          if (carriedIds.has(String(reactionDamageEvents[index].effect_id || ''))) reactionDamageEvents.splice(index, 1);
        }
        const initialEffect = seedLoopInitialReaction(reaction, snapshot, 0);
        if (!initialEffect) return;
        initialEffect.looped = carriedEffects.length > 0;
        if (reaction === '浊燃') {
          const inheritedFrequency = Math.min(3, Math.max(1, carriedFrequency));
          initialEffect.stack_count = inheritedFrequency;
          initialEffect.frequency_multiplier = inheritedFrequency;
          reactionDamageEvents.forEach((event) => {
            if (String(event.effect_id || '') === String(initialEffect.id || '')) {
              event.frequency_multiplier = inheritedFrequency;
            }
          });
        }
      });
    }

    let runtimeFrontSlot = options.loop_enabled ? loopOpeningFrontSlot : null;
    let runtimeFrontSinceTick = 0;
    const personalResourceDrains = new Map();

    function settlePersonalResourceDrains(tick) {
      personalResourceDrains.forEach((drain, key) => {
        const settleTick = Math.min(int(tick), int(drain.end_tick));
        const elapsedTicks = Math.max(0, settleTick - int(drain.last_tick));
        if (elapsedTicks > 0) {
          const resources = personalResources.get(int(drain.slot)) || {};
          const resource = String(drain.resource || '');
          resources[resource] = Math.max(0, num(resources[resource]) - elapsedTicks / 10 * num(drain.per_second));
          drain.last_tick = settleTick;
        }
        if (int(tick) >= int(drain.end_tick)) personalResourceDrains.delete(key);
      });
    }

    function registerPersonalResourceDrain(rule, instance, triggerTick) {
      const config = rule?.activation?.duration_from_personal_resource;
      if (!config || typeof config !== 'object' || config.drain === false || !config.resource || num(config.per_second) <= 0 || !instance) return;
      const slot = int(instance.owner_slot);
      const resource = String(config.resource);
      personalResourceDrains.set(`${slot}:${resource}`, {
        slot,
        resource,
        per_second: num(config.per_second),
        last_tick: int(triggerTick),
        end_tick: int(instance.end_tick),
      });
    }

    function activeResourceEffectConfigs(activeBuffsAtTick, slot, tick, kind, resource, step, action, snapshot, isBackground) {
      return activeBuffsAtTick.filter((buff) => {
        if (int(buff.owner_slot) !== int(slot) || tick < int(buff.start_tick) || tick >= int(buff.end_tick)) return false;
        const config = buff.rule?.resource_effects?.[kind];
        if (!config || typeof config !== 'object' || !Object.prototype.hasOwnProperty.call(config, resource)) return false;
        const requirement = config[resource];
        if (requirement === true) return true;
        if (!requirement || typeof requirement !== 'object') return false;
        const tags = strSet(requirement.tags);
        if (tags.size && !Array.from(actionTags(action)).some((tag) => tags.has(tag))) return false;
        const actionIds = strSet(requirement.action_ids);
        if (actionIds.size && !actionIds.has(String(action.id || ''))) return false;
        return activeBuffApplies(buff, step, action, snapshot, isBackground, { tick });
      });
    }

    function syncFrontTimeBuffs(tick) {
      buffRules
        .filter((rule) => String(rule.trigger?.event || '') === 'front_time')
        .forEach((rule) => {
          const ownerSlot = int(rule.owner_slot);
          const stacking = rule.stacking && typeof rule.stacking === 'object' ? rule.stacking : {};
          const definitionId = String(stacking.key || rule.id || '');
          const intervalTicks = Math.max(1, int(stacking.interval_ticks, 10));
          const maxStacks = Math.max(1, int(stacking.max_stacks, 1));
          const stackCount = ownerSlot === runtimeFrontSlot
            ? Math.min(maxStacks, Math.max(0, Math.floor((tick - runtimeFrontSinceTick) / intervalTicks)))
            : 0;
          const existing = activeBuffs.find(
            (instance) => int(instance.owner_slot) === ownerSlot && String(instance.definition_id || '') === definitionId,
          );
          if (stackCount <= 0) {
            if (existing) activeBuffs.splice(activeBuffs.indexOf(existing), 1);
            return;
          }
          if (existing) {
            existing.stack_count = stackCount;
            existing.end_tick = PERMANENT_BUFF_END_TICK;
            return;
          }
          activeBuffs.push({
            rule,
            definition_id: definitionId,
            name: rule.name || '',
            owner_slot: ownerSlot,
            start_tick: runtimeFrontSinceTick + intervalTicks,
            end_tick: PERMANENT_BUFF_END_TICK,
            stack_count: stackCount,
          });
        });
    }

    scheduledSteps.forEach((scheduled) => {
      const step = scheduled.step;
      const slot = int(scheduled.slot);
      const snapshot = snapshots.get(slot);
      if (!snapshot) return;
      const action = scheduled.action;
      periodicDamageWindows
        .filter((window) => String(window.step_id || '') === String(step.id || ''))
        .forEach((window) => { window.activated = true; });
      const startTick = int(scheduled.start_tick);
      const visualStartTick = int(scheduled.visual_start_tick, startTick);
      const isBackground = Boolean(scheduled.is_background);
      const actionMultiplier = backgroundActionMultiplier(step, action);
      const durationTicks = Math.max(0, int(scheduled.duration_ticks));
      const endTick = int(scheduled.end_tick);
      const visualEndTick = int(scheduled.visual_end_tick);
      const cooldownKey = actionCooldownKey(slot, action);
      const availableTick = cooldownUntil.get(cooldownKey) || 0;
      const usesCooldowns = snapshot.character?.uses_cooldowns !== false;
      const usesEnergy = snapshot.character?.uses_energy !== false;
      const isInstantForegroundQ = isZeroForegroundQStep(step, action);
      const warnings = [];
      settleActionEnergy(isInstantForegroundQ ? startTick - 1 : startTick);
      if (int(action.required_awakening) > 0 && !hasAwakeningNode(snapshot, action.required_awakening)) {
        const awakeningLabel = 'ABCDEF'[int(action.required_awakening) - 1] || String(int(action.required_awakening));
        warnings.push(`动作需要 ${awakeningLabel} 觉醒节点。`);
      }
      if (usesCooldowns && startTick < availableTick) warnings.push(`动作 CD 尚未结束，需等到 ${(availableTick / 10).toFixed(1)}s。`);
      const energyCost = usesEnergy ? num(action.energy_cost) * actionMultiplier : 0;
      let slotEnergy = energyBySlot.get(slot) ?? initialEnergyBySlot.get(slot) ?? 0;
      if (energyCost > slotEnergy) warnings.push('终结技能量不足。');
      const buffTick = startTick;
      const previousRuntimeFrontSlot = runtimeFrontSlot;
      if (!isBackground && runtimeFrontSlot !== slot) {
        runtimeFrontSlot = slot;
        runtimeFrontSinceTick = buffTick;
      }
      settlePersonalResourceDrains(buffTick);
      settlePeriodicHealing(buffTick);
      settleReactionDamage(buffTick);
      enemyDebuffs = activeEnemyDebuffs(enemyDebuffs, buffTick);
      activeBuffs = activeBuffs.filter((buff) => (
        buffTick < int(buff.end_tick)
        && !activeBuffResetsOnActionStart(
          buff,
          step,
          action,
          isBackground,
          buffTick,
          previousRuntimeFrontSlot,
          visualStartTick,
        )
      ));
      syncBuffLayerResources(buffTick);
      syncFrontTimeBuffs(buffTick);
      const triggeredBuffs = [];
      if (!isBackground && previousRuntimeFrontSlot != null && previousRuntimeFrontSlot !== slot) {
        const previousSnapshot = snapshots.get(int(previousRuntimeFrontSlot));
        if (previousSnapshot) {
          triggeredBuffs.push(...triggerBuffsForEvent(
            'foreground_leave',
            startTick,
            {id: String(step.id || ''), slot: previousRuntimeFrontSlot},
            {id: 'foreground_leave', name: '离开前台', action_type: '切人', damage_type: '切人', tags: []},
            previousSnapshot,
            false,
            {visual_trigger_tick: visualStartTick},
          ));
        }
      }
      if (!isBackground && previousRuntimeFrontSlot !== slot) {
        triggeredBuffs.push(...triggerBuffsForEvent(
          'foreground_enter',
          startTick,
          step,
          action,
          snapshot,
          false,
          {visual_trigger_tick: visualStartTick},
        ));
      }
      for (let copyIndex = 0; copyIndex < actionMultiplier; copyIndex += 1) {
        triggeredBuffs.push(...triggerBuffsForEvent(
          'action_start',
          startTick,
          step,
          action,
          snapshot,
          isBackground,
          { visual_trigger_tick: visualStartTick },
        ));
      }
      slotEnergy = energyBySlot.get(slot) ?? slotEnergy;
      const requiredBuffPlacements = strSet(action.required_buff_placements);
      const shouldValidateRequiredBuff = !requiredBuffPlacements.size
        || requiredBuffPlacements.has(placement(isBackground));
      const requiredBuffKey = String(action.required_buff_key || '');
      if (shouldValidateRequiredBuff && requiredBuffKey && !activeBuffs.some((buff) => (
        String(buff.definition_id || '') === requiredBuffKey
        && int(buff.owner_slot) === slot
        && buffTick >= int(buff.start_tick)
        && buffTick < int(buff.end_tick)
      ))) {
        warnings.push(`动作需要处于 ${String(action.required_buff_name || requiredBuffKey)} 状态。`);
      }
      const requiredBuffAnyKeys = strSet(action.required_buff_any_keys);
      if (shouldValidateRequiredBuff && requiredBuffAnyKeys.size && !activeBuffs.some((buff) => (
        requiredBuffAnyKeys.has(String(buff.definition_id || ''))
        && int(buff.owner_slot) === slot
        && buffTick >= int(buff.start_tick)
        && buffTick < int(buff.end_tick)
      ))) {
        warnings.push(`动作需要处于 ${String(action.required_buff_name || Array.from(requiredBuffAnyKeys).join(' / '))} 状态。`);
      }
      if (String(snapshot.character?.id || '') === CANHONG_CHARACTER_ID) {
        const bloodBanquetActive = activeBuffs.some((buff) => (
          String(buff.definition_id || '') === CANHONG_BLOOD_BANQUET_BUFF_ID
          && int(buff.owner_slot) === slot
          && buffTick >= int(buff.start_tick)
          && buffTick < int(buff.end_tick)
        ));
        const awakeningDAutoActive = hasAwakeningNode(snapshot, 4) && slotEnergy >= energyCost && energyCost > 0;
        if (action.canhong_blood_activation_required === true && !bloodBanquetActive && !awakeningDAutoActive) {
          warnings.push('血宴尚未激活。');
        }
        if (String(action.id || '') === CANHONG_FENTIAN_ACTION_ID && (bloodBanquetActive || awakeningDAutoActive)) {
          warnings.push('血宴已激活，应优先释放血宴。');
        }
      }
      const appliedBuffs = [];
      const buffModifiers = mods();
      const personalResourceGainPct = {};
      const activeAtTick = activeBuffs.filter((buff) => buffTick >= int(buff.start_tick) && buffTick < int(buff.end_tick));
      const buffContext = {
        enemy,
        action,
        tick: buffTick,
        enemy_debuffs: activeEnemyDebuffs(enemyDebuffs, buffTick),
        active_buffs: activeAtTick,
        active_buff_keys: activeAtTick.map((buff) => String(buff.definition_id || '')),
        active_periodic_action_ids: activePeriodicActionIds(buffTick),
        active_damage_sources: activeDamageSources(buffTick),
        active_dot_layer_count: activeDotLayerCount(buffTick, action),
      };
      applicableBuffContributions(activeBuffs, step, action, snapshot, isBackground, buffContext)
        .forEach(({ buff, effects }) => {
          mergeMods(buffModifiers, effects);
          Object.entries(effects || {}).forEach(([key, value]) => {
            const prefix = 'personal_resource_gain_pct_';
            if (!key.startsWith(prefix)) return;
            const resourceKey = key.slice(prefix.length);
            personalResourceGainPct[resourceKey] = num(personalResourceGainPct[resourceKey]) + num(value);
          });
          appliedBuffs.push(buffSummary(buff, buffContext));
        });
      const slotResources = personalResources.get(slot) || {};
      Object.entries(resourceMap(action.personal_resource_threshold)).forEach(([key, threshold]) => {
        if ((slotResources[key] || 0) < threshold && action.personal_resource_threshold_warning === true) {
          warnings.push(`个人资源 ${key} 未达到 ${threshold} 点阈值。`);
        }
      });
      Object.entries(resourceMap(action.personal_resource_cost)).forEach(([key, cost]) => {
        const totalCost = cost * actionMultiplier;
        if (activeResourceEffectConfigs(activeBuffs, slot, buffTick, 'ignore_cost', key, step, action, snapshot, isBackground).length) {
          return;
        }
        if ((slotResources[key] || 0) < totalCost) warnings.push(`个人资源 ${key} 不足。`);
        slotResources[key] = Math.max(0, (slotResources[key] || 0) - totalCost);
      });
      Object.entries(resourceMap(action.personal_resource_gain)).forEach(([key, gain]) => {
        if (activeResourceEffectConfigs(activeBuffs, slot, buffTick, 'block_gain', key, step, action, snapshot, isBackground).length) {
          warnings.push(`${key} 正在持续消耗，期间无法积攒。`);
          return;
        }
        const cap = num(personalResourceCaps[String(snapshot.character?.id || '')]?.[key], Number.POSITIVE_INFINITY);
        const gainMultiplier = Math.max(0, 1 + num(personalResourceGainPct[key]));
        slotResources[key] = Math.min(cap, (slotResources[key] || 0) + gain * gainMultiplier * actionMultiplier);
      });
      triggeredBuffs.forEach((summary) => {
        const instance = activeBuffs.find((buff) => (
          int(buff.owner_slot) === int(summary.owner_slot)
          && String(buff.definition_id || '') === String(summary.definition_id || '')
          && int(buff.start_tick) === int(summary.start_tick)
          && int(buff.end_tick) === int(summary.end_tick)
        ));
        if (instance) registerPersonalResourceDrain(instance.rule, instance, startTick);
      });
      const energyBeforeAction = slotEnergy;
      slotEnergy = Math.max(0, slotEnergy - energyCost);
      if (
        String(snapshot.character?.id || '') === ZHENHONG_CHARACTER_ID
        && String(action.id || '') === ZHENHONG_ASCENDANT_ENTRY_ACTION_ID
      ) {
        zhenhongAscendantBySlot.set(slot, true);
        slotEnergy = Math.min(slotEnergy, ZHENHONG_ASCENDANT_ENERGY_CAPACITY);
      } else if (
        String(snapshot.character?.id || '') === ZHENHONG_CHARACTER_ID
        && String(action.id || '') === ZHENHONG_ASCENDANT_EXIT_ACTION_ID
      ) {
        zhenhongAscendantBySlot.set(slot, false);
        slotEnergy = 0;
      }
      const calc = calculateActionDamage(snapshot, action, enemy, buffModifiers);
      const consumedBuffs = new Set(
        activeBuffs
          .filter((buff) => (
            buffTick >= int(buff.start_tick)
            && buffTick < int(buff.end_tick)
            && (buff.rule?.consume_on_apply === true || buff.rule?.consume?.on_apply === true)
            && activeBuffApplies(buff, step, action, snapshot, isBackground, {
              enemy,
              tick: buffTick,
              enemy_debuffs: activeEnemyDebuffs(enemyDebuffs, buffTick),
              active_buffs: activeAtTick,
              active_buff_keys: activeAtTick.map((candidate) => String(candidate.definition_id || '')),
            })
          )),
      );
      if (consumedBuffs.size) {
        activeBuffs = activeBuffs.filter((buff) => !consumedBuffs.has(buff));
      }
      const criticalHitsPerAction = expectedCriticalHits(action, calc);
      const criticalHits = criticalHitsPerAction * actionMultiplier;
      const appliedEnemyDebuffs = applyEnemyDebuffs(enemyDebuffs, action, buffTick);
      if (!action.periodic_damage) {
        for (let copyIndex = 0; copyIndex < actionMultiplier; copyIndex += 1) {
          triggeredBuffs.push(...triggerBuffsForEvent('action_hit', startTick, step, action, snapshot, isBackground, {
            visual_trigger_tick: visualStartTick,
            expected_critical_hits: criticalHitsPerAction,
            applied_enemy_debuffs: appliedEnemyDebuffs,
            enemy_debuffs: activeEnemyDebuffs(enemyDebuffs, buffTick),
          }));
        }
      }
      const reactionAmplification = reactionAmplificationMultiplier(snapshot, calc.panel, buffTick);
      const fuwenAmplification = reactionAmplificationMultiplier(snapshot, calc.panel, buffTick, '覆纹');
      const baseActionDirectDamage = calc.direct_damage * actionMultiplier;
      const amplifiedActionDamage = baseActionDirectDamage * reactionAmplification;
      const fuwenDamage = Math.max(0, baseActionDirectDamage * (fuwenAmplification - 1));
      const multipliedDirectDamage = Math.max(0, amplifiedActionDamage - fuwenDamage);
      let multipliedStagger = calc.stagger_amount * actionMultiplier;
      let forcedStagger = false;
      const forceStaggerNodes = action.force_stagger_by_awakening_node || {};
      if (
        Object.entries(forceStaggerNodes).some(([node, enabled]) => enabled === true && hasAwakeningNode(snapshot, node))
        && options.loop_enabled !== true
      ) {
        const targetKey = 'single_target';
        if (!forcedStaggerTargets.has(targetKey)) {
          multipliedStagger = Math.max(multipliedStagger, Math.max(0, STAGGER_LIMIT - totalStagger));
          forcedStaggerTargets.add(targetKey);
          forcedStagger = true;
        }
      }
      const baseActionHarmony = num(action.harmony);
      const awakeningActionHarmony = actionValueForAwakeningNodes(
        action,
        snapshot,
        'harmony',
        'harmony_by_awakening_node',
      );
      const actionHarmonyScale = baseActionHarmony !== 0 ? calc.harmony / baseActionHarmony : 1;
      const multipliedHarmony = awakeningActionHarmony * actionHarmonyScale * actionMultiplier;
      const baseActionEnergyGain = num(action.energy_gain) * actionMultiplier;
      const actorActionEnergyGain = calc.energy_gain * actionMultiplier;
      let actionEnergyReturnPerAction = num(action.energy_return);
      Object.entries(action.energy_return_by_awakening_node || {}).forEach(([level, value]) => {
        if (hasAwakeningNode(snapshot, level)) actionEnergyReturnPerAction += num(value);
      });
      const actionEnergyReturn = actionEnergyReturnPerAction * actionMultiplier;
      directDamage += multipliedDirectDamage + fuwenDamage;
      if (fuwenDamage > 0) {
        specialDamageBySource.set('覆纹', (specialDamageBySource.get('覆纹') || 0) + fuwenDamage);
      }
      const damageSource = specialDamageSource(action);
      if (damageSource) {
        specialDamageBySource.set(damageSource, (specialDamageBySource.get(damageSource) || 0) + multipliedDirectDamage);
      }
      totalStagger += multipliedStagger;
      staggerProfileSamplesBySlot.get(slot).push({
        weight: actionMultiplier,
        profile: calc.stagger_profile,
      });
      harmonyBySlot.set(slot, Math.min(HARMONY_CAPACITY, (harmonyBySlot.get(slot) || 0) + multipliedHarmony));
      energyBySlot.set(slot, slotEnergy);
      recordEnergyState(slot, energyBeforeAction, startTick, {
        kind: energyCost > 0 ? 'action_cost' : 'action_state',
        step_id: step.id,
        action_id: action.id,
      });
      if (isInstantForegroundQ) {
        settleActionEnergy(startTick);
      }
      const plannedEnergy = scheduleActionEnergy({
        actorSlot: slot,
        baseActionEnergy: baseActionEnergyGain,
        actorEnergyGain: actorActionEnergyGain,
        extraEnergyReturn: actionEnergyReturn,
        startTick: buffTick,
        durationTicks,
        currentFrontSlot: runtimeFrontSlot,
        stepId: step.id,
        actionId: action.id,
      });
      slotEnergy = energyBySlot.get(slot) ?? slotEnergy;
      const displayedEnergyGain = plannedEnergy.get(slot) || 0;
      const actionCooldownTicks = Math.max(0, int(actionValueForAwakeningNodes(
        action,
        snapshot,
        'cooldown_ticks',
        'cooldown_ticks_by_awakening_node',
      )));
      if (usesCooldowns && durationTicks > 0) {
        cooldownUntil.set(cooldownKey, Math.max(cooldownUntil.get(cooldownKey) || 0, startTick + Math.max(durationTicks, actionCooldownTicks)));
      } else if (usesCooldowns && actionCooldownTicks > 0) {
        cooldownUntil.set(cooldownKey, Math.max(cooldownUntil.get(cooldownKey) || 0, startTick + actionCooldownTicks));
      }
      if (!isBackground) {
        frontEvents.push({
          slot,
          start_tick: startTick,
          end_tick: endTick,
          visual_start_tick: visualStartTick,
          visual_end_tick: visualEndTick,
          order: frontEvents.length,
        });
      }
      const reactionTick = calculationTickFromVisualIntervals(reactionTriggerTick(scheduled), qVirtualIntervals);
      const reactionTrigger = triggerReaction(scheduled, reactionTick);
      const triggeredReaction = reactionTrigger.effect;
      if (triggeredReaction) {
        triggeredBuffs.push(...triggerBuffsForEvent(
          'reaction_trigger',
          reactionTick,
          step,
          action,
          snapshot,
          isBackground,
          {
            reaction: triggeredReaction,
            visual_trigger_tick: reactionTriggerTick(scheduled),
          },
        ));
      }
      if (reactionTrigger.warning) {
        warnings.push(reactionTrigger.warning);
      }
      for (let copyIndex = 0; copyIndex < actionMultiplier; copyIndex += 1) {
        triggeredBuffs.push(...triggerBuffsForEvent(
          'action_end',
          endTick,
          step,
          action,
          snapshot,
          isBackground,
          { visual_trigger_tick: visualEndTick },
        ));
      }
      syncBuffLayerResources(endTick);
      slotEnergy = energyBySlot.get(slot) ?? slotEnergy;
      details.push({
        resource_sequence: details.length,
        step_id: step.id || '',
        slot,
        character_id: snapshot.character.id,
        character_name: snapshot.character.name,
        action_id: action.id,
        action_name: action.name,
        action_type: action.action_type,
        damage_type: action.damage_type,
        damage_element: action.damage_element || snapshot.character.element || '',
        damage_source: damageSource || '',
        raw_start_tick: int(step.start_tick),
        switch_gap_ticks: Math.max(0, int(scheduled.switch_loss_ticks)),
        foreground_lock_ticks: Math.max(0, int(scheduled.foreground_lock_ticks)),
        start_tick: startTick,
        calculation_start_sequence: int(scheduled.calculation_start_sequence),
        end_tick: endTick,
        calculation_end_sequence: int(scheduled.calculation_end_sequence),
        duration_ticks: durationTicks,
        visual_start_tick: visualStartTick,
        visual_end_tick: visualEndTick,
        display_start_tick: visualStartTick,
        display_end_tick: visualEndTick,
        display_duration_ticks: Math.max(0, int(scheduled.original_duration_ticks)),
        tick_duration_ticks: Math.max(0, visualEndTick - visualStartTick),
        display_visual_end_tick: visualEndTick,
        original_start_tick: int(scheduled.original_start_tick, startTick),
        original_calculation_start_sequence: int(scheduled.original_calculation_start_sequence),
        original_duration_ticks: int(scheduled.original_duration_ticks, durationTicks),
        original_end_tick: int(scheduled.original_end_tick, endTick),
        original_calculation_end_sequence: int(scheduled.original_calculation_end_sequence),
        original_visual_end_tick: int(scheduled.original_visual_end_tick, visualEndTick),
        q_instant_release: Boolean(scheduled.q_instant_release),
        q_instant_release_kind: scheduled.q_instant_release_kind || '',
        q_instant_release_tick: scheduled.q_instant_release_tick,
        q_instant_release_anchor_tick: scheduled.q_instant_release_anchor_tick,
        q_instant_release_calculation_tick: scheduled.q_instant_release_calculation_tick,
        q_instant_release_anchor_step_id: scheduled.q_instant_release_anchor_step_id || '',
        q_instant_release_start_sequence: scheduled.q_instant_release_start_sequence,
        q_instant_release_end_sequence: scheduled.q_instant_release_end_sequence,
        q_cover_visual_end_tick: scheduled.q_cover_visual_end_tick,
        q_cover_target_step_ids: asList(scheduled.q_cover_target_step_ids),
        is_background_damage: isBackground,
        is_basic_background: isBasicBackgroundOverride(step, action),
        action_multiplier: actionMultiplier,
        action_tags: Array.from(actionTagsForSnapshot(action, snapshot)).sort(),
        hit_count: actionHitCount(action) * actionMultiplier,
        expected_critical_hits: criticalHits,
        applied_enemy_debuffs: appliedEnemyDebuffs,
        enemy_debuffs: activeEnemyDebuffs(enemyDebuffs, buffTick),
        direct_damage: multipliedDirectDamage,
        fuwen_damage: fuwenDamage,
        stagger_amount: multipliedStagger,
        forced_stagger: forcedStagger,
        harmony: multipliedHarmony,
        energy_gain: displayedEnergyGain,
        base_energy_gain: actorActionEnergyGain,
        energy_return: actionEnergyReturn,
        energy_gain_timing: durationTicks > 0 ? 'uniform' : 'instant',
        energy_gain_start_tick: startTick,
        energy_gain_end_tick: startTick + durationTicks,
        energy_after: slotEnergy,
        energy_capacity_after: energyCapacityForSlot(slot),
        harmony_after: harmonyBySlot.get(slot) || 0,
        personal_resources_after: Object.assign({}, slotResources),
        resources_after_by_slot: Array.from(snapshots.keys())
          .sort((left, right) => left - right)
          .map((resourceSlot) => ({
            slot: resourceSlot,
            energy: energyBySlot.get(resourceSlot) ?? initialEnergyBySlot.get(resourceSlot) ?? 0,
            energy_capacity: energyCapacityForSlot(resourceSlot),
            harmony: harmonyBySlot.get(resourceSlot) || 0,
            personal_resources: Object.assign({}, personalResources.get(resourceSlot) || {}),
          })),
        nightmare_stacks: action.nightmare_stacks == null ? action.nightmare_stacks : num(action.nightmare_stacks) * actionMultiplier,
        sin_recovery: action.sin_recovery == null ? action.sin_recovery : num(action.sin_recovery) * actionMultiplier,
        triggered_reaction: triggeredReaction,
        applied_buffs: appliedBuffs,
        triggered_buffs: triggeredBuffs,
        panel: calc.panel,
        formula_parts: Object.assign({}, calc.formula_parts, {
          action_multiplier: actionMultiplier,
          reaction_amplification: reactionAmplification,
          fuwen_amplification: fuwenAmplification,
          active_dot_layer_count: buffContext.active_dot_layer_count,
        }),
        stagger_profile: calc.stagger_profile,
        warnings,
      });
    });

    settleReactionDamage(options.loop_enabled ? loopDurationTicks : Number.POSITIVE_INFINITY);
    reactionEffects.forEach((effect) => {
      effect.visual_start_tick = visualTickFromCalculationTick(int(effect.start_tick), qVirtualIntervals);
      effect.visual_end_tick = visualTickFromCalculationTick(int(effect.end_tick), qVirtualIntervals);
    });
    reactionDamageEvents.forEach((event) => {
      event.visual_tick = visualTickFromCalculationTick(int(event.tick), qVirtualIntervals);
    });
    frontEvents.sort((a, b) => int(a.visual_start_tick, int(a.start_tick)) - int(b.visual_start_tick, int(b.start_tick)) || int(a.order) - int(b.order));
    const dedupedFrontEvents = [];
    frontEvents.forEach((event) => {
      const last = dedupedFrontEvents[dedupedFrontEvents.length - 1];
      if (last && int(last.visual_start_tick, int(last.start_tick)) === int(event.visual_start_tick, int(event.start_tick))) {
        dedupedFrontEvents[dedupedFrontEvents.length - 1] = event;
      } else {
        dedupedFrontEvents.push(event);
      }
    });
    const frontWindows = [];
    dedupedFrontEvents.forEach((event, index) => {
      const startTick = int(event.visual_start_tick, int(event.start_tick));
      const endTick = index + 1 < dedupedFrontEvents.length
        ? int(dedupedFrontEvents[index + 1].visual_start_tick, int(dedupedFrontEvents[index + 1].start_tick))
        : Math.max(int(event.end_tick), int(event.visual_end_tick), startTick + 1);
      if (endTick <= startTick) return;
      const last = frontWindows[frontWindows.length - 1];
      if (last && int(last.slot) === int(event.slot) && startTick <= int(last.end_tick)) {
        last.end_tick = Math.max(int(last.end_tick), endTick);
        last.visual_end_tick = last.end_tick;
        return;
      }
      frontWindows.push({ slot: int(event.slot), start_tick: startTick, end_tick: endTick, visual_end_tick: endTick });
    });

    const durationTicks = Math.max(
      0,
      ...scheduledSteps
        .filter((scheduled) => !scheduled.is_background)
        .map((scheduled) => Math.max(int(scheduled.end_tick), int(scheduled.start_tick))),
    );
    settlePeriodicHealing(options.loop_enabled ? loopDurationTicks : durationTicks);
    healingEvents.forEach((event) => {
      event.visual_tick = visualTickFromCalculationTick(int(event.tick), qVirtualIntervals);
    });
    settleActionEnergy(durationTicks);
    settlePersonalResourceDrains(durationTicks);
    syncBuffLayerResources(durationTicks);
    const timelineTicks = Math.max(
      0,
      ...scheduledSteps
        .filter((scheduled) => !scheduled.is_background)
        .map((scheduled) => Math.max(int(scheduled.visual_end_tick), int(scheduled.visual_start_tick))),
    );
    const frozenTicks = qVirtualIntervals.reduce(
      (sum, interval) => sum + Math.max(0, int(interval.end_tick) - int(interval.start_tick)),
      0,
    );
    const harmonyDamage = HARMONY_DAMAGE_SOURCES.reduce(
      (sum, source) => sum + (specialDamageBySource.get(source) || 0),
      0,
    );
    const durationSeconds = durationTicks / 10;
    const staggerRecoverySeconds = STAGGER_RECOVERY_SECONDS + (
      Array.from(snapshots.values()).some((snapshot) => String(snapshot.arc?.id || '') === LAST_ROSE_ARC_ID)
        ? LAST_ROSE_STAGGER_EXTENSION_SECONDS
        : 0
    );
    const staggerFrequency = totalStagger > 0 && durationSeconds > 0
      ? 1 / (STAGGER_LIMIT / totalStagger + staggerRecoverySeconds / durationSeconds)
      : 0;
    const staggerContributionsBySlot = Array.from(snapshots.entries())
      .sort(([left], [right]) => left - right)
      .map(([slot, snapshot]) => {
        const averageProfile = averageStaggerProfile(snapshot, staggerProfileSamplesBySlot.get(slot));
        const damagePerTrigger = staggerContribution(snapshot, averageProfile, enemy);
        return {
          slot,
          character_id: snapshot.character.id,
          character_name: snapshot.character.name,
          average_stagger_strength: averageProfile.stagger_strength,
          average_stagger_damage_bonus: averageStaggerDamageBonus(snapshot, averageProfile),
          average_def_ignore: averageProfile.def_ignore,
          average_def_down: averageProfile.def_down,
          average_res_down: averageProfile.res_down,
          damage_per_trigger: damagePerTrigger,
          damage: damagePerTrigger * staggerFrequency,
        };
      });
    const staggerDamagePerTrigger = staggerContributionsBySlot.reduce(
      (sum, contribution) => sum + contribution.damage_per_trigger,
      0,
    );
    staggerDamage = staggerDamagePerTrigger * staggerFrequency;
    staggerContributionsBySlot.forEach((contribution) => {
      contribution.percent = staggerDamage > 0 ? contribution.damage / staggerDamage * 100 : 0;
    });
    const totalDamage = directDamage + staggerDamage;
    const characterDamage = Math.max(0, directDamage - harmonyDamage);
    const teamEnergy = Array.from(energyBySlot.values()).reduce((sum, value) => sum + value, 0);
    const totalHarmony = Array.from(harmonyBySlot.values()).reduce((sum, value) => sum + value, 0);
    const damageBySlot = new Map();
    const damageByActionBySlot = new Map();
    const harmonyDamageBySourceBySlot = new Map(
      Array.from(snapshots.keys()).map((slot) => [slot, new Map()]),
    );
    details.forEach((detail) => {
      const fuwenDamage = Math.max(0, num(detail.fuwen_damage));
      if (fuwenDamage > 0) {
        const sources = harmonyDamageBySourceBySlot.get(detail.slot);
        sources.set('覆纹', (sources.get('覆纹') || 0) + fuwenDamage);
      }
      if (HARMONY_DAMAGE_SOURCES.includes(String(detail.damage_source || ''))) {
        const sources = harmonyDamageBySourceBySlot.get(detail.slot);
        sources.set(
          detail.damage_source,
          (sources.get(detail.damage_source) || 0) + detail.direct_damage,
        );
        return;
      }
      damageBySlot.set(detail.slot, (damageBySlot.get(detail.slot) || 0) + detail.direct_damage);
      if (!damageByActionBySlot.has(detail.slot)) {
        damageByActionBySlot.set(detail.slot, new Map());
      }
      const actionDamage = damageByActionBySlot.get(detail.slot);
      const actionKey = detail.action_id || detail.action_name;
      const current = actionDamage.get(actionKey) || {
        action_id: detail.action_id,
        action_name: detail.action_name,
        action_type: detail.action_type,
        damage_type: detail.damage_type,
        damage_element: detail.damage_element,
        damage: 0,
      };
      current.damage += detail.direct_damage;
      actionDamage.set(actionKey, current);
    });
    reactionDamageEvents
      .filter((event) => num(event.damage) > 0)
      .forEach((event) => {
      const slot = int(event.contributor_slot);
      const isHarmonyDamage = !event.kind && HARMONY_DAMAGE_SOURCES.includes(String(event.reaction || ''));
      if (isHarmonyDamage) {
        const sources = harmonyDamageBySourceBySlot.get(slot);
        sources.set(event.reaction, (sources.get(event.reaction) || 0) + num(event.damage));
        return;
      }
      damageBySlot.set(slot, (damageBySlot.get(slot) || 0) + num(event.damage));
      if (!damageByActionBySlot.has(slot)) {
        damageByActionBySlot.set(slot, new Map());
      }
          const isPeriodicBuffDamage = Boolean(event.kind);
          const periodicActionType = String(event.action_type || '周期伤害');
          const periodicDamageType = String(event.damage_type || periodicActionType);
          const actionKey = isPeriodicBuffDamage
            ? `periodic:${event.reaction}`
            : `reaction:${event.reaction}`;
          const actionDamage = damageByActionBySlot.get(slot);
          const current = actionDamage.get(actionKey) || {
            action_id: actionKey,
            action_name: event.reaction,
            action_type: isPeriodicBuffDamage ? periodicActionType : '环合',
            damage_type: isPeriodicBuffDamage ? periodicDamageType : '环合',
            damage_element: isPeriodicBuffDamage ? snapshots.get(slot).character.element || '' : '',
            damage: 0,
          };
          current.damage += num(event.damage);
          actionDamage.set(actionKey, current);
    });
    const sortedSlots = Array.from(snapshots.keys()).sort((a, b) => a - b);
    const harmonyContributionsBySlot = sortedSlots.map((slot) => {
      const rawSources = harmonyDamageBySourceBySlot.get(slot) || new Map();
      const sources = DAMAGE_SHARE_SOURCE_GROUPS.map((group) => ({
        source: group.source,
        damage: group.members.reduce((sum, source) => sum + (rawSources.get(source) || 0), 0),
      }))
        .filter((item) => item.damage > 0)
        .sort((left, right) => right.damage - left.damage || left.source.localeCompare(right.source, 'zh-CN'));
      const damage = sources.reduce((sum, item) => sum + item.damage, 0);
      return {
        slot,
        character_id: snapshots.get(slot).character.id,
        character_name: snapshots.get(slot).character.name,
        damage,
        percent: harmonyDamage > 0 ? damage / harmonyDamage * 100 : 0,
        sources: sources.map((item) => ({
          ...item,
          percent: damage > 0 ? item.damage / damage * 100 : 0,
        })),
      };
    });
    const damageBySource = [
      ...DAMAGE_SHARE_SOURCE_GROUPS.map((group) => ({
        source: group.source,
        damage: group.members.reduce((sum, source) => sum + (specialDamageBySource.get(source) || 0), 0),
      })),
      { source: '倾陷', damage: staggerDamage },
    ]
      .filter((item) => item.damage > 0)
      .map((item) => ({
        ...item,
        percent: totalDamage > 0 ? item.damage / totalDamage * 100 : 0,
      }));
    return {
      ok: true,
      summary: {
        duration_ticks: durationTicks,
        duration_seconds: durationTicks / 10,
        timeline_ticks: timelineTicks,
        frozen_ticks: frozenTicks,
        direct_damage: directDamage,
        character_damage: characterDamage,
        harmony_damage: harmonyDamage,
        stagger_damage: staggerDamage,
        stagger_damage_per_trigger: staggerDamagePerTrigger,
        stagger_frequency: staggerFrequency,
        stagger_recovery_seconds: staggerRecoverySeconds,
        total_stagger: totalStagger,
        total_damage: totalDamage,
        dps: totalDamage / Math.max(durationTicks / 10, 0.1),
        team_energy: teamEnergy,
        total_harmony: totalHarmony,
      },
      damage_by_slot: sortedSlots.map((slot) => ({
        slot,
        character_id: snapshots.get(slot).character.id,
        character_name: snapshots.get(slot).character.name,
        damage: damageBySlot.get(slot) || 0,
        direct_damage: damageBySlot.get(slot) || 0,
        stagger_damage: staggerContributionsBySlot.find((item) => item.slot === slot)?.damage || 0,
        harmony_damage: harmonyContributionsBySlot.find((item) => item.slot === slot)?.damage || 0,
        percent: characterDamage > 0 ? (damageBySlot.get(slot) || 0) / characterDamage * 100 : 0,
      })),
      damage_by_action_by_slot: sortedSlots.map((slot) => {
        const characterDamage = damageBySlot.get(slot) || 0;
        return {
          slot,
          character_id: snapshots.get(slot).character.id,
          character_name: snapshots.get(slot).character.name,
          total_damage: characterDamage,
          actions: Array.from((damageByActionBySlot.get(slot) || new Map()).values())
            .filter((item) => item.damage > 0)
            .sort((left, right) => right.damage - left.damage || left.action_name.localeCompare(right.action_name, 'zh-CN'))
            .map((item) => ({
              ...item,
              percent: characterDamage > 0 ? item.damage / characterDamage * 100 : 0,
            })),
        };
      }),
      damage_by_source: damageBySource,
      harmony_contributions_by_slot: harmonyContributionsBySlot,
      stagger_contributions_by_slot: staggerContributionsBySlot,
      resources_by_slot: sortedSlots.map((slot) => ({
        slot,
        character_id: snapshots.get(slot).character.id,
        character_name: snapshots.get(slot).character.name,
        initial_energy: initialEnergyBySlot.get(slot) ?? 0,
        initial_energy_capacity: energyCapacityBySlot.get(slot) || 0,
        energy_capacity: energyCapacityForSlot(slot),
        energy: energyBySlot.get(slot) ?? initialEnergyBySlot.get(slot) ?? 0,
        initial_harmony: initialHarmonyBySlot.get(slot) || 0,
        harmony: harmonyBySlot.get(slot) || 0,
        initial_reaction: options.loop_enabled
          ? String(loopInitialResources[String(snapshots.get(slot).character?.id || '')]?.reaction || '')
          : '',
        initial_personal_resources: initialPersonalResourcesBySlot.get(slot) || {},
        personal_resources: personalResources.get(slot) || {},
      })),
      energy_events: energyEvents,
      build_panels_by_slot: sortedSlots.map((slot) => buildPanelProjection(snapshots.get(slot))),
      time_axis: {
        tick_seconds: 0.1,
        timeline_ticks: timelineTicks,
        real_duration_ticks: durationTicks,
        frozen_intervals: qVirtualIntervals.map((interval) => ({ ...interval })),
      },
      details,
      reaction_effects: reactionEffects,
      reaction_damage_events: reactionDamageEvents
        .filter((event) => event.damage != null && !event.kind)
        .map((event) => Object.fromEntries(Object.entries(event).filter(([key]) => !key.startsWith('_')))),
      periodic_damage_events: reactionDamageEvents
        .filter((event) => num(event.damage) > 0 && Boolean(event.kind))
        .map((event) => Object.fromEntries(Object.entries(event).filter(([key]) => !key.startsWith('_')))),
      healing_events: healingEvents,
      front_windows: frontWindows,
      enemy,
    };
  }

  const SUBSTAT_CONTRIBUTION_FIELDS = [
    { key: 'all_dmg', label: '通伤' },
    { key: 'crit_rate', label: '暴击' },
    { key: 'crit_dmg', label: '暴伤' },
    { key: 'harmony_strength', label: '环合' },
    { key: 'stagger_strength', label: '倾陷' },
    { key: 'atk_pct', label: '攻击%' },
    { key: 'hp_pct', label: '生命%' },
    { key: 'def_pct', label: '防御%' },
    { key: 'flat_atk', label: '攻击' },
    { key: 'flat_hp', label: '生命' },
    { key: 'flat_def', label: '防御' },
  ];

  const BASELINE_SUBSTAT_FIELDS = SUBSTAT_CONTRIBUTION_FIELDS.slice(0, 8);

  function axisWithContributionSubstats(axisPayload, baseCount = 15) {
    const axis = JSON.parse(JSON.stringify(axisPayload || {}));
    const baseSubstats = Object.fromEntries(SUBSTAT_CONTRIBUTION_FIELDS.map((field) => [field.key, 0]));
    BASELINE_SUBSTAT_FIELDS.forEach((field) => {
      baseSubstats[field.key] = baseCount;
    });
    axis.team = asList(axis.team).map((member) => ({
      ...member,
      substat_counts: { ...baseSubstats },
    }));
    return axis;
  }

  function analyzeSubstatContributions(axisPayload, catalog, options = {}) {
    if (!asList(axisPayload?.steps).length) {
      throw new Error('请先在动作轴中添加动作。');
    }
    const baseCount = Math.max(0, int(options.base_count, 15));
    const incrementCount = Math.max(1, int(options.increment_count, 5));
    const baselineAxis = axisWithContributionSubstats(axisPayload, baseCount);
    const baselineResult = simulateAxis(baselineAxis, catalog);
    const baselineTotalDamage = num(baselineResult?.summary?.total_damage);
    const rows = [];

    asList(baselineAxis.team).forEach((member) => {
      SUBSTAT_CONTRIBUTION_FIELDS.forEach((field) => {
        const candidateAxis = JSON.parse(JSON.stringify(baselineAxis));
        const candidateMember = asList(candidateAxis.team)
          .find((item) => int(item.slot) === int(member.slot));
        if (!candidateMember) return;
        candidateMember.substat_counts[field.key] = num(candidateMember.substat_counts[field.key]) + incrementCount;
        const candidateResult = simulateAxis(candidateAxis, catalog);
        const candidateTotalDamage = num(candidateResult?.summary?.total_damage);
        const increaseDamage = candidateTotalDamage - baselineTotalDamage;
        const epsilon = Math.max(1e-8, Math.abs(baselineTotalDamage) * 1e-12);
        if (increaseDamage <= epsilon) return;
        rows.push({
          slot: int(member.slot),
          character_id: String(member.character_id || ''),
          character_name: String(member.character_name || ''),
          stat_key: field.key,
          stat_label: field.label,
          boosted_total_damage: candidateTotalDamage,
          increase_damage: increaseDamage,
          increase_percent: baselineTotalDamage > 0 ? increaseDamage / baselineTotalDamage * 100 : 0,
        });
      });
    });

    const maxIncreaseDamage = Math.max(0, ...rows.map((row) => row.increase_damage));
    rows.forEach((row) => {
      row.contribution_percent = maxIncreaseDamage > 0
        ? row.increase_damage / maxIncreaseDamage * 100
        : 0;
    });
    rows.sort((left, right) => (
      right.contribution_percent - left.contribution_percent
      || right.increase_damage - left.increase_damage
      || left.slot - right.slot
      || SUBSTAT_CONTRIBUTION_FIELDS.findIndex((field) => field.key === left.stat_key)
        - SUBSTAT_CONTRIBUTION_FIELDS.findIndex((field) => field.key === right.stat_key)
    ));

    return {
      base_count: baseCount,
      increment_count: incrementCount,
      baseline_total_damage: baselineTotalDamage,
      max_increase_damage: maxIncreaseDamage,
      fields: SUBSTAT_CONTRIBUTION_FIELDS.map((field) => ({ ...field })),
      rows,
    };
  }

  return {
    ELEMENTS,
    ZERO_ACTION_VISUAL_TICKS,
    MIN_FOREGROUND_START_GAP_TICKS,
    calculateAxisDurationTicks,
    buildSnapshot,
    buildPanelProjection,
    simulateAxis,
    analyzeSubstatContributions,
  };
}));
