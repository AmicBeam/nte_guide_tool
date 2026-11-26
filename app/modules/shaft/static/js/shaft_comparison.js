(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ShaftComparison = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const clone = value => JSON.parse(JSON.stringify(value));
  const MAX_COMBINATIONS = 600;

  function candidateRefinement(arc) {
    const rarity = String(arc?.rarity || '').toUpperCase();
    if (rarity === 'S' || rarity === '5') return 1;
    if (['A', 'B', '4', '3'].includes(rarity)) return 5;
    throw new Error(`无法确定弧盘「${arc?.name || ''}」的稀有度。`);
  }

  function plan(axis, catalog, selection, engine) {
    if (!Array.isArray(axis?.steps) || !axis.steps.length) throw new Error('请先选择包含动作的排轴。');
    const member = axis.team?.find(item => String(item.character_id) === String(selection.character_id));
    if (!member) throw new Error('所选排轴中没有这个角色。');
    const character = catalog.characters.find(item => item.id === member.character_id);
    if (!character || character.selection_disabled) throw new Error('当前账号不能使用这个角色进行配装试算。');
    const unique = values => [...new Set(values || [])];
    const arcIds = unique(selection.arc_ids);
    const cartridgeIds = unique(selection.cartridge_ids);
    const mainStats = unique(selection.main_stats);
    if (!arcIds.length || !cartridgeIds.length || !mainStats.length) throw new Error('弧盘、卡带和主词条各至少选择一项。');
    for (const id of arcIds) {
      const arc = catalog.arcs.find(item => item.id === id);
      if (!arc || arc.adaptation !== character.adaptation) throw new Error('弧盘与角色适应类型不匹配。');
      candidateRefinement(arc);
    }
    if (cartridgeIds.some(id => !catalog.cartridges.some(item => item.id === id))) throw new Error('存在未知卡带。');
    if (mainStats.some(stat => !Object.hasOwn(catalog.formula_constants.cartridge_main_stat_options, stat))) throw new Error('存在未知主词条。');
    const total = arcIds.length * cartridgeIds.length * mainStats.length;
    if (total > MAX_COMBINATIONS) throw new Error(`一次最多比较 ${MAX_COMBINATIONS} 个组合，请减少勾选项。`);
    const snapshot = engine.buildSnapshot(member, catalog, axis.team_panel_bonus);
    return {member, total, arcIds, cartridgeIds, mainStats, snapshot};
  }

  function warningsFor(result, axis, catalog, selfCheck) {
    const warnings = (result.details || []).flatMap(detail => (detail.warnings || [])
      .map(message => `${detail.character_name || ''} ${detail.action_name || ''}：${message}`));
    if (selfCheck) warnings.push(...selfCheck.inspectAxis(axis, catalog, result));
    return [...new Set(warnings)];
  }

  function summarize(result, member, axis, catalog, selfCheck) {
    return {
      ...result.summary,
      own_damage: result.damage_by_slot.find(item => Number(item.slot) === Number(member.slot))?.damage || 0,
      warnings: warningsFor(result, axis, catalog, selfCheck),
    };
  }

  function candidateAxis(axis, member, option) {
    const candidate = clone(axis);
    const selected = candidate.team.find(item => Number(item.slot) === Number(member.slot));
    Object.assign(selected, {
      arc_id: option.arc_id, arc_refinement: option.arc_refinement,
      cartridge_id: option.cartridge_id, cartridge_main_stat: option.main_stat,
    });
    // Keep the serialized build mirror consistent without changing other characters.
    if (candidate.character_builds?.[member.character_id]) {
      Object.assign(candidate.character_builds[member.character_id], clone(selected));
    }
    return candidate;
  }

  function* compare(axis, catalog, selection, engine, selfCheck) {
    const prepared = plan(axis, catalog, selection, engine);
    const {member, total, snapshot} = prepared;
    const baselineResult = engine.simulateAxis(clone(axis), catalog);
    const baseline = summarize(baselineResult, member, axis, catalog, selfCheck);
    const baselineOptions = {
      arc_id: member.arc_id || '', arc_refinement: snapshot.arc_refinement,
      cartridge_id: member.cartridge_id || '', main_stat: snapshot.build_options.cartridge_main_stat,
    };
    yield {type: 'baseline', baseline, baseline_options: baselineOptions, total};
    let completed = 0;
    for (const arcId of prepared.arcIds) for (const cartridgeId of prepared.cartridgeIds) for (const mainStat of prepared.mainStats) {
      const option = {arc_id: arcId, arc_refinement: candidateRefinement(catalog.arcs.find(arc => arc.id === arcId)), cartridge_id: cartridgeId, main_stat: mainStat};
      const candidate = candidateAxis(axis, member, option);
      const result = engine.simulateAxis(candidate, catalog);
      const summary = summarize(result, member, candidate, catalog, selfCheck);
      const originalActions = baselineResult.damage_by_action_by_slot.find(item => Number(item.slot) === Number(member.slot))?.actions || [];
      const actions = result.damage_by_action_by_slot.find(item => Number(item.slot) === Number(member.slot))?.actions || [];
      const actionIds = [...new Set([...originalActions, ...actions].map(item => item.action_id))];
      const actionChanges = actionIds.map(id => {
        const current = actions.find(item => item.action_id === id);
        const original = originalActions.find(item => item.action_id === id);
        return {name: current?.action_name || original?.action_name, damage: current?.damage || 0,
          delta: (current?.damage || 0) - (original?.damage || 0)};
      }).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 5);
      completed++;
      yield {type: 'row', completed, total, row: {...option, ...summary, action_changes: actionChanges,
        baseline_percent: baseline.total_damage > 0 ? summary.total_damage / baseline.total_damage * 100 : null,
        increase_percent: baseline.total_damage > 0 ? (summary.total_damage / baseline.total_damage - 1) * 100 : null}};
    }
  }

  return {MAX_COMBINATIONS, plan, compare, candidateAxis, candidateRefinement};
}));
