(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
  root.ShaftSelfCheck = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const MIN_FOREGROUND_RETURN_TICKS = 12;

  function isBasicAction(action) {
    return String(action?.action_type || '') === '普攻' || String(action?.damage_type || '') === '普攻';
  }

  function isBackgroundStep(step, action) {
    const marker = `${action?.name || ''} ${action?.extra_tag || ''}`;
    const isNativeBackground = Boolean(action?.is_background_damage) || marker.includes('后台');
    const isManualBackground = Boolean(action?.can_background_override) &&
      (isBasicAction(action) || Boolean(action?.is_instant_switch)) &&
      step?.placement === 'background';
    return isNativeBackground || isManualBackground;
  }

  function basicAttackStage(action) {
    if (!isBasicAction(action)) {
      return null;
    }
    const match = String(action?.name || '').match(/(?:^|[^a-z])a([1-9]\d*)/i);
    return match ? Number(match[1]) : null;
  }

  function isQAction(action) {
    return String(action?.action_type || '') === 'Q' || String(action?.damage_type || '') === 'Q';
  }

  function actionDurationTicks(step, action) {
    return Boolean(step?.detached) && Boolean(action?.can_detach)
      ? Math.max(0, Number(action?.detached_duration_ticks || 0))
      : Math.max(0, Number(action?.duration_ticks || 0));
  }

  function isMaskingForegroundQ(step, action) {
    return !isBackgroundStep(step, action) &&
      isQAction(action) &&
      actionDurationTicks(step, action) === 0;
  }

  function hasUnimplementedForegroundDuration(step, action) {
    return Boolean(action?.id) &&
      !isBackgroundStep(step, action) &&
      !isQAction(action) &&
      !Boolean(action?.is_instant_switch) &&
      actionDurationTicks(step, action) === 0;
  }

  function foregroundReturnWarnings(orderedSteps) {
    const warnings = [];
    const pendingReturns = new Map();
    let foregroundSlot = null;

    orderedSteps.forEach(({ step, action, calculationTick }) => {
      if (isBackgroundStep(step, action) && !Boolean(action?.is_instant_switch)) {
        return;
      }
      const slot = Number(step?.slot || 0);
      const tick = Math.max(0, Number(calculationTick || 0));
      if (foregroundSlot === slot) {
        if (isMaskingForegroundQ(step, action)) {
          pendingReturns.forEach((pending) => { pending.masked = true; });
        }
        return;
      }

      const returning = pendingReturns.get(slot);
      if (isMaskingForegroundQ(step, action)) {
        pendingReturns.forEach((pending, pendingSlot) => {
          if (pendingSlot !== slot) pending.masked = true;
        });
      }
      if (returning) {
        const elapsedTicks = tick - Number(returning.departure_tick || 0);
        if (!returning.masked && elapsedTicks < MIN_FOREGROUND_RETURN_TICKS) {
          const characterName = String(action?.character_name || '该角色');
          const actionName = String(action?.name || '动作');
          const availableTick = Number(returning.departure_tick || 0) + MIN_FOREGROUND_RETURN_TICKS;
          warnings.push(
            `${characterName}「${actionName}」：角色切换 CD 尚未结束，需等到 ${(availableTick / 10).toFixed(1)}s。`,
          );
        }
        pendingReturns.delete(slot);
      }
      if (foregroundSlot !== null) {
        pendingReturns.set(foregroundSlot, {
          departure_tick: tick,
          masked: isMaskingForegroundQ(step, action),
        });
      }
      foregroundSlot = slot;
    });

    return warnings;
  }

  function loopInitialReactionWarnings(axis, catalog, simulationResult) {
    if (!axis?.options?.loop_enabled) return [];
    const configuredResources = axis.options.loop_initial_resources;
    if (!configuredResources || typeof configuredResources !== 'object') return [];

    const policies = new Map((catalog?.formula_constants?.loop_initial_reaction_options || []).map((option) => [
      String(option?.id || ''),
      option || {},
    ]));
    const characterNames = new Map((catalog?.characters || []).map((character) => [
      String(character?.id || ''),
      String(character?.name || character?.id || '未知角色'),
    ]));
    const teamCharacterIds = new Set((axis?.team || []).map((member) => String(member?.character_id || '')));
    const configuredByReaction = new Map();
    Object.entries(configuredResources).forEach(([characterId, resources]) => {
      if (!teamCharacterIds.has(String(characterId)) || !resources || typeof resources !== 'object') return;
      const reaction = String(resources.reaction || '');
      if (!reaction || !policies.has(reaction)) return;
      if (!configuredByReaction.has(reaction)) configuredByReaction.set(reaction, []);
      configuredByReaction.get(reaction).push({
        character_id: String(characterId),
        character_name: characterNames.get(String(characterId)) || String(characterId),
      });
    });

    const axisEffectsByReaction = new Map();
    const reactionsRecordedFromDetails = new Set();
    function recordAxisEffect(effect) {
      if (!effect || typeof effect !== 'object' || effect?.source_reaction) return;
      const reaction = String(effect?.reaction || '');
      if (!configuredByReaction.has(reaction)) return;
      if (!axisEffectsByReaction.has(reaction)) axisEffectsByReaction.set(reaction, []);
      axisEffectsByReaction.get(reaction).push(effect);
    }
    (simulationResult?.details || []).forEach((detail) => {
      // A current-cycle trigger may refresh the carried loop-initial instance in place,
      // so reaction_effects alone cannot prove whether the axis produced it again.
      const effect = detail?.triggered_reaction;
      recordAxisEffect(effect);
      if (effect?.reaction) reactionsRecordedFromDetails.add(String(effect.reaction));
    });
    (simulationResult?.reaction_effects || []).forEach((effect) => {
      if (effect?.loop_initial === true) return;
      if (reactionsRecordedFromDetails.has(String(effect?.reaction || ''))) return;
      recordAxisEffect(effect);
    });

    const warnings = [];
    configuredByReaction.forEach((carriers, reaction) => {
      const policy = policies.get(reaction) || {};
      const maxInstances = Math.max(1, Number(policy.max_instances || 1));
      const axisEffects = axisEffectsByReaction.get(reaction) || [];
      const logicalInstanceCount = carriers.length + axisEffects.length;
      if (logicalInstanceCount <= maxInstances) return;

      const carrierNames = Array.from(new Set(carriers.map((carrier) => carrier.character_name)));
      const axisProducerNames = Array.from(new Set(axisEffects.map((effect) => (
        String(effect?.trigger_character_name || effect?.contributor_character_name || '轴内动作')
      ))));
      const sources = [];
      if (carrierNames.length) sources.push(`${carrierNames.join('、')}自带`);
      if (axisProducerNames.length) sources.push(`轴内由${axisProducerNames.join('、')}产生`);
      warnings.push(
        `循环轴自带环合「${reaction}」冲突：${sources.join('，且')}；原始机制最多同时存在 ${maxInstances} 个实例，刷新或额外加层能力不放宽此自检。${reaction === '黯星' ? ' 黯星冲突会立即结算上一个实例的伤害。' : ''}`,
      );
    });
    return warnings;
  }

  function inspectAxis(axis, catalog, simulationResult = {}) {
    const resultDetails = Array.isArray(simulationResult)
      ? simulationResult
      : (simulationResult?.details || []);
    const actions = new Map((catalog?.actions || []).map((action) => [String(action.id || ''), action]));
    const details = new Map((resultDetails || []).map((detail) => [String(detail?.step_id || ''), detail]));
    const orderedSteps = (axis?.steps || [])
      .map((step, order) => {
        const detail = details.get(String(step?.id || '')) || {};
        return {
          step,
          order,
          action: actions.get(String(step?.action_id || '')) || {},
          calculationTick: Number(detail?.start_tick ?? step?.start_tick ?? 0),
          calculationSequence: Number(detail?.calculation_start_sequence ?? 0),
          visualTick: Number(detail?.visual_start_tick ?? step?.start_tick ?? 0),
        };
      })
      .sort((left, right) => (
        left.calculationTick - right.calculationTick ||
        left.calculationSequence - right.calculationSequence ||
        left.visualTick - right.visualTick ||
        left.order - right.order
      ));
    const warnings = [];
    const abnormalCharacters = new Set();
    const missingDurationActions = new Set();

    warnings.push(...foregroundReturnWarnings(orderedSteps));
    warnings.push(...loopInitialReactionWarnings(axis, catalog, Array.isArray(simulationResult) ? {} : simulationResult));

    orderedSteps.forEach(({ step, action }) => {
      if (!hasUnimplementedForegroundDuration(step, action)) {
        return;
      }
      const actionKey = String(action?.id || `${action?.character_name || ''}:${action?.name || ''}`);
      if (missingDurationActions.has(actionKey)) {
        return;
      }
      missingDurationActions.add(actionKey);
      const characterName = String(action?.character_name || '该角色');
      const actionName = String(action?.name || '动作');
      warnings.push(`${characterName}「${actionName}」：该动作的时长未实装。`);
    });

    for (let index = 1; index < orderedSteps.length; index += 1) {
      const previous = orderedSteps[index - 1];
      const current = orderedSteps[index];
      if (
        Number(previous.step?.slot) !== Number(current.step?.slot) ||
        isBackgroundStep(previous.step, previous.action) ||
        isBackgroundStep(current.step, current.action)
      ) {
        continue;
      }
      const previousStage = basicAttackStage(previous.action);
      const currentStage = basicAttackStage(current.action);
      if (previousStage == null || currentStage == null || currentStage === 1 || currentStage >= previousStage) {
        continue;
      }
      const characterName = String(current.action?.character_name || previous.action?.character_name || '该角色');
      if (!abnormalCharacters.has(characterName)) {
        abnormalCharacters.add(characterName);
        warnings.push(`${characterName}存在普攻段数异常`);
      }
    }

    return warnings;
  }

  return {
    MIN_FOREGROUND_RETURN_TICKS,
    basicAttackStage,
    hasUnimplementedForegroundDuration,
    isMaskingForegroundQ,
    loopInitialReactionWarnings,
    inspectAxis,
  };
}));
