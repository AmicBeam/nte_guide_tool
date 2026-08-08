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
    return Boolean(action?.is_background_damage) || marker.includes('后台') || step?.placement === 'background';
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

  function isMaskingForegroundQ(step, action) {
    return !isBackgroundStep(step, action) &&
      isQAction(action) &&
      Math.max(0, Number(action?.duration_ticks || 0)) === 0;
  }

  function foregroundReturnWarnings(orderedSteps) {
    const warnings = [];
    const pendingReturns = new Map();
    let foregroundSlot = null;

    orderedSteps.forEach(({ step, action, calculationTick }) => {
      if (isBackgroundStep(step, action)) {
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

  function inspectAxis(axis, catalog, resultDetails = []) {
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

    warnings.push(...foregroundReturnWarnings(orderedSteps));

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
    isMaskingForegroundQ,
    inspectAxis,
  };
}));
