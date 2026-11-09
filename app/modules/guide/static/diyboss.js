'use strict';

for (const selector of document.querySelectorAll('[data-diy-selector]')) {
  const stageLinks = [...selector.querySelectorAll('[data-diy-stage-link]')];
  const stages = [...selector.querySelectorAll('[data-diy-stage]')];
  const difficultyLinks = [...selector.querySelectorAll('[data-diy-difficulty-link]')];
  const panels = [...selector.querySelectorAll('[data-diy-difficulty]')];
  const calculators = new Map();
  for (const calculator of selector.querySelectorAll('[data-diy-calculator]')) {
    let pending;
    let revision = 0;
    const total = calculator.querySelector('[data-diy-total]');
    const breakdown = calculator.querySelector('[data-diy-breakdown]');
    const error = calculator.querySelector('[data-diy-score-error]');
    const targetScore = calculator.querySelector('[data-diy-target-score]');
    const refresh = async difficulty => {
      calculator.dataset.difficulty = difficulty;
      pending?.abort();
      const controller = new AbortController();
      pending = controller;
      const requestRevision = ++revision;
      calculator.setAttribute('aria-busy', 'true');
      error.hidden = true;
      total.textContent = '计算中…';
      breakdown.hidden = true;
      if (targetScore) targetScore.textContent = '…';
      const buffs = [...calculator.querySelectorAll('input[type="radio"]:checked')]
        .map(input => input.value).filter(Boolean);
      const targets = [...calculator.querySelectorAll('[data-diy-target]:checked')]
        .map(input => input.value);
      try {
        const response = await fetch(calculator.dataset.scoreUrl, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({stage: calculator.dataset.stageId, difficulty, buffs, targets}),
          signal: controller.signal,
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error?.message || '积分计算失败');
        if (requestRevision !== revision) return;
        const score = payload.data;
        total.textContent = score.total.toLocaleString('zh-CN');
        breakdown.textContent = `（基础 ${score.base_score} + 强化 ${score.buff_score} + 目标 ${score.target_score}）× ${score.score_rate}`;
        breakdown.hidden = false;
        if (targetScore) targetScore.textContent = score.target_score.toLocaleString('zh-CN');
      } catch (failure) {
        if (failure.name === 'AbortError' || requestRevision !== revision) return;
        total.textContent = '暂无法计算';
        error.textContent = failure.message;
        error.hidden = false;
      } finally {
        if (requestRevision === revision) calculator.removeAttribute('aria-busy');
      }
    };
    calculator.addEventListener('change', () => refresh(calculator.dataset.difficulty));
    calculator.querySelector('[data-diy-clear]').addEventListener('click', () => {
      for (const input of calculator.querySelectorAll('input')) {
        input.checked = input.type === 'radio' && input.value === '';
      }
      refresh(calculator.dataset.difficulty);
    });
    calculators.set(calculator.dataset.stageId, refresh);
  }
  const show = (stageId, difficultyId) => {
    if (!stages.some(stage => stage.dataset.diyStage === stageId)) return;
    for (const stage of stages) stage.hidden = stage.dataset.diyStage !== stageId;
    for (const panel of panels) panel.hidden = panel.dataset.diyDifficulty !== difficultyId;
    for (const link of stageLinks) {
      if (link.dataset.diyStageLink === stageId) link.setAttribute('aria-current', 'true');
      else link.removeAttribute('aria-current');
      link.href = `?stage=${encodeURIComponent(link.dataset.diyStageLink)}&difficulty=${encodeURIComponent(difficultyId)}`;
    }
    calculators.get(stageId)?.(difficultyId);
    for (const link of difficultyLinks) {
      if (link.dataset.diyDifficultyLink === difficultyId) link.setAttribute('aria-current', 'true');
      else link.removeAttribute('aria-current');
    }
  };
  selector.addEventListener('click', event => {
    const link = event.target.closest('[data-diy-stage-link], [data-diy-difficulty-link]');
    if (!link || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    const url = new URL(link.href);
    show(url.searchParams.get('stage'), url.searchParams.get('difficulty'));
    history.pushState(null, '', url);
  });
  window.addEventListener('popstate', () => {
    const url = new URL(location.href);
    show(url.searchParams.get('stage') || selector.dataset.defaultStage,
         url.searchParams.get('difficulty') || '1');
  });
}
