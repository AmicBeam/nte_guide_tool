'use strict';

for (const selector of document.querySelectorAll('[data-diy-selector]')) {
  const stageLinks = [...selector.querySelectorAll('[data-diy-stage-link]')];
  const stages = [...selector.querySelectorAll('[data-diy-stage]')];
  const difficultyLinks = [...selector.querySelectorAll('[data-diy-difficulty-link]')];
  const panels = [...selector.querySelectorAll('[data-diy-difficulty]')];
  const show = (stageId, difficultyId) => {
    if (!stages.some(stage => stage.dataset.diyStage === stageId)) return;
    for (const stage of stages) stage.hidden = stage.dataset.diyStage !== stageId;
    for (const panel of panels) panel.hidden = panel.dataset.diyDifficulty !== difficultyId;
    for (const link of stageLinks) {
      if (link.dataset.diyStageLink === stageId) link.setAttribute('aria-current', 'true');
      else link.removeAttribute('aria-current');
      link.href = `?stage=${encodeURIComponent(link.dataset.diyStageLink)}&difficulty=${encodeURIComponent(difficultyId)}`;
    }
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
