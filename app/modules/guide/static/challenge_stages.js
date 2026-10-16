'use strict';

for (const selector of document.querySelectorAll('[data-challenge-stage-selector]')) {
  const options = Array.from(selector.querySelectorAll('input[name="challenge-stage"]'));
  const stages = Array.from(selector.querySelectorAll('[data-challenge-stage]'));
  const show = (value) => {
    for (const stage of stages) stage.hidden = stage.dataset.challengeStage !== value;
  };
  selector.addEventListener('change', event => {
    if (!options.includes(event.target)) return;
    const value = event.target.value;
    show(value);
    const url = new URL(location.href);
    url.searchParams.set('stage', value);
    // Keep the user's position while making refresh/share restore the selection.
    history.replaceState(null, '', url);
  });
  window.addEventListener('popstate', () => {
    const value = new URL(location.href).searchParams.get('stage') ||
      (options.some(option => option.value === '12') ? '12' : options[0]?.value);
    const option = options.find(option => option.value === value);
    if (option) { option.checked = true; show(value); }
  });
}
