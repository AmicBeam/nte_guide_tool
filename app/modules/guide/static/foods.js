'use strict';
(() => {
  const dialog = document.getElementById('food-detail-dialog');
  if (!dialog) return;
  const content = dialog.querySelector('[data-food-content]');
  let trigger = null;
  let previousOverflow = '';
  function open(id, button) {
    const template = document.getElementById(`food-detail-${id}`);
    if (!(template instanceof HTMLTemplateElement)) return;
    trigger = button || document.querySelector(`[data-food-open="${CSS.escape(id)}"]`);
    content.replaceChildren(template.content.cloneNode(true));
    dialog.scrollTop = 0;
    if (!dialog.open) {
      previousOverflow = document.body.style.overflow;
      document.body.style.overflow = 'hidden';
      dialog.showModal();
    }
    const url = new URL(location.href);
    url.searchParams.set('food', id);
    history.replaceState(null, '', url);
  }
  document.addEventListener('click', event => {
    const button = event.target.closest('[data-food-open]');
    if (button) open(button.dataset.foodOpen, button);
  });
  dialog.querySelector('[data-food-close]').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => {
    document.body.style.overflow = previousOverflow;
    const url = new URL(location.href);
    url.searchParams.delete('food');
    history.replaceState(null, '', url);
    if (trigger?.isConnected && !trigger.hidden) trigger.focus();
    else document.getElementById('guide-query')?.focus();
    content.replaceChildren();
  });
  const id = new URL(location.href).searchParams.get('food');
  if (id) open(id);
})();
