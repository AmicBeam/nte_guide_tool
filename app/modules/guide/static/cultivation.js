'use strict';

const cultivationForm = document.querySelector('form[data-boundary-levels]');
if (cultivationForm) {
  const boundaries = JSON.parse(cultivationForm.dataset.boundaryLevels);
  for (const field of cultivationForm.querySelectorAll('[data-boundary-side]')) {
    const level = cultivationForm.elements.namedItem(`${field.dataset.boundarySide}_level`);
    const select = field.querySelector('select');
    const update = () => {
      const boundary = boundaries.includes(Number(level.value));
      field.hidden = !boundary;
      select.disabled = !boundary;
    };
    level.addEventListener('input', () => {
      select.value = '0';
      update();
    });
    update();
  }
}
