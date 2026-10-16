/* Display stored level values only; no combat or growth formulas. */
'use strict';
for (const group of document.querySelectorAll('[data-level-group]')) {
    const control = group.querySelector('[data-level-control]');
    const output = group.querySelector('[data-level-output]');
    if (!control) continue;
    const cells = [...group.querySelectorAll('[data-level-values]')].map(cell => ({
        cell, id: cell.dataset.valueId, values: JSON.parse(cell.dataset.levelValues),
    }));
    const update = () => {
        const level = Number(control.value);
        output.textContent = String(level);
        for (const inline of group.querySelectorAll('[data-level-inline]')) inline.textContent = String(level);
        control.setAttribute('aria-valuetext', `等级 ${level}`);
        for (const {cell, values} of cells) cell.textContent = values[level - 1] ?? '未提供';
        const description = group.querySelector('[data-level-template]');
        if (description) {
            description.textContent = description.dataset.levelTemplate.replace(/\{(\d+)\}/g, (token, id) => {
                const row = cells.find(cell => cell.id === id);
                return row?.values[level - 1] ?? token;
            });
        }
    };
    control.addEventListener('input', update);
    control.addEventListener('change', update);
    update();
}
