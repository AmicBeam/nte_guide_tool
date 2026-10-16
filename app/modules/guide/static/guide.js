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

const toc = document.querySelector('[data-guide-toc]');
const tocToggle = document.querySelector('[data-toc-toggle]');
if (toc && tocToggle) {
    const links = [...toc.querySelectorAll('a[href^="#"]')];
    const entries = links.map(link => ({link, target: document.getElementById(link.hash.slice(1))})).filter(entry => entry.target);
    const setOpen = open => {
        toc.dataset.open = String(open);
        tocToggle.setAttribute('aria-expanded', String(open));
    };
    tocToggle.addEventListener('click', () => setOpen(toc.dataset.open !== 'true'));
    for (const link of links) link.addEventListener('click', () => setOpen(false));
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && toc.dataset.open === 'true') {
            setOpen(false);
            tocToggle.focus();
        }
    });
    document.addEventListener('click', event => {
        if (!toc.contains(event.target) && !tocToggle.contains(event.target)) setOpen(false);
    });
    let queued = false;
    const updateCurrent = () => {
        queued = false;
        let current = entries[0];
        for (const entry of entries) {
            if (entry.target.getBoundingClientRect().top <= 100) current = entry;
        }
        for (const entry of entries) {
            if (entry === current) entry.link.setAttribute('aria-current', 'location');
            else entry.link.removeAttribute('aria-current');
        }
    };
    window.addEventListener('scroll', () => {
        if (!queued) {
            queued = true;
            requestAnimationFrame(updateCurrent);
        }
    }, {passive: true});
    window.addEventListener('hashchange', updateCurrent);
    updateCurrent();
}
