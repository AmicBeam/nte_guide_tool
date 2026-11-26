/* Display stored level values only; no combat or growth formulas. */
'use strict';
(() => {
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
        if (group.dataset.abilityId) {
            for (const linked of document.querySelectorAll('[data-linked-ability]')) {
                if (linked.dataset.linkedAbility !== group.dataset.abilityId) continue;
                const values = JSON.parse(linked.dataset.linkedValues);
                linked.querySelectorAll('[data-linked-value]').forEach(value => { value.textContent = values[level - 1] ?? '未提供'; });
                linked.querySelectorAll('[data-linked-level]').forEach(value => { value.textContent = level; });
            }
        }
        const description = group.querySelector('[data-level-template]');
        if (description) {
            const parameters = JSON.parse(description.dataset.levelParameters);
            const resolved = description.dataset.levelTemplate.replace(/\{(\d+)\}/g, (token, id) => {
                const row = parameters.find(row => String(row.id_value) === id);
                return row?.values[level - 1] ?? token;
            });
            const fragment = document.createDocumentFragment();
            let previous = 0;
            for (const match of resolved.matchAll(/[+-]?\d+(?:\.\d+)?%?/g)) {
                fragment.append(document.createTextNode(resolved.slice(previous, match.index)));
                const number = document.createElement('span');
                number.className = 'guide-effect-number';
                number.textContent = match[0];
                fragment.append(number);
                previous = match.index + match[0].length;
            }
            fragment.append(document.createTextNode(resolved.slice(previous)));
            description.replaceChildren(fragment);
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

const skillCards = [...document.querySelectorAll('[data-ability-id][data-character-id]')];
const characterId = skillCards[0]?.dataset.characterId;
if (characterId) {
    const readToken = () => {
        try { return localStorage.getItem('nte_token') || ''; } catch (_) { return ''; }
    };
    let requestId = 0;
    const updateCombatDetails = async () => {
        const id = ++requestId;
        const token = readToken();
        const previous = [...document.querySelectorAll('.guide-combat-details')];
        const openMoves = new Set(previous.filter(detail => detail.open).map(detail => detail.parentElement));
        previous.forEach(detail => detail.remove());
        if (!token) return;

        try {
            const url = `/api/guide/characters/${encodeURIComponent(characterId)}?section=abilities&skill_level=1&hit_details=0`;
            const response = await fetch(url, {
                headers: { Authorization: `Bearer ${token}` }, cache: 'no-store', redirect: 'error',
            });
            if (!response.ok) return;
            const { data } = await response.json();
            if (id !== requestId || token !== readToken()) return;

            const abilities = new Map(data.abilities.map(ability => [ability.id, ability.groups]));
            for (const card of skillCards) {
                const groups = new Map((abilities.get(card.dataset.abilityId) || [])
                    .filter(group => group.combat_details).map(group => [group.name, group.combat_details]));
                for (const move of card.querySelectorAll('[data-move-name]')) {
                    const combat = groups.get(move.dataset.moveName);
                    if (!combat) continue;
                    const detail = document.createElement('details');
                    detail.className = 'guide-hit-details guide-combat-details';
                    detail.open = openMoves.has(move);
                    const summary = document.createElement('summary');
                    summary.textContent = '判定与时长';
                    const scroll = document.createElement('div');
                    scroll.className = 'guide-table-scroll';
                    scroll.tabIndex = 0;
                    scroll.setAttribute('role', 'region');
                    scroll.setAttribute('aria-label', `${move.dataset.moveName}判定与时长`);
                    const table = document.createElement('table');
                    table.className = 'guide-data-table guide-combat-table';
                    const headings = table.createTHead().insertRow();
                    for (const label of combat.columns) {
                        const heading = document.createElement('th');
                        heading.scope = 'col';
                        heading.textContent = label;
                        headings.append(heading);
                    }
                    const body = table.createTBody();
                    for (const [label, ...entries] of combat.rows) {
                        const row = body.insertRow();
                        const title = document.createElement('th');
                        title.scope = 'row';
                        title.textContent = label;
                        row.append(title);
                        for (const [index, entry] of entries.entries()) {
                            const cell = row.insertCell();
                            if (combat.columns[index + 1] === '伤害判定') cell.className = 'guide-combat-range';
                            cell.textContent = entry ?? '—';
                        }
                    }
                    scroll.append(table);
                    detail.append(summary, scroll);
                    for (const note of combat.notes || []) {
                        const paragraph = document.createElement('p');
                        paragraph.className = 'guide-combat-note';
                        paragraph.textContent = note;
                        detail.append(paragraph);
                    }
                    move.append(detail);
                }
            }
        } catch (_) { /* Keep the public character page usable if the authenticated read fails. */ }
    };
    updateCombatDetails();
    window.addEventListener('storage', event => {
        if (event.key === 'nte_token' || event.key === null) updateCombatDetails();
    });
    window.addEventListener('pageshow', event => {
        if (event.persisted) updateCombatDetails();
    });
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') updateCombatDetails();
    });
}

})();
