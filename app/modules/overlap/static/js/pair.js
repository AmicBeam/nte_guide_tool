(() => {
    'use strict';
    const root = document.querySelector('.overlap-pair-page');
    const $ = id => document.getElementById(`pair-${id}`);
    const escape = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
    const n = v => Number(v.toFixed(3)).toLocaleString('zh-CN');
    const percent = v => `${n(v * 100)}%`;
    let dataset, worker, plans = [], shown = 20, summary = '';
    const allocationChoices = new Map();
    const pendingVariants = new Set();
    function bubble(step, offset, repeat, period, px, segments, role) {
        const start = offset + step.start_seconds + repeat * period;
        const end = offset + step.end_seconds + repeat * period;
        const input = step.input_seconds == null ? null : offset + step.input_seconds + repeat * period;
        const front = segments.some(s => s.role === role && start >= s.start + repeat * period - 1e-8 && start < s.end + repeat * period - 1e-8);
        const linger = step.linger_seconds ? ` · 含滞留 ${n(step.linger_seconds)}s，不增加伤害与资源` : '';
        const title = `${step.group ? `组 ${step.group} · ` : ''}${step.action} · 占用 ${n(start)}–${n(end)}s${linger}${input == null ? ' · 自动衔接' : ` · 输入 ${n(input)}s`}`;
        return `<span class="pair-bubble ${front ? '' : 'is-background'} ${repeat ? 'is-repeat' : ''}" style="left:${start * px}px;width:${step.duration_seconds * px}px" title="${escape(title)}"><span>${escape(step.action)}</span><small>${n(step.duration_seconds)}s${step.linger_seconds ? ` · 滞留${n(step.linger_seconds)}s` : ''}</small></span>`
            + (input == null ? '' : `<span class="pair-input ${repeat ? 'is-repeat' : ''}" style="left:${input * px}px" title="${escape(`${n(input)}s 输入 ${step.action}${input < start ? `；${n(start)}s 开始释放` : ''}`)}">◆</span>`);
    }
    function card(group, index) {
        const chars = [$('first').value, $('second').value].map(id => dataset.characters.find(c => c.id === id));
        const key = group.id;
        const selection = allocationChoices.get(key) ?? group.defaultVariant ?? 0;
        const variant = group.variants?.[selection];
        if (!variant) return `<article class="glass-card pair-plan" data-rate="${group.multiplierRate}" data-group="${escape(key)}" data-plan-index="${index}"><header><div><span class="pair-plan-number">方案 ${index + 1}</span><h2>${n(group.period)}<small> 秒 / 循环</small></h2></div><div class="pair-plan-yield"><strong>${percent(group.multiplierRate)}<small> / 秒</small></strong></div></header><p class="pair-variant-loading">正在准备动作组时间轴…</p></article>`;
        const plan = window.OverlapMultiEngine.materialize(variant ? {...group, ...variant} : group, chars);
        const px = Number($('zoom').value), period = plan.period, width = period * 2 * px;
        const valueText = `解 ${selection + 1} / ${group.variants?.length || 1} · ${chars[0].name} ${plan.foregroundSeconds[0].toFixed(1)}s / ${chars[1].name} ${plan.foregroundSeconds[1].toFixed(1)}s`;
        const allocation = group.variants?.length > 1 ? `<section class="pair-allocation" aria-label="本方案前台分配"><div class="pair-allocation-labels"><span>${escape(chars[0].name)}更多前台</span><output>${escape(valueText)}</output><span>${escape(chars[1].name)}更多前台</span></div><input type="range" class="pair-allocation-slider" min="0" max="${group.variants.length - 1}" step="1" value="${selection}" aria-label="方案${index + 1}前台分配" aria-valuetext="${escape(valueText)}"><p>各档均为合法代表解，保持动作、循环时间和倍率不变；按${escape(chars[1].name)}前台时长递增。</p></section>` : '';
        let ticks = '';
        const stride = period > 12 ? 2 : 1;
        for (let t = 0; t < period * 2; t += stride) ticks += `<span style="left:${t * px}px">${t}s</span>`;
        const seam = `<span class="pair-seam" style="left:${period * px}px"><b>↻ 下一轮 · ${n(period)}s</b></span>`;
        const rows = chars.map((character, globalRole) => {
            const role = plan.roles.indexOf(globalRole), row = plan.rows[role], offset = plan.offsets[role];
            let content = row.timeline.filter(step => offset + step.end_seconds > period).map(step => bubble(step, offset, -1, period, px, plan.segments, role)).join('');
            for (let repeat = 0; repeat < 2; repeat++) {
                content += plan.segments.filter(s => s.role === role).map(s => `<span class="pair-occupation ${repeat ? 'is-repeat' : ''}" style="left:${(s.start + repeat * period) * px}px;width:${(s.end - s.start) * px}px" title="前台 ${n(s.start + repeat * period)}–${n(s.end + repeat * period)}s"></span>`).join('');
                content += row.timeline.map(step => bubble(step, offset, repeat, period, px, plan.segments, role)).join('');
            }
            return `<div class="pair-lane-label pair-color-${globalRole ? 'b' : 'a'}"><img src="${escape(character.avatar)}" alt=""><strong>${escape(character.name)}</strong></div><div class="pair-lane pair-role-${globalRole}">${content}${seam}</div>`;
        }).join('');
        const switches = plan.segments.slice(1).map(s => ({time: s.start, role: s.role}));
        if (!plan.sameEnd) switches.push({time: period, role: 0});
        const switchText = switches.map(s => `${n(s.time)}s → ${chars[plan.roles[s.role]].name}`).join('　·　');
        const delays = plan.segments.filter(s => s.end > s.earliestEnd + 0.0005 && !(plan.sameEnd && s === plan.segments.at(-1)))
            .map(s => `${chars[plan.roles[s.role]].name}在 ${n(s.earliestEnd)}s 已可切人，延后至 ${n(s.end)}s`);
        const restarts = (plan.restartWarnings || []).map(w => `${chars[w.role].name} ${w.from} → ${w.to}：不属于已有衔接，须在末动作结束后等待至少 0.5s`);
        const metrics = `<aside class="pair-metrics" aria-label="每轮倍率"><div class="pair-metrics-heading">每轮攻击倍率</div>${chars.map((character, i) => `<div class="pair-metric pair-color-${i ? 'b' : 'a'}"><strong>${percent(plan.multipliers[i].atk || 0)}</strong><small class="pair-metric-rate">${percent((plan.multipliers[i].atk || 0) / period)} / 秒</small><small>${escape(character.name)} · 前台 ${plan.foregroundSeconds[i].toFixed(1)}s</small></div>`).join('')}</aside>`;
        const cutLines = switches.flatMap(s => [0, 1].filter(r => s.time + r * period < period * 2).map(r => `<span class="pair-cut" style="left:${(s.time + r * period) * px}px" title="${n(s.time + r * period)}s 切至${escape(chars[plan.roles[s.role]].name)}"></span>`)).join('');
        return `<article class="glass-card pair-plan" data-rate="${plan.multiplierRate}" data-group="${escape(key)}" data-plan-index="${index}"><header><div><span class="pair-plan-number">方案 ${index + 1}</span><h2>${n(period)}<small> 秒 / 循环</small></h2></div><div class="pair-plan-yield"><strong>${percent(plan.multiplierRate)}<small> / 秒</small></strong><span>双人合计 ${percent(plan.totalMultiplier)} / 循环</span><span class="pair-plan-count">${plan.groups.map(g => g.length).join(' / ')} 组 · ${plan.actions.map(a => a.length).join(' + ')} 个动作 · ${plan.sameEnd ? '同角色打满接回' : '异角色接回'}</span></div></header>
            ${allocation || (group.variants == null ? '<p class="pair-variant-loading">正在准备前台分配…</p>' : '')}<div class="pair-content"><div class="pair-timeline-layout"><div class="pair-timeline-scroll" tabindex="0" aria-label="方案${index + 1}时间轴，可横向滚动"><div class="pair-timeline" style="--pair-width:${width}px"><div class="pair-ruler-label">时间 / 秒</div><div class="pair-ruler">${ticks}${seam}</div>${rows}<div class="pair-cuts">${cutLines}</div></div></div>${metrics}</div>
            <p class="pair-switches">${escape(switchText)}</p>${delays.length ? `<p class="pair-delays">${escape(delays.join('；'))}。</p>` : ''}${restarts.length ? `<p class="pair-delays">${escape(restarts.join('；'))}。</p>` : ''}
            <details class="pair-detail"><summary>动作与循环衔接</summary>${chars.map((c, i) => `<p class="pair-color-${i ? 'b' : 'a'}"><strong>${escape(c.name)}</strong>　${plan.groups[i].map((g, j) => `<span class="pair-group-line">组 ${j + 1}：${escape(g.actions.join(' → '))}</span>`).join('')}</p>`).join('')}<p>${plan.sameEnd ? '末段完整释放后，由同一角色衔接下一轮；跨轮两次切人之间也满足 1.2 秒。' : '周期末切回起手角色，至下一次切人至少 1.2 秒；上轮最后一次切人至周期末同样至少 1.2 秒。'} 每位角色本轮动作均在其下一轮起手前释放完毕。</p></details></div></article>`;
    }
    function render() {
        const personalRate = (role) => (a, b) => (b.multipliers[role].atk || 0) / b.period - (a.multipliers[role].atk || 0) / a.period || b.multiplierRate - a.multiplierRate || a.period - b.period;
        const sorters = {'first-rate': personalRate(0), 'second-rate': personalRate(1), rate: (a, b) => b.multiplierRate - a.multiplierRate || a.period - b.period, actions: (a, b) => b.actions.flat().length - a.actions.flat().length || a.period - b.period, time: (a, b) => a.period - b.period};
        const sorted = [...plans].sort(sorters[$('sort').value]);
        const visible = sorted.slice(0, shown);
        $('results').innerHTML = visible.map(card).join('');
        for (const group of visible) if (group.variants == null && worker && !pendingVariants.has(group.id)) {
            pendingVariants.add(group.id); worker.postMessage({type: 'variants', planId: group.id});
        }
        $('more').hidden = shown >= plans.length;
        $('status').textContent = summary + (plans.length ? ` · 已显示 ${Math.min(shown, plans.length)} 个` : ' · 当前动作范围内没有符合条件的双角色循环（有条件动作已排除）。');
    }
    function calculate() {
        if (!dataset) return;
        if (worker) {worker.onmessage = null; worker.onerror = null; worker.terminate();}
        worker = null;
        plans = []; shown = 20; allocationChoices.clear(); pendingVariants.clear();
        $('results').innerHTML = ''; $('more').hidden = true;
        $('results').setAttribute('aria-busy', 'false');
        const chars = [$('first').value, $('second').value].map(id => dataset.characters.find(c => c.id === id));
        if (!chars[0] || !chars[1] || chars[0].id === chars[1].id) {
            $('status').textContent = '请选择两个不同的角色。'; return;
        }
        $('results').setAttribute('aria-busy', 'true');
        $('status').textContent = '正在匹配循环与首尾切人节点…';
        const activeWorker = new Worker(root.dataset.workerUrl);
        worker = activeWorker;
        worker.onmessage = ({data}) => {
            if (worker !== activeWorker) return;
            if (data.progress) $('status').textContent = data.progress.phase === 'prune' ? '正在合并相同方案并去除冗余…' : `正在匹配 ${data.progress.checked} / ${data.progress.total} 组路线…`;
            if (data.result) {
                plans = data.result.plans;
                summary = `匹配完成 · 检查 ${data.result.checked} 组路线 · 保留 ${plans.length} 个非冗余循环`;
                $('results').setAttribute('aria-busy', 'false');
                render();
            }
            if (data.variants) {
                const group = plans.find(p => p.id === data.planId);
                if (!group) return;
                group.variants = data.variants;
                group.defaultVariant = 0;
                const article = [...$('results').querySelectorAll('.pair-plan')].find(el => el.dataset.group === group.id);
                if (article) {
                    const expanded = article.querySelector('details')?.open || false;
                    const template = document.createElement('template');
                    template.innerHTML = card(group, Number(article.dataset.planIndex));
                    template.content.querySelector('details').open = expanded;
                    article.replaceWith(template.content.firstElementChild);
                }
            }
            if (data.error) fail(data.error);
        };
        worker.onerror = event => {
            if (worker !== activeWorker) return;
            console.error('合轴匹配错误', event.message);
            fail('匹配未能完成，请重新匹配或刷新页面。');
        };
        worker.postMessage({characters: chars, options: {withoutDodge: $('no-dodge').checked, maxGroups: 2, groupCount: $('group-count').value, containsGroups: [$('first-group').value, $('second-group').value]}});
    }
    function fail(message) {
        if (worker) worker.terminate(); worker = null;
        $('results').setAttribute('aria-busy', 'false');
        $('status').textContent = message;
    }
    function updateGroupOptions() {
        if (!dataset) return;
        for (const field of ['first', 'second']) {
            const character = dataset.characters.find(c => c.id === $(field).value), select = $(`${field}-group`);
            const previous = select.dataset.characterId === character.id ? select.value : '';
            const groups = window.OverlapMultiEngine.groupOptions(character, {withoutDodge: $('no-dodge').checked});
            select.innerHTML = '<option value="">不限动作组</option>' + groups.map(g => `<option value="${escape(g.key)}">${escape(g.actions.join(' → '))}</option>`).join('');
            select.value = groups.some(g => g.key === previous) ? previous : '';
            select.dataset.characterId = character.id;
            $(`${field}-group-label`).textContent = `${character.name}包含动作组`;
        }
    }
    function updateChoices() {
        for (const [own, other] of [['first', 'second'], ['second', 'first']]) {
            for (const option of $(own).options) option.disabled = option.value === $(other).value;
        }
        updateGroupOptions();
    }
    for (const field of ['first', 'second']) $(field).addEventListener('change', () => {updateChoices(); calculate();});
    $('no-dodge').addEventListener('change', () => {updateGroupOptions(); calculate();});
    for (const field of ['first-group', 'second-group', 'group-count']) $(field).addEventListener('change', calculate);
    $('calculate').addEventListener('click', calculate);
    $('swap').addEventListener('click', () => {
        const first = $('first').value;
        const filters = [$('first-group').value, $('second-group').value];
        $('first').value = $('second').value;
        $('second').value = first;
        updateChoices();
        $('first-group').value = filters[1]; $('second-group').value = filters[0];
        calculate();
    });
    for (const field of ['sort', 'zoom']) $(field).addEventListener('change', render);
    $('results').addEventListener('input', event => {
        const input = event.target;
        if (!input.matches('.pair-allocation-slider')) return;
        const article = input.closest('.pair-plan'), key = article.dataset.group;
        const group = plans.find(p => p.id === key);
        allocationChoices.set(key, Number(input.value));
        const template = document.createElement('template');
        template.innerHTML = card(group, Number(article.dataset.planIndex));
        const replacement = template.content.firstElementChild;
        const oldContent = article.querySelector('.pair-content');
        const scroll = oldContent.querySelector('.pair-timeline-scroll').scrollLeft;
        const expanded = oldContent.querySelector('details').open;
        oldContent.innerHTML = replacement.querySelector('.pair-content').innerHTML;
        oldContent.querySelector('.pair-timeline-scroll').scrollLeft = scroll;
        oldContent.querySelector('details').open = expanded;
        article.querySelector('.pair-plan-count').textContent = replacement.querySelector('.pair-plan-count').textContent;
        article.querySelector('output').textContent = replacement.querySelector('output').textContent;
        input.setAttribute('aria-valuetext', replacement.querySelector('input[type="range"]').getAttribute('aria-valuetext'));
    });
    $('more').addEventListener('click', () => {shown += 20; render();});
    fetch(root.dataset.tablesUrl).then(response => {if (!response.ok) throw new Error('无法读取动作表'); return response.json();}).then(data => {
        dataset = data;
        const options = data.characters.map(c => `<option value="${escape(c.id)}">${escape(c.name)}</option>`).join('');
        $('first').innerHTML = options; $('second').innerHTML = options;
        $('second').selectedIndex = 1;
        $('calculate').disabled = false;
        $('swap').disabled = false;
        updateChoices(); calculate();
    }).catch(error => fail(`${error.message}，请刷新重试。`));
})();
