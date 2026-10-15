(() => {
    'use strict';
    const root = document.querySelector('.overlap-pair-page');
    const $ = id => document.getElementById(`pair-${id}`);
    const escape = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
    const n = v => Number(v.toFixed(3)).toLocaleString('zh-CN');
    const percent = v => `${n(v * 100)}%`;
    let dataset, worker, plans = [], shown = 20, summary = '';
    function bubble(step, offset, repeat, period, px, segments, role) {
        const start = offset + step.start_seconds + repeat * period;
        const end = offset + step.end_seconds + repeat * period;
        const input = step.input_seconds == null ? null : offset + step.input_seconds + repeat * period;
        const front = segments.some(s => s.role === role && start >= s.start + repeat * period - 1e-8 && start < s.end + repeat * period - 1e-8);
        const title = `${step.action} · 释放 ${n(start)}–${n(end)}s${input == null ? ' · 自动衔接' : ` · 输入 ${n(input)}s`}`;
        return `<span class="pair-bubble ${front ? '' : 'is-background'} ${repeat ? 'is-repeat' : ''}" style="left:${start * px}px;width:${step.duration_seconds * px}px" title="${escape(title)}"><span>${escape(step.action)}</span><small>${n(step.duration_seconds)}s</small></span>`
            + (input == null ? '' : `<span class="pair-input ${repeat ? 'is-repeat' : ''}" style="left:${input * px}px" title="${escape(`${n(input)}s 输入 ${step.action}${input < start ? `；${n(start)}s 开始释放` : ''}`)}">◆</span>`);
    }
    function card(plan, index) {
        const px = Number($('zoom').value), period = plan.period, width = period * 2 * px;
        const chars = [$('first').value, $('second').value].map(id => dataset.characters.find(c => c.id === id));
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
        const metrics = `<aside class="pair-metrics" aria-label="每轮倍率"><div class="pair-metrics-heading">每轮攻击倍率</div>${chars.map((character, i) => `<div class="pair-metric pair-color-${i ? 'b' : 'a'}"><strong>${percent(plan.multipliers[i].atk || 0)}</strong><small>${escape(character.name)}</small></div>`).join('')}</aside>`;
        const cutLines = switches.flatMap(s => [0, 1].filter(r => s.time + r * period < period * 2).map(r => `<span class="pair-cut" style="left:${(s.time + r * period) * px}px" title="${n(s.time + r * period)}s 切至${escape(chars[plan.roles[s.role]].name)}"></span>`)).join('');
        return `<article class="glass-card pair-plan" data-rate="${plan.multiplierRate}"><header><div><span class="pair-plan-number">方案 ${index + 1}</span><h2>${n(period)}<small> 秒 / 循环</small></h2></div><div class="pair-plan-yield"><strong>${percent(plan.multiplierRate)}<small> / 秒</small></strong><span>双人合计 ${percent(plan.totalMultiplier)} / 循环</span><span class="pair-plan-count">${plan.actions.map(a => a.length).join(' + ')} 个动作 · ${plan.sameEnd ? '同角色打满接回' : '异角色接回'}</span></div></header>
            <div class="pair-timeline-layout"><div class="pair-timeline-scroll" tabindex="0" aria-label="方案${index + 1}时间轴，可横向滚动"><div class="pair-timeline" style="--pair-width:${width}px"><div class="pair-ruler-label">时间 / 秒</div><div class="pair-ruler">${ticks}${seam}</div>${rows}<div class="pair-cuts">${cutLines}</div></div></div>${metrics}</div>
            <p class="pair-switches">${escape(switchText)}</p>${delays.length ? `<p class="pair-delays">${escape(delays.join('；'))}。</p>` : ''}
            <details class="pair-detail"><summary>动作与循环衔接</summary>${chars.map((c, i) => `<p class="pair-color-${i ? 'b' : 'a'}"><strong>${escape(c.name)}</strong>　${escape(plan.actions[i].join(' → '))}</p>`).join('')}<p>${plan.sameEnd ? '末段完整释放后，由同一角色衔接下一轮；跨轮两次切人之间也满足 1.2 秒。' : '周期末切回起手角色，至下一次切人至少 1.2 秒；上轮最后一次切人至周期末同样至少 1.2 秒。'} 每位角色本轮动作均在其下一轮起手前释放完毕。</p></details></article>`;
    }
    function render() {
        const sorters = {rate: (a, b) => b.multiplierRate - a.multiplierRate || a.period - b.period, actions: (a, b) => b.actions.flat().length - a.actions.flat().length || a.period - b.period, time: (a, b) => a.period - b.period};
        const sorted = [...plans].sort(sorters[$('sort').value]);
        $('results').innerHTML = sorted.slice(0, shown).map(card).join('');
        $('more').hidden = shown >= plans.length;
        $('status').textContent = summary + (plans.length ? ` · 已显示 ${Math.min(shown, plans.length)} 个` : ' · 当前动作范围内没有符合条件的双角色循环（有条件动作已排除）。');
    }
    function calculate() {
        if (!dataset) return;
        if (worker) worker.terminate();
        worker = null;
        plans = []; shown = 20;
        $('results').innerHTML = ''; $('more').hidden = true;
        $('results').setAttribute('aria-busy', 'false');
        const chars = [$('first').value, $('second').value].map(id => dataset.characters.find(c => c.id === id));
        if (!chars[0] || !chars[1] || chars[0].id === chars[1].id) {
            $('status').textContent = '请选择两个不同的角色。'; return;
        }
        $('results').setAttribute('aria-busy', 'true');
        $('status').textContent = '正在匹配循环与首尾切人节点…';
        worker = new Worker(root.dataset.workerUrl);
        worker.onmessage = ({data}) => {
            if (data.progress) $('status').textContent = `正在匹配 ${data.progress.checked} / ${data.progress.total} 组路线…`;
            if (data.result) {
                plans = data.result.plans;
                summary = `匹配完成 · 检查 ${data.result.checked} 组路线 · 保留 ${plans.length} 个非冗余循环`;
                $('results').setAttribute('aria-busy', 'false');
                worker.terminate(); worker = null; render();
            }
            if (data.error) fail(data.error);
        };
        worker.onerror = () => fail('匹配未能完成，请重新匹配或刷新页面。');
        worker.postMessage({characters: chars, options: {withoutDodge: $('no-dodge').checked}});
    }
    function fail(message) {
        if (worker) worker.terminate(); worker = null;
        $('results').setAttribute('aria-busy', 'false');
        $('status').textContent = message;
    }
    function updateChoices() {
        for (const [own, other] of [['first', 'second'], ['second', 'first']]) {
            for (const option of $(own).options) option.disabled = option.value === $(other).value;
        }
    }
    for (const field of ['first', 'second']) $(field).addEventListener('change', () => {updateChoices(); calculate();});
    $('no-dodge').addEventListener('change', calculate);
    $('calculate').addEventListener('click', calculate);
    for (const field of ['sort', 'zoom']) $(field).addEventListener('change', render);
    $('more').addEventListener('click', () => {shown += 20; render();});
    fetch(root.dataset.tablesUrl).then(response => {if (!response.ok) throw new Error('无法读取动作表'); return response.json();}).then(data => {
        dataset = data;
        const options = data.characters.map(c => `<option value="${escape(c.id)}">${escape(c.name)}</option>`).join('');
        $('first').innerHTML = options; $('second').innerHTML = options;
        $('second').selectedIndex = 1;
        $('calculate').disabled = false;
        updateChoices(); calculate();
    }).catch(error => fail(`${error.message}，请刷新重试。`));
})();
