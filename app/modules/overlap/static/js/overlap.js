/* Render/filter the precomputed dataset. No scheduling or benefit calculations. */
(() => {
    'use strict';
    const root = document.querySelector('.overlap-page');
    const $ = (id) => document.getElementById(`overlap-${id}`);
    const escape = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
    const number = (value, digits = 3) => value == null ? '待补' : Number(value).toLocaleString('zh-CN', {maximumFractionDigits: digits});
    const seconds = (value) => `${number(value)}${value == null ? '' : 's'}`;
    const bases = {atk: '攻击', hp: '生命', def: '防御', flat: '固定'};
    const multiplier = (values) => Object.entries(values).filter(([, value]) => value !== 0 && value != null).map(([key, value]) => key === 'flat' ? `${number(value)}固定` : `${number(value * 100)}%${bases[key]}`).join(' + ') || '—';
    const isCycleHold = (row) => row.switch_method === 'hold' && row.hold_reason === 'cycle_to_start';
    let dataset;
    let selected;
    let shown = 60;
    let generation = 0;
    const actionFilters = new Map();

    function updateActionFilters() {
        const names = [...new Set(selected.rows.flatMap((row) => row.timeline.map((step) => step.action)))];
        for (const name of actionFilters.keys()) {
            if (!names.includes(name)) actionFilters.delete(name);
        }
        $('contains').innerHTML = names.map((name) => {
            const mode = actionFilters.get(name) || '';
            return `<label class="overlap-contains-option" data-mode="${mode}"><span>${escape(name)}</span><select data-action="${escape(name)}" aria-label="${escape(name)}筛选"><option value=""${!mode ? ' selected' : ''}>不限</option><option value="include"${mode === 'include' ? ' selected' : ''}>包含</option><option value="exclude"${mode === 'exclude' ? ' selected' : ''}>排除</option></select></label>`;
        }).join('');
    }

    function fillActionOptions(field, names) {
        const previous = $(field).value;
        $(field).innerHTML = '<option value="">全部</option>' + names.map((name) => `<option value="${escape(name)}">${escape(name)}</option>`).join('');
        $(field).value = names.includes(previous) ? previous : '';
    }

    function updateLastActions() {
        if (!selected) return;
        const first = $('first').value;
        const rows = selected.rows.filter((row) => !first || row.timeline[0].action === first);
        fillActionOptions('last', [...new Set(rows.map((row) => row.timeline[row.timeline.length-1].action))]);
    }

    function renderActions() {
        $('actions').innerHTML = selected.actions.map((action) => {
            const predecessors = [...new Set([
                ...selected.actions.filter((source) => source.successors.some((edge) => edge.to === action.name)).map((source) => source.name),
                ...selected.rows.flatMap((row) => row.timeline.flatMap((step, index) => step.action === action.name && index > 0 ? [row.timeline[index-1].action] : [])),
            ])];
            const entryLabel = action.can_start ? '可作为起手' : predecessors.length ? `需从 ${predecessors.join('、')} 衔接` : '衔接来源待补';
            const successors = action.successors.map((edge) => `${edge.to}${edge.automatic ? '（自动）' : ''}${edge.can_preinput === false ? '（仅顺接，不支持合轴）' : ''}${edge.requires_previous ? `（${edge.requires_previous.join('/')}插重击后）` : ''}`).join('、') || '尚无后续记录';
            const points = action.switch_points.map((point) => `${point.label || point.id} ${seconds(point.after_seconds)}`).join('；');
            const preinputs = action.successors.filter((edge) => edge.can_preinput !== false && !edge.automatic);
            const preinputText = !preinputs.length ? '无后续预输入合轴记录'
                : preinputs.some((edge) => edge.input_delay_seconds != null || edge.switch_delay_seconds)
                    ? preinputs.map((edge) => {
                        const input = edge.input_delay_seconds ?? action.preinput_seconds;
                        const cut = input + (edge.switch_delay_seconds ?? 0);
                        return `${edge.to}：动作开始后 ${seconds(input)} 预输入，开始后 ${seconds(cut)} 可切人`;
                    }).join('；')
                    : `开始后 ${seconds(action.preinput_seconds)} 可预输入下一段并切人`;
            const recovery = action.recovery_seconds ? `<p>恢复后摇 ${seconds(action.recovery_seconds)}：合法衔接可提前接，重新起手须等后摇结束；后台接前台可在窗口内回场。</p>` : '';
            return `<article class="overlap-action"><h3>${escape(action.name)} <small>${escape(seconds(action.duration_seconds))}</small></h3><p>${escape(entryLabel)}${action.preinput_forbidden ? ' · 不可被预输入' : ''}${action.locks_foreground_switch ? ' · 必须前台打完' : ''}</p><p>衔接：${escape(successors)}</p><p>${escape(preinputText)}</p>${recovery}${points ? `<p>${escape(points)}</p>` : ''}${action.gaps.length ? `<p class="overlap-pending">待补：${escape(action.gaps.join('、'))}</p>` : ''}</article>`;
        }).join('');
    }

    function rowHtml(row) {
        const windows = row.overlap_windows;
        const bubbles = row.timeline.map((step, index) => {
            const previous = row.timeline[index - 1];
            let separator = index ? '<span class="overlap-chip-arrow" aria-hidden="true">→</span>' : '';
            if (previous && step.placement !== previous.placement) {
                separator = `<span class="overlap-chip-switch">${step.placement === 'background' ? '切走' : '回场'}</span>`;
            }
            const dodged = Boolean(step.dodge_replaces);
            const label = step.action + (step.terminal_switch ? '切' : '');
            const badges = `${step.preinput_before_switch ? '<small>预</small>' : ''}${step.detached ? '<small>脱手</small>' : ''}${step.recovery_used_seconds ? `<small>后摇${seconds(step.recovery_used_seconds)}</small>` : ''}`;
            const hint = `${step.action}${dodged ? `（跳过${step.dodge_replaces}全部收益）` : ''} · ${step.placement === 'background' ? '后台释放' : '前台起手'} · 输入${step.automatic ? '自动' : seconds(step.input_seconds)} · 动作${seconds(step.start_seconds)}–${seconds(step.start_seconds == null ? null : step.start_seconds + (step.base_duration_seconds ?? step.duration_seconds))}${step.recovery_used_seconds ? ` · 恢复后摇${seconds(step.recovery_used_seconds)}` : ''}`;
            return `${separator}<span class="overlap-action-chip ${step.placement === 'background' ? 'is-background' : ''} ${dodged ? 'is-dodge' : ''}" title="${escape(hint)}" aria-label="${escape(hint)}">${escape(label)}${badges}</span>`;
        }).join('');
        const timeline = row.timeline.map((step) => `<tr><td>${escape(step.action)}</td><td>${escape(step.automatic ? '自动' : seconds(step.input_seconds))}</td><td>${escape(seconds(step.start_seconds))}</td><td>${escape(seconds(step.start_seconds == null ? null : step.start_seconds + (step.base_duration_seconds ?? step.duration_seconds)))}</td><td>${step.recovery_seconds ? `${escape(seconds(step.recovery_used_seconds))} / ${escape(seconds(step.recovery_seconds))}` : '—'}</td></tr>`).join('');
        const personal = Object.entries(row.personal_resources).map(([name, value]) => `${name} ${value > 0 ? '+' : ''}${number(value)}`).join('，');
        const foregroundHtml = `<div class="overlap-foreground-times"><strong>${row.foreground_seconds == null ? '待补' : `${number(row.foreground_seconds)}秒`}</strong><small class="overlap-release-end" title="动作结束时间">${escape(seconds(row.release_end_seconds))}</small></div>`;
        const phases = [
            ...row.foreground_segments.map((segment) => ({...segment, placement: 'foreground'})),
            ...windows.map((window) => ({...window, placement: 'background'})),
        ].sort((first, second) => first.start_seconds - second.start_seconds);
        const windowHtml = `<div class="overlap-duration-list">${phases.map((phase) => {
            const title = `${phase.placement === 'background' ? '回场后台窗口' : '前台耗时'}：${seconds(phase.duration_seconds)}（${seconds(phase.start_seconds)}–${seconds(phase.end_seconds)}）`;
            return `<span class="overlap-action-chip overlap-time-bubble ${phase.placement === 'background' ? 'is-background' : ''}" title="${escape(title)}" aria-label="${escape(title)}">${number(phase.duration_seconds)}秒</span>`;
        }).join('')}</div>`;
        const windowDetail = `<div class="overlap-window-details">${row.foreground_segments.map((segment) => `<p><strong>前台${segment.number}：${number(segment.duration_seconds)}秒</strong>（${seconds(segment.start_seconds)}–${seconds(segment.end_seconds)}）</p>`).join('')}${windows.map((window) => `<p><strong>回场后台窗口${window.number}：${number(window.duration_seconds)}秒</strong><br>${seconds(window.start_seconds)}切走 → ${seconds(window.end_seconds)}回场续连段。切人冷却为1.2秒，可在${seconds(window.return_start_seconds)}–${seconds(window.return_end_seconds)}切回；本表按${seconds(window.end_seconds)}回场。</p>`).join('')}</div>`;
        const returnDetail = windows.length ? `<p>${row.requires_three_characters
            ? `回场人数：需三人合轴（第${row.three_character_segments.join('、')}段前台耗时不足1.2秒，且以切人收尾）。`
            : '回场人数：两人可合轴。'}</p>` : '';
        const detailId = `overlap-detail-${row.id}`;
        const columns = root.querySelector('.overlap-table > thead > tr').children.length;
        return `<tr class="overlap-main-row"><td><details class="overlap-route"><summary aria-controls="${escape(detailId)}" aria-expanded="false"><strong class="${isCycleHold(row) ? 'is-cycle-hold' : ''}">${escape(row.operation)}</strong><span class="overlap-bubble-route">${bubbles}</span><span class="overlap-badge">${escape(row.switch_label)}${row.has_dodge ? ' · 闪避替代' : ''}</span></summary></details></td><td class="overlap-emphasis">${foregroundHtml}</td><td class="overlap-window">${windowHtml}</td><td>${multiplier(row.multipliers)}</td><td>${multiplier(row.multiplier_rates)}</td><td>${number(row.harmony)}${row.harmony_on_start ? ` + ${number(row.harmony_on_start)}释放` : ''}</td><td>${row.harmony_rate == null ? '—' : number(row.harmony_rate)}</td><td>${number(row.energy_gain)}</td><td>${row.energy_rate == null ? '—' : number(row.energy_rate)}</td><td>${number(row.stagger)}</td></tr><tr class="overlap-detail-row" id="${escape(detailId)}" hidden><td colspan="${columns}"><div class="overlap-route-detail">${returnDetail}${windowDetail}<p>${row.ends_with_switch ? `切人时刻 ${escape(seconds(row.switch_seconds))}` : `前台完整释放至 ${escape(seconds(row.release_end_seconds))}`}，整条路线释放结束 ${escape(seconds(row.release_end_seconds))}。</p><table aria-label="动作输入与释放时刻"><thead><tr><th>动作</th><th>输入</th><th>开始释放</th><th>本体结束</th><th>已用后摇 / 上限</th></tr></thead><tbody>${timeline}</tbody></table><p>耗能 ${number(row.energy_cost)}，返还 ${number(row.energy_return)}，净能量 ${number(row.energy_net)}；释放时环合 ${number(row.harmony_on_start)}${personal ? `；${escape(personal)}` : ''}。</p>${row.notes.map((note) => `<p>${escape(note)}</p>`).join('')}</div></td></tr>`;
    }

    function render() {
        if (!selected) return;
        const kind = $('kind').value;
        const dodge = $('dodge').value;
        const returnFilter = $('return').value;
        const firstAction = $('first').value;
        const lastAction = $('last').value;
        const sort = $('sort').value;
        const rows = selected.rows.filter((row) => (kind === 'all' || row.kind === kind || kind === '合轴' && ['分段合轴','打满'].includes(row.kind)
            || kind === 'hold' && row.switch_method === 'hold'
            || kind === 'cycle_hold' && isCycleHold(row)
            || kind === 'normal' && row.switch_method === 'normal' || kind === 'preinput' && row.switch_method === 'preinput')
            && (returnFilter === 'all' || returnFilter === 'with' && Boolean(row.window_1)
                || returnFilter === 'without' && !row.window_1
                || returnFilter === 'three' && Boolean(row.window_1) && row.requires_three_characters
                || returnFilter === 'two' && Boolean(row.window_1) && !row.requires_three_characters)
            && (dodge === 'all' || row.has_dodge === (dodge === 'with'))
            && (!firstAction || row.timeline[0].action === firstAction)
            && (!lastAction || row.timeline[row.timeline.length-1].action === lastAction)
            && [...actionFilters].every(([name, mode]) => row.timeline.some((step) => step.action === name) === (mode === 'include')));
        $('contains-clear').disabled = actionFilters.size === 0;
        const metric = (row) => sort === 'multiplier_rate' ? row.multiplier_rates.atk : row[sort];
        rows.sort((a, b) => {
            const first = metric(a), second = metric(b);
            if (first == null || second == null) return (first == null) - (second == null);
            return (sort === 'foreground_seconds' ? first - second : second - first) || a.id.localeCompare(b.id);
        });
        $('title').textContent = `${selected.name} · ${rows.length} 条操作`;
        $('status').textContent = `${selected.element}属性 · ${selected.coverage.actions} 个动作 · ${selected.coverage.rows} 条已生成操作 · ${selected.dodge_preserves_combo ? '支持闪避保留连段' : '不支持闪避保留连段'}`;
        $('rows').innerHTML = rows.slice(0, shown).map(rowHtml).join('');
        $('empty').hidden = rows.length > 0;
        $('more').hidden = rows.length <= shown;
        $('more').textContent = `显示更多（已显示 ${Math.min(shown, rows.length)} / ${rows.length}）`;
    }

    function selectCharacter(character) {
        selected = character;
        shown = 60;
        fillActionOptions('first', [...new Set(character.rows.map((row) => row.timeline[0].action))]);
        updateLastActions();
        updateActionFilters();
        document.querySelectorAll('[data-character]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.character === character.id)));
        const url = new URL(window.location.href);
        url.searchParams.set('character', character.id);
        history.replaceState(null, '', url);
        renderActions();
        render();
    }

    async function load() {
        const current = ++generation;
        try {
            const response = await fetch(root.dataset.tablesUrl);
            if (!response.ok) throw new Error('读取收益表失败');
            const data = await response.json();
            if (current !== generation) return;
            dataset = data;
            $('characters').innerHTML = data.characters.map((character) => `<button type="button" class="overlap-character" data-character="${escape(character.id)}" aria-pressed="false"><img src="${escape(character.avatar)}" alt="" loading="lazy"><span>${escape(character.name)}</span></button>`).join('');
            document.querySelectorAll('[data-character]').forEach((button) => button.addEventListener('click', () => selectCharacter(data.characters.find((character) => character.id === button.dataset.character))));
            const id = new URL(location.href).searchParams.get('character');
            selectCharacter(data.characters.find((character) => character.id === id) || data.characters[0]);
        } catch (error) {
            $('status').textContent = `${error.message}，请刷新重试。`;
        }
    }
    ['kind', 'return', 'dodge', 'sort', 'first', 'last'].forEach((name) => $(name).addEventListener('change', () => {
        if (name === 'first') updateLastActions();
        shown = 60;
        render();
    }));
    $('contains').addEventListener('change', (event) => {
        const input = event.target;
        if (!input.matches('select[data-action]')) return;
        if (input.value) actionFilters.set(input.dataset.action, input.value);
        else actionFilters.delete(input.dataset.action);
        input.closest('label').dataset.mode = input.value;
        shown = 60;
        render();
    });
    $('contains-clear').addEventListener('click', () => {
        actionFilters.clear();
        updateActionFilters();
        shown = 60;
        render();
    });
    $('rows').addEventListener('toggle', (event) => {
        const disclosure = event.target;
        if (!disclosure.matches('.overlap-route')) return;
        const summary = disclosure.querySelector('summary');
        const detailRow = disclosure.closest('tr').nextElementSibling;
        detailRow.hidden = !disclosure.open;
        summary.setAttribute('aria-expanded', String(disclosure.open));
    }, true);
    $('more').addEventListener('click', () => { shown += 60; render(); });
    load();
})();
