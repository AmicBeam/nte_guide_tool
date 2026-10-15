/* Two distinct action groups per actor, once each. Pure browser-side scheduling. */
(function (scope) {
    'use strict';
    const core = scope.OverlapPairEngine || (typeof require === 'function' ? require('./pair_engine.js') : null);
    const ms = s => Math.round(s * 1000), sec = t => t / 1000;
    const key = row => JSON.stringify(row.actions);
    const tailDurations = new WeakMap();
    let groupMetadata = new Map();
    const metadata = row => {const id = key(row); if (!groupMetadata.has(id)) groupMetadata.set(id, {key: id, actions: row.actions}); return groupMetadata.get(id);};
    const firstInput = row => ms(row.timeline[0].input_seconds ?? row.timeline[0].start_seconds);
    function rawRows(character, options = {}) {
        const heiyu = character.name === '黑羽';
        const allowed = new Set(character.actions.filter(a => heiyu || core.unrestricted(a)).map(a => a.name));
        return character.rows.filter(r => core.eligible(r) && (!options.withoutDodge || !r.has_dodge)
            && (r.original_actions || r.actions).every(n => allowed.has(n))
            && (!heiyu || r.actions.every((n, i, names) => n !== '魔e' || i === 0 || names[i - 1] === '魔a5')));
    }
    function groupOptions(character, options = {}) {
        return [...new Map(rawRows(character, options).map(r => [key(r), {key: key(r), actions: r.actions}])).values()];
    }
    function transition(character, previous, next) {
        if (character.name === '黑羽' && next.actions[0] === '魔e' && previous.actions.at(-1) !== '魔a5') return null;
        // restartDelay needs the previous LAST action and its original prefix.
        const composite = {...previous, timeline: [next.timeline[0], previous.timeline.at(-1)]};
        return ms(core.restartDelay(character, composite));
    }
    function pool(character, options) {
        const rows = rawRows(character, options);
        // Same inputs/releases and a smaller mandatory foreground set can always
        // reproduce a larger set by staying on field; retain only minimal sets.
        const signatures = new Map();
        for (const row of rows) {
            const signature = JSON.stringify(row.timeline.map(s => [s.action, s.input_seconds, s.start_seconds, s.end_seconds, !!s.detached]));
            if (!signatures.has(signature)) signatures.set(signature, []);
            signatures.get(signature).push(row);
        }
        return [...signatures.values()].flatMap(group => group.filter((row, i) => !group.some((other, j) => j !== i
            && other.foreground_intervals.length === row.foreground_intervals.length
            && other.foreground_intervals.every(([l, r]) => row.foreground_intervals.some(([a, b]) => a <= l && b >= r))
            && (other.foreground_seconds < row.foreground_seconds - 1e-8 || (other.foreground_seconds === row.foreground_seconds && j < i)))));
    }
    function withTailDuration(row, duration) {
        if (Math.abs(row.timeline.at(-1).duration_seconds - duration) < 1e-9) return row;
        const timeline = row.timeline.map((step, i) => i + 1 === row.timeline.length
            ? {...step, duration_seconds: duration, end_seconds: step.start_seconds + duration, detached: true} : step);
        const release = Math.max(...timeline.map(step => step.end_seconds));
        const foreground_intervals = row.foreground_intervals.map(([start, end]) => [start, Math.min(end, release)]);
        return {...row, id: `${row.id.split('~')[0]}~${duration}`, timeline, foreground_intervals,
            release_end_seconds: release, foreground_seconds: foreground_intervals.reduce((sum,[a,b])=>sum+b-a,0)};
    }
    function rowById(character, id) {
        const [base, duration] = id.split('~'), row = character.rows.find(r => r.id === base);
        return duration == null ? row : withTailDuration(row, Number(duration));
    }
    function adaptTail(character, row, next) {
        const last = row.timeline.at(-1), action = character.actions.find(a => a.name === last.action);
        if (!action?.successors.some(edge => edge.to === next.actions[0] && edge.detach_previous)) return row;
        // Reuse a recorded detached transition's release duration. Crossing a group
        // boundary does not turn magic-a5 -> magic-E back into the full a5 timing.
        if (!tailDurations.has(character)) tailDurations.set(character, new Map());
        const cache = tailDurations.get(character), signature = `${last.action}>${next.actions[0]}`;
        if (!cache.has(signature)) {
            const durations = character.rows.flatMap(candidate => candidate.timeline.slice(0, -1).flatMap((step, i) =>
                step.action === last.action && step.detached && candidate.timeline[i + 1].action === next.actions[0] ? [step.duration_seconds] : []));
            cache.set(signature, durations.length ? Math.min(...durations) : null);
        }
        return cache.get(signature) == null ? row : withTailDuration(row, cache.get(signature));
    }
    function sequences(character, options, required) {
        const rows = pool(character, options), result = [];
        function add(sequence) {
            sequence = sequence.map((row, i) => adaptTail(character, row, sequence[(i + 1) % sequence.length]));
            if (required && !sequence.some(r => key(r) === required)) return;
            const gaps = sequence.map((r, i) => transition(character, r, sequence[(i + 1) % sequence.length]));
            if (gaps.some(g => g === null)) return;
            result.push({rows: sequence, actions: sequence.flatMap(r => r.actions), gaps, groups: sequence.map(key),
                lower: sequence.reduce((sum, r, i) => sum + Math.max(ms(r.release_end_seconds), ms(r.timeline.at(-1).end_seconds) + gaps[i] - firstInput(sequence[(i + 1) % sequence.length])), 0)});
        }
        rows.forEach(r => add([r]));
        if (options.maxGroups !== 1) for (let i = 0; i < rows.length; i++) for (let j = i + 1; j < rows.length; j++) {
            if (key(rows[i]) !== key(rows[j])) add([rows[i], rows[j]]);
        }
        return result;
    }
    function orders(sequences) {
        const queues = sequences.map((sequence, role) => sequence.rows.flatMap((row, occurrence) => row.foreground_intervals.map((_, segment) => ({role, occurrence, segment}))));
        const result = [];
        function walk(indices, current) {
            if (indices[0] === queues[0].length && indices[1] === queues[1].length) {result.push(current); return;}
            for (let role = 0; role < 2; role++) {
                const next = queues[role][indices[role]];
                if (!next) continue;
                const previous = current.at(-1);
                if (!previous && role !== 0) continue; // canonical cycle anchor
                if (previous?.role === role && previous.occurrence === next.occurrence) continue;
                const counts = [...indices]; counts[role]++;
                walk(counts, [...current, next]);
            }
        }
        walk([0, 0], []);
        return result;
    }
    function model(sequences, order) {
        const jobs = [], queues = [[], []];
        sequences.forEach((sequence, role) => sequence.rows.forEach((row, occurrence) => {
            queues[role].push(jobs.length); jobs.push({row, role, occurrence});
        }));
        const blocks = [];
        order.forEach(item => {
            if (blocks.at(-1)?.role !== item.role) blocks.push({role: item.role, items: []});
            blocks.at(-1).items.push({...item, job: queues[item.role][item.occurrence]});
        });
        if (blocks.length < 2) return null;
        const sameEnd = blocks[0].role === blocks.at(-1).role;
        const cuts = [0, ...blocks.slice(1).map((_, i) => jobs.length + 1 + i)];
        const size = jobs.length + blocks.length;
        const constant = [], periodic = [], links = [];
        jobs.forEach((_, j) => constant.push([j + 1, 0, 0]));
        blocks.forEach((block, i) => block.items.forEach(item => {
            const [start, end] = jobs[item.job].row.foreground_intervals[item.segment].map(ms);
            constant.push([item.job + 1, cuts[i], start]);
            if (i + 1 < blocks.length) constant.push([cuts[i + 1], item.job + 1, -end]);
            else periodic.push([0, item.job + 1, -end]);
        }));
        const switches = sameEnd ? cuts.slice(1) : cuts;
        for (let i = 1; i < switches.length; i++) constant.push([switches[i], switches[i - 1], -1200]);
        if (sameEnd) periodic.push([switches[0], switches.at(-1), -1200]);
        else periodic.push([0, cuts.at(-1), -1200]);
        // Ending with the starting actor requires a full release before restarting.
        if (sameEnd) {
            const actualLast = queues[0].at(-1);
            periodic.push([0, actualLast + 1, -ms(jobs[actualLast].row.release_end_seconds)]);
        }
        sequences.forEach((sequence, role) => sequence.rows.forEach((row, occurrence) => {
            const next = (occurrence + 1) % sequence.rows.length;
            const required = Math.max(ms(row.release_end_seconds), ms(row.timeline.at(-1).end_seconds) + sequence.gaps[occurrence] - firstInput(sequence.rows[next]));
            const normal = ms(row.release_end_seconds);
            const entry = [queues[role][next] + 1, queues[role][occurrence] + 1, -required];
            const wrap = next === 0;
            (wrap ? periodic : constant).push(entry);
            links.push({entry, wrap, required, normal, gap: sequence.gaps[occurrence], role,
                from: row.actions.at(-1), to: sequence.rows[next].actions[0]});
        }));
        return {size, constant, periodic, jobs, queues, cuts, blocks, sameEnd, links, sequences, order};
    }
    function edgesAt(model, period, ignore = null) {
        return [...model.constant, ...model.periodic.map(e => [e[0], e[1], e[2] + period])].map(e => {
            if (ignore && e[0] === ignore.entry[0] && e[1] === ignore.entry[1]
                && e[2] === -ignore.required + (ignore.wrap ? period : 0)) return [e[0], e[1], -ignore.normal + (ignore.wrap ? period : 0)];
            return e;
        });
    }
    // Exact minimum period for constraints with constant edges and edges (w+T).
    // Collapse constant paths, then use Karp's minimum cycle mean on wrap edges.
    function minimum(model) {
        if (!core.solve(model.size, model.constant)) return null;
        const p = model.periodic, n = p.length;
        const paths = new Map();
        for (const [, end] of p) {
            if (paths.has(end)) continue;
            const distances = Array(model.size).fill(Infinity); distances[end] = 0;
            for (let pass = 1; pass < model.size; pass++) {
                let changed = false;
                for (const [u, v, w] of model.constant) if (distances[v] > distances[u] + w) {distances[v] = distances[u] + w; changed = true;}
                if (!changed) break;
            }
            paths.set(end, distances);
        }
        const dp = [Array(n).fill(0)];
        for (let k = 1; k <= n; k++) {
            const next = Array(n).fill(Infinity);
            for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) next[j] = Math.min(next[j], dp[k - 1][i] + paths.get(p[i][1])[p[j][0]] + p[j][2]);
            dp.push(next);
        }
        let mean = Infinity;
        for (let v = 0; v < n; v++) if (Number.isFinite(dp[n][v])) {
            let largest = -Infinity;
            for (let k = 0; k < n; k++) if (Number.isFinite(dp[k][v])) largest = Math.max(largest, (dp[n][v] - dp[k][v]) / (n - k));
            mean = Math.min(mean, largest);
        }
        const period = Math.max(1, Math.ceil(-mean - 1e-7));
        const values = core.solve(model.size, edgesAt(model, period));
        return values ? {period, values} : null;
    }
    function realize(model, period, values, characters, warnings = false) {
        const jobs = model.jobs.map((j, i) => ({role: j.role, rowId: j.row.id, occurrence: j.occurrence, offset: sec(values[i + 1])}));
        const segments = model.blocks.map((block, i) => ({role: block.role, index: i,
            start: sec(values[model.cuts[i]]), end: sec(i + 1 < model.cuts.length ? values[model.cuts[i + 1]] : period),
            earliestEnd: sec(Math.max(...block.items.map(item => values[item.job + 1] + ms(model.jobs[item.job].row.foreground_intervals[item.segment][1]))))}));
        const offsets = [0, 1].map(role => jobs.find(j => j.role === role).offset);
        let overlap = 0;
        model.jobs.forEach((job, index) => job.row.timeline.forEach(step => {
            const offset = sec(values[index + 1]);
            for (const segment of segments.filter(s => s.role !== job.role)) for (const shift of [-sec(period), 0, sec(period)]) overlap += Math.max(0,
                Math.min(offset + step.end_seconds, segment.end + shift) - Math.max(offset + step.start_seconds, segment.start + shift));
        }));
        if (overlap < 0.0005) return null;
        const groups = model.sequences.map(s => s.rows.map(metadata));
        const multipliers = model.sequences.map(s => ({atk: s.rows.reduce((sum, r) => sum + (r.multipliers.atk || 0), 0)}));
        const restartWarnings = warnings ? model.links.filter(link => link.gap && core.solve(model.size, edgesAt(model, period - 1, link)))
            .map(link => ({role: link.role, from: link.from, to: link.to})) : [];
        return {jobs, offsets, roles: [0, 1], segments, period: sec(period), sameEnd: model.sameEnd, overlap,
            groups, actions: model.sequences.map(s => s.actions || s.rows.flatMap(r => r.actions)), multipliers, restartWarnings,
            foregroundSeconds: [0, 1].map(role => sec(segments.filter(s => s.role === role).reduce((sum, s) => sum + ms(s.end) - ms(s.start), 0))),
            totalMultiplier: multipliers[0].atk + multipliers[1].atk,
            multiplierRate: (multipliers[0].atk + multipliers[1].atk) / sec(period)};
    }
    function prune(plans) {
        const unique = new Map();
        for (const plan of plans) {
            const signature = plan.id || JSON.stringify(plan.actions), old = unique.get(signature);
            if (!old || plan.period < old.period) unique.set(signature, plan);
        }
        const all = [...unique.values()];
        const catalogs = [0, 1].map(role => [...new Map(all.map(p => [JSON.stringify(p.actions[role]), p.actions[role]])).entries()]);
        const supers = catalogs.map(catalog => new Map(catalog.map(([id, actions]) => [id, catalog.filter(([, candidate]) => candidate.length >= actions.length && core.subsequence(actions, candidate)).map(([k]) => k)])));
        const matrix = new Map();
        for (const plan of all) {
            const [a, b] = plan.actions.map(actions => JSON.stringify(actions));
            if (!matrix.has(a)) matrix.set(a, new Map());
            if (!matrix.get(a).has(b)) matrix.get(a).set(b, []);
            matrix.get(a).get(b).push(plan);
        }
        return all.filter(plan => {
            const [a, b] = plan.actions.map(actions => JSON.stringify(actions));
            for (const superA of supers[0].get(a)) {
                const row = matrix.get(superA);
                for (const superB of supers[1].get(b)) {
                    for (const other of row?.get(superB) || []) if (other !== plan && other.period <= plan.period
                        && (other.period < plan.period || superA !== a || superB !== b)) return false;
                }
            }
            return true;
        });
    }
    function match(characters, options = {}, progress = () => {}) {
        if (characters.length !== 2 || characters[0].id === characters[1].id) throw new Error('请选择两个不同的角色');
        const choices = characters.map((c, role) => sequences(c, options, options.containsGroups?.[role] || ''));
        groupMetadata = new Map();
        const codes = new Map(characters.flatMap(c => groupOptions(c, options)).map((g, i) => [g.key, i]));
        const idFor = sequences => sequences.map(s => s.groups.map(g => codes.get(g)).sort((a,b)=>a-b).join('+')).join('|');
        const best = new Map(), total = choices[0].length * choices[1].length;
        let checked = 0, feasible = 0;
        for (const a of choices[0]) {
            for (const initialB of choices[1]) {
                if ((options.groupCount === 'single' && (a.rows.length !== 1 || initialB.rows.length !== 1))
                    || (options.groupCount === 'with-two' && a.rows.length === 1 && initialB.rows.length === 1)
                    || (options.groupCount === 'both-two' && (a.rows.length !== 2 || initialB.rows.length !== 2))) {checked++; continue;}
                const id = idFor([a, initialB]), previous = best.get(id);
                if (previous && Math.max(a.lower, initialB.lower) > ms(previous.period)) {checked++; continue;}
                const rotations = initialB.rows.length === 1 ? [initialB] : [initialB,
                    {...initialB, rows: [...initialB.rows].reverse(), actions: [...initialB.rows].reverse().flatMap(r => r.actions), gaps: [...initialB.gaps].reverse(), groups: [...initialB.groups].reverse()}];
                for (const b of rotations) for (const order of orders([a, b])) {
                    const m = model([a, b], order), solution = minimum(m);
                    if (!solution) continue;
                    if (previous && solution.period > ms(previous.period)) continue;
                    const plan = realize(m, solution.period, solution.values, characters);
                    if (!plan) continue;
                    feasible++;
                    const current = best.get(id);
                    if (!current || plan.period < current.period) {
                        plan.id = id;
                        plan.recipe = {sequences: m.sequences.map(s => s.rows.map(r => r.id)), order: m.order};
                        best.set(id, plan);
                    }
                }
                checked++;
            }
            progress({checked, total});
        }
        progress({checked, total, phase: 'prune'});
        const candidates = [...best.values()];
        // Group boundaries are part of identity. Compare ordered action content for
        // Pareto pruning only after the requested contains-group filters apply.
        const retained = prune(candidates);
        for (const plan of retained) {
            plan.variants = null; // computed lazily for displayed pages
        }
        return {plans: retained, checked, feasible};
    }
    function materialize(plan, characters) {
        if (!plan.jobs) return plan;
        const rows = [0, 1].map(role => {
            const timeline = [];
            plan.jobs.filter(j => j.role === role).forEach(job => {
                const row = rowById(characters[role], job.rowId), shift = job.offset - plan.offsets[role];
                row.timeline.forEach(step => timeline.push({...step, start_seconds: step.start_seconds + shift,
                    end_seconds: step.end_seconds + shift, input_seconds: step.input_seconds == null ? null : step.input_seconds + shift, group: job.occurrence + 1}));
            });
            return {timeline, actions: timeline.map(s => s.action)};
        });
        return {...plan, rows};
    }
    function variants(plan, characters) {
        const sequences = plan.recipe.sequences.map((ids, role) => {
            const rows = ids.map(id => rowById(characters[role], id));
            return {rows, actions: rows.flatMap(r => r.actions), groups: rows.map(key), gaps: rows.map((r, i) => transition(characters[role], r, rows[(i + 1) % rows.length]))};
        });
        const m = model(sequences, plan.recipe.order), period = ms(plan.period);
        const extremes = [0, 1].map(role => {
            const objective = Array(m.size).fill(0);
            m.blocks.forEach((b, i) => {if (b.role === role) {objective[m.cuts[i]]--; if (i + 1 < m.cuts.length) objective[m.cuts[i + 1]]++;}});
            objective[0] -= objective.reduce((sum, x) => sum + x, 0);
            return core.optimize(m.size, edgesAt(m, period), objective);
        });
        const values = [core.solve(m.size, edgesAt(m, period)), ...extremes];
        const ends = extremes.map(v => realize(m, period, v, characters));
        if (ends[0] && ends[1]) {
            const steps = Math.ceil(Math.abs(ms(ends[0].foregroundSeconds[1]) - ms(ends[1].foregroundSeconds[1])) / 100);
            for (let i = 1; i < steps; i++) values.push(extremes[0].map((v, j) => Math.round(v * (1 - i / steps) + extremes[1][j] * i / steps)));
        }
        const unique = new Map();
        for (const value of values) {
            const variant = realize(m, period, value, characters, true);
            if (!variant) continue;
            const signature = JSON.stringify([variant.jobs.map(j => j.offset), variant.segments.map(s => [s.start, s.end])]);
            delete variant.rows;
            if (!unique.has(signature)) unique.set(signature, variant);
        }
        return [...unique.values()].sort((a, b) => a.foregroundSeconds[1] - b.foregroundSeconds[1]);
    }
    scope.OverlapMultiEngine = {match, variants, materialize, groupOptions, rawRows, transition, rowById, prune, sequences, model, minimum, realize};
    if (typeof module !== 'undefined') module.exports = scope.OverlapMultiEngine;
}(typeof self !== 'undefined' ? self : globalThis));
