/* Pure browser-side scheduler. Integer milliseconds; no Flask or shaft runtime. */
(function (scope) {
    'use strict';
    const tick = (s) => Math.round(s * 1000);
    const seconds = (t) => t / 1000;
    const COOLDOWN = 1200;
    function eligible(row) {
        return !row.requires_three_characters && Number.isFinite(row.release_end_seconds)
            && row.release_end_seconds > 0 && row.foreground_intervals.length > 0
            && row.timeline.every(s => Number.isFinite(s.start_seconds) && Number.isFinite(s.end_seconds));
    }
    // x[v] <= x[u] + w. Negative cycles mean no legal arrangement.
    function solve(size, edges) {
        const values = Array(size).fill(0);
        for (let pass = 0; pass < size; pass++) {
            let changed = false;
            for (const [u, v, w] of edges) {
                if (values[v] > values[u] + w) {
                    values[v] = values[u] + w;
                    changed = true;
                }
            }
            if (!changed) return values.map(x => x - values[0]);
        }
        return null;
    }
    // Dual of max c*x under difference constraints: min-cost flow. Flow on a
    // constraint makes it tight; residual potentials recover an optimal schedule.
    function optimize(size, edges, objective) {
        const source = size, sink = size + 1, graph = Array.from({length: size + 2}, () => []);
        function add(u, v, cost, cap) {
            const forward = {v, cost, cap, reverse: graph[v].length};
            const backward = {v: u, cost: -cost, cap: 0, reverse: graph[u].length};
            graph[u].push(forward); graph[v].push(backward);
        }
        edges.forEach(([u, v, cost]) => add(u, v, cost, 100));
        let remaining = 0;
        objective.forEach((coefficient, node) => {
            if (coefficient < 0) add(source, node, 0, -coefficient);
            if (coefficient > 0) {add(node, sink, 0, coefficient); remaining += coefficient;}
        });
        while (remaining > 0) {
            const distances = Array(size + 2).fill(Infinity), previous = [];
            distances[source] = 0;
            for (let pass = 0; pass < size + 1; pass++) {
                let changed = false;
                graph.forEach((list, u) => list.forEach((edge, index) => {
                    if (edge.cap > 0 && distances[edge.v] > distances[u] + edge.cost) {
                        distances[edge.v] = distances[u] + edge.cost;
                        previous[edge.v] = [u, index]; changed = true;
                    }
                }));
                if (!changed) break;
            }
            if (!Number.isFinite(distances[sink])) throw new Error('无法优化前台分配');
            for (let v = sink; v !== source;) {
                const [u, index] = previous[v], edge = graph[u][index];
                edge.cap--; graph[v][edge.reverse].cap++; v = u;
            }
            remaining--;
        }
        return solve(size, graph.flatMap((list, u) => u < size ? list.filter(e => e.v < size && e.cap > 0).map(e => [u, e.v, e.cost]) : []));
    }
    // User 2026-10-15: restarting a basic attack outside the recorded graph
    // requires 0.5s after the previous own action finishes, even across switches.
    function restartDelay(character, row) {
        const first = row.timeline[0], last = row.timeline.at(-1);
        const next = character.actions.find(a => a.name === first.action);
        if (!next || next.type !== '普攻' || /z\d*$/.test(first.action)) return 0;
        const previous = character.actions.find(a => a.name === last.action);
        const names = row.original_actions || row.actions;
        const linked = (previous?.successors || []).some(edge => edge.to === first.action
            && (!edge.requires_previous || edge.requires_previous.includes(names.at(-2)))
            && (!edge.requires_prefix || JSON.stringify(edge.requires_prefix) === JSON.stringify(names))
            && (!edge.detach_previous || last.detached));
        return linked ? 0 : 0.5;
    }
    function schedule(first, second, restartDelays = [0, 0], foregroundRole = null) {
        if (!eligible(first) || !eligible(second)) return null;
        const rows = [first, second];
        const counts = rows.map(r => r.foreground_intervals.length);
        const sameEnd = counts[0] === counts[1] + 1;
        if (counts[0] !== counts[1] && !sameEnd) return null;
        if (sameEnd && first.hold_reason !== 'cycle_to_start') return null;
        const order = [];
        for (let i = 0; i < counts[0]; i++) {
            order.push({role: 0, index: i});
            if (i < counts[1]) order.push({role: 1, index: i});
        }
        // Nodes 0: cycle origin; 1/2: route starts; 3: cycle end;
        // all remaining nodes are actual handovers, including the cyclic end when needed.
        const cuts = [0, ...order.slice(1).map((_, i) => i + 4)];
        if (!sameEnd) cuts.push(3);
        const base = [[1, 0, 0], [2, 0, 0]]; // action input may follow the handover
        if (sameEnd) base.push([0, 1, 0]); // same-character seam starts with its next action
        order.forEach(({role, index}, i) => {
            const [start, end] = rows[role].foreground_intervals[index].map(tick);
            base.push([role + 1, cuts[i], start]); // enter no later than next required input
            if (i + 1 < cuts.length) base.push([cuts[i + 1], role + 1, -end]); // may leave later
        });
        const actualCuts = sameEnd ? cuts.slice(1) : cuts;
        for (let i = 1; i < actualCuts.length; i++) base.push([actualCuts[i], actualCuts[i - 1], -COOLDOWN]);
        const releaseLimits = rows.map(r => tick(r.release_end_seconds));
        const restartLimits = rows.map((r, role) => tick(r.timeline.at(-1).end_seconds)
            + tick(restartDelays[role]) - tick(r.timeline[0].input_seconds ?? r.timeline[0].start_seconds));
        const lower = Math.max(...releaseLimits, ...restartLimits);
        function constraints(period) {
            const edges = [...base, [0, 3, period], [3, 0, -period]];
            // In an A…A loop, the seam has no switch. Check last -> next first switch.
            if (sameEnd) edges.push([actualCuts[0], actualCuts.at(-1), period - COOLDOWN]);
            return edges;
        }
        const feasible = period => solve(order.length + 3, constraints(period));
        let low = lower, high = rows.reduce((sum, r) => sum + tick(r.release_end_seconds), 0) + COOLDOWN * (order.length + 1);
        if (!feasible(high)) return null;
        while (low < high) {
            const mid = Math.floor((low + high) / 2);
            if (feasible(mid)) high = mid; else low = mid + 1;
        }
        // Warn only when relaxing THIS restart rule permits a shorter cycle.
        // Existing cooldown/partner timing can naturally leave the same gap.
        const restartLimited = restartDelays.map((delay, role) => delay > 0
            && low === restartLimits[role]
            && low > Math.max(...releaseLimits, ...restartLimits.filter((_, i) => i !== role))
            && feasible(low - 1) !== null);
        let values = feasible(low);
        if (foregroundRole === 0 || foregroundRole === 1) {
            const objective = Array(order.length + 3).fill(0);
            order.forEach(({role}, i) => {
                if (role !== foregroundRole) return;
                objective[cuts[i]]--;
                objective[i + 1 < cuts.length ? cuts[i + 1] : 3]++;
            });
            values = optimize(order.length + 3, constraints(low), objective);
            if (!values) throw new Error('前台分配违反时序约束');
        }
        const offsets = [seconds(values[1]), seconds(values[2])];
        const segments = order.map(({role, index}, i) => ({
            role, index, start: seconds(values[cuts[i]]),
            end: seconds(i + 1 < cuts.length ? values[cuts[i + 1]] : low),
            earliestEnd: seconds(tick(offsets[role]) + tick(rows[role].foreground_intervals[index][1])),
        }));
        const period = seconds(low);
        // A pair must actually release damage while the other character occupies the field.
        let overlap = 0;
        rows.forEach((row, role) => row.timeline.forEach(step => {
            const start = offsets[role] + step.start_seconds, end = offsets[role] + step.end_seconds;
            for (const segment of segments.filter(s => s.role !== role)) {
                for (const shift of [0, period]) overlap += Math.max(0, Math.min(end, segment.end + shift) - Math.max(start, segment.start + shift));
            }
        }));
        if (overlap < 0.0005) return null;
        return {rows, offsets, segments, period, sameEnd, overlap, restartDelays, restartLimited};
    }
    function subsequence(small, large) {
        let index = 0;
        for (const action of large) if (action === small[index]) index++;
        return index === small.length;
    }
    function prune(plans) {
        // Cyclic rotations and different switch methods with identical actions share one result.
        const best = new Map();
        for (const plan of plans) {
            const key = JSON.stringify(plan.actions);
            const old = best.get(key);
            if (!old || plan.period < old.period) best.set(key, plan);
        }
        const unique = [...best.values()].sort((a, b) => a.period - b.period || b.actions.flat().length - a.actions.flat().length);
        return unique.filter((plan, i) => !unique.some((other, j) => j !== i && other.period <= plan.period + 1e-8
            && other.actions[0].length >= plan.actions[0].length && other.actions[1].length >= plan.actions[1].length
            && subsequence(plan.actions[0], other.actions[0]) && subsequence(plan.actions[1], other.actions[1])
            && (other.period < plan.period - 1e-8 || other.actions.flat().length > plan.actions.flat().length)));
    }
    function unrestricted(action) {
        const c = action.conditions || {};
        return !Object.entries(c).some(([key, value]) =>
            ((key.startsWith('required_') || key === 'forbidden_buff_key' || key === 'personal_resource_threshold')
                && value != null && value !== '' && (!Array.isArray(value) || value.length))
            || (['cooldown_ticks', 'energy_cost'].includes(key) && Number(value) > 0)
            || (key === 'personal_resource_cost' && Object.values(value || {}).some(v => Number(v) > 0)));
    }
    // User 2026-10-15: in this planner, Heiyu has only the magic-E gate.
    // Its previous OWN action must be magic-a5, including across the cycle seam.
    // This overrides imported shaft buff/malice conditions only in pair matching.
    function availableRows(character, options = {}) {
        const heiyu = character.name === '黑羽';
        const allowed = new Set(character.actions.filter(a => heiyu || unrestricted(a)).map(a => a.name));
        return character.rows.filter(r => eligible(r) && (!options.withoutDodge || !r.has_dodge)
            && (r.original_actions || r.actions).every(name => allowed.has(name))
            && (!heiyu || r.actions.every((name, i, names) => name !== '魔e' || names[(i + names.length - 1) % names.length] === '魔a5')));
    }
    function match(characters, options = {}, progress = () => {}) {
        if (characters.length !== 2 || characters[0].id === characters[1].id) throw new Error('请选择两个不同的角色');
        const candidates = characters.map(c => availableRows(c, options));
        const delays = new Map(characters.flatMap((c, i) => candidates[i].map(r => [r, restartDelay(c, r)])));
        const plans = [];
        let checked = 0;
        for (const a of candidates[0]) {
            for (const b of candidates[1]) {
                for (const reverse of [false, true]) {
                    const ordered = reverse ? [b, a] : [a, b];
                    const plan = schedule(...ordered, ordered.map(r => delays.get(r)));
                    if (plan) {
                        plan.foregroundSeconds = [0, 0];
                        for (const segment of plan.segments) plan.foregroundSeconds[(reverse ? [1, 0] : [0, 1])[segment.role]] += segment.end - segment.start;
                        plan.roles = reverse ? [1, 0] : [0, 1];
                        plan.actions = [a.actions, b.actions];
                        plan.multipliers = [a.multipliers, b.multipliers];
                        plan.totalMultiplier = (a.multipliers.atk || 0) + (b.multipliers.atk || 0);
                        plan.multiplierRate = plan.totalMultiplier / plan.period;
                        plans.push(plan);
                    }
                }
                checked++;
            }
            progress({checked, total: candidates[0].length * candidates[1].length});
        }
        const groups = prune(plans);
        const retained = new Map(groups.map(p => [JSON.stringify(p.actions), p]));
        const variants = new Map(groups.map(p => [p, new Map()]));
        function compact(plan) {
            return {rowIds: plan.rows.map(r => r.id), roles: plan.roles, offsets: plan.offsets,
                segments: plan.segments, period: plan.period, sameEnd: plan.sameEnd,
                restartDelays: plan.restartDelays, restartLimited: plan.restartLimited, overlap: plan.overlap,
                foregroundSeconds: [0, 1].map(role => seconds(plan.segments.filter(s => plan.roles[s.role] === role)
                    .reduce((sum, s) => sum + tick(s.end) - tick(s.start), 0)))};
        }
        function add(group, plan) {
            const value = compact(plan);
            // Anchor to character one's first input so rotating a cyclic timeline,
            // or using an equivalent row ID, does not create another slider stop.
            const anchor = tick(plan.offsets[plan.roles.indexOf(0)]);
            const period = tick(plan.period);
            const phase = time => ((tick(time) - anchor) % period + period) % period;
            const actions = [0, 1].map(role => {
                const local = plan.roles.indexOf(role), offset = plan.offsets[local];
                return plan.rows[local].timeline.map(step => [step.action,
                    step.input_seconds == null ? null : phase(offset + step.input_seconds),
                    phase(offset + step.start_seconds), tick(step.duration_seconds)]);
            });
            const switches = plan.segments.slice(plan.sameEnd ? 1 : 0)
                .map(segment => [phase(segment.start), plan.roles[segment.role]])
                .sort((a, b) => a[0] - b[0]);
            const key = JSON.stringify([actions, switches]);
            if (!variants.get(group).has(key)) variants.get(group).set(key, value);
            return key;
        }
        for (const group of groups) group.defaultVariantKey = add(group, group);
        for (const plan of plans) {
            const group = retained.get(JSON.stringify(plan.actions));
            if (!group || plan.period !== group.period) continue;
            add(group, plan);
            const extremes = [0, 1].map(role => {
                const optimized = schedule(...plan.rows, plan.restartDelays, role);
                if (optimized) {optimized.roles = plan.roles; add(group, optimized);}
                return optimized;
            });
            if (!extremes[0] || !extremes[1]) continue;
            const span = Math.abs(compact(extremes[0]).foregroundSeconds[1] - compact(extremes[1]).foregroundSeconds[1]);
            // Discrete representative allocations, <=0.1s between foreground-duration
            // levels, with both exact extrema retained. Interpolate only within the
            // SAME route and handover topology; never across incompatible solutions.
            const steps = Math.ceil(tick(span) / 100);
            for (let step = 1; step < steps; step++) {
                const ratio = step / steps;
                const blend = (a, b) => seconds(Math.round(tick(a) * (1 - ratio) + tick(b) * ratio));
                const left = extremes[0], right = extremes[1];
                const offsets = left.offsets.map((v, i) => blend(v, right.offsets[i]));
                const segments = left.segments.map((s, i) => ({...s, start: blend(s.start, right.segments[i].start),
                    end: blend(s.end, right.segments[i].end), earliestEnd: seconds(tick(offsets[s.role]) + tick(plan.rows[s.role].foreground_intervals[s.index][1]))}));
                let overlap = 0;
                plan.rows.forEach((row, role) => row.timeline.forEach(action => {
                    for (const s of segments.filter(s => s.role !== role)) for (const shift of [0, plan.period]) {
                        overlap += Math.max(0, Math.min(offsets[role] + action.end_seconds, s.end + shift) - Math.max(offsets[role] + action.start_seconds, s.start + shift));
                    }
                }));
                if (overlap > 0.0005) add(group, {...plan, offsets, segments, overlap});
            }
        }
        for (const group of groups) {
            const entries = [...variants.get(group)].sort(([, a], [, b]) => a.foregroundSeconds[1] - b.foregroundSeconds[1]);
            group.defaultVariant = entries.findIndex(([key]) => key === group.defaultVariantKey);
            group.variants = entries.map(([, value]) => value);
            delete group.defaultVariantKey;
        }
        return {plans: groups, checked, feasible: plans.length};
    }
    scope.OverlapPairEngine = {schedule, match, prune, subsequence, eligible, unrestricted, availableRows, restartDelay};
    if (typeof module !== 'undefined') module.exports = scope.OverlapPairEngine;
}(typeof self !== 'undefined' ? self : globalThis));
