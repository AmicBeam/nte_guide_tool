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
    function schedule(first, second, restartDelays = [0, 0]) {
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
        const lower = Math.max(...rows.map((r, role) => Math.max(tick(r.release_end_seconds),
            tick(r.timeline.at(-1).end_seconds) + tick(restartDelays[role]) - tick(r.timeline[0].input_seconds ?? r.timeline[0].start_seconds))));
        function feasible(period) {
            const edges = [...base, [0, 3, period], [3, 0, -period]];
            // In an A…A loop, the seam has no switch. Check last -> next first switch.
            if (sameEnd) edges.push([actualCuts[0], actualCuts.at(-1), period - COOLDOWN]);
            return solve(order.length + 3, edges);
        }
        let low = lower, high = rows.reduce((sum, r) => sum + tick(r.release_end_seconds), 0) + COOLDOWN * (order.length + 1);
        if (!feasible(high)) return null;
        while (low < high) {
            const mid = Math.floor((low + high) / 2);
            if (feasible(mid)) high = mid; else low = mid + 1;
        }
        const values = feasible(low);
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
        return {rows, offsets, segments, period, sameEnd, overlap, restartDelays};
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
        return {plans: prune(plans), checked, feasible: plans.length};
    }
    scope.OverlapPairEngine = {schedule, match, prune, subsequence, eligible, unrestricted, availableRows, restartDelay};
    if (typeof module !== 'undefined') module.exports = scope.OverlapPairEngine;
}(typeof self !== 'undefined' ? self : globalThis));
