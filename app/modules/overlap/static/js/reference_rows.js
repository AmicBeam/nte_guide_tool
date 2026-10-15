/* Presentation-only grouping; the pair planner continues to use every raw row. */
(function (root) {
    'use strict';
    function routeKey(row) {
        return JSON.stringify([
            row.original_actions, row.kind, row.switch_method, row.hold_reason,
            row.id.slice(16), row.preinput_action_indices,
            row.timeline.map((step) => [step.action, step.placement, step.detached,
                step.dodge_replaces, step.automatic, step.preinput_before_switch,
                step.terminal_switch, step.base_duration_seconds ?? step.duration_seconds]),
        ]);
    }
    function summarize(rows) {
        const groups = new Map();
        for (const row of rows) {
            const key = routeKey(row);
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push(row);
        }
        return rows.filter((row) => row.timeline.every((step) => !step.recovery_used_seconds)).map((row) => {
            const variants = groups.get(routeKey(row));
            return {...row, overlap_windows: row.overlap_windows.map((window, index) => {
                const durations = variants.map((variant) => variant.overlap_windows[index].duration_seconds);
                return {...window, min_duration_seconds: Math.min(...durations), max_duration_seconds: Math.max(...durations)};
            })};
        });
    }
    if (typeof module !== 'undefined' && module.exports) module.exports = {summarize, routeKey};
    else root.OverlapReferenceRows = {summarize};
})(typeof globalThis !== 'undefined' ? globalThis : this);
