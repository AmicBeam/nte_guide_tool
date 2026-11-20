'use strict';
const assert = require('node:assert/strict');
const {summarize, routeKey} = require('../app/modules/overlap/static/js/reference_rows.js');
const data = require('../app/modules/overlap/static/data/tables.json');
for (const character of data.characters) {
    const before = JSON.stringify(character.rows);
    const zeroRows = character.rows.filter(row => row.timeline.every(step => !step.recovery_used_seconds));
    assert.equal(new Set(zeroRows.map(routeKey)).size, zeroRows.length, 'distinct zero-duration routes must not share an identity');
    const result = summarize(character.rows);
    assert.equal(result.length, zeroRows.length);
    for (const row of result) {
        const original = zeroRows.find(candidate => candidate.id === row.id);
        const {overlap_windows, ...rest} = row;
        const {overlap_windows: originalWindows, ...originalRest} = original;
        assert.deepEqual(rest, originalRest, 'all metrics and timeline data come from the zero-duration route');
        const variants = character.rows.filter(candidate => routeKey(candidate) === routeKey(row));
        assert(variants.every(candidate => JSON.stringify(candidate.multipliers) === JSON.stringify(row.multipliers)));
        for (let index = 0; index < overlap_windows.length; index++) {
            const window = overlap_windows[index];
            const durations = variants.map(candidate => candidate.overlap_windows[index].duration_seconds);
            assert.equal(window.min_duration_seconds, originalWindows[index].duration_seconds);
            assert.equal(window.max_duration_seconds, Math.max(...durations));
        }
    }
    assert.equal(JSON.stringify(character.rows), before, 'raw planner candidates remain unchanged');
    console.log(`${character.name}: ${character.rows.length} → ${result.length}`);
}
const heiyu = data.characters.find(character => character.id === 'char_heiyu');
const result = summarize(heiyu.rows);
assert(result.some(row => row.overlap_windows.some(window => window.min_duration_seconds === 1.5 && window.max_duration_seconds === 2)), 'the 0.5s recovery window is retained');
const sample = heiyu.rows.find(row => row.timeline.some(step => step.recovery_used_seconds));
assert.deepEqual(summarize([sample]), [], 'a positive-duration-only route is not presented as a zero-duration route');
console.log('Reference route aggregation passed');
