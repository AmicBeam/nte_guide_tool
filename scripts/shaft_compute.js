#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const engine = require(path.resolve(__dirname, '../app/modules/shaft/static/js/shaft_engine.js'));

function readStdin() {
  return fs.readFileSync(0, 'utf8');
}

try {
  const input = JSON.parse(readStdin() || '{}');
  const result = input.operation === 'analyze_substats'
    ? engine.analyzeSubstatContributions(input.axis || {}, input.catalog || {}, input.options || {})
    : engine.simulateAxis(input.axis || {}, input.catalog || {});
  process.stdout.write(`${JSON.stringify({ ok: true, result })}\n`);
} catch (error) {
  process.stdout.write(`${JSON.stringify({ ok: false, error: error && error.message ? error.message : String(error) })}\n`);
  process.exitCode = 1;
}
