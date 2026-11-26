const fs = require('node:fs');

const [filename, stepId, tickText] = process.argv.slice(2);
if (!filename || !stepId || !/^\d+$/.test(tickText || '')) {
  console.error('用法 node move-step.js <排轴 JSON> <步骤 ID> <新 start_tick>');
  process.exit(2);
}

try {
  const document = JSON.parse(fs.readFileSync(filename, 'utf8'));
  const axis = document.axis ?? document;
  const matches = axis.steps.filter(step => step.id === stepId);
  const tick = Number(tickText);
  if (matches.length !== 1 || !Number.isSafeInteger(tick)) {
    throw new Error('步骤 ID 必须唯一，且 start_tick 须为有效整数。');
  }
  matches[0].start_tick = tick;
  axis.steps.sort((left, right) => left.start_tick - right.start_tick);
  delete document.result;
  delete document.axis_sha256;
  delete document.self_check;
  console.log(JSON.stringify(document, null, 2));
} catch {
  console.error('编辑失败，请核对排轴 JSON、唯一的步骤 ID 和 start_tick。');
  process.exitCode = 1;
}
