const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const context = vm.createContext({document: {querySelector: () => null}, URL,
  location: {href: 'https://example.invalid/guide'}});
vm.runInContext(fs.readFileSync('app/modules/guide/static/cultivation.js', 'utf8'), context);
const read = context.readGuidePrivateResponse;
const response = (status, body, extra = {}) => ({status, ok: status < 400,
  json: async () => JSON.parse(body), ...extra});

test('private data and delete success', async () => {
  assert.equal(await read(response(200, '{"data":42}')), 42);
  assert.equal(await read(response(200, '{"ok":true}')), undefined);
});
test('known JSON errors retain readable messages', async () => {
  await assert.rejects(read(response(503, '{"error":{"message":"服务未就绪"}}')), /服务未就绪/);
});
test('HTML and malformed responses never expose parser or page contents', async () => {
  for (const status of [200, 500, 502]) {
    await assert.rejects(read(response(status, '<!doctype html>private-page')), /网站服务暂时不可用/);
  }
  await assert.rejects(read(response(404, '<html>')), /同步接口暂不可用/);
  await assert.rejects(read(response(200, '{')), /网站服务暂时不可用/);
  await assert.rejects(read(response(200, '{}')), /资料格式异常/);
});
test('expired login and login redirect', async () => {
  await assert.rejects(read(response(401, '<html>')), /请先登录本站/);
  await assert.rejects(read(response(200, '<html>', {redirected: true, url: 'https://example.invalid/login?next=x'})), /请先登录本站/);
});


test('saved display uses weapon mix rather than character slev and includes skill levels', () => {
  const lines = context.guideSavedCharacterLines({level: 80, awaken_level: 2, resonance_level: 6,
    skills: [{id: 'melee', level: 8}, {id: 'skill', name: '变轨攻击', level: null}],
    equipment: {fork: {name: '测试弧盘', alev: '80', slev: '5'}, suit: {}}}, {Melee: '普通攻击'}).join('\n');
  assert.match(lines, /混频 5/);
  assert.doesNotMatch(lines, /混频 6/);
  assert.match(lines, /普通攻击：8级/);
  assert.match(lines, /变轨攻击：等级未提供/);
  for (const slev of ['6', '', null]) {
    assert.match(context.guideSavedCharacterLines({equipment: {fork: {slev}}}).join('\n'), /混频 未提供/);
  }
});
