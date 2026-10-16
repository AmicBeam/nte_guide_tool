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
