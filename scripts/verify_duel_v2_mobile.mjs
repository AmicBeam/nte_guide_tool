#!/usr/bin/env node
/**
 * Mobile/layout verification for V2 table.
 * Run by the main agent, not this worker. Uses a fresh Playwright context.
 *
 * Env:
 *   BASE_URL            default http://127.0.0.1:5001
 *   PLAYER_UID          login player id
 *   LOGIN_CODE          login password/code
 *   OUTPUT_DIR          optional screenshot/json dump directory
 *   CHROME_EXECUTABLE   optional Chromium path
 *   NODE_PATH           may point at an installed Playwright
 *   ALLOW_RESET_TEST_ROOM=1  required to leave an existing room and start a new one
 */
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

function fail(message) {
  throw new Error(message);
}

function env(name, fallback) {
  const value = process.env[name];
  return value == null || value === '' ? fallback : value;
}

function assertLocalBase(url) {
  const parsed = new URL(url);
  const host = parsed.hostname;
  const allowed = host === '127.0.0.1' || host === 'localhost' || host === '::1';
  if (!allowed && process.env.ALLOW_REMOTE_VERIFY !== '1') {
    fail('BASE_URL must be localhost unless ALLOW_REMOTE_VERIFY=1');
  }
}

function clipped(rect, viewport) {
  return rect.top < -1 || rect.left < -1 || rect.bottom > viewport.height + 1 || rect.right > viewport.width + 1;
}

async function launchBrowser(playwright) {
  const executablePath = env('CHROME_EXECUTABLE', undefined);
  return playwright.chromium.launch({
    headless: true,
    executablePath,
  });
}

async function login(page, baseUrl, uid, code) {
  await page.goto(baseUrl + '/login', { waitUntil: 'domcontentloaded' });
  await page.fill('#player-uid-input', uid);
  await page.fill('#code-input', code);
  await Promise.all([
    page.waitForURL((url) => !String(url).includes('/login'), { timeout: 20000 }),
    page.click('#login-btn'),
  ]);
}

async function ensureOwnRoom(page, request, token, baseUrl) {
  const headers = { Authorization: 'Bearer ' + token };
  const stateRes = await request.get(baseUrl + '/api/duel-v2/state', { headers });
  let payload = {};
  try { payload = await stateRes.json(); } catch (_error) { payload = {}; }
  if (stateRes.ok() && payload.room && payload.room.room_code) {
    if (env('ALLOW_RESET_TEST_ROOM', '') !== '1') {
      fail('existing test room found; set ALLOW_RESET_TEST_ROOM=1 to reopen');
    }
    await request.post(baseUrl + '/api/duel-v2/leave', { headers, data: {} });
  }
  await page.goto(baseUrl + '/card-game', { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('#v2-start-btn', { timeout: 20000 });
  await page.click('#v2-mode-solo');
  await Promise.all([
    page.waitForURL(/\/table/, { timeout: 20000 }),
    page.click('#v2-start-btn'),
  ]);
}

async function measure(page, viewport) {
  return page.evaluate((view) => {
    function rect(id) {
      const node = document.getElementById(id);
      if (!node) return null;
      const box = node.getBoundingClientRect();
      return { id, x: box.x, y: box.y, width: box.width, height: box.height, top: box.top, right: box.right, bottom: box.bottom, left: box.left };
    }
    const arts = Array.from(document.querySelectorAll('.v2-character-art img, .v2-character-art .v2-image-fallback')).map((node) => {
      const box = node.getBoundingClientRect();
      return { width: box.width, height: box.height, top: box.top, bottom: box.bottom, left: box.left, right: box.right };
    });
    return {
      viewport: view,
      top: rect('v2-opponent-life'),
      bottom: rect('v2-hand-dock'),
      hp: rect('v2-opponent-hp'),
      playerHp: rect('v2-player-hp'),
      ap: rect('v2-player-ap'),
      endTurn: rect('v2-end-turn-btn'),
      playerFront: rect('v2-player-front'),
      opponentFront: rect('v2-opponent-front'),
      arts,
    };
  }, viewport);
}

function assertLayout(report, minAvatar) {
  const viewport = report.viewport;
  for (const key of ['top', 'bottom', 'hp', 'playerHp', 'ap', 'endTurn']) {
    const node = report[key];
    if (!node) fail('missing ' + key);
    if (node.width <= 0 || node.height <= 0) fail(key + ' has zero size');
    if (clipped(node, viewport)) fail(key + ' clipped: ' + JSON.stringify(node));
  }
  if (report.playerFront.width > viewport.width * 0.55) fail('player front too wide');
  if (report.opponentFront.width > viewport.width * 0.55) fail('opponent front too wide');
  if (!report.arts.length) fail('no character art nodes');
  const zero = report.arts.filter((item) => item.height <= 0 || item.width <= 0);
  if (zero.length) fail('character art has zero bbox: ' + JSON.stringify(zero));
  const tooSmall = report.arts.filter((item) => item.width < minAvatar - 0.5 || item.height < minAvatar - 0.5);
  if (tooSmall.length) fail('character art smaller than ' + minAvatar + ': ' + JSON.stringify(tooSmall));
}

async function verifyGestures(page) {
  let actionCalls = 0;
  await page.route('**/api/duel-v2/action', async (route) => {
    actionCalls += 1;
    await route.continue();
  });
  const before = actionCalls;
  const invalidTarget = await page.$('[data-entity-id="b:nanali"]') || await page.$('#v2-opponent-bench');
  const card = await page.$('#v2-player-hand .v2-hand-card');
  if (card && invalidTarget) {
    const a = await card.boundingBox();
    const b = await invalidTarget.boundingBox();
    await page.mouse.move(a.x + 8, a.y + 8);
    await page.mouse.down();
    await page.mouse.move(b.x + 8, b.y + 8, { steps: 6 });
    await page.mouse.up();
  }
  if (actionCalls !== before) fail('invalid drop sent an action request');
  const attackSource = await page.$('#v2-player-bench [data-drag-kind="character"]');
  const front = await page.$('#v2-player-front');
  if (attackSource && front) {
    const a = await attackSource.boundingBox();
    const b = await front.boundingBox();
    await page.mouse.move(a.x + a.width / 2, a.y + a.height / 2);
    await page.mouse.down();
    await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 8 });
    await page.mouse.up();
    await page.waitForTimeout(400);
  }
  if (actionCalls - before > 1) fail('valid drag sent more than one request');
  const rail = await page.$('#v2-player-hand');
  if (rail) {
    await rail.evaluate((node) => { node.scrollLeft = 40; });
  }
}

async function main() {
  const baseUrl = env('BASE_URL', 'http://127.0.0.1:5001').replace(/\/$/, '');
  assertLocalBase(baseUrl);
  const uid = env('PLAYER_UID');
  const code = env('LOGIN_CODE');
  if (!uid || !code) fail('PLAYER_UID and LOGIN_CODE are required');
  const outputDir = env('OUTPUT_DIR', '');
  if (outputDir) fs.mkdirSync(outputDir, { recursive: true });

  const { chromium } = await import('playwright');
  const browser = await launchBrowser({ chromium });
  const results = [];
  try {
    const viewports = [
      { name: 'landscape-944', width: 944, height: 427, isMobile: true, hasTouch: true, minAvatar: 28 },
      { name: 'portrait-390', width: 390, height: 844, isMobile: true, hasTouch: true, minAvatar: 28 },
    ];
    for (const viewport of viewports) {
      const context = await browser.newContext({
        viewport: { width: viewport.width, height: viewport.height },
        isMobile: viewport.isMobile,
        hasTouch: viewport.hasTouch,
        locale: 'zh-CN',
      });
      const page = await context.newPage();
      await login(page, baseUrl, uid, code);
      const token = await page.evaluate(() => window.localStorage.getItem('nte_token') || '');
      if (!token) fail('login did not store token');
      await ensureOwnRoom(page, context.request, token, baseUrl);
      await page.waitForSelector('#v2-table-page', { timeout: 20000 });
      await page.waitForSelector('[data-entity-id="a:nanali"]', { timeout: 20000 });
      const report = await measure(page, { width: viewport.width, height: viewport.height });
      assertLayout(report, viewport.minAvatar);
      if (viewport.name === 'landscape-944') {
        await verifyGestures(page);
      }
      if (outputDir) {
        const file = path.join(outputDir, viewport.name + '.json');
        fs.writeFileSync(file, JSON.stringify(report, null, 2));
        await page.screenshot({ path: path.join(outputDir, viewport.name + '.png') });
      }
      results.push({ viewport: viewport.name, ok: true, report });
      await context.close();
    }
  } finally {
    await browser.close();
  }
  process.stdout.write(JSON.stringify({ ok: true, results: results.map((item) => item.viewport) }) + '\n');
}

main().catch((error) => {
  process.stderr.write(String(error && error.stack || error) + '\n');
  process.exit(1);
});
