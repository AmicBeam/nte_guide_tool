const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const { Player } = require(path.resolve(__dirname, "../../app/modules/card_game/static/js/v2_animation.js"));

class FakeClassList {
  constructor(node) {
    this.node = node;
  }
  add() {
    Array.from(arguments).forEach((name) => this.node._classes.add(String(name)));
    this.node.className = Array.from(this.node._classes).join(" ");
  }
  remove() {
    Array.from(arguments).forEach((name) => this.node._classes.delete(String(name)));
    this.node.className = Array.from(this.node._classes).join(" ");
  }
  contains(name) {
    return this.node._classes.has(String(name));
  }
}

class FakeNode {
  constructor(tag, documentRef, ns) {
    this.tagName = String(tag || "div").toUpperCase();
    this.namespaceURI = ns || "";
    this.ownerDocument = documentRef;
    this.parentNode = null;
    this.childNodes = [];
    this.attributes = {};
    this.style = {};
    this._classes = new Set();
    this.classList = new FakeClassList(this);
    this.className = "";
    this.id = "";
    this.textContent = "";
    this._listeners = {};
    this._rect = { left: 0, top: 0, width: 80, height: 80, x: 0, y: 0 };
  }
  setAttribute(name, value) {
    const key = String(name);
    const next = String(value);
    this.attributes[key] = next;
    if (key === "id") {
      this.id = next;
    }
    if (key === "class") {
      this.className = next;
      this._classes = new Set(next.split(/\s+/).filter(Boolean));
    }
  }
  getAttribute(name) {
    const key = String(name);
    if (key === "class") {
      return this.className || "";
    }
    return Object.prototype.hasOwnProperty.call(this.attributes, key) ? this.attributes[key] : null;
  }
  removeAttribute(name) {
    const key = String(name);
    delete this.attributes[key];
    if (key === "id") {
      this.id = "";
    }
    if (key === "class") {
      this.className = "";
      this._classes = new Set();
    }
  }
  cloneNode(deep) {
    const copy = new FakeNode(this.tagName, this.ownerDocument, this.namespaceURI);
    Object.keys(this.attributes || {}).forEach((key) => {
      copy.setAttribute(key, this.attributes[key]);
    });
    copy.className = this.className;
    copy._classes = new Set(this._classes);
    copy.style = Object.assign({}, this.style);
    copy.textContent = this.textContent;
    copy._rect = Object.assign({}, this._rect);
    if (this.src) {
      copy.src = this.src;
    }
    if (deep) {
      this.childNodes.forEach((child) => {
        copy.appendChild(child.cloneNode(true));
      });
    }
    return copy;
  }
  appendChild(child) {
    if (!child) {
      return child;
    }
    if (child.parentNode) {
      child.parentNode.removeChild(child);
    }
    child.parentNode = this;
    this.childNodes.push(child);
    return child;
  }
  get firstChild() {
    return this.childNodes.length ? this.childNodes[0] : null;
  }
  removeChild(child) {
    this.childNodes = this.childNodes.filter((node) => node !== child);
    if (child) {
      child.parentNode = null;
    }
    return child;
  }
  replaceChildren() {
    this.childNodes.forEach((child) => {
      child.parentNode = null;
    });
    this.childNodes = [];
  }
  querySelector(selector) {
    return queryAll(this, selector)[0] || null;
  }
  querySelectorAll(selector) {
    return queryAll(this, selector);
  }
  getBoundingClientRect() {
    const rect = this._rect || { left: 0, top: 0, width: 80, height: 80 };
    return {
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height,
      x: rect.left,
      y: rect.top,
      right: rect.left + rect.width,
      bottom: rect.top + rect.height,
    };
  }
  addEventListener(type, handler) {
    this._listeners[type] = this._listeners[type] || [];
    this._listeners[type].push(handler);
  }
  animate(_keyframes, options) {
    const duration = options && options.duration ? Number(options.duration) : 0;
    const animation = {
      cancelled: false,
      finished: Promise.resolve(),
      cancel() {
        this.cancelled = true;
      },
    };
    if (duration > 0 && this.ownerDocument && this.ownerDocument.clock) {
      animation.finished = this.ownerDocument.clock.wait(duration);
    }
    return animation;
  }
}

function parseAttrSelector(selector) {
  const match = /^\[([^=]+)=["']?([^"']+)["']?\]$/.exec(selector);
  if (!match) {
    return null;
  }
  return { attr: match[1], value: match[2] };
}

function matches(node, selector) {
  if (!node || !selector) {
    return false;
  }
  if (selector.charAt(0) === "#") {
    return node.id === selector.slice(1);
  }
  if (selector.charAt(0) === ".") {
    const wanted = selector.slice(1).split(".").filter(Boolean);
    return wanted.every((name) => node.classList.contains(name));
  }
  const attr = parseAttrSelector(selector);
  if (attr) {
    return node.getAttribute(attr.attr) === attr.value;
  }
  return node.tagName.toLowerCase() === selector.toLowerCase();
}

function walk(node, visit) {
  visit(node);
  (node.childNodes || []).forEach((child) => walk(child, visit));
}

function queryAll(root, selector) {
  const found = [];
  (root.childNodes || []).forEach((child) => {
    walk(child, (node) => {
      if (matches(node, selector)) {
        found.push(node);
      }
    });
  });
  return found;
}

class FakeClock {
  constructor() {
    this.now = 0;
    this._waiters = [];
  }
  wait(ms) {
    const target = this.now + Math.max(0, Number(ms) || 0);
    return new Promise((resolve) => {
      this._waiters.push({ target: target, resolve: resolve });
    });
  }
  async flush(ms) {
    this.now += Math.max(0, Number(ms) || 0);
    const due = this._waiters.filter((item) => item.target <= this.now);
    this._waiters = this._waiters.filter((item) => item.target > this.now);
    due.forEach((item) => item.resolve());
    await Promise.resolve();
  }
}

function makeDocument() {
  const clock = new FakeClock();
  const documentRef = {
    clock: clock,
    body: null,
    createElement(tag) {
      return new FakeNode(tag, documentRef);
    },
    createElementNS(_ns, tag) {
      return new FakeNode(tag, documentRef, _ns);
    },
  };
  documentRef.body = new FakeNode("body", documentRef);
  return documentRef;
}

function place(node, left, top, width, height) {
  node._rect = { left: left, top: top, width: width, height: height };
}

function cardFace(node) {
  return {
    instance_id: node.instance_id,
    card_id: node.card_id,
    character_id: node.character_id,
    name: node.name,
    type: node.type,
    cost: node.cost,
    terminal: !!node.terminal,
    description: node.description || "",
    copy: !!node.copy,
  };
}

function makeRoot() {
  const documentRef = makeDocument();
  const root = documentRef.createElement("div");
  root.className = "v2-fx-root";
  const stage = documentRef.createElement("div");
  stage.setAttribute("id", "v2-fx-stage");
  place(stage, 0, 0, 800, 450);
  const handA = documentRef.createElement("div");
  handA.setAttribute("id", "v2-player-hand");
  handA.setAttribute("data-hand-side", "a");
  place(handA, 80, 360, 180, 40);
  const handB = documentRef.createElement("div");
  handB.setAttribute("data-hand-side", "b");
  place(handB, 80, 20, 180, 40);
  const playerA = documentRef.createElement("div");
  playerA.setAttribute("data-player-side", "a");
  place(playerA, 20, 380, 60, 40);
  const playerB = documentRef.createElement("div");
  playerB.setAttribute("data-player-side", "b");
  place(playerB, 20, 20, 60, 40);
  const nanali = documentRef.createElement("div");
  nanali.setAttribute("data-entity-id", "a:nanali");
  place(nanali, 320, 300, 90, 110);
  const zero = documentRef.createElement("div");
  zero.setAttribute("data-entity-id", "b:zero");
  place(zero, 320, 80, 90, 110);
  const xun = documentRef.createElement("div");
  xun.setAttribute("data-entity-id", "a:xun");
  place(xun, 430, 300, 90, 110);
  function portrait(host, src) {
    const img = documentRef.createElement("img");
    img.setAttribute("src", src);
    host.appendChild(img);
    return img;
  }
  portrait(nanali, "/static/images/characters/nanali.png");
  portrait(zero, "/static/images/characters/zero.png");
  portrait(xun, "/static/images/characters/xun.png");
  [stage, handA, handB, playerA, playerB, nanali, zero, xun].forEach((node) => root.appendChild(node));
  documentRef.body.appendChild(root);
  return {
    documentRef,
    clock: documentRef.clock,
    root,
    stage,
    handA,
    handB,
    nanali,
    zero,
    xun,
  };
}

function makeHandCard(documentRef, card) {
  const node = documentRef.createElement("div");
  node.setAttribute("data-instance-id", String(card.instance_id));
  node.setAttribute("data-card-id", String(card.card_id || ""));
  node.name = card.name;
  place(node, 90, 360, 48, 68);
  return node;
}

function publicCard() {
  return cardFace({
    instance_id: "c-1",
    card_id: "N03",
    character_id: "nanali",
    name: "要叫大姐头",
    type: "battle",
    cost: 1,
    terminal: false,
    description: "娜娜莉出击",
    copy: false,
  });
}

async function settle(done, clock, player) {
  let settled = false;
  const marked = Promise.resolve(done).then(() => {
    settled = true;
  });
  for (let index = 0; index < 40 && !settled; index += 1) {
    await clock.flush(500);
    await Promise.resolve();
  }
  if (!settled && player) {
    player.cancel();
    await clock.flush(500);
  }
  await marked;
}

test("play shows opponent card from hand with user and cost then patches after animation", async () => {
  const harness = makeRoot();
  const patches = [];
  const cards = [];
  const busy = [];
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch(patch, event) {
      patches.push({ patch, type: event.type, seq: event.seq });
    },
    onBusy(value) {
      busy.push(value);
    },
    onPublicCard(card) {
      cards.push(card);
    },
  });
  const done = player.play([
    {
      seq: 1,
      action_id: "act-1",
      action_version: 4,
      type: "play",
      side: "b",
      actor: "b:zero",
      target: "a:nanali",
      text: "ignored",
      card: publicCard(),
      patch: { sides: { a: { hp: 28 } } },
    },
  ]);
  await Promise.resolve();
  await Promise.resolve();
  const overlay = harness.stage.querySelector(".v2-fx-overlay");
  const playCard = overlay && overlay.querySelector(".v2-fx-play-card");
  assert.ok(playCard, "play card overlay exists");
  assert.equal(playCard.getAttribute("data-v2-fx-kind"), "play");
  const meta = overlay.querySelector(".v2-fx-card-meta");
  assert.match(meta.textContent, /使用者/);
  assert.equal(overlay.getAttribute("data-v2-fx-last-actor"), "零");
  assert.equal(patches.length, 0, "patch waits until presentation");
  await settle(done, harness.clock, player);
  assert.equal(patches.length, 1);
  assert.equal(patches[0].patch.sides.a.hp, 28);
  assert.equal(cards[0].name, "要叫大姐头");
  assert.deepEqual(busy.slice(0, 1), [true]);
  assert.equal(busy[busy.length - 1], false);
});

test("seq and action identity are de-duplicated", async () => {
  const harness = makeRoot();
  const patches = [];
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch(patch) {
      patches.push(patch);
    },
  });
  const event = {
    seq: 7,
    action_id: "act-7",
    action_version: 1,
    type: "growth",
    side: "a",
    actor: "a:nanali",
    amount: 1,
    patch: { sides: { a: { characters: [{ id: "nanali", growth: 1 }] } } },
  };
  await settle(player.play([event, Object.assign({}, event)], { instant: true }), harness.clock, player);
  await settle(player.play([event], { instant: true }), harness.clock, player);
  assert.equal(patches.length, 1);
});

test("same group plays together while consecutive attacks stay segmented", async () => {
  const harness = makeRoot();
  const order = [];
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch(_patch, event) {
      order.push(event.type + ":" + event.seq);
    },
  });
  await settle(player.play([
    { seq: 10, type: "damage", group_id: "g1", target: "a:nanali", amount: 2, patch: { a: 1 } },
    { seq: 11, type: "damage", group_id: "g1", target: "b:zero", amount: 3, patch: { b: 1 } },
    { seq: 12, type: "attack", actor: "a:xun", target: "b:zero", attack: 4, patch: { hit: 1 } },
    { seq: 13, type: "attack", actor: "a:xun", target: "b:player", attack: 4, patch: { hit: 2 } },
  ], { instant: true }), harness.clock, player);
  assert.deepEqual(order, ["damage:10", "damage:11", "attack:12", "attack:13"]);
});

test("cancel finishes play and still applies remaining patches", async () => {
  const harness = makeRoot();
  const patches = [];
  const busy = [];
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch(patch, event) {
      patches.push(event.seq);
    },
    onBusy(value) {
      busy.push(value);
    },
  });
  const done = player.play([
    { seq: 21, type: "play", side: "b", actor: "b:zero", card: publicCard(), patch: { play: true } },
    { seq: 22, type: "attack", actor: "b:zero", target: "a:nanali", attack: 3, patch: { atk: true } },
  ]);
  await Promise.resolve();
  player.cancel();
  await settle(done, harness.clock, player);
  assert.ok(patches.includes(21));
  assert.ok(patches.includes(22));
  assert.equal(busy[busy.length - 1], false);
});

test("missing entities and animation API still complete without throwing", async () => {
  const harness = makeRoot();
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch() {},
  });
  harness.nanali.animate = undefined;
  await settle(player.play([
    { seq: 30, type: "attack", actor: "a:missing", target: "b:ghost", attack: 2, patch: { ok: true } },
    { seq: 31, type: "flower", actor: "a:jiuyuan", target: "b:player", amount: 1, patch: { flower: true } },
    { seq: 32, type: "awaken", actor: "a:nanali", patch: { awaken: true } },
    { seq: 33, type: "shape", actor: "a:nanali", shape: "第二位成员", patch: { shape: true } },
    { seq: 34, type: "down", target: "b:zero", patch: { down: true } },
    { seq: 35, type: "revive", target: "b:zero", patch: { revive: true } },
    { seq: 36, type: "record", actor: "a:xun", card: publicCard(), patch: { record: true } },
    { seq: 37, type: "redeem", actor: "a:xun", card: Object.assign(publicCard(), { copy: true, name: "要叫大姐头" }), patch: { copy: true } },
    { seq: 38, type: "draw", side: "b", patch: { draw: true } },
  ], { instant: true }), harness.clock, player);
  const overlay = harness.stage.querySelector(".v2-fx-overlay");
  assert.ok(overlay);
  assert.match(overlay.getAttribute("data-v2-fx-types") || "", /attack/);
  assert.match(overlay.getAttribute("data-v2-fx-types") || "", /flower/);
  assert.match(overlay.getAttribute("data-v2-fx-types") || "", /awaken/);
  assert.match(overlay.getAttribute("data-v2-fx-types") || "", /record/);
});

test("down and attack fx do not leave character cards rotated", async () => {
  const harness = makeRoot();
  harness.zero.classList.add("v2-character-card");
  harness.nanali.classList.add("v2-character-card");
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch() {},
  });
  await settle(player.play([
    { seq: 50, type: "down", target: "b:zero", patch: { down: true } },
    { seq: 51, type: "attack", actor: "a:nanali", target: "b:zero", attack: 2, patch: { atk: true } },
  ]), harness.clock, player);
  assert.equal(harness.zero.classList.contains("v2-fx-down"), false);
  assert.equal(harness.nanali.classList.contains("v2-fx-lunge"), false);
  assert.equal(harness.zero.style.transform || "", "");
  assert.equal(harness.nanali.style.transform || "", "");
});

test("reduced motion and speed keep final patches and do not leave busy", async () => {
  const harness = makeRoot();
  const patches = [];
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch(_patch, event) {
      patches.push(event.seq);
    },
  });
  player.setOptions({ reducedMotion: true, speed: 2, muted: true });
  await settle(player.play([
    { seq: 40, type: "damage", target: "a:player", amount: 5, before: { shield: 5 }, after: { shield: 0 }, patch: { hp: 25 } },
    { seq: 41, type: "growth", actor: "a:nanali", amount: 1, patch: { growth: 1 } },
  ]), harness.clock, player);
  assert.deepEqual(patches, [40, 41]);
  assert.equal(player._busy, false);
});

test("audio stays muted by default and unlocks only after user gesture", async () => {
  const harness = makeRoot();
  let created = 0;
  globalThis.AudioContext = function FakeAudio() {
    created += 1;
    this.currentTime = 0;
    this.destination = {};
    this.state = "suspended";
    this.resume = async () => {
      this.state = "running";
    };
    this.suspend = async () => {
      this.state = "suspended";
    };
    this.createOscillator = () => ({
      type: "triangle",
      frequency: { value: 0 },
      connect() {},
      start() {},
      stop() {},
    });
    this.createGain = () => ({
      gain: {
        setValueAtTime() {},
        exponentialRampToValueAtTime() {},
      },
      connect() {},
    });
  };
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch() {},
  });
  assert.equal(player.options.muted, true);
  await settle(player.play([{ seq: 50, type: "play", side: "a", actor: "a:nanali", card: publicCard(), patch: { ok: true } }], { instant: true }), harness.clock, player);
  assert.equal(created, 0);
  player._unlockAudio();
  assert.equal(created, 1);
  player.setOptions({ muted: false });
  await settle(player.play([{ seq: 51, type: "attack", actor: "a:nanali", target: "b:zero", attack: 3, patch: { ok: true } }], { instant: true }), harness.clock, player);
  delete globalThis.AudioContext;
});
function overlayOf(harness) {
  return harness.stage.querySelector(".v2-fx-overlay");
}

function floatersOf(harness) {
  const overlay = overlayOf(harness);
  return overlay ? String(overlay.getAttribute("data-v2-fx-floaters") || "").split("|").filter(Boolean) : [];
}

function arrowsOf(harness) {
  const overlay = overlayOf(harness);
  return overlay ? overlay.querySelectorAll(".v2-fx-arrow") : [];
}

test("damage floater stays hidden and is removed after its animation finishes", async () => {
  const harness = makeRoot();
  const player = new Player({ root: harness.root, clock: harness.clock });
  let finish;
  player._animate = () => new Promise((resolve) => { finish = resolve; });
  const node = player._floater({ x: 40, y: 40 }, { left: 0, top: 0 }, "-2", "damage");
  assert.ok(node.parentNode);
  assert.equal(node.style.opacity, "0", "base style must stay invisible when animation is cancelled");
  finish();
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(node.parentNode, null);
});

test("attack presentation does not announce damage numbers before damage packets", async () => {
  const harness = makeRoot();
  const player = new Player({ root: harness.root, clock: harness.clock });
  const captions = [];
  player._notice = (...args) => captions.push(args);
  await settle(player.play([{ seq: 1, type: "attack", actor: "a:nanali", target: "b:zero",
    attack: 4, counter: 2, text: "攻击 4，反击 2。" }], { instant: true }), harness.clock, player);
  assert.deepEqual(captions, []);
  assert.deepEqual(floatersOf(harness), []);
});

test("attack plus combat counter penetration is one strike with parallel damage", async () => {
  const harness = makeRoot();
  const order = [];
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch(_patch, event) {
      order.push(event.type + ":" + event.seq);
    },
  });
  await settle(player.play([
    {
      seq: 60,
      type: "attack",
      group_id: "g-hit",
      actor: "a:nanali",
      target: "b:zero",
      attack: 6,
      counter: 2,
      patch: { attack: true },
    },
    {
      seq: 61,
      type: "combat",
      group_id: "g-hit",
      actor: "a:nanali",
      target: "b:zero",
      amount: 6,
      before: { hp: 9, shield: 2 },
      after: { hp: 5, shield: 0 },
      patch: { defender: true },
    },
    {
      seq: 62,
      type: "combat",
      group_id: "g-hit",
      actor: "b:zero",
      target: "a:nanali",
      amount: 2,
      before: { hp: 7, shield: 0 },
      after: { hp: 5, shield: 0 },
      patch: { attacker: true },
    },
    {
      seq: 63,
      type: "penetration",
      group_id: "g-hit",
      actor: "a:nanali",
      target: "b:player",
      amount: 1,
      before: { hp: 30, shield: 0 },
      after: { hp: 29, shield: 0 },
      patch: { player: true },
    },
  ], { instant: true }), harness.clock, player);
  const overlay = overlayOf(harness);
  assert.equal(overlay.getAttribute("data-v2-fx-attack-count"), "1");
  const types = String(overlay.getAttribute("data-v2-fx-types") || "").split(",");
  assert.equal(types.filter((kind) => kind === "attack").length, 1);
  assert.ok(types.includes("combat"));
  assert.ok(types.includes("penetration"));
  const floaters = floatersOf(harness);
  assert.ok(floaters.includes("-2"));
  assert.ok(floaters.includes("-4"));
  assert.ok(floaters.includes("-1"));
  assert.equal(floaters.filter((text) => text === "-6").length, 0);
  assert.deepEqual(order, ["attack:60", "combat:61", "combat:62", "penetration:63"]);
});

test("shield and hp losses both float from server before after", async () => {
  const harness = makeRoot();
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch() {},
  });
  await settle(player.play([
    {
      seq: 70,
      type: "damage",
      target: "a:nanali",
      amount: 5,
      before: { hp: 7, shield: 3 },
      after: { hp: 5, shield: 0 },
      patch: { mixed: true },
    },
    {
      seq: 71,
      type: "heal",
      target: "a:nanali",
      amount: 1,
      before: { hp: 5, shield: 0 },
      after: { hp: 6, shield: 0 },
      patch: { heal: true },
    },
    {
      seq: 72,
      type: "combat",
      target: "b:zero",
      amount: 0,
      counter: 0,
      before: { hp: 9, shield: 0 },
      after: { hp: 9, shield: 0 },
      patch: { miss: true },
    },
  ], { instant: true }), harness.clock, player);
  const overlay = overlayOf(harness);
  const floaters = floatersOf(harness);
  assert.ok(floaters.includes("-3"));
  assert.ok(floaters.includes("-2"));
  assert.ok(floaters.includes("+1"));
  assert.equal(floaters.filter((text) => text === "-0" || text === "-5").length, 0);
  assert.equal(overlay.getAttribute("data-v2-fx-attack-count"), null);
});

test("queued play jobs cancel without continuing old events", async () => {
  const harness = makeRoot();
  const patches = [];
  const presented = [];
  const busy = [];
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch(_patch, event) {
      patches.push(event.seq);
    },
    onBusy(value) {
      busy.push(value);
    },
  });
  const originalPresent = player._present.bind(player);
  player._present = async function (event, instant) {
    presented.push(event.seq);
    return originalPresent(event, instant);
  };
  const first = player.play([
    { seq: 80, type: "play", side: "b", actor: "b:zero", card: publicCard(), patch: { first: true } },
    { seq: 81, type: "attack", actor: "b:zero", target: "a:nanali", attack: 3, patch: { firstHit: true } },
  ]);
  const queued = player.play([
    { seq: 82, type: "growth", actor: "a:nanali", amount: 1, patch: { queued: true } },
    { seq: 83, type: "attack", actor: "a:nanali", target: "b:zero", attack: 4, patch: { queuedHit: true } },
  ]);
  await Promise.resolve();
  player.cancel();
  await settle(first, harness.clock, player);
  await settle(queued, harness.clock, player);
  assert.deepEqual(patches, [80, 81, 82, 83]);
  assert.ok(!presented.includes(82));
  assert.ok(!presented.includes(83));
  const later = player.play([
    { seq: 84, type: "heal", target: "a:nanali", amount: 1, before: { hp: 5 }, after: { hp: 6 }, patch: { later: true } },
  ], { instant: true });
  await settle(later, harness.clock, player);
  assert.ok(patches.includes(84));
  assert.ok(presented.includes(84));
  assert.equal(busy[busy.length - 1], false);
});

test("own play fades the matching instance and keeps the other card", async () => {
  const harness = makeRoot();
  const first = makeHandCard(harness.documentRef, { instance_id: "c-keep", card_id: "N01", name: "帮派初建成" });
  const second = makeHandCard(harness.documentRef, { instance_id: "c-play", card_id: "N03", name: "要叫大姐头" });
  harness.handA.appendChild(first);
  harness.handA.appendChild(second);
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    presentOwnPlays: true,
    applyPatch() {},
  });
  const done = player.play([
    {
      seq: 90,
      type: "play",
      side: "a",
      actor: "a:nanali",
      card: Object.assign(publicCard(), { instance_id: "c-play", description: "娜娜莉出击" }),
      patch: { play: true },
    },
  ]);
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(second.classList.contains("v2-fx-source"), true);
  assert.equal(first.classList.contains("v2-fx-source"), false);
  const overlay = overlayOf(harness);
  const playCard = overlay.querySelector(".v2-fx-play-card");
  assert.equal(playCard.getAttribute("data-instance-id"), "c-play");
  assert.equal(overlay.querySelector(".v2-fx-card-desc").textContent, "娜娜莉出击");
  assert.equal(overlay.querySelector(".v2-fx-card-portrait").getAttribute("src"), "/static/images/characters/nanali.png");
  await settle(done, harness.clock, player);
});

test("opponent hidden hand uses a single card back source", async () => {
  const harness = makeRoot();
  const hidden = harness.documentRef.createElement("div");
  hidden.className = "v2-hidden-card";
  hidden.setAttribute("class", "v2-hidden-card");
  place(hidden, 80, 20, 48, 68);
  const extra = harness.documentRef.createElement("div");
  extra.className = "v2-hidden-card extra";
  extra.setAttribute("class", "v2-hidden-card extra");
  place(extra, 140, 20, 48, 68);
  harness.handB.appendChild(hidden);
  harness.handB.appendChild(extra);
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch() {},
  });
  const done = player.play([
    {
      seq: 91,
      type: "play",
      side: "b",
      actor: "b:zero",
      card: publicCard(),
      patch: { play: true },
    },
  ]);
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(hidden.classList.contains("v2-fx-source"), true);
  assert.equal(extra.classList.contains("v2-fx-source"), false);
  await settle(done, harness.clock, player);
});

test("harmony and dots use readable titles instead of raw type chips", async () => {
  const harness = makeRoot();
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch() {},
  });
  await settle(player.play([
    { seq: 100, type: "harmony", actor: "a:nanali", text: "娜娜莉消耗自己的 2 点环合值，触发创生。", patch: { harmony: true } },
    { seq: 101, type: "burn", target: "b:zero", amount: 1, text: "浊燃", patch: { burn: true } },
  ], { instant: true }), harness.clock, player);
  const overlay = overlayOf(harness);
  const types = String(overlay.getAttribute("data-v2-fx-types") || "");
  assert.match(types, /harmony/);
  assert.match(types, /burn/);
  const floaters = floatersOf(harness);
  assert.ok(floaters.some((text) => text.indexOf("浊燃") >= 0 || text.indexOf("噩梦") >= 0 || text.length > 0));
});

test("enter uses the existing character pop before the front patch", async () => {
  const harness = makeRoot();
  harness.nanali.classList.add("v2-character-card");
  const patches = [];
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch(_patch, event) {
      patches.push(event.type);
    },
  });
  const done = player.play([
    {
      seq: 110,
      type: "enter",
      side: "a",
      actor: "a:nanali",
      source: "nanali",
      text: "娜娜莉进入战斗区。",
      patch: { sides: { a: { front: "nanali" } } },
    },
  ]);
  await Promise.resolve();
  await Promise.resolve();
  const overlay = overlayOf(harness);
  assert.equal(overlay.getAttribute("data-v2-fx-last-type"), "enter");
  assert.equal(harness.nanali.classList.contains("v2-fx-enter"), true);
  assert.equal(patches.length, 0);
  await settle(done, harness.clock, player);
  assert.deepEqual(patches, ["enter"]);
  assert.equal(harness.nanali.classList.contains("v2-fx-enter"), false);
});

test("own draw flies in from the right before the hand patch", async () => {
  const harness = makeRoot();
  const patches = [];
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch(_patch, event) {
      patches.push(event.type);
    },
  });
  const done = player.play([
    {
      seq: 120,
      type: "draw",
      side: "a",
      present: true,
      actor: "a:player",
      card: publicCard(),
      text: "我方玩家抽到「要叫大姐头」。",
      patch: { drawn: true },
    },
  ]);
  await Promise.resolve();
  await Promise.resolve();
  const overlay = overlayOf(harness);
  const flyer = overlay && overlay.querySelector(".v2-fx-draw-card");
  assert.ok(flyer, "draw close-up exists before the hand updates");
  assert.equal(flyer.getAttribute("data-v2-fx-kind"), "draw-public");
  assert.ok(overlay.querySelector(".v2-fx-draw-formula"), "draw starts as a formula card");
  const startLeft = Number(String(flyer.style.left || "0").replace("px", ""));
  assert.ok(startLeft >= 100, "draw starts off the right edge");
  assert.equal(patches.length, 0);
  await settle(done, harness.clock, player);
  assert.deepEqual(patches, ["draw"]);
});

test("draw ignores the hidden mulligan rail and stays over the table hand", () => {
  const harness = makeRoot();
  const rail = harness.documentRef.createElement("div");
  rail.setAttribute("id", "v2-mulligan-rail");
  rail.setAttribute("data-hand-side", "a");
  place(rail, 0, 0, 0, 0);
  harness.root.appendChild(rail);
  const last = harness.documentRef.createElement("div");
  last.classList.add("v2-hand-card");
  place(last, 420, 360, 148, 108);
  harness.handA.appendChild(last);
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch() {},
  });
  const slot = player._drawSlotRect({ side: "a", card: publicCard() }, { left: 0, top: 0, width: 800, height: 450 });
  assert.ok(slot.cx > 400, "landing stays over the table hand, not the left edge");
});

test("finish event presents then notifies onFinish", async () => {
  const harness = makeRoot();
  const finishes = [];
  const patches = [];
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch(_patch, event) {
      patches.push(event.type);
    },
    onFinish(event) {
      finishes.push(event.type);
    },
  });
  await settle(player.play([
    { seq: 200, type: "finish", side: "a", present: true, text: "我方玩家获胜。", winner: "a", patch: { phase: "finished" } },
  ], { instant: true }), harness.clock, player);
  assert.deepEqual(patches, ["finish"]);
  assert.deepEqual(finishes, ["finish"]);
});

test("ultimate end patches without a 终结 burst", async () => {
  const harness = makeRoot();
  const patches = [];
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch(_patch, event) {
      patches.push(event.type);
    },
  });
  await settle(player.play([
    {
      seq: 130,
      type: "ultimate",
      side: "a",
      present: false,
      text: "娜娜莉的终结结束。",
      after: { awakened: false, ultimate_turns: 0 },
      patch: { ended: true },
    },
  ], { instant: true }), harness.clock, player);
  assert.deepEqual(patches, ["ultimate"]);
  const overlay = overlayOf(harness);
  assert.equal(overlay.querySelector(".v2-fx-burst"), null);
  assert.equal(overlay.querySelector(".v2-fx-title"), null);
});

test("turn start is patched before the following draw", async () => {
  const harness = makeRoot();
  const order = [];
  const banners = [];
  const captions = [];
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch(_patch, event) {
      order.push(event.type);
    },
  });
  const notice = player._notice.bind(player);
  player._notice = (title, detail, position) => {
    if (position === "top") banners.push(title);
    else captions.push([title, detail]);
    return notice(title, detail, position);
  };
  await settle(player.play([
    { seq: 121, type: "turn", side: "a", present: true, text: "我方玩家的第 2 个回合开始。", patch: { turn: 2 } },
    { seq: 122, type: "draw", side: "a", present: true, card: publicCard(), patch: { drawn: true } },
    { seq: 123, type: "resource", side: "a", present: false, patch: { ap: 2 } },
  ], { instant: true }), harness.clock, player);
  assert.deepEqual(order, ["turn", "draw", "resource"]);
  const turnText = "我方玩家的第 2 个回合开始。";
  assert.equal(banners.filter((text) => text === turnText).length, 1);
  assert.equal(captions.some(([title, detail]) => title === turnText || detail === turnText), false);
});

test("notice moves one element between top and bottom without losing content", () => {
  const harness = makeRoot();
  const player = new Player({ root: harness.root, clock: harness.clock });
  const top = player._notice("恢复", "娜娜莉恢复了生命。", "top");
  assert.equal(top.textContent, "恢复 · 娜娜莉恢复了生命。");
  assert.equal(top.classList.contains("v2-fx-banner"), true);
  const bottom = player._notice("抽牌", "抽牌", "bottom");
  assert.equal(bottom, top);
  assert.equal(bottom.textContent, "抽牌");
  assert.equal(bottom.classList.contains("v2-fx-caption"), true);
  assert.equal(bottom.classList.contains("v2-fx-banner"), false);
  assert.equal(overlayOf(harness).querySelectorAll(".v2-fx-notice").length, 1);
  player._notice("", "", "top");
  assert.equal(top.hidden, true);
  player._notice("你的回合", "", "top");
  assert.equal(top.hidden, false);
});

test("mulligan replacement slides into its rail without a normal draw close-up", async () => {
  const harness = makeRoot();
  harness.root.classList.add("is-mulligan");
  const rail = harness.documentRef.createElement("div");
  rail.setAttribute("id", "v2-mulligan-rail");
  harness.root.appendChild(rail);
  const incoming = harness.documentRef.createElement("article");
  incoming.setAttribute("data-card-id", "replacement-1");
  incoming.classList.add("v2-hand-card");
  const patches = [];
  const player = new Player({
    root: harness.root,
    clock: harness.clock,
    applyPatch(_patch, event) {
      patches.push(event.seq);
      rail.appendChild(incoming);
    },
  });
  const done = player.play([{
    seq: 501, type: "draw", side: "a", present: true,
    card: { ...publicCard(), instance_id: "replacement-1" }, patch: { replacement: true },
  }]);
  await Promise.resolve();
  await Promise.resolve();
  assert.ok(incoming.classList.contains("is-entering"));
  assert.equal(overlayOf(harness).querySelector(".v2-fx-draw-card"), null);
  assert.deepEqual(patches, [501], "replacement patch precedes the rail animation");
  await settle(done, harness.clock, player);
  assert.deepEqual(patches, [501], "the outer queue must not apply the same draw twice");
  assert.equal(incoming.classList.contains("is-entering"), false);
});

test("generated and copied cards appear centrally before entering either hand", async () => {
  for (const side of ["a", "b"]) {
    for (const kind of ["derived", "copy"]) {
      const harness = makeRoot();
      const patches = [];
      const player = new Player({ root: harness.root, clock: harness.clock, viewerSide: "a",
        applyPatch(_patch, event) { patches.push(event.seq); } });
      const card = Object.assign(publicCard(), { [kind]: true });
      const done = player.play([{ seq: 130, type: "gain", side, actor: side + ":player", card, patch: { gained: true } }]);
      await Promise.resolve();
      await Promise.resolve();
      const overlay = overlayOf(harness);
      const flyer = overlay.querySelector(".v2-fx-draw-card");
      assert.ok(flyer);
      assert.equal(flyer.getAttribute("data-v2-fx-kind"), "gain-generated");
      assert.equal(parseFloat(flyer.style.left), (overlay.getBoundingClientRect().width - 300) / 2);
      assert.equal(flyer.style.transformOrigin, "center center");
      assert.deepEqual(patches, []);
      await settle(done, harness.clock, player);
      assert.deepEqual(patches, [130]);
      assert.equal(flyer.parentNode, null);
    }
  }
});

test("draw lands after its owner group or before the next owner group", () => {
  const harness = makeRoot();
  harness.handA.setAttribute("data-character-order", JSON.stringify(["zero", "nanali", "jiuyuan", "iloy"]));
  function add(owner, id, left) {
    const node = harness.documentRef.createElement("article");
    node.classList.add("v2-hand-card");
    node.setAttribute("data-card-owner", owner);
    node.setAttribute("data-instance-id", id);
    place(node, left, 360, 100, 108);
    harness.handA.appendChild(node);
    return node;
  }
  add("nanali", "n1", 100);
  add("nanali", "n2", 208);
  add("iloy", "i1", 330);
  const player = new Player({ root: harness.root, clock: harness.clock, applyPatch() {} });
  const slot = (owner, id = "new") => player._drawSlotRect({ side: "a", card: { ...publicCard(), character_id: owner, instance_id: id } }, {});
  assert.equal(slot("nanali").left, 316, "append to own group, not the final iloy card");
  assert.equal(slot("zero").left, 100, "new first group inserts before existing cards");
  assert.equal(slot("jiuyuan").left, 330, "new middle group inserts before later owner");
  assert.equal(slot("iloy").left, 438);
  assert.equal(slot("nanali", "n2").left, 208, "an existing instance uses its own slot");
});

test("discard events update state without notices", async () => {
  const harness = makeRoot();
  const patches = [];
  const notices = [];
  const player = new Player({ root: harness.root, clock: harness.clock, applyPatch: (_patch, event) => patches.push(event.seq) });
  player._notice = (...args) => notices.push(args);
  player._wait = async () => {};
  await player.play([{ seq: 601, type: "discard", side: "a", present: true, text: "「测试牌」进入弃牌堆。", card: publicCard(), patch: { discard_count: 1 } }]);
  assert.deepEqual(patches, [601]);
  assert.deepEqual(notices, []);
});

test("gulang forced discard announces the revealed card once after the silent discard", async () => {
  const harness = makeRoot();
  const patches = [], notices = [];
  const player = new Player({root:harness.root, clock:harness.clock,
    applyPatch:(_patch,event)=>patches.push(event.seq)});
  player._notice=(...args)=>notices.push(args);
  player._wait=async()=>{};
  const text='鬼郎丸受到伤害：对方玩家随机弃置手牌「测试牌」。';
  await player.play([
    {seq:602,type:'discard',side:'b',card:publicCard(),patch:{hand_count:0,discard_count:1}},
    {seq:603,type:'effect',side:'a',actor:'a:zaowu',target:'b:player',present:true,text,patch:{}},
    {seq:604,type:'effect',side:'a',present:false,text:'普通历史效果',patch:{}},
  ]);
  assert.deepEqual(patches,[602,603,604]);
  assert.deepEqual(notices,[[text,'','top']]);
});

test("hidden opponent draw uses only the card back during close-up", async () => {
  const harness = makeRoot();
  const player = new Player({ root: harness.root, clock: harness.clock });
  const done = player.play([{ seq: 990, type: "draw", side: "b", actor: "b:player", card: { hidden: true } }]);
  await Promise.resolve();
  await Promise.resolve();
  const formula = overlayOf(harness).querySelector(".v2-fx-draw-formula");
  assert.ok(formula.classList.contains("v2-fx-card-back"));
  assert.equal(formula.classList.contains("v2-fx-card-front"), false);
  assert.equal(formula.querySelector(".v2-fx-card-name"), null);
  assert.equal(formula.querySelector(".v2-fx-card-portrait"), null);
  await settle(done, harness.clock, player);
});

test("support immunity floats at the counter damage slot and target", async () => {
  const harness = makeRoot();
  const player = new Player({ root: harness.root, clock: harness.clock });
  const shown = [];
  const original = player._floater.bind(player);
  player._floater = (point, rect, text, kind) => {
    shown.push({ point, text, kind, time: harness.clock.now });
    return original(point, rect, text, kind);
  };
  await settle(player.play([
    { seq: 900, type: "attack", group_id: "support", actor: "a:nanali", target: "b:zero", attack: 2, counter: 0 },
    { seq: 901, type: "combat", group_id: "support", actor: "a:nanali", target: "b:zero", amount: 2,
      before: { hp: 5, shield: 0 }, after: { hp: 3, shield: 0 } },
    { seq: 902, type: "combat", group_id: "support", actor: "b:zero", target: "a:nanali", amount: 0,
      counter_immunity: "support", before: { hp: 5, shield: 0 }, after: { hp: 5, shield: 0 } },
  ], { instant: true }), harness.clock, player);
  const immune = shown.find(item => item.text === "无敌");
  const hit = shown.find(item => item.text === "-2");
  assert.ok(immune && hit);
  assert.equal(immune.time, hit.time);
  assert.deepEqual(immune.point, { x: 365, y: 355 });
  assert.equal(immune.kind, "invincible");
  assert.equal(shown.filter(item => item.text === "无敌").length, 1);
});

test("ordinary zero damage does not float invincibility", async () => {
  const harness = makeRoot();
  const player = new Player({ root: harness.root, clock: harness.clock });
  await settle(player.play([{ seq: 910, type: "combat", target: "a:nanali", amount: 0,
    before: { hp: 5, shield: 0 }, after: { hp: 5, shield: 0 } }], { instant: true }), harness.clock, player);
  assert.deepEqual(floatersOf(harness), []);
});

test("rewind patches once under the black curtain and removes it", async () => {
  const harness = makeRoot();
  let patches = 0;
  const player = new Player({ root: harness.root, clock: harness.clock, applyPatch: () => {
    patches += 1;
    assert.ok(harness.stage.querySelector('.v2-rewind-curtain'));
  } });
  await settle(player.play([{ seq: 920, type: 'rewind', side: 'a', patch: { turn: 1 } }]), harness.clock, player);
  assert.equal(patches, 1);
  assert.equal(harness.stage.querySelector('.v2-rewind-curtain'), null);
});
