(() => {
  const CHARACTER_NAMES = {
    nanali: "娜娜莉",
    zero: "零",
    jiuyuan: "九原",
    xun: "浔",
    anhunqu: "安魂曲",
    canhong: "残虹",
    zaowu: "早雾",
    lingke: "灵可",
    zhenhong: "真红",
    hathor: "哈索尔",
    iloy: "伊洛伊",
  };
  const TYPE_LABELS = {
    battle: "战斗",
    tactic: "战术",
    form: "武备",
    shape: "武备",
  };
  const NS = "http://www.w3.org/2000/svg";
  const MAX_EVENT_MS = 3600;
  const ACTOR_FX_CLASSES = [
    "v2-fx-lunge", "v2-fx-enter", "v2-fx-hurt", "v2-fx-shield", "v2-fx-heal",
    "v2-fx-down", "v2-fx-revive", "v2-fx-awaken", "v2-fx-shape", "v2-fx-growth",
  ];
  const BEATS = {
    playFly: 640,
    playFlip: 240,
    playHold: 1200,
    playAim: 320,
    attackLunge: 460,
    attackHold: 300,
    damageHold: 640,
    flowerHold: 560,
    statusHold: 720,
    titleHold: 820,
    bannerHold: 520,
    moveHold: 420,
    drawFly: 420,
    drawHold: 640,
    drawInsert: 380,
    markHold: 640,
    floaterHold: 520,
    gap: 280,
    linger: 240,
  };
  const TYPE_TITLES = {
    play: "出牌",
    attack: "出击",
    enter: "入场",
    move: "返回备战区",
    harmony: "环合",
    flower: "创生花",
    delay: "延滞",
    burn: "浊燃",
    nightmare: "噩梦",
    star: "黯星",
    awaken: "觉醒",
    ultimate: "终结",
    shape: "武备",
    growth: "成长",
    family: "家族壮大",
    resource: "资源",
    record: "记录",
    redeem: "兑现",
    draw: "抽牌",
    heal: "回复",
    shield: "护盾",
    down: "倒地",
    revive: "恢复",
    turn: "回合",
    finish: "胜负",
    gain: "获得",
    discard: "弃牌",
    remove: "移出",
    fatigue: "疲劳",
  };

  function noop() {}

  function asArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function sideOf(event) {
    if (event && event.side) {
      return String(event.side);
    }
    const source = String((event && (event.actor || event.source)) || "");
    const prefix = source.split(":")[0];
    return prefix === "a" || prefix === "b" ? prefix : "";
  }

  function characterIdFrom(ref) {
    const raw = String(ref || "");
    if (!raw) {
      return "";
    }
    if (raw.indexOf(":") >= 0) {
      return raw.split(":")[1] || "";
    }
    return raw;
  }

  function displayName(ref, card) {
    const characterId = characterIdFrom(ref) || (card && card.character_id) || "";
    if (characterId === "player") {
      return "玩家";
    }
    return CHARACTER_NAMES[characterId] || (card && card.name) || characterId || "";
  }

  function typeLabel(card) {
    if (!card) {
      return "";
    }
    return TYPE_LABELS[card.type] || String(card.type || "");
  }

  function eventKey(event) {
    if (!event || typeof event !== "object") {
      return "";
    }
    if (event.seq != null && event.seq !== "") {
      return "seq:" + String(event.seq);
    }
    return [
      "action",
      event.action_id || "",
      event.action_version || "",
      event.type || "",
      event.actor || event.source || "",
      event.target || "",
    ].join(":");
  }

  function hasPublicCard(card) {
    if (!card || typeof card !== "object" || card.hidden) {
      return false;
    }
    return Boolean(card.name || card.card_id || card.instance_id);
  }

  function cssEscape(value) {
    return String(value || "").replace(/\\/g, "\\\\").replace(/"/g, '\\"');
  }

  function readRect(node) {
    if (!node || typeof node.getBoundingClientRect !== "function") {
      return null;
    }
    try {
      const rect = node.getBoundingClientRect();
      if (!rect) {
        return null;
      }
      const width = Number(rect.width || 0);
      const height = Number(rect.height || 0);
      const left = Number(rect.left != null ? rect.left : rect.x || 0);
      const top = Number(rect.top != null ? rect.top : rect.y || 0);
      return {
        left: left,
        top: top,
        width: width,
        height: height,
        right: left + width,
        bottom: top + height,
        cx: left + width / 2,
        cy: top + height / 2,
      };
    } catch (_error) {
      return null;
    }
  }

  function centerOf(rect, fallback) {
    if (rect) {
      return { x: rect.cx, y: rect.cy };
    }
    return fallback;
  }

  function amountOf(event) {
    if (!event) {
      return null;
    }
    if (event.amount != null && event.amount !== "") {
      return event.amount;
    }
    if (event.damage != null && event.damage !== "") {
      return event.damage;
    }
    if (event.attack != null && event.attack !== "") {
      return event.attack;
    }
    return null;
  }

  function beforeAfter(event, field) {
    const before = event && event.before ? event.before[field] : undefined;
    const after = event && event.after ? event.after[field] : undefined;
    return { before: before, after: after };
  }

  function isDamagePacket(event) {
    const kind = event && event.type;
    return kind === "damage" || kind === "combat" || kind === "followup" || kind === "penetration" || kind === "heal";
  }

  function isAttackEvent(event) {
    return !!(event && event.type === "attack");
  }

  function numericValue(value) {
    if (value == null || value === "") {
      return null;
    }
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function fieldDelta(event, field) {
    const pair = beforeAfter(event, field);
    const before = numericValue(pair.before);
    const after = numericValue(pair.after);
    if (before == null || after == null) {
      return null;
    }
    return before - after;
  }

  function safeLocalSrc(value) {
    const src = String(value || "").trim();
    if (!src) {
      return "";
    }
    const lower = src.toLowerCase();
    if (lower.indexOf("http://") === 0 || lower.indexOf("https://") === 0 || lower.indexOf("//") === 0) {
      return "";
    }
    if (lower.indexOf("javascript:") === 0 || lower.indexOf("data:") === 0 || lower.indexOf("blob:") === 0) {
      return "";
    }
    return src;
  }

  class Player {
    constructor(options) {
      options = options || {};
      this.root = options.root || null;
      this.applyPatch = typeof options.applyPatch === "function" ? options.applyPatch : noop;
      this.onBusy = typeof options.onBusy === "function" ? options.onBusy : noop;
      this.onPublicCard = typeof options.onPublicCard === "function" ? options.onPublicCard : noop;
      this.onFinish = typeof options.onFinish === "function" ? options.onFinish : noop;
      this._clock = options.clock || null;
      this.options = {
        muted: options.muted == null ? true : !!options.muted,
        reducedMotion: !!options.reducedMotion,
        speed: options.speed === 2 ? 2 : 1,
        presentOwnPlays: !!options.presentOwnPlays,
      };
      this._generation = 0;
      this._cancelled = false;
      this._busy = false;
      this._seen = new Set();
      this._patched = new Set();
      this._waiters = new Set();
      this._anims = new Set();
      this._overlay = null;
      this._layer = null;
      this._svg = null;
      this._audio = null;
      this._audioReady = false;
      this._unlockBound = this._unlockAudio.bind(this);
      this._queue = Promise.resolve();
      this._cancelEpoch = 0;
      this._ensureOverlay();
      this._listenUnlock();
    }

    setOptions(next) {
      next = next || {};
      if (next.muted != null) {
        this.options.muted = !!next.muted;
      }
      if (next.reducedMotion != null) {
        this.options.reducedMotion = !!next.reducedMotion;
      }
      if (next.speed === 1 || next.speed === 2) {
        this.options.speed = next.speed;
      }
      if (next.presentOwnPlays != null) {
        this.options.presentOwnPlays = !!next.presentOwnPlays;
      }
      this._syncOverlayFlags();
      if (this.options.muted) {
        this._suspendAudio();
      }
    }

    cancel() {
      this._cancelled = true;
      this._cancelEpoch += 1;
      this._generation += 1;
      this._resumeWaiters();
      this._stopAnims();
    }

    play(events, playOptions) {
      const enqueueEpoch = this._cancelEpoch;
      const job = this._queue.then(() => this._runPlay(asArray(events), playOptions || {}, enqueueEpoch));
      this._queue = job.then(noop, noop);
      return job;
    }

    async _runPlay(events, playOptions, enqueueEpoch) {
      if (enqueueEpoch !== this._cancelEpoch) {
        this._flushPatches(events);
        return;
      }
      const token = ++this._generation;
      this._cancelled = false;
      const instant = !!(playOptions && playOptions.instant) || this.options.reducedMotion;
      this._instant = instant;
      this._setBusy(true);
      try {
        const batches = this._batch(events);
        let played = 0;
        for (let index = 0; index < batches.length; index += 1) {
          if (this._cancelled || token !== this._generation) {
            break;
          }
          const batch = batches[index].filter((event) => this._take(event));
          if (!batch.length) {
            continue;
          }
          if (played > 0) {
            await this._wait(this._duration(instant, BEATS.gap));
            this._clearFx(false);
          }
          await this._playBatch(batch, instant);
          played += 1;
        }
        if (played > 0 && !this._cancelled && token === this._generation) {
          await this._wait(this._duration(instant, BEATS.linger));
        }
      } catch (_error) {
        this._flushPatches(events);
      } finally {
        if (this._cancelled || token !== this._generation) {
          this._flushPatches(events);
        }
        this._resumeWaiters();
        this._clearFx(false);
        this._instant = false;
        this._setBusy(false);
      }
    }

    _take(event) {
      const key = eventKey(event);
      if (!key || this._seen.has(key)) {
        return false;
      }
      this._seen.add(key);
      return true;
    }

    _batch(events) {
      const batches = [];
      let current = [];
      let currentGroup = null;
      events.forEach((event) => {
        const groupId = event && event.group_id != null && event.group_id !== "" ? String(event.group_id) : null;
        if (!current.length) {
          current = [event];
          currentGroup = groupId;
          return;
        }
        if (groupId && groupId === currentGroup) {
          current.push(event);
          return;
        }
        if (isAttackEvent(event) && current.length) {
          const previous = current[current.length - 1];
          if (isAttackEvent(previous) && !groupId && !previous.group_id) {
            batches.push(current);
            current = [event];
            currentGroup = groupId;
            return;
          }
        }
        batches.push(current);
        current = [event];
        currentGroup = groupId;
      });
      if (current.length) {
        batches.push(current);
      }
      return batches;
    }

    async _playBatch(batch, instant) {
      const packets = batch.filter((event) => isDamagePacket(event));
      const others = batch.filter((event) => !isDamagePacket(event));
      for (let index = 0; index < others.length; index += 1) {
        if (this._cancelled) {
          return;
        }
        await this._present(others[index], instant);
        if (others[index].type !== 'rewind') this._patch(others[index]);
        const silent = this._isSilent(others[index]);
        if (!silent && (index < others.length - 1 || packets.length)) {
          await this._wait(this._duration(instant, BEATS.gap));
        }
      }
      if (!packets.length) {
        return;
      }
      await Promise.all(packets.map((event) => this._present(event, instant)));
      packets.forEach((event) => this._patch(event));
      await this._wait(this._duration(instant, BEATS.linger));
    }

    async _present(event, instant) {
      const kind = event && event.type ? String(event.type) : "instant";
      try {
        this._note(event, kind);
        if (this._isSilent(event)) {
          return;
        }
        if (kind === "rewind") {
          const curtain = this._el("div", "v2-rewind-curtain");
          this._ensureOverlay().appendChild(curtain);
          try {
            await this._animate(curtain, [{ opacity: 0 }, { opacity: 1 }], this._duration(instant, 220));
            this._patch(event);
            await this._animate(curtain, [{ opacity: 1 }, { opacity: 0 }], this._duration(instant, 280));
          } finally { if (curtain.parentNode) curtain.parentNode.removeChild(curtain); }
          return;
        }
        if (kind === "draw") {
          await this._presentDraw(event, instant);
          return;
        }
        if (kind === "gain" && hasPublicCard(event.card) && (event.card.derived || event.card.copy)) {
          await this._presentDraw(event, instant);
          return;
        }
        this._tone(kind);
        if (kind === "initiative") {
          await this._presentInitiative(event, instant);
          return;
        }
        if (kind === "mulligan") {
          await this._presentMulligan(event, instant);
          return;
        }
        if (kind === "finish") {
          await this._presentBanner(event, instant);
          try {
            this.onFinish(event);
          } catch (_error) {}
          return;
        }
        if (kind === "play") {
          if (this._isOpponentPlay(event) || this.options.presentOwnPlays) {
            await this._presentPlay(event, instant);
          }
          return;
        }
        if (kind === "attack") {
          await this._presentAttack(event, instant);
          return;
        }
        if (kind === "damage" || kind === "combat" || kind === "followup" || kind === "penetration" || kind === "heal") {
          await this._presentDamage(event, instant);
          return;
        }
        if (kind === "flower") {
          await this._presentFlower(event, instant);
          return;
        }
        if (kind === "harmony") {
          await this._presentHarmony(event, instant);
          return;
        }
        if (kind === "delay" || kind === "burn" || kind === "nightmare" || kind === "star" || kind === "fatigue") {
          await this._presentDot(event, instant, kind);
          return;
        }
        if (kind === "enter") {
          await this._presentEnter(event, instant);
          return;
        }
        if (kind === "move") {
          await this._presentMove(event, instant);
          return;
        }
        if (kind === "down") {
          await this._presentStatus(event, instant, "down", "倒地");
          return;
        }
        if (kind === "revive") {
          await this._presentStatus(event, instant, "revive", "恢复");
          return;
        }
        if (kind === "awaken" || kind === "ultimate") {
          await this._presentStatus(event, instant, "awaken", kind === "ultimate" ? "终结" : "觉醒");
          return;
        }
        if (kind === "shape") {
          await this._presentStatus(event, instant, "shape", event.shape || (event.card && event.card.name) || "武备");
          return;
        }
        if (kind === "family") {
          await this._presentFloater(event, instant, "growth", "家族壮大");
          return;
        }
        if (kind === "growth") {
          await this._presentFloater(event, instant, "growth", "成长");
          return;
        }
        if (kind === "record") {
          await this._presentMark(event, instant, "record", "记录");
          return;
        }
        if (kind === "redeem") {
          await this._presentMark(event, instant, "copy", "复制");
          return;
        }
        if (kind === "gain" || kind === "remove" || kind === "discard" || kind === "turn") {
          await this._presentBanner(event, instant);
          return;
        }
        await this._presentBanner(event, instant);
      } catch (_error) {
        await this._wait(instant ? 0 : 16);
      }
    }

    async _presentPlay(event, instant) {
      const overlay = this._ensureOverlay();
      const card = event && event.card;
      const side = sideOf(event);
      const hand = this._hand(side);
      const stageRect = readRect(overlay) || { left: 0, top: 0, width: 800, height: 450, cx: 400, cy: 225, right: 800, bottom: 450 };
      const sourceCard = this._sourceCard(hand, card, side);
      const fromRect = readRect(sourceCard) || readRect(hand) || readRect(this._entity(event && (event.actor || event.source))) || stageRect;
      const node = this._el("div", "v2-fx-play-card");
      node.setAttribute("data-v2-fx-kind", "play");
      if (card && card.card_id) {
        node.setAttribute("data-card-id", String(card.card_id));
      }
      if (card && card.instance_id) {
        node.setAttribute("data-instance-id", String(card.instance_id));
      }
      if (card && (card.type === "battle" || card.type === "tactic" || card.type === "form")) {
        node.setAttribute("data-card-type", String(card.type));
      }
      const portraitSrc = this._safePortrait(event && (event.actor || event.source));
      if (portraitSrc) {
        overlay.setAttribute("data-v2-fx-last-portrait", portraitSrc);
      }
      const shell = this._el("div", "v2-fx-card-3d");
      const back = this._el("div", "v2-fx-card-face v2-fx-card-back");
      if (portraitSrc) {
        back.style.backgroundImage = "url(" + JSON.stringify(portraitSrc) + ")";
      }
      shell.appendChild(back);
      const front = this._el("div", "v2-fx-card-face v2-fx-card-front");
      if (portraitSrc) {
        const art = this._el("div", "v2-fx-card-art");
        const img = this._el("img", "v2-fx-card-portrait");
        img.setAttribute("src", portraitSrc);
        img.setAttribute("alt", "");
        art.appendChild(img);
        front.appendChild(art);
      } else {
        front.appendChild(this._el("div", "v2-fx-card-art is-fallback"));
      }
      const api = typeof window !== "undefined" ? window.NTE_V2 : null;
      const statMarkup = hasPublicCard(card) && api && api.cardFrameStat ? api.cardFrameStat(card) : "";
      if (statMarkup) {
        const stats = this._el("div", "v2-fx-card-stats");
        stats.innerHTML = statMarkup;
        front.firstChild.appendChild(stats);
      }
      const name = this._el("strong", "v2-fx-card-name");
      name.textContent = hasPublicCard(card) ? String(card.name || card.card_id || "") : "";
      const kind = this._el("span", "v2-fx-card-type");
      kind.textContent = typeLabel(card);
      const owner = this._el("span", "v2-fx-card-owner");
      const userName = displayName(event && (event.actor || event.source), card);
      owner.textContent = userName;
      front.appendChild(name);
      front.appendChild(kind);
      front.appendChild(owner);
      if (card && card.description) {
        const desc = this._el("p", "v2-fx-card-desc");
        const api = typeof window !== "undefined" ? window.NTE_V2 : null;
        if (api && api.cardDescriptionMarkup) {
          desc.innerHTML = api.cardDescriptionMarkup(card.description);
        } else {
          desc.textContent = String(card.description);
        }
        front.appendChild(desc);
      }
      if (card && card.copy) {
        const copy = this._el("span", "v2-fx-card-copy");
        copy.textContent = "复制";
        front.appendChild(copy);
      }
      shell.appendChild(front);
      node.appendChild(shell);
      const meta = this._el("div", "v2-fx-card-meta");
      meta.textContent = "使用者 " + (userName || "未知");
      node.appendChild(meta);
      overlay.appendChild(node);
      if (sourceCard && sourceCard.classList) {
        sourceCard.classList.add("v2-fx-source");
      }
      const mobile = typeof window !== "undefined" && typeof window.matchMedia === "function"
        && window.matchMedia("(pointer: coarse) and (max-width: 1200px)").matches;
      const cardW = mobile ? Math.min(220, stageRect.width * 0.3) : 300;
      node.style.width = cardW + "px";
      node.style.height = "auto";
      const measured = readRect(node);
      const cardH = mobile ? Math.min(300, stageRect.height * 0.65)
        : Math.max(430, (measured && measured.height) || 430);
      node.style.height = cardH + "px";
      const landing = this._playLanding(stageRect, cardW, cardH);
      const startScale = Math.max(0.1, Math.min(0.55, (fromRect.width || 18) / cardW));
      const startX = fromRect.left - stageRect.left + (fromRect.width || 0) / 2 - cardW / 2;
      const startY = fromRect.top - stageRect.top + (fromRect.height || 0) / 2 - cardH / 2;
      const endX = landing.x;
      const endY = landing.y;
      node.style.left = startX + "px";
      node.style.top = startY + "px";
      const opponent = this._isOpponentPlay(event);
      if (!opponent) {
        shell.classList.add("is-flipped");
      }
      if (hasPublicCard(card)) {
        try {
          this.onPublicCard(card, event);
        } catch (_error) {}
      }
      this._veil(true);
      this._notice(hasPublicCard(card) ? String(card.name || "") : "出牌", event && event.text, "bottom");
      const fly = this._duration(instant, BEATS.playFly);
      const flip = opponent ? this._duration(instant, BEATS.playFlip) : 0;
      const dx = endX - startX;
      const dy = endY - startY;
      const flyAnim = this._animate(node, [
        { transform: "translate(0px, 0px) scale(" + startScale + ")", opacity: 0.95 },
        { transform: "translate(" + dx + "px, " + dy + "px) scale(1.03)", offset: 0.82, opacity: 1 },
        { transform: "translate(" + dx + "px, " + dy + "px) scale(1)", opacity: 1 },
      ], fly);
      if (flip) {
        await Promise.all([
          flyAnim,
          (async () => {
            await this._wait(Math.max(0, Math.round(fly * 0.62)));
            await this._animate(shell, [
              { transform: "rotateY(0deg)" },
              { transform: "rotateY(180deg)" },
            ], flip);
            shell.classList.add("is-flipped");
          })(),
        ]);
      } else {
        await flyAnim;
      }
      this._settlePlayCard(node, endX, endY);
      await this._wait(this._duration(instant, BEATS.playHold));
      if (event && event.target) {
        this._arrow(centerOf(readRect(node) || { cx: stageRect.cx, cy: stageRect.cy }), this._point(event.target, stageRect), "cast");
        await this._wait(this._duration(instant, BEATS.playAim));
      }
      this._veil(false);
    }

    async _presentAttack(event, instant) {
      const actor = this._entity(event && (event.actor || event.source));
      const target = this._entity(event && event.target);
      const overlay = this._ensureOverlay();
      const stageRect = readRect(overlay) || { cx: 400, cy: 225, left: 0, top: 0, width: 800, height: 450 };
      const from = centerOf(readRect(actor), { x: stageRect.cx - 80, y: stageRect.cy });
      const to = centerOf(readRect(target), { x: stageRect.cx + 80, y: stageRect.cy });
      this._arrow(from, to, "attack");
      const attackCount = Number(overlay.getAttribute("data-v2-fx-attack-count") || 0) + 1;
      overlay.setAttribute("data-v2-fx-attack-count", String(attackCount));
      overlay.setAttribute("data-v2-fx-last-attack", "1");
      this._impact(to, stageRect, "attack");
      if (actor) {
        actor.classList.add("v2-fx-lunge");
        try {
          await this._animate(actor, [
            { transform: "translate(0px, 0px) scale(1)" },
            { transform: "translate(" + Math.max(-36, Math.min(36, (to.x - from.x) * 0.14)) + "px, " + Math.max(-28, Math.min(28, (to.y - from.y) * 0.14)) + "px) scale(1.06)" },
            { transform: "translate(0px, 0px) scale(1)" },
          ], this._duration(instant, BEATS.attackLunge));
        } finally {
          actor.classList.remove("v2-fx-lunge");
          this._resetActorStyle(actor);
        }
      }
      const counterValue = numericValue(event && event.counter);
      if (counterValue != null && counterValue > 0 && target) {
        this._arrow(to, from, "counter");
      }
      if (target) {
        target.classList.add("v2-fx-hurt");
      }
      await this._wait(this._duration(instant, BEATS.attackHold));
      if (target) {
        target.classList.remove("v2-fx-hurt");
      }
    }

    async _presentDamage(event, instant) {
      const overlay = this._ensureOverlay();
      const stageRect = readRect(overlay) || { cx: 400, cy: 225, left: 0, top: 0, width: 800, height: 450 };
      const target = this._entity(event && event.target) || this._entity(event && (event.actor || event.source));
      const point = centerOf(readRect(target), { x: stageRect.cx, y: stageRect.cy });
      const hpLost = fieldDelta(event, "hp");
      const shieldLost = fieldDelta(event, "shield");
      const hpGained = hpLost == null ? null : -hpLost;
      const shieldGained = shieldLost == null ? null : -shieldLost;
      const rawAmount = numericValue(amountOf(event));
      const isHeal = event && event.type === "heal";
      if (event && event.type === "combat" && event.counter_immunity === "support") {
        this._floater(point, stageRect, "无敌", "invincible");
        await this._wait(this._duration(instant, BEATS.damageHold));
        return;
      }
      let shown = false;
      if (isHeal) {
        const healValue = hpGained != null && hpGained > 0 ? hpGained : (rawAmount != null && rawAmount > 0 ? rawAmount : 0);
        if (healValue > 0) {
          if (target) {
            target.classList.add("v2-fx-heal");
          }
          this._floater(point, stageRect, "+" + String(healValue), "heal");
          shown = true;
        }
      } else {
        if (shieldLost != null && shieldLost > 0) {
          if (target) {
            target.classList.add("v2-fx-shield");
          }
          this._floater(point, stageRect, "-" + String(shieldLost), "shield");
          shown = true;
        }
        if (hpLost != null && hpLost > 0) {
          if (target) {
            target.classList.add("v2-fx-hurt");
          }
          this._floater(point, stageRect, "-" + String(hpLost), "damage");
          shown = true;
        } else if (shieldGained != null && shieldGained > 0) {
          if (target) {
            target.classList.add("v2-fx-shield");
          }
          this._floater(point, stageRect, "+" + String(shieldGained), "shield");
          shown = true;
        }
      }
      if (!shown) {
        return;
      }
      const spark = this._el("div", event && event.type === "penetration" ? "v2-fx-burst" : (isHeal ? "v2-fx-flower" : "v2-fx-spark"));
      spark.setAttribute("data-v2-fx-kind", event && event.type ? String(event.type) : "damage");
      spark.style.left = (point.x - stageRect.left) + "px";
      spark.style.top = (point.y - stageRect.top) + "px";
      overlay.appendChild(spark);
      if (!isHeal) {
        this._flash();
        this._shake();
      }
      await this._wait(this._duration(instant, BEATS.damageHold));
      if (target) {
        target.classList.remove("v2-fx-hurt");
        target.classList.remove("v2-fx-shield");
        target.classList.remove("v2-fx-heal");
      }
    }

    async _presentFlower(event, instant) {
      const overlay = this._ensureOverlay();
      const stageRect = readRect(overlay) || { cx: 400, cy: 225, left: 0, top: 0, width: 800, height: 450 };
      const from = this._point(event && (event.actor || event.source), stageRect);
      const to = this._point(event && event.target, stageRect);
      this._arrow(from, to, "flower");
      const flower = this._el("div", "v2-fx-flower");
      flower.setAttribute("data-v2-fx-kind", "flower");
      flower.style.left = (to.x - stageRect.left) + "px";
      flower.style.top = (to.y - stageRect.top) + "px";
      overlay.appendChild(flower);
      const value = amountOf(event);
      this._notice("创生花", event && event.text, "bottom");
      this._floater(to, stageRect, value != null ? "-" + String(value) : "创生花", "harmony");
      await this._wait(this._duration(instant, BEATS.flowerHold));
    }

    async _presentMove(event, instant) {
      const actor = this._entity(event && (event.actor || event.source || event.target));
      if (actor) {
        actor.classList.add("v2-fx-lunge");
        try {
          await this._animate(actor, [
            { transform: "translateY(0px)", opacity: 1 },
            { transform: "translateY(-10px)", opacity: 0.7 },
            { transform: "translateY(0px)", opacity: 1 },
          ], this._duration(instant, BEATS.moveHold));
        } finally {
          actor.classList.remove("v2-fx-lunge");
          this._resetActorStyle(actor);
        }
      } else {
        await this._presentBanner(event, instant);
      }
    }

    async _presentEnter(event, instant) {
      const actor = this._entity(event && (event.actor || event.source || event.target));
      this._notice("入场", event && event.text, "bottom");
      if (actor) {
        actor.classList.add("v2-fx-enter");
        try {
          await this._animate(actor, [
            { transform: "translateY(28px) scale(0.92)", opacity: 0.35 },
            { transform: "translateY(-8px) scale(1.06)", opacity: 1 },
            { transform: "translateY(0px) scale(1)", opacity: 1 },
          ], this._duration(instant, BEATS.moveHold + 80));
        } finally {
          actor.classList.remove("v2-fx-enter");
          this._resetActorStyle(actor);
        }
      } else {
        await this._presentTitle(event, instant, "入场");
      }
    }

    async _presentHarmony(event, instant) {
      const overlay = this._ensureOverlay();
      const stageRect = readRect(overlay) || { cx: 400, cy: 225, left: 0, top: 0 };
      const names = { "创生": "01", "覆纹": "02", "黯星": "03", "浊燃": "04", "浸染": "05", "延滞": "06", "盈蓄": "07", "失谐": "08" };
      const text = String(event && event.text || "");
      const name = Object.keys(names).find((value) => text.includes(value)) || "环合";
      const area = this.root && this.root.querySelector ? this.root.querySelector('#v2-play-area') : null;
      const point = centerOf(readRect(area), { x: stageRect.cx, y: stageRect.cy });
      const art = this._el("div", "v2-fx-harmony-art");
      art.setAttribute("aria-label", name);
      art.style.left = (point.x - stageRect.left) + "px";
      art.style.top = (point.y - stageRect.top) + "px";
      if (names[name]) {
        ["01", "02"].forEach((part) => {
          const img = this._el("img", "v2-fx-harmony-glyph");
          img.src = "/static/card_game/images/harmony/fanying" + names[name] + "_" + part + "_8.png";
          img.alt = "";
          art.appendChild(img);
        });
      } else {
        art.textContent = name;
      }
      overlay.appendChild(art);
      await this._animate(art, [
        { transform: "translate(-50%, -50%) scale(0.35)", opacity: 0 },
        { transform: "translate(-50%, -50%) scale(1.12)", opacity: 1, offset: 0.2 },
        { transform: "translate(-50%, -50%) scale(1)", opacity: 1, offset: 0.35 },
        { transform: "translate(-50%, -50%) scale(1)", opacity: 1, offset: 0.78 },
        { transform: "translate(-50%, -56%) scale(1.04)", opacity: 0 },
      ], this._duration(instant, 1100));
      art.remove();
    }

    async _presentDot(event, instant, kind) {
      const overlay = this._ensureOverlay();
      const stageRect = readRect(overlay) || { cx: 400, cy: 225, left: 0, top: 0 };
      const target = this._entity(event && (event.target || event.actor || event.source));
      const point = centerOf(readRect(target), { x: stageRect.cx, y: stageRect.cy });
      const label = TYPE_TITLES[kind] || kind;
      this._notice(label, event && event.text, "bottom");
      if (target) {
        target.classList.add("v2-fx-" + kind);
      }
      const spark = this._el("div", "v2-fx-spark is-" + kind);
      spark.setAttribute("data-v2-fx-kind", kind);
      spark.style.left = (point.x - stageRect.left) + "px";
      spark.style.top = (point.y - stageRect.top) + "px";
      overlay.appendChild(spark);
      const value = amountOf(event);
      this._floater(point, stageRect, value != null ? label + " -" + String(value) : label, kind);
      await this._wait(this._duration(instant, BEATS.statusHold));
      if (target) {
        target.classList.remove("v2-fx-" + kind);
      }
    }

    async _presentTitle(event, instant, fallback) {
      const overlay = this._ensureOverlay();
      const title = this._el("div", "v2-fx-title");
      title.textContent = (event && event.text) || fallback || TYPE_TITLES[event && event.type] || "";
      overlay.appendChild(title);
      this._notice("", "", "bottom");
      await this._wait(this._duration(instant, BEATS.titleHold));
    }

    async _presentStatus(event, instant, klass, label) {
      const node = this._entity(event && (event.target || event.actor || event.source));
      const className = "v2-fx-" + klass;
      try {
        if (node) {
          node.classList.add(className);
        }
        const overlay = this._ensureOverlay();
        const stageRect = readRect(overlay) || { cx: 400, cy: 225, left: 0, top: 0, width: 800, height: 450 };
        const point = centerOf(readRect(node), { x: stageRect.cx, y: stageRect.cy });
        const burst = this._el("div", klass === "awaken" ? "v2-fx-burst" : "v2-fx-spark");
        burst.setAttribute("data-v2-fx-kind", klass);
        burst.style.left = (point.x - stageRect.left) + "px";
        burst.style.top = (point.y - stageRect.top) + "px";
        overlay.appendChild(burst);
        if (klass === "awaken" || klass === "shape" || klass === "revive") {
          this._floater(point, stageRect, label, klass === "awaken" ? "awaken" : klass);
        }
        this._notice(label, event && event.text, "top");
        await this._wait(this._duration(instant, BEATS.statusHold));
      } finally {
        if (node) {
          node.classList.remove(className);
          this._resetActorStyle(node);
        }
      }
    }

    async _presentFloater(event, instant, klass, label) {
      const overlay = this._ensureOverlay();
      const stageRect = readRect(overlay) || { cx: 400, cy: 225, left: 0, top: 0, width: 800, height: 450 };
      const point = this._point(event && (event.target || event.actor || event.source), stageRect);
      const value = amountOf(event);
      const node = this._entity(event && (event.target || event.actor || event.source));
      if (node) {
        node.classList.add("v2-fx-" + klass);
      }
      this._floater(point, stageRect, value != null ? label + " +" + String(value) : label, klass);
      this._notice(label, event && event.text, "bottom");
      await this._wait(this._duration(instant, BEATS.floaterHold));
      if (node) {
        node.classList.remove("v2-fx-" + klass);
      }
    }

    _isSilent(event) {
      if (!event) {
        return true;
      }
      if (event.type === "resource" || event.type === "discard") {
        return true;
      }
      if (event.present === false) {
        return true;
      }
      if (event.type === "ultimate" && /终结结束/.test(String(event.text || ""))) {
        return true;
      }
      if (event.present === true) {
        return false;
      }
      return false;
    }

    async _presentInitiative(event, instant) {
      const overlay = this.root && typeof this.root.querySelector === "function"
        ? this.root.querySelector("#v2-initiative-overlay")
        : null;
      const first = (event && (event.first_side || event.side)) || "";
      const viewer = this._viewerSide();
      const isFirst = Boolean(viewer && first && first === viewer);
      this._notice(isFirst ? "先手" : "后手", event && event.text, "bottom");
      if (!overlay) {
        await this._presentBanner(event, instant);
        return;
      }
      overlay.classList.toggle("is-second", !isFirst);
      const label = typeof overlay.querySelector === "function" ? overlay.querySelector("#v2-initiative-label") : null;
      const copy = typeof overlay.querySelector === "function" ? overlay.querySelector("#v2-initiative-copy") : null;
      if (label) {
        label.textContent = isFirst ? "先手" : "后手";
      }
      if (copy) {
        copy.textContent = isFirst ? "你先行动。" : "对手先行动。你有 5 点开局护盾。";
      }
      overlay.classList.add("is-playing");
      overlay.classList.remove("is-leaving");
      if (typeof overlay.removeAttribute === "function") {
        overlay.removeAttribute("hidden");
      }
      overlay.hidden = false;
      await this._wait(this._duration(instant, 1400));
      overlay.classList.add("is-leaving");
      await this._wait(this._duration(instant, 280));
      overlay.hidden = true;
      overlay.classList.remove("is-playing", "is-leaving");
    }

    async _presentMulligan(event, instant) {
      this._notice("起手换牌", event && event.text, "bottom");
      const table = typeof window !== "undefined" ? window.NTE_V2_TABLE : null;
      const rail = this.root && typeof this.root.querySelector === "function"
        ? this.root.querySelector("#v2-mulligan-rail")
        : null;
      const outgoing = ((event && event.card_ids) || []).map(String).filter(Boolean);
      if (this._isOpponentPlay(event) || !outgoing.length || !table || typeof table.animateMulliganSwap !== "function" || !rail) {
        await this._presentBanner(event, instant);
        return;
      }
      await table.animateMulliganSwap({
        rail: rail,
        outgoingIds: outgoing,
        incomingIds: [],
        reducedMotion: instant || (this.options && this.options.reducedMotion),
        replace: function () {},
      });
    }

    _drawSlotRect(event, stageRect) {
      const hand = this._hand(sideOf(event));
      const card = event && event.card;
      let slot = null;
      let insertBefore = false;
      let existing = false;
      if (hand && typeof hand.querySelector === "function") {
        if (card && card.instance_id) {
          try {
            slot = hand.querySelector('[data-instance-id="' + cssEscape(card.instance_id) + '"]')
              || hand.querySelector('[data-card-id="' + cssEscape(card.instance_id) + '"]');
          } catch (_error) {}
        }
        existing = Boolean(slot);
        if (!slot && typeof hand.querySelectorAll === "function") {
          const cards = Array.from(hand.querySelectorAll(".v2-hand-card") || []);
          const backs = Array.from(hand.querySelectorAll(".v2-hidden-card") || []);
          const nodes = cards.concat(backs);
          const owner = card && (card.hand_face || card).character_id;
          const owned = owner ? cards.filter((node) => node.getAttribute("data-card-owner") === owner) : [];
          slot = owned.length ? owned[owned.length - 1] : null;
          if (!slot && owner) {
            let order = [];
            try { order = JSON.parse(hand.getAttribute("data-character-order") || "[]"); } catch (_error) {}
            const ownerIndex = order.indexOf(owner);
            if (ownerIndex >= 0) {
              slot = cards.find((node) => order.indexOf(node.getAttribute("data-card-owner")) > ownerIndex) || null;
              insertBefore = Boolean(slot);
            }
          }
          if (!slot) slot = nodes.length ? nodes[nodes.length - 1] : null;
        }
      }
      const slotRect = readRect(slot);
      const handRect = readRect(hand);
      const usable = function (rect) {
        return Boolean(rect && rect.width > 8 && rect.height > 8);
      };
      const width = (usable(slotRect) && slotRect.width) || (hasPublicCard(event && event.card) ? 148 : 22);
      const height = (usable(slotRect) && slotRect.height) || (hasPublicCard(event && event.card) ? 108 : 32);
      if (usable(slotRect)) {
        const left = existing || insertBefore ? slotRect.left : slotRect.right + 8;
        return {
          left: left,
          top: slotRect.top,
          width: width,
          height: height,
          cx: left + width / 2,
          cy: slotRect.cy,
          right: left + width,
          bottom: slotRect.bottom,
        };
      }
      if (usable(handRect)) {
        const left = handRect.left + Math.max(0, ((handRect.width || width) - width) / 2);
        return {
          left: left,
          top: handRect.top,
          width: width,
          height: height,
          cx: left + width / 2,
          cy: handRect.top + height / 2,
          right: left + width,
          bottom: handRect.top + height,
        };
      }
      const stageW = (stageRect && stageRect.width) || 800;
      const stageH = (stageRect && stageRect.height) || 450;
      return {
        left: stageW / 2 - width / 2,
        top: stageH - height - 24,
        width: width,
        height: height,
        cx: stageW / 2,
        cy: stageH - height / 2 - 24,
        right: stageW / 2 + width / 2,
        bottom: stageH - 24,
      };
    }

    async _presentMulliganDraw(event, instant) {
      // Replacement draws belong in the opening rail, not the normal draw close-up.
      this._patch(event);
      const card = event && event.card;
      const rail = this.root.querySelector("#v2-mulligan-rail");
      if (this._isOpponentPlay(event) || !rail || !card || !card.instance_id) return;
      const id = cssEscape(card.instance_id);
      const node = rail.querySelector('[data-instance-id="' + id + '"]')
        || rail.querySelector('[data-card-id="' + id + '"]');
      if (!node) return;
      if (instant || this.options.reducedMotion) return;
      node.classList.add("is-entering");
      try {
        await this._wait(this._duration(instant, 420));
      } finally {
        node.classList.remove("is-entering");
      }
    }

    async _presentDraw(event, instant) {
      const generated = event && event.type === "gain";
      if (!generated && this.root && this.root.classList && this.root.classList.contains("is-mulligan")) {
        await this._presentMulliganDraw(event, instant);
        return;
      }
      const overlay = this._ensureOverlay();
      const rawCard = event && event.card;
      const api = typeof window !== "undefined" ? window.NTE_V2 : null;
      const card = api && api.handCardFace ? api.handCardFace(rawCard) : rawCard;
      const publicCard = hasPublicCard(card);
      const opponent = this._isOpponentPlay(event);
      const portraitSrc = this._safePortrait(event && (event.actor || event.source))
        || this._safePortrait(sideOf(event) + ":" + ((card && card.character_id) || ""));
      const node = this._el("div", "v2-fx-draw-card");
      node.setAttribute("data-v2-fx-kind", generated ? "gain-generated" : (publicCard ? "draw-public" : "draw-hidden"));
      const show = this._el("div", publicCard
        ? "v2-fx-card-front v2-fx-draw-formula"
        : "v2-fx-card-back v2-fx-draw-formula");
      if (card && card.type) {
        show.setAttribute("data-card-type", String(card.type));
        node.setAttribute("data-card-type", String(card.type));
      }
      if (publicCard) {
        if (portraitSrc) {
          const art = this._el("div", "v2-fx-card-art");
          const img = this._el("img", "v2-fx-card-portrait");
          img.setAttribute("src", portraitSrc);
          img.setAttribute("alt", "");
          art.appendChild(img);
          show.appendChild(art);
        } else {
          show.appendChild(this._el("div", "v2-fx-card-art is-fallback"));
        }
        const statMarkup = api && api.cardFrameStat ? api.cardFrameStat(card) : "";
        if (statMarkup) {
          const stats = this._el("div", "v2-fx-card-stats");
          stats.innerHTML = statMarkup;
          show.firstChild.appendChild(stats);
        }
        const name = this._el("strong", "v2-fx-card-name");
        name.textContent = String(card.name || card.card_id || "");
        show.appendChild(name);
        const kind = this._el("span", "v2-fx-card-type");
        kind.textContent = typeLabel(card);
        show.appendChild(kind);
        if (card.description) {
          const desc = this._el("p", "v2-fx-card-desc");
          const api = typeof window !== "undefined" ? window.NTE_V2 : null;
          if (api && api.cardDescriptionMarkup) {
            desc.innerHTML = api.cardDescriptionMarkup(card.description);
          } else {
            desc.textContent = String(card.description);
          }
          show.appendChild(desc);
        }
        try {
          this.onPublicCard(card, event);
        } catch (_error) {}
      }
      const handFace = this._el("article", "v2-hand-card v2-fx-draw-hand");
      if (!publicCard || (generated && opponent)) {
        handFace.className = "v2-hidden-card v2-fx-draw-hand";
      } else {
        if (card && card.type) {
          handFace.setAttribute("data-card-type", String(card.type));
        }
        if (portraitSrc && handFace.style) {
          const art = "url(" + JSON.stringify(portraitSrc) + ")";
          if (typeof handFace.style.setProperty === "function") {
            handFace.style.setProperty("--card-art", art);
          } else {
            handFace.style["--card-art"] = art;
          }
        }
        const handName = this._el("strong");
        handName.textContent = String(card.name || card.card_id || "卡牌");
        handFace.appendChild(handName);
        if (card && card.description) {
          const handDesc = this._el("p", "subtle");
          const api = typeof window !== "undefined" ? window.NTE_V2 : null;
          if (api && api.cardDescriptionMarkup) {
            handDesc.innerHTML = api.cardDescriptionMarkup(card.description);
          } else {
            handDesc.textContent = String(card.description);
          }
          handFace.appendChild(handDesc);
        }
        const statMarkup = api && api.cardFrameStat ? api.cardFrameStat(card) : "";
        if (statMarkup) {
          const stats = this._el("span");
          stats.innerHTML = statMarkup;
          handFace.appendChild(stats);
        }
      }
      node.appendChild(show);
      node.appendChild(handFace);
      overlay.appendChild(node);
      const stageRect = readRect(overlay) || { left: 0, top: 0, width: 800, height: 450, cx: 400, cy: 225 };
      const compactBack = !publicCard && typeof window !== "undefined" && typeof window.matchMedia === "function"
        && window.matchMedia("(pointer: coarse) and (max-width: 1200px)").matches;
      const cardW = compactBack ? Math.min(160, stageRect.width * 0.22) : 300;
      node.style.width = cardW + "px";
      node.style.height = "auto";
      const measured = readRect(node);
      const cardH = compactBack ? Math.min(230, stageRect.height * 0.5)
        : Math.max(430, (measured && measured.height) || 430);
      node.style.height = cardH + "px";
      const slot = this._drawSlotRect(event, stageRect);
      const slotLeft = slot.left - stageRect.left;
      const slotTop = slot.top - stageRect.top;
      let hoverX = slot.cx - stageRect.left - cardW / 2;
      hoverX = Math.max(12, Math.min(hoverX, (stageRect.width || 800) - cardW - 12));
      let hoverY = opponent
        ? slot.bottom - stageRect.top + 12
        : slot.top - stageRect.top - 16 - cardH;
      hoverY = Math.max(8, Math.min(hoverY, (stageRect.height || 450) - cardH - 8));
      if (generated) {
        hoverX = ((stageRect.width || 800) - cardW) / 2;
        hoverY = ((stageRect.height || 450) - cardH) / 2;
      }
      const startX = generated ? hoverX : (stageRect.width || 800) + 36;
      node.style.left = startX + "px";
      node.style.top = hoverY + "px";
      if (generated) node.style.transformOrigin = "center center";
      this._notice(generated ? "" : (publicCard ? String(card.name || "抽牌") : "抽牌"), generated ? "" : event && event.text, "bottom");
      const fly = this._duration(instant, BEATS.drawFly);
      await this._animate(node, generated ? [
        { transform: "scale(0.6)", opacity: 0 },
        { transform: "scale(1.05)", opacity: 1, offset: 0.75 },
        { transform: "scale(1)", opacity: 1 },
      ] : [
        { transform: "translate(0px, 0px)", opacity: 0.35 },
        { transform: "translate(" + (hoverX - startX) + "px, 0px)", opacity: 1 },
      ], fly);
      this._settlePlayCard(node, hoverX, hoverY);
      await this._wait(this._duration(instant, publicCard ? BEATS.drawHold : BEATS.linger));
      node.classList.add("is-dropping");
      if (generated) node.style.transformOrigin = "top left";
      const insert = this._duration(instant, BEATS.drawInsert);
      const scaleX = Math.max(0.18, (slot.width || 148) / cardW);
      const scaleY = Math.max(0.18, (slot.height || 108) / cardH);
      await Promise.all([
        this._animate(node, [
          { transform: "translate(0px, 0px) scale(1, 1)", opacity: 1 },
          { transform: "translate(" + (slotLeft - hoverX) + "px, " + (slotTop - hoverY) + "px) scale(" + scaleX + ", " + scaleY + ")", opacity: 1 },
        ], insert),
        this._animate(show, [{ opacity: 1 }, { opacity: 0 }], insert),
        this._animate(handFace, [{ opacity: 0 }, { opacity: 1 }], insert),
      ]);
      if (node.parentNode) {
        try {
          node.parentNode.removeChild(node);
        } catch (_error) {}
      }
    }

    async _presentMark(event, instant, klass, label) {
      const overlay = this._ensureOverlay();
      const stageRect = readRect(overlay) || { cx: 400, cy: 225, left: 0, top: 0, width: 800, height: 450 };
      const point = this._point(event && (event.actor || event.source || event.target), stageRect);
      const chip = this._el("div", "v2-fx-" + klass);
      chip.setAttribute("data-v2-fx-kind", klass);
      const card = event && event.card;
      const cardName = hasPublicCard(card) ? String(card.name || card.card_id) : "";
      chip.textContent = cardName ? label + " " + cardName : label;
      chip.style.left = (point.x - stageRect.left) + "px";
      chip.style.top = (point.y - stageRect.top) + "px";
      overlay.appendChild(chip);
      if (klass === "copy" && hasPublicCard(card)) {
        try {
          this.onPublicCard(card, event);
        } catch (_error) {}
      }
      this._notice(chip.textContent, event && event.text, "bottom");
      await this._wait(this._duration(instant, BEATS.markHold));
    }

    async _presentBanner(event, instant) {
      const label = (event && event.text) || TYPE_TITLES[event && event.type] || (event && event.type) || "";
      this._notice(label, "", "top");
      await this._wait(this._duration(instant, BEATS.bannerHold));
    }

    _note(event, kind) {
      const overlay = this._ensureOverlay();
      overlay.setAttribute("data-v2-fx-last-type", kind);
      const card = event && event.card;
      if (hasPublicCard(card)) {
        overlay.setAttribute("data-v2-fx-last-card", String(card.name || card.card_id || ""));
        overlay.setAttribute("data-v2-fx-last-desc", card.description ? String(card.description) : "");
      }
      const actor = displayName(event && (event.actor || event.source), card);
      if (actor) {
        overlay.setAttribute("data-v2-fx-last-actor", actor);
      }
      const seen = overlay.getAttribute("data-v2-fx-types") || "";
      overlay.setAttribute("data-v2-fx-types", seen ? seen + "," + kind : kind);
    }

    _notice(title, detail = "", position = "bottom") {
      const overlay = this._ensureOverlay();
      let node = overlay.querySelector(".v2-fx-notice");
      if (!node) {
        node = this._el("div", "v2-fx-notice");
        overlay.appendChild(node);
      }
      const placement = position === "top" ? "top" : "bottom";
      node.setAttribute("class", "v2-fx-notice " + (placement === "top" ? "v2-fx-banner" : "v2-fx-caption"));
      node.setAttribute("data-v2-fx-position", placement);
      const headline = String(title || "").trim();
      const body = String(detail || "").trim();
      node.textContent = body && body !== headline ? headline + " · " + body : (headline || body);
      node.hidden = !node.textContent;
      return node;
    }

    _veil(on) {
      const overlay = this._ensureOverlay();
      overlay.classList.toggle("is-veiled", !!on);
    }

    _flash() {
      const overlay = this._ensureOverlay();
      const flash = this._el("div", "v2-fx-flash");
      overlay.appendChild(flash);
    }

    _shake() {
      const stage = this._stage();
      if (!stage || !stage.classList) {
        return;
      }
      stage.classList.add("v2-fx-shake");
      this._wait(this._duration(this._instant, 280)).then(() => {
        if (stage.classList) {
          stage.classList.remove("v2-fx-shake");
        }
      }).catch(noop);
    }

    _impact(point, stageRect, klass) {
      const overlay = this._ensureOverlay();
      const node = this._el("div", "v2-fx-impact" + (klass ? " is-" + klass : ""));
      node.style.left = (point.x - stageRect.left) + "px";
      node.style.top = (point.y - stageRect.top) + "px";
      overlay.appendChild(node);
      return node;
    }

    _floater(point, stageRect, text, klass) {
      const overlay = this._ensureOverlay();
      const node = this._el("div", "v2-fx-floater is-" + klass);
      node.textContent = String(text);
      node.style.left = (point.x - stageRect.left) + "px";
      node.style.top = (point.y - stageRect.top) + "px";
      overlay.appendChild(node);
      overlay.setAttribute("data-v2-fx-last-floater", String(text));
      overlay.setAttribute("data-v2-fx-last-floater-kind", String(klass || ""));
      const seen = overlay.getAttribute("data-v2-fx-floaters") || "";
      overlay.setAttribute("data-v2-fx-floaters", seen ? seen + "|" + String(text) : String(text));
      const duration = this._duration(this._instant, 900);
      if (duration > 0) {
        // Cancelling the finished Web Animation must not reveal the base element again.
        node.style.opacity = "0";
        this._animate(node, [
          { transform: "translate(-50%, -50%) scale(0.8)", opacity: 0.2 },
          { transform: "translate(-50%, -120%) scale(1)", opacity: 1 },
          { transform: "translate(-50%, -160%) scale(1)", opacity: 0 },
        ], duration).catch(noop).finally(() => {
          if (node.parentNode) node.parentNode.removeChild(node);
        });
      }
      return node;
    }

    _arrow(from, to, klass) {
      const svg = this._ensureSvg();
      const path = this._svgEl("path");
      const classes = ["v2-fx-arrow"];
      if (klass === "cast") {
        classes.push("is-cast");
      } else if (klass === "flower") {
        classes.push("is-flower");
      } else if (klass === "counter") {
        classes.push("is-counter");
      }
      path.setAttribute("class", classes.join(" "));
      const midX = (from.x + to.x) / 2;
      const midY = Math.min(from.y, to.y) - 36;
      path.setAttribute("d", "M " + from.x + " " + from.y + " Q " + midX + " " + midY + " " + to.x + " " + to.y);
      svg.appendChild(path);
      return path;
    }

    _point(ref, stageRect) {
      const node = this._entity(ref);
      return centerOf(readRect(node), { x: stageRect.cx || 400, y: stageRect.cy || 225 });
    }

    _patch(event) {
      const key = eventKey(event);
      if (key && this._patched.has(key)) {
        return;
      }
      if (key) {
        this._patched.add(key);
      }
      if (!event || event.patch == null) {
        return;
      }
      try {
        this.applyPatch(event.patch, event);
      } catch (_error) {}
    }

    _flushPatches(events) {
      asArray(events).forEach((event) => {
        const key = eventKey(event);
        if (key && this._seen.has(key) === false) {
          this._seen.add(key);
        }
        this._patch(event);
      });
    }

    _duration(instant, base) {
      if (instant || this._cancelled) {
        return 0;
      }
      if (this.options.reducedMotion) {
        return 24;
      }
      const scaled = Math.round(base / (this.options.speed || 1));
      return Math.max(0, Math.min(MAX_EVENT_MS, scaled));
    }

    _wait(ms) {
      if (!ms || ms <= 0 || this._cancelled) {
        return Promise.resolve();
      }
      const clock = this._clock;
      if (clock && typeof clock.wait === "function") {
        return this._cancellable(clock.wait(ms));
      }
      return this._cancellable(new Promise((resolve) => {
        const timer = setTimeout(resolve, ms);
        this._waiters.add(() => {
          clearTimeout(timer);
          resolve();
        });
      }));
    }

    _cancellable(promise) {
      return new Promise((resolve) => {
        let done = false;
        const finish = () => {
          if (done) {
            return;
          }
          done = true;
          resolve();
        };
        this._waiters.add(finish);
        Promise.resolve(promise).then(finish, finish);
      });
    }

    async _animate(node, keyframes, duration) {
      if (!node || duration <= 0 || this._cancelled) {
        return;
      }
      if (typeof node.animate !== "function") {
        await this._wait(duration);
        return;
      }
      let animation = null;
      try {
        animation = node.animate(keyframes, { duration: duration, easing: "ease-out", fill: "forwards" });
        if (animation) {
          this._anims.add(animation);
        }
        const finished = animation && animation.finished
          ? Promise.resolve(animation.finished).catch(noop)
          : this._wait(duration);
        await Promise.race([
          finished,
          this._wait(duration + 48),
        ]);
      } catch (_error) {
        await this._wait(Math.min(duration, 16));
      } finally {
        if (animation) {
          this._anims.delete(animation);
          try {
            if (typeof animation.cancel === "function") {
              animation.cancel();
            }
          } catch (_error) {}
        }
        if (node && node.classList && node.classList.contains("v2-character-card")) {
          this._resetActorStyle(node);
        }
      }
    }

    _resetActorStyle(node) {
      if (!node || !node.style) {
        return;
      }
      node.style.transform = "";
      node.style.opacity = "";
    }

    _resetActorFx() {
      const root = this.root;
      if (!root || typeof root.querySelectorAll !== "function") {
        return;
      }
      Array.from(root.querySelectorAll(".v2-character-card")).forEach((node) => {
        if (node.classList) {
          ACTOR_FX_CLASSES.forEach((name) => {
            try {
              node.classList.remove(name);
            } catch (_error) {}
          });
        }
        this._resetActorStyle(node);
      });
    }

    _stopAnims() {
      this._anims.forEach((animation) => {
        try {
          if (animation && typeof animation.cancel === "function") {
            animation.cancel();
          }
        } catch (_error) {}
      });
      this._anims.clear();
    }

    _resumeWaiters() {
      const waiters = Array.from(this._waiters);
      this._waiters.clear();
      waiters.forEach((resume) => {
        try {
          resume();
        } catch (_error) {}
      });
    }

    _setBusy(value) {
      this._busy = !!value;
      if (this._overlay) {
        this._overlay.setAttribute("data-v2-fx-busy", this._busy ? "1" : "0");
      }
      try {
        this.onBusy(this._busy);
      } catch (_error) {}
    }

    _ensureOverlay() {
      if (this._overlay && this._overlay.parentNode) {
        this._syncOverlayFlags();
        return this._overlay;
      }
      const documentRef = this._document();
      const overlay = this._el("div", "v2-fx-overlay");
      overlay.setAttribute("data-v2-fx-root", "1");
      const layer = this._el("div", "v2-fx-layer");
      overlay.appendChild(layer);
      this._overlay = overlay;
      this._layer = layer;
      this._svg = null;
      const host = this._stage() || this.root;
      if (host && typeof host.appendChild === "function") {
        host.appendChild(overlay);
      } else if (documentRef && documentRef.body) {
        documentRef.body.appendChild(overlay);
      }
      this._syncOverlayFlags();
      return overlay;
    }

    _ensureSvg() {
      this._ensureOverlay();
      if (this._svg && this._svg.parentNode) {
        return this._svg;
      }
      const svg = this._svgEl("svg");
      svg.setAttribute("class", "v2-fx-svg");
      this._overlay.appendChild(svg);
      this._svg = svg;
      return svg;
    }

    _clearFx(removeOverlay) {
      this._stopAnims();
      this._resetActorFx();
      const stage = this._stage();
      if (stage && stage.classList) {
        stage.classList.remove("v2-fx-shake");
      }
      if (this._overlay && this._overlay.classList) {
        this._overlay.classList.remove("is-veiled");
      }
      if (this._layer) {
        this._empty(this._layer);
      }
      if (this._overlay) {
        Array.from(this._overlay.childNodes || []).forEach((child) => {
          if (child !== this._layer) {
            try {
              this._overlay.removeChild(child);
            } catch (_error) {}
          }
        });
        this._svg = null;
      }
      if (removeOverlay && this._overlay && this._overlay.parentNode) {
        try {
          this._overlay.parentNode.removeChild(this._overlay);
        } catch (_error) {}
      }
    }

    _empty(node) {
      if (!node) {
        return;
      }
      if (typeof node.replaceChildren === "function") {
        node.replaceChildren();
        return;
      }
      while (node.firstChild) {
        node.removeChild(node.firstChild);
      }
    }

    _syncOverlayFlags() {
      if (!this._overlay) {
        return;
      }
      this._overlay.setAttribute("data-v2-fx-reduced", this.options.reducedMotion ? "1" : "0");
      this._overlay.setAttribute("data-v2-fx-muted", this.options.muted ? "1" : "0");
      this._overlay.setAttribute("data-v2-fx-speed", String(this.options.speed || 1));
    }

    _stage() {
      const root = this.root;
      if (!root) {
        return null;
      }
      if (root.id === "v2-fx-stage") {
        return root;
      }
      if (typeof root.querySelector === "function") {
        return root.querySelector("#v2-fx-stage") || root;
      }
      return root;
    }

    _document() {
      if (this.root && this.root.ownerDocument) {
        return this.root.ownerDocument;
      }
      return typeof document !== "undefined" ? document : null;
    }

    _el(tag, className) {
      const documentRef = this._document();
      const node = documentRef && typeof documentRef.createElement === "function"
        ? documentRef.createElement(tag)
        : { tagName: String(tag).toUpperCase(), children: [], style: {}, classList: { add: noop, remove: noop, contains: function () { return false; } }, setAttribute: noop, appendChild: noop };
      if (className) {
        node.className = className;
        if (typeof node.setAttribute === "function") {
          node.setAttribute("class", className);
        }
      }
      return node;
    }

    _svgEl(tag) {
      const documentRef = this._document();
      if (documentRef && typeof documentRef.createElementNS === "function") {
        return documentRef.createElementNS(NS, tag);
      }
      return this._el(tag);
    }

    _viewerSide() {
      if (this.options && this.options.viewerSide) {
        return String(this.options.viewerSide);
      }
      if (!this.root || typeof this.root.querySelector !== "function") {
        return "";
      }
      const hand = this.root.querySelector("#v2-player-hand[data-hand-side]");
      return hand ? String(hand.getAttribute("data-hand-side") || "") : "";
    }

    _isOpponentPlay(event) {
      const side = sideOf(event);
      const viewer = this._viewerSide();
      if (viewer) {
        return !!(side && side !== viewer);
      }
      return side === "b";
    }

    _playLanding(stageRect, cardW, cardH) {
      const board = this.root && typeof this.root.querySelector === "function"
        ? this.root.querySelector("#v2-turn-board")
        : null;
      const rect = readRect(board);
      if (rect && rect.width) {
        return {
          x: rect.left - stageRect.left + rect.width / 2 - cardW / 2,
          y: rect.top - stageRect.top + Math.max(8, rect.height / 2 - cardH / 2),
        };
      }
      return {
        x: Math.max(8, (stageRect.width || 800) * 0.06),
        y: Math.max(8, (stageRect.height || 450) / 2 - cardH / 2),
      };
    }

    _settlePlayCard(node, endX, endY) {
      if (!node || !node.style) {
        return;
      }
      try {
        if (typeof node.getAnimations === "function") {
          node.getAnimations().forEach((anim) => {
            try {
              anim.cancel();
            } catch (_error) {}
          });
        }
      } catch (_error) {}
      node.style.left = endX + "px";
      node.style.top = endY + "px";
      node.style.transform = "none";
    }

    _hand(side) {
      if (!this.root || typeof this.root.querySelector !== "function" || !side) {
        return null;
      }
      if (this.root.classList && this.root.classList.contains("is-mulligan")) {
        const rail = this.root.querySelector("#v2-mulligan-rail");
        if (rail) {
          return rail;
        }
      }
      const viewer = this._viewerSide();
      try {
        if (viewer && side === viewer) {
          const playerHand = this.root.querySelector("#v2-player-hand");
          if (playerHand) {
            return playerHand;
          }
        }
        if (viewer && side && side !== viewer) {
          const opponentHand = this.root.querySelector("#v2-opponent-hand");
          if (opponentHand) {
            return opponentHand;
          }
        }
        const nodes = typeof this.root.querySelectorAll === "function"
          ? Array.from(this.root.querySelectorAll('[data-hand-side="' + cssEscape(side) + '"]'))
          : [];
        for (let index = 0; index < nodes.length; index += 1) {
          const node = nodes[index];
          if (!node || node.id === "v2-mulligan-rail") {
            continue;
          }
          const rect = readRect(node);
          if (rect && rect.width > 8 && rect.height > 8) {
            return node;
          }
        }
        return this.root.querySelector("#v2-player-hand") || nodes[0] || null;
      } catch (_error) {
        return null;
      }
    }

    _sourceCard(hand, card, _side) {
      if (!hand) {
        return null;
      }
      const instanceId = card && card.instance_id != null && card.instance_id !== "" ? String(card.instance_id) : "";
      if (instanceId && typeof hand.querySelector === "function") {
        try {
          const exact = hand.querySelector('[data-instance-id="' + cssEscape(instanceId) + '"]');
          if (exact) {
            return exact;
          }
        } catch (_error) {}
      }
      if (typeof hand.querySelector === "function") {
        const hidden = hand.querySelector(".v2-hidden-card");
        if (hidden) {
          return hidden;
        }
      }
      const fallback = hand.firstChild || null;
      if (!fallback) {
        return null;
      }
      const fallbackId = typeof fallback.getAttribute === "function" ? fallback.getAttribute("data-instance-id") : "";
      if (instanceId && fallbackId && fallbackId !== instanceId) {
        return null;
      }
      return fallback;
    }

    _safePortrait(ref) {
      const node = this._entity(ref);
      if (!node) {
        return "";
      }
      let img = null;
      if (node.tagName && String(node.tagName).toLowerCase() === "img") {
        img = node;
      } else if (typeof node.querySelector === "function") {
        img = node.querySelector("img");
      }
      const raw = img
        ? ((typeof img.getAttribute === "function" && img.getAttribute("src")) || img.src || "")
        : "";
      return safeLocalSrc(raw);
    }

    _entity(ref) {
      if (!ref || !this.root || typeof this.root.querySelector !== "function") {
        return null;
      }
      const raw = String(ref);
      try {
        const exact = this.root.querySelector('[data-entity-id="' + cssEscape(raw) + '"]');
        if (exact) {
          return exact;
        }
        const parts = raw.split(":");
        if (parts[1] === "player") {
          return this.root.querySelector('[data-player-side="' + cssEscape(parts[0]) + '"]');
        }
      } catch (_error) {
        return null;
      }
      return null;
    }

    _listenUnlock() {
      const target = typeof globalThis !== "undefined" ? globalThis : null;
      if (!target || typeof target.addEventListener !== "function") {
        return;
      }
      target.addEventListener("pointerdown", this._unlockBound, true);
      target.addEventListener("keydown", this._unlockBound, true);
      target.addEventListener("touchstart", this._unlockBound, true);
    }

    _unlockAudio() {
      if (this._audioReady) {
        return;
      }
      try {
        const Context = (typeof AudioContext !== "undefined" && AudioContext)
          || (typeof webkitAudioContext !== "undefined" && webkitAudioContext);
        if (!Context) {
          return;
        }
        this._audio = this._audio || new Context();
        if (this._audio && typeof this._audio.resume === "function") {
          this._audio.resume().catch(noop);
        }
        this._audioReady = true;
      } catch (_error) {
        this._audioReady = false;
      }
    }

    _suspendAudio() {
      if (this._audio && typeof this._audio.suspend === "function") {
        try {
          this._audio.suspend().catch(noop);
        } catch (_error) {}
      }
    }

    _tone(kind) {
      if (this.options.muted || !this._audioReady || !this._audio) {
        return;
      }
      try {
        const context = this._audio;
        const oscillator = context.createOscillator();
        const gain = context.createGain();
        const now = context.currentTime || 0;
        const map = {
          play: 392,
          attack: 220,
          combat: 180,
          damage: 180,
          heal: 523,
          flower: 523,
          awaken: 659,
          shape: 494,
          record: 440,
          redeem: 587,
          down: 140,
          revive: 349,
        };
        oscillator.type = kind === "down" ? "sawtooth" : "triangle";
        oscillator.frequency.value = map[kind] || 330;
        gain.gain.setValueAtTime(0.0001, now);
        gain.gain.exponentialRampToValueAtTime(0.04, now + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.14);
        oscillator.connect(gain);
        gain.connect(context.destination);
        oscillator.start(now);
        oscillator.stop(now + 0.16);
      } catch (_error) {}
    }
  }

  const api = { Player: Player };
  if (typeof window !== "undefined") {
    window.NTEDuelPresentation = Object.assign({}, window.NTEDuelPresentation || {}, api);
  }
  if (typeof globalThis !== "undefined") {
    globalThis.NTEDuelPresentation = Object.assign({}, globalThis.NTEDuelPresentation || {}, api);
  }
  if (typeof module === "object" && module && module.exports) {
    module.exports = api;
  }
})();
