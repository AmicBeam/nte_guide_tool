(() => {
  const CHARACTER_NAMES = {
    nanali: "娜娜莉",
    zero: "主角「零」",
    jiuyuan: "九原",
    xun: "浔",
  };
  const TYPE_LABELS = {
    battle: "战斗",
    tactic: "战术",
    form: "形态",
    shape: "形态",
  };
  const NS = "http://www.w3.org/2000/svg";
  const MAX_EVENT_MS = 2400;

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
    const base = TYPE_LABELS[card.type] || String(card.type || "");
    return card.terminal ? base + "·终结" : base;
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
      this._clock = options.clock || null;
      this.options = {
        muted: options.muted == null ? true : !!options.muted,
        reducedMotion: !!options.reducedMotion,
        speed: options.speed === 2 ? 2 : 1,
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
        for (let index = 0; index < batches.length; index += 1) {
          if (this._cancelled || token !== this._generation) {
            break;
          }
          const batch = batches[index].filter((event) => this._take(event));
          if (!batch.length) {
            continue;
          }
          await this._playBatch(batch, instant);
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
        this._patch(others[index]);
      }
      if (!packets.length) {
        return;
      }
      await Promise.all(packets.map((event) => this._present(event, instant)));
      packets.forEach((event) => this._patch(event));
      this._clearFx(false);
    }

    async _present(event, instant) {
      const kind = event && event.type ? String(event.type) : "instant";
      try {
        this._note(event, kind);
        this._tone(kind);
        if (kind === "play") {
          await this._presentPlay(event, instant);
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
        if (kind === "move" || kind === "enter") {
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
        if (kind === "awaken") {
          await this._presentStatus(event, instant, "awaken", "觉醒");
          return;
        }
        if (kind === "shape") {
          await this._presentStatus(event, instant, "shape", event.shape || (event.card && event.card.name) || "形态");
          return;
        }
        if (kind === "growth") {
          await this._presentFloater(event, instant, "growth", "成长");
          return;
        }
        if (kind === "resource") {
          await this._presentFloater(event, instant, "resource", "资源");
          return;
        }
        if (kind === "draw") {
          await this._presentDraw(event, instant);
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
        if (kind === "gain" || kind === "remove" || kind === "discard" || kind === "turn" || kind === "finish") {
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
      const cost = this._el("span", "v2-fx-card-cost");
      cost.textContent = card && card.cost != null ? String(card.cost) : "";
      const name = this._el("strong", "v2-fx-card-name");
      name.textContent = hasPublicCard(card) ? String(card.name || card.card_id || "") : "";
      const kind = this._el("span", "v2-fx-card-type");
      kind.textContent = typeLabel(card);
      const owner = this._el("span", "v2-fx-card-owner");
      const userName = displayName(event && (event.actor || event.source), card);
      owner.textContent = userName;
      front.appendChild(cost);
      front.appendChild(name);
      front.appendChild(kind);
      front.appendChild(owner);
      if (card && card.description) {
        const desc = this._el("p", "v2-fx-card-desc");
        desc.textContent = String(card.description);
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
      const costText = card && card.cost != null ? String(card.cost) : "-";
      meta.textContent = "使用者 " + (userName || "未知") + " · 费用 " + costText;
      node.appendChild(meta);
      overlay.appendChild(node);
      if (sourceCard && sourceCard.classList) {
        sourceCard.classList.add("v2-fx-source");
      }
      const startX = fromRect.left - stageRect.left;
      const startY = fromRect.top - stageRect.top;
      const endX = stageRect.width / 2 - 84;
      const endY = stageRect.height / 2 - 118;
      node.style.left = startX + "px";
      node.style.top = startY + "px";
      if (hasPublicCard(card)) {
        try {
          this.onPublicCard(card, event);
        } catch (_error) {}
      }
      const fly = this._duration(instant, 420);
      await this._animate(node, [
        { transform: "translate(0px, 0px) scale(0.42)" },
        { transform: "translate(" + (endX - startX) + "px, " + (endY - startY) + "px) scale(1)" },
      ], fly);
      node.style.left = endX + "px";
      node.style.top = endY + "px";
      const flip = this._duration(instant, 280);
      await this._animate(shell, [
        { transform: "rotateY(0deg)" },
        { transform: "rotateY(180deg)" },
      ], flip);
      shell.classList.add("is-flipped");
      if (event && event.target) {
        this._arrow(centerOf(readRect(node) || { cx: stageRect.cx, cy: stageRect.cy }), this._point(event.target, stageRect), "cast");
        await this._wait(this._duration(instant, 220));
      } else {
        await this._wait(this._duration(instant, 260));
      }
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
      if (actor) {
        actor.classList.add("v2-fx-lunge");
        await this._animate(actor, [
          { transform: "translate(0px, 0px)" },
          { transform: "translate(" + Math.max(-24, Math.min(24, (to.x - from.x) * 0.08)) + "px, " + Math.max(-16, Math.min(16, (to.y - from.y) * 0.08)) + "px)" },
          { transform: "translate(0px, 0px)" },
        ], this._duration(instant, 280));
        actor.classList.remove("v2-fx-lunge");
      }
      const counterValue = numericValue(event && event.counter);
      if (counterValue != null && counterValue > 0 && target) {
        this._arrow(to, from, "counter");
      }
      await this._wait(this._duration(instant, 160));
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
      await this._wait(this._duration(instant, 260));
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
      this._floater(to, stageRect, value != null ? "-" + String(value) : "创生花", "harmony");
      await this._wait(this._duration(instant, 280));
    }

    async _presentMove(event, instant) {
      const actor = this._entity(event && (event.actor || event.source || event.target));
      if (actor) {
        actor.classList.add("v2-fx-lunge");
        await this._animate(actor, [
          { transform: "translateY(0px)", opacity: 1 },
          { transform: "translateY(-10px)", opacity: 0.7 },
          { transform: "translateY(0px)", opacity: 1 },
        ], this._duration(instant, 240));
        actor.classList.remove("v2-fx-lunge");
      } else {
        await this._presentBanner(event, instant);
      }
    }

    async _presentStatus(event, instant, klass, label) {
      const node = this._entity(event && (event.target || event.actor || event.source));
      if (node) {
        node.classList.add("v2-fx-" + klass);
      }
      this._banner(label);
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
      await this._wait(this._duration(instant, 300));
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
      await this._wait(this._duration(instant, 240));
      if (node) {
        node.classList.remove("v2-fx-" + klass);
      }
    }

    async _presentDraw(event, instant) {
      const overlay = this._ensureOverlay();
      const card = event && event.card;
      const publicCard = hasPublicCard(card);
      const node = this._el("div", "v2-fx-play-card");
      node.setAttribute("data-v2-fx-kind", publicCard ? "draw-public" : "draw-hidden");
      const shell = this._el("div", "v2-fx-card-3d");
      shell.appendChild(this._el("div", "v2-fx-card-face v2-fx-card-back"));
      if (publicCard) {
        const front = this._el("div", "v2-fx-card-face v2-fx-card-front");
        const name = this._el("strong", "v2-fx-card-name");
        name.textContent = String(card.name || "");
        front.appendChild(name);
        shell.appendChild(front);
        shell.classList.add("is-flipped");
        try {
          this.onPublicCard(card, event);
        } catch (_error) {}
      }
      node.appendChild(shell);
      overlay.appendChild(node);
      const stageRect = readRect(overlay) || { width: 800, height: 450, left: 0, top: 0 };
      const hand = this._hand(sideOf(event));
      const from = readRect(hand) || { left: stageRect.left + 40, top: stageRect.top + 40 };
      node.style.left = (from.left - stageRect.left) + "px";
      node.style.top = (from.top - stageRect.top) + "px";
      await this._animate(node, [
        { transform: "translateY(-24px) scale(0.6)", opacity: 0.4 },
        { transform: "translateY(0px) scale(0.8)", opacity: 1 },
      ], this._duration(instant, 220));
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
      await this._wait(this._duration(instant, 280));
    }

    async _presentBanner(event, instant) {
      const label = event && event.type ? String(event.type) : "";
      this._banner(label);
      await this._wait(this._duration(instant, 140));
    }

    _note(event, kind) {
      const overlay = this._ensureOverlay();
      overlay.setAttribute("data-v2-fx-last-type", kind);
      const card = event && event.card;
      if (hasPublicCard(card)) {
        overlay.setAttribute("data-v2-fx-last-card", String(card.name || card.card_id || ""));
        overlay.setAttribute("data-v2-fx-last-cost", card.cost == null ? "" : String(card.cost));
        overlay.setAttribute("data-v2-fx-last-desc", card.description ? String(card.description) : "");
      }
      const actor = displayName(event && (event.actor || event.source), card);
      if (actor) {
        overlay.setAttribute("data-v2-fx-last-actor", actor);
      }
      const seen = overlay.getAttribute("data-v2-fx-types") || "";
      overlay.setAttribute("data-v2-fx-types", seen ? seen + "," + kind : kind);
    }

    _banner(text) {
      if (!text) {
        return;
      }
      const overlay = this._ensureOverlay();
      const node = this._el("div", "v2-fx-banner");
      node.textContent = String(text);
      overlay.appendChild(node);
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
      const duration = this._duration(this._instant, 420);
      if (duration > 0) {
        this._animate(node, [
          { transform: "translate(-50%, -50%) scale(0.8)", opacity: 0.2 },
          { transform: "translate(-50%, -120%) scale(1)", opacity: 1 },
          { transform: "translate(-50%, -160%) scale(1)", opacity: 0 },
        ], duration).catch(noop);
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
        }
      }
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

    _hand(side) {
      if (!this.root || typeof this.root.querySelector !== "function" || !side) {
        return null;
      }
      try {
        return this.root.querySelector('[data-hand-side="' + cssEscape(side) + '"]');
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
