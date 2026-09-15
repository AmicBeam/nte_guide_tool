const V2 = window.NTE_V2;

if (V2.ensureLogin()) {
  bootstrapBuild();
}

async function bootstrapBuild() {
  const banner = document.getElementById('v2-build-banner');
  const groupsNode = document.getElementById('v2-build-groups');
  const teamNode = document.getElementById('v2-build-team');
  const libraryNode = document.getElementById('v2-build-library');
  const rosterNode = document.getElementById('v2-build-roster');
  const totalNode = document.getElementById('v2-build-total');
  const sourceNode = document.getElementById('v2-build-source');
  const nameInput = document.getElementById('v2-build-name');
  const saveBtn = document.getElementById('v2-save-build-btn');
  const newBtn = document.getElementById('v2-build-new-btn');
  const deleteBtn = document.getElementById('v2-build-delete-btn');
  let catalog = null;
  let counts = {};
  let selectedIds = ['nanali', 'zero', 'jiuyuan', 'iloy'];
  let focusId = selectedIds[0];
  let activeBuildId = '';
  let savedBuilds = [];
  let sourceLabel = '尚未读取构筑。';
  let busy = false;
  let pickingSlot = null;
  let dragState = null;
  let suppressOpen = false;

  function cards() {
    return (catalog && catalog.cards) || [];
  }

  function characters() {
    return V2.sortedCharacters(catalog && catalog.characters);
  }

  function cardOrder() {
    return cards().map(function (card) { return card.id; });
  }

  function starterDeck() {
    const starter = (catalog && catalog.starter_deck) || {};
    if (Array.isArray(starter.card_ids) && starter.card_ids.length) {
      return starter;
    }
    const ids = [];
    cards().forEach(function (card) {
      const copies = Math.max(0, Number(card.starter_copies || 0));
      for (let index = 0; index < copies; index += 1) {
        ids.push(card.id);
      }
    });
    return Object.assign({}, starter, {
      id: starter.id || 'starter',
      name: starter.name || '基础预组',
      character_ids: starter.character_ids || V2.CHARACTER_ORDER.slice(),
      card_ids: ids,
    });
  }

  function emptyDraft() {
    return { id: '', name: '新的构筑', character_ids: [], card_ids: [] };
  }

  function applyDeck(deck, label) {
    counts = V2.countIds(deck && deck.card_ids);
    if (deck && Array.isArray(deck.character_ids)) {
      selectedIds = deck.character_ids.slice();
    }
    if (selectedIds.indexOf(focusId) < 0) {
      focusId = selectedIds[0] || '';
    }
    if (nameInput && deck && deck.name) {
      nameInput.value = deck.name;
    }
    pickingSlot = null;
    sourceLabel = label;
    render();
  }

  function applyStore(payload) {
    if (!payload) return;
    if (Array.isArray(payload.saved_builds)) savedBuilds = payload.saved_builds;
    if (Object.prototype.hasOwnProperty.call(payload, 'active_build_id')) {
      activeBuildId = payload.active_build_id || '';
    }
    if (payload.saved_build) catalog.saved_build = payload.saved_build;
    else if (Object.prototype.hasOwnProperty.call(payload, 'saved_build')) catalog.saved_build = null;
    if (payload.build) catalog.saved_build = payload.build;
  }

  function currentBuild() {
    return {
      id: activeBuildId || '',
      name: (nameInput && nameInput.value.trim()) || '自定义构筑',
      character_ids: selectedIds.slice(),
      card_ids: V2.expandCounts(counts, cardOrder()),
    };
  }

  function selectedCharacters() {
    const byId = V2.indexById(characters());
    return selectedIds.map(function (id) {
      return byId[id] || V2.characterFallback(id);
    }).filter(Boolean);
  }

  function countFor(cardId) {
    return Number(counts[cardId] || 0);
  }

  function characterCount(characterId) {
    return V2.cardsForCharacter(cards(), characterId, { buildableOnly: true }).reduce(function (sum, card) {
      return sum + countFor(card.id);
    }, 0);
  }

  function totalCount() {
    return Object.keys(counts).reduce(function (sum, key) {
      return sum + countFor(key);
    }, 0);
  }

  function setCount(cardId, next) {
    const card = V2.indexById(cards())[cardId];
    const current = countFor(cardId);
    let value = Math.max(0, Math.min(2, Number(next) || 0));
    if (card && value > current) {
      const room = 8 - characterCount(card.character_id);
      value = current + Math.min(value - current, room);
    }
    if (value === 0) {
      delete counts[cardId];
    } else {
      counts[cardId] = value;
    }
    sourceLabel = '未保存草稿，不会因为校验失败而丢失。';
    render();
  }

  function render() {
    const total = totalCount();
    totalNode.textContent = '总牌数 ' + total + ' / 32';
    sourceNode.textContent = sourceLabel;
    if (deleteBtn) deleteBtn.disabled = !activeBuildId || savedBuilds.length < 1;
    if (libraryNode) {
      libraryNode.innerHTML = '<p class="subtle">我的构筑 ' + savedBuilds.length + '/8' +
        (savedBuilds.length > 1 ? ' · 可拖动排序' : '') + '</p><div class="v2-deck-row">' +
        (savedBuilds.length
          ? savedBuilds.map(function (deck) {
              const on = deck.id === activeBuildId;
              return '<button class="v2-deck-chip' + (on ? ' selected' : '') + '" type="button" data-open-build="' +
                V2.escapeAttr(deck.id) + '"' + (savedBuilds.length > 1 ? ' title="拖动排序"' : '') + '>' +
                V2.escapeHtml(deck.name || deck.id) + '</button>';
            }).join('')
          : '<span class="subtle">还没有构筑</span>') +
        '</div>';
    }
    const team = selectedCharacters();
    if (!team.some(function (item) { return item.id === focusId; })) {
      focusId = team[0] && team[0].id;
    }
    const character = team.find(function (item) { return item.id === focusId; }) || team[0];
    if (teamNode) {
      const slots = [0, 1, 2, 3].map(function (index) {
        const item = team[index];
        if (!item) {
          return '<button class="v2-team-slot is-empty" type="button" data-pick-slot="' + index + '">选择角色</button>';
        }
        const used = characterCount(item.id);
        const art = V2.characterAsset(item, 'avatar') || V2.characterAsset(item, 'portrait');
        return '<button class="v2-team-slot' + (item.id === focusId ? ' selected' : '') + '" type="button" data-focus-character="' + V2.escapeAttr(item.id) + '">' +
          '<div class="v2-team-art">' + V2.imageMarkup(art, item.name, 'v2-avatar') + '</div>' +
          '<div class="v2-team-meta">' +
            '<strong>' + V2.escapeHtml(item.name) + '</strong>' +
            '<span class="v2-count ' + (used === 8 ? 'ok' : 'bad') + '">' + used + '/8</span>' +
            '<span class="secondary-btn v2-team-change" data-pick-slot="' + index + '">更换</span>' +
          '</div></button>';
      });
      teamNode.innerHTML = slots.join('');
    }
    if (!character) {
      groupsNode.innerHTML = '<div class="empty-state">从下方四个位置选择出战角色。</div>';
    } else {
      const owned = V2.cardsForCharacter(cards(), character.id, { buildableOnly: true });
      const used = characterCount(character.id);
      const portrait = V2.characterAsset(character, 'portrait') || V2.characterAsset(character, 'avatar');
      groupsNode.innerHTML =
        '<section class="v2-build-focus">' +
          '<div class="v2-build-focus-art">' + V2.imageMarkup(portrait, character.name, 'v2-portrait') + '</div>' +
          '<div class="v2-build-focus-copy">' +
            '<h2>' + V2.escapeHtml(character.name) + '</h2>' +
            V2.attributeMarkup(character.attribute) +
            '<p class="v2-character-meta">' + V2.characterStatMarkup(character) +
            '<span class="v2-count ' + (used === 8 ? 'ok' : 'bad') + '">' + used + ' / 8</span></p>' +
            '<div class="v2-passive-box">' +
              '<p>异能：' + V2.cardDescriptionMarkup(character.passive || '尚未提供') + '</p>' +
              '<p>' + V2.escapeHtml(V2.characterUltimateLabel(character)) + '：' + V2.cardDescriptionMarkup(character.awakened_passive || '尚未提供') + '</p>' +
            '</div>' +
          '</div>' +
        '</section>' +
        '<div class="v2-build-cards">' + owned.map(function (card) {
          const copies = countFor(card.id);
          return (
            '<article class="v2-build-card' + (copies ? '' : ' is-empty') + '" data-card-id="' + V2.escapeAttr(card.id) + '"' +
              (V2.cardTypeToken(card.type) ? ' data-card-type="' + V2.escapeAttr(V2.cardTypeToken(card.type)) + '"' : '') + '>' +
              V2.cardCostBadge(card) +
              '<span class="v2-build-copies">×' + copies + '</span>' +
              '<div class="v2-build-card-body">' +
                '<h3>' + V2.escapeHtml(card.name) + '</h3>' +
                '<div class="v2-build-card-meta">' + V2.cardTypeChip(card.type, card.terminal) + '</div>' +
                V2.cardDescriptionHtml(card.description) +
              '</div>' +
              '<div class="v2-build-card-foot">' +
                V2.cardFrameStat(card) +
                '<div class="v2-stepper">' +
                  '<button class="secondary-btn" type="button" data-delta="-1" data-card-id="' + V2.escapeAttr(card.id) + '"' + (copies <= 0 ? ' disabled' : '') + '>-</button>' +
                  '<strong>' + copies + '</strong>' +
                  '<button class="secondary-btn" type="button" data-delta="1" data-card-id="' + V2.escapeAttr(card.id) + '"' + (copies >= 2 || used >= 8 ? ' disabled' : '') + '>+</button>' +
                '</div>' +
              '</div>' +
            '</article>'
          );
        }).join('') + '</div>';
    }
    if (rosterNode) {
      const picking = pickingSlot != null;
      V2.setHidden(rosterNode, !picking);
      rosterNode.innerHTML = picking
        ? '<div class="v2-gallery-head"><strong>选择出战角色</strong><button class="secondary-btn" type="button" data-close-gallery="1">取消</button></div>' +
          '<div class="v2-roster-row">' + characters().map(function (item) {
            const on = selectedIds.indexOf(item.id) >= 0;
            const art = V2.characterAsset(item, 'portrait') || V2.characterAsset(item, 'avatar');
            return '<button class="v2-roster-card' + (on ? ' selected' : '') + '" type="button" data-pick-character="' + V2.escapeAttr(item.id) + '">' +
              V2.imageMarkup(art, item.name, 'v2-portrait') +
              '<strong>' + V2.escapeHtml(item.name) + '</strong></button>';
          }).join('') + '</div>'
        : '';
    }
  }

  function fillStarterCopies(characterId) {
    V2.cardsForCharacter(cards(), characterId, { buildableOnly: true }).forEach(function (card) {
      const copies = Math.max(0, Number(card.starter_copies || 0));
      if (copies) counts[card.id] = copies;
      else delete counts[card.id];
    });
  }

  function clearCharacterCards(characterId) {
    V2.cardsForCharacter(cards(), characterId, { buildableOnly: true }).forEach(function (card) {
      delete counts[card.id];
    });
  }

  function pickCharacter(characterId) {
    const already = selectedIds.indexOf(characterId);
    if (already >= 0) {
      focusId = characterId;
      pickingSlot = null;
      render();
      return;
    }
    const slot = pickingSlot == null ? selectedIds.length : pickingSlot;
    if (slot < selectedIds.length) {
      clearCharacterCards(selectedIds[slot]);
      selectedIds[slot] = characterId;
    } else if (selectedIds.length < 4) {
      selectedIds.push(characterId);
    } else {
      return;
    }
    fillStarterCopies(characterId);
    focusId = characterId;
    pickingSlot = null;
    sourceLabel = '未保存草稿，不会因为校验失败而丢失。';
    render();
  }

  async function openSaved(buildId) {
    if (busy) return;
    busy = true;
    try {
      const payload = await V2.selectBuild(buildId);
      applyStore(payload);
      const deck = (payload.saved_builds || []).find(function (item) { return item.id === buildId; }) || payload.saved_build;
      if (deck) applyDeck(deck, '已打开「' + (deck.name || '构筑') + '」。');
    } catch (error) {
      V2.showBanner(banner, error && error.message ? error.message : '无法打开构筑。', 'error');
    } finally {
      busy = false;
    }
  }

  function orderedIds() {
    return savedBuilds.map(function (deck) { return deck.id; });
  }

  function insertionIdAt(x, y) {
    const chips = Array.from(libraryNode.querySelectorAll('[data-open-build]'));
    const draggingId = dragState && dragState.id;
    for (let index = 0; index < chips.length; index += 1) {
      const chip = chips[index];
      const chipId = chip.getAttribute('data-open-build');
      if (chipId === draggingId) continue;
      const box = chip.getBoundingClientRect();
      if (y < box.top - 6) return chipId;
      if (y <= box.bottom + 6 && x < box.left + box.width / 2) return chipId;
    }
    return '';
  }

  function markDropTarget(beforeId) {
    const chips = Array.from(libraryNode.querySelectorAll('[data-open-build]'));
    const draggingId = dragState && dragState.id;
    chips.forEach(function (chip) {
      const chipId = chip.getAttribute('data-open-build');
      chip.classList.toggle('is-drop-before', Boolean(beforeId) && chipId === beforeId);
      chip.classList.remove('is-drop-after');
    });
    if (!beforeId) {
      const last = chips.filter(function (chip) {
        return chip.getAttribute('data-open-build') !== draggingId;
      }).pop();
      if (last) last.classList.add('is-drop-after');
    }
  }

  function moveIds(fromId, beforeId) {
    const ids = orderedIds();
    const from = ids.indexOf(fromId);
    if (from < 0) return ids;
    ids.splice(from, 1);
    if (beforeId) {
      const to = ids.indexOf(beforeId);
      ids.splice(to < 0 ? ids.length : to, 0, fromId);
    } else {
      ids.push(fromId);
    }
    return ids;
  }

  async function persistOrder(ids) {
    if (ids.join('\0') === orderedIds().join('\0')) return;
    const previous = savedBuilds.slice();
    busy = true;
    savedBuilds = ids.map(function (id) {
      return previous.find(function (deck) { return deck.id === id; });
    }).filter(Boolean);
    render();
    try {
      const payload = await V2.reorderBuilds(ids);
      applyStore(payload);
      render();
    } catch (error) {
      savedBuilds = previous;
      render();
      V2.showBanner(banner, error && error.message ? error.message : '无法调整构筑顺序。', 'error');
    } finally {
      busy = false;
    }
  }

  function endDrag(event) {
    if (!dragState) return;
    const moved = dragState.moved;
    const fromId = dragState.id;
    const beforeId = moved ? insertionIdAt(event.clientX, event.clientY) : null;
    const chip = dragState.chip;
    dragState = null;
    if (chip && chip.releasePointerCapture && event.pointerId != null) {
      try { chip.releasePointerCapture(event.pointerId); } catch (error) { /* already released */ }
    }
    Array.from(libraryNode.querySelectorAll('[data-open-build]')).forEach(function (item) {
      item.classList.remove('is-dragging', 'is-drop-before', 'is-drop-after');
    });
    if (moved) {
      suppressOpen = true;
      persistOrder(moveIds(fromId, beforeId));
    }
  }

  if (libraryNode) {
    libraryNode.addEventListener('click', function (event) {
      if (suppressOpen) {
        event.preventDefault();
        event.stopPropagation();
        suppressOpen = false;
        return;
      }
      const btn = event.target.closest('[data-open-build]');
      if (btn) openSaved(btn.getAttribute('data-open-build'));
    });
    libraryNode.addEventListener('pointerdown', function (event) {
      if (busy || event.button) return;
      const chip = event.target.closest('[data-open-build]');
      if (!chip || savedBuilds.length < 2) return;
      dragState = {
        id: chip.getAttribute('data-open-build'),
        chip: chip,
        x: event.clientX,
        y: event.clientY,
        moved: false,
      };
      if (chip.setPointerCapture) chip.setPointerCapture(event.pointerId);
    });
    libraryNode.addEventListener('pointermove', function (event) {
      if (!dragState) return;
      const dx = event.clientX - dragState.x;
      const dy = event.clientY - dragState.y;
      if (!dragState.moved && (dx * dx + dy * dy) < 64) return;
      dragState.moved = true;
      if (dragState.chip) dragState.chip.classList.add('is-dragging');
      markDropTarget(insertionIdAt(event.clientX, event.clientY));
    });
    libraryNode.addEventListener('pointerup', endDrag);
    libraryNode.addEventListener('pointercancel', endDrag);
  }
  if (rosterNode) {
    rosterNode.addEventListener('click', function (event) {
      if (event.target.closest('[data-close-gallery]')) {
        pickingSlot = null;
        render();
        return;
      }
      const btn = event.target.closest('[data-pick-character]');
      if (btn) pickCharacter(btn.getAttribute('data-pick-character'));
    });
  }

  if (teamNode) {
    teamNode.addEventListener('click', function (event) {
      const pick = event.target.closest('[data-pick-slot]');
      if (pick) {
        event.preventDefault();
        event.stopPropagation();
        pickingSlot = Number(pick.getAttribute('data-pick-slot'));
        render();
        return;
      }
      const slot = event.target.closest('[data-focus-character]');
      if (slot) {
        focusId = slot.getAttribute('data-focus-character');
        pickingSlot = null;
        render();
      }
    });
  }

  groupsNode.addEventListener('click', function (event) {
    const deltaBtn = event.target.closest('[data-delta]');
    if (deltaBtn) {
      event.stopPropagation();
      const cardId = deltaBtn.getAttribute('data-card-id');
      setCount(cardId, countFor(cardId) + Number(deltaBtn.getAttribute('data-delta')));
      return;
    }
    const face = event.target.closest('[data-card-id]');
    if (face) {
      const cardId = face.getAttribute('data-card-id');
      setCount(cardId, countFor(cardId) + 1);
    }
  });

  if (newBtn) {
    newBtn.addEventListener('click', function () {
      if (busy) return;
      activeBuildId = '';
      focusId = '';
      if (nameInput) nameInput.value = '新的构筑';
      applyDeck(emptyDraft(), '已创建「新的构筑」，请选择四名出战角色。');
    });
  }

  if (deleteBtn) {
    deleteBtn.addEventListener('click', async function () {
      if (busy || !activeBuildId) return;
      busy = true;
      deleteBtn.disabled = true;
      try {
        const payload = await V2.deleteBuild(activeBuildId);
        applyStore(payload);
        const next = payload.saved_build;
        activeBuildId = payload.active_build_id || '';
        if (next) applyDeck(next, '已删除，已切换到「' + (next.name || '构筑') + '」。');
        else applyDeck(emptyDraft(), '已删除全部构筑，可新建一套。');
        V2.showBanner(banner, '构筑已删除。', 'ok');
      } catch (error) {
        V2.showBanner(banner, error && error.message ? error.message : '删除失败。', 'error');
      } finally {
        busy = false;
        render();
      }
    });
  }

  const transfer = document.getElementById('v2-build-transfer');
  const transferText = document.getElementById('v2-build-text');
  const transferError = document.getElementById('v2-build-transfer-error');
  const importApply = document.getElementById('v2-build-import-apply');
  function openTransfer(exporting, text) {
    document.getElementById('v2-build-transfer-title').textContent = exporting ? '导出构筑' : '导入构筑';
    transferText.value = text || '';
    transferText.readOnly = exporting;
    importApply.hidden = exporting;
    document.getElementById('v2-build-copy').hidden = !exporting;
    document.getElementById('v2-build-download').hidden = !exporting;
    transferError.textContent = '';
    transfer.showModal();
    transferText.focus();
  }
  document.getElementById('v2-build-import-btn').addEventListener('click', function () {
    if (!busy && catalog) openTransfer(false);
  });
  document.getElementById('v2-build-export-btn').addEventListener('click', async function () {
    if (busy || !catalog) return;
    busy = true;
    try {
      const payload = await V2.request('/build-export', { method: 'POST', body: currentBuild() });
      openTransfer(true, payload.text);
    } catch (error) {
      V2.showBanner(banner, error.message || '导出失败。', 'error');
    } finally { busy = false; }
  });
  importApply.addEventListener('click', async function () {
    if (busy) return;
    busy = true;
    importApply.disabled = true;
    transferError.textContent = '';
    try {
      const payload = await V2.request('/build-import', { method: 'POST', body: { text: transferText.value } });
      activeBuildId = '';
      applyDeck(payload.build, '已导入新草稿，点击保存后加入我的构筑。');
      transfer.close();
    } catch (error) {
      transferError.textContent = error.message || '导入失败，原草稿仍保留。';
    } finally { busy = false; importApply.disabled = false; }
  });
  document.getElementById('v2-build-transfer-close').addEventListener('click', function () {
    if (!busy) transfer.close();
  });
  transfer.addEventListener('cancel', function (event) { if (busy) event.preventDefault(); });
  document.getElementById('v2-build-copy').addEventListener('click', async function () {
    try {
      await navigator.clipboard.writeText(transferText.value);
      transferError.textContent = '已复制。';
    } catch (error) {
      transferText.focus();
      transferText.select();
      transferError.textContent = '已选中文本，请长按或按 Ctrl/Cmd+C 复制。';
    }
  });
  document.getElementById('v2-build-download').addEventListener('click', function () {
    const url = URL.createObjectURL(new Blob([transferText.value], { type: 'text/plain;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = '异能对决构筑.txt';
    link.click();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  });

  saveBtn.addEventListener('click', async function () {
    if (busy) return;
    busy = true;
    saveBtn.disabled = true;
    V2.showBanner(banner, '', '');
    try {
      const payload = await V2.saveBuild(currentBuild());
      applyStore(payload);
      activeBuildId = (payload.build && payload.build.id) || activeBuildId;
      applyDeck(payload.build || currentBuild(), '已保存「' + ((payload.build && payload.build.name) || '构筑') + '」。');
      V2.showBanner(banner, '构筑已保存。', 'ok');
    } catch (error) {
      V2.showBanner(banner, error && error.message ? error.message : '保存失败，草稿仍保留。', 'error');
      sourceLabel = '保存失败，当前草稿仍保留在本页。';
      sourceNode.textContent = sourceLabel;
    } finally {
      busy = false;
      saveBtn.disabled = false;
    }
  });

  try {
    catalog = await V2.getCatalog();
    applyStore(catalog);
    if (catalog.saved_build && Array.isArray(catalog.saved_build.card_ids) && catalog.saved_build.card_ids.length) {
      activeBuildId = catalog.active_build_id || catalog.saved_build.id || '';
      applyDeck(catalog.saved_build, '已加载「' + (catalog.saved_build.name || '保存的构筑') + '」。');
    } else {
      applyDeck(emptyDraft(), '还没有构筑，可新建或等待赠送的预组。');
    }
  } catch (error) {
    V2.showBanner(banner, error && error.message ? error.message : '无法读取卡池。', 'error');
    catalog = { characters: [], cards: [], starter_deck: { character_ids: V2.CHARACTER_ORDER.slice(), card_ids: [] } };
    applyDeck(starterDeck(), '卡池尚未到达，骨架仍可浏览。');
  }
}
