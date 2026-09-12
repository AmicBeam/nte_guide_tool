const V2 = window.NTE_V2;

if (V2.ensureLogin()) {
  bootstrapBuild();
}

async function bootstrapBuild() {
  const banner = document.getElementById('v2-build-banner');
  const groupsNode = document.getElementById('v2-build-groups');
  const totalNode = document.getElementById('v2-build-total');
  const sourceNode = document.getElementById('v2-build-source');
  const saveBtn = document.getElementById('v2-save-build-btn');
  const resetBtn = document.getElementById('v2-reset-starter-btn');
  const detail = document.getElementById('v2-build-detail');
  let catalog = null;
  let counts = {};
  let sourceLabel = '尚未读取构筑。';
  let busy = false;

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
      id: starter.id || "starter",
      name: starter.name || "基础预组",
      character_ids: starter.character_ids || V2.CHARACTER_ORDER.slice(),
      card_ids: ids,
    });
  }

  function applyDeck(deck, label) {
    counts = V2.countIds(deck && deck.card_ids);
    sourceLabel = label;
    render();
  }

  function currentBuild() {
    const starter = catalog && catalog.starter_deck || {};
    return {
      id: (catalog && catalog.saved_build && catalog.saved_build.id) || starter.id || 'custom',
      name: (catalog && catalog.saved_build && catalog.saved_build.name) || '自定义构筑',
      character_ids: V2.CHARACTER_ORDER.slice(),
      card_ids: V2.expandCounts(counts, cardOrder()),
    };
  }

  function countFor(cardId) {
    return Number(counts[cardId] || 0);
  }

  function characterCount(characterId) {
    return V2.cardsForCharacter(cards(), characterId).reduce(function (sum, card) {
      return sum + countFor(card.id);
    }, 0);
  }

  function totalCount() {
    return Object.keys(counts).reduce(function (sum, key) {
      return sum + countFor(key);
    }, 0);
  }

  function setCount(cardId, next) {
    const value = Math.max(0, Math.min(2, Number(next) || 0));
    if (value === 0) {
      delete counts[cardId];
    } else {
      counts[cardId] = value;
    }
    sourceLabel = '未保存草稿，不会因为校验失败而丢失。';
    render();
  }

  function showDetail(title, body) {
    detail.innerHTML = (
      '<div class="v2-detail-card">' +
        '<div class="section-title-row"><h2>' + V2.escapeHtml(title) + '</h2><button class="icon-btn" id="v2-detail-close" type="button" aria-label="关闭">×</button></div>' +
        body +
      '</div>'
    );
    detail.classList.add('open');
    detail.setAttribute('aria-hidden', 'false');
    document.getElementById('v2-detail-close').addEventListener('click', closeDetail);
  }

  function closeDetail() {
    detail.classList.remove('open');
    detail.setAttribute('aria-hidden', 'true');
    detail.innerHTML = '';
  }

  function render() {
    const total = totalCount();
    totalNode.textContent = '总牌数 ' + total + ' / 32';
    sourceNode.textContent = sourceLabel;
    groupsNode.innerHTML = characters().map(function (character) {
      const owned = V2.cardsForCharacter(cards(), character.id);
      const used = characterCount(character.id);
      const portrait = V2.characterAsset(character, 'portrait') || V2.characterAsset(character, 'avatar');
      const countClass = used === 8 ? 'ok' : 'bad';
      return (
        '<article class="glass-card v2-build-group" data-character-id="' + V2.escapeAttr(character.id) + '">' +
          '<div class="v2-build-group-head">' +
            '<div class="v2-build-group-art">' + V2.imageMarkup(portrait, character.name, 'v2-portrait') + '</div>' +
            '<div>' +
              '<h2>' + V2.escapeHtml(character.name) + '</h2>' +
              V2.attributeMarkup(character.attribute) +
              '<p class="subtle">攻击 ' + V2.escapeHtml(character.attack) + ' · 生命 ' + V2.escapeHtml(character.max_hp) + '</p>' +
            '</div>' +
            '<strong class="v2-count ' + countClass + '">' + used + ' / 8</strong>' +
          '</div>' +
          '<div class="v2-passive-box">' +
            '<p><strong>基础被动</strong> ' + V2.escapeHtml(character.passive || '尚未提供') + '</p>' +
            '<p><strong>觉醒后</strong> ' + V2.escapeHtml(character.awakened_passive || '尚未提供') + '</p>' +
            '<div class="hero-actions">' +
              '<button class="secondary-btn" type="button" data-view-character="' + V2.escapeAttr(character.id) + '">查看被动/觉醒</button>' +
            '</div>' +
          '</div>' +
          '<div class="v2-build-cards">' + owned.map(function (card) {
            const copies = countFor(card.id);
            return (
              '<article class="v2-build-card" data-card-id="' + V2.escapeAttr(card.id) + '">' +
                '<div>' +
                  '<h3>' + V2.escapeHtml(card.name) + '</h3>' +
                  '<p class="subtle">' + V2.escapeHtml(V2.cardTypeLabel(card.type, card.terminal)) + ' · 费用 ' + V2.escapeHtml(card.cost) + '</p>' +
                '</div>' +
                '<div class="v2-stepper">' +
                  '<button class="secondary-btn" type="button" data-delta="-1" data-card-id="' + V2.escapeAttr(card.id) + '"' + (copies <= 0 ? ' disabled' : '') + '>-</button>' +
                  '<strong>' + copies + '</strong>' +
                  '<button class="secondary-btn" type="button" data-delta="1" data-card-id="' + V2.escapeAttr(card.id) + '"' + (copies >= 2 ? ' disabled' : '') + '>+</button>' +
                  '<button class="secondary-btn" type="button" data-view-card="' + V2.escapeAttr(card.id) + '">查看</button>' +
                '</div>' +
              '</article>'
            );
          }).join('') + '</div>' +
        '</article>'
      );
    }).join('');
  }

  groupsNode.addEventListener('click', function (event) {
    const deltaBtn = event.target.closest('[data-delta]');
    if (deltaBtn) {
      const cardId = deltaBtn.getAttribute('data-card-id');
      setCount(cardId, countFor(cardId) + Number(deltaBtn.getAttribute('data-delta')));
      return;
    }
    const cardBtn = event.target.closest('[data-view-card]');
    if (cardBtn) {
      const card = V2.indexById(cards())[cardBtn.getAttribute('data-view-card')];
      if (!card) {
        return;
      }
      showDetail(card.name, '<p class="subtle">' + V2.escapeHtml(V2.cardTypeLabel(card.type, card.terminal)) + ' · 费用 ' + V2.escapeHtml(card.cost) + '</p><p>' + V2.escapeHtml(card.description || '效果尚未提供。') + '</p>');
      return;
    }
    const charBtn = event.target.closest('[data-view-character]');
    if (charBtn) {
      const character = characters().find(function (item) { return item.id === charBtn.getAttribute('data-view-character'); });
      if (!character) {
        return;
      }
      showDetail(character.name, '<p><strong>基础被动</strong></p><p>' + V2.escapeHtml(character.passive || '尚未提供') + '</p><p><strong>觉醒后</strong></p><p>' + V2.escapeHtml(character.awakened_passive || '尚未提供') + '</p>');
    }
  });
  detail.addEventListener('click', function (event) {
    if (event.target === detail) {
      closeDetail();
    }
  });

  resetBtn.addEventListener('click', function () {
    applyDeck(starterDeck(), '已重置为基础预组草稿，尚未保存。');
  });

  saveBtn.addEventListener('click', async function () {
    if (busy) {
      return;
    }
    busy = true;
    saveBtn.disabled = true;
    V2.showBanner(banner, '', '');
    try {
      const payload = await V2.saveBuild(currentBuild());
      catalog.saved_build = payload.build || currentBuild();
      applyDeck(catalog.saved_build, '已保存当前构筑。');
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
    if (catalog.saved_build && Array.isArray(catalog.saved_build.card_ids) && catalog.saved_build.card_ids.length) {
      applyDeck(catalog.saved_build, '已加载保存的构筑。');
    } else {
      applyDeck(starterDeck(), '无保存构筑，已载入基础预组。');
    }
  } catch (error) {
    V2.showBanner(banner, error && error.message ? error.message : '无法读取卡池。', 'error');
    catalog = { characters: [], cards: [], starter_deck: { character_ids: V2.CHARACTER_ORDER.slice(), card_ids: [] } };
    applyDeck(starterDeck(), '卡池尚未到达，骨架仍可浏览。');
  }
}
