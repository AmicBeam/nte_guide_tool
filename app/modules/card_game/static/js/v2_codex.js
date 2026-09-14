const V2 = window.NTE_V2;

if (V2.ensureLogin()) {
  bootstrapCodex();
}

async function bootstrapCodex() {
  const banner = document.getElementById('v2-codex-banner');
  const nav = document.getElementById('v2-codex-nav');
  const body = document.getElementById('v2-codex-body');
  const summary = document.getElementById('v2-codex-summary');
  let catalog = null;
  let selectedId = 'nanali';

  function characters() {
    return V2.sortedCharacters(catalog && catalog.characters);
  }

  function render() {
    const list = characters();
    if (!list.some(function (item) { return item.id === selectedId; })) {
      selectedId = (list[0] && list[0].id) || 'nanali';
    }
    nav.innerHTML = list.map(function (character) {
      const avatar = V2.characterAsset(character, 'avatar') || V2.characterAsset(character, 'portrait');
      return '<button class="' + (character.id === selectedId ? ' selected' : '') + '" type="button" data-character-id="' + V2.escapeAttr(character.id) + '">' +
        V2.imageMarkup(avatar, character.name, 'v2-avatar') +
        '<span>' + V2.escapeHtml(character.name) + '</span></button>';
    }).join('');
    const character = list.find(function (item) { return item.id === selectedId; }) || list[0];
    if (!character) {
      body.innerHTML = '<div class="empty-state">图鉴数据尚未到达。</div>';
      return;
    }
    const portrait = V2.characterAsset(character, 'portrait') || V2.characterAsset(character, 'avatar');
    const owned = V2.cardsForCharacter((catalog && catalog.cards) || [], character.id);
    body.innerHTML = (
      '<section class="v2-codex-hero">' +
        '<div class="v2-codex-hero-art">' + V2.imageMarkup(portrait, character.name, 'v2-portrait') + '</div>' +
        '<div>' +
          '<p class="eyebrow">角色</p>' +
          '<h2>' + V2.escapeHtml(character.name) + '</h2>' +
          V2.attributeMarkup(character.attribute) +
          '<p class="v2-character-meta">' + V2.characterStatMarkup(character) + '</p>' +
          '<p>异能：' + V2.cardDescriptionMarkup(character.passive || '尚未提供') + '</p>' +
          '<p>' + V2.escapeHtml(V2.characterUltimateLabel(character)) + '：' + V2.cardDescriptionMarkup(character.awakened_passive || '尚未提供') + '</p>' +
        '</div>' +
      '</section>' +
      '<section class="v2-codex-cards">' + owned.map(function (card) {
        return (
          '<article class="v2-codex-card"' + (V2.cardTypeToken(card.type) ? ' data-card-type="' + V2.escapeAttr(V2.cardTypeToken(card.type)) + '"' : '') + '>' +
            '<div class="v2-stat-row">' +
              V2.cardTypeChip(card.type, card.terminal) +
              V2.cardDerivedChip(card) +
              V2.cardCostBadge(card) +
              V2.cardFrameStat(card) +
            '</div>' +
            '<h3>' + V2.escapeHtml(card.name) + '</h3>' +
            V2.cardDescriptionHtml(card.description, '效果尚未提供。') +
          '</article>'
        );
      }).join('') + '</section>'
    );
    if (summary) summary.textContent = character.name + ' · 八种卡牌';
  }

  nav.addEventListener('click', function (event) {
    const button = event.target.closest('[data-character-id]');
    if (!button) {
      return;
    }
    selectedId = button.getAttribute('data-character-id');
    render();
  });
  try {
    catalog = await V2.getCatalog();
    render();
  } catch (error) {
    V2.showBanner(banner, error && error.message ? error.message : '无法读取图鉴。', 'error');
    catalog = { characters: [], cards: [] };
    render();
  }
}
