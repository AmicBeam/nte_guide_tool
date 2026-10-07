const V2 = window.NTE_V2;

const glossary = document.getElementById('v2-codex-glossary');
const glossaryHelp = document.getElementById('v2-codex-help');
const glossarySearch = document.getElementById('v2-glossary-search');
const glossaryEntries = Array.from(glossary.querySelectorAll('article'));

glossaryHelp.addEventListener('click', function () {
  glossary.showModal();
  glossarySearch.focus();
});
document.getElementById('v2-glossary-close').addEventListener('click', function () {
  glossary.close();
});
glossary.addEventListener('keydown', function (event) {
  if (event.key === 'Escape') {
    event.preventDefault();
    glossary.close();
  }
});
glossary.addEventListener('click', function (event) {
  const rect = glossary.getBoundingClientRect();
  if (event.target === glossary && (event.clientX < rect.left || event.clientX > rect.right ||
      event.clientY < rect.top || event.clientY > rect.bottom)) {
    glossary.close();
  }
});
glossary.addEventListener('close', function () {
  glossaryHelp.focus();
});
glossarySearch.addEventListener('input', function () {
  const query = glossarySearch.value.trim().toLocaleLowerCase();
  glossaryEntries.forEach(function (entry) {
    entry.hidden = !entry.textContent.toLocaleLowerCase().includes(query);
  });
  glossary.querySelectorAll('[data-glossary-group]').forEach(function (group) {
    group.hidden = !Array.from(group.querySelectorAll('article')).some(function (entry) { return !entry.hidden; });
  });
  document.getElementById('v2-glossary-empty').hidden = glossaryEntries.some(function (entry) { return !entry.hidden; });
});

if (V2.ensureLogin()) {
  bootstrapCodex();
}

async function bootstrapCodex() {
  const banner = document.getElementById('v2-codex-banner');
  const nav = document.getElementById('v2-codex-nav');
  const body = document.getElementById('v2-codex-body');
  let catalog = null;
  let selectedId = 'nanali';

  function characters() {
    return V2.sortedCharacters(catalog && catalog.characters);
  }

  function unpublishedBadge(character) {
    return character.access_level === 'test' ? '<span class="v2-codex-unpublished">未公开</span>' : '';
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
        '<span>' + V2.escapeHtml(character.name) + '</span>' + unpublishedBadge(character) + '</button>';
    }).join('');
    const character = list.find(function (item) { return item.id === selectedId; }) || list[0];
    if (!character) {
      body.innerHTML = '<div class="empty-state">图鉴数据尚未到达。</div>';
      return;
    }
    const portrait = V2.characterAsset(character, 'portrait') || V2.characterAsset(character, 'avatar');
    const owned = V2.cardsForCharacter((catalog && catalog.cards) || [], character.id);
    const displayed = owned.concat(character.mechanisms || []);
    body.innerHTML = (
      '<section class="v2-codex-hero">' +
        '<div class="v2-codex-hero-art">' + V2.imageMarkup(portrait, character.name, 'v2-portrait') + '</div>' +
        '<div>' +
          '<p class="eyebrow">角色</p>' +
          '<div class="v2-codex-title"><h2>' + V2.escapeHtml(character.name) + '</h2>' + unpublishedBadge(character) + '</div>' +
          V2.attributeMarkup(character.attribute) +
          '<p class="v2-character-meta">' + V2.characterStatMarkup(character) + '</p>' +
          '<p>异能：' + V2.cardDescriptionMarkup(character.passive || '尚未提供') + '</p>' +
          '<p>' + V2.escapeHtml(V2.characterUltimateLabel(character)) + '：' + V2.cardDescriptionMarkup(character.awakened_passive || '尚未提供') + '</p>' +
        '</div>' +
      '</section>' +
      '<section class="v2-codex-cards">' + displayed.map(function (card) {
        return (
          '<article class="v2-codex-card' + (card.type === 'mechanism' ? ' is-mechanism' : '') + '"' + (card.tooltip ? ' title="' + V2.escapeAttr(card.tooltip) + '"' : '') + (V2.cardTypeToken(card.type) ? ' data-card-type="' + V2.escapeAttr(V2.cardTypeToken(card.type)) + '"' : '') + '>' +
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
