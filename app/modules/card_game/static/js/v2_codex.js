const V2 = window.NTE_V2;

if (V2.ensureLogin()) {
  bootstrapCodex();
}

async function bootstrapCodex() {
  const banner = document.getElementById('v2-codex-banner');
  const nav = document.getElementById('v2-codex-nav');
  const body = document.getElementById('v2-codex-body');
  const summary = document.getElementById('v2-codex-summary');
  const detail = document.getElementById('v2-codex-detail');
  let catalog = null;
  let selectedId = 'nanali';

  function characters() {
    return V2.sortedCharacters(catalog && catalog.characters);
  }

  function showDetail(card) {
    detail.innerHTML = (
      '<div class="v2-detail-card">' +
        '<div class="section-title-row"><h2>' + V2.escapeHtml(card.name) + '</h2><button class="icon-btn" id="v2-codex-close" type="button" aria-label="关闭">×</button></div>' +
        '<p class="subtle">' + V2.escapeHtml(V2.cardTypeLabel(card.type, card.terminal)) + ' · 费用 ' + V2.escapeHtml(card.cost) + '</p>' +
        '<p>' + V2.escapeHtml(card.description || '效果尚未提供。') + '</p>' +
      '</div>'
    );
    detail.classList.add('open');
    detail.setAttribute('aria-hidden', 'false');
    document.getElementById('v2-codex-close').addEventListener('click', closeDetail);
  }

  function closeDetail() {
    detail.classList.remove('open');
    detail.setAttribute('aria-hidden', 'true');
    detail.innerHTML = '';
  }

  function render() {
    const list = characters();
    if (!list.some(function (item) { return item.id === selectedId; })) {
      selectedId = (list[0] && list[0].id) || 'nanali';
    }
    nav.innerHTML = list.map(function (character) {
      return '<button class="secondary-btn' + (character.id === selectedId ? ' selected' : '') + '" type="button" data-character-id="' + V2.escapeAttr(character.id) + '">' + V2.escapeHtml(character.name) + '</button>';
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
          '<p class="subtle">攻击 ' + V2.escapeHtml(character.attack) + ' · 生命 ' + V2.escapeHtml(character.max_hp) + '</p>' +
          '<p><strong>基础被动</strong> ' + V2.escapeHtml(character.passive || '尚未提供') + '</p>' +
          '<p><strong>觉醒后</strong> ' + V2.escapeHtml(character.awakened_passive || '尚未提供') + '</p>' +
        '</div>' +
      '</section>' +
      '<section class="v2-codex-cards">' + owned.map(function (card) {
        return (
          '<article class="v2-codex-card" data-card-id="' + V2.escapeAttr(card.id) + '">' +
            '<div class="v2-stat-row">' +
              '<span class="v2-chip">' + V2.escapeHtml(V2.cardTypeLabel(card.type, card.terminal)) + '</span>' +
              '<span class="v2-chip">费用 ' + V2.escapeHtml(card.cost) + '</span>' +
            '</div>' +
            '<h3>' + V2.escapeHtml(card.name) + '</h3>' +
            '<p class="subtle">' + V2.escapeHtml(card.description || '效果尚未提供。') + '</p>' +
          '</article>'
        );
      }).join('') + '</section>'
    );
    summary.textContent = '查看四名角色的核心能力，以及各自八种卡牌的效果与配合。';
  }

  nav.addEventListener('click', function (event) {
    const button = event.target.closest('[data-character-id]');
    if (!button) {
      return;
    }
    selectedId = button.getAttribute('data-character-id');
    render();
  });
  body.addEventListener('click', function (event) {
    const cardNode = event.target.closest('[data-card-id]');
    if (!cardNode) {
      return;
    }
    const card = V2.indexById((catalog && catalog.cards) || [])[cardNode.getAttribute('data-card-id')];
    if (card) {
      showDetail(card);
    }
  });
  detail.addEventListener('click', function (event) {
    if (event.target === detail) {
      closeDetail();
    }
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
