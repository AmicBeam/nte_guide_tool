const kongmuState = {
  catalog: null,
  selectedCharacterId: '',
  selectedCartridgeId: '',
  plan: null,
  filterQueues: {
    2: [],
    3: [],
    4: [],
  },
};

const kongmuEls = {
  characterGrid: document.getElementById('kongmu-character-grid'),
  cartridgeGrid: document.getElementById('kongmu-cartridge-grid'),
  characterSearch: document.getElementById('kongmu-character-search'),
  cartridgeSearch: document.getElementById('kongmu-cartridge-search'),
  planBtn: document.getElementById('kongmu-plan-btn'),
  resultShell: document.getElementById('kongmu-result-shell'),
  resultTitle: document.getElementById('kongmu-result-title'),
  passiveLine: document.getElementById('kongmu-passive-line'),
  visibleCount: document.getElementById('kongmu-visible-count'),
  totalCount: document.getElementById('kongmu-total-count'),
  filterPanel: document.getElementById('kongmu-filter-panel'),
  emptyResult: document.getElementById('kongmu-empty-result'),
  solutionGrid: document.getElementById('kongmu-solution-grid'),
};

bootstrapKongmu();

async function bootstrapKongmu() {
  try {
    kongmuEls.planBtn.textContent = '读取中';
    kongmuState.catalog = await kongmuRequest('/api/kongmu/catalog');
    const params = new URLSearchParams(window.location.search);
    const characters = kongmuState.catalog.characters || [];
    const cartridges = kongmuState.catalog.cartridges || [];
    kongmuState.selectedCharacterId = pickExistingId(characters, params.get('character')) || characters[0]?.id || '';
    kongmuState.selectedCartridgeId = pickExistingId(cartridges, params.get('cartridge')) || cartridges[0]?.id || '';
    renderKongmuPickers();
    bindKongmuEvents();
    updatePlanButton();
  } catch (error) {
    kongmuEls.planBtn.textContent = '读取失败';
    kongmuEls.characterGrid.innerHTML = `<div class="empty-state">${escapeHtml(error.message || error)}</div>`;
  }
}

function bindKongmuEvents() {
  kongmuEls.characterGrid.addEventListener('click', (event) => {
    const card = event.target.closest('[data-character-id]');
    if (!card) return;
    kongmuState.selectedCharacterId = card.dataset.characterId;
    renderCharacterCards();
    updatePlanButton();
  });

  kongmuEls.cartridgeGrid.addEventListener('click', (event) => {
    const card = event.target.closest('[data-cartridge-id]');
    if (!card) return;
    kongmuState.selectedCartridgeId = card.dataset.cartridgeId;
    renderCartridgeCards();
    updatePlanButton();
  });

  kongmuEls.characterSearch.addEventListener('input', renderCharacterCards);
  kongmuEls.cartridgeSearch.addEventListener('input', renderCartridgeCards);
  kongmuEls.planBtn.addEventListener('click', runKongmuPlan);
}

function pickExistingId(records, id) {
  const value = String(id || '');
  const record = records.find((item) => {
    const ids = [item.id, ...(item.merged_ids || [])].map((itemId) => String(itemId));
    return ids.includes(value);
  });
  return record ? String(record.id) : '';
}

function renderKongmuPickers() {
  renderCharacterCards();
  renderCartridgeCards();
}

function renderCharacterCards() {
  const query = normalizeKongmuQuery(kongmuEls.characterSearch.value);
  const characters = kongmuState.catalog?.characters || [];
  kongmuEls.characterGrid.innerHTML = characters.map((character) => {
    const haystack = normalizeKongmuQuery([
      character.name,
      character.id,
      ...(character.merged_ids || []),
      character.element,
      ...(Object.values(character.names || {})),
    ].join(' '));
    const hidden = query && !haystack.includes(query);
    const selected = String(character.id) === String(kongmuState.selectedCharacterId);
    return `
      <button class="kongmu-select-card${selected ? ' selected' : ''}" type="button" data-character-id="${escapeAttr(character.id)}"${hidden ? ' hidden' : ''}>
        <span class="kongmu-elem-badge" title="${escapeAttr(character.element || '')}">
          <img src="${escapeAttr(character.element_icon || '')}" alt="${escapeAttr(character.element || '')}" loading="lazy">
        </span>
        ${imageHtml(character.avatar, character.name, 'kongmu-avatar', character.source_icon)}
        <span class="kongmu-name">${escapeHtml(character.name)}</span>
      </button>
    `;
  }).join('');
  bindImageFallbacks(kongmuEls.characterGrid);
}

function renderCartridgeCards() {
  const query = normalizeKongmuQuery(kongmuEls.cartridgeSearch.value);
  const cartridges = kongmuState.catalog?.cartridges || [];
  kongmuEls.cartridgeGrid.innerHTML = cartridges.map((cartridge) => {
    const haystack = normalizeKongmuQuery([
      cartridge.name,
      cartridge.raw_name,
      cartridge.id,
      ...(cartridge.aliases || []),
    ].join(' '));
    const hidden = query && !haystack.includes(query);
    const selected = String(cartridge.id) === String(kongmuState.selectedCartridgeId);
    return `
      <button class="kongmu-select-card${selected ? ' selected' : ''}" type="button" data-cartridge-id="${escapeAttr(cartridge.id)}"${hidden ? ' hidden' : ''}>
        ${imageHtml(cartridge.icon, cartridge.name, 'kongmu-cartridge-icon', cartridge.source_icon)}
        <span class="kongmu-name">${escapeHtml(cartridge.name)}</span>
      </button>
    `;
  }).join('');
  bindImageFallbacks(kongmuEls.cartridgeGrid);
}

function updatePlanButton() {
  const ready = Boolean(kongmuState.selectedCharacterId && kongmuState.selectedCartridgeId);
  kongmuEls.planBtn.disabled = !ready;
  if (ready && kongmuEls.planBtn.textContent !== '计算中') {
    kongmuEls.planBtn.textContent = '开始计算';
  }
}

async function runKongmuPlan() {
  if (!kongmuState.selectedCharacterId || !kongmuState.selectedCartridgeId) return;
  kongmuEls.planBtn.disabled = true;
  kongmuEls.planBtn.textContent = '计算中';
  try {
    kongmuState.plan = await kongmuRequest('/api/kongmu/plan', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        character_id: kongmuState.selectedCharacterId,
        cartridge_id: kongmuState.selectedCartridgeId,
      }),
    });
    kongmuState.driveMap = null;
    resetKongmuFilters();
    const params = new URLSearchParams();
    params.set('character', kongmuState.selectedCharacterId);
    params.set('cartridge', kongmuState.selectedCartridgeId);
    window.history.replaceState(null, '', `${window.location.pathname}?${params.toString()}`);
    renderKongmuPlan();
  } catch (error) {
    kongmuEls.resultShell.hidden = false;
    kongmuEls.resultTitle.textContent = '计算失败';
    kongmuEls.filterPanel.innerHTML = '';
    kongmuEls.solutionGrid.innerHTML = '';
    kongmuEls.emptyResult.hidden = false;
    kongmuEls.emptyResult.textContent = error.message || String(error);
  } finally {
    kongmuEls.planBtn.textContent = '开始计算';
    updatePlanButton();
  }
}

function resetKongmuFilters() {
  kongmuState.filterQueues = {
    2: [],
    3: [],
    4: [],
  };
}

function renderKongmuPlan() {
  const plan = kongmuState.plan;
  if (!plan) return;
  const character = plan.character || {};
  const cartridge = plan.cartridge || {};
  const passiveText = character.kongmu_passive?.text || '';
  kongmuEls.resultShell.hidden = false;
  kongmuEls.resultTitle.textContent = `${character.name || '-'} + ${cartridge.name || '-'}`;
  if (kongmuEls.passiveLine) {
    kongmuEls.passiveLine.textContent = passiveText;
    kongmuEls.passiveLine.hidden = !passiveText;
  }
  kongmuEls.totalCount.textContent = String(plan.result?.total_solution_count || 0);
  renderKongmuFilters();
  renderKongmuSolutions();
}

function renderKongmuFilters() {
  const result = kongmuState.plan?.result || {};
  const requiredDrives = result.required_drives || [];
  const optionalRows = result.optional_rows || [];
  const rows = [];

  rows.push(`
    <div class="kongmu-filter-row">
      <div class="kongmu-filter-row-title">必须驱动</div>
      <div class="kongmu-drive-strip">
        ${requiredDrives.map((drive) => driveChipHtml(drive)).join('')}
      </div>
    </div>
  `);

  optionalRows.forEach((row) => {
    const maxSelect = Math.max(1, Number(row.max_select || 1));
    for (let slotIndex = 0; slotIndex < maxSelect; slotIndex += 1) {
      rows.push(`
        <div class="kongmu-filter-row" data-filter-row="${escapeAttr(row.grid_count)}" data-slot-index="${slotIndex}">
          <div class="kongmu-filter-row-title">${escapeHtml(`${row.label} 筛选${slotIndex + 1}`)}</div>
          <div class="kongmu-drive-strip">
            ${(row.options || []).map((drive) => driveOptionHtml(row, drive, slotIndex)).join('')}
          </div>
        </div>
      `);
    }
  });

  kongmuEls.filterPanel.innerHTML = rows.join('');
  kongmuEls.filterPanel.querySelectorAll('.kongmu-drive-option').forEach((button) => {
    button.addEventListener('click', () => toggleKongmuFilter(
      button.dataset.gridCount,
      button.dataset.slotIndex,
      button.dataset.geometry,
    ));
  });
  bindImageFallbacks(kongmuEls.filterPanel);
}

function driveChipHtml(drive) {
  return `
    <span class="kongmu-drive-chip" title="${escapeAttr(`${drive.label || drive.name} x${drive.count || 0}`)}">
      ${imageHtml(drive.icon, drive.label || drive.name, '', drive.source_icon)}
      <span class="kongmu-drive-count">x${escapeHtml(drive.count || 0)}</span>
    </span>
  `;
}

function driveOptionHtml(row, drive, slotIndex) {
  const slots = kongmuState.filterQueues[Number(row.grid_count)] || [];
  const selected = slots[slotIndex] === drive.geometry;
  return `
    <button class="kongmu-drive-option${selected ? ' selected' : ''}" type="button"
      data-grid-count="${escapeAttr(row.grid_count)}"
      data-slot-index="${escapeAttr(slotIndex)}"
      data-geometry="${escapeAttr(drive.geometry)}"
      title="${escapeAttr(drive.label || drive.name)}">
      ${imageHtml(drive.icon, drive.label || drive.name, '', drive.source_icon)}
    </button>
  `;
}

function toggleKongmuFilter(gridCountValue, slotIndexValue, geometry) {
  const gridCount = Number(gridCountValue);
  const slotIndex = Number(slotIndexValue);
  const row = (kongmuState.plan?.result?.optional_rows || []).find((item) => Number(item.grid_count) === gridCount);
  if (!row || !geometry || !Number.isInteger(slotIndex) || slotIndex < 0) return;
  const slots = kongmuState.filterQueues[gridCount] || [];
  if (slots[slotIndex] === geometry) {
    slots[slotIndex] = '';
  } else {
    slots[slotIndex] = geometry;
  }
  kongmuState.filterQueues[gridCount] = slots.slice(0, Number(row.max_select || 1));
  renderKongmuFilters();
  renderKongmuSolutions();
}

function renderKongmuSolutions() {
  const plan = kongmuState.plan;
  const solutions = plan?.result?.solutions || [];
  const visible = solutions
    .map((solution, index) => ({solution, index}))
    .filter(({solution}) => kongmuSolutionMatches(solution));
  kongmuEls.visibleCount.textContent = String(visible.length);
  kongmuEls.emptyResult.hidden = visible.length > 0;
  kongmuEls.solutionGrid.innerHTML = visible.map(({solution, index}) => renderKongmuSolution(solution, index + 1)).join('');
  bindImageFallbacks(kongmuEls.solutionGrid);
}

function kongmuSolutionMatches(solution) {
  const optionalCounts = solution.optional_counts || {};
  const selectedCounts = selectedOptionalGeometryCounts();
  for (const [geometry, count] of Object.entries(selectedCounts)) {
    if (Number(optionalCounts[geometry] || 0) < count) {
      return false;
    }
  }
  return true;
}

function selectedOptionalGeometryCounts() {
  const counts = {};
  Object.values(kongmuState.filterQueues).flat().forEach((geometry) => {
    if (!geometry) return;
    counts[geometry] = Number(counts[geometry] || 0) + 1;
  });
  return counts;
}

function renderKongmuSolution(solution, displayIndex) {
  return `
    <article class="kongmu-solution">
      ${renderKongmuBoard(kongmuState.plan.character.equip_slots.slots, solution)}
    </article>
  `;
}

function renderKongmuBoard(slots, solution) {
  const rows = slots.length;
  const cols = Math.max(...slots.map((row) => row.length), 0);
  const cells = [];
  for (let y = 0; y < rows; y += 1) {
    const row = slots[y] || [];
    for (let x = 0; x < cols; x += 1) {
      const value = x < row.length ? row[x] : -1;
      cells.push(`<div class="kongmu-slot ${value === -1 ? 'invalid' : 'valid'}"></div>`);
    }
  }
  const pieces = (solution.pieces || []).map((piece) => pieceHtml(piece)).join('');
  return `
    <div class="kongmu-board" style="--cols:${cols}; --rows:${rows}">
      <div class="kongmu-board-cells">${cells.join('')}</div>
      <div class="kongmu-board-pieces">${pieces}</div>
    </div>
  `;
}

function pieceHtml(piece) {
  const drive = getDriveMap().get(piece.geometry) || {};
  const [minX, minY, maxX, maxY] = pieceBounds(piece);
  const pieceCols = maxX - minX;
  const pieceRows = maxY - minY;
  const selectedGeometries = new Set(Object.keys(selectedOptionalGeometryCounts()));
  const classes = ['kongmu-piece'];
  if (piece.is_required_drive) classes.push('required');
  if (!piece.is_required_drive && selectedGeometries.has(piece.geometry)) classes.push('optional-highlight');
  return `
    <div class="${classes.join(' ')}"
      style="grid-column:${minX + 1} / span ${pieceCols}; grid-row:${minY + 1} / span ${pieceRows}; --piece-cols:${pieceCols}; --piece-rows:${pieceRows}"
      title="${escapeAttr(piece.label || drive.label || '')}">
      ${imageHtml(drive.icon_url || drive.source_icon, piece.label || drive.label, 'kongmu-piece-icon', drive.source_icon)}
    </div>
  `;
}

function getDriveMap() {
  if (!kongmuState.driveMap) {
    kongmuState.driveMap = new Map((kongmuState.plan?.drives || []).map((drive) => [drive.geometry, drive]));
  }
  return kongmuState.driveMap;
}

function pieceBounds(piece) {
  const xs = piece.cells.map((cell) => Number(cell[0]));
  const ys = piece.cells.map((cell) => Number(cell[1]));
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs) + 1, Math.max(...ys) + 1];
}

function imageHtml(src, alt, className = '', fallback = '', style = '') {
  const attrs = [
    `src="${escapeAttr(src || fallback || '')}"`,
    `alt="${escapeAttr(alt || '')}"`,
    'loading="lazy"',
  ];
  if (className) attrs.push(`class="${escapeAttr(className)}"`);
  if (fallback && fallback !== src) attrs.push(`data-fallback="${escapeAttr(fallback)}"`);
  if (style) attrs.push(`style="${escapeAttr(style)}"`);
  return `<img ${attrs.join(' ')}>`;
}

function bindImageFallbacks(root) {
  root.querySelectorAll('img[data-fallback]').forEach((image) => {
    image.addEventListener('error', () => {
      const fallback = image.dataset.fallback;
      if (fallback && image.src !== fallback) {
        image.removeAttribute('data-fallback');
        image.src = fallback;
      }
    }, {once: true});
  });
}

function normalizeKongmuQuery(value) {
  return String(value || '').trim().toLowerCase().replace(/[\s_\-·・:：,，。/\\「」『』"'`]+/g, '');
}

async function kongmuRequest(url, options = {}) {
  const headers = Object.assign({}, options.headers || {});
  const token = typeof getToken === 'function' ? getToken() : '';
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  headers['X-Log-Id'] = `kongmu-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
  const response = await fetch(url, Object.assign({}, options, {headers}));
  let payload = {};
  try {
    payload = await response.json();
  } catch (error) {
    payload = {};
  }
  if (!response.ok || payload.error) {
    const logId = response.headers.get('X-Log-Id') || headers['X-Log-Id'];
    throw new Error(`${payload.error || '请求失败'}（logId: ${logId}）`);
  }
  return payload;
}

function escapeHtml(value) {
  if (typeof nteEscapeHtml === 'function') {
    return nteEscapeHtml(value);
  }
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[char]));
}

function escapeAttr(value) {
  if (typeof nteEscapeAttr === 'function') {
    return nteEscapeAttr(value);
  }
  return escapeHtml(value).replace(/`/g, '&#96;');
}
