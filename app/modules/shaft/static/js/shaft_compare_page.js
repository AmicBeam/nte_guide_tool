(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const page = document.querySelector('.compare-page');
  let lockedCharacter = '';
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const num = value => Number.isFinite(Number(value)) ? Number(value) : 0;
  const format = value => num(value).toLocaleString('zh-CN', {maximumFractionDigits: 0});
  const clone = value => JSON.parse(JSON.stringify(value));
  const token = () => { try { return localStorage.getItem('nte_token') || ''; } catch { return ''; } };
  const state = {catalog: null, axis: null, source: null, worker: null, timer: null, rows: [], baseline: null, baselineOptions: null, axisRequest: 0};

  async function request(url) {
    const response = await fetch(url, {headers: token() ? {Authorization: `Bearer ${token()}`} : {}, cache: 'no-store'});
    const payload = await response.json();
    if (!response.ok || payload.error) throw new Error(response.status === 401 ? '请先登录，或重新登录后读取个人排轴。' : String(payload.error || '读取失败，请重试。'));
    return payload;
  }
  function stop(message = '') {
    state.worker?.terminate(); state.worker = null;
    clearTimeout(state.timer);
    $('compare-cancel').hidden = true;
    $('compare-run').disabled = false;
    if (message) $('compare-progress').textContent = message;
  }
  function clearResults() {
    if ($('compare-detail').open) $('compare-detail').close();
    $('compare-matrices').replaceChildren();
    stop(); state.rows = []; state.baseline = null; state.baselineOptions = null;
    $('compare-results').hidden = true;
    $('compare-download').disabled = true;
    $('compare-progress').textContent = '';
  }
  function count() {
    const lengths = ['arcs', 'cartridges', 'stats'].map(id => document.querySelectorAll(`#compare-${id} input:checked`).length);
    const total = lengths.reduce((a, b) => a * b, 1);
    $('compare-count').textContent = `${total} 个组合 / 最多 600 个`;
    $('compare-run').disabled = !state.axis || total === 0 || total > 600;
  }
  function checks(id, options, selected) {
    $(`compare-${id}`).innerHTML = options.map(option => `<label><input type="checkbox" value="${esc(option.id)}" ${selected.has(option.id) ? 'checked' : ''}>${esc(option.name)}</label>`).join('');
  }
  function setupOptions() {
    clearResults();
    const member = state.axis.team.find(item => item.character_id === lockedCharacter);
    if (!member) return;
    const character = state.catalog.characters.find(item => item.id === member.character_id);
    const arcs = state.catalog.arcs.filter(arc => arc.adaptation === character.adaptation);
    const defaultArcs = new Set([member.arc_id || '', ...arcs.slice(0, 2).map(item => item.id)]);
    checks('arcs', arcs.map(arc => ({id:arc.id, name:`${arc.name} · ${ShaftComparison.candidateRefinement(arc)}精`})), defaultArcs);
    checks('cartridges', state.catalog.cartridges, new Set([member.cartridge_id || '', state.catalog.cartridges[0]?.id].filter(id => id !== undefined)));
    const stats = Object.entries(state.catalog.formula_constants.cartridge_main_stat_options).map(([id, option]) => ({id, name: option.label + (option.modifier_key ? '' : '（不参与伤害）')}));
    checks('stats', stats, new Set([member.cartridge_main_stat, '属伤', '暴击', '暴伤', '攻击']));
    count();
  }
  function renderAxis() {
    const {axis, source, catalog} = state;
    const byId = new Map(catalog.characters.map(item => [item.id, item]));
    const actions = new Map(catalog.actions.map(item => [item.id, item]));
    $('compare-axis-info').hidden = false;
    $('compare-axis-info').innerHTML = `<h3>${esc(source.title)}</h3><div class="compare-team">${axis.team.map(member => {
      const character = byId.get(member.character_id) || {};
      return `<span>${character.avatar ? `<img src="${esc(character.avatar)}" alt="">` : ''}${esc(character.name)} · ${esc(member.awakening_nodes?.length ?? member.awakening ?? 0)} 觉</span>`;
    }).join('')}</div><p class="compare-source-meta">${esc(source.owner?.nickname || '')}${source.source_version ? ` · 保存版本 ${esc(source.source_version)}` : ''}<br>${esc(source.description || '')}</p><details><summary>查看操作顺序与参考配置</summary><p class="compare-note">动作时间为轴内编辑位置，零秒时停的展示位置不计入真实轴长。</p><div class="compare-axis-steps">${axis.team.map(member => {
      const arc = catalog.arcs.find(item => item.id === member.arc_id);
      const cartridge = catalog.cartridges.find(item => item.id === member.cartridge_id);
      const skills = Object.entries({basic:'普攻',skill:'变轨',ultimate:'终结',support:'援护'}).map(([key,name]) => `${name} ${member.skill_levels?.[key] ?? '默认'}`).join(' / ');
      const substats = Object.entries(member.substat_counts || {}).filter(([,value]) => num(value) > 0).map(([key,value]) => `${catalog.formula_constants.substat_units[key]?.label || key} ${value}`).join('、') || '无额外词条';
      return `<p>${esc(byId.get(member.character_id)?.name)}：${esc(arc?.name || '未装备弧盘')} · ${member.arc_refinement ? `${esc(member.arc_refinement)} 精` : '默认精炼'} · ${esc(cartridge?.name || '未装备卡带')} · 主词条 ${esc(catalog.formula_constants.cartridge_main_stat_options[member.cartridge_main_stat]?.label || '原轴默认')}<br>技能等级：${esc(skills)}；副词条数：${esc(substats)}</p>`;
    }).join('')}<p>敌人等级：${esc(axis.enemy?.level ?? '原轴默认')}；弱点：${esc((axis.enemy?.weakness_elements || []).join('、') || '未设置')}。其余敌人参数均沿用原轴。</p>${[...axis.steps].sort((a,b) => num(a.start_tick) - num(b.start_tick)).map(step => `<div>${(num(step.start_tick) / 10).toFixed(1)}s · ${esc(byId.get(axis.team.find(item => num(item.slot) === num(step.slot))?.character_id)?.name)} · ${esc(actions.get(step.action_id)?.name || step.action_name || step.action_id)}${step.placement === 'background' ? '（后台）' : ''}</div>`).join('')}</div></details>`;
    const selectable = axis.team.filter(member => !byId.get(member.character_id)?.selection_disabled && (!lockedCharacter || member.character_id === lockedCharacter));
    $('compare-form').hidden = !selectable.length;
    $('compare-source-link').hidden = !source.id;
    if (source.id) $('compare-source-link').href = `/shaft/rotation?preview=${encodeURIComponent(source.id)}`;
    $('compare-source-status').textContent = selectable.length ? `已载入「${source.title}」。以下比较使用当前排轴数据重新计算，不改写参考轴。` : '该轴没有当前账号可试算的角色。';
    if (selectable.length) setupOptions();
  }
  function useAxis(source) {
    if (!Array.isArray(source.axis?.team) || !source.axis.team.length || !Array.isArray(source.axis.steps) || !source.axis.steps.length) throw new Error('这条轴没有可比较的角色或动作。');
    const known = new Map(state.catalog.characters.map(item => [item.id, item]));
    if (lockedCharacter && !source.axis.team.some(member => member.character_id === lockedCharacter)) throw new Error('这条轴没有当前角色，请选择包含该角色的排轴。');
    if (source.axis.team.some(member => !known.has(member.character_id) || known.get(member.character_id).selection_disabled)) throw new Error('这条轴包含当前账号不可用的角色，请选择其他排轴。');
    clearResults(); state.axis = clone(source.axis); state.source = source;
    renderAxis();
  }
  async function loadAxis(id) {
    const requestId = ++state.axisRequest;
    $('compare-source-status').textContent = '正在读取参考轴…';
    try {
      const source = await request(`/api/shaft/axes/${encodeURIComponent(id)}`);
      if (requestId !== state.axisRequest) return;
      useAxis(source);
    } catch (error) { if (requestId === state.axisRequest) $('compare-source-status').textContent = error.message; }
  }
  function label(kind, id) {
    return state.catalog[kind].find(item => item.id === id)?.name || (kind === 'arcs' ? '不装备弧盘' : '不装备卡带');
  }
  const percent = value => value == null ? '—' : `${num(value).toFixed(2)}%`;
  const delta = (value, baseline) => `${num(value) - num(baseline) >= 0 ? '+' : ''}${format(num(value) - num(baseline))}`;
  function renderBaseline() {
    const b = state.baseline;
    const o = state.baselineOptions;
    $('compare-baseline').innerHTML = `<p>原轴基准（100%）：${esc(label('arcs', o.arc_id))} · ${o.arc_refinement} 精 / ${esc(label('cartridges', o.cartridge_id))} / ${esc(state.catalog.formula_constants.cartridge_main_stat_options[o.main_stat]?.label || o.main_stat)}</p><div class="compare-baseline-grid">${[['全队总伤', b.total_damage], ['全队 DPS', b.dps], ['角色自身伤害', b.own_damage], ['轴长（秒）', b.duration_seconds]].map(([name,value]) => `<div><small>${name}</small><strong>${name.includes('轴长') ? num(value).toFixed(1) : format(value)}</strong></div>`).join('')}</div>${b.warnings.length ? `<details class="compare-warning"><summary>原轴有 ${b.warnings.length} 项自检提示</summary><ul>${b.warnings.map(w => `<li>${esc(w)}</li>`).join('')}</ul></details>` : ''}`;
  }
  function renderRows() {
    state.rows.sort((a,b) => b.total_damage - a.total_damage);
    const ratios = state.rows.map(row => row.baseline_percent).filter(Number.isFinite);
    const minimum = Math.min(100, ...ratios);
    const maximum = Math.max(100, ...ratios);
    const tone = ratio => {
      if (!Number.isFinite(ratio)) return '--compare-cell-color:var(--dim);--compare-cell-bg:transparent';
      const distance = ratio < 100 ? (100 - ratio) / Math.max(100 - minimum, 1e-9) : (ratio - 100) / Math.max(maximum - 100, 1e-9);
      const weight = Math.max(0, Math.min(1, distance));
      const edge = ratio < 100 ? [255, 153, 132] : [132, 230, 168];
      const color = [231, 235, 239].map((value, index) => Math.round(value + (edge[index] - value) * weight));
      return `--compare-cell-color:rgb(${color.join(',')});--compare-cell-bg:rgba(${edge.join(',')},${(weight * 0.14).toFixed(3)})`;
    };
    const indexed = state.rows.map((row, index) => ({row, index}));
    const cartridges = [...new Set(state.rows.map(row => row.cartridge_id))];
    const statNames = state.catalog.formula_constants.cartridge_main_stat_options;
    $('compare-matrices').innerHTML = cartridges.map(cartridgeId => {
      const entries = indexed.filter(entry => entry.row.cartridge_id === cartridgeId);
      const stats = Object.keys(statNames).filter(stat => entries.some(entry => entry.row.main_stat === stat));
      const arcs = [...new Set(entries.map(entry => entry.row.arc_id))];
      const cellMap = new Map(entries.map(entry => [JSON.stringify([entry.row.arc_id, entry.row.main_stat]), entry]));
      return `<section class="compare-matrix-section"><h3>${esc(label('cartridges', cartridgeId))}</h3><div class="compare-table-scroll"><table class="compare-matrix" aria-label="${esc(label('cartridges', cartridgeId))}配装收益"><thead><tr><th scope="col">弧盘 / 卡带主词条</th>${stats.map(stat => `<th scope="col">${esc(statNames[stat].label)}</th>`).join('')}</tr></thead><tbody>${arcs.map(arcId => `<tr><th scope="row">${esc(label('arcs', arcId))}<small>${entries.find(entry => entry.row.arc_id === arcId).row.arc_refinement} 精</small></th>${stats.map(stat => {
        const entry = cellMap.get(JSON.stringify([arcId, stat]));
        if (!entry) return '<td>—</td>';
        const ratio = entry.row.baseline_percent;
        return `<td><button type="button" class="compare-cell" style="${tone(ratio)}" data-result-index="${entry.index}" aria-haspopup="dialog" aria-label="${esc(label('arcs', arcId))}，${esc(statNames[stat].label)}，${percent(ratio)}，查看详情">${percent(ratio)}</button></td>`;
      }).join('')}</tr>`).join('')}</tbody></table></div></section>`;
    }).join('');
  }
  function openResult(index) {
    const row = state.rows[index];
    const b = state.baseline;
    if (!row || !b) return;
    const dialog = $('compare-detail');
    const toneStyle = $('compare-matrices').querySelector(`[data-result-index="${index}"]`)?.getAttribute('style') || '';
    $('compare-detail-title').textContent = `${label('arcs', row.arc_id)} · ${row.arc_refinement} 精`;
    const stat = state.catalog.formula_constants.cartridge_main_stat_options[row.main_stat]?.label || row.main_stat;
    const metrics = [['全队总伤','total_damage'],['全队 DPS','dps'],['角色自身伤害','own_damage'],['全队环合伤害','harmony_damage'],['全队倾陷伤害','stagger_damage'],['其他伤害','other_damage']];
    $('compare-detail-body').innerHTML = `<p>${esc(label('cartridges', row.cartridge_id))} · 主词条 ${esc(stat)}</p><p class="compare-detail-ratio" style="${esc(toneStyle)}">${percent(row.baseline_percent)}<small>原轴基准为100%</small></p><div class="compare-table-scroll"><table class="compare-detail-table"><thead><tr><th>指标</th><th>原轴</th><th>该组合</th><th>变化</th></tr></thead><tbody>${metrics.map(([name,key]) => `<tr><th scope="row">${name}</th><td>${format(b[key])}</td><td>${format(row[key])}</td><td>${delta(row[key], b[key])}</td></tr>`).join('')}</tbody></table></div><h3>主要动作变化</h3><ul>${row.action_changes.map(item => `<li>${esc(item.name)}：${format(item.damage)}（${delta(item.delta, 0)}）</li>`).join('')}</ul>${row.warnings.length ? `<h3 class="compare-warning">自检提示（${row.warnings.length}）</h3><ul class="compare-warning">${row.warnings.map(w => `<li>${esc(w)}</li>`).join('')}</ul>` : '<p class="compare-note">本组合未检出动作与资源问题。</p>'}`;
    dialog.showModal();
  }
  function run(event) {
    event.preventDefault(); clearResults();
    const selected = id => [...document.querySelectorAll(`#compare-${id} input:checked`)].map(input => input.value);
    const selection = {character_id:lockedCharacter, arc_ids:selected('arcs'), cartridge_ids:selected('cartridges'), main_stats:selected('stats')};
    let worker;
    try { worker = new Worker(page.dataset.workerUrl); } catch { $('compare-progress').textContent = '浏览器无法启动计算，请在完整浏览器中重试。'; return; }
    state.worker = worker;
    $('compare-run').disabled = true; $('compare-cancel').hidden = false;
    $('compare-progress').textContent = '正在重算原轴基准…';
    state.timer = setTimeout(() => { stop('计算超时，已停止。请减少组合后重试。'); $('compare-results').hidden = true; }, 120000);
    worker.onmessage = ({data}) => {
      if (state.worker !== worker) return;
      if (data.type === 'error') { stop(data.error); $('compare-results').hidden = true; return; }
      if (data.type === 'baseline') { state.baseline = data.baseline; state.baselineOptions = data.baseline_options; renderBaseline(); }
      if (data.type === 'row') { state.rows.push(data.row); $('compare-progress').textContent = `已计算 ${data.completed} / ${data.total} 个组合`; }
      if (data.type === 'done') { renderRows(); $('compare-results').hidden = false; $('compare-download').disabled = false; stop(`已完成 ${state.rows.length} 个组合，按卡带分表展示。`); }
    };
    worker.onerror = () => { if (state.worker === worker) { stop('计算未完成，请刷新后重试。'); $('compare-results').hidden = true; } };
    worker.postMessage({axis:state.axis, catalog:state.catalog, selection});
  }
  function download() {
    const cell = value => `"${String(typeof value === 'string' && /^[=+\-@]/.test(value) ? `'${value}` : value ?? '').replace(/"/g, '""')}"`;
    const rows = [['参考轴',state.source.title,'角色',state.catalog.characters.find(c => c.id === lockedCharacter)?.name], ['条件','固定原轴动作、队友、敌人、觉醒、技能、空幕与副词条；S级弧盘1精、A级与B级5精'], ['基准总伤',state.baseline.total_damage,'基准DPS',state.baseline.dps], ['弧盘','精炼','卡带','主词条','全队总伤','DPS','原轴为100%的比例%','角色自身伤害','全队环合','全队倾陷','自检提示']];
    for (const row of state.rows) rows.push([label('arcs',row.arc_id),row.arc_refinement,label('cartridges',row.cartridge_id),state.catalog.formula_constants.cartridge_main_stat_options[row.main_stat]?.label,row.total_damage,row.dps,row.baseline_percent,row.own_damage,row.harmony_damage,row.stagger_damage,row.warnings.join('；')]);
    const url = URL.createObjectURL(new Blob(['\ufeff' + rows.map(row => row.map(cell).join(',')).join('\r\n')], {type:'text/csv;charset=utf-8'}));
    const link = document.createElement('a'); link.href = url; link.download = '配装收益对比.csv'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function init() {
    $('compare-login').href = `/login?next=${encodeURIComponent(location.pathname + location.search)}`;
    $('compare-login').textContent = token() ? '账号 / 登录' : '登录';
    $('compare-form').onsubmit = run;
    $('compare-form').onchange = () => { clearResults(); count(); };
    $('compare-form').onclick = event => {
      const button = event.target.closest('[data-select-all], [data-select-none]'); if (!button) return;
      const id = button.dataset.selectAll || button.dataset.selectNone;
      document.querySelectorAll(`#compare-${id} input`).forEach(input => { input.checked = Boolean(button.dataset.selectAll); }); clearResults(); count();
    };
    $('compare-cancel').onclick = () => { stop('已取消计算，参考轴保持原样。'); state.rows = []; };
    $('compare-download').onclick = download;
    $('compare-matrices').onclick = event => {
      const button = event.target.closest('[data-result-index]');
      if (button) openResult(Number(button.dataset.resultIndex));
    };
    $('compare-detail-close').onclick = () => $('compare-detail').close();
    $('compare-detail').addEventListener('click', event => { if (event.target === event.currentTarget) event.currentTarget.close(); });
    try {
      state.catalog = await request('/api/shaft/catalog');
      const name = ['1046','1051'].includes(page.dataset.guideCharacterId) ? '主角' : page.dataset.guideCharacterName;
      const character = state.catalog.characters.find(item => item.name === name && !item.selection_disabled);
      if (!character) throw new Error('当前角色尚不可用于配装比较，请返回角色页查看相关排轴。');
      lockedCharacter = character.id;
      const rarities = JSON.parse(page.dataset.arcRarities);
      for (const arc of state.catalog.arcs) if (!arc.rarity && rarities[arc.name]) arc.rarity = rarities[arc.name];
      if (!/^\d+$/.test(page.dataset.axisId || '')) throw new Error('请从角色排轴与配装页面选择实际排轴。');
      await loadAxis(page.dataset.axisId);
    } catch (error) { $('compare-source-status').textContent = error.message; }
  }
  window.addEventListener('pagehide', () => stop());
  init();
}());
