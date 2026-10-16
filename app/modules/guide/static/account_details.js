'use strict';

function guideEquipmentReference(entries, item, kind) {
  const normalize = value => String(value || '').trim().replace(/^[「『“"]|[」』”"]$/g, '').toLowerCase();
  const candidates = Object.entries(entries).filter(([, entry]) => !kind || entry.kind === kind);
  const identifier = String(item.id || '').toLowerCase();
  const exact = candidates.find(([id]) => id.toLowerCase() === identifier);
  if (exact) return exact[1];
  const name = normalize(item.name);
  if (!name) return null;
  const matching = candidates.filter(([, entry]) => [entry.name, ...(entry.aliases || [])].some(alias => normalize(alias) === name));
  // Several quality IDs can point at the same shape, but a shared model name cannot select a shape.
  const images = new Set(matching.map(([, entry]) => entry.image));
  return images.size === 1 ? matching[0][1] : null;
}

(() => {
  const dialog = document.getElementById('account-detail-dialog');
  if (!dialog) return;
  const content = document.getElementById('account-detail-content');
  const arcs = JSON.parse(document.getElementById('account-arcs').textContent);
  const equipmentImages = JSON.parse(document.getElementById('account-equipment-images').textContent);
  const node = (tag, className, text) => {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (text !== undefined) result.textContent = text;
    return result;
  };
  const display = value => value === null || value === undefined || value === '' ? '未提供' : String(value);
  const image = (src, className) => {
    const img = node('img', className);
    img.src = src; img.alt = '';
    img.addEventListener('error', () => { img.hidden = true; });
    return img;
  };
  const propertyRows = (properties, className = '') => {
    const list = node('dl', 'account-stat-list ' + className);
    if (!properties?.length) list.append(node('p', 'account-missing', '属性未提供'));
    for (const property of properties || []) {
      const row = node('div');
      row.append(node('dt', '', display(property.name || property.id)), node('dd', '', display(property.value)));
      list.append(row);
    }
    return list;
  };
  const open = (saved, profile, local) => {
    content.replaceChildren();
    const hero = node('section', 'account-build-hero');
    const art = node('div', 'account-build-art');
    if (local?.portrait || local?.avatar) art.append(image(local.portrait || local.avatar, 'account-build-portrait'));
    else art.append(node('p', 'account-missing', '暂无角色立绘'));
    const awakenBar = node('div', 'account-awaken-bar');
    awakenBar.append(node('span', 'account-awaken-badge', `觉醒 ${display(saved.awaken_level)}`));
    const effects = (saved.awaken_effects || []).filter(value => String(value).trim());
    const icons = node('div', 'account-awaken-icons');
    icons.setAttribute('aria-label', '觉醒效果');
    for (const value of effects) {
      const metadata = local?.awakens?.[value];
      if (!metadata?.icon) continue;
      const effect = node('button', 'account-awaken-effect');
      effect.type = 'button'; effect.title = value;
      effect.setAttribute('aria-label', value);
      effect.append(image(metadata.icon, 'account-awaken-icon'));
      effect.addEventListener('click', () => {
        const wasOpen = effect.classList.contains('is-open');
        for (const other of icons.querySelectorAll('.is-open')) other.classList.remove('is-open');
        if (!wasOpen) effect.classList.add('is-open');
      });
      effect.append(node('span', 'account-awaken-name', value));
      icons.append(effect);
    }
    awakenBar.append(icons);
    const unmatched = effects.filter(value => !local?.awakens?.[value]?.icon);
    if (unmatched.length) {
      const notice = node('span', 'account-awaken-missing', `${unmatched.length}项效果未识别`);
      notice.title = unmatched.join('、'); awakenBar.append(notice);
    }
    art.append(awakenBar);
    const panel = node('div', 'account-build-panel');
    const identity = node('div', 'account-build-identity');
    identity.append(node('span', 'account-level-badge', `Lv.${display(saved.level)}`));
    const title = node('h2', '', local?.name || saved.name || saved.id);
    title.id = 'account-detail-title'; identity.append(title);
    const properties = saved.properties || [];
    const featuredNames = new Set(['通用伤害增强']);
    if (local?.element_name) featuredNames.add(`${local.element_name}属性异能伤害增强`);
    const featured = properties.filter(property => featuredNames.has(property.name));
    const remaining = properties.filter(property => !featuredNames.has(property.name));
    const main = [...remaining.slice(0, Math.max(0, 8 - featured.length)), ...featured];
    const extraProperties = properties.filter(property => !main.includes(property));
    panel.append(identity, node('p', 'account-build-uid', `UID：${profile.role_id}`), propertyRows(main));
    if (extraProperties.length) {
      const extra = node('details', 'account-extra-properties');
      extra.append(node('summary', '', '查看其余属性'), propertyRows(extraProperties));
      panel.append(extra);
    }
    hero.append(art, panel);
    const skills = node('div', 'account-build-skills');
    for (const skill of saved.skills || []) {
      const metadata = local?.skill_details?.[String(skill.id || '').toLowerCase()];
      const skillCard = node('div', 'account-build-skill');
      if (metadata?.icon) skillCard.append(image(metadata.icon, 'account-skill-icon'));
      const label = node('span', '', metadata?.name || skill.name || skill.id || '未知技能');
      skillCard.append(node('strong', '', skill.level == null ? '未提供' : `Lv.${skill.level}`), label);
      skills.append(skillCard);
    }
    if (!saved.skills?.length) skills.append(node('p', 'account-missing', '技能等级未提供'));
    hero.append(skills); content.append(hero);

    const equipment = saved.equipment || {};
    const fork = equipment.fork || {};
    const arc = guideEquipmentReference(arcs, fork);
    const arcPanel = node('section', 'account-build-arc');
    if (arc?.image) arcPanel.append(image(arc.image, 'account-arc-image'));
    const arcInfo = node('div');
    arcInfo.append(node('p', 'account-section-label', '已装备弧盘'), node('h3', '', arc?.name || fork.name || fork.id || '未提供'));
    const mix = /^[1-5]$/.test(String(fork.slev)) ? fork.slev : '未提供';
    arcInfo.append(node('p', 'account-equipment-level', `Lv.${display(fork.alev)} · 混频 ${mix}`));
    arcPanel.append(arcInfo, propertyRows(fork.properties)); content.append(arcPanel);

    const suit = equipment.suit || {};
    const gear = node('section', 'account-build-gear');
    gear.append(node('h3', 'account-gear-heading', `空幕 / 卡带 · ${suit.name || suit.id || '未提供'}`));
    const pieces = node('div', 'account-gear-grid');
    for (const [category, entries] of [['核心', suit.core], ['驱动块', suit.pie]]) {
      for (const item of entries || []) {
        const card = node('section', 'account-gear-piece');
        const header = node('div', 'account-gear-piece-heading');
        header.append(node('span', 'account-section-label', category), node('span', '', `Lv.${display(item.level)}`));
        const identity = node('div', 'account-piece-identity');
        const reference = guideEquipmentReference(equipmentImages, item, category === '核心' ? 'core' : 'drive');
        if (reference?.image) identity.append(image(reference.image, 'account-piece-image'));
        identity.append(node('h4', '', item.name || reference?.name || item.id || '未提供'));
        card.append(header, identity);
        if (item.main_properties?.length) card.append(propertyRows(item.main_properties, 'account-main-properties'));
        if (item.properties?.length) card.append(propertyRows(item.properties));
        if (!item.main_properties?.length && !item.properties?.length) card.append(node('p', 'account-missing', '属性未提供'));
        pieces.append(card);
      }
    }
    if (!pieces.children.length) pieces.append(node('p', 'account-missing', '尚无已保存的空幕卡带明细'));
    gear.append(pieces); content.append(gear);
    const footer = node('footer', 'account-build-footer');
    footer.append(node('p', '', `同步于 ${new Date(profile.updated_at).toLocaleString('zh-CN')} · 技能为同步展示等级`));
    if (local) {
      const links = node('div', 'guide-links');
      for (const [label, suffix] of [['角色资料', ''], ['养成计算', '/cultivation']]) {
        const link = node('a', '', label);
        link.href = '/guide/characters/' + encodeURIComponent(saved.id) + suffix; links.append(link);
      }
      footer.append(links);
    }
    content.append(footer);
    dialog.showModal(); dialog.scrollTop = 0;
  };
  const close = () => { if (dialog.open) dialog.close(); };
  document.getElementById('account-detail-close').addEventListener('click', close);
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) close();
  });
  window.guideAccountDetails = { open, close };
})();
