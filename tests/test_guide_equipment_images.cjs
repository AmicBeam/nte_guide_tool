const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({document: {getElementById: () => null}});
vm.runInContext(fs.readFileSync('app/modules/guide/static/account_details.js', 'utf8'), context);
const resolve = context.guideEquipmentReference;
const equipment = JSON.parse(fs.readFileSync('app/modules/guide/data/equipment_images.json', 'utf8'));

test('arc case variants and instance IDs resolve through a unique local name', () => {
  const arcs = {fork_Rose: {name: '「最后一朵玫瑰」', image: '/rose.webp'}};
  assert.equal(resolve(arcs, {id: 'fork_rose'}).image, '/rose.webp');
  assert.equal(resolve(arcs, {id: 'instance-only', name: '最后一朵玫瑰'}).image, '/rose.webp');
  assert.equal(resolve(arcs, {name: '未收录弧盘'}), null);
});

test('drive shape follows the explicit template ID rather than the shared model name', () => {
  const horizontal = resolve(equipment, {id: 'CELL3_STYLE1_1_ORANGE', name: 'Ⅲ型驱动'}, 'drive');
  const corner = resolve(equipment, {id: 'cell3_style3_1', name: 'Ⅲ型驱动'}, 'drive');
  assert.ok(horizontal.image.endsWith('EquipmentGeometry_Hen3.webp'));
  assert.ok(corner.image.endsWith('EquipmentGeometry_ZhiJiao1.webp'));
  assert.equal(resolve(equipment, {id: 'unknown-instance', name: 'Ⅲ型驱动'}, 'drive'), null);
});

test('core images resolve from the local unique name and never cross equipment categories', () => {
  assert.ok(resolve(equipment, {name: '迪亚波罗斯'}, 'core').image.endsWith('/chaos.webp'));
  assert.equal(resolve(equipment, {id: 'chaos'}, 'drive'), null);
});

test('all local equipment image bindings point to delivered guide assets', () => {
  const paths = new Set(Object.values(equipment).map(item => item.image));
  assert.equal(paths.size, 25);
  for (const path of paths) {
    assert.ok(path.startsWith('/static/guide/equipment/'));
    assert.ok(fs.existsSync('app/modules/guide/static/' + path.slice('/static/guide/'.length)), path);
  }
});
