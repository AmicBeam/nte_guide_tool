"""Build Guide combat tables from exported UE ability and animation assets."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'app/modules/guide/data'
TYPES = ('Melee', 'Skill', 'UltraSkill', 'QTE', 'Peculiarity')
COLUMNS = ('动作', '连段窗口（秒）', '连段占时推算（秒）', '动作衔接（秒）', '伤害通知（秒）', '命中顿帧（秒）',
           '切人锁定（秒）', '解除暂停事件（秒）', '伤害判定')
MELEE = re.compile(r'(Ultra)?Melee([A-Z]?)(\d+)(.*)', re.I)
LABELS = {
    'MeleePre': '接近起手', 'AppearMelee': '入场攻击', 'AirAttack': '下落起始', 'AirAttackEnd': '着地攻击',
    'AirAttackLoop': '下落循环', 'Atk': '反击攻击', 'PerfectEvadeBack': '后向极限闪避',
    'PerfectEvadeFront': '前向极限闪避', 'Skill': '变轨技能', 'Default': '技能起手', 'QTE': '援护攻击',
    'Skillhold': '长按', 'Skill_Short': '强化短按', 'Skill_Long': '强化长按', 'UltraSkill': '极轨终结',
    'Fire': '射击', 'Fire_LoopAim': '瞄准射击', 'Branch': '分支攻击', 'End': '收招', 'Loop': '循环',
    'UltraSkillB': '变身终结', 'Atk_tran': '变身反击', 'QTE_tran': '变身援护',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def props(export):
    return export.get('Properties') or {}


def game_path(ref):
    path = (ref or {}).get('ObjectPath', '')
    return path[6:].split('.', 1)[0] if path.startswith('/Game/') else ''


def ref_name(ref):
    return game_path(ref).rsplit('/', 1)[-1]


def fmt(number):
    return f'{number:.3f}'.rstrip('0').rstrip('.')


def vec(value, separator=', '):
    return separator.join(fmt(value.get(axis, 0)) for axis in 'XYZ')


def move_label(section, ability_type):
    if match := MELEE.fullmatch(section):
        enhanced, form, num, suffix = match.groups()
        prefix = '强化' if enhanced else ''
        if form:
            prefix += f'{form.upper()} 形态'
        variant = f'（变体 {suffix.lstrip("_")}）' if suffix else ''
        return f'{prefix}第 {int(num)} 段{variant}'
    if section == 'Default':
        return {'Skill': '变轨技能', 'UltraSkill': '极轨终结', 'QTE': '援护攻击'}.get(ability_type, '技能起手')
    if match := re.fullmatch(r'(Skill|ExtraAttack|Branch)(\d+)', section, re.I):
        title = {'skill': '变轨', 'extraattack': '强化攻击', 'branch': '分支攻击'}[match[1].lower()]
        return f'{title}第 {int(match[2])} 段'
    return LABELS.get(section, section)


def tag_name(entry):
    tag = entry.get('Key', {})
    return tag.get('TagName', '') if isinstance(tag, dict) else tag


def export_index(ref):
    index = (ref or {}).get('ObjectPath', '').rsplit('.', 1)[-1]
    return int(index) if index.isdecimal() else None


def note_export(notify, names):
    ref = notify.get('NotifyStateClass') or notify.get('Notify') or {}
    name = ref.get('ObjectName', '').rsplit(':', 1)[-1].strip("'")
    return names.get(name, {})


def notify_time(notify):
    if notify.get('LinkMethod') == 'EAnimLinkMethod::Relative':
        return notify['SegmentBeginTime'] + notify['LinkValue']
    if notify.get('LinkMethod') == 'EAnimLinkMethod::Proportional':
        return notify['SegmentBeginTime'] + notify['SegmentLength'] * notify['LinkValue']
    if notify.get('LinkMethod') in (None, 'EAnimLinkMethod::Absolute'):
        return notify.get('LinkValue')
    return None


class Assets:
    def __init__(self, blueprints, montages=None):
        self.blueprints = {p.relative_to(blueprints).with_suffix('').as_posix().casefold(): p
                           for p in blueprints.rglob('*.json')}
        self.montages = {}
        if montages is not None:
            self.montages = {path.relative_to(montages).with_suffix('').as_posix().casefold().removeprefix('ht/content/'): path
                             for path in montages.rglob('*.json')}
        self.abilities = {p.stem.casefold(): k for k, p in self.blueprints.items()
                          if k.startswith('blueprints/abilities/player/') and p.stem.startswith('GA_')}

    @lru_cache(maxsize=None)
    def blueprint(self, key):
        path = self.blueprints.get(key.casefold())
        return read_json(path) if path else []

    @lru_cache(maxsize=None)
    def animation(self, key):
        path = self.montages.get(key.casefold())
        return read_json(path) if path else []

    @lru_cache(maxsize=None)
    def defaults(self, key):
        exports = self.blueprint(key)
        defaults = next((props(o) for o in exports if o['Name'].startswith('Default__')), {})
        parent = next((o.get('Super') or o.get('SuperStruct') for o in exports if o['Type']=='BlueprintGeneratedClass'), None)
        inherited = self.defaults(game_path(parent)) if game_path(parent) else ({}, '')
        return {**inherited[0], **defaults}, inherited[1] or (parent or {}).get('ObjectName', '')

    @lru_cache(maxsize=None)
    def damage_range(self, key):
        p, native = self.defaults(key)
        if 'BoxTrace' in native and p.get('HalfSize'):
            shape = '盒体半尺寸 ' + vec(p['HalfSize'], ' × ')
        elif 'SphereTrace' in native and p.get('SphereRadius'):
            shape = '球形半径 ' + fmt(p['SphereRadius'])
        elif 'CapsuleTrace' in native and p.get('CapsuleRadius') and p.get('CapsuleHalfHeight'):
            shape = f"胶囊半径 {fmt(p['CapsuleRadius'])}、半高 {fmt(p['CapsuleHalfHeight'])}"
        else:
            return ''
        values = [shape]
        offset, sweep = p.get('OffsetFromActor'), p.get('TraceVector')
        if offset and sweep and offset == sweep and any(offset.values()):
            values.append('偏移／扫掠 (' + vec(offset) + ')')
        else:
            for title, value in [('偏移', offset), ('扫掠', sweep)]:
                if value and any(value.values()):
                    values.append(f'{title} ({vec(value)})')
        if p.get('TraceLength'):
            values.append('检测长度 ' + fmt(p['TraceLength']))
        if p.get('MaxCount', 0) > 0:
            values.append(f"目标上限 {p['MaxCount']}")
        return '，'.join(values)

    def attack_hit_pause(self, effect):
        pause = self.defaults(game_path(effect))[0].get('HitPauseData') or {}
        seconds = pause.get('AttackDilationTime')
        return seconds if isinstance(seconds, (int, float)) and seconds > 0 else None

    def resolve_export(self, ref):
        exports = self.blueprint(game_path(ref))
        index = export_index(ref)
        return exports[index] if index is not None and index < len(exports) else {}

    def projectile_collision(self, actor):
        defaults = next((props(export) for export in actor if export['Name'].startswith('Default__')), {})
        if not defaults.get('ProjectileMovementComponent'):
            return None
        collision = props(self.resolve_export(defaults.get('BoxComponent')))
        if extent := collision.get('BoxExtent'):
            return '弹体碰撞盒半尺寸 ' + vec(extent, ' × ')
        return None

    def hit_containers(self, ability):
        result = defaultdict(list)
        for entry in ability.get('EffectContainerMap', []):
            result[tag_name(entry)].append((entry['Value'], False))
        for entry in ability.get('SpawnActorMaps', []):
            actor = self.resolve_export(entry.get('Value'))
            for value in nested_dicts(props(actor)):
                container = value.get('EffectContainer')
                if container:
                    result[tag_name(entry)].append((container, True))
                reference = value.get('SpawnClass') or value.get('SapwnActorClass')
                actor_exports = self.blueprint(game_path(reference))
                collision = self.projectile_collision(actor_exports)
                for component in actor_exports:
                    for key,data in props(component).items():
                        if key.endswith('EffectContainer') and isinstance(data,dict):
                            if collision:
                                data = {**data, 'projectile_collision': collision}
                            result[tag_name(entry)].append((data, True))
        return result


def nested_dicts(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from nested_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_dicts(item)


def order_unique(values):
    return list(dict.fromkeys(values))


def section_window(montage, key):
    sections = montage.get('CompositeSections') or []
    if len(sections) == 1:
        return (notify_time(sections[0]) or 0), montage['SequenceLength']
    times = [(section['SectionName'].casefold(), notify_time(section)) for section in sections]
    bounds = next((t for name, t in times if name == key.casefold()), None)
    if bounds is None:
        return None
    return bounds, min((t for _, t in times if t is not None and t > bounds), default=montage['SequenceLength'])


def followup_thresholds(assets, ability):
    thresholds = {}
    for section in ability.get('JumpSectionMap', []):
        jump = assets.resolve_export(section.get('Value'))
        for reference in props(jump).get('JumpSectionTypes', []):
            trigger = assets.resolve_export(reference)
            config = props(trigger)
            if (trigger.get('Type') != 'JumpSection_LocalClientGenericCheck'
                    or not MELEE.fullmatch(config.get('NextSectionName', ''))):
                continue
            requirements = config.get('OptionalRequirements', [])
            if not any('InputID_Melee' in rule.get('SpecifiedInputTypeArr', []) for rule in requirements):
                continue
            for rule in requirements:
                for position in rule.get('MontagePositionRequirement', []):
                    lower = position.get('LowerBound', {})
                    if lower.get('Type') in ('ERangeBoundTypes::Inclusive', 'ERangeBoundTypes::Exclusive'):
                        thresholds.setdefault(section['Key'], []).append(lower['Value'])
    return {key: min(values) for key, values in thresholds.items()}


def inspect_montage(assets, exports, key, ability, definition, action_label=None):
    montage = next((props(o) for o in exports if o['Type'] == 'AnimMontage'), None)
    if not montage or not (window := section_window(montage, key)):
        return None
    start, end = window
    names = {o['Name']: o for o in exports}
    rate = montage.get('CommonTargetFrameRate') or {}
    fps = rate.get('Numerator', 0) / (rate.get('Denominator') or 1)
    times = []
    windows = []
    predictions = []
    cancel = []
    locks = []
    pauses = []
    ranges = []
    hit_pauses = Counter()
    direct_effects = set()
    spawn_effects = set()
    for notify in montage.get('Notifies') or []:
        time = notify_time(notify)
        if time is None or not start <= time < end:
            continue
        relative = time - start
        component = note_export(notify, names)
        p, kind = props(component), component.get('Type', '')
        end_time = notify_time(notify.get('EndLink') or {})
        if kind.startswith('BP_JumpSection') and p.get('JumpSections') and end_time is not None and end_time > time:
            windows.append(f'{fmt(relative)}–{fmt(min(end_time, end) - start)}')
            frame = p.get('nFrame')
            if fps and isinstance(frame, int) and frame > 0 and time + frame / fps <= min(end_time,end):
                predictions.append(fmt(relative + frame / fps))
        elif kind.startswith('BP_CanBreakSkill') and any(t.endswith('BreakSkill_AnySkill') for t in p.get('BreakSkillTypes',[])):
            cancel.append(relative)
        elif 'CantChangePlayer' in kind and end_time is not None:
            locks.append(f'{fmt(relative)}–{fmt(min(end_time, end) - start)}')
        elif kind.startswith('BP_SendGamePlayEvent'):
            tag = (p.get('TriggerEventTag') or {}).get('TagName','')
            if tag in definition['pause_end']:
                pauses.append(fmt(relative))
            hit_pauses_at_event = set()
            for container, is_spawn in definition['hits'].get(tag, []):
                damage = [ref for ref in container.get('TargetGameplayEffectClasses', [])
                          if ref_name(ref).casefold().startswith('ge_') and 'damage' in ref_name(ref).casefold()]
                if not damage:
                    continue
                (spawn_effects if is_spawn else direct_effects).update(ref_name(ref).casefold() for ref in damage)
                target = container.get('TargetType') or {}
                geometry = assets.damage_range(game_path(target))
                if target.get('ObjectName') == "Class'HTTargetType_AbilityTarget'":
                    geometry = '技能目标'
                if geometry:
                    ranges.append(('生成物：' if is_spawn else '') + geometry)
                elif collision := container.get('projectile_collision'):
                    ranges.append(collision)
                if not is_spawn:
                    times.append(fmt(relative))
                    for effect in damage:
                        if seconds := assets.attack_hit_pause(effect):
                            hit_pauses_at_event.add(seconds)
            hit_pauses.update(hit_pauses_at_event)
    pause_column = ' / '.join(f'{fmt(value)}*{count}' if count > 1 else fmt(value)
                              for value, count in sorted(hit_pauses.items()))
    threshold = definition['followup'].get(key)
    if threshold is not None and not start <= threshold < end:
        threshold = None
    row = (
        action_label or move_label(key, ability['type_name']),
        ' / '.join(order_unique(windows)) or None,
        ' / '.join(order_unique(predictions)) or None,
        fmt(threshold - start) if threshold is not None else (fmt(min(cancel)) if cancel else None),
        ' / '.join(order_unique(times)) or None,
        pause_column or None,
        ' / '.join(order_unique(locks)) or None,
        ' / '.join(order_unique(pauses)) or None,
        ' / '.join(order_unique(ranges)) or None,
    )
    return (row, direct_effects, spawn_effects) if any(row[1:]) else None


def related(assets, ability):
    main_id = ability['id'].casefold()
    primary = assets.abilities.get(main_id)
    if not primary:
        return []
    directory = primary.rsplit('/', 1)[0]
    terminal = main_id.removesuffix('_ultraskill') + '_subultraskill' if ability['type_name'] == 'UltraSkill' else None
    variants = []
    for key in assets.abilities.values():
        parent, name = key.rsplit('/', 1)
        if parent == directory and (name == terminal or name.startswith(main_id + '_') or (
            ability['type_name'] == 'Melee' and 'perfectevade' in name
        )):
            variants.append(key)
    return [primary, *sorted(variants)]


def montage_bindings(definition):
    for bind in definition.get('MontageToPlays', []):
        key = bind['Key']
        animation = bind['Value'].get('AssetPathName', '')
        if animation.startswith('/Game/') and not key.startswith('Dissolve'):
            yield key, animation[6:].split('.', 1)[0]


def requested_montages(assets, characters):
    paths = {animation for character in characters for ability in character['abilities']
             if ability['type_name'] in TYPES
             for key in related(assets, ability)
             for _, animation in montage_bindings(assets.defaults(key)[0])}
    return [f'HT/Content/{path}.uasset' for path in sorted(paths)]


def match_group(ability, direct_effects, spawn_effects, primary, key):
    groups = ability.get('groups') or []
    matches = []
    for group in groups:
        effects = {hit['effect_id'].casefold() for row in group['param_rows']
                   for hit in row.get('hit_details') or []}
        score = (len(effects & direct_effects), len(effects & spawn_effects))
        if any(score):
            matches.append((score, group['name']))
    if matches:
        top = max(score for score, _ in matches)
        names = [name for score, name in matches if score == top]
        if len(names) == 1:
            return names[0]
        groups = [group for group in groups if group['name'] in names]
    if not primary:
        return None
    if len(groups) == 1:
        return groups[0]['name']
    preferred = [group['name'] for group in groups if group['name'].split('：')[-1] == ability['name']]
    if len(preferred) == 1 and (MELEE.fullmatch(key) or key in ('MeleePre', 'AppearMelee', 'Default',
                                                              'Skill', 'QTE', 'UltraSkill')):
        return preferred[0]
    if ability['type_name'] == 'Melee':
        keywords = {'AirAttack': ('下落', '滞空'), 'AirAttackEnd': ('下落', '滞空'),
                    'Fire': ('瞄准',), 'Fire_LoopAim': ('瞄准',)}
        if tokens := keywords.get(key):
            likely = [group['name'] for group in groups
                      if group['name'].startswith('普通攻击：') and
                      any(token in (group.get('description') or '') for token in tokens)]
            if len(likely) == 1:
                return likely[0]
    return None

def build_character(assets, character, curated):
    record = deepcopy(curated)
    for ability in character['abilities']:
        if ability['type_name'] not in TYPES:
            continue
        candidates = related(assets, ability)
        if not candidates:
            continue
        main = candidates[0]
        rows = defaultdict(list)
        for path in candidates:
            raw = assets.blueprint(path)
            definition = assets.defaults(path)[0]
            pause = {tag for o in raw if o['Type']=='HTGAComponent_AbilityGamePaused'
                         for tag in props(o).get('TriggerEndEventTags',[])}
            data = {'hits': assets.hit_containers(definition), 'pause_end': pause,
                    'followup': followup_thresholds(assets, definition)}
            terminal = ability['type_name'] == 'UltraSkill' and path.endswith(('_end', '_subultraskill'))
            for key, animation in montage_bindings(definition):
                label = '终结终段' if terminal and key == 'Default' else None
                measured = inspect_montage(assets, assets.animation(animation), key, ability, data, label)
                if not measured:
                    continue
                row, direct_effects, spawn_effects = measured
                group = match_group(ability, direct_effects, spawn_effects,
                                    path == main or path.startswith(main + '_') or terminal, key)
                if group:
                    rows[group].append(row)
        for name, moves in rows.items():
            columns = [i for i in range(len(COLUMNS)) if any(row[i] is not None for row in moves)]
            record.setdefault(ability['id'], {}).setdefault(name, {
                'columns': [COLUMNS[i] for i in columns],
                'rows': [[row[i] for i in columns] for row in order_unique(moves)],
            })
    return record


def build_tables(assets, characters, curated):
    result = {}
    missing = []
    for character in characters:
        identifier = str(character['id'])
        tables = build_character(assets, character, curated.get(identifier, {}))
        if tables:
            result[identifier] = tables
        for ability in character['abilities']:
            if ability['type_name'] in TYPES[:4] and ability['id'] not in tables:
                missing.append(f"{character['name']} ({ability['id']})")
    return result, missing


def write_snapshot(path, data):
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blueprints', type=Path, required=True, help='Root containing the NTEData Blueprints directory')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--montages', type=Path, help='Root containing exported HT/Content animation JSON')
    mode.add_argument('--list-montages', action='store_true', help='Print client asset paths for an export manifest')
    parser.add_argument('--catalog', type=Path, default=DATA / 'catalog.json')
    parser.add_argument('--curated', type=Path, default=DATA / 'skill_combat_overrides.json')
    parser.add_argument('--output', type=Path, default=DATA / 'skill_combat_details.json')
    args = parser.parse_args()
    catalog = read_json(args.catalog)['characters']
    assets = Assets(args.blueprints, args.montages)
    if not assets.abilities:
        parser.error('No ability assets')
    if args.list_montages:
        print('\n'.join(requested_montages(assets, catalog)))
        return
    if not assets.montages:
        parser.error('No animation assets')
    tables, missing = build_tables(assets, catalog, read_json(args.curated))
    if missing:
        parser.error('Missing core combat bindings: ' + ', '.join(missing))
    write_snapshot(args.output, tables)
    action_count = sum(len(table['rows']) for skills in tables.values() for groups in skills.values()
                       for table in groups.values())
    print(f'{len(tables)} characters, {action_count} action rows')


if __name__ == '__main__':
    main()
