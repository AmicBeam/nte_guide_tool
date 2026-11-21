"""Build Guide combat tables from exported UE ability and animation assets."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from guide_skill_details import LACRIMOSA_NORMAL_ATTACKS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'app/modules/guide/data'
TYPES = ('Melee', 'Skill', 'UltraSkill', 'QTE', 'Peculiarity')
COLUMNS = ('动作', '连段窗口（秒）', '动作占时（秒）', '伤害通知（秒）', '命中顿帧（秒）',
           '切人锁定（秒）', '解除暂停事件（秒）', '伤害判定')
MELEE = re.compile(r'(Ultra|Magic)?Melee([A-Z]?)(\d+)(.*)', re.I)
ADDITIONAL_ABILITIES = {
    'ga_lacrimosa_melee': ('ga_lacrimosa_meleemoda', 'ga_lacrimosa_switchskill', 'ga_lacrimosa_switchskillmodb'),
    'ga_lacrimosa_ultraskill': ('ga_lacrimosa_ultraskillmoda',),
    'ga_hathor_skill': ('ga_hathor_skill2',),
    'ga_zankou_melee': ('ga_zankou_meleemagic',),
    'ga_zankou_ultraskill': ('ga_zankou_ultraskillforce', 'ga_zankou_ultraskillmagic'),
    'ga_blackbird_melee': ('ga_blackbird_evolve_melee',),
    'ga_blackbird_skill': ('ga_blackbird_evolve_skill',),
    'ga_blackbird_ultraskill': ('ga_blackbird_evolve_ultraskill',),
    'ga_daffodill_melee': ('ga_daffodill_evade',),
    'ga_mitsuki_melee': ('ga_mitsuki_evade',),
    'ga_elyms_melee': ('ga_elyms_meleeb',),
    'ga_elyms_ultraskill': ('ga_elyms_ultraskillb',),
}
MONTAGE_GROUPS = {
    ('ga_sagiri_skill', 'Skill1'): '变轨技能：打嗝',
    ('ga_haniel_melee', 'End'): '普通攻击：超异科魔法炮',
    ('ga_cang_melee', 'Branch1'): '普通攻击：无状之言',
    ('ga_cang_melee', 'Branch2'): '普通攻击：无讳之语',
    ('ga_hathor_melee', 'AirAttackEnd'): '普通攻击：压轴派送',
    ('ga_hathor_melee', 'Branch'): '普通攻击：制高清算',
    ('ga_hathor_melee', 'BranchEnd'): '普通攻击：制高清算',
    ('ga_adler_melee', 'Branch'): '蓄力攻击',
    ('ga_zankou_meleemagic', 'MagicMelee1'): '普通攻击：魇舞',
    ('ga_zankou_meleemagic', 'MagicMeleePre'): '普通攻击：魇舞',
    ('ga_jin_melee', 'Branch1'): '普通攻击：漪澜',
    ('ga_daffodill_melee', 'Branch1'): '普通攻击：临水照影',
    ('ga_daffodill_evade', 'AttackEvade'): '普通攻击：错位锋芒',
    ('ga_daffodill_skill', 'Skill1'): '变轨技能：同振',
    ('ga_daffodill_skill', 'Skill2'): '变轨技能：如响斯应',
    ('ga_mitsuki_evade', 'EvadeBranchEntry'): '普通攻击：顿音',
    ('ga_mitsuki_evade', 'End'): '普通攻击：顿音',
    ('ga_chaos071_melee', 'Branch1'): '普通攻击：裁断',
    ('ga_chaos071_melee', 'Branch2'): '普通攻击：裁断',
    ('ga_chaos071_melee', 'Branch1_Ultra'): '普通攻击：裁断',
    ('ga_chaos071_melee', 'Branch2_Ultra'): '普通攻击：裁断',
    ('ga_oneiroi_perfectevade', 'Atk'): '无意识响应',
    ('ga_shinku_melee', 'AirAttackEnd'): '普通攻击：空箭',
    ('ga_shinku_melee', 'Branch1'): '普通攻击：特种格斗术—变式',
    ('ga_shinku_melee_rage', 'AirAttackEnd'): '普通攻击：落锤',
    ('ga_shinku_melee_rage', 'Branch'): '普通攻击：惊骇急袭—轻蔑',
    ('ga_shinku_skill', 'Skill'): '变轨技能：高速突破',
    ('ga_exe_melee', 'Branch'): '普通攻击：纵深侵入',
}
MONTAGE_LABELS = {
    ('ga_lacrimosa_ultraskill', 'UltraSkillB'): '近战终结',
    ('ga_lacrimosa_ultraskillmoda', 'UltraSkill'): '远程终结',
    ('ga_elyms_ultraskill', 'UltraSkill'): '对群终结',
    ('ga_elyms_ultraskillb', 'UltraSkill'): '对单终结',
    ('ga_elyms_meleeb', 'MeleeB1'): '第 1 段',
    ('ga_elyms_meleeb', 'MeleeB2'): '第 2 段',
    ('ga_elyms_meleeb', 'MeleeB3'): '第 3 段',
    ('ga_hathor_melee', 'Branch'): '环绕起手',
    ('ga_hathor_melee', 'BranchEnd'): '着地攻击',
    ('ga_haniel_melee', 'End'): '蓄力炮',
    ('ga_chaos071_melee', 'Branch1_Ultra'): '强化第 1 段',
    ('ga_chaos071_melee', 'Branch2_Ultra'): '强化第 2 段',
    ('ga_mitsuki_evade', 'EvadeBranchEntry'): '冲刺起手',
    ('ga_daffodill_evade', 'AttackEvade'): '特殊闪避',
}
LACRIMOSA_SWITCH_MOVES = {'SwitchModA': '近切远', 'SwitchModB': '远切近'}
COUNTER_SPAWN_PAUSES = {
    'GA_Exe_Melee': 0.066666,
    'GA_Lacrimosa_Melee': 0.3,
    'GA_Mitsuki_Melee': 0.3,
}
LABELS = {
    'MeleePre': '接近起手', 'AppearMelee': '入场攻击', 'AirAttack': '下落起始', 'AirAttackEnd': '着地攻击',
    'AirAttackLoop': '下落循环', 'Atk': '反击攻击', 'PerfectEvadeBack': '后向极限闪避',
    'PerfectEvadeFront': '前向极限闪避', 'Skill': '变轨技能', 'Default': '技能起手', 'QTE': '援护攻击',
    'Skillhold': '长按', 'Skill_Short': '强化短按', 'Skill_Long': '强化长按', 'UltraSkill': '极轨终结',
    'Fire': '射击', 'Fire_LoopAim': '瞄准射击', 'Branch': '分支攻击', 'End': '收招', 'Loop': '循环',
    'UltraSkillB': '变身终结', 'Atk_tran': '变身反击', 'QTE_tran': '变身援护',
    'MagicMeleePre': '接近起手', 'MeleeBPre': '接近起手', 'AppearMeleeB': '入场攻击',
    'Skill2_1': '第 1 段', 'Skill2_2': '第 2 段', 'Skill2_3': '第 3 段',
    'UltraSkillForce': '强化终结', 'UltraskillMagic': '幻境终结',
    **LACRIMOSA_SWITCH_MOVES,
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
        prefix = '强化' if (enhanced or '').casefold() == 'ultra' else ''
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
    def __init__(self, blueprints, montages=None, damage_triggers=None):
        self.blueprints = {p.relative_to(blueprints).with_suffix('').as_posix().casefold(): p
                           for p in blueprints.rglob('*.json')}
        self.montages = {}
        if montages is not None:
            self.montages = {path.relative_to(montages).with_suffix('').as_posix().casefold().removeprefix('ht/content/'): path
                             for path in montages.rglob('*.json')}
        self.abilities = {p.stem.casefold(): k for k, p in self.blueprints.items()
                          if k.startswith('blueprints/abilities/player/') and p.stem.startswith('GA_')}
        self.damage_triggers = damage_triggers or {}

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
            return '盒体半尺寸 ' + vec(p['HalfSize'], ' × ')
        if 'SphereTrace' in native and p.get('SphereRadius'):
            return '球形半径 ' + fmt(p['SphereRadius'])
        if 'CapsuleTrace' in native and p.get('CapsuleRadius') and p.get('CapsuleHalfHeight'):
            return f"胶囊半径 {fmt(p['CapsuleRadius'])}、半高 {fmt(p['CapsuleHalfHeight'])}"
        return ''

    def attack_hit_pause(self, effect):
        pause = self.defaults(game_path(effect))[0].get('HitPauseData') or {}
        seconds = pause.get('AttackDilationTime')
        rate = pause.get('AttackDilationRate', 0)
        return seconds if isinstance(seconds, (int, float)) and seconds > 0 and rate < 1 else None

    def fixed_damage_offsets(self, container):
        trigger = self.damage_triggers.get(container.get('damage_trigger'), {})
        count = trigger.get('MaxTriggerCount')
        period = trigger.get('Period', 0)
        if (not trigger.get('TriggerOnSpawn') or not isinstance(count, int) or count <= 0
                or (count > 1 and period <= 0)):
            return ()
        spawn = trigger.get('DelaySpawnTime', 0)
        first_hit = trigger.get('DelayTriggerTime', 0)
        life = trigger.get('duration', 0)
        return tuple(spawn + first_hit + i * period for i in range(count)
                     if not life or first_hit + i * period <= life)

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
                    if (actor.get('Type') == 'HTAbilitySpawnDamageTriggerActor'
                            and not value.get('SpawnConditionObjs') and not value.get('RandomCountCurve')):
                        container = {**container, 'damage_trigger': value.get('DamageTriggerConfigID')}
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


def with_hit_pause(cutoff, hit_pause_events):
    return cutoff + sum(duration for time, duration in hit_pause_events if time <= cutoff)


def section_window(montage, key, *, include_battle_continuation=False):
    sections = montage.get('CompositeSections') or []
    if len(sections) == 1:
        return (notify_time(sections[0]) or 0), montage['SequenceLength']
    section = next((entry for entry in sections if entry['SectionName'].casefold() == key.casefold()), {})
    bounds = notify_time(section)
    if bounds is None:
        return None
    continuation = section.get('NextSectionName') if include_battle_continuation else None
    times = (notify_time(entry) for entry in sections
             if not (continuation == 'CR_Battle' and entry['SectionName'] == continuation))
    return bounds, min((t for t in times if t is not None and t > bounds), default=montage['SequenceLength'])


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


def evade_entry_times(assets, bindings, attack):
    suffix = attack.removeprefix('Atk')
    entries = []
    for direction, label in (('Front', '前闪反击'), ('Back', '后闪反击')):
        key = 'PerfectEvade' + direction + suffix
        exports = assets.animation(bindings.get(key, ''))
        montage = next((props(o) for o in exports if o['Type'] == 'AnimMontage'), None)
        if not montage or not (window := section_window(montage, key)):
            continue
        start, end = window
        names = {o['Name']: o for o in exports}
        rate = montage.get('CommonTargetFrameRate') or {}
        fps = rate.get('Numerator', 0) / (rate.get('Denominator') or 1)
        if not fps:
            continue
        times = []
        for notify in montage.get('Notifies') or []:
            time = notify_time(notify)
            if time is None or not start <= time < end:
                continue
            config = props(note_export(notify, names))
            frame = config.get('nFrame')
            if {attack, 'Atk'}.intersection(config.get('JumpSections') or []) and isinstance(frame, int):
                times.append(time - start + frame / fps + config.get('fDelayTriggerTime', 0))
        if times:
            entries.append((label, min(times)))
    return entries


def inspect_montage(assets, exports, key, ability, definition, action_label=None, *, include_battle_continuation=False):
    montage = next((props(o) for o in exports if o['Type'] == 'AnimMontage'), None)
    if not montage or not (window := section_window(
        montage, key, include_battle_continuation=include_battle_continuation
    )):
        return None
    start, end = window
    names = {o['Name']: o for o in exports}
    rate = montage.get('CommonTargetFrameRate') or {}
    fps = rate.get('Numerator', 0) / (rate.get('Denominator') or 1)
    times = []
    windows = []
    predictions = []
    cancel = []
    ability_end = []
    locks = []
    pauses = []
    ranges = []
    hit_pauses = Counter()
    hit_pause_events = []
    spawn_pause_events = set()
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
                predictions.append(relative + frame / fps)
        elif kind.startswith('BP_CanBreakSkill') and any(t.endswith('BreakSkill_AnySkill') for t in p.get('BreakSkillTypes',[])):
            cancel.append(relative)
        elif kind.startswith('BP_TriggerEndAbilityEffect'):
            ability_end.append(relative)
        elif 'CantChangePlayer' in kind and end_time is not None:
            locks.append(f'{fmt(relative)}–{fmt(min(end_time, end) - start)}')
        elif kind.startswith('BP_SendGamePlayEvent'):
            tag = (p.get('TriggerEventTag') or {}).get('TagName','')
            if tag in definition['pause_end']:
                pauses.append(fmt(relative))
            hit_pauses_at_event = set()
            containers = definition['hits'].get(tag, [])
            direct_at_event = {game_path(ref) for container, spawned in containers if not spawned
                               for ref in container.get('TargetGameplayEffectClasses', [])}
            for container, is_spawn in containers:
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
                    ranges.append(geometry)
                elif collision := container.get('projectile_collision'):
                    ranges.append(collision)
                if is_spawn:
                    offsets = assets.fixed_damage_offsets(container)
                    for effect in damage:
                        if game_path(effect) not in direct_at_event and (seconds := assets.attack_hit_pause(effect)):
                            spawn_pause_events.update((relative + offset, seconds)
                                                      for offset in offsets)
                else:
                    times.append(fmt(relative))
                    for effect in damage:
                        if seconds := assets.attack_hit_pause(effect):
                            hit_pauses_at_event.add(seconds)
            hit_pauses.update(hit_pauses_at_event)
            if hit_pauses_at_event:
                hit_pause_events.append((relative, sum(hit_pauses_at_event)))
    threshold = definition['followup'].get(key)
    if threshold is not None and not start <= threshold < end:
        threshold = None
    style_switch = ability.get('id') == 'GA_Lacrimosa_Melee' and key in LACRIMOSA_SWITCH_MOVES
    followup = threshold - start if threshold is not None else min(cancel or ability_end, default=None)
    endpoints = predictions if predictions and not style_switch else ([] if followup is None else [followup])
    counter_cutoff = min(cancel, default=None) if key in ('Atk', 'Atk_tran') else None
    cutoff = counter_cutoff if counter_cutoff is not None else max(endpoints, default=None)
    if cutoff is not None:
        fixed_spawns = sorted((time, pause) for time, pause in spawn_pause_events if time <= cutoff)
        hit_pause_events.extend(fixed_spawns)
        hit_pauses.update(pause for _, pause in fixed_spawns)
    occupancy = ' / '.join(order_unique(fmt(with_hit_pause(t, hit_pause_events)) for t in endpoints)) or None
    pause_column = ' / '.join(f'{fmt(value)}*{count}' if count > 1 else fmt(value)
                              for value, count in sorted(hit_pauses.items()))
    row = (
        action_label or move_label(key, ability['type_name']),
        None if style_switch else (' / '.join(order_unique(windows)) or None),
        occupancy,
        ' / '.join(order_unique(times)) or None,
        pause_column or None,
        ' / '.join(order_unique(locks)) or None,
        ' / '.join(order_unique(pauses)) or None,
        ' / '.join(order_unique(ranges)) or None,
    )
    if not any(row[1:]):
        return None
    counter_duration = None
    if counter_cutoff is not None:
        counter_duration = with_hit_pause(counter_cutoff, hit_pause_events)
    return row, direct_effects, spawn_effects, counter_duration


def related(assets, ability):
    main_id = ability['id'].casefold()
    primary = assets.abilities.get(main_id)
    if not primary:
        return []
    directory = primary.rsplit('/', 1)[0]
    terminal = main_id.removesuffix('_ultraskill') + '_subultraskill' if ability['type_name'] == 'UltraSkill' else None
    additional = ADDITIONAL_ABILITIES.get(main_id, ())
    variants = []
    for key in assets.abilities.values():
        parent, name = key.rsplit('/', 1)
        same_folder_variant = parent == directory and (
            name == terminal or name.startswith(main_id + '_')
            or (ability['type_name'] == 'Melee' and 'perfectevade' in name)
        )
        if name in additional or same_folder_variant:
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
    record = {
        skill: {name: deepcopy(table) for name, table in groups.items() if 'columns' in table}
        for skill, groups in curated.items()
        if any('columns' in table for table in groups.values())
    }
    for ability in character['abilities']:
        if ability['type_name'] not in TYPES:
            continue
        candidates = related(assets, ability)
        if not candidates:
            continue
        main = candidates[0]
        rows = defaultdict(list)
        for path in candidates:
            blueprint_name = path.rsplit('/', 1)[1]
            raw = assets.blueprint(path)
            definition = assets.defaults(path)[0]
            ranged_lacrimosa = path.endswith('/ga_lacrimosa_meleemoda')
            pause = {tag for o in raw if o['Type']=='HTGAComponent_AbilityGamePaused'
                         for tag in props(o).get('TriggerEndEventTags',[])}
            data = {'hits': assets.hit_containers(definition), 'pause_end': pause,
                    'followup': followup_thresholds(assets, definition)}
            terminal = ability['type_name'] == 'UltraSkill' and path.endswith(('_end', '_subultraskill'))
            bindings = dict(montage_bindings(definition))
            for key, animation in bindings.items():
                label = MONTAGE_LABELS.get((blueprint_name, key))
                if label is None and terminal and key == 'Default':
                    label = '终结终段'
                measured = inspect_montage(
                    assets, assets.animation(animation), key, ability, data, label,
                    include_battle_continuation=ranged_lacrimosa,
                )
                if not measured:
                    continue
                row, direct_effects, spawn_effects, counter_duration = measured
                if (
                    ability['id'] == 'GA_Lacrimosa_Melee' and (path == main or ranged_lacrimosa)
                    and (MELEE.fullmatch(key) or key in ('MeleePre', 'AppearMelee'))
                ):
                    form = '近战普攻' if path == main else '远程普攻'
                    group = LACRIMOSA_NORMAL_ATTACKS[form]
                else:
                    group = MONTAGE_GROUPS.get((blueprint_name, key)) or match_group(
                        ability, direct_effects, spawn_effects,
                        path == main or path.startswith(main + '_') or terminal, key,
                    )
                if 'perfectevade' in path and key in ('Atk', 'Atk_tran'):
                    if not group:
                        counter_groups = [g['name'] for g in ability['groups'] if g['name'].startswith('极限反击')]
                        if len(counter_groups) == 1:
                            group = counter_groups[0]
                    entries = evade_entry_times(assets, bindings, key)
                    if group and counter_duration is not None and len(entries) == 2:
                        _, _, _, hit_notifies, hit_pause, *details = row
                        fixed_pause = COUNTER_SPAWN_PAUSES.get(ability['id'], 0) if not hit_pause else 0
                        if fixed_pause:
                            hit_pause = ' / '.join(value for value in (hit_pause, fmt(fixed_pause)) if value)
                        for label, entry in entries:
                            damage_times = ' / '.join(
                                fmt(entry + float(time)) for time in (hit_notifies or '').split(' / ') if time
                            )
                            rows[group].append((label, None, fmt(entry + counter_duration + fixed_pause),
                                                damage_times or None, hit_pause, *details))
                        continue
                if group:
                    rows[group].append(row)
        for name, moves in rows.items():
            groups = record.setdefault(ability['id'], {})
            if name in groups:
                continue
            columns = [i for i in range(len(COLUMNS)) if any(row[i] is not None for row in moves)]
            groups[name] = {
                'columns': [COLUMNS[i] for i in columns],
                'rows': [[row[i] for i in columns] for row in order_unique(moves)],
                **curated.get(ability['id'], {}).get(name, {}),
            }
    return record


def build_tables(assets, characters, curated):
    result = {}
    missing = []
    for character in characters:
        identifier = str(character['id'])
        overrides = curated.get(identifier, {})
        tables = build_character(assets, character, overrides)
        for skill, groups in overrides.items():
            for name in groups.keys() - tables.get(skill, {}).keys():
                missing.append(f"{character['name']} ({skill} / {name})")
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
    parser.add_argument('--damage-trigger-table', type=Path, help='DamageTriggerActorDataTable JSON for fixed spawned hits')
    parser.add_argument('--output', type=Path, default=DATA / 'skill_combat_details.json')
    args = parser.parse_args()
    catalog = read_json(args.catalog)['characters']
    if args.montages and not args.damage_trigger_table:
        parser.error('--damage-trigger-table is required to include fixed spawned hit pauses')
    damage_triggers = None
    if args.damage_trigger_table:
        damage_triggers = next((export['Rows'] for export in read_json(args.damage_trigger_table)
                                if export.get('Name') == 'DamageTriggerActorDataTable'), None)
        if not damage_triggers:
            parser.error('Missing DamageTriggerActorDataTable rows')
    assets = Assets(args.blueprints, args.montages, damage_triggers)
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
