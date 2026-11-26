"""Fixed action timings for tests of timeline mechanics and boundary conditions."""
from functools import wraps
from unittest.mock import patch

from app.modules.shaft.domain import catalog


SCENARIO_TICKS = {
    '主角': {'援护': 14, 'e': 15},
    '真红': {'援护': 15, 'a1': 3, 'a2': 5, 'a3': 10, 'a4': 7, 'a5': 10, '龙a1': 3, '龙a2': 7, '龙a3': 15, '龙a4': 8, 'e': 10, '龙e': 13},
    '浔': {'援护': 13},
    '小吱': {'援护': 13, '闪反': 13, 'e1': 7, 'e3': 15},
    '埃德嘉': {'援护': 13},
    '娜娜莉': {'援护': 13, 'a2': 3, 'a5': 13, '闪反': 21, 'e': 11},
    '九原': {'援护': 15, 'e': 15},
    '薄荷': {'援护': 15, '闪反': 20, 'e': 7},
    '白藏': {'援护': 15, '长闪': 7, '闪反': 20},
    '早雾': {'援护': 20, 'e': 20, '长e': 20},
    '阿德勒': {'e': 15},
    '安魂曲': {'援护': 15, 'a1远': 4, 'a2远': 4, 'a4远': 12, 'a5远': 20, '切换': 13, 'e': 8},
    '达芙蒂尔': {'援护': 15, '移行': 10, 'e': 15, '1层e': 15, '2层e': 15},
    '法帝娅': {'援护': 15, 'e': 15},
    '哈尼娅': {'e': 15},
    '海月': {'援护': 15, '闪反': 20, 'e': 13},
    '卡厄斯': {'a2': 6, 'a3': 5, 'a4': 10, '闪反': 12, 'z2': 13, 'e': 15},
    '哈索尔': {'长e尾刀': 22, 'E1': 13},
    '翳': {'援护': 15, 'e': 15},
    '伊洛伊': {'a1': 3, 'a2': 7, '清明梦': 70, 'e': 4},
    '残虹': {'a1': 6, 'a2': 6, 'a3': 8, 'a4': 11, 'a5': 12, 'z': 10, '幻a2': 7, '幻a3': 9, '幻a4': 11, '幻z': 11, '闪反': 10, 'e': 10, '强化e': 10, '幻e': 22},
    '灵可': {'援护': 15},
    '黑羽': {'a1': 4, 'a2': 6, 'a3': 8, 'a4': 10, 'a5': 12, 'z': 10, '闪反': 12, '魔a1': 7, '魔a2': 6, '魔a3': 10, '魔a4': 6, '魔a5': 7, '魔z1': 10, '魔闪反': 10, 'e': 12, '魔e': 10, '魔援护': 15},
    '明音凛': {'a1': 5, 'a3': 10, 'a4': 8, '援护': 15, 'e': 5, '长e': 10, '长强E': 15},
    '曳尔沐': {'a1': 4, 'a2': 6, 'a3': 8, 'a4': 8, 'a5': 10, '强a1': 6, '强a2': 8, '强a3': 6, '重击': 10, '闪反': 10, 'e1': 10, '援护': 15},
    'Exe': {'a1': 6, 'a2': 6, 'a5': 8, '闪反': 10, 'e': 12, '援护': 15},
}


def use_scenario_ticks(actions):
    for action in actions:
        ticks = SCENARIO_TICKS.get(action['character_name'], {}).get(action['name'])
        if ticks is not None:
            action.update(duration_ticks=ticks, duration_seconds=ticks / 10)
    return actions


def with_scenario_ticks(test):
    @wraps(test)
    def run(*args, **kwargs):
        load_json = catalog._load_json

        def load_fixture(filename):
            data = load_json(filename)
            return use_scenario_ticks(data) if filename == 'actions.json' else data

        catalog.load_shaft_catalog.cache_clear()
        try:
            with patch.object(catalog, '_load_json', load_fixture):
                return test(*args, **kwargs)
        finally:
            catalog.load_shaft_catalog.cache_clear()

    return run
