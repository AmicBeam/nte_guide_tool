"""Character tool directory using the existing character-page destinations."""
from pathlib import Path

from flask import render_template, request

from . import presentation, service

TOOLS = {
    'cultivation': ('养成计算器', '选择目标练度，汇总升级材料与花费'),
    'builds': ('排轴与配装', '查看相关排轴，比较装备收益'),
    'kongmu': ('空幕计算', '选择角色，查看空幕搭配方案'),
    'infographic': ('制作角色一图流攻略', '自由排版、编辑配装与队伍，导出高清图片'),
}


def register(blueprint):
    @blueprint.get('/guide/tools')
    def character_tools():
        selected = request.args.get('tool', '')
        if selected and selected not in TOOLS:
            raise LookupError('未知角色工具')
        characters = []
        if selected:
            for character in service.catalog()['characters']:
                if character.get('status') == 'preview':
                    continue
                if selected == 'kongmu':
                    entry = presentation.kongmu_entry(character)
                    if not entry:
                        continue
                    url = entry['url']
                else:
                    url = f"/guide/characters/{character['id']}/{selected}"
                characters.append(dict(character, tool_url=url))
        return render_template(
            'guide/tools.html', tools=TOOLS, selected_tool=selected,
            characters=characters, tools_page=True, kind=None, kinds=service.KINDS,
            counts={key: len(service.public_records(key)) for key in service.KINDS},
            catalog_asset_version=(Path(__file__).parent / 'static/catalog_cards.css').stat().st_mtime_ns,
        )
