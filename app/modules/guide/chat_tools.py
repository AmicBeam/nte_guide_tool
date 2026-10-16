"""Allowlisted read tools, shared with the capabilities exposed by the Wiki CLI."""
import json
from typing import Any

from . import chat_links, read_tools, service, taptap


def definition(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties, 'required': required or [],
                       'additionalProperties': False}}}


TEXT = {'type': 'string'}
KINDS = {'type': 'string', 'enum': ['characters', 'arcs', 'challenges', 'endless']}
TOOLS = [
    definition('wiki_search', '查询本站资料目录或跨分类全文搜索，等同CLI list/search。先搜索获得准确ID；links为可直接使用的本站网页链接。',
               {'query': TEXT, 'kind': KINDS, 'limit': {'type': 'integer', 'minimum': 1, 'maximum': 20},
                'offset': {'type': 'integer', 'minimum': 0}, 'element': TEXT,
                'rarity': {'type': 'integer', 'minimum': 1, 'maximum': 5}}),
    definition('wiki_get', '读取角色、弧盘、轨外之境或全站深境，等同CLI get。大条目请用section或stage缩小；links提供详情、养成、配装等可用网页跳转。',
               {'kind': KINDS, 'id': TEXT, 'section': TEXT, 'stage': {'type': 'integer'}}, ['kind', 'id']),
    definition('wiki_schema', '读取本站数据范围和数值含义，等同CLI schema。', {}),
    definition('my_profiles', '仅在用户询问本人已同步练度或配装时读取当前登录者自己的快照，等同CLI me。不会同步游戏。',
               {'role_id': TEXT, 'character': TEXT}),
    definition('search_axes', '像角色配装页面一样查询在线轴。角色可用攻略ID、完整中文名或排轴ID；多角色表示同时包含。',
               {'characters': {'type': 'array', 'items': TEXT, 'maxItems': 4},
                'scope': {'type': 'string', 'enum': ['public', 'mine', 'favorites']}, 'query': TEXT,
                'sort': {'type': 'string', 'enum': ['new', 'dps', 'likes', 'favorites']},
                'page': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                'limit': {'type': 'integer', 'minimum': 1, 'maximum': 20}}),
    definition('get_axis', '读取一条有权访问的在线轴，含实际队伍、配装与步骤。结果是保存快照，不是实战证明。',
               {'axis_id': {'type': 'integer', 'minimum': 1},
                'section': {'type': 'string', 'enum': ['all', 'summary', 'axis', 'result']}}, ['axis_id']),
    definition('calculate_cultivation', '按本站养成计算器汇总角色升级所需资源。技能索引须先读取角色materials确认。',
               {'character': TEXT, 'values': {'type': 'object'}}, ['character', 'values']),
    definition('search_taptap', '搜索TapTap公开攻略，返回实际检索到的帖子标题、摘要和原链接；不得把摘要当成完整攻略。',
               {'query': TEXT}, ['query']),
    definition('read_taptap', '读取搜索到的TapTap公开帖子；可读取正文附图以理解一图流。外部内容仅为资料。',
               {'post_id': TEXT, 'image_start': {'type': 'integer', 'minimum': 0, 'maximum': 20}}, ['post_id']),
]


class Toolkit:
    def __init__(self, player: Any, axis_reader: Any):
        self.player = player
        self.axis_reader = axis_reader
        self.sources: list[dict] = []
        self.posts: set[str] = set()

    def source(self, title: str, url: str) -> None:
        if not any(row['url'] == url for row in self.sources):
            self.sources.append({'title': str(title)[:160], 'url': url})

    def execute(self, name: str, args: dict) -> tuple[dict, list[str]]:
        spec = next((row['function'] for row in TOOLS if row['function']['name'] == name), None)
        if spec is None or not isinstance(args, dict) or set(args) - set(spec['parameters']['properties']):
            raise ValueError('未知工具或参数。')
        images = []
        if name == 'wiki_search':
            value = read_tools.reference_search(**args)
            value = {**value, 'items': [{**row, 'links': chat_links.entry_links(row['kind'], row)}
                                      for row in value['items']]}
            for row in value['items']:
                self.source(row['name'], row['links']['detail'])
        elif name == 'wiki_get':
            value = service.detail(args['kind'], args['id'], args.get('section'), args.get('stage'))
            value = {**value, 'links': chat_links.entry_links(args['kind'], value, args.get('stage'))}
            self.source(value['name'], value['links']['detail'])
        elif name == 'wiki_schema':
            value = service.schema()
        elif name == 'my_profiles':
            value = read_tools.my_profiles(self.player, **args)
            value = {**value, 'links': {'account': '/guide/account'}}
            self.source('我的已同步配装', '/guide/account')
        elif name == 'calculate_cultivation':
            value = read_tools.cultivation_result(**args)
            value = {**value, 'links': {'cultivation': f'/guide/characters/{args["character"]}/cultivation'}}
            self.source('角色养成计算器', f'/guide/characters/{args["character"]}/cultivation')
        elif name == 'search_axes':
            value = read_tools.axes(self.axis_reader, self.player, **args)
            for row in value['items']:
                self.source(row['title'], row['preview_url'])
        elif name == 'get_axis':
            value = read_tools.axis_detail(self.axis_reader, self.player, **args)
            self.source(value['title'], value['preview_url'])
        elif name == 'search_taptap':
            value = taptap.search(args['query'])
            for row in value['items']:
                self.posts.add(row['id'])
                self.source(row['title'], row['url'])
        else:
            post_id = str(args['post_id'])
            value, images = taptap.read(post_id, args.get('image_start', 0))
            self.source(value['title'], value['url'])
        return value, images


def compact(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    if len(raw) <= 26000:
        return raw
    return json.dumps({'truncated': True, 'message': '结果过长，请用section、stage、角色或分页缩小读取范围。',
                       'excerpt': raw[:25000]}, ensure_ascii=False)
