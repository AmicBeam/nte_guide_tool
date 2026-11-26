#!/usr/bin/env python3
"""NTE combat reference CLI. Python standard library only; no Flask imports."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CLI_VERSION = '1.0.0'
KINDS = ('characters', 'arcs', 'foods', 'permanent', 'dungeons', 'challenges', 'endless', 'diyboss', 'beginner', 'articles')

SHAFT_NODE_RUNNER = """
const fs = require('node:fs');
const vm = require('node:vm');
const {engine, self_check, axis, catalog} = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = vm.createContext({input: JSON.stringify({axis, catalog})}, {
    codeGeneration: {strings: false, wasm: false},
});
vm.runInContext(engine, context, {filename: 'shaft_engine.js'});
vm.runInContext(self_check, context, {filename: 'shaft_self_check.js'});
const output = vm.runInContext(
    "const {axis, catalog} = JSON.parse(input);"
    + "const result = ShaftEngine.simulateAxis(axis, catalog);"
    + "JSON.stringify({result, self_check: ShaftSelfCheck.inspectAxis(axis, catalog, result)});",
    context,
);
process.stdout.write(output);
"""


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('个人数据请求不接受重定向，请核对站点地址。')


class ShaftCliError(ValueError):
    pass


def add_shaft_commands(commands):
    shaft = commands.add_parser(
        'shaft', help='排轴模拟器，需先登录并设置 NTE_GUIDE_TOKEN',
        epilog='Skill ZIP on the selected site: /guide/skills/shaft-axis-workflow.zip.',
    )
    operations = shaft.add_subparsers(dest='shaft_command', required=True)
    operations.add_parser('catalog', help='读取当前账号的排轴目录')
    axes = operations.add_parser('axes', help='列出私有轴、收藏或公开轴')
    axes.add_argument('--scope', choices=('mine', 'favorites', 'public'), default='mine')
    axes.add_argument('--page', type=int, choices=range(1, 201), metavar='1–200', default=1)
    axis = operations.add_parser('axis', help='读取已保存的轴及结果')
    axis.add_argument('id', type=int)
    shared = operations.add_parser('shared', help='读取同站分享链接')
    shared.add_argument('link')
    simulate = operations.add_parser('simulate', help='使用目标站点的 JS 和本机 Node.js 复算及自检')
    simulate.add_argument('file', type=Path)
    create = operations.add_parser('create', help='从 simulate 的输出创建私有轴')
    create.add_argument('file', type=Path)
    create.add_argument('--title', required=True)
    update = operations.add_parser('update', help='从 simulate 的输出更新本人私有轴')
    update.add_argument('id', type=int)
    update.add_argument('file', type=Path)


def shaft_share_token(link, base):
    share = urllib.parse.urlsplit(link)
    if share.scheme or share.netloc:
        target = urllib.parse.urlsplit(base)
        if (share.scheme, share.netloc) != (target.scheme, target.netloc):
            raise ShaftCliError('分享链接与当前站点不同，请先确认站点。')
    if share.path != '/shaft/rotation':
        raise ShaftCliError('请提供 /shaft/rotation?share=… 分享链接。')
    tokens = urllib.parse.parse_qs(share.query).get('share', [])
    if len(tokens) != 1 or not re.fullmatch(r'[A-Za-z0-9_-]{20,64}', tokens[0]):
        raise ShaftCliError('排轴分享链接无效。')
    return tokens[0]


def shaft_axis_digest(axis):
    data = json.dumps(axis, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def simulate_shaft(document, axis, runtime, timeout):
    node = shutil.which('node')
    if not node:
        raise ShaftCliError('本地未找到 Node.js，复算需要 Node.js。')
    catalog = runtime['catalog']
    allowed = {character['id'] for character in catalog['characters'] if not character.get('selection_disabled')}
    team = axis.get('team')
    if not isinstance(team, list) or any(
        not isinstance(member, dict) or member.get('character_id') not in allowed
        for member in team
    ):
        raise ShaftCliError('队伍配置无效，或包含当前账号无权使用的角色。')
    env = {key: value for key, value in os.environ.items()
           if key not in ('NTE_GUIDE_TOKEN', 'NODE_OPTIONS', 'NODE_PATH')}
    try:
        result = subprocess.run(
            [node, '-e', SHAFT_NODE_RUNNER], env=env, timeout=timeout, check=True,
            input=json.dumps({
                'axis': axis, 'catalog': catalog, 'engine': runtime['engine'],
                'self_check': runtime['self_check'],
            }, ensure_ascii=False).encode('utf-8'),
            capture_output=True,
        )
        calculated = json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        raise ShaftCliError('本地排轴复算超时，可增大 --timeout 后重试。') from None
    except (subprocess.CalledProcessError, ValueError):
        raise ShaftCliError('排轴 JS 复算失败，请核对轴 JSON 和目标站点版本。') from None
    return {**document, 'axis': axis, **calculated, 'axis_sha256': shaft_axis_digest(axis)}


def run_shaft(args, base, token):
    if not token:
        raise ShaftCliError('请先登录排轴站，并通过 NTE_GUIDE_TOKEN 提供本站登录凭据。')
    opener = urllib.request.build_opener(NoRedirects())

    def request(path, method='GET', payload=None):
        headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/json'}
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
        with opener.open(req, timeout=args.timeout) as response:
            return json.load(response)

    request('/api/me')
    command = args.shaft_command
    axis_id = getattr(args, 'id', None)
    if axis_id is not None and axis_id <= 0:
        raise ShaftCliError('排轴 ID 必须为正整数。')
    if command == 'catalog':
        return request('/api/shaft/catalog')
    if command == 'axes':
        query = urllib.parse.urlencode({'scope': args.scope, 'page': args.page})
        return request('/api/guide/axes?' + query)
    if command == 'axis':
        return request(f'/api/shaft/axes/{axis_id}')
    if command == 'shared':
        return request('/api/shaft/shared/' + shaft_share_token(args.link, base))

    with args.file.open(encoding='utf-8') as file:
        document = json.load(file)
    if not isinstance(document, dict):
        raise ShaftCliError('排轴文件须为 JSON 对象。')
    axis = document.get('axis', document)
    if not isinstance(axis, dict) or not isinstance(axis.get('steps'), list):
        raise ShaftCliError('排轴文件必须包含 axis.steps。')
    if command == 'simulate':
        metadata = document if 'axis' in document else {}
        runtime = request('/api/shaft/cli-runtime')
        return simulate_shaft(metadata, axis, runtime, args.timeout)

    result = document.get('result')
    if not isinstance(result, dict) or document.get('axis_sha256') != shaft_axis_digest(axis):
        raise ShaftCliError('保存前请对当前轴运行 shaft simulate，并使用其输出。')
    payload = {'axis': axis, 'result': result, 'description': document.get('description')}
    if command == 'create':
        if not args.title.strip():
            raise ShaftCliError('请提供新私有轴的标题。')
        payload['title'] = args.title
        saved = request('/api/shaft/axes', 'POST', payload)
    else:
        if (document.get('id') != axis_id or not document.get('updated_at')
                or not document.get('is_owner') or document.get('visibility') != 'private'):
            raise ShaftCliError('更新需要先读取指定的本人私有轴，并保留其 ID 和版本。')
        payload.update(title=document.get('title'), expected_updated_at=document['updated_at'])
        saved = request(f'/api/shaft/axes/{axis_id}', 'PUT', payload)
    saved_id = saved['id']
    try:
        verified = request(f'/api/shaft/axes/{saved_id}')
        if (verified.get('id') != saved_id or not verified.get('is_owner')
                or verified.get('visibility') != 'private'
                or verified.get('updated_at') != saved.get('updated_at')
                or verified.get('summary') != saved.get('summary')):
            raise ValueError('readback mismatch')
    except (OSError, ValueError):
        raise ShaftCliError(f'轴 {saved_id} 已写入，但回读校验未完成，请先重新读取该轴。') from None
    return verified


def filter_profiles(payload, role_id=None, character_id=None):
    if (not isinstance(payload, dict) or not isinstance(payload.get('data'), list)
            or not all(isinstance(profile, dict) and isinstance(profile.get('characters'), list)
                       and all(isinstance(character, dict) for character in profile['characters'])
                       for profile in payload['data'])):
        raise ValueError('个人资料返回格式错误')
    profiles = payload['data']
    if role_id:
        profiles = [profile for profile in profiles if str(profile.get('role_id')) == str(role_id)]
        if not profiles:
            raise ValueError('未找到此账号的已同步资料')
    if character_id:
        profiles = [{**profile, 'characters': [character for character in profile.get('characters', [])
                     if str(character.get('id')) == str(character_id)]} for profile in profiles]
        profiles = [profile for profile in profiles if profile['characters']]
        if not profiles:
            raise ValueError('未找到此角色的已同步资料')
    return {**payload, 'data': profiles}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='Skill ZIPs on the selected site: /guide/skills/nte-wiki-cli.zip and '
               '/guide/skills/shaft-axis-workflow.zip.',
    )
    parser.add_argument('--base-url', default=os.environ.get('NTE_GUIDE_URL', 'http://127.0.0.1:5001'))
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--version', action='version', version=f'nte-guide {CLI_VERSION}')
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list', help='List a collection')
    listing.add_argument('kind', choices=KINDS)
    searching = commands.add_parser('search', help='Search all combat data')
    searching.add_argument('query')
    for cmd in (listing, searching):
        cmd.add_argument('--limit', type=int, default=20)
        cmd.add_argument('--offset', type=int, default=0)
        cmd.add_argument('--element')
        cmd.add_argument('--rarity', type=int)
    getting = commands.add_parser('get', help='Read one item by its stable ID')
    getting.add_argument('kind', choices=KINDS)
    getting.add_argument('id')
    getting.add_argument('--section')
    getting.add_argument('--stage', type=int)
    getting.add_argument('--encounter', help='Select one dungeon encounter')
    getting.add_argument('--ability', dest='ability_id', help='Read one exact character ability ID (section=abilities)')
    levels = getting.add_mutually_exclusive_group()
    levels.add_argument('--level', type=int, choices=range(1, 14), help='Skill level (characters default: 10)')
    levels.add_argument('--all-levels', action='store_true', help='Return every character skill level')
    getting.add_argument('--hit-details', action='store_true', help='Include per-hit energy, harmony and stagger data')
    getting.add_argument('--format', choices=('json', 'markdown'), default='json')
    commands.add_parser('schema', help='Read API and numeric conventions')
    beginner = commands.add_parser('beginner', help='拉取新手入坑攻略全文与图片链接')
    beginner.add_argument('--section', choices=('intro', 'gameplay', 'currencies', 'events', 'wishes', 'closing', 'progression'))
    beginner.add_argument('--format', choices=('json', 'markdown'), default='json')
    personal = commands.add_parser('me', help='Read your saved game profiles (NTE_GUIDE_TOKEN required)')
    personal.add_argument('--role-id', help='Select one of your own game accounts')
    personal.add_argument('--character', help='Select a character ID in the saved profiles')
    axes = commands.add_parser('axes', help='Search online axes with the same visibility as the website')
    axes.add_argument('--character', action='append', default=[], help='Character ID or full name; repeat up to four times')
    axes.add_argument('--scope', choices=('public', 'mine', 'favorites'), default='public')
    axes.add_argument('--query', default='')
    axes.add_argument('--sort', choices=('new', 'dps', 'likes', 'favorites'), default='new')
    axes.add_argument('--page', type=int, default=1)
    axes.add_argument('--limit', type=int, default=10)
    axis = commands.add_parser('axis', help='Read a visible online axis and its actual actions')
    axis.add_argument('id', type=int)
    axis.add_argument('--section', choices=('all', 'summary', 'axis', 'result'), default='all')
    add_shaft_commands(commands)
    args = parser.parse_args(argv)
    base = args.base_url.rstrip('/')
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.query or parsed.fragment:
        parser.error('--base-url must be an http(s) URL without query or fragment')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    query = {}
    token = os.environ.get('NTE_GUIDE_TOKEN', '').strip()
    protected = bool(token) or args.command in ('me', 'axes', 'axis')
    needs_token = args.command == 'me' or (args.command == 'axes' and args.scope != 'public')
    if needs_token and not token:
        print('读取个人资料需要本站登录凭据 NTE_GUIDE_TOKEN；请先登录网站并同步游戏数据。', file=sys.stderr)
        return 1
    if parsed.username or parsed.password:
        parser.error('--base-url must not contain credentials')
    if args.command == 'shaft':
        try:
            print(json.dumps(run_shaft(args, base, token), ensure_ascii=False, indent=2))
            return 0
        except ShaftCliError as error:
            print(error, file=sys.stderr)
        except urllib.error.HTTPError as error:
            if error.code == 409:
                print('HTTP 409: 排轴版本或标题冲突，请重新读取目标轴或检查标题。', file=sys.stderr)
            else:
                print(f'HTTP {error.code}: 排轴请求失败，请检查本站登录及访问权限。', file=sys.stderr)
        except (OSError, ValueError, KeyError, TypeError):
            print('排轴请求失败，请检查站点地址、网络和输入 JSON。', file=sys.stderr)
        return 1
    if args.command in ('list', 'search'):
        path = '/api/guide/' + (args.kind if args.command == 'list' else 'search')
        query = {k: getattr(args, k) for k in ('limit', 'offset', 'element', 'rarity') if getattr(args, k) is not None}
        if args.command == 'search':
            query['q'] = args.query
    elif args.command == 'get':
        ident = urllib.parse.quote(args.id, safe='')
        path = f'/guide/{args.kind}/{ident}.md' if args.format == 'markdown' else f'/api/guide/{args.kind}/{ident}'
        query = {k: getattr(args, k) for k in ('section', 'stage', 'encounter', 'ability_id', 'level') if getattr(args, k) is not None}
        if args.kind == 'characters':
            query.pop('level', None)
            if not args.all_levels:
                query['skill_level'] = args.level if args.level is not None else 10
            query['hit_details'] = int(args.hit_details)
        elif args.all_levels or args.hit_details:
            parser.error('--all-levels and --hit-details are only supported for characters')
    elif args.command == 'beginner':
        path = ('/guide/beginner/getting-started.md' if args.format == 'markdown'
                else '/api/guide/beginner/getting-started')
        if args.section:
            query['section'] = args.section
    elif args.command == 'me':
        path = '/api/guide/me/profiles'
    elif args.command == 'axes':
        if len(args.character) > 4 or not 1 <= args.limit <= 20 or not 1 <= args.page <= 200:
            parser.error('axes requires at most four characters, limit 1–20 and page 1–200')
        path = '/api/guide/axes'
        query = {'character': args.character, 'scope': args.scope, 'q': args.query,
                 'sort': args.sort, 'page': args.page, 'limit': args.limit}
    elif args.command == 'axis':
        if args.id <= 0:
            parser.error('axis ID must be positive')
        path = f'/api/guide/axes/{args.id}'
        query = {'section': args.section}
    else:
        path = '/api/guide/schema'
    url = base + path + ('?' + urllib.parse.urlencode(query, doseq=True) if query else '')
    try:
        if token:
            request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token})
            response_context = urllib.request.build_opener(NoRedirects()).open(request, timeout=args.timeout)
        else:
            response_context = urllib.request.urlopen(url, timeout=args.timeout)
        with response_context as response:
            content = response.read().decode('utf-8')
            if args.command == 'me' or response.headers.get_content_type() == 'application/json':
                payload = json.loads(content)
                if args.command == 'me':
                    payload = filter_profiles(payload, args.role_id, args.character)
                content = json.dumps(payload, ensure_ascii=False, indent=2)
        print(content.rstrip())
        return 0
    except urllib.error.HTTPError as error:
        if protected:
            print(f'HTTP {error.code}: 无法读取资料，请检查本站登录及访问权限。', file=sys.stderr)
        else:
            print(f'HTTP {error.code}: {error.read().decode("utf-8", errors="replace")}', file=sys.stderr)
    except (OSError, ValueError) as error:
        message = '资料读取失败，请检查站点地址、网络及登录状态。' if protected else f'查询失败: {error}'
        print(message, file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
