#!/usr/bin/env python3
"""NTE combat reference CLI. Python standard library only; no Flask imports."""
from __future__ import annotations
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

KINDS = ('characters', 'arcs', 'foods', 'permanent', 'dungeons', 'challenges', 'endless', 'diyboss', 'beginner', 'articles')


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default=os.environ.get('NTE_GUIDE_URL', 'http://127.0.0.1:5001'))
    parser.add_argument('--timeout', type=float, default=20)
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
    args = parser.parse_args(argv)
    base = args.base_url.rstrip('/')
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.query or parsed.fragment:
        parser.error('--base-url must be an http(s) URL without query or fragment')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    query = {}
    protected = args.command in ('me', 'axes', 'axis')
    token = os.environ.get('NTE_GUIDE_TOKEN', '').strip() if protected else ''
    needs_token = args.command == 'me' or (args.command == 'axes' and args.scope != 'public')
    if needs_token and not token:
        print('读取个人资料需要本站登录凭据 NTE_GUIDE_TOKEN；请先登录网站并同步游戏数据。', file=sys.stderr)
        return 1
    if parsed.username or parsed.password:
        parser.error('--base-url must not contain credentials')
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
            class NoRedirects(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    raise ValueError('个人数据请求不接受重定向，请核对站点地址。')
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
            print(f'HTTP {error.code}: 无法读取个人资料，请检查本站登录是否有效并确认已同步。', file=sys.stderr)
        else:
            print(f'HTTP {error.code}: {error.read().decode("utf-8", errors="replace")}', file=sys.stderr)
    except (OSError, ValueError) as error:
        message = '资料读取失败，请检查站点地址、网络及登录状态。' if protected else f'查询失败: {error}'
        print(message, file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
