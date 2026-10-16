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

KINDS = ('characters', 'arcs', 'challenges')


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
    getting.add_argument('--format', choices=('json', 'markdown'), default='json')
    commands.add_parser('schema', help='Read API and numeric conventions')
    args = parser.parse_args(argv)
    base = args.base_url.rstrip('/')
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.query or parsed.fragment:
        parser.error('--base-url must be an http(s) URL without query or fragment')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    query = {}
    if args.command in ('list', 'search'):
        path = '/api/guide/' + (args.kind if args.command == 'list' else 'search')
        query = {k: getattr(args, k) for k in ('limit', 'offset', 'element', 'rarity') if getattr(args, k) is not None}
        if args.command == 'search':
            query['q'] = args.query
    elif args.command == 'get':
        ident = urllib.parse.quote(args.id, safe='')
        path = f'/guide/{args.kind}/{ident}.md' if args.format == 'markdown' else f'/api/guide/{args.kind}/{ident}'
        query = {k: getattr(args, k) for k in ('section', 'stage') if getattr(args, k) is not None}
    else:
        path = '/api/guide/schema'
    url = base + path + ('?' + urllib.parse.urlencode(query) if query else '')
    try:
        with urllib.request.urlopen(url, timeout=args.timeout) as response:
            content = response.read().decode('utf-8')
            if response.headers.get_content_type() == 'application/json':
                content = json.dumps(json.loads(content), ensure_ascii=False, indent=2)
        print(content.rstrip())
        return 0
    except urllib.error.HTTPError as error:
        print(f'HTTP {error.code}: {error.read().decode("utf-8", errors="replace")}', file=sys.stderr)
    except (OSError, ValueError) as error:
        print(f'查询失败: {error}', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
