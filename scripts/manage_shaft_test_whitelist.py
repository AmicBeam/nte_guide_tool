import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import init_db
from app.models import Player


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='管理排轴受邀账号与测试账号权限。')
    parser.add_argument('action', choices=(
        'add', 'remove', 'status', 'list',
        'invite', 'uninvite', 'list-invited',
    ))
    parser.add_argument('player_uid', nargs='?', help='玩家账号；list 操作不需要。')
    args = parser.parse_args()
    list_actions = {'list', 'list-invited'}
    if args.action not in list_actions and not str(args.player_uid or '').strip():
        parser.error(f'{args.action} 操作需要 player_uid')
    if args.action in list_actions and args.player_uid:
        parser.error(f'{args.action} 操作不接受 player_uid')
    return args


def main() -> int:
    args = parse_args()
    init_db([Player])

    if args.action in {'list', 'list-invited'}:
        permission_field = (
            Player.shaft_invited if args.action == 'list-invited'
            else Player.shaft_test_whitelisted
        )
        players = (
            Player.select()
            .where(permission_field)
            .order_by(Player.player_uid)
        )
        for player in players:
            print(f'{player.player_uid}\t{player.nickname}')
        return 0

    player_uid = str(args.player_uid).strip()
    player = Player.get_or_none(Player.player_uid == player_uid)
    if player is None:
        print(f'账号不存在：{player_uid}', file=sys.stderr)
        return 1

    if args.action == 'status':
        if player.shaft_test_whitelisted:
            status = 'test'
        elif player.shaft_invited:
            status = 'invited'
        else:
            status = 'public'
        print(f'{player.player_uid}\t{status}')
        return 0

    if args.action in {'invite', 'uninvite'}:
        player.shaft_invited = args.action == 'invite'
        player.save(only=[Player.shaft_invited])
        status = '已设为' if player.shaft_invited else '已取消'
        print(f'{status}排轴受邀用户：{player.player_uid}')
    else:
        player.shaft_test_whitelisted = args.action == 'add'
        player.save(only=[Player.shaft_test_whitelisted])
        status = '已加入' if player.shaft_test_whitelisted else '已移出'
        print(f'{status}排轴测试账号白名单：{player.player_uid}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
