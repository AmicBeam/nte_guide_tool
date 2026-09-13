import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dao import issue_mock_login
from app.db import init_db
from app.models import AccessToken, DeckBuild, DuelV2Build, GameRun, LoginCode, Player
from app.modules.card_game.engine.application.v2_service import complete_starter_presets
from app.utils.logger import get_logger, setup_logging


def main():
    setup_logging()
    logger = get_logger('nte.seed_mock_account')
    init_db([Player, LoginCode, AccessToken, DeckBuild, DuelV2Build, GameRun])
    player = issue_mock_login('10001', 'Mock Runner', '654321')
    player.shaft_test_whitelisted = True
    player.save(only=[Player.shaft_test_whitelisted])
    store = complete_starter_presets(player)
    names = [item.get('name') for item in store.get('builds') or []]
    logger.info('Mock account seeded into shared database. v2_presets=%s', names)
    print('mock-account-seeded 10001 654321')
    print('v2-presets ' + ','.join(str(name) for name in names))


if __name__ == '__main__':
    main()
