import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

NTE_ENV = os.getenv('NTE_ENV', 'production').strip().lower()
IS_DEV_ENV = NTE_ENV == 'development'
SECRET_KEY = os.getenv('NTE_SECRET_KEY', 'nte-board-game-dev-secret')
DATABASE_PATH = os.getenv('NTE_DATABASE_PATH', str(BASE_DIR / 'nte_board_game.db'))
TOKEN_TTL_HOURS = int(os.getenv('NTE_TOKEN_TTL_HOURS', '72'))
LOG_DIR = Path(os.getenv('NTE_LOG_DIR', str(BASE_DIR / 'logs')))
DUEL_SUPPORT_GROUP = os.getenv('NTE_DUEL_SUPPORT_GROUP', '575377480')
SHAFT_SUPPORT_GROUP = os.getenv('NTE_SHAFT_SUPPORT_GROUP', '1005485948')
SHAFT_LOGIN_REQUIRED = os.getenv('NTE_SHAFT_LOGIN_REQUIRED', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
DUEL_AI_MODEL_DIR = Path(os.getenv('NTE_DUEL_AI_MODEL_DIR', str(BASE_DIR / 'app' / 'modules' / 'card_game' / 'engine' / 'ai' / 'models')))
