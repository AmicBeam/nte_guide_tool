from flask import Flask, g, request

from app.config import DUEL_SUPPORT_GROUP, IS_DEV_ENV, SECRET_KEY, SHAFT_LOGIN_REQUIRED, SHAFT_SUPPORT_GROUP
from app.db import init_db
from app.models import (
    AccessToken,
    DeckBuild,
    GameRun,
    LoginCode,
    Player,
    PlayerTutorial,
    Room,
    RoomMember,
    ShaftAxis,
    ShaftAxisCharacter,
    ShaftCharacterPublication,
    ShaftAxisDislike,
    ShaftAxisFavorite,
    ShaftAxisLike,
)
from app.utils.logger import get_logger, setup_logging


def create_app() -> Flask:
    setup_logging()
    logger = get_logger('nte.app')
    app = Flask(__name__)
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['IS_DEV_ENV'] = IS_DEV_ENV
    app.config['DUEL_SUPPORT_GROUP'] = DUEL_SUPPORT_GROUP
    app.config['SHAFT_SUPPORT_GROUP'] = SHAFT_SUPPORT_GROUP
    app.config['SHAFT_LOGIN_REQUIRED'] = SHAFT_LOGIN_REQUIRED
    init_db([
        Player,
        PlayerTutorial,
        LoginCode,
        AccessToken,
        DeckBuild,
        Room,
        RoomMember,
        GameRun,
        ShaftCharacterPublication,
        ShaftAxis,
        ShaftAxisCharacter,
        ShaftAxisLike,
        ShaftAxisDislike,
        ShaftAxisFavorite,
    ])
    from app.modules.shaft.service import initialize_shaft_character_publications
    initialize_shaft_character_publications()

    @app.before_request
    def attach_log_id():
        g.log_id = request.headers.get('X-Log-Id') or 'log-unknown'
        if request.path.startswith('/api/'):
            logger.info('api request log_id=%s method=%s path=%s', g.log_id, request.method, request.path)

    @app.after_request
    def add_static_cache_headers(response):
        if request.path.startswith('/static/images/characters/'):
            response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
        if request.path.startswith('/api/'):
            response.headers['X-Log-Id'] = getattr(g, 'log_id', 'log-unknown')
            logger.info(
                'api response log_id=%s method=%s path=%s status=%s',
                getattr(g, 'log_id', 'log-unknown'),
                request.method,
                request.path,
                response.status_code,
            )
        return response

    from app.modules.card_game import blueprint as card_game_module
    from app.modules.kongmu import blueprint as kongmu_module
    from app.modules.preteam import blueprint as preteam_module
    from app.modules.shaft import blueprint as shaft_module
    from .routes import main_bp

    app.register_blueprint(card_game_module)
    app.register_blueprint(kongmu_module)
    app.register_blueprint(preteam_module)
    app.register_blueprint(shaft_module)
    app.register_blueprint(main_bp)
    logger.info('Flask app initialized successfully.')
    return app
