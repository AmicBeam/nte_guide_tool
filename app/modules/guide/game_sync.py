"""Short-lived SMS sessions; only sanitized character snapshots reach storage."""
import re
import secrets
import threading
import time
from dataclasses import dataclass, field

from .game_provider import GameClient, ProviderError
from . import user_data

_lock = threading.RLock()
_sessions = {}
_cooldowns = {}
_hourly = {}
_TTL = 600


class SyncError(ValueError):
    pass


@dataclass
class Session:
    player_id: int
    mobile: str
    client: GameClient
    expires: float
    attempts: int = 0
    roles: list = field(default_factory=list)
    busy: bool = False


def expire_session(token):
    with _lock:
        session = _sessions.get(token)
        if session is not None and session.expires <= time.monotonic():
            _sessions.pop(token, None)


def send_sms(player, payload):
    mobile = payload.get('mobile')
    if not isinstance(mobile, str) or not re.fullmatch(r'1\d{10}', mobile):
        raise SyncError('请输入11位中国大陆手机号。')
    now = time.monotonic()
    with _lock:
        for key in list(_sessions):
            if _sessions[key].expires <= now:
                del _sessions[key]
        for key in list(_cooldowns):
            if _cooldowns[key] <= now:
                del _cooldowns[key]
        import hashlib
        phone_key = ('phone', hashlib.sha256(mobile.encode()).hexdigest())
        if any(_cooldowns.get(k, 0) > now for k in (('player', player.id), phone_key)):
            raise SyncError('验证码已发送，请60秒后再试。')
        for key in list(_hourly):
            if _hourly[key][0] <= now:
                del _hourly[key]
        if any(_hourly.get(k, (0, 0))[1] >= 5 for k in (('player', player.id), phone_key)):
            raise SyncError('本小时获取验证码次数已达上限，请稍后再试。')
        if len(_sessions) >= 512 or len(_hourly) >= 4096:
            raise SyncError('同步人数较多，请稍后重试。')
        _cooldowns[('player', player.id)] = _cooldowns[phone_key] = now + 60
        for key in (('player', player.id), phone_key):
            until, count = _hourly.get(key, (now + 3600, 0))
            _hourly[key] = (until, count + 1)
    client = GameClient()
    client.send_sms(mobile)
    token = secrets.token_urlsafe(32)
    with _lock:
        for key in list(_sessions):
            if _sessions[key].player_id == player.id:
                del _sessions[key]
        _sessions[token] = Session(player.id, mobile, client, now + _TTL)
        timer = threading.Timer(_TTL, expire_session, args=(token,))
        timer.daemon = True
        timer.start()
    return dict(session_id=token, expires_in=600)


def operate(player, payload, *, select=False):
    token = payload.get('session_id')
    with _lock:
        session = _sessions.get(token) if isinstance(token, str) else None
        if session is not None and session.expires <= time.monotonic():
            _sessions.pop(token, None)
            session = None
        if session is None or session.player_id != player.id:
            raise SyncError('验证会话已过期，请重新获取验证码。')
        if session.busy:
            raise SyncError('正在同步，请稍候。')
        session.busy = True
    try:
        if not select and session.roles:
            if len(session.roles) > 1:
                return dict(status='select_role', roles=session.roles)
            role = session.roles[0]
        elif not select:
            code = payload.get('code')
            if not isinstance(code, str) or not re.fullmatch(r'\d{4,8}', code):
                raise SyncError('请输入4至8位短信验证码。')
            session.attempts += 1
            if session.attempts > 5:
                with _lock:
                    _sessions.pop(token, None)
                raise SyncError('验证次数过多，请重新获取验证码。')
            session.roles = session.client.login(session.mobile, code)
            session.mobile = ''
            if not session.roles:
                raise SyncError('此账号没有可读取的异环角色，请先在塔吉多绑定游戏账号。')
            if len(session.roles) > 1:
                return dict(status='select_role', roles=session.roles)
            role = session.roles[0]
        else:
            role = next((r for r in session.roles if r['role_id'] == payload.get('role_id')), None)
            if role is None:
                raise SyncError('请选择本次验证返回的游戏账号。')
        raw = session.client.characters(role['role_id'])
        profile = user_data.save_profile(player, role, raw)
        with _lock:
            _sessions.pop(token, None)
        return dict(status='success', profile=profile)
    finally:
        with _lock:
            session.busy = False
