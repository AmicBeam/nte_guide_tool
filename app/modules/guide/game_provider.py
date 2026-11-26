"""Bounded, explicit read-only game-data import through SMS authentication."""
import base64
import hashlib
import importlib
import json
import secrets
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, HTTPRedirectHandler, build_opener


class ProviderError(RuntimeError):
    pass


class LoginDependencyError(ProviderError):
    pass


def login_cipher():
    try:
        aes = importlib.import_module('Crypto.Cipher.AES')
        pad = importlib.import_module('Crypto.Util.Padding').pad
        cipher = aes.new(GameClient.APP_KEY[-16:].encode(), aes.MODE_ECB)
        cipher.encrypt(pad(b'', 16))
        return aes, pad
    except (ImportError, OSError):
        raise LoginDependencyError('网站短信登录服务暂未就绪，请联系管理员。') from None


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError('游戏服务地址发生变化，请稍后重试。')


def request_json(url, fields=None, headers=None, *, form=False):
    body = None if fields is None else (urlencode(fields).encode() if form else json.dumps(fields).encode())
    headers = dict(headers or {})
    if body is not None:
        headers['Content-Type'] = 'application/x-www-form-urlencoded' if form else 'application/json'
    try:
        with build_opener(NoRedirects).open(Request(url, data=body, headers=headers), timeout=20) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ProviderError('游戏接口返回数据过大，请稍后重试。')
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get('code') not in (0, '0'):
            raise ProviderError('游戏接口未接受请求，请核对验证码或稍后重试；如遇风控，请到塔吉多完成验证。')
        return payload.get('result') if 'result' in payload else payload.get('data')
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        raise ProviderError('暂时无法连接游戏服务，请稍后重试。') from None


class GameClient:
    # These identify the public client protocol; they are not user credentials.
    APP_KEY = '89155cc4e8634ec5b1b6364013b23e3e'

    def __init__(self, device=None):
        self.device = device or 'HT' + secrets.token_hex(7).upper()
        self.access = ''

    def _laohu(self, path, extra, *, millis=False):
        fields = dict(appId='10550', channelId='1', deviceId=self.device,
            deviceType='Pixel 6', deviceModel='Pixel 6', deviceName='Pixel 6',
            deviceSys='Android 14', adm=self.device, idfa='', sdkVersion='4.273.0',
            bid='com.pwrd.htassistant', t=str(int(time.time() * (1000 if millis else 1))))
        fields.update({'version': '12', 'mac': ''} if millis else {'versionCode': '12', 'imei': ''})
        fields.update(extra)
        fields['sign'] = hashlib.md5((''.join(fields[k] for k in sorted(fields)) + self.APP_KEY).encode()).hexdigest()
        if not millis:
            fields = {k: v for k, v in fields.items() if v != ''}
        return request_json('https://user.laohu.com' + path, fields,
                            {'User-Agent': 'okhttp/4.9.0'}, form=True)

    def send_sms(self, mobile):
        login_cipher()
        self._laohu('/m/newApi/sendPhoneCaptchaWithOutLogin',
                    dict(cellphone=mobile, areaCodeId='1', type='16'))

    def login(self, mobile, code):
        AES, pad = login_cipher()
        self._laohu('/m/newApi/checkPhoneCaptchaWithOutLogin', dict(cellphone=mobile, captcha=code))
        def encrypt(value):
            cipher = AES.new(self.APP_KEY[-16:].encode(), AES.MODE_ECB)
            return base64.b64encode(cipher.encrypt(pad(value.encode(), 16))).decode()
        result = self._laohu('/openApi/sms/new/login', dict(cellphone=encrypt(mobile),
            captcha=encrypt(code), areaCodeId='1', type='16'), millis=True)
        if not isinstance(result, dict) or not result.get('token') or not result.get('userId'):
            raise ProviderError('游戏登录结果不完整，请重新验证。')
        session = self._tajiduo('/usercenter/api/login', dict(token=result['token'],
            userIdentity=str(result['userId']), appId='10551'), authenticated=False)
        if not isinstance(session, dict) or not session.get('accessToken'):
            raise ProviderError('未取得游戏资料读取权限，请重新验证。')
        self.access = str(session['accessToken'])
        roles = self._tajiduo('/usercenter/api/v2/getGameRoles?gameId=1289')
        roles = roles.get('roles') if isinstance(roles, dict) else roles
        if not isinstance(roles, list):
            raise ProviderError('无法读取游戏账号列表。')
        return [{'role_id': str(r['roleId']), 'role_name': str(r.get('roleName', ''))[:80]}
                for r in roles if isinstance(r, dict) and r.get('roleId')]

    def _tajiduo(self, path, fields=None, *, authenticated=True):
        timestamp, nonce = str(int(time.time())), secrets.token_hex(4)
        version = '1.2.4'
        signature = hashlib.md5((timestamp + nonce + version + 'pUds3dfMkl').encode()).hexdigest()
        headers = {'User-Agent': 'okhttp/4.12.0', 'platform': 'android', 'deviceid': self.device,
            'appversion': version, 'uid': '0', 'authorization': self.access if authenticated else '',
            'ds': f'{timestamp},{nonce},{signature}'}
        return request_json('https://bbs-api.tajiduo.com' + path, fields, headers, form=True)

    def characters(self, role_id):
        result = self._tajiduo('/apihub/awapi/yh/characters?' + urlencode({'roleId': role_id}))
        if not isinstance(result, list) or len(result) > 200:
            raise ProviderError('角色资料格式异常，本次没有覆盖已保存的数据。')
        return result
