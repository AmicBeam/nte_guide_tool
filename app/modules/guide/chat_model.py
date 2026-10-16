"""OpenAI-compatible multimodal transport. No browser or bot-runtime dependency."""
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request


class ChatModelError(RuntimeError):
    pass


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ChatModelError('模型服务发生重定向，请检查后端配置。')


def configured(settings: dict) -> bool:
    parsed = urllib.parse.urlsplit(settings.get('base_url', ''))
    return bool(parsed.scheme in ('http', 'https') and parsed.netloc and not parsed.username
                and not parsed.password and not parsed.query and not parsed.fragment and settings.get('model'))


def complete(settings: dict, messages: list[dict], tools: list[dict]) -> dict:
    if not configured(settings):
        raise ChatModelError('尚未配置对话模型，请联系站点管理员。')
    payload = {'model': settings['model'], 'messages': messages,
               'max_tokens': max(256, min(16000, int(settings.get('max_tokens', 4096))))}
    if tools:
        payload.update(tools=tools, tool_choice='auto')
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    if settings.get('api_key'):
        headers['Authorization'] = 'Bearer ' + settings['api_key']
    context = ssl.create_default_context(cafile=settings.get('ca_bundle') or None)
    opener = urllib.request.build_opener(NoRedirects(), urllib.request.HTTPSHandler(context=context))
    request = urllib.request.Request(settings['base_url'].rstrip('/') + '/chat/completions',
        data=json.dumps(payload, ensure_ascii=False).encode(), headers=headers, method='POST')
    try:
        with opener.open(request, timeout=max(5, min(120, float(settings.get('timeout', 90))))) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ChatModelError('模型返回内容过多，请缩小问题范围。')
        result = json.loads(raw)
        message = result['choices'][0]['message']
        if not isinstance(message, dict):
            raise ValueError
        return message
    except urllib.error.HTTPError as error:
        # Upstream error bodies can contain private prompts or credentials.
        raise ChatModelError(f'模型服务暂不可用（HTTP {error.code}），请稍后重试。') from None
    except (OSError, ValueError, KeyError, IndexError):
        raise ChatModelError('模型连接失败或返回无效，请稍后重试。') from None
