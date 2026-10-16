"""OpenAI-compatible multimodal transport. No browser or bot-runtime dependency."""
import json
import ssl
import time
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


def stream(settings: dict, messages: list[dict], tools: list[dict]):
    if not configured(settings):
        raise ChatModelError('尚未配置对话模型，请联系站点管理员。')
    payload = {'model': settings['model'], 'messages': messages, 'stream': True,
               'max_tokens': max(256, min(16000, int(settings.get('max_tokens', 4096))))}
    if tools:
        payload.update(tools=tools, tool_choice='auto')
    headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream'}
    if settings.get('api_key'):
        headers['Authorization'] = 'Bearer ' + settings['api_key']
    context = ssl.create_default_context(cafile=settings.get('ca_bundle') or None)
    opener = urllib.request.build_opener(NoRedirects(), urllib.request.HTTPSHandler(context=context))
    request = urllib.request.Request(settings['base_url'].rstrip('/') + '/chat/completions',
        data=json.dumps(payload, ensure_ascii=False).encode(), headers=headers, method='POST')
    timeout = max(5, min(120, float(settings.get('timeout', 90))))
    deadline = time.monotonic() + timeout
    try:
        with opener.open(request, timeout=timeout) as response:
            content, calls, size, data, finished = '', {}, 0, [], False
            while True:
                line = response.readline(65537)
                if time.monotonic() > deadline:
                    raise ChatModelError('本轮回答用时较长，请稍后重试。')
                size += len(line)
                if size > 2 * 1024 * 1024 or len(line) > 65536:
                    raise ChatModelError('模型返回内容过多，请缩小问题范围。')
                if line.startswith(b'data:'):
                    data.append(line[5:].strip())
                if line.strip() and line:
                    continue
                if data:
                    raw = b'\n'.join(data)
                    data = []
                    if raw == b'[DONE]':
                        break
                    packet = json.loads(raw)
                    if packet.get('error'):
                        raise ValueError
                    choices = packet.get('choices', [])
                    if choices:
                        choice = choices[0]
                        delta = choice.get('delta') or {}
                        text = delta.get('content') or ''
                        if not isinstance(text, str):
                            raise ValueError
                        if text:
                            content += text
                            if len(content) > 24000:
                                raise ChatModelError('模型返回内容过多，请缩小问题范围。')
                            yield {'type': 'delta', 'text': text}
                        for part in delta.get('tool_calls') or []:
                            index = part['index']
                            if not isinstance(index, int) or not 0 <= index < 4:
                                raise ValueError
                            call = calls.setdefault(index, {'id': '', 'type': 'function',
                                'function': {'name': '', 'arguments': ''}})
                            call['id'] += part.get('id') or ''
                            function = part.get('function') or {}
                            for key in ('name', 'arguments'):
                                call['function'][key] += function.get(key) or ''
                        reason = choice.get('finish_reason')
                        if reason:
                            if reason not in ('stop', 'tool_calls'):
                                raise ChatModelError('回答未完成，请缩小问题范围后重试。')
                            finished = True
                if not line:
                    break
            if not finished:
                raise ChatModelError('回答传输中断，请重试。')
            yield {'type': 'message', 'message': {'content': content,
                'tool_calls': [calls[index] for index in sorted(calls)]}}
    except urllib.error.HTTPError as error:
        # Upstream error bodies can contain private prompts or credentials.
        raise ChatModelError(f'模型服务暂不可用（HTTP {error.code}），请稍后重试。') from None
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        raise ChatModelError('模型连接失败或返回无效，请稍后重试。') from None


def complete(settings: dict, messages: list[dict], tools: list[dict]) -> dict:
    for event in stream(settings, messages, tools):
        if event['type'] == 'message':
            return event['message']
    raise ChatModelError('模型没有返回完整回答，请重试。')
