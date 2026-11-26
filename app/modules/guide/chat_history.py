"""Validate browser-owned history and build bounded multimodal model messages."""
import base64
import math
import re
import time

GAP_MS = 6 * 60 * 60 * 1000
MAX_MESSAGES = 32
MAX_IMAGE_BYTES = 1536 * 1024


def image_data(value: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_IMAGE_BYTES * 4 // 3 + 100:
        raise ValueError('单张图片过大，请压缩后重试。')
    match = re.fullmatch(r'data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=]+)', value)
    if not match:
        raise ValueError('图片须为PNG、JPEG或WebP，不接受远程图片地址。')
    try:
        raw = base64.b64decode(match[2], validate=True)
    except ValueError:
        raise ValueError('图片内容无效。') from None
    valid = {'image/png': raw.startswith(b'\x89PNG\r\n\x1a\n'),
             'image/jpeg': raw.startswith(b'\xff\xd8\xff'),
             'image/webp': raw[:4] == b'RIFF' and raw[8:12] == b'WEBP'}
    if len(raw) > MAX_IMAGE_BYTES or not valid[match[1]]:
        raise ValueError('图片格式或大小不符合要求。')
    return value


def prepare(payload: dict, now: float | None = None) -> list[dict]:
    rows = payload.get('messages')
    now_ms = (time.time() if now is None else now) * 1000
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise ValueError('请发送有效的对话记录。')
    validated = []
    previous = None
    for row in rows:
        if not isinstance(row, dict) or row.get('role') not in ('user', 'assistant'):
            raise ValueError('只接受用户消息和助手消息。')
        stamp = row.get('timestamp')
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp < 0 or stamp > now_ms + 60000:
            raise ValueError('消息时间无效。')
        if previous is not None and stamp < previous:
            raise ValueError('消息时间顺序无效。')
        if previous is not None and stamp - previous > GAP_MS:
            validated = []
        previous = stamp
        text = row.get('text', '')
        images = row.get('images', [])
        if not isinstance(text, str) or len(text) > 12000 or not isinstance(images, list) or len(images) > 2:
            raise ValueError('单条消息最多12000字、2张图片。')
        if row['role'] == 'assistant' and images:
            raise ValueError('助手历史不接受图片字段。')
        if not text.strip() and not images:
            raise ValueError('消息内容不能为空。')
        validated.append({'role': row['role'], 'text': text, 'images': images})
    if validated[-1]['role'] != 'user' or now_ms - previous > GAP_MS:
        raise ValueError('请发送一条新问题。')
    validated = validated[-MAX_MESSAGES:]
    remaining_images = 6
    remaining_text = 60000
    result = []
    for row in reversed(validated):
        if len(row['text']) > remaining_text:
            break
        remaining_text -= len(row['text'])
        content = [{'type': 'text', 'text': row['text'] or '请查看图片并回答。'}]
        if len(row['images']) <= remaining_images:
            for value in row['images']:
                content.append({'type': 'image_url', 'image_url': {'url': image_data(value)}})
            remaining_images -= len(row['images'])
        elif row['images']:
            content[0]['text'] += '\n[早前附图未包含在本轮图片窗口内，需要时请用户重发。]'
        result.append({'role': row['role'], 'content': content if row['role'] == 'user' else row['text']})
    return list(reversed(result))
