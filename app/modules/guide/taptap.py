"""Bounded, unauthenticated reads using TapTap's public search and detail APIs."""
import base64
from html.parser import HTMLParser
import json
import os
import re
import ssl
import urllib.parse
import urllib.request

from .chat_model import NoRedirects

CLIENT = 'V=1&PN=WebApp&LANG=zh_CN&VN_CODE=100000000&LOC=CN&PLT=PC'


class TapTapError(RuntimeError):
    pass


class Text(HTMLParser):
    def __init__(self, value):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.feed(value)

    def handle_data(self, data):
        self.parts.append(data)


def plain(value: str) -> str:
    return ''.join(Text(str(value)).parts).strip()


def fetch(url: str, limit: int = 5 * 1024 * 1024) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ''
    if (parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port not in (None, 443)
            or not (host == 'www.taptap.cn' or host.endswith('.tapimg.com'))):
        raise TapTapError('只能读取TapTap公开接口和帖子附图。')
    try:
        request = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 NTE-Wiki/1.0', 'Accept': 'application/json,image/*'})
        context = ssl.create_default_context(cafile=os.getenv('NTE_GUIDE_CHAT_CA_BUNDLE') or None)
        opener = urllib.request.build_opener(NoRedirects(), urllib.request.HTTPSHandler(context=context))
        with opener.open(request, timeout=18) as response:
            content = response.read(limit + 1)
        if len(content) > limit:
            raise ValueError
        return content
    except Exception:
        raise TapTapError('TapTap公开内容暂时无法读取，请稍后重试或打开原链接。') from None


def api(path: str, query: dict) -> dict:
    url = 'https://www.taptap.cn/webapiv2/' + path + '?' + urllib.parse.urlencode({'X-UA': CLIENT, **query})
    try:
        value = json.loads(fetch(url))
        if value.get('success') is False or not isinstance(value.get('data'), dict):
            raise ValueError
        return value['data']
    except (ValueError, TypeError, AttributeError):
        raise TapTapError('TapTap未返回可核实的查询结果。') from None


def search(query: str) -> dict:
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 100:
        raise ValueError('搜索词须为1–100字。')
    query = query.strip()
    keyword = query if '异环' in query or '異環' in query else '异环 ' + query
    data = api('search/v4/agg-search', {'kw': keyword, 'types': 'community', 'limit': 10, 'from': 0})
    rows = {}
    for group in data.get('list') or []:
        for item in group.get('list') or []:
            moment = item.get('moment') or {}
            identifier = str(moment.get('id_str') or '')
            topic = moment.get('topic') or {}
            title = topic.get('title')
            if not re.fullmatch(r'\d{10,24}', identifier) or not isinstance(title, str) or not title.strip():
                continue
            author = (moment.get('author') or {}).get('user') or {}
            rows[identifier] = {'id': identifier, 'title': plain(title)[:200],
                'summary': plain(str(topic.get('summary') or ''))[:900], 'author': str(author.get('name', ''))[:80],
                'published_at': moment.get('publish_time'), 'url': 'https://www.taptap.cn/moment/' + identifier}
    ordered = sorted(rows.values(), key=lambda row: not any(word in row['title'] for word in ('攻略', '配队', '养成', '手法', '教学', '测评')))
    return {'query': query, 'search_url': 'https://www.taptap.cn/search/topics/' + urllib.parse.quote(keyword, safe=''),
            'items': ordered[:8]}


def block_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ''.join(block_text(child) for child in value)
    if isinstance(value, dict):
        return str(value.get('text') or '') + block_text(value.get('children') or [])
    return ''


def read(post_id: str, image_start: int = 0) -> tuple[dict, list[str]]:
    if not re.fullmatch(r'\d{10,24}', post_id):
        raise ValueError('无效帖子ID。')
    if isinstance(image_start, bool) or not isinstance(image_start, int) or not 0 <= image_start <= 20:
        raise ValueError('无效图片起点。')
    data = api('moment/v3/detail', {'id': post_id})
    article = data.get('moment') or {}
    if str(article.get('id_str')) != post_id:
        raise TapTapError('帖子详情不完整，请打开原链接。')
    topic = article.get('topic') or {}
    contents = ((data.get('first_post') or {}).get('contents') or {}).get('json') or []
    text = '\n'.join(filter(None, (plain(block_text(block)) for block in contents)))
    image_urls = []
    for item in topic.get('footer_images') or []:
        value = item.get('original_url') or item.get('url')
        if isinstance(value, str) and '.tapimg.com/moment/' in value:
            image_urls.append(value)
    image_urls = list(dict.fromkeys(image_urls))
    images = []
    for image_url in image_urls[image_start:image_start + 2]:
        try:
            image = fetch(image_url, 3 * 1024 * 1024)
            mime = ('image/png' if image.startswith(b'\x89PNG\r\n\x1a\n') else
                    'image/jpeg' if image.startswith(b'\xff\xd8\xff') else
                    'image/webp' if image[:4] == b'RIFF' and image[8:12] == b'WEBP' else '')
            if mime:
                images.append(f'data:{mime};base64,' + base64.b64encode(image).decode())
        except TapTapError:
            continue
    return {'id': post_id, 'url': 'https://www.taptap.cn/moment/' + post_id,
            'title': plain(topic.get('title', ''))[:200], 'text': text[:14000],
            'summary': str(topic.get('summary') or '')[:900], 'text_truncated': len(text) > 14000,
            'image_urls': image_urls[:24], 'image_start': image_start, 'image_count': len(image_urls),
            'images_read': len(images), 'note': '正文与摘要分开；帖子为作者观点，仅引用必要片段，完整内容请查看原帖。'}, images
