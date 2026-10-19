"""Server-rendered Markdown with escaped HTML and a bounded Bilibili extension."""
import re
from urllib.parse import urlsplit, parse_qs
from html import escape, unescape

import mistune
from markupsafe import Markup

BV = re.compile(r'BV[0-9A-Za-z]{10}\Z')


def bilibili_id(value):
    value = value.strip()
    if BV.fullmatch(value):
        return value
    try:
        parsed = urlsplit(value)
    except ValueError:
        raise ValueError('无效视频链接。') from None
    if (parsed.scheme not in ('http', 'https') or parsed.username or parsed.password
            or parsed.hostname not in ('www.bilibili.com', 'bilibili.com', 'player.bilibili.com')):
        raise ValueError('请输入完整的B站视频链接或BV号。')
    match = re.fullmatch(r'/video/(BV[0-9A-Za-z]{10})/?', parsed.path)
    result = match[1] if match else parse_qs(parsed.query).get('bvid', [''])[0] if parsed.hostname == 'player.bilibili.com' else ''
    if not BV.fullmatch(result):
        raise ValueError('未识别到有效的BV号。')
    return result


def safe_url(value, *, image=False):
    value = unescape(value).strip()
    if any(ord(c) < 32 for c in value) or '\\' in value:
        return ''
    try:
        parts = urlsplit(value)
    except ValueError:
        return ''
    if parts.scheme in ('https', 'http') and parts.hostname and not parts.username and not parts.password:
        return value
    if value.startswith('/') and not value.startswith('//'):
        return value
    if not image and (value.startswith('#') or parts.scheme == 'mailto'):
        return value
    return ''


class ArticleRenderer(mistune.HTMLRenderer):
    def __init__(self):
        super().__init__(escape=True)
        self.toc = []

    def heading(self, text, level, **attrs):
        ident = f'article-heading-{len(self.toc) + 1}'
        title = unescape(re.sub(r'<[^>]*>', '', text))
        self.toc.append({'id': ident, 'title': title, 'level': level})
        return f'<h{level} id="{ident}">{text}</h{level}>\n'

    def link(self, text, url, title=None):
        url = safe_url(url)
        if not url:
            return text
        return f'<a href="{escape(url, quote=True)}" rel="noopener noreferrer">{text}</a>'

    def image(self, text, url, title=None):
        url = safe_url(url, image=True)
        alt = escape(unescape(re.sub(r'<[^>]*>', '', text)), quote=True)
        if not url:
            return alt
        return f'<img src="{escape(url, quote=True)}" alt="{alt}" loading="lazy" decoding="async">'

    def block_code(self, code, info=None):
        if (info or '').strip() == 'bilibili':
            try:
                ident = bilibili_id(code)
            except ValueError:
                return super().block_code(code, info)
            return ('<div class="guide-article-video"><iframe title="哔哩哔哩视频" '
                    f'src="https://player.bilibili.com/player.html?bvid={ident}&amp;page=1&amp;autoplay=0" '
                    'allow="fullscreen; picture-in-picture; encrypted-media" allowfullscreen '
                    'referrerpolicy="strict-origin-when-cross-origin"></iframe></div>\n')
        return super().block_code(code, info)


def render(text):
    renderer = ArticleRenderer()
    parser = mistune.create_markdown(renderer=renderer, plugins=['table', 'strikethrough'])
    html = parser(text)
    return Markup(html), renderer.toc
