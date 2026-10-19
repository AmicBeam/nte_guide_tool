"""Guide authoring HTTP adapters; writes always use the authenticated owner."""
from pathlib import Path
import json

from flask import Response, current_app, g, jsonify, render_template, request
from app.auth import token_required
from . import articles, article_markdown


def public_items(query='', **target):
    if not current_app.extensions.get('guide_articles_ready'):
        return []
    return articles.public_list(query, **target)


def register(blueprint):
    @blueprint.context_processor
    def article_assets():
        static = Path(__file__).parent / 'static'
        return {'article_asset_version': max((static / name).stat().st_mtime_ns
                                            for name in ('articles.css', 'article_editor.js', 'article_view.js')),
                'guide_authored_count': articles.GuideArticle.select().where(articles.GuideArticle.published_at.is_null(False)).count()
                if current_app.extensions.get('guide_articles_ready') else 0}

    @blueprint.errorhandler(articles.Conflict)
    def conflict(error):
        return jsonify(error={'code': 'draft_conflict', 'message': str(error)}), 409

    @blueprint.after_request
    def private_drafts(response):
        if request.path.startswith('/api/guide/me/articles') or request.path == '/api/guide/articles/preview':
            response.headers['Cache-Control'] = 'no-store'
        return response

    @blueprint.get('/guide/write')
    def write_article():
        return render_template('guide/article_editor.html')

    @blueprint.get('/api/guide/article-targets')
    def article_targets():
        return jsonify(items=articles.targets())

    def payload():
        if not request.is_json or (request.content_length or 0) > 1024 * 1024:
            raise ValueError('请提交不超过1MB的JSON正文。')
        raw = request.stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError('正文不能超过1MB。')
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ValueError('请求须为有效JSON对象。') from None
        if not isinstance(value, dict):
            raise ValueError('请求须为JSON对象。')
        return value

    @blueprint.get('/api/guide/me/articles')
    @token_required
    def article_drafts():
        return jsonify(items=articles.my_drafts(g.current_player))

    @blueprint.get('/api/guide/me/articles/<ident>')
    @token_required
    def article_draft(ident):
        return jsonify(data=articles.draft(articles.owned(g.current_player, ident)))

    @blueprint.put('/api/guide/me/articles/<ident>')
    @token_required
    def save_article(ident):
        return jsonify(data=articles.save(g.current_player, ident, payload()))

    @blueprint.post('/api/guide/me/articles/<ident>/publish')
    @token_required
    def publish_article(ident):
        return jsonify(data=articles.publish(g.current_player, ident, payload().get('revision')))

    @blueprint.post('/api/guide/articles/preview')
    @token_required
    def preview_article():
        value = payload()
        text = value.get('markdown')
        if not isinstance(text, str) or len(text) > 150000:
            raise ValueError('正文须为不超过150000字的文字。')
        html, toc = article_markdown.render(text)
        return jsonify(html=str(html), toc=toc)

    @blueprint.get('/api/guide/articles')
    def list_articles():
        return jsonify(items=public_items(request.args.get('q', '')[:120]))

    @blueprint.get('/api/guide/articles/<ident>')
    def read_article(ident):
        return jsonify(schema_version=1, kind='articles', data=articles.public_detail(ident))

    @blueprint.get('/guide/articles/<ident>.md')
    def article_markdown_page(ident):
        row = articles.public_detail(ident)
        return Response(f'# {row["name"]}\n\n{row["markdown"]}', content_type='text/markdown; charset=utf-8')

    @blueprint.get('/guide/articles/<ident>')
    def article_page(ident):
        row = articles.public_detail(ident)
        html, toc = article_markdown.render(row['markdown'])
        target = None
        if row['target_kind']:
            from . import service
            try:
                target = service.summary(row['target_kind'], service.detail(row['target_kind'], row['target_id']))
            except LookupError:
                pass
        return render_template('guide/article.html', article=row, article_html=html, toc=toc, target=target)
