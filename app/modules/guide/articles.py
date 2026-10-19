"""Synchronous account-owned drafts and separately published article snapshots."""
from datetime import datetime
import re
from uuid import UUID

from peewee import CharField, DateTimeField, ForeignKeyField, IntegerField, TextField
from app.db import atomic_transaction
from app.models import BaseModel, Player
from . import service

TARGET_KINDS = ('characters', 'arcs', 'challenges', 'endless')


class GuideArticle(BaseModel):
    id = CharField(primary_key=True, max_length=36)
    owner = ForeignKeyField(Player, backref='guide_articles', on_delete='CASCADE')
    title = CharField(max_length=160, default='')
    markdown = TextField(default='')
    target_kind = CharField(max_length=20, default='')
    target_id = CharField(max_length=100, default='')
    revision = IntegerField(default=0)
    published_title = CharField(max_length=160, default='')
    published_markdown = TextField(default='')
    published_excerpt = CharField(max_length=220, default='')
    published_target_kind = CharField(max_length=20, default='')
    published_target_id = CharField(max_length=100, default='')
    published_at = DateTimeField(null=True)
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        indexes = ((('owner', 'updated_at'), False), (('published_target_kind', 'published_target_id'), False))


class Conflict(ValueError):
    pass


def identifier(value):
    try:
        if str(UUID(value)) != value:
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise LookupError('未找到攻略。') from None
    return value


def revision(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('无效草稿版本。')
    return value


def validate_target(kind, ident):
    if not kind and not ident:
        return
    if kind not in TARGET_KINDS or not isinstance(ident, str) or not ident:
        raise ValueError('请选择有效的挂载页面，或选择不挂载。')
    try:
        record = service.detail(kind, ident)
        if record.get('status') == 'preview':
            raise LookupError
    except LookupError:
        raise ValueError('挂载页面不存在或尚未开放。') from None


def fields(payload):
    if not isinstance(payload, dict) or set(payload) - {'title', 'markdown', 'target_kind', 'target_id', 'revision'}:
        raise ValueError('草稿字段无效。')
    result = {key: payload.get(key, '') for key in ('title', 'markdown', 'target_kind', 'target_id')}
    for key, limit in (('title', 160), ('markdown', 150000), ('target_kind', 20), ('target_id', 100)):
        if not isinstance(result[key], str) or len(result[key]) > limit:
            raise ValueError(f'{key} 格式或长度无效。')
    result['title'] = result['title'].strip()
    validate_target(result['target_kind'], result['target_id'])
    return result


def owned(player, ident):
    row = GuideArticle.get_or_none((GuideArticle.id == identifier(ident)) & (GuideArticle.owner == player.id))
    if row is None:
        raise LookupError('未找到本人草稿。')
    return row


def draft(row, *, body=True):
    result = {'id': row.id, 'title': row.title, 'target_kind': row.target_kind, 'target_id': row.target_id,
              'revision': row.revision, 'published': row.published_at is not None,
              'updated_at': row.updated_at.isoformat() + 'Z', 'url': f'/guide/articles/{row.id}'}
    if body:
        result['markdown'] = row.markdown
    return result


def my_drafts(player):
    return [draft(row, body=False) for row in GuideArticle.select().where(GuideArticle.owner == player.id)
            .order_by(GuideArticle.updated_at.desc())]


def save(player, ident, payload):
    ident = identifier(ident)
    expected = revision(payload.get('revision')) if isinstance(payload, dict) else revision(None)
    values = fields(payload)
    with atomic_transaction():
        row = GuideArticle.get_or_none(GuideArticle.id == ident)
        if row is not None and row.owner_id != player.id:
            raise LookupError('未找到本人草稿。')
        if row is None:
            if expected != 0:
                raise Conflict('草稿版本已变化，请重新打开草稿。')
            row = GuideArticle(id=ident, owner=player)
        elif row.revision != expected:
            raise Conflict('另一页面已保存这篇草稿，请重新打开后继续编辑。本机备份仍保留。')
        for key, value in values.items():
            setattr(row, key, value)
        row.revision += 1
        row.updated_at = datetime.utcnow()
        row.save(force_insert=row.revision == 1)
        return draft(row)


def publish(player, ident, expected):
    expected = revision(expected)
    with atomic_transaction():
        row = owned(player, ident)
        if row.revision != expected:
            raise Conflict('草稿版本已变化，请重新打开后再发布。')
        if not row.title or not row.markdown.strip():
            raise ValueError('发布前请填写标题和正文。')
        validate_target(row.target_kind, row.target_id)
        row.published_title = row.title
        row.published_markdown = row.markdown
        excerpt = re.sub(r'```bilibili\s*.*?```', '视频演示', row.markdown, flags=re.S)
        excerpt = re.sub(r'!?\[([^\]]*)\]\([^)]*\)', r'\1', excerpt)
        row.published_excerpt = re.sub(r'\s+', ' ', re.sub(r'[#*`>\[\]]', '', excerpt)).strip()[:220]
        row.published_target_kind = row.target_kind
        row.published_target_id = row.target_id
        row.published_at = row.updated_at = datetime.utcnow()
        row.revision += 1
        row.save()
        return draft(row)


def public_summary(row):
    return {'id': row.id, 'name': row.published_title, 'kind': 'articles',
            'description': row.published_excerpt, 'author': row.owner.nickname or '玩家',
            'target_kind': row.published_target_kind, 'target_id': row.published_target_id,
            'url': f'/guide/articles/{row.id}', 'json_url': f'/api/guide/articles/{row.id}',
            'markdown_url': f'/guide/articles/{row.id}.md'}


def public_list(query='', *, target_kind=None, target_id=None):
    columns = [GuideArticle.id, GuideArticle.owner, GuideArticle.published_title, GuideArticle.published_excerpt,
               GuideArticle.published_target_kind, GuideArticle.published_target_id, Player.nickname]
    rows = GuideArticle.select(*columns).join(Player).where(GuideArticle.published_at.is_null(False))
    if query:
        rows = rows.where(GuideArticle.published_title.contains(query) | GuideArticle.published_markdown.contains(query))
    if target_kind is not None:
        rows = rows.where((GuideArticle.published_target_kind == target_kind) & (GuideArticle.published_target_id == target_id))
    return [public_summary(row) for row in rows.order_by(GuideArticle.published_at.desc(), GuideArticle.id)]


def public_detail(ident):
    row = GuideArticle.select(GuideArticle, Player).join(Player).where(
        (GuideArticle.id == identifier(ident)) & GuideArticle.published_at.is_null(False)).first()
    if row is None:
        raise LookupError('未找到已发布攻略。')
    return {**public_summary(row), 'markdown': row.published_markdown}


def targets():
    return [{'kind': kind, 'id': str(row['id']), 'name': row['name'], 'kind_name': service.KINDS[kind]}
            for kind in TARGET_KINDS for row in service.public_records(kind) if row.get('status') != 'preview']
