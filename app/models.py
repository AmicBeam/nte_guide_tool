from datetime import datetime

from peewee import AutoField, BooleanField, CharField, DateTimeField, ForeignKeyField, IntegerField, Model, TextField

from app.db import db


class BaseModel(Model):
    class Meta:
        database = db


class Player(BaseModel):
    id = AutoField()
    player_uid = CharField(unique=True, max_length=64)
    nickname = CharField(default='')
    shaft_test_whitelisted = BooleanField(default=False)
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)


class ShaftCharacterPublication(BaseModel):
    id = AutoField()
    character_id = CharField(unique=True, max_length=64)
    character_name = CharField(unique=True, max_length=64)
    is_published = BooleanField(default=False)
    updated_at = DateTimeField(default=datetime.utcnow)


class PlayerTutorial(BaseModel):
    id = AutoField()
    player = ForeignKeyField(Player, backref='tutorials', on_delete='CASCADE')
    scope = CharField(max_length=32)
    completed = BooleanField(default=False)
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        indexes = (
            (('player', 'scope'), True),
        )


class LoginCode(BaseModel):
    id = AutoField()
    player = ForeignKeyField(Player, backref='login_codes', on_delete='CASCADE')
    code = CharField(max_length=16)
    purpose = CharField(default='web_login')
    used = BooleanField(default=False)
    expires_at = DateTimeField()
    created_at = DateTimeField(default=datetime.utcnow)


class AccessToken(BaseModel):
    id = AutoField()
    player = ForeignKeyField(Player, backref='tokens', on_delete='CASCADE')
    token = CharField(unique=True, max_length=128)
    revoked = BooleanField(default=False)
    expires_at = DateTimeField()
    created_at = DateTimeField(default=datetime.utcnow)


class DeckBuild(BaseModel):
    id = AutoField()
    player = ForeignKeyField(Player, backref='builds', on_delete='CASCADE', unique=True)
    character_id = CharField(default='')
    item_ids = TextField(default='[]')
    updated_at = DateTimeField(default=datetime.utcnow)


class Room(BaseModel):
    id = AutoField()
    room_code = CharField(unique=True, max_length=12)
    mode = CharField(default='solo')
    status = CharField(default='waiting')
    host = ForeignKeyField(Player, backref='hosted_rooms', on_delete='CASCADE')
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)


class RoomMember(BaseModel):
    id = AutoField()
    room = ForeignKeyField(Room, backref='members', on_delete='CASCADE')
    player = ForeignKeyField(Player, backref='room_memberships', on_delete='CASCADE')
    seat = CharField(default='host')
    is_host = BooleanField(default=False)
    is_ready = BooleanField(default=False)
    joined_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        indexes = (
            (('room', 'player'), True),
        )


class GameRun(BaseModel):
    id = AutoField()
    room = ForeignKeyField(Room, backref='run', on_delete='CASCADE', unique=True)
    status = CharField(default='idle')
    snapshot = TextField(default='{}')
    updated_at = DateTimeField(default=datetime.utcnow)


class ShaftAxis(BaseModel):
    id = AutoField()
    owner = ForeignKeyField(Player, backref='shaft_axes', on_delete='CASCADE')
    title = CharField(max_length=80, default='')
    description = TextField(default='')
    visibility = CharField(max_length=16, default='private')
    source_version = CharField(max_length=64, default='')
    team_json = TextField(default='[]')
    axis_json = TextField(default='{}')
    enemy_json = TextField(default='{}')
    result_json = TextField(default='{}')
    duration_ticks = IntegerField(default=0)
    direct_damage = IntegerField(default=0)
    stagger_damage = IntegerField(default=0)
    total_damage = IntegerField(default=0)
    dps_x100 = IntegerField(default=0)
    like_count = IntegerField(default=0)
    dislike_count = IntegerField(default=0)
    favorite_count = IntegerField(default=0)
    dedupe_hash = CharField(max_length=64, default='')
    share_token = CharField(max_length=64, null=True, unique=True)
    forked_from = ForeignKeyField('self', backref='published_snapshots', null=True, on_delete='SET NULL')
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    published_at = DateTimeField(null=True)

    class Meta:
        indexes = (
            (('visibility', 'updated_at'), False),
            (('visibility', 'dedupe_hash'), False),
            (('forked_from', 'visibility'), False),
        )


class ShaftAxisCharacter(BaseModel):
    id = AutoField()
    axis = ForeignKeyField(ShaftAxis, backref='characters', on_delete='CASCADE')
    character_id = CharField(max_length=64)
    slot = IntegerField(default=0)

    class Meta:
        indexes = (
            (('axis', 'slot'), True),
            (('character_id',), False),
        )


class ShaftAxisLike(BaseModel):
    id = AutoField()
    axis = ForeignKeyField(ShaftAxis, backref='likes', on_delete='CASCADE')
    player = ForeignKeyField(Player, backref='shaft_likes', on_delete='CASCADE')
    created_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        indexes = (
            (('axis', 'player'), True),
        )


class ShaftAxisDislike(BaseModel):
    id = AutoField()
    axis = ForeignKeyField(ShaftAxis, backref='dislikes', on_delete='CASCADE')
    player = ForeignKeyField(Player, backref='shaft_dislikes', on_delete='CASCADE')
    created_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        indexes = (
            (('axis', 'player'), True),
        )


class ShaftAxisFavorite(BaseModel):
    id = AutoField()
    axis = ForeignKeyField(ShaftAxis, backref='favorites', on_delete='CASCADE')
    player = ForeignKeyField(Player, backref='shaft_favorites', on_delete='CASCADE', null=True)
    visitor_key = CharField(max_length=96, default='')
    created_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        indexes = (
            (('axis', 'player'), False),
            (('axis', 'visitor_key'), False),
        )
