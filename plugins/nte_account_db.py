import os
from datetime import datetime
from pathlib import Path
from random import randint

from dotenv import load_dotenv
from peewee import AutoField, BooleanField, CharField, DateTimeField, ForeignKeyField, Model, SqliteDatabase


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / '.env')
DATABASE_PATH = os.getenv('NTE_DATABASE_PATH', str(BASE_DIR / 'nte_board_game.db'))
PERMANENT_PASSWORD_EXPIRES_AT = datetime(2999, 12, 31, 23, 59, 59)
MAX_NICKNAME_LENGTH = 8

nte_account_db = SqliteDatabase(DATABASE_PATH, pragmas={'foreign_keys': 1})


class NTEBaseModel(Model):
    class Meta:
        database = nte_account_db


class NTEPlayer(NTEBaseModel):
    id = AutoField()
    player_uid = CharField(unique=True, max_length=64)
    nickname = CharField(default='')
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        table_name = 'player'


class NTELoginCode(NTEBaseModel):
    id = AutoField()
    player = ForeignKeyField(NTEPlayer, backref='login_codes', on_delete='CASCADE')
    code = CharField(max_length=16)
    purpose = CharField(default='web_login')
    used = BooleanField(default=False)
    expires_at = DateTimeField()
    created_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        table_name = 'logincode'


class NTEShaftCharacterPublication(NTEBaseModel):
    id = AutoField()
    character_id = CharField(unique=True, max_length=64)
    character_name = CharField(unique=True, max_length=64)
    access_level = CharField(default='test', max_length=16)
    is_published = BooleanField(default=False)
    updated_at = DateTimeField(default=datetime.utcnow)

    class Meta:
        table_name = 'shaftcharacterpublication'


DEFAULT_UNPUBLISHED_CHARACTERS = {
    '灵可': 'char_0846d632e0',
}
RELEASED_CHARACTERS = {
    '伊洛伊': 'char_a01c39f576',
    '残虹': 'char_076a1f4e53',
}


def ensure_nte_tables():
    if nte_account_db.is_closed():
        nte_account_db.connect(reuse_if_open=True)
    if nte_account_db.table_exists('shaftcharacterpublication'):
        existing_columns = {
            column.name for column in nte_account_db.get_columns('shaftcharacterpublication')
        }
        if 'access_level' not in existing_columns:
            nte_account_db.execute_sql(
                'ALTER TABLE "shaftcharacterpublication" '
                'ADD COLUMN "access_level" VARCHAR(16) NOT NULL DEFAULT \'test\''
            )
            nte_account_db.execute_sql(
                'UPDATE "shaftcharacterpublication" SET "access_level" = \'public\' '
                'WHERE "is_published" = 1'
            )
    nte_account_db.create_tables([NTEPlayer, NTELoginCode, NTEShaftCharacterPublication])
    for character_name, character_id in DEFAULT_UNPUBLISHED_CHARACTERS.items():
        publication, _ = NTEShaftCharacterPublication.get_or_create(
            character_id=character_id,
            defaults={
                'character_name': character_name,
                'access_level': 'test',
                'is_published': False,
            },
        )
        fields_to_update = []
        if publication.character_name != character_name:
            publication.character_name = character_name
            fields_to_update.append(NTEShaftCharacterPublication.character_name)
        if not publication.is_published and publication.access_level != 'test':
            publication.access_level = 'test'
            fields_to_update.append(NTEShaftCharacterPublication.access_level)
        if fields_to_update:
            publication.updated_at = datetime.utcnow()
            publication.save(only=[
                *fields_to_update,
                NTEShaftCharacterPublication.updated_at,
            ])
    for character_name, character_id in RELEASED_CHARACTERS.items():
        publication, _ = NTEShaftCharacterPublication.get_or_create(
            character_id=character_id,
            defaults={
                'character_name': character_name,
                'access_level': 'public',
                'is_published': True,
            },
        )
        fields_to_update = []
        if publication.character_name != character_name:
            publication.character_name = character_name
            fields_to_update.append(NTEShaftCharacterPublication.character_name)
        if not publication.is_published:
            publication.is_published = True
            fields_to_update.append(NTEShaftCharacterPublication.is_published)
        if publication.access_level != 'public':
            publication.access_level = 'public'
            fields_to_update.append(NTEShaftCharacterPublication.access_level)
        if fields_to_update:
            publication.updated_at = datetime.utcnow()
            publication.save(only=[
                *fields_to_update,
                NTEShaftCharacterPublication.updated_at,
            ])


def normalize_nickname(nickname: str):
    return (nickname or '').strip()[:MAX_NICKNAME_LENGTH]


def issue_account_register_code(player_uid: str, nickname: str = ''):
    ensure_nte_tables()
    nickname = normalize_nickname(nickname)
    player, created = NTEPlayer.get_or_create(
        player_uid=player_uid,
        defaults={'nickname': nickname},
    )
    if nickname and player.nickname != nickname:
        player.nickname = nickname
        player.updated_at = datetime.utcnow()
        player.save()

    NTELoginCode.update(used=True).where(
        (NTELoginCode.player == player) &
        (NTELoginCode.used == False) &
        (NTELoginCode.purpose == 'web_login')
    ).execute()

    code = f'{randint(0, 999999):06d}'
    NTELoginCode.create(
        player=player,
        code=code,
        purpose='web_login',
        used=False,
        expires_at=PERMANENT_PASSWORD_EXPIRES_AT,
    )
    return code, created


def publish_shaft_character(character_name: str) -> str:
    ensure_nte_tables()
    normalized_name = (character_name or '').strip()
    publication = NTEShaftCharacterPublication.get_or_none(
        NTEShaftCharacterPublication.character_name == normalized_name
    )
    if publication is None:
        return 'not_found'
    if publication.is_published:
        return 'already_published'
    publication.is_published = True
    publication.access_level = 'public'
    publication.updated_at = datetime.utcnow()
    publication.save(only=[
        NTEShaftCharacterPublication.is_published,
        NTEShaftCharacterPublication.access_level,
        NTEShaftCharacterPublication.updated_at,
    ])
    return 'published'
