"""Trusted bot-runtime bridge to an existing website login; never an agent tool."""
from contextlib import closing
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sqlite3


def get_token_for_qq_id(qq_id_str, database_path_str=None):
    """Read an active token for the authenticated event sender, without creating one.

    The caller must obtain QQ identity from the bot event, never from model arguments.
    The returned secret is for runtime HTTP headers/environment only, not tool output.
    """
    sender = str(qq_id_str)
    if not re.fullmatch(r'[1-9][0-9]{4,19}', sender):
        return None
    configured = database_path_str or os.environ.get('NTE_DATABASE_PATH')
    if not configured:
        return None
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        return None
    now = datetime.now(timezone.utc)
    try:
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=3)) as connection:
            rows = connection.execute('''
                SELECT access.token, access.expires_at
                FROM accesstoken AS access JOIN player ON access.player_id = player.id
                WHERE player.player_uid = ? AND access.revoked = 0
                ORDER BY access.created_at DESC, access.id DESC
            ''', (sender,))
            for token, expires in rows:
                try:
                    expiration = datetime.fromisoformat(str(expires).replace('Z', '+00:00'))
                    if expiration.tzinfo is None:
                        expiration = expiration.replace(tzinfo=timezone.utc)
                except (TypeError, ValueError):
                    continue
                if expiration > now and isinstance(token, str) and token:
                    return token
    except sqlite3.Error:
        # Missing schema, locked/invalid DB: let the caller fall back to website login.
        return None
    return None


def get_sender_token(sender_qq_id, database_path=None):
    return get_token_for_qq_id(sender_qq_id, database_path)
