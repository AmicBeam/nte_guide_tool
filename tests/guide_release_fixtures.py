"""Full-content regressions run after publication; boundary tests set their own clock."""
from datetime import datetime
from unittest.mock import patch

AFTER_CHARACTER_RELEASES = datetime.fromisoformat('2026-11-27T12:00:00+08:00')


def published_character_data(cls):
    return patch('app.modules.guide.character_release.current_time',
                 new=lambda: AFTER_CHARACTER_RELEASES)(cls)
