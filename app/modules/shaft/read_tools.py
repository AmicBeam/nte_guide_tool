"""Read-only interface shared by the Wiki CLI HTTP adapter and chat tools."""
from typing import Any

from .service import get_shaft_axis, get_shaft_catalog_payload, list_filtered_shaft_axes, list_shaft_market


class ShaftReader:
    def characters(self, player: Any) -> list[dict]:
        return get_shaft_catalog_payload(player)['characters']

    def search(self, player: Any, *, scope: str = 'public', character_ids: list[str] | None = None,
               query: str = '', sort: str = 'new', page: int = 1, limit: int = 10) -> dict:
        if scope not in ('public', 'mine', 'favorites'):
            raise ValueError('范围须为 public、mine 或 favorites。')
        if scope == 'public':
            return list_shaft_market(player=player, character_ids=character_ids, query_text=query,
                                     sort=sort, page=page, page_size=limit)
        if player is None:
            raise PermissionError('读取个人排轴需要登录。')
        payload = list_filtered_shaft_axes(player=player, scope=scope, character_ids=character_ids,
                                          query_text=query, sort=sort)
        offset = (page - 1) * limit
        rows = payload['items']
        return {'items': rows[offset:offset + limit], 'page': page, 'page_size': limit,
                'has_more': len(rows) > offset + limit}

    def get(self, player: Any, axis_id: int) -> dict:
        return get_shaft_axis(axis_id, player=player)
