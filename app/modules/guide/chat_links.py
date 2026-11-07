"""HTML destinations for verified guide records, kept separate from API URLs."""
from urllib.parse import quote, urlencode

from . import presentation


def entry_links(kind: str, record: dict, stage=None, encounter=None) -> dict:
    path = f'/guide/{kind}/{quote(str(record["id"]), safe="")}'
    if kind == 'foods':
        return {'detail': '/guide?' + urlencode({'kind': 'foods', 'food': str(record['id'])})}
    query = {'stage': stage} if stage is not None else {}
    if kind == 'dungeons' and encounter:
        query['encounter'] = encounter
    links = {'detail': path + ('?' + urlencode(query) if query else '')}
    if kind == 'characters' and record.get('status') != 'preview':
        links.update(cultivation=path + '/cultivation', builds=path + '/builds')
        kongmu = presentation.kongmu_entry(record)
        if kongmu:
            links['kongmu'] = kongmu['url']
        overlap = presentation.overlap_entry(record)
        if overlap:
            links['overlap'] = overlap
    return links
