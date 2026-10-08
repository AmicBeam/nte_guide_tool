import unittest
from copy import deepcopy
from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.application.v2_preset_migrations import (
    retire_crimson, _OLD_CRIMSON_TEAM, _OLD_CRIMSON_CARDS,
)


class PresetMigrationTest(unittest.TestCase):
    def store(self):
        return dict(active_id='gift', granted_preset_ids=['crimson'], builds=[dict(
            id='gift', name='盈蓄预组', character_ids=list(_OLD_CRIMSON_TEAM),
            card_ids=list(_OLD_CRIMSON_CARDS.elements()))])

    def test_replaces_untouched_gift_preserving_selected_id_and_is_idempotent(self):
        s=self.store()
        self.assertTrue(retire_crimson(s,STARTER_DECKS))
        self.assertEqual(s['active_id'],'gift')
        self.assertEqual(s['builds'][0]['name'],'真红预组')
        self.assertEqual(s['builds'][0]['character_ids'],['zhenhong','zero','iloy','yi'])
        self.assertFalse(retire_crimson(s,STARTER_DECKS))

    def test_existing_replacement_removes_duplicate_and_redirects_selection(self):
        s=self.store();new=deepcopy(next(d for d in STARTER_DECKS if d['id']=='zhenhong'))
        new['id']='new-gift';s['builds'].append(new)
        self.assertTrue(retire_crimson(s,STARTER_DECKS))
        self.assertEqual(s['builds'],[new]);self.assertEqual(s['active_id'],'new-gift')

    def test_customized_or_unattributed_builds_and_unavailable_replacement_are_untouched(self):
        for kind in ('name','cards','team','ledger','visibility'):
            s=self.store();presets=STARTER_DECKS
            if kind=='name':s['builds'][0]['name']='我的盈蓄'
            if kind=='cards':s['builds'][0]['card_ids'][0]='Z04'
            if kind=='team':s['builds'][0]['character_ids'].reverse()
            if kind=='ledger':s['granted_preset_ids']=[]
            if kind=='visibility':presets=[]
            before=deepcopy(s)
            self.assertFalse(retire_crimson(s,presets));self.assertEqual(s,before)
