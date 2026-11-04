from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('gift_supplement',Path(__file__).resolve().parents[1]/'scripts/supplement_guide_affinity_gifts.py')
sync=importlib.util.module_from_spec(spec);spec.loader.exec_module(sync)


class AffinityGiftSupplementTest(unittest.TestCase):
    def test_explicit_gifts_work_without_level_rows_and_preserve_other_roles(self):
        data={'characters':[{'id':'1090','affinity':{'available':False,'levels':[],
            'gifts':[],'full_affinity_bonus':[]}}, {'id':'1003','affinity':{'available':True,
            'gifts':[{'id':'Food_143','name':'旧名','offers':[{'price':120}]}]}}]}
        before=deepcopy(data)
        gifts={
            'Food_143':{'SpeicalGiftDatas':[{'RoleId':'1090','LikeabilityValue':100}],
                'DefaultLikeabilityValue':50,'bUnlimit':False,'ExclusiveLikeabilityID':'None'},
            'SpecialGift_1090':{'SpeicalGiftDatas':[{'RoleId':'1090','LikeabilityValue':1200}],
                'DefaultLikeabilityValue':300,'bUnlimit':True,'ExclusiveLikeabilityID':'1090'},
            'OtherGift':{'SpeicalGiftDatas':[{'RoleId':'1003','LikeabilityValue':400}],
                'DefaultLikeabilityValue':200,'bUnlimit':False,'ExclusiveLikeabilityID':'None'}}
        result=sync.supplement(data,'1090',gifts,{'Food_0143_name':'空想特饮','SpecialGift_1090_name':'复古风手机壳'})
        role=result['characters'][0]['affinity']
        self.assertFalse(role['available']);self.assertEqual(role['levels'],[])
        self.assertEqual(role['full_affinity_bonus'],[])
        self.assertEqual([(g['name'],g['affinity_gain'])for g in role['gifts']], [('空想特饮',100),('复古风手机壳',1200)])
        self.assertEqual(role['gifts'][0]['offers'],[{'price':120}])
        self.assertEqual(role['gifts'][1]['offers'],[])
        self.assertTrue(role['gifts'][1]['exclusive']);self.assertTrue(role['gifts'][1]['unlimited'])
        self.assertEqual(data,before)
        self.assertEqual(result['characters'][1],before['characters'][1])

    def test_unresolved_names_or_invalid_binding_fail_without_modifying_data(self):
        data={'characters':[{'id':'1090','affinity':{'gifts':[]}}]}
        row={'SpeicalGiftDatas':[{'RoleId':'1090','LikeabilityValue':100}],
             'DefaultLikeabilityValue':50,'bUnlimit':False,'ExclusiveLikeabilityID':'None'}
        with self.assertRaisesRegex(ValueError,'Missing gift name'):
            sync.supplement(data,'1090',{'unknown':row},{})
        self.assertEqual(data['characters'][0]['affinity']['gifts'],[])
        row['SpeicalGiftDatas'][0]['LikeabilityValue']=True
        with self.assertRaisesRegex(ValueError,'Invalid gift'):
            sync.supplement(data,'1090',{'unknown':row},{'unknown_name':'测试礼物'})
