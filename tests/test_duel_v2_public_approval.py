"""Synthetic evidence fixtures validate admission; they are not gameplay results."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy required')
class PublicApprovalTest(unittest.TestCase):
    def test_matching_evidence_and_stale_or_incomplete_report(self):
        import numpy as np
        from app.modules.card_game.rl.public_schema import SCHEMA,state_dim,rule_identity,CAND_DIM
        from app.modules.card_game.rl.capability import PUBLIC_CHARACTERS
        from app.modules.card_game.rl.gpu_duel.catalog import GPU_LOCK,CARD_IDS,SEATS
        from app.modules.card_game.rl.public_export import approve_candidate
        from app.modules.card_game.engine.ai.advanced_model import FrozenModel
        shapes={'state_net.0.weight':(1,state_dim()),'state_net.0.bias':(1,),
                'state_net.2.weight':(1,1),'state_net.2.bias':(1,),
                'cand_net.0.weight':(1,CAND_DIM),'cand_net.0.bias':(1,),
                'cand_net.2.weight':(1,1),'cand_net.2.bias':(1,),
                'score.weight':(1,1),'score.bias':(1,)}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'starter.npz'
            np.savez_compressed(path,**{k:np.zeros(shape,np.float32) for k,shape in shapes.items()})
            digest=hashlib.sha256(path.read_bytes()).hexdigest();identity=rule_identity()
            manifest={'schema':SCHEMA,'deck':'starter','gpu_lock':GPU_LOCK,'card_ids':list(CARD_IDS),
                      'seat_ids':list(SEATS),'sha256':digest,'rule_identity':{'rule_hash':identity},
                      'hidden':1,'state_dim':state_dim(),'cand_dim':CAND_DIM,'update':1,
                      'capability':{'kind':'public_candidate_v1','bot_preset':'starter','human_public_custom':False}}
            path.with_suffix('.json').write_text(json.dumps(manifest))
            teams=list(itertools.combinations(PUBLIC_CHARACTERS,4))
            report={'model':{'sha256':digest,'schema':SCHEMA,'rule_hash':identity},'preset':'starter',
                    'sampled_opponent':True,'by_opponent_team':{str(i):{} for i in range(15)},'pairs':32,
                    'engine':{'escalation':True,'first_turn_draw':True},
                    'games_index':[{'pair':i,'position':position,'opponent_team':teams[i%15],
                                    'terminated':True,'winner':'a'} for i in range(32) for position in ('first','second')]}
            source=root/'report.json'
            report['model']['sha256']='wrong';source.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'does not match'):approve_candidate(path,source)
            report['model']['sha256']=digest;report['games_index'][0]['truncated']=True
            source.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'incomplete'):approve_candidate(path,source)
            report['games_index'][0]['truncated']=False;source.write_text(json.dumps(report))
            evidence=approve_candidate(path,source)
            self.assertEqual(evidence['games'],64)
            self.assertTrue(FrozenModel(root,'starter').capability['human_public_custom'])
