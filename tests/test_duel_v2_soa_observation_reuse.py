"""SoA leaf-value reuse removes duplicate queries without changing Gumbel32."""
import unittest
from unittest.mock import patch


class SoAObservationReuseTests(unittest.TestCase):
    def test_factory_defaults_and_explicit_baseline_override_keep_search_contract(self):
        from app.modules.card_game.rl.batched_search.device_soa_rules import DeviceSoARules,create_soa_search
        rules=object.__new__(DeviceSoARules);rules.device='cpu'
        for options,expected in (({},True),({'reuse_interior_observation':False},False)):
            with patch('app.modules.card_game.rl.batched_search.device_search.DeviceGumbelSearch') as search:
                create_soa_search(rules,object(),object(),object(),object(),rows=2,random=object(),zhenhong_prior=False,**options)
                config=search.call_args.kwargs
                self.assertIs(config['reuse_interior_observation'],expected)
                self.assertEqual((config['simulations'],config['gumbel_candidates'],config['terminal_horizon']), (32,16,0))
                self.assertEqual((config['teacher_mode'],config['opponent_mode']),('network_value','policy'))
                self.assertTrue(config['reuse_root_decision']);self.assertTrue(config['reuse_sampled_world'])

    def test_full32_policy_opponent_results_rng_and_leases_match_with_fewer_queries(self):
        import torch
        from tests.test_duel_v2_device_policy_opponent import PolicyOpponentTest
        helper=PolicyOpponentTest();outputs=[];queries=[];rng=[]
        for enabled in (False,True):
            search,rules,bank,roots,seeds,ids=helper.make()
            try:
                search.reuse_interior_observation=enabled
                calls=[];original=rules.value_observation_device
                def observed(states,viewers):
                    if bool(states.active.any()):calls.append(int(states.active.sum()))
                    return original(states,viewers)
                rules.value_observation_device=observed
                search.initialize(roots,rules.viewer,seeds,ids,ids)
                result=helper.finish(search)
                self.assertEqual(result.completed.tolist(),[32,32,0,32])
                self.assertEqual(result.errors.tolist(),[0]*4)
                outputs.append({key:getattr(result,key).clone() for key in ('actions','visits','policy','information_keys','errors','complete','completed')})
                queries.append(sum(calls));rng.append(bank.export_numpy_states())
                self.assertEqual(int(rules.live.sum()),4)
            finally:bank.close()
        for key in outputs[0]:torch.testing.assert_close(outputs[0][key],outputs[1][key])
        self.assertEqual(rng[0],rng[1]);self.assertLess(queries[1],queries[0]);self.assertGreater(queries[1],0)


if __name__=='__main__':unittest.main()
