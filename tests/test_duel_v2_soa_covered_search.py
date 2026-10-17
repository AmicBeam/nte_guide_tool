"""Covered search construction/ownership mocks, never real CUDA acceptance."""
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

from app.modules.card_game.rl.batched_search.soa_covered_search import create_soa_covered_search_bundle


class SoACoveredSearchBundleTests(unittest.TestCase):
    def inputs(self):
        import torch
        from app.modules.card_game.rl.batched_search.device_soa_rules import DeviceSoARules
        from app.modules.card_game.rl.seven_schedule import ROSTER
        # Actual class, explicitly unconstructed CPU fixture metadata. Only
        # numerical constructors below are mocked; no CUDA run is claimed.
        rules=object.__new__(DeviceSoARules)
        rules.rows=8;rules.max_actions=800;rules.device=torch.device('cuda:0')
        rules.require_teacher=True;rules.staged_teacher=True;rules.close=Mock()
        policy=NS(device=rules.device,close=Mock())
        statistics=NS(close=Mock());keys=NS(close=Mock());random=object()
        models={key:NS(version='aux-'+key) for key in ROSTER}
        report=dict(schema='seven_expert_value_v2',passed=True,
            objective='visible_rule_after_six_exploration_actions',rule_hash='rule',algorithm_sha256='algorithm',
            models={key:model.version for key,model in models.items()})
        auxiliary=NS(device=rules.device,keys=tuple(sorted(models)),
                     versions=tuple(models[key].version for key in sorted(models)),close=Mock())
        search=NS(close=Mock())
        return NS(rules=rules,policy=policy,statistics=statistics,keys=keys,random=random,
                  models=models,report=report,auxiliary=auxiliary,search=search,scope=Path('explicit-scoped-auxiliary'))

    def make(self,inputs,**options):
        return create_soa_covered_search_bundle(inputs.rules,inputs.policy,inputs.statistics,inputs.keys,
            auxiliary_scope=inputs.scope,rows=inputs.rules.rows,random=inputs.random,**options)

    def mocks(self,inputs,**kwargs):
        from contextlib import ExitStack
        stack=ExitStack();self.addCleanup(stack.close)
        loader=stack.enter_context(patch('app.modules.card_game.rl.seven_covered_value.load_scope',
            return_value=(inputs.models,inputs.report)))
        bank=stack.enter_context(patch('app.modules.card_game.rl.batched_search.device_value_only_inference.DeviceValueOnlyInference',
            return_value=inputs.auxiliary))
        search=stack.enter_context(patch('app.modules.card_game.rl.batched_search.device_search.DeviceGumbelSearch',
            return_value=inputs.search))
        return loader,bank,search

    def test_real_scope_loader_entry_seven_versions_independent_aux_and_fixed_contract(self):
        values=self.inputs();loader,bank,search=self.mocks(values)
        bundle=self.make(values,noise=True,inference_rows=3)
        loader.assert_called_once_with(values.scope)
        self.assertIs(bank.call_args.args[0],values.models)
        self.assertEqual(bank.call_args.kwargs,dict(device='cuda:0',max_rows=3,max_candidate_elements=3*800*1576))
        self.assertEqual(search.call_args.args,(values.rules,values.policy,values.auxiliary,values.statistics,values.keys))
        config=search.call_args.kwargs
        self.assertEqual((config['simulations'],config['gumbel_candidates'],config['max_depth']),(32,16,10))
        self.assertEqual((config['teacher_mode'],config['terminal_horizon'],config['opponent_mode']),('covered_value',3,'rule'))
        self.assertIs(config['prior_guide'],None)
        for option in ('reuse_sampled_world','reuse_root_decision','reuse_root_policy_value','verify_root_value_reuse'):
            self.assertIs(config[option],False)
        self.assertNotIn('allow_cpu_reference',config);self.assertNotIn('allow_synthetic',config)
        self.assertEqual(config['value_batch_rows'],3);self.assertIs(config['random'],values.random)
        self.assertIs(config['noise'],True)
        self.assertIs(bundle.search,values.search);self.assertIs(bundle.auxiliary,values.auxiliary)
        self.assertIsNot(bundle.auxiliary,values.policy)
        self.assertEqual(bundle.auxiliary_keys,values.auxiliary.keys)
        self.assertEqual(bundle.auxiliary_versions,values.auxiliary.versions)
        self.assertEqual(bundle.scope_receipt['models'],values.report['models'])
        self.assertEqual(bundle.scope_receipt['world_sampling'],'per_simulation')
        for qualification in ('gpu_qualified','full_search_qualified','training_qualified'):
            self.assertIs(bundle.scope_receipt[qualification],False)

    def test_close_is_idempotent_and_owns_only_new_search_and_auxiliary(self):
        values=self.inputs();self.mocks(values)
        bundle=self.make(values);bundle.close();bundle.close()
        values.search.close.assert_called_once();values.auxiliary.close.assert_called_once()
        for borrowed in (values.rules,values.policy,values.statistics,values.keys):borrowed.close.assert_not_called()
        self.assertTrue(bundle.closed)

    def test_auxiliary_closes_even_if_owned_search_close_fails(self):
        values=self.inputs();self.mocks(values);bundle=self.make(values)
        values.search.close.side_effect=RuntimeError('search close failed')
        with self.assertRaisesRegex(RuntimeError,'search close failed'):bundle.close()
        values.auxiliary.close.assert_called_once();values.policy.close.assert_not_called()
        bundle.close();values.search.close.assert_called_once()

    def test_search_constructor_failure_releases_constructed_auxiliary(self):
        values=self.inputs();_,_,search=self.mocks(values)
        search.side_effect=RuntimeError('search construction failed')
        with self.assertRaisesRegex(RuntimeError,'construction failed'):self.make(values)
        values.auxiliary.close.assert_called_once()
        values.policy.close.assert_not_called();values.rules.close.assert_not_called()

    def test_scope_rejection_or_mismatched_seven_versions_never_constructs_numeric_bank(self):
        values=self.inputs();loader,bank,search=self.mocks(values)
        loader.side_effect=ValueError('Scope heldout evidence invalid')
        with self.assertRaisesRegex(ValueError,'heldout'):self.make(values)
        bank.assert_not_called();search.assert_not_called()
        loader.side_effect=None
        values.models['starter'].version='wrong-checkpoint'
        with self.assertRaisesRegex(ValueError,'versions'):self.make(values)
        bank.assert_not_called()
        values.models.pop('lingke')
        with self.assertRaisesRegex(ValueError,'seven'):self.make(values)
        bank.assert_not_called()

    def test_wrong_aux_keys_versions_or_device_fail_and_close_auxiliary(self):
        import torch
        for field,value in (('keys',('starter',)),('versions',('wrong',)*7),('device',torch.device('cuda:1'))):
            inputs=self.inputs();_,_,search=self.mocks(inputs)
            setattr(inputs.auxiliary,field,value)
            with self.assertRaises(ValueError):self.make(inputs)
            inputs.auxiliary.close.assert_called_once();search.assert_not_called()
            inputs.policy.close.assert_not_called()

    def test_aliasing_policy_as_auxiliary_is_refused_without_closing_borrowed_policy(self):
        values=self.inputs();_,bank,search=self.mocks(values)
        bank.return_value=values.policy
        with self.assertRaisesRegex(ValueError,'independent'):self.make(values)
        values.policy.close.assert_not_called();search.assert_not_called()

    def test_staged_rule_device_and_bounded_configuration_are_checked_before_loading(self):
        import torch
        for field,value in (('require_teacher',False),('staged_teacher',False),('device',torch.device('cpu'))):
            inputs=self.inputs();loader,_,_=self.mocks(inputs)
            setattr(inputs.rules,field,value)
            with self.assertRaises(ValueError):self.make(inputs)
            loader.assert_not_called()
        inputs=self.inputs();loader,_,_=self.mocks(inputs)
        for options in (dict(inference_rows=0),dict(inference_rows=33),dict(inference_rows=True),
                        dict(max_actions=799),dict(noise=1),dict(row_work_seconds=float('inf')),
                        dict(row_work_seconds=True),dict(queued_row_work=1)):
            with self.assertRaises((ValueError,TypeError)):self.make(inputs,**options)
        loader.assert_not_called()
        with self.assertRaises(TypeError):
            create_soa_covered_search_bundle(object(),inputs.policy,inputs.statistics,inputs.keys,
                auxiliary_scope=inputs.scope,rows=8,random=inputs.random)

    def test_reserved_budget_prior_world_and_cpu_switches_are_never_overridden(self):
        values=self.inputs();loader,_,_=self.mocks(values)
        overrides=dict(simulations=1,gumbel_candidates=1,max_depth=1,teacher_mode='network_value',
            terminal_horizon=0,opponent_mode='policy',prior_guide=object(),reuse_sampled_world=True,
            reuse_root_decision=True,reuse_root_policy_value=True,verify_root_value_reuse=True,
            value_batch_rows=1,allow_cpu_reference=True,allow_synthetic=True,device='cpu')
        for name,value in overrides.items():
            with self.assertRaisesRegex(ValueError,'contract'):self.make(values,**{name:value})
        loader.assert_not_called()

    def test_explicit_performance_options_and_inference_tile_do_not_change_formal_budget(self):
        values=self.inputs();_,_,search=self.mocks(values)
        bundle=self.make(values,inference_rows=32,queued_row_work=False,row_work_seconds=30,
            group_ready_leaves=False,reuse_root_observation=True,reuse_interior_observation=True)
        config=search.call_args.kwargs
        self.assertEqual(config['value_batch_rows'],8)
        self.assertIs(config['queued_row_work'],False);self.assertEqual(config['row_work_seconds'],30)
        self.assertIs(config['group_ready_leaves'],False)
        self.assertIs(config['reuse_root_observation'],True);self.assertIs(config['reuse_interior_observation'],True)
        self.assertEqual(config['simulations'],32);self.assertEqual(config['terminal_horizon'],3)
        self.assertIs(config['reuse_sampled_world'],False);bundle.close()


if __name__=='__main__':unittest.main()
