"""CLI guards for the bounded SoA CUDA acceptance script. No real device run."""
import ast
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CLI = REPO / 'scripts' / 'benchmark_duel_v2_soa_device.py'


def _main(argv):
    from scripts.benchmark_duel_v2_soa_device import main
    return main(argv)


def _stderr(argv):
    buf = io.StringIO()
    try:
        with redirect_stderr(buf):
            _main(argv)
    except SystemExit as exc:
        if exc.code in (0, None):
            raise AssertionError(argv)
        return buf.getvalue()
    raise AssertionError('expected SystemExit for %r' % (argv,))


class SoADeviceCliTests(unittest.TestCase):
    def test_fused_steps_are_explicit_without_changing_search_budget(self):
        buf=io.StringIO()
        with redirect_stdout(buf):self.assertEqual(_main(['--fused-rule-steps']),0)
        result=json.loads(buf.getvalue())
        self.assertTrue(result['search']['fused_rule_steps'])
        self.assertEqual(result['search']['simulations'],32)
        self.assertEqual(result['search']['gumbel_candidates'],16)
        self.assertFalse(result['cuda_imported'])
        self.assertFalse(result['gpu_qualified'])

    def test_wide_lanes_full_native_short_continuous_search_plan(self):
        flags=['--native-complete','--indexed-attributes','--rows','65','--rule-work-rows','64',
               '--cohort-only','--max-decisions','2','--warmup','0','--repeats','1',
               '--execution-seconds','60','--proof-seconds','30']
        buf=io.StringIO()
        with redirect_stdout(buf):self.assertEqual(_main(flags),0)
        result=json.loads(buf.getvalue())
        self.assertEqual((result['rows'],result['capacity']['effective_rule_lanes']),(65,64))
        self.assertEqual(result['max_decisions'],2)
        self.assertTrue(result['cohort_only']);self.assertTrue(result['rule_executor']['native_functions'])
        self.assertFalse(result['rule_executor']['execute_only'])
        self.assertGreaterEqual(result['rule_executor']['stack_bytes'],131072)
        self.assertTrue(result['capacity']['distinct_roots_required'])
        self.assertEqual(result['search']['simulations'],32)
        self.assertFalse(result['capacity']['capacity_qualified'])
        self.assertFalse(result['gpu_qualified']);self.assertFalse(result['passed'])
        self.assertIn('rule-work-rows',_stderr(['--rule-work-rows','65']))
        self.assertIn('workspace',_stderr(['--rows','65','--rule-work-rows','64',
            '--gpu-budget-mib','4096','--state-budget-mib','2048']))

    def test_root_identity_rejects_copies_and_lane_receipt_checks_tail_errors(self):
        import torch as t
        from types import SimpleNamespace as NS
        from scripts.benchmark_duel_v2_soa_device import root_identity_receipt,work_lane_receipt
        roots=NS(slots=t.arange(65),generations=t.ones(65,dtype=t.int64),owners=t.arange(65))
        jobs=[dict(seed=100+i) for i in range(65)]
        result=root_identity_receipt(roots,jobs)
        self.assertEqual(result['independent_roots'],65)
        self.assertFalse(result['copied_state_capacity_qualified'])
        jobs[1]['seed']=jobs[0]['seed']
        with self.assertRaisesRegex(AssertionError,'Distinct'):root_identity_receipt(roots,jobs)
        q=NS(total_rows=t.tensor([65]),total_dispatches=t.tensor([2]),peak_selected=t.tensor([64]),
             root_visits=t.ones(65,dtype=t.int64),tail_dispatches=t.tensor([1]),
             errored_rows=t.tensor([0]),loop=NS(errors=t.tensor([0],dtype=t.int32)))
        rules=NS(rows=65,work_rows=64,_queues=[q])
        receipt=work_lane_receipt(rules)
        self.assertEqual(receipt['sites'][0]['tail_dispatches'],1)
        self.assertEqual(receipt['distinct_roots'],65)
        self.assertFalse(receipt['capacity_qualified'])
        q.loop.errors[0]=1
        with self.assertRaisesRegex(AssertionError,'tail/error'):work_lane_receipt(rules)

    def test_complete_batch_lease_receipt_rejects_orphans_and_stale_owner(self):
        import torch as t
        from types import SimpleNamespace as NS
        from scripts.benchmark_duel_v2_soa_device import completed_lease_receipt
        store=NS(live=t.tensor([True,True,False]),generation=t.tensor([2,3,0]),owner=t.tensor([7,8,-1]))
        actions=NS(live=t.zeros(3,dtype=t.bool))
        states=NS(slots=t.tensor([0,1]),generations=t.tensor([2,3]),owners=t.tensor([7,8]))
        self.assertEqual(completed_lease_receipt(store,actions,states,2)['checked_physical_leases'],2)
        store.live[2]=True
        with self.assertRaisesRegex(AssertionError,'leaked'):completed_lease_receipt(store,actions,states,2)
        store.live[2]=False;states.owners[1]=9
        with self.assertRaisesRegex(AssertionError,'lease mismatch'):completed_lease_receipt(store,actions,states,2)

    def test_native_complete_and_attribute_indexing_are_explicit_in_plan(self):
        for flags,native,indexed in (([],False,False),(['--native-complete'],True,False),
                                     (['--native-complete','--indexed-attributes'],True,True)):
            buf=io.StringIO()
            with redirect_stdout(buf):self.assertEqual(_main(flags),0)
            result=json.loads(buf.getvalue())
            self.assertEqual(result['rule_executor'],dict(native_functions=native,native_values=native,
                separate_compilation=native,indexed_attributes=indexed,indexed_entity_queries=False,
                runtime_primitive_compare=False,direct_deepcopy_calls=False,
                execute_only=False,stack_bytes=131072))
            self.assertEqual(result['search']['simulations'],32)
            self.assertEqual(result['search']['gumbel_candidates'],16)
            self.assertFalse(result['passed']);self.assertFalse(result['gpu_qualified'])
            self.assertFalse(result['cuda_imported']);self.assertFalse(result['models_loaded'])
            self.assertEqual(result['learning_updates'],0)
        self.assertIn('--native-complete',_stderr(['--indexed-attributes']))
        self.assertIn('--native-complete',_stderr(['--indexed-entity-queries']))
        buf=io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(_main(['--native-complete','--indexed-entity-queries']),0)
        self.assertTrue(json.loads(buf.getvalue())['rule_executor']['indexed_entity_queries'])

    def test_native_plan_options_reach_actual_full_module_factory(self):
        from unittest.mock import Mock
        from scripts.benchmark_duel_v2_soa_device import (
            build_parser,create_rule_executor,rule_executor_options)
        for flags in ([],['--native-complete'],['--native-complete','--indexed-attributes'],
                ['--native-complete','--runtime-primitive-compare'],
                ['--native-complete','--direct-deepcopy-calls'],
                ['--native-complete','--runtime-primitive-compare','--direct-deepcopy-calls']):
            args=build_parser().parse_args(flags)
            factory=Mock();image=object();schema=object();limits=object()
            executor=create_rule_executor(factory,image,schema,args,device='cuda:0',limits=limits)
            self.assertIs(executor,factory.return_value)
            factory.assert_called_once_with(image,schema,device='cuda:0',limits=limits,
                **rule_executor_options(args))
            kwargs=factory.call_args.kwargs
            self.assertIs(kwargs['execute_only'],False)
            self.assertGreaterEqual(kwargs['stack_bytes'],131072)
            for name in ('direct_runtime_calls','indexed_container_methods','direct_type_queries'):
                self.assertNotIn(name,kwargs)
            self.assertEqual(kwargs['runtime_primitive_compare'],'--runtime-primitive-compare' in flags)
            self.assertEqual(kwargs['direct_deepcopy_calls'],'--direct-deepcopy-calls' in flags)

    def test_optional_runtime_routes_require_native_mode_and_are_bound_in_metadata(self):
        for flag,name in (('--runtime-primitive-compare','runtime_primitive_compare'),
                ('--direct-deepcopy-calls','direct_deepcopy_calls')):
            self.assertIn('--native-complete',_stderr([flag]))
            output=io.StringIO()
            with redirect_stdout(output):self.assertEqual(_main(['--native-complete',flag]),0)
            metadata=json.loads(output.getvalue())
            self.assertIs(metadata['rule_executor'][name],True)
            self.assertIs(metadata['passed'],False)
            self.assertIs(metadata['gpu_qualified'],False)

    def test_optional_runtime_routes_reject_invalid_cuda_options_before_allocation(self):
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_duel.native_soa_cuda import CudaSoARules
        with patch('torch.cuda.is_available',return_value=True):
            for name in ('runtime_primitive_compare','direct_deepcopy_calls'):
                for value in (0,1,None,'true'):
                    with self.assertRaisesRegex(ValueError,'Runtime specialization'):
                        CudaSoARules(None,None,**{name:value})
                with self.assertRaisesRegex(ValueError,'separate compilation'):
                    CudaSoARules(None,None,**{name:True})
                with self.assertRaisesRegex(ValueError,'complete native functions'):
                    CudaSoARules(None,None,separate_compilation=True,**{name:True})

    def test_query_forwards_explicit_runtime_routes_without_changing_scratch(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        from app.modules.card_game.rl.batched_duel.native_soa_cuda_query import NativeSoACUDAQuery
        image=SimpleNamespace(nodes=np.zeros((1,4),dtype=np.int32),
            edges=np.zeros((1,2),dtype=np.int32),payload=np.zeros(1,dtype=np.uint8))
        with patch('app.modules.card_game.rl.batched_duel.native_soa_cuda_query.CudaSoARules') as factory:
            with NativeSoACUDAQuery(image,None,native_functions=True,native_values=True,
                    separate_compilation=True,runtime_primitive_compare=True,direct_deepcopy_calls=True) as query:
                options=factory.call_args.kwargs
                self.assertIs(options['runtime_primitive_compare'],True)
                self.assertIs(options['direct_deepcopy_calls'],True)
                self.assertIs(options['separate_compilation'],True)
                self.assertEqual(query.max_nodes,1048576)
                self.assertEqual(options['stack_bytes'],131072)

    def test_runtime_routes_compose_once_and_keep_other_object_sources_identical(self):
        from unittest.mock import patch
        import torch
        from app.modules.card_game.rl.batched_duel.native_soa_cuda import CudaSoARules
        from app.modules.card_game.rl.batched_duel.native_runtime_calls import transform_runtime_calls
        from app.modules.card_game.rl.batched_duel.native_runtime_compare import specialize_runtime_compare
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from tests.test_duel_v2_native_functions import fixture_program,fixture_schema
        image=lower_program(fixture_program());schema=fixture_schema()
        tensor,full=torch.tensor,torch.full
        def cpu_tensor(*args,**kwargs):
            kwargs.pop('device',None);return tensor(*args,**kwargs)
        def cpu_full(*args,**kwargs):
            kwargs.pop('device',None);return full(*args,**kwargs)
        with patch('torch.cuda.is_available',return_value=True), \
                patch('torch.cuda.current_device',return_value=0), \
                patch('torch.cuda.is_current_stream_capturing',return_value=False), \
                patch('torch.tensor',side_effect=cpu_tensor),patch('torch.full',side_effect=cpu_full), \
                patch('app.modules.card_game.rl.batched_duel.cuda_runtime.CudaModule') as factory:
            original=CudaSoARules(image,schema,native_functions=True,native_values=True,
                separate_compilation=True,indexed_attributes=True)
            baseline=factory.call_args.kwargs
            with patch('app.modules.card_game.rl.batched_duel.native_runtime_calls.transform_runtime_calls',
                    wraps=transform_runtime_calls) as route, \
                    patch('app.modules.card_game.rl.batched_duel.native_runtime_compare.specialize_runtime_compare',
                    wraps=specialize_runtime_compare) as comparison:
                candidate=CudaSoARules(image,schema,native_functions=True,native_values=True,
                    separate_compilation=True,indexed_attributes=True,direct_runtime_calls=True,
                    direct_type_queries=True,direct_deepcopy_calls=True,runtime_primitive_compare=True)
                configured=factory.call_args.kwargs
                self.assertEqual(route.call_count,1)
                self.assertEqual(comparison.call_count,1)
                self.assertIs(route.call_args.kwargs['deepcopy_calls'],True)
                self.assertIs(route.call_args.kwargs['function_calls'],True)
            self.assertEqual(baseline['compilation_unit_specs'],configured['compilation_unit_specs'])
            self.assertNotEqual(baseline['compilation_units'][0],configured['compilation_units'][0])
            self.assertEqual(baseline['compilation_units'][1:],configured['compilation_units'][1:])
            original.close();candidate.close()

    def test_search_cli_does_not_expose_execute_only_or_small_stack(self):
        for flags in (['--execute-only'],['--stack-bytes','65536'],['--direct-runtime-calls'],
                      ['--indexed-container-methods'],['--direct-type-queries']):
            self.assertIn('unrecognized arguments',_stderr(flags))

    def test_narrow_indices_require_the_supported_dispatch_and_are_explicit_in_plan(self):
        self.assertIn('tiled',_stderr(['--candidate-int32','--rule-dispatch','direct']))
        buf=io.StringIO()
        with redirect_stdout(buf):self.assertEqual(_main(['--candidate-int32']),0)
        self.assertEqual(json.loads(buf.getvalue())['candidate_index_bits'],32)

    def test_freeze_check_detects_resident_tensor_changes_and_gradient_inputs(self):
        import torch
        from types import SimpleNamespace
        from scripts.benchmark_duel_v2_soa_device import inference_weight_digest
        bank=SimpleNamespace(weights={'a':torch.tensor([1.,2.])},legacy_weights={'b':torch.tensor([3.])})
        before=inference_weight_digest(bank)
        self.assertEqual(before,inference_weight_digest(bank))
        bank.legacy_weights['b'][0]+=1
        self.assertNotEqual(before,inference_weight_digest(bank))
        bank.weights['a'].requires_grad_(True)
        with self.assertRaises(ValueError):inference_weight_digest(bank)

    def test_memory_budget_rejects_driver_use_and_allocator_cache_separately(self):
        from scripts.benchmark_duel_v2_soa_device import check_memory_budget
        from app.modules.card_game.rl.batched_search.soa_capacity import SoACapacityConfig
        cfg=SoACapacityConfig(gpu_budget_mib=4096,state_budget_mib=2048)
        receipt=dict(device_used=3<<30,reserved=2<<30,process_peak_reserved=2<<30,
                     action_store=100<<20)
        check_memory_budget(receipt,cfg)
        for field in ('device_used','reserved','process_peak_reserved'):
            with self.assertRaises(MemoryError):
                check_memory_budget(dict(receipt,**{field:(4096<<20)+1}),cfg)
        with self.assertRaises(MemoryError):
            check_memory_budget(dict(receipt,action_store=(512<<20)+1),cfg)
        with self.assertRaises(MemoryError):
            check_memory_budget(dict(receipt,training_tape=(512<<20)+1),cfg)

    def test_lane_receipt_checks_tail_coverage_without_counting_work_slots_as_roots(self):
        import torch
        from types import SimpleNamespace as NS
        from scripts.benchmark_duel_v2_soa_device import work_lane_receipt
        q=NS(total_rows=torch.tensor([3]),total_dispatches=torch.tensor([2]),
             peak_selected=torch.tensor([2]),root_visits=torch.tensor([1,1,1]))
        rules=NS(rows=3,work_rows=2,_queues=[q])
        result=work_lane_receipt(rules)
        self.assertEqual((result['roots'],result['work_rows']),(3,2))
        q.root_visits=torch.tensor([2,1,0])
        with self.assertRaisesRegex(AssertionError,'never reached'):
            work_lane_receipt(rules)
        q.root_visits=torch.tensor([1,1,1]);q.peak_selected[0]=3
        with self.assertRaisesRegex(AssertionError,'coverage'):
            work_lane_receipt(rules)

    def test_direct_script_large_capacity_plan_without_cuda(self):
        result=subprocess.run([sys.executable,str(CLI),'--rows','1600'],
            cwd=REPO,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        plan=json.loads(result.stdout)
        self.assertEqual(plan['rows'],1600)
        self.assertEqual(plan['capacity']['effective_rule_lanes'],8)
        self.assertEqual(plan['capacity']['effective_inference_rows'],2)
        self.assertFalse(plan['cuda_imported'])
        self.assertFalse(plan['passed'])

    def test_cohort_only_requires_multiple_physical_decisions(self):
        self.assertIn('cohort-only',_stderr(['--cohort-only']))
        buf=io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(_main(['--cohort-only','--max-decisions','4']),0)
        plan=json.loads(buf.getvalue())
        self.assertTrue(plan['cohort_only'])
        self.assertFalse(plan['passed'])

    def test_training_capture_is_one_cohort_not_repeated_timing_exports(self):
        self.assertIn('--cohort-only',_stderr(['--capture-training']))
        buf=io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(_main(['--capture-training','--cohort-only','--max-decisions','4']),0)
        plan=json.loads(buf.getvalue());self.assertTrue(plan['capture_training'])
        self.assertFalse(plan['passed'])

    def test_numerical_receipt_with_numpy_reference_q_serializes(self):
        import time
        import numpy as np
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts.benchmark_duel_v2_soa_device import verify_last_searches, save_report
        # Import aliases before mocking the public entry; first import inside
        # the patch would permanently bind information_search.acting_side to it.
        from app.modules.card_game.rl import recovery_search, information_search
        from app.modules.card_game.engine.duel_v2 import acting_side
        tensor = lambda x: torch.tensor(x, dtype=torch.float64)
        cohort = SimpleNamespace(
            diagnostics=dict(q=tensor([[.2, .3]]), prior=tensor([[.5, .5]]),
                             mean=tensor([[.2, .3]]), value=tensor([.25]),
                             gumbel=tensor([[0., 0.]]), errors=torch.tensor([False])),
            light=SimpleNamespace(_rec_count=torch.tensor([2]), _rec_offset=torch.tensor([0]),
                                  _packed_visits=torch.tensor([16, 16]),
                                  _packed_pi=tensor([.5, .5]), _rec_chosen=torch.tensor([0])))
        result=dict(complete=True, visits=np.array([16, 16]), pi=np.array([.5, .5]),
                    root_diagnostics={'completed_q':np.array([.2, .3])},
                    root_prior=np.array([.5, .5]), mean_values=np.array([.2, .3]),
                    value=.25, root_gumbel=np.array([0., 0.]), search_choice=0)
        shared=dict(inference=None, models={'m':object()},
                    jobs=[dict(seed=1, keys={'a':'m', 'b':'m'})])
        with patch('app.modules.card_game.engine.duel_v2.acting_side', return_value='a'), \
             patch('app.modules.card_game.rl.recovery_search.search', return_value=result):
            rows=verify_last_searches(cohort, shared, [{}], [1], {(0, 0):0}, time.monotonic()+10)
        self.assertIs(information_search.acting_side, acting_side)
        with tempfile.TemporaryDirectory() as directory:
            save_report(Path(directory), {'postbatch':rows}, 'complete', time.monotonic())
            saved=json.loads((Path(directory)/'report.json').read_text())
        self.assertEqual(saved['postbatch'][0]['reference_completed_q'], [.2, .3])
        self.assertTrue(all(saved['postbatch'][0]['checks'].values()))

    def test_postbatch_value_oracle_recomputes_and_rejects_changed_forward(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from scripts.benchmark_duel_v2_soa_device import PostbatchValueOracle
        expected = np.array([.2, .3, .5], dtype=np.float32)
        class Bank:
            device = 'cpu'
            max_rows = 2
            candidate_dim = 3
            keys = ('frozen',)
            calls = 0
            changed = False
            def predict_device(self, x, c, valid, ids, context, *, value_only):
                self.calls += 1
                self.context = context.clone()
                self.features = x.clone()
                self_empty = c.shape == (2, 0, 3) and valid.shape == (2, 0)
                assert self_empty and value_only.all()
                p = torch.as_tensor(expected).repeat(2, 1)
                if self.changed:
                    p[:, 0] += .01
                return SimpleNamespace(wdl=p, invalid_rows=torch.zeros(2, dtype=torch.bool))
        bank = Bank()
        model = SimpleNamespace(wdl=lambda x, actor: expected,
                                scores=lambda x, c: np.arange(len(c)))
        oracle = PostbatchValueOracle(bank, {'frozen': model})
        policy = oracle.policy('frozen')
        x = np.array([1., 2., 3.], dtype=np.float32)
        np.testing.assert_array_equal(policy.wdl(x, False), expected)
        np.testing.assert_array_equal(bank.features.numpy(), np.stack([x, x]))
        self.assertFalse(bank.context.any())
        policy.wdl(x.copy(), False)
        self.assertEqual(bank.calls, 1)
        _, value = policy.scores_value(x, [0, 1])
        self.assertAlmostEqual(value, .3)
        self.assertTrue(bank.context.all())
        bank.changed = True
        with self.assertRaisesRegex(AssertionError, 'WDL component parity'):
            policy.wdl(x + 1, True)

    def test_module_import_does_not_load_cuda(self):
        result = subprocess.run(
            [sys.executable, '-c',
             "import sys; import scripts.benchmark_duel_v2_soa_device as m; "
             "assert 'torch' not in sys.modules; assert m.SEARCH['simulations']==32"],
            cwd=REPO, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_plan_has_no_cuda_models_or_files(self):
        before = sys.modules.get('torch')
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'unused-out'
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(_main(['--output', str(target)]), 0)
            self.assertIs(sys.modules.get('torch'), before)
            self.assertFalse(target.exists())
            plan = json.loads(buf.getvalue())
        self.assertEqual(plan['mode'], 'plan')
        self.assertFalse(plan['cuda_imported'])
        self.assertFalse(plan['models_loaded'])
        self.assertFalse(plan['files_created'])
        self.assertFalse(plan['passed'])
        self.assertFalse(plan['gpu_qualified'])
        self.assertEqual(plan['learning_updates'], 0)
        self.assertEqual(plan['rows'], 8)
        self.assertEqual(plan['max_decisions'], 1)
        self.assertEqual(plan['warmup'], 2)
        self.assertEqual(plan['repeats'], 3)
        self.assertFalse(plan['require_natural'])
        self.assertEqual(plan['search'], {
            'simulations': 32, 'gumbel_candidates': 16, 'terminal_horizon': 0,
            'teacher_mode': 'network_value', 'opponent_mode': 'policy',
            'prior': 'array_zhenhong', 'extra_guidance_simulations': 0,
            'cpu_fallback': False,
            'world_sampling': 'per_decision', 'fused_rule_steps': False,
        })
        self.assertEqual(plan['presets'], [
            'starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk', 'collapse', 'lingke'])
        self.assertTrue(plan['mixed_first_sides'])

    def test_plan_rejects_cpu_and_out_of_range_schema(self):
        self.assertIn('CUDA-only', _stderr(['--device', 'cpu']))
        self.assertIn('rows', _stderr(['--rows', '0']))
        self.assertIn('rows', _stderr(['--rows', '1601']))
        self.assertIn('Direct SoA', _stderr(['--rows','128','--rule-dispatch','direct']))
        self.assertIn('max-decisions', _stderr(['--max-decisions', '0']))
        self.assertIn('max-decisions', _stderr(['--max-decisions', '401']))
        self.assertIn('warmup', _stderr(['--warmup', '-1']))
        self.assertIn('repeats', _stderr(['--repeats', '0']))
        self.assertIn('setup-seconds', _stderr(['--setup-seconds', '0']))
        self.assertIn('execution-seconds', _stderr(['--execution-seconds', '86401']))
        self.assertIn('seed',_stderr(['--seed','-1']))

    def test_run_requires_frozen_inputs_and_new_evals_directory(self):
        self.assertIn('--program --models --teachers --output', _stderr(['--run']))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            program = root / 'frozen.json'
            program.write_text('{}', encoding='utf8')
            models = root / 'models'
            teachers = root / 'teachers'
            models.mkdir()
            teachers.mkdir()
            self.assertIn('artifacts/rl-evals', _stderr([
                '--run', '--program', str(program), '--models', str(models),
                '--teachers', str(teachers), '--output', str(root / 'out')]))
            existing = REPO / 'artifacts' / 'rl-evals'
            self.assertTrue(existing.is_dir())
            self.assertIn('Use a new output directory', _stderr([
                '--run', '--program', str(program), '--models', str(models),
                '--teachers', str(teachers), '--output', str(existing)]))

    def test_fixed_search_semantics_are_wired_in_source(self):
        source = CLI.read_text(encoding='utf8')
        tree = ast.parse(source)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertIn('create_soa_search', names)
        self.assertIn('DeviceGameLoop', names)
        self.assertIn('DeviceSoARules', names)
        self.assertIn('RawActionPolicySink', names)
        self.assertIn('DeviceValueOnlyInference', names)
        self.assertIn('bind_game_loop', attrs)
        self.assertNotIn('DeviceZhenhongGuide', names)
        self.assertNotIn('DeviceBatchTeacherActions', names)
        self.assertNotIn('DeviceFrozenRules', names)
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'create_soa_search']
        self.assertTrue(calls)
        reserved = {'teacher_mode', 'terminal_horizon', 'opponent_mode', 'prior_guide',
                    'simulations', 'gumbel_candidates', 'device'}
        for call in calls:
            found = {kw.arg for kw in call.keywords}
            self.assertFalse(found & reserved)
            self.assertIn('zhenhong_prior', found)
        rules_calls = [node for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'DeviceSoARules']
        self.assertTrue(any('raw_tape' in {kw.arg for kw in call.keywords} for call in rules_calls))

    def test_plan_does_not_create_output_even_under_evals_name(self):
        target = REPO / 'artifacts' / 'rl-evals' / 'soa-device-cli-plan-should-not-exist'
        self.assertFalse(target.exists())
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(_main(['--output', str(target), '--rows', '8']), 0)
        self.assertFalse(target.exists())
        plan = json.loads(buf.getvalue())
        self.assertEqual(plan['mode'], 'plan')
        self.assertFalse(plan['files_created'])

    def test_require_natural_is_plan_only_until_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(_main(['--require-natural', '--max-decisions', '16']), 0)
        plan = json.loads(buf.getvalue())
        self.assertTrue(plan['require_natural'])
        self.assertEqual(plan['max_decisions'], 16)
        self.assertFalse(plan['passed'])
        self.assertFalse(plan['cuda_imported'])

    def test_root_diagnostic_capture_uses_real_statistics_and_survives_release(self):
        import torch
        from types import SimpleNamespace
        from scripts.benchmark_duel_v2_soa_device import install_root_diagnostics
        from app.modules.card_game.rl.batched_search.device_statistics import DeviceStatisticsPool
        stats=DeviceStatisticsPool(2,3,device='cpu',allow_cpu_reference=True,max_batch=2,max_path_len=2)
        slots=torch.tensor([0,1],dtype=torch.int64)
        generations=torch.ones(2,dtype=torch.int64)
        active=torch.tensor([True,False])
        written=stats.write_nodes(slots,torch.tensor([1,2]),torch.tensor([[.6,.4,0.],[.2,.3,.5]],dtype=torch.float64),
            torch.tensor([.2,-.2],dtype=torch.float64),torch.tensor([[True,True,False],[True,True,True]]),
            expected_generations=generations)
        self.assertFalse(written.error.any())
        stats._visits[0,:2]=torch.tensor([2,1],dtype=torch.int32)
        stats._total[0,:2]=torch.tensor([.5,-.2],dtype=torch.float64)
        noise=torch.zeros((2,3),dtype=torch.float64)
        cohort=SimpleNamespace(stats=stats,search=SimpleNamespace(_noise=noise))
        expected=stats.completed_q(slots,written.generations,active=active).completed_q.clone()
        install_root_diagnostics(cohort,torch,2,3)
        stats.snapshot(slots,written.generations,active=active)
        self.assertTrue(torch.equal(cohort.diagnostics['q'],expected))
        self.assertFalse(cohort.diagnostics['errors'].any())
        stats.release_nodes(slots,written.generations)
        self.assertTrue(torch.equal(cohort.diagnostics['q'],expected))


if __name__ == '__main__':
    unittest.main()
