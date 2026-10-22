"""CPU oracle and optional CUDA checks for domain action grouping/scatter."""
from unittest.mock import patch
import unittest
from dataclasses import replace

import torch

from app.modules.card_game.rl.batched_duel.domain_soa_action_batch import (
    COUNT_OVERFLOW, DEFAULT_MAX_ROOTS, DUPLICATE_RESULT, FOREIGN_RESULT,
    INCOMPLETE_GROUP, STALE_RESULT, UNKNOWN_KIND, DomainActionJobResults,
    DomainHandlerSpec, DomainSoAActionBatch, cpu_oracle, supports_full_duel_rules,
)


HANDLERS = (
    DomainHandlerSpec(1, 10),
    DomainHandlerSpec(2, 7),
    DomainHandlerSpec(3, 10),
)


def _cuda_available():
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def batch(device='cpu', max_roots=4, max_candidates=4):
    if device == 'cpu':
        return cpu_oracle(HANDLERS, max_roots=max_roots, max_candidates=max_candidates)
    return DomainSoAActionBatch(
        HANDLERS, max_roots=max_roots, max_candidates=max_candidates, device=device)


def tensors(rows, width, device='cpu'):
    tokens = torch.zeros((rows, width), dtype=torch.int32, device=device)
    counts = torch.full((rows,), width, dtype=torch.int32, device=device)
    mask = torch.ones((rows, width), dtype=torch.bool, device=device)
    selected = torch.zeros((rows, width), dtype=torch.bool, device=device)
    active = torch.ones(rows, dtype=torch.bool, device=device)
    owners = torch.arange(rows, dtype=torch.int64, device=device)
    generations = torch.ones(rows, dtype=torch.int64, device=device)
    return tokens, counts, mask, selected, active, owners, generations


def compact_jobs(queue):
    present = queue.present.tolist()
    jobs = []
    for index, flag in enumerate(present):
        if not flag:
            continue
        jobs.append({
            'handler': int(queue.handler_id[index]),
            'root': int(queue.root_index[index]),
            'action': int(queue.action_index[index]),
            'ordinal': int(queue.candidate_ordinal[index]),
            'owner': int(queue.owners[index]),
            'generation': int(queue.generations[index]),
            'group': int(queue.group_id[index]),
            'group_size': int(queue.group_size[index]),
            'order': int(queue.original_order[index]),
        })
    return jobs


def results_from_queue(arena, queue, values):
    present = queue.present.clone()
    packed = torch.zeros(arena.work_width, dtype=torch.int32, device=arena.device)
    packed = torch.where(present, values.to(device=arena.device, dtype=torch.int32), packed)
    return DomainActionJobResults(
        present=present,
        values=packed,
        root_index=queue.root_index.clone(),
        action_index=queue.action_index.clone(),
        owners=queue.owners.clone(),
        generations=queue.generations.clone(),
        handler_id=queue.handler_id.clone(),
        candidate_ordinal=queue.candidate_ordinal.clone(),
        epoch=queue.epoch.clone(),
    )


class DomainActionBatchCPUTests(unittest.TestCase):
    def setUp(self):
        self.arena = batch()

    def test_refuses_full_duel_rules_and_reports_capacity_bytes(self):
        self.assertFalse(supports_full_duel_rules)
        self.assertFalse(DomainSoAActionBatch.supports_full_duel_rules)
        self.assertFalse(self.arena.supports_full_duel_rules)
        self.assertGreater(self.arena.capacity_bytes, 0)
        self.assertEqual(self.arena.capacity_bytes, self.arena.allocated_bytes)
        self.assertEqual(self.arena.max_roots, 4)
        self.assertLessEqual(self.arena.max_roots, DEFAULT_MAX_ROOTS)

    def test_cpu_required_explicit_oracle(self):
        with self.assertRaisesRegex(ValueError, 'explicit action-batch oracle'):
            DomainSoAActionBatch(HANDLERS, max_roots=2, max_candidates=4, device='cpu')

    def test_groups_by_handler_preserving_root_action_order(self):
        tokens, counts, mask, selected, active, owners, generations = tensors(3, 4)
        tokens[0] = torch.tensor([1, 2, 3, 0])
        tokens[1] = torch.tensor([2, 1, 0, 0])
        tokens[2] = torch.tensor([3, 3, 2, 1])
        counts[2] = 3
        mask[2, 3] = False
        selected[0, :3] = True
        selected[1, :2] = True
        selected[2, :2] = True
        active[2] = False
        snapshot = [value.clone() for value in (
            tokens, counts, mask, selected, active, owners, generations)]
        queue = self.arena.group_selected_actions(
            tokens, counts, mask, selected, active, owners, generations,preview=True)
        self.assertEqual([value.tolist() for value in snapshot],
                         [tokens.tolist(), counts.tolist(), mask.tolist(),
                          selected.tolist(), active.tolist(), owners.tolist(),
                          generations.tolist()])
        self.assertEqual(int(queue.compact_count), 5)
        self.assertEqual(queue.errors.tolist(), [0, 0, 0])
        jobs = compact_jobs(queue)
        self.assertEqual([(j['handler'], j['root'], j['action']) for j in jobs], [
            (7, 0, 1), (7, 1, 0), (10, 0, 0), (10, 0, 2), (10, 1, 1),
        ])
        self.assertEqual(jobs[0]['group_size'], 2)
        self.assertEqual(jobs[2]['group_size'], 3)
        self.assertEqual(jobs[2]['group'], jobs[4]['group'])
        self.assertNotEqual(jobs[0]['group'], jobs[2]['group'])

    def test_unknown_kind_and_count_overflow_fail_closed(self):
        tokens, counts, mask, selected, active, owners, generations = tensors(3, 4)
        tokens[0] = torch.tensor([1, 9, 2, 0])
        tokens[1] = torch.tensor([1, 1, 1, 1])
        tokens[2] = torch.tensor([2, 2, 2, 2])
        selected[0, :2] = True
        selected[1, :2] = True
        selected[2, :1] = True
        counts[1] = 5
        mask[2, 3] = False
        counts[2] = 4
        queue = self.arena.group_selected_actions(
            tokens, counts, mask, selected, active, owners, generations,preview=True)
        self.assertEqual(queue.errors.tolist(), [UNKNOWN_KIND, COUNT_OVERFLOW, COUNT_OVERFLOW])
        self.assertEqual(int(queue.compact_count), 0)
        self.assertFalse(bool(queue.present.any()))

    def test_preview_keeps_multiple_jobs_and_ordinal_tie_order(self):
        tokens, counts, mask, selected, active, owners, generations = tensors(2, 4)
        tokens[0] = torch.tensor([1, 1, 2, 0])
        tokens[1] = torch.tensor([2, 1, 1, 0])
        selected[0, :3] = True
        selected[1, :2] = True
        ordinals = torch.tensor([[2, 0, 1, 3], [0, 3, 1, 2]], dtype=torch.int32)
        before = tokens.clone()
        queue = self.arena.group_selected_actions(
            tokens, counts, mask, selected, active, owners, generations,
            candidate_ordinals=ordinals, preview=True)
        self.assertTrue(queue.preview)
        self.assertEqual(tokens.tolist(), before.tolist())
        jobs = compact_jobs(queue)
        self.assertEqual([(j['handler'], j['root'], j['action'], j['ordinal']) for j in jobs], [
            (7, 0, 2, 1), (7, 1, 0, 0), (10, 0, 1, 0), (10, 0, 0, 2), (10, 1, 1, 3),
        ])
        values = torch.zeros(self.arena.work_width, dtype=torch.int32)
        for index, job in enumerate(jobs):
            values[index] = 100 + index
        scattered = self.arena.scatter_results(queue, results_from_queue(self.arena, queue, values))
        self.assertEqual(scattered.errors.tolist(), [0, 0])
        self.assertEqual(scattered.values.tolist(), [
            [103, 102, 100, 0],
            [101, 104, 0, 0],
        ])
        self.assertEqual(scattered.present.tolist(), [
            [True, True, True, False],
            [True, True, False, False],
        ])

    def test_stale_foreign_duplicate_and_incomplete_fail_closed(self):
        tokens, counts, mask, selected, active, owners, generations = tensors(2, 4)
        tokens[0, 0] = 1
        tokens[1, 0] = 2
        selected[:, 0] = True
        queue = self.arena.group_selected_actions(
            tokens, counts, mask, selected, active, owners, generations)
        base = results_from_queue(
            self.arena, queue, torch.tensor([11, 22] + [0] * (self.arena.work_width - 2)))
        stale = DomainActionJobResults(
            present=base.present.clone(), values=base.values.clone(),
            root_index=base.root_index.clone(), action_index=base.action_index.clone(),
            owners=base.owners.clone(),
            generations=base.generations.clone() + 3,
            handler_id=base.handler_id.clone())
        out = self.arena.scatter_results(queue, stale)
        self.assertTrue((out.errors & STALE_RESULT).all())
        self.assertFalse(bool(out.present.any()))
        self.assertEqual(out.values.tolist(), [[0, 0, 0, 0], [0, 0, 0, 0]])

        foreign = DomainActionJobResults(
            present=base.present.clone(), values=base.values.clone(),
            root_index=base.root_index.clone(), action_index=base.action_index.clone(),
            owners=base.owners.clone() + 9,
            generations=base.generations.clone(),
            handler_id=base.handler_id.clone())
        out = self.arena.scatter_results(queue, foreign)
        self.assertTrue((out.errors & FOREIGN_RESULT).all())
        self.assertFalse(bool(out.present.any()))

        dup_present = base.present.clone()
        dup_root = base.root_index.clone()
        dup_action = base.action_index.clone()
        dup_present[1] = True
        dup_root[1] = dup_root[0]
        dup_action[1] = dup_action[0]
        duplicate = DomainActionJobResults(
            present=dup_present, values=base.values.clone(),
            root_index=dup_root, action_index=dup_action,
            owners=base.owners.clone(), generations=base.generations.clone(),
            handler_id=base.handler_id.clone())
        out = self.arena.scatter_results(queue, duplicate)
        self.assertTrue((out.errors & DUPLICATE_RESULT).any() or (out.errors & INCOMPLETE_GROUP).any())
        self.assertFalse(bool(out.present.any()))

        incomplete = DomainActionJobResults(
            present=torch.zeros_like(base.present), values=base.values.clone(),
            root_index=base.root_index.clone(), action_index=base.action_index.clone(),
            owners=base.owners.clone(), generations=base.generations.clone(),
            handler_id=base.handler_id.clone())
        incomplete.present[0] = True
        out = self.arena.scatter_results(queue, incomplete)
        self.assertTrue((out.errors & INCOMPLETE_GROUP).all())
        self.assertFalse(bool(out.present.any()))

    def test_hot_path_does_not_sync_or_loop_to_host(self):
        tokens, counts, mask, selected, active, owners, generations = tensors(2, 4)
        tokens[:, 0] = 1
        selected[:, 0] = True
        queue = self.arena.group_selected_actions(
            tokens, counts, mask, selected, active, owners, generations)
        packed = results_from_queue(
            self.arena, queue, torch.tensor([4, 5] + [0] * (self.arena.work_width - 2)))
        banned = AssertionError('host sync')
        with patch.object(torch.Tensor, 'item', side_effect=banned), \
                patch.object(torch.Tensor, 'cpu', side_effect=banned), \
                patch.object(torch.Tensor, 'numpy', side_effect=banned), \
                patch.object(torch.Tensor, 'tolist', side_effect=banned):
            again = self.arena.group_selected_actions(
                tokens, counts, mask, selected, active, owners, generations)
            self.arena.scatter_results(again, packed)
        self.assertEqual(int(again.compact_count), 2)

    def test_unavailable_cuda_is_refused(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'Actual CUDA required'):
                DomainSoAActionBatch(HANDLERS, max_roots=2, max_candidates=4, device='cuda')

    def test_full_800_legal_actions_keep_large_ordinals_and_roots_separate(self):
        arena=batch(max_roots=2,max_candidates=800)
        args=list(tensors(2,800));args[0].fill_(1);args[3][:,799]=True
        queue=arena.group_selected_actions(*args)
        self.assertEqual([(v['root'],v['ordinal'])for v in compact_jobs(queue)],[(0,799),(1,799)])
        values=torch.arange(arena.work_width,dtype=torch.int32)
        out=arena.scatter_results(queue,results_from_queue(arena,queue,values))
        self.assertEqual(out.errors.tolist(),[0,0])
        self.assertTrue(out.present[:,799].all())

    def test_old_same_root_results_and_wrong_preview_ordinal_are_rejected(self):
        args=list(tensors(2,4));args[0].fill_(1);args[3][:,0]=True
        old=self.arena.group_selected_actions(*args)
        values=torch.arange(self.arena.work_width,dtype=torch.int32)
        stale=results_from_queue(self.arena,old,values)
        new=self.arena.group_selected_actions(*args)
        out=self.arena.scatter_results(new,stale)
        self.assertTrue((out.errors&STALE_RESULT).all())
        self.assertFalse(out.present.any())
        # A self-consistent old queue/result pair is stale after reissuance too.
        out=self.arena.scatter_results(old,stale)
        self.assertTrue((out.errors&STALE_RESULT).all())
        self.assertFalse(out.present.any())
        preview=self.arena.group_selected_actions(*args,preview=True)
        result=results_from_queue(self.arena,preview,values)
        wrong=replace(result,candidate_ordinal=result.candidate_ordinal+1)
        out=self.arena.scatter_results(preview,wrong)
        self.assertTrue((out.errors&FOREIGN_RESULT).all())
        self.assertFalse(out.present.any())

    def test_physical_selection_cannot_commit_multiple_actions_for_one_root(self):
        args=list(tensors(2,4));args[0].fill_(1);args[3][:,:2]=True
        queue=self.arena.group_selected_actions(*args)
        self.assertEqual(queue.errors.tolist(),[COUNT_OVERFLOW,COUNT_OVERFLOW])
        self.assertFalse(queue.present.any())


class DomainActionBatchCUDATests(unittest.TestCase):
    @unittest.skipUnless(_cuda_available(), 'Actual CUDA required')
    def test_cuda_groups_and_scatters_like_cpu_oracle(self):
        cpu = batch('cpu')
        gpu = batch('cuda')
        tokens, counts, mask, selected, active, owners, generations = tensors(3, 4)
        tokens[0] = torch.tensor([1, 2, 3, 0])
        tokens[1] = torch.tensor([2, 1, 0, 0])
        tokens[2] = torch.tensor([1, 1, 2, 2])
        selected[0, :3] = True
        selected[1, :2] = True
        selected[2, :2] = True
        cpu_queue = cpu.group_selected_actions(
            tokens, counts, mask, selected, active, owners, generations,preview=True)
        gpu_queue = gpu.group_selected_actions(
            *(value.cuda() for value in (tokens, counts, mask, selected, active, owners, generations)),preview=True)
        self.assertEqual(int(gpu_queue.compact_count),7)
        self.assertEqual(cpu_queue.errors.tolist(), gpu_queue.errors.cpu().tolist())
        self.assertEqual(compact_jobs(cpu_queue), compact_jobs(type(gpu_queue)(
            **{name: (value.cpu() if torch.is_tensor(value) else value)
               for name, value in gpu_queue.__dict__.items()})))
        values = torch.arange(cpu.work_width, dtype=torch.int32)
        cpu_out = cpu.scatter_results(cpu_queue, results_from_queue(cpu, cpu_queue, values))
        gpu_out = gpu.scatter_results(
            gpu_queue, results_from_queue(gpu, gpu_queue, values.cuda()))
        self.assertEqual(cpu_out.values.tolist(), gpu_out.values.cpu().tolist())
        self.assertEqual(cpu_out.present.tolist(), gpu_out.present.cpu().tolist())
        self.assertEqual(cpu_out.errors.tolist(), gpu_out.errors.cpu().tolist())
