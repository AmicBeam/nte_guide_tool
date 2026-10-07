"""Episode-weighted WDL metrics using the checked batched inference bridge.

This scores already collected visible observations. It does not execute rules,
generate targets, change weights, or claim full GPU game/search concurrency.
"""
import numpy as np


def compute(games, policies, *, device='cuda'):
    from .inference import BatchedInference, InferenceRequest, CANDIDATE_DIM
    from scripts.check_duel_v2_value_calibration import probability_metrics
    if device not in ('cpu', 'cuda'):
        raise ValueError('Explicit CPU oracle or CUDA metrics required')
    engine = BatchedInference(policies, device=device, max_batch_rows=256)
    empty = np.empty((0, CANDIDATE_DIM), dtype=np.float32)
    empty.flags.writeable = False
    result = {}
    for key, policy in policies.items():
        requests = []; labels = []; weights = []; count = 0
        for game in games:
            rows = [r for r in game['value_rows'] if r['key'] == key]
            if not rows:
                continue
            count += 1
            for row in rows:
                requests.append(InferenceRequest(
                    request_id=(key, len(requests)), model_key=key,
                    model_version=policy.version, x=np.asarray(row['x'], dtype=np.float32),
                    candidates=empty, value_actor=row['value_actor'], value_only=True))
                labels.append(int(row['z']) + 1)
                weights.append(1 / len(rows))
        answers = engine.predict(requests)
        if (len(answers) != len(requests) or any(
                a.expired or a.request_id != q.request_id or a.model_key != key
                or a.model_version != policy.version or a.logits.size
                for a, q in zip(answers, requests))):
            raise ValueError('Batched value metrics response identity mismatch')
        result[key] = dict(**probability_metrics([a.wdl for a in answers], labels, weights),
                           normal_games=count)
    return result
