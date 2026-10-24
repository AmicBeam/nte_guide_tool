"""Phase-homogeneous bounded replay; fresh coverage and reuse accounting retained."""
import math
import time
from .episode_replay import EpisodePool


def row_phase(row):
    from .phase_policy import phase_of_observation
    phase = phase_of_observation(row['x'])
    declared = row.get('phase')
    if declared not in ('mulligan', 'playing', 'choice'):
        raise ValueError('Policy replay requires an explicit legal game phase')
    if phase != ('opening' if declared == 'mulligan' else 'playing'):
        raise ValueError('Replay phase disagrees with the encoded observation')
    return phase


def balanced_slices(count, maximum):
    if type(maximum) is not int or maximum < 1:
        raise ValueError('Positive batch capacity required')
    if not count:
        return
    batches = math.ceil(count / maximum)
    width, extra = divmod(count, batches)
    start = 0
    for i in range(batches):
        stop = start + width + (i < extra)
        yield slice(start, stop)
        start = stop


class PhaseEpisodePool(EpisodePool):
    def _iter_refs(self, refs, rng, batch_size, deadline, source):
        groups = {'opening': [], 'playing': []}
        for ref in refs:
            groups[row_phase(ref[0]['rows'][ref[1]])].append(ref)
        # Host metadata routing: no CUDA synchronization or per-row forward calls.
        for phase, group in groups.items():
            rng.shuffle(group)
            for selection in balanced_slices(len(group), batch_size):
                if deadline is not None and time.time() >= deadline:
                    return
                rows, chosen, stats = self._take(group[selection], source)
                if rows:
                    yield rows, chosen, dict(stats, policy_phase=phase)
