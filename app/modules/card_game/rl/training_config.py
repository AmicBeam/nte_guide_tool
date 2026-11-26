"""Training limits usable by CLI validation without optional ML imports."""
import os

MAX_N_ENVS = 128
DEFAULT_N_ENVS_CAP = 32
DEFAULT_COLLECTION_REPLICAS = 1
MAX_COLLECTION_REPLICAS = 5
COLLECTION_GAMES_PER_REPLICA = 56
COLLECTION_HISTORICAL_GAMES_PER_REPLICA = 112
COLLECTION_REPLICA_SEED_STRIDE = 1000
COLLECTION_EPISODE_POOL_LIMIT_PER_REPLICA = 64
COLLECTION_LIVE_SEARCH_ROOTS_MAX = 1600
COLLECTION_SUPPORTED_BACKENDS = ('resident_cpu', 'cuda_vm')


def default_n_envs():
    return max(8, min(DEFAULT_N_ENVS_CAP, int(os.cpu_count() or 8)))


def collection_replicas(config=None, value=None):
    if value is None:
        value = DEFAULT_COLLECTION_REPLICAS if config is None else config.get(
            'collection_replicas', DEFAULT_COLLECTION_REPLICAS)
    if type(value) is not int or value < 1:
        raise ValueError('Positive collection replica count required')
    if value > MAX_COLLECTION_REPLICAS:
        raise ValueError('Collection replica count exceeds the named bound')
    return value


def planned_complete_games(replicas, *, historical=False):
    replicas = collection_replicas(value=replicas)
    unit = COLLECTION_HISTORICAL_GAMES_PER_REPLICA if historical else COLLECTION_GAMES_PER_REPLICA
    return unit * replicas


def collection_episode_pool_limit(replicas):
    return COLLECTION_EPISODE_POOL_LIMIT_PER_REPLICA * collection_replicas(value=replicas)


def collection_uses_historical(config):
    return not (config.get('smoke') or config.get('preflight'))


def validate_collection_memory(
    replicas,
    *,
    training_buffer_mib=None,
    gpu_memory_budget_mib=None,
    historical=False,
):
    replicas = collection_replicas(value=replicas)
    games = planned_complete_games(replicas, historical=historical)
    if training_buffer_mib is not None:
        if type(training_buffer_mib) is not int or training_buffer_mib < 1:
            raise ValueError('Positive training trajectory buffer budget required')
        # A fixed per-game guess cannot qualify trajectory capacity. The
        # collector enforces its measured retained-byte budget; preflight must
        # actually finish the configured complete games before readiness.
    if gpu_memory_budget_mib is not None:
        if type(gpu_memory_budget_mib) is not int or gpu_memory_budget_mib < 1:
            raise ValueError('Positive GPU memory budget required')
    return games


def validate_collection_option(
    value,
    *,
    search_roots=None,
    training_buffer_mib=None,
    gpu_memory_budget_mib=None,
    rule_backend=None,
    runtime=None,
    historical=False,
):
    replicas = collection_replicas(value=value)
    if replicas > 1:
        if runtime not in (None, 'seven'):
            raise ValueError('Collection replicas are a seven-team training option')
        if rule_backend not in COLLECTION_SUPPORTED_BACKENDS:
            raise ValueError('Larger collection replicas require resident_cpu or cuda_vm')
    if search_roots is not None:
        if type(search_roots) is not int or not 1 <= search_roots <= COLLECTION_LIVE_SEARCH_ROOTS_MAX:
            raise ValueError('Bounded search roots required')
    validate_collection_memory(
        replicas,
        training_buffer_mib=training_buffer_mib,
        gpu_memory_budget_mib=gpu_memory_budget_mib,
        historical=historical,
    )
    return replicas
