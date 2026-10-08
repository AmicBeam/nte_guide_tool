"""Training limits usable by CLI validation without optional ML imports."""
import os

MAX_N_ENVS = 128
DEFAULT_N_ENVS_CAP = 32


def default_n_envs():
    return max(8, min(DEFAULT_N_ENVS_CAP, int(os.cpu_count() or 8)))
