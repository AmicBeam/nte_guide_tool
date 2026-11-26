"""Cache relocatable objects independently, then cache the linked CUDA image."""
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from hashlib import sha256
import os
from pathlib import Path
import time


def validate_compilation_units(units):
    if units is None:
        return None
    if not isinstance(units, (tuple, list)) or not units:
        raise ValueError('Nonempty ordered CUDA compilation units required')
    result, names = [], set()
    for pair in units:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError('Compilation units must be (name, source) pairs')
        name, source = pair
        if (not isinstance(name, str) or not name.strip() or '\0' in name or name in names
                or not isinstance(source, str) or not source.strip()):
            raise ValueError('Unique named nonempty CUDA source units required')
        names.add(name)
        result.append((name, source))
    return tuple(result)


def validate_compilation_unit_specs(specs, host_specs):
    """Copy a nonempty, exactly typed subset used only for object identities."""
    if specs is None:
        return None
    from .cuda_runtime import KernelSpec
    if not isinstance(specs, Mapping) or not specs:
        raise ValueError('Nonempty compilation_unit_specs mapping required')
    result = {}
    for name, spec in specs.items():
        if not isinstance(name, str) or name not in host_specs:
            raise ValueError('Compilation unit specs must be a subset of host kernel specs')
        parsed = KernelSpec.from_spec(spec)
        if parsed != KernelSpec.from_spec(host_specs[name]):
            raise ValueError('Compilation unit specs must exactly match host kernel signatures')
        result[name] = parsed
    return result


def compile_linked_binary(module, units, flags, *, workers=2, compilation_unit_specs=None):
    from .cuda_binary_cache import CudaBinaryCache
    from .cuda_linker import link_cuda_objects
    units = validate_compilation_units(units)
    if units is None or type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('Named compilation units and one to four workers required')
    cache_specs = validate_compilation_unit_specs(compilation_unit_specs,
        module._kernel_specs if compilation_unit_specs is not None else {})
    profile = module._compilation_profile
    object_flags = [*flags, '--device-c']
    objects, receipts = [], []
    started = time.perf_counter()
    profile['compilation_units'] = receipts
    profile['unit_compile_workers'] = workers

    def compile_one(pair):
        name, source = pair
        source_sha = sha256(source.encode('utf8')).hexdigest()
        receipt = {}
        # NVRTC owns a distinct program per call. Source identity and timing
        # are local to this job; no shared module fields are temporarily changed.
        options = dict(source_sha=source_sha, profile=receipt)
        if cache_specs is not None:
            options['cache_specs'] = cache_specs
        binary = module._compiled_binary(source, object_flags, 'cubin', **options)
        return (name, binary), dict(name=name, source_sha256=source_sha,
            binary_sha256=sha256(binary).hexdigest(), binary_bytes=len(binary), **receipt)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='cuda-compile') as pool:
        futures = [pool.submit(compile_one, pair) for pair in units]
        try:
            # Link order is independent of scheduling/completion order.
            for future in futures:
                obj, receipt = future.result()
                objects.append(obj)
                receipts.append(receipt)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    profile['object_prepare_seconds'] = time.perf_counter()-started
    profile['compile_seconds'] = sum(row.get('compile_seconds', 0.0) for row in receipts)
    identity = dict(module._cache_inputs(flags), format='linked_relocatable_cubin_v1',
        object_compile_options=object_flags,
        objects=[dict(name=name, sha256=sha256(data).hexdigest()) for name, data in objects])
    default = Path(__file__).resolve().parents[5]/'artifacts'/'rl-evals'/'cuda-cache'
    directory = os.environ.get('NTE_CUDA_BINARY_CACHE_DIR', str(default))
    cache = CudaBinaryCache(directory) if directory else None
    profile['cache_status'] = 'disabled'
    if cache:
        try:
            binary, status = cache.read(identity)
            profile.update(cache_status=status, cache_key=cache.key(identity))
            if binary is not None:
                profile['link_seconds'] = 0.0
                return binary
        except OSError as error:
            profile.update(cache_status='unavailable', cache_error=str(error))
            cache = None
    before = time.perf_counter()
    binary = link_cuda_objects(module._drv, objects,
                              jit_level=profile['jit_optimization_level'])
    profile['link_seconds'] = time.perf_counter()-before
    if cache:
        try:
            cache.write(identity, binary)
        except OSError as error:
            profile['cache_write_error'] = str(error)
    return binary
