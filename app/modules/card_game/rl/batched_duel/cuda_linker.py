"""Link relocatable CUBIN objects through the CUDA Driver linker."""

from __future__ import annotations

import ctypes
from typing import Any, List, Optional, Sequence, Tuple

from .cuda_runtime import CudaDriverError

CU_JIT_INPUT_CUBIN = 0
CU_JIT_INFO_LOG_BUFFER = 3
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4
CU_JIT_ERROR_LOG_BUFFER = 5
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
CU_JIT_OPTIMIZATION_LEVEL = 7
LOG_BUFFER_SIZE = 8192

_CREATE_ARGTYPES = [
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
]
_ADD_DATA_ARGTYPES = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_void_p),
]
_COMPLETE_ARGTYPES = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
]
_DESTROY_ARGTYPES = [ctypes.c_void_p]
_ERROR_ARGTYPES = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]


def link_cuda_objects(
    driver: Any,
    objects: Sequence[Tuple[str, bytes]],
    *,
    jit_level: Optional[int] = None,
) -> bytes:
    """Link relocatable CUBIN inputs and return a copied final image."""
    prepared = _prepare_objects(objects)
    if jit_level is not None and (isinstance(jit_level, bool) or not isinstance(jit_level, int)):
        raise TypeError('jit_level must be int or None')

    create, create_name = _resolve_symbol(driver, 'cuLinkCreate_v2', 'cuLinkCreate')
    add_data, add_name = _resolve_symbol(driver, 'cuLinkAddData_v2', 'cuLinkAddData')
    complete = _require_symbol(driver, 'cuLinkComplete')
    destroy = _require_symbol(driver, 'cuLinkDestroy')
    _configure_linker_signatures(driver)

    info_log = ctypes.create_string_buffer(LOG_BUFFER_SIZE)
    error_log = ctypes.create_string_buffer(LOG_BUFFER_SIZE)
    option_keys: List[int] = [
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    ]
    option_vals: List[int] = [
        ctypes.addressof(info_log),
        LOG_BUFFER_SIZE,
        ctypes.addressof(error_log),
        LOG_BUFFER_SIZE,
    ]
    if jit_level is not None:
        option_keys.append(CU_JIT_OPTIMIZATION_LEVEL)
        option_vals.append(jit_level)
    count = len(option_keys)
    options = (ctypes.c_int * count)(*option_keys)
    values = (ctypes.c_void_p * count)(*option_vals)
    keepalive: List[Any] = [info_log, error_log, options, values]
    keepalive.extend(prepared)

    state = ctypes.c_void_p()
    created = False
    pending: Optional[BaseException] = None
    try:
        rc = create(count, options, values, ctypes.byref(state))
        if rc != 0:
            raise CudaDriverError(_format_link_error(driver, rc, create_name, info_log, error_log))
        created = True
        for name_buf, data_buf, size in prepared:
            rc = add_data(
                state,
                CU_JIT_INPUT_CUBIN,
                data_buf,
                size,
                name_buf,
                0,
                None,
                None,
            )
            if rc != 0:
                raise CudaDriverError(_format_link_error(driver, rc, add_name, info_log, error_log))
        cubin_ptr = ctypes.c_void_p()
        cubin_size = ctypes.c_size_t()
        rc = complete(state, ctypes.byref(cubin_ptr), ctypes.byref(cubin_size))
        if rc != 0:
            raise CudaDriverError(_format_link_error(driver, rc, 'cuLinkComplete', info_log, error_log))
        if not cubin_ptr.value or cubin_size.value == 0:
            raise CudaDriverError('cuLinkComplete returned an empty CUBIN image')
        return ctypes.string_at(cubin_ptr.value, cubin_size.value)
    except BaseException as exc:
        pending = exc
        raise
    finally:
        keepalive.append(state)
        if created:
            try:
                rc = destroy(state)
                if rc != 0:
                    raise CudaDriverError(
                        _format_link_error(driver, rc, 'cuLinkDestroy', info_log, error_log)
                    )
            except BaseException:
                if pending is None:
                    raise


def _prepare_objects(objects: Sequence[Tuple[str, bytes]]) -> List[Tuple[Any, Any, int]]:
    if isinstance(objects, (str, bytes)):
        raise TypeError('CUDA link objects must be a sequence of (name, data) pairs')
    try:
        items = list(objects)
    except TypeError as exc:
        raise TypeError('CUDA link objects must be a sequence of (name, data) pairs') from exc
    if not items:
        raise ValueError('CUDA link requires at least one relocatable CUBIN object')

    seen = set()
    prepared = []
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError('CUDA link objects must be (name, data) pairs')
        name, data = item
        if not isinstance(name, str):
            raise TypeError('CUDA link object names must be str')
        if not name:
            raise ValueError('CUDA link object names must be non-empty')
        if '\0' in name:
            raise ValueError('CUDA link object names must not contain NUL')
        if name in seen:
            raise ValueError('CUDA link object names must be unique')
        if not isinstance(data, bytes):
            raise TypeError('CUDA link object data must be bytes')
        if not data:
            raise ValueError('CUDA link object data must be a non-empty CUBIN')
        seen.add(name)
        name_buf = ctypes.create_string_buffer(name.encode('utf-8'))
        data_buf = (ctypes.c_char * len(data)).from_buffer_copy(data)
        prepared.append((name_buf, data_buf, len(data)))
    return prepared


def _resolve_symbol(driver: Any, preferred: str, fallback: str) -> Tuple[Any, str]:
    if hasattr(driver, preferred):
        return getattr(driver, preferred), preferred
    if hasattr(driver, fallback):
        return getattr(driver, fallback), fallback
    raise CudaDriverError(f'CUDA driver is missing {preferred} or {fallback}')


def _require_symbol(driver: Any, name: str) -> Any:
    if not hasattr(driver, name):
        raise CudaDriverError(f'CUDA driver is missing {name}')
    return getattr(driver, name)


def _configure_linker_signatures(driver: Any) -> None:
    for name in ('cuLinkCreate_v2', 'cuLinkCreate'):
        if hasattr(driver, name):
            _bind(getattr(driver, name), _CREATE_ARGTYPES)
    for name in ('cuLinkAddData_v2', 'cuLinkAddData'):
        if hasattr(driver, name):
            _bind(getattr(driver, name), _ADD_DATA_ARGTYPES)
    if hasattr(driver, 'cuLinkComplete'):
        _bind(driver.cuLinkComplete, _COMPLETE_ARGTYPES)
    if hasattr(driver, 'cuLinkDestroy'):
        _bind(driver.cuLinkDestroy, _DESTROY_ARGTYPES)
    for name in ('cuGetErrorName', 'cuGetErrorString'):
        if hasattr(driver, name):
            _bind(getattr(driver, name), _ERROR_ARGTYPES)


def _bind(fn: Any, argtypes: Sequence[Any]) -> None:
    fn.argtypes = list(argtypes)
    fn.restype = ctypes.c_int


def _format_link_error(
    driver: Any,
    rc: int,
    operation: str,
    info_log: ctypes.Array,
    error_log: ctypes.Array,
) -> str:
    msg = f'{operation} failed with CUDA driver error code {rc}'
    details = _driver_error_details(driver, rc)
    if details:
        msg += f' ({details})'
    error_text = _buffer_text(error_log)
    info_text = _buffer_text(info_log)
    if error_text:
        msg += f'\n{error_text}'
    if info_text:
        msg += f'\n{info_text}'
    return msg


def _driver_error_details(driver: Any, rc: int) -> str:
    parts = []
    for name in ('cuGetErrorName', 'cuGetErrorString'):
        fn = getattr(driver, name, None)
        if fn is None:
            continue
        text = ctypes.c_char_p()
        try:
            query_rc = fn(rc, ctypes.byref(text))
        except Exception:
            continue
        if query_rc == 0 and text.value:
            parts.append(text.value.decode('utf-8', errors='replace'))
    return ': '.join(parts)


def _buffer_text(buf: ctypes.Array) -> str:
    raw = buf.raw.split(b'\x00', 1)[0]
    if not raw:
        return ''
    return raw.decode('utf-8', errors='replace')
