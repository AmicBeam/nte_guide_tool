"""CUDA NVRTC compilation and launch runtime for batched game state execution.

This module provides a generic, pure-CUDA compilation and launch runtime:
- No CPU fallbacks or fake emulation.
- Deferred torch import and CUDA initialization (only upon explicit CudaModule construction).
- In-memory NVRTC compilation to PTX using PyTorch's bundled NVRTC DLL/SO without requiring nvcc.
- CUDA Driver API module loading and kernel dispatch with explicit 64-bit ctypes signatures.
- Dedicated module stream lifecycle management with producer stream synchronization and event recording.
- Strict argument validation (rejects boolean scalars, integer overflows, non-finite floats,
  invalid tensor device/contiguity/dtype/arity).
"""

from __future__ import annotations

import ctypes
from functools import wraps
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    import torch


class CudaError(RuntimeError):
    """Base exception for CUDA compilation and runtime errors."""
    pass


class CudaCompilationError(CudaError):
    """Raised when NVRTC fails to compile CUDA source code, containing full compiler logs."""
    pass


class CudaDriverError(CudaError):
    """Raised when CUDA driver API operations return a non-zero status."""
    pass


SUPPORTED_POINTER_TYPES = {
    "int32*": "int32",
    "uint32*": "uint32",
    "int64*": "int64",
    "uint64*": "uint64",
    "uint8*": "uint8",
    "float32*": "float32",
    "float64*": "float64",
}

SUPPORTED_SCALAR_TYPES = {
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
}

TYPE_ALIASES = {
    # Scalar aliases
    "int": "int32",
    "int32_t": "int32",
    "uint": "uint32",
    "uint32_t": "uint32",
    "int64_t": "int64",
    "long": "int64",
    "long long": "int64",
    "uint64_t": "uint64",
    "ulong": "uint64",
    "unsigned long long": "uint64",
    "float": "float32",
    "double": "float64",
    # Pointer aliases
    "int*": "int32*",
    "int32_t*": "int32*",
    "uint*": "uint32*",
    "uint32_t*": "uint32*",
    "int64_t*": "int64*",
    "long*": "int64*",
    "long long*": "int64*",
    "uint64_t*": "uint64*",
    "ulong*": "uint64*",
    "unsigned long long*": "uint64*",
    "uint8_t*": "uint8*",
    "uchar*": "uint8*",
    "unsigned char*": "uint8*",
    "float*": "float32*",
    "double*": "float64*",
}


@dataclass(frozen=True)
class ParamType:
    """Parsed representation of a single kernel parameter type specification."""

    raw_str: str
    normalized: str
    is_pointer: bool
    is_scalar: bool
    is_struct: bool
    dtype_name: str
    struct_name: Optional[str] = None

    def __str__(self) -> str:
        return self.normalized


@dataclass(frozen=True)
class KernelSpec:
    """Specification of a kernel function's parameters and optional struct kind."""

    params: Tuple[ParamType, ...]
    struct_kind: Optional[str] = None

    @classmethod
    def from_spec(
        cls,
        spec: Union[KernelSpec, Sequence[str], Tuple[str, ...]],
        struct_kind: Optional[str] = None,
    ) -> KernelSpec:
        if isinstance(spec, KernelSpec):
            if struct_kind is not None and spec.struct_kind != struct_kind:
                return KernelSpec(params=spec.params, struct_kind=struct_kind)
            return spec

        if not isinstance(spec, (list, tuple)):
            raise TypeError(
                f"Kernel spec must be a KernelSpec or sequence of type strings, got {type(spec).__name__}"
            )

        parsed_params = tuple(parse_param_type(s) for s in spec)
        return cls(params=parsed_params, struct_kind=struct_kind)


def parse_param_type(spec_str: str) -> ParamType:
    """Parse a parameter type string into a validated ParamType."""
    if not isinstance(spec_str, str):
        raise TypeError(f"Parameter type specification must be a string, got {type(spec_str).__name__}")

    raw = spec_str.strip()
    if not raw:
        raise ValueError("Parameter type specification cannot be empty")

    # Clean const qualifiers
    cleaned = raw
    if cleaned.startswith("const "):
        cleaned = cleaned[6:].strip()

    # Normalization check
    aliased = TYPE_ALIASES.get(cleaned, cleaned)

    # Raw structs and host addresses have no validated device memory ABI here.
    if aliased.startswith("struct ") or aliased.startswith("struct:"):
        raise ValueError("Struct parameters require an explicit validated ABI; use typed tensors")

    # Check pointer types
    if aliased.endswith("*"):
        base = aliased[:-1].strip()
        aliased_base = TYPE_ALIASES.get(base, base)
        norm_ptr = f"{aliased_base}*"
        if norm_ptr in SUPPORTED_POINTER_TYPES:
            return ParamType(
                raw_str=raw,
                normalized=norm_ptr,
                is_pointer=True,
                is_scalar=False,
                is_struct=False,
                dtype_name=SUPPORTED_POINTER_TYPES[norm_ptr],
            )
        raise ValueError(
            f"Unsupported pointer parameter type: '{spec_str}' (supported: {sorted(SUPPORTED_POINTER_TYPES.keys())})"
        )

    # Check scalar types
    if aliased in SUPPORTED_SCALAR_TYPES:
        return ParamType(
            raw_str=raw,
            normalized=aliased,
            is_pointer=False,
            is_scalar=True,
            is_struct=False,
            dtype_name=aliased,
        )

    raise ValueError(
        f"Unsupported parameter type: '{spec_str}' (supported scalars: {sorted(SUPPORTED_SCALAR_TYPES)}, "
        f"supported pointers: {sorted(SUPPORTED_POINTER_TYPES.keys())})"
    )


def convert_scalar(param_type: ParamType, val: Any) -> ctypes._SimpleCData:
    """Validate and convert a Python scalar value to a ctypes scalar.

    Strictly rejects boolean values, out-of-range integer values, and non-finite floats.
    """
    if isinstance(val, bool):
        raise TypeError(
            f"Scalar parameter '{param_type.normalized}' strictly rejects bool values, got {val!r}"
        )

    dtype = param_type.dtype_name
    if dtype == "int32":
        if not isinstance(val, int):
            raise TypeError(f"Expected int for int32, got {type(val).__name__}: {val!r}")
        if not (-2_147_483_648 <= val <= 2_147_483_647):
            raise ValueError(f"Value {val} out of bounds for int32 [-2^31, 2^31-1]")
        return ctypes.c_int32(val)

    elif dtype == "uint32":
        if not isinstance(val, int):
            raise TypeError(f"Expected int for uint32, got {type(val).__name__}: {val!r}")
        if not (0 <= val <= 4_294_967_295):
            raise ValueError(f"Value {val} out of bounds for uint32 [0, 2^32-1]")
        return ctypes.c_uint32(val)

    elif dtype == "int64":
        if not isinstance(val, int):
            raise TypeError(f"Expected int for int64, got {type(val).__name__}: {val!r}")
        if not (-9_223_372_036_854_775_808 <= val <= 9_223_372_036_854_775_807):
            raise ValueError(f"Value {val} out of bounds for int64 [-2^63, 2^63-1]")
        return ctypes.c_int64(val)

    elif dtype == "uint64":
        if not isinstance(val, int):
            raise TypeError(f"Expected int for uint64, got {type(val).__name__}: {val!r}")
        if not (0 <= val <= 18_446_744_073_709_551_615):
            raise ValueError(f"Value {val} out of bounds for uint64 [0, 2^64-1]")
        return ctypes.c_uint64(val)

    elif dtype == "float32":
        if not isinstance(val, (int, float)):
            raise TypeError(f"Expected float/int for float32, got {type(val).__name__}: {val!r}")
        fval = float(val)
        if not math.isfinite(fval):
            raise ValueError(f"Non-finite float value {val} rejected for float32")
        if abs(fval) > 3.4028234663852886e+38:
            raise ValueError(f"Value {val} overflows 32-bit float range")
        return ctypes.c_float(fval)

    elif dtype == "float64":
        if not isinstance(val, (int, float)):
            raise TypeError(f"Expected float/int for float64, got {type(val).__name__}: {val!r}")
        fval = float(val)
        if not math.isfinite(fval):
            raise ValueError(f"Non-finite float value {val} rejected for float64")
        return ctypes.c_double(fval)

    raise ValueError(f"Unknown scalar dtype: {dtype}")


def normalize_dim3(val: Union[int, Sequence[int]], name: str) -> Tuple[int, int, int]:
    """Validate and normalize grid/block dimension specification into a 3-tuple of positive ints."""
    if isinstance(val, bool):
        raise TypeError(f"{name} dimension strictly rejects bool, got {val!r}")
    if isinstance(val, int):
        if val <= 0:
            raise ValueError(f"{name} dimension must be positive (> 0), got {val}")
        if val > 0xFFFFFFFF:
            raise ValueError(f"{name} dimension exceeds the CUDA uint32 ABI")
        return (val, 1, 1)
    if isinstance(val, (tuple, list)):
        if len(val) == 0:
            raise ValueError(f"{name} tuple cannot be empty")
        if len(val) > 3:
            raise ValueError(f"{name} tuple must have at most 3 dimensions, got {len(val)}")
        dims: List[int] = []
        for i, d in enumerate(val):
            if isinstance(d, bool):
                raise TypeError(f"{name}[{i}] dimension strictly rejects bool, got {d!r}")
            if not isinstance(d, int):
                raise TypeError(f"{name}[{i}] dimension must be int, got {type(d).__name__}")
            if d <= 0:
                raise ValueError(f"{name}[{i}] dimension must be positive (> 0), got {d}")
            if d > 0xFFFFFFFF:
                raise ValueError(f"{name}[{i}] dimension exceeds the CUDA uint32 ABI")
            dims.append(d)
        while len(dims) < 3:
            dims.append(1)
        return (dims[0], dims[1], dims[2])

    raise TypeError(f"{name} must be an int or a sequence of 1-3 ints, got {type(val).__name__}")


def validate_shared_bytes(shared_bytes: int) -> int:
    """Validate shared memory byte count."""
    if isinstance(shared_bytes, bool):
        raise TypeError("shared_bytes strictly rejects bool")
    if not isinstance(shared_bytes, int):
        raise TypeError(f"shared_bytes must be an int, got {type(shared_bytes).__name__}")
    if shared_bytes < 0:
        raise ValueError(f"shared_bytes must be non-negative (>= 0), got {shared_bytes}")
    if shared_bytes > 0xFFFFFFFF:
        raise ValueError("shared_bytes exceeds the CUDA uint32 ABI")
    return shared_bytes


def _bound_cuda_device(method):
    """Driver API calls require the module's context on the calling thread."""
    @wraps(method)
    def bound(self, *args, **kwargs):
        if not hasattr(self, '_device'):
            return method(self, *args, **kwargs)
        import torch
        with torch.cuda.device(self._device):
            return method(self, *args, **kwargs)
    return bound


def _find_nvrtc_path(torch_lib_dir: Path) -> str:
    """Find the NVRTC shared library path from PyTorch lib directory or environment."""
    if torch_lib_dir.is_dir():
        if os.name == 'nt':
            candidates = list(torch_lib_dir.glob('nvrtc64_*.dll'))
        else:
            candidates = list(torch_lib_dir.glob('libnvrtc.so*'))
        valid = [p for p in candidates if '.alt.' not in p.name]
        if valid:
            return str(sorted(valid)[-1])

    # Check site-packages/nvidia/cuda_nvrtc/lib
    site_packages = torch_lib_dir.parent.parent
    nvidia_dir = site_packages / 'nvidia'
    if nvidia_dir.is_dir():
        pattern = 'nvrtc64_*.dll' if os.name == 'nt' else 'libnvrtc.so*'
        candidates = list(nvidia_dir.glob(f'**/lib/{pattern}'))
        valid = [p for p in candidates if '.alt.' not in p.name]
        if valid:
            return str(sorted(valid)[-1])

    # Fallback to system library finder
    import ctypes.util
    found = ctypes.util.find_library('nvrtc')
    if found:
        return found
    if os.name == 'nt':
        found = ctypes.util.find_library('nvrtc64')
        if found:
            return found

    raise RuntimeError("NVRTC shared library not found in PyTorch lib directory or system paths")


def _setup_nvrtc_signatures(nv: Any) -> None:
    """Set explicit argtypes and restypes for all NVRTC functions to prevent 64-bit truncation."""
    nv.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    nv.nvrtcCreateProgram.restype = ctypes.c_int

    nv.nvrtcCompileProgram.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    nv.nvrtcCompileProgram.restype = ctypes.c_int

    nv.nvrtcGetProgramLogSize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    nv.nvrtcGetProgramLogSize.restype = ctypes.c_int

    nv.nvrtcGetProgramLog.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    nv.nvrtcGetProgramLog.restype = ctypes.c_int

    nv.nvrtcGetPTXSize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    nv.nvrtcGetPTXSize.restype = ctypes.c_int

    nv.nvrtcGetPTX.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    nv.nvrtcGetPTX.restype = ctypes.c_int

    nv.nvrtcDestroyProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
    ]
    nv.nvrtcDestroyProgram.restype = ctypes.c_int

    if hasattr(nv, "nvrtcGetErrorString"):
        nv.nvrtcGetErrorString.argtypes = [ctypes.c_int]
        nv.nvrtcGetErrorString.restype = ctypes.c_char_p


def _setup_cuda_driver_signatures(drv: Any) -> None:
    """Set explicit argtypes and restypes for CUDA driver API to prevent 64-bit truncation."""
    drv.cuInit.argtypes = [ctypes.c_uint]
    drv.cuInit.restype = ctypes.c_int

    drv.cuModuleLoadData.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
    ]
    drv.cuModuleLoadData.restype = ctypes.c_int

    if hasattr(drv, "cuModuleLoadDataEx"):
        drv.cuModuleLoadDataEx.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        drv.cuModuleLoadDataEx.restype = ctypes.c_int

    drv.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    drv.cuModuleGetFunction.restype = ctypes.c_int

    drv.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    drv.cuLaunchKernel.restype = ctypes.c_int

    drv.cuModuleUnload.argtypes = [ctypes.c_void_p]
    drv.cuModuleUnload.restype = ctypes.c_int

    if hasattr(drv, "cuGetErrorString"):
        drv.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        drv.cuGetErrorString.restype = ctypes.c_int


def _format_cuda_error(drv: Any, rc: int, operation: str) -> str:
    msg = f"{operation} failed with CUDA driver error code {rc}"
    if drv is not None and hasattr(drv, "cuGetErrorString"):
        err_str = ctypes.c_char_p()
        if drv.cuGetErrorString(rc, ctypes.byref(err_str)) == 0 and err_str.value:
            msg += f" ({err_str.value.decode('utf-8', errors='replace')})"
    return msg


def _format_nvrtc_error(nv: Any, rc: int, operation: str) -> str:
    msg = f"{operation} failed with NVRTC error code {rc}"
    if nv is not None and hasattr(nv, "nvrtcGetErrorString"):
        err_str = nv.nvrtcGetErrorString(rc)
        if err_str:
            msg += f" ({err_str.decode('utf-8', errors='replace')})"
    return msg


class CudaModule:
    """Generic CUDA module compiled via NVRTC and dispatched via CUDA Driver API.

    Features:
    - Lazy torch and CUDA dependency (only imported/initialized on CudaModule instantiation).
    - No CPU fallback or emulation: requires functional CUDA hardware and libraries.
    - Explicit compilation flags (--std=c++17, --gpu-architecture=compute_NN, --fmad=false).
    - Owns a dedicated torch.cuda.Stream; coordinates caller producer stream and records events.
    - Strict lifecycle cleanup on close or initialization failure.
    """

    def __init__(
        self,
        source: str,
        kernel_specs: Mapping[str, Union[KernelSpec, Sequence[str], Tuple[str, ...]]],
        device: str = "cuda:0",
        *,
        extra_compile_options: Optional[Sequence[str]] = None,
    ) -> None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string containing CUDA C/C++ code")
        if not isinstance(kernel_specs, Mapping) or not kernel_specs:
            raise ValueError("kernel_specs must be a non-empty mapping of kernel names to specs")

        # Parse kernel specs before doing heavy CUDA operations
        self._kernel_specs: Dict[str, KernelSpec] = {}
        for name, spec in kernel_specs.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Kernel name must be a non-empty string, got {name!r}")
            self._kernel_specs[name] = KernelSpec.from_spec(spec)

        self._source = source
        self._source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
        self._closed = False
        self._module: Optional[ctypes.c_void_p] = None
        self._kernels: Dict[str, ctypes.c_void_p] = {}
        self._nv: Any = None
        self._drv: Any = None
        self._stream: Any = None

        # Deferred torch import and CUDA verification
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available; CudaModule requires a real, functional CUDA GPU environment"
            )

        # Device normalization and binding
        device_obj = torch.device(device)
        if device_obj.type != "cuda":
            raise ValueError(f"device must be a CUDA device, got {device!r}")
        self._device_idx: int = (
            device_obj.index if device_obj.index is not None else torch.cuda.current_device()
        )
        self._device = torch.device(f"cuda:{self._device_idx}")

        with torch.cuda.device(self._device):
            # Retain Torch primary context on the target device
            torch.cuda.init()
            # Touch device to ensure runtime primary context is active on this thread
            _ = torch.zeros(1, device=self._device)

            capability = torch.cuda.get_device_capability(self._device_idx)
            self._gpu_capability = (capability[0], capability[1])

            # Load NVRTC and CUDA Driver libraries
            torch_lib_dir = Path(torch.__file__).parent / "lib"
            nvrtc_path = _find_nvrtc_path(torch_lib_dir)
            self._nv = ctypes.WinDLL(nvrtc_path) if os.name == "nt" else ctypes.CDLL(nvrtc_path)
            _setup_nvrtc_signatures(self._nv)

            if os.name == "nt":
                try:
                    self._drv = ctypes.WinDLL("nvcuda.dll")
                except OSError as e:
                    raise RuntimeError(f"Failed to load CUDA driver DLL nvcuda.dll: {e}") from e
            else:
                driver_loaded = False
                for libname in ("libcuda.so.1", "libcuda.so"):
                    try:
                        self._drv = ctypes.CDLL(libname)
                        driver_loaded = True
                        break
                    except OSError:
                        continue
                if not driver_loaded:
                    raise RuntimeError("Failed to load CUDA driver library (libcuda.so.1 / libcuda.so)")
            _setup_cuda_driver_signatures(self._drv)

            # Initialize CUDA driver API
            rc = self._drv.cuInit(0)
            if rc != 0:
                raise CudaDriverError(_format_cuda_error(self._drv, rc, "cuInit"))

            # Strict NVRTC compiler options: C++17, explicit architecture, no fast-math FMAD relaxation
            arch_flag = f"--gpu-architecture=compute_{self._gpu_capability[0]}{self._gpu_capability[1]}"
            flags = [
                "--std=c++17",
                arch_flag,
                "--fmad=false",
            ]
            if extra_compile_options:
                if any(not isinstance(flag, str) or not flag.startswith('-D')
                       for flag in extra_compile_options):
                    raise ValueError("Only explicit -D compiler definitions may extend frozen CUDA flags")
                flags.extend(extra_compile_options)
            self._compile_options = tuple(flags)

            # Compile CUDA source code to PTX
            ptx = self._compile_source_to_ptx(source, flags)

            # Load module into CUDA primary context
            mod = ctypes.c_void_p()
            try:
                rc = self._drv.cuModuleLoadData(ctypes.byref(mod), ptx)
                if rc != 0:
                    raise CudaDriverError(_format_cuda_error(self._drv, rc, "cuModuleLoadData"))

                self._module = mod

                # Lookup kernel function handles
                for k_name in self._kernel_specs:
                    fn = ctypes.c_void_p()
                    rc = self._drv.cuModuleGetFunction(
                        ctypes.byref(fn), self._module, k_name.encode("utf-8")
                    )
                    if rc != 0:
                        raise KeyError(
                            f"Kernel '{k_name}' not found in compiled module: "
                            f"{_format_cuda_error(self._drv, rc, 'cuModuleGetFunction')}"
                        )
                    self._kernels[k_name] = fn

            except Exception:
                if mod.value:
                    try:
                        self._drv.cuModuleUnload(mod)
                    except Exception:
                        pass
                self._module = None
                self._kernels.clear()
                raise

            # Dedicated stream owned by this CudaModule
            self._stream = torch.cuda.Stream(device=self._device)

    def _compile_source_to_ptx(self, source: str, flags: Sequence[str]) -> bytes:
        """Compile source string to PTX bytes with strictly managed NVRTC program lifecycle."""
        prog = ctypes.c_void_p()
        src_bytes = source.encode("utf-8")
        prog_name = b"batched_kernel.cu"

        rc = self._nv.nvrtcCreateProgram(
            ctypes.byref(prog),
            src_bytes,
            prog_name,
            0,
            None,
            None,
        )
        if rc != 0:
            raise CudaCompilationError(_format_nvrtc_error(self._nv, rc, "nvrtcCreateProgram"))

        try:
            c_flags = [f.encode("utf-8") for f in flags]
            options = (ctypes.c_char_p * len(c_flags))(*c_flags)
            compile_rc = self._nv.nvrtcCompileProgram(prog, len(c_flags), options)

            if compile_rc != 0:
                log_size = ctypes.c_size_t()
                self._nv.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
                buf = ctypes.create_string_buffer(log_size.value)
                self._nv.nvrtcGetProgramLog(prog, buf)
                log_text = buf.value.decode("utf-8", errors="replace")
                raise CudaCompilationError(
                    f"NVRTC compilation failed (code {compile_rc}) with flags {flags}:\n{log_text}"
                )

            ptx_size = ctypes.c_size_t()
            rc = self._nv.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
            if rc != 0:
                raise CudaCompilationError(_format_nvrtc_error(self._nv, rc, "nvrtcGetPTXSize"))

            ptx_buf = ctypes.create_string_buffer(ptx_size.value)
            rc = self._nv.nvrtcGetPTX(prog, ptx_buf)
            if rc != 0:
                raise CudaCompilationError(_format_nvrtc_error(self._nv, rc, "nvrtcGetPTX"))

            return ptx_buf.raw

        finally:
            if prog.value:
                self._nv.nvrtcDestroyProgram(ctypes.byref(prog))

    @property
    def source(self) -> str:
        return self._source

    @property
    def source_sha(self) -> str:
        return self._source_sha

    @property
    def gpu_capability(self) -> Tuple[int, int]:
        return self._gpu_capability

    @property
    def compile_options(self) -> Tuple[str, ...]:
        return self._compile_options

    @property
    def backend(self) -> str:
        return "cuda"

    @property
    def actual_backend(self) -> str:
        return "cuda"

    @property
    def device(self) -> Any:
        return self._device

    @property
    def device_idx(self) -> int:
        return self._device_idx

    @property
    def stream(self) -> Any:
        return self._stream

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def kernel_specs(self) -> Dict[str, KernelSpec]:
        return dict(self._kernel_specs)

    @_bound_cuda_device
    def launch(
        self,
        name: str,
        args: Sequence[Any],
        *,
        grid: Union[int, Sequence[int]],
        block: Union[int, Sequence[int]],
        shared_bytes: int = 0,
    ) -> Any:
        """Asynchronously launch a kernel on the module's stream and return a completion event.

        - Coordinates caller stream with self._stream.wait_stream.
        - Calls Tensor.record_stream for all tensor arguments.
        - Guarantees ctypes parameter pointer structures stay alive across cuLaunchKernel.
        - Does NOT synchronize on CPU or evaluate elements synchronously.
        - Returns a real torch.cuda.Event recorded on completion.
        """
        if self._closed:
            raise RuntimeError("CudaModule is closed")

        if name not in self._kernels:
            raise KeyError(f"Kernel '{name}' not found in compiled module")

        spec = self._kernel_specs[name]
        if len(args) != len(spec.params):
            raise ValueError(
                f"Kernel '{name}' expects {len(spec.params)} arguments, got {len(args)}"
            )

        gx, gy, gz = normalize_dim3(grid, "grid")
        bx, by, bz = normalize_dim3(block, "block")
        shmem = validate_shared_bytes(shared_bytes)

        import torch

        uint32_dtype = getattr(torch, "uint32", torch.int32)
        uint64_dtype = getattr(torch, "uint64", torch.int64)

        tensor_dtype_map = {
            "int32": (torch.int32,),
            "uint32": (uint32_dtype, torch.int32),
            "int64": (torch.int64,),
            "uint64": (uint64_dtype, torch.int64),
            "uint8": (torch.uint8,),
            "float32": (torch.float32,),
            "float64": (torch.float64,),
        }

        c_values: List[Any] = []
        tensors_to_record: List[Any] = []

        for idx, (param_type, arg) in enumerate(zip(spec.params, args)):
            if param_type.is_pointer:
                if not isinstance(arg, torch.Tensor):
                    raise TypeError(
                        f"Kernel '{name}' argument {idx} ('{param_type.normalized}') requires a torch.Tensor, "
                        f"got {type(arg).__name__}"
                    )
                if arg.device.type != "cuda":
                    raise ValueError(
                        f"Kernel '{name}' argument {idx} ('{param_type.normalized}') must be on CUDA, "
                        f"got device={arg.device}"
                    )
                t_idx = arg.device.index if arg.device.index is not None else 0
                if t_idx != self._device_idx:
                    raise ValueError(
                        f"Kernel '{name}' argument {idx} ('{param_type.normalized}') device cuda:{t_idx} "
                        f"does not match module device cuda:{self._device_idx}"
                    )
                if not arg.is_contiguous():
                    raise ValueError(
                        f"Kernel '{name}' argument {idx} ('{param_type.normalized}') must be contiguous"
                    )

                expected_dtypes = tensor_dtype_map[param_type.dtype_name]
                if arg.dtype not in expected_dtypes:
                    expected_str = (
                        str(expected_dtypes[0])
                        if len(expected_dtypes) == 1
                        else f"one of {expected_dtypes}"
                    )
                    raise TypeError(
                        f"Kernel '{name}' argument {idx} ('{param_type.normalized}') dtype mismatch: "
                        f"expected {expected_str}, got {arg.dtype}"
                    )

                c_ptr = ctypes.c_void_p(arg.data_ptr())
                c_values.append(c_ptr)
                tensors_to_record.append(arg)

            else:
                c_scalar = convert_scalar(param_type, arg)
                c_values.append(c_scalar)

        # Coordinate caller producer stream
        caller_stream = torch.cuda.current_stream(device=self._device)
        if caller_stream != self._stream:
            self._stream.wait_stream(caller_stream)

        # Extend asynchronous memory lifetime for tensor arguments
        for t in tensors_to_record:
            t.record_stream(self._stream)

        # Construct parameter pointers array; elements remain alive on Python stack across call
        param_ptrs = (ctypes.c_void_p * len(c_values))()
        for idx, val in enumerate(c_values):
            param_ptrs[idx] = ctypes.cast(ctypes.byref(val), ctypes.c_void_p)

        kernel_fn = self._kernels[name]
        stream_handle = ctypes.c_void_p(self._stream.cuda_stream)

        rc = self._drv.cuLaunchKernel(
            kernel_fn,
            ctypes.c_uint(gx),
            ctypes.c_uint(gy),
            ctypes.c_uint(gz),
            ctypes.c_uint(bx),
            ctypes.c_uint(by),
            ctypes.c_uint(bz),
            ctypes.c_uint(shmem),
            stream_handle,
            param_ptrs,
            None,
        )
        if rc != 0:
            raise CudaDriverError(_format_cuda_error(self._drv, rc, f"cuLaunchKernel for '{name}'"))

        # Record completion event on the module stream
        event = torch.cuda.Event()
        event.record(self._stream)
        return event

    @_bound_cuda_device
    def close(self) -> None:
        """Idempotently close the module, synchronize its own stream, and unload module."""
        if self._closed:
            return
        self._closed = True

        if self._stream is not None:
            try:
                self._stream.synchronize()
            except Exception:
                pass

        if self._module is not None and self._drv is not None:
            try:
                self._drv.cuModuleUnload(self._module)
            except Exception:
                pass
            self._module = None

        self._kernels.clear()

    def __enter__(self) -> CudaModule:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
