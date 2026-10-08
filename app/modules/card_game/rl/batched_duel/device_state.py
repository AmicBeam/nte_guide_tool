"""Device-resident lossless graph storage, independent of game-rule execution.

Row cloning is a real device index_select, not a Python per-state decode/copy.
This storage component does not claim a GPU rule engine or policy encoder.
"""
import numpy as np
from .schema import GraphLayout,PackedStateBatch,CodecCapacityError


_NAMES=('nodes','edges','payload','counts','roots')


class DeviceStateBatch:
    def __init__(self):
        raise TypeError('Use DeviceStateBatch.from_packed explicitly')

    @classmethod
    def from_packed(cls,batch,*,device='cuda',max_device_bytes=None):
        if not isinstance(batch,PackedStateBatch):raise TypeError('PackedStateBatch required')
        from .codec import unpack_states
        # Host validation happens once at ingress, never on the device fork path.
        unpack_states(batch)
        import torch
        dev=torch.device(device)
        if dev.type not in ('cpu','cuda'):raise ValueError('CPU oracle or CUDA device required')
        if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
        cls._check_bytes(batch.allocated_bytes,max_device_bytes)
        arrays={name:torch.as_tensor(getattr(batch,name).copy(),device=dev) for name in _NAMES}
        return cls._from_tensors(batch.layout,arrays,max_device_bytes)

    @staticmethod
    def _check_bytes(nbytes,limit):
        if limit is not None:
            if type(limit) is not int or limit<1:raise ValueError('Positive device allocation limit required')
            if nbytes>limit:raise MemoryError('Graph storage exceeds explicit byte allowance')

    @classmethod
    def _from_tensors(cls,layout,arrays,limit):
        obj=object.__new__(cls);obj.layout=layout;obj._arrays=arrays;obj._byte_limit=limit
        obj.device=arrays['nodes'].device;obj._stream=None
        if obj.device.type=='cuda':
            import torch
            obj._stream=torch.cuda.current_stream(obj.device)
        return obj

    @property
    def n_rows(self):return self._arrays['nodes'].shape[0]

    @property
    def allocated_bytes(self):
        return sum(t.numel()*t.element_size() for t in self._arrays.values())

    def _stream_check(self):
        if self._stream is not None:
            import torch
            if torch.cuda.current_stream(self.device)!=self._stream:
                raise RuntimeError('Graph batch requires its owning CUDA stream')

    def clone_rows(self,indices):
        self._stream_check()
        import torch
        if isinstance(indices,torch.Tensor):
            if indices.device!=self.device or indices.dtype not in (torch.int32,torch.int64) or indices.ndim!=1:
                raise ValueError('Row indices must be a device-local integer vector')
            index=indices.to(dtype=torch.int64)
            if index.numel() and bool(((index<0)|(index>=self.n_rows)).any()):raise IndexError('Graph row index outside batch')
        else:
            values=list(indices)
            if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in values):
                raise TypeError('Integer row indices required, not bool/float')
            if any(v<0 or v>=self.n_rows for v in values):raise IndexError('Graph row index outside batch')
            index=torch.tensor(values,dtype=torch.int64,device=self.device)
        n=index.numel()
        if n>self.layout.game_capacity:raise CodecCapacityError('Fork exceeds graph batch row capacity')
        per_row=self.allocated_bytes//self.n_rows if self.n_rows else 0
        self._check_bytes(n*per_row,self._byte_limit)
        arrays={name:value.index_select(0,index).contiguous() for name,value in self._arrays.items()}
        return self._from_tensors(self.layout,arrays,self._byte_limit)

    def to_packed(self):
        """Explicit host snapshot for validation/archival; not a simulation hot path."""
        self._stream_check()
        arrays={name:value.detach().cpu().numpy().copy() for name,value in self._arrays.items()}
        return PackedStateBatch(layout=self.layout,**arrays)

    def ready_event(self):
        """Record a completion event on the owning stream, or None for CPU oracle."""
        self._stream_check()
        if self.device.type!='cuda':return None
        import torch
        event=torch.cuda.Event();event.record(self._stream);return event
