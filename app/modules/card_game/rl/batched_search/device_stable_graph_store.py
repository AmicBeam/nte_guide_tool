"""Collector specialization of page admission; existing buffers stay intact.

Only each newly created store's node/edge/payload admission method is replaced
by the checked stable-prefix implementation. No global Torch patch, no other
store mutation, no ownership/capacity change and no rule implementation.
"""
from types import MethodType
from ..batched_duel.device_graph_pool import DeviceGraphPagePool
from ..batched_duel.device_graph_storage import CompactDeviceGraphStore
from .device_stable_page_pool import DeviceStableGraphPagePool


class DeviceStableGraphStore(CompactDeviceGraphStore):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for name in ('nodes','edges','payload'):
            pool=getattr(self,name)
            if type(pool) is not DeviceGraphPagePool:
                self.close()
                raise TypeError('Explicit original page pool required')
            pool._admit=MethodType(DeviceStableGraphPagePool._admit,pool)
