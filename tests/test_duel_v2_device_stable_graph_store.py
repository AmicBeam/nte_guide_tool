"""Reuse complete graph-store lifecycle/fault suite with stable page admission."""
from tests import test_duel_v2_device_graph_storage as suite
from app.modules.card_game.rl.batched_search.device_stable_graph_store import DeviceStableGraphStore

class StableGraphStorageTest(suite.DeviceGraphStorageTest):
    def store(self,device='cpu',slots=4,pages=16,page_bytes=128,per_slot=8):
        return DeviceStableGraphStore(slots=slots,node_pages=pages,edge_pages=pages,payload_pages=pages,
            node_page_bytes=page_bytes,edge_page_bytes=page_bytes,payload_page_bytes=page_bytes,
            max_node_pages_per_slot=per_slot,max_edge_pages_per_slot=per_slot,max_payload_pages_per_slot=per_slot,
            device=device,allow_cpu_reference=device=='cpu')
