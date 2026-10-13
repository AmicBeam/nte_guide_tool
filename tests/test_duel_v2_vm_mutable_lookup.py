"""Actual CUDA checked string-index hints, mutation and GC parity."""
from pathlib import Path
import os,sys,unittest


def _touch(d):
    return d['focus']


def _ephemeral(value):
    yield value
    yield value+1


def mutations(count):
    d={'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'focus':10,'last':8}
    alias=d;trace=[]
    for i in range(count):
        trace.append(_touch(d))
        d['focus']=d['focus']+1
        if i%3==0:
            value=d.pop('focus');d['focus']=value
        if i%5==0:
            d.pop('a',None);d['a']=i
        garbage={'a':i,'b':i+1,'c':i+2,'d':i+3,'e':i+4,'f':i+5,'g':i+6,'focus':i+7}
        trace.append(_touch(garbage))
        for unused in _ephemeral(i):
            break
    return {'d':d,'alias':alias,'trace':trace}


def program():
    from app.modules.card_game.rl.rule_ir.portable_program import capture_program,FrozenRuleProgram
    path=os.environ.get('NTE_VM_MUTABLE_LOOKUP_PROGRAM')
    if path:return FrozenRuleProgram.from_json(Path(path).read_text())
    if sys.version_info[:2]!=(3,10):raise RuntimeError('Explicit CPython3.10 fixture required')
    return capture_program({'mutations':mutations},allowed_module_prefixes=(__name__,),
        scope_character_ids=None,scope_card_ids=None)


class MutableLookupCudaTest(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')

    def test_dict_replacement_delete_reinsert_and_gc_match_uncached_vm(self):
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm,VmLimits
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        image=lower_program(program());values=[1,10,30,100];results=[]
        for enabled in (False,True):
            with CudaVm(image,limits=VmLimits(frames=8),constant_lookup=True,mutable_lookup_cache=enabled) as vm:
                call=vm.launch(vm.ingest(pack_states(values,GraphLayout(4,16384,32768,131072))),'mutations')
                decoded=call.decode();self.assertEqual(decoded,[mutations(v) for v in values])
                for row in decoded:self.assertIs(row['d'],row['alias'])
                results.append(decoded)
        self.assertEqual(results[0],results[1])

    def test_cache_rechecks_current_edge_after_key_reorder_with_unchanged_header(self):
        import torch as t
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.cuda_runtime import CudaModule
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.vm_heap import emit_vm_heap_source
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout,TAG_STR
        image=lower_program(program())
        candidates=[i for i,n in enumerate(image.nodes) if n[0]==TAG_STR and bytes(image.payload[n[1]:n[1]+n[2]])==b'focus']
        self.assertTrue(candidates);query=-candidates[0]-1
        source='#define DUEL_VM_MUTABLE_LOOKUP_CACHE 1\n'+emit_vm_heap_source()+r'''
using namespace duel_vm_heap;
extern "C" __global__ void reorder(int* nodes,int* edges,unsigned char* payload,int* counts,int* roots,
 int* cn,int* ce,unsigned char* cp,int nc,int ec,int pc,int query,int* out,int maxn,int maxe,int maxp){
 Heap h;init_heap(&h,nodes,edges,payload,counts,maxn,maxe,maxp,cn,ce,cp,nc,ec,pc);
 int before=dict_find(&h,roots[0],query);NodeView n;get_node(&h,roots[0],&n);
 int other=before==0?1:0,first=(n.offset+before)*2,last=(n.offset+other)*2;
 int k=edges[first],v=edges[first+1];edges[first]=edges[last];edges[first+1]=edges[last+1];
 edges[last]=k;edges[last+1]=v;
 out[0]=before;out[1]=dict_find(&h,roots[0],query);out[2]=other;out[3]=h.error;
}
'''
        with CudaVm(image) as vm:
            graph=vm.ingest(pack_states([dict(a=0,b=1,c=2,d=3,e=4,f=5,g=6,focus=7,last=8)],GraphLayout(1,128,128,1024)))
            b=vm._buffers;out=t.zeros(4,dtype=t.int32,device=vm.device)
            with CudaModule(source,{'reorder':['int32*','int32*','uint8*','int32*','int32*','int32*','int32*','uint8*','int32','int32','int32','int32','int32*','int32','int32','int32']},device=str(vm.device)) as module:
                module.launch('reorder',[graph.nodes,graph.edges,graph.payload,graph.counts,graph.roots,
                    b['nodes'],b['edges'],b['payload'],len(image.nodes),len(image.edges),len(image.payload),query,out,128,128,1024],grid=1,block=1)
                actual=out.cpu().tolist();self.assertEqual(actual[1],actual[2]);self.assertNotEqual(actual[0],actual[1]);self.assertEqual(actual[3],0)


if __name__=='__main__':unittest.main()
