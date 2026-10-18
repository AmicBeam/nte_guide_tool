"""Actual CUDA rule queries for diagnostic parity, with explicit host transfers.

This interface never runs CPU rules. Its host controller and column transfers
make it unsuitable for throughput measurement; closed search uses DeviceSoARules.
"""
from pathlib import Path

import numpy as np

from .cuda_vm import VmLimits, DeviceGraph
from .native_soa_cuda import CudaSoARules
from .native_soa_cpu import NativeSoACall
from .native_array_cpu import pack_call_arguments
from .record_soa import RecordSoABatch, RecordSoAError
from .schema import GraphLayout


class NativeSoACUDAQuery:
    execution_mode='native_domain_soa_cuda_query'
    interpreter_fallback=False
    runtime_instruction_fetch=False
    gpu_throughput_qualified=False

    def __init__(self,image,schema,*,limits=VmLimits(),device='cuda',directory=None,
                 max_nodes=1048576,max_edges=2097152,max_payload_bytes=33554432,
                 seconds=120,stack_runs=False,native_functions=False,native_values=False,
                 execute_only=False, separate_compilation=False, indexed_attributes=False,
                 stack_bytes=131072, direct_runtime_calls=False, indexed_container_methods=False,
                 direct_type_queries=False, indexed_entity_queries=False,
                 runtime_primitive_compare=False,direct_deepcopy_calls=False,
                 native_hydration_probe=False,hydration_program=None,
                 native_preview_arena=False,preview_program=None):
        if type(seconds) not in (int,float) or not 0<seconds<=3600:
            raise ValueError('Bounded CUDA diagnostic call budget required')
        for value in (max_nodes,max_edges,max_payload_bytes):
            if type(value) is not int or not 0<value<2**31:
                raise ValueError('Positive bounded diagnostic scratch required')
        self.executor=CudaSoARules(image,schema,device=device,limits=limits,stack_bytes=stack_bytes,
            stack_runs=stack_runs,native_functions=native_functions,native_values=native_values,
            execute_only=execute_only, separate_compilation=separate_compilation,
            indexed_attributes=indexed_attributes, direct_runtime_calls=direct_runtime_calls,
            indexed_container_methods=indexed_container_methods,
            direct_type_queries=direct_type_queries, indexed_entity_queries=indexed_entity_queries,
            runtime_primitive_compare=runtime_primitive_compare,direct_deepcopy_calls=direct_deepcopy_calls,
            native_hydration_probe=native_hydration_probe,hydration_program=hydration_program,
            native_preview_arena=native_preview_arena,preview_program=preview_program)
        self.image,self.schema,self.limits,self.seconds=image,schema,limits,seconds
        self.max_nodes,self.max_edges,self.max_payload_bytes=max_nodes,max_edges,max_payload_bytes
        self.source_sha256=self.executor.source_sha256
        self.hydration_probe_binding=self.executor.hydration_probe_binding
        self.canonical_rule_source_sha256=self.executor.canonical_rule_source_sha256
        self.rule_source_sha256=self.executor.rule_source_sha256
        self.preview_arena_binding=self.executor.preview_arena_binding
        self.preview_arena_identity=self.executor.preview_arena_identity
        self._constants={key:np.ascontiguousarray(getattr(image,key)) for key in ('nodes','edges','payload')}
        self.closed=False
        if directory is not None:
            from .native_soa_source import emit_native_soa_source
            output=Path(directory);output.mkdir(parents=True,exist_ok=True)
            source = emit_native_soa_source(image,schema,limits,
                stack_runs=stack_runs,native_functions=native_functions,native_values=native_values)[0]
            if execute_only:
                from .native_soa_source import execute_only_source
                source = execute_only_source(source)
            (output/'native_soa.cu').write_text(source,encoding='utf8')
            if self.hydration_probe_binding is not None:
                import json
                (output/'native_hydration_probe_binding.json').write_text(
                    json.dumps(dict(unpatched_monolithic_source_sha256=self.source_sha256,
                        actual_post_split_binding=self.hydration_probe_binding.to_dict()),indent=2)+'\n',encoding='utf8')
            if self.preview_arena_binding is not None:
                import json
                (output/'native_preview_arena_binding.json').write_text(
                    json.dumps(dict(unpatched_monolithic_source_sha256=self.source_sha256,
                        canonical_rule_source_sha256=self.canonical_rule_source_sha256,
                        actual_rule_source_sha256=self.rule_source_sha256,
                        identity=self.preview_arena_identity,
                        actual_post_split_binding=self.preview_arena_binding),indent=2)+'\n',encoding='utf8')

    def call(self,entry,state,arguments,*,state_positions=(0,),steps=None):
        if self.closed:
            raise RuntimeError('CUDA SoA diagnostic executor is closed')
        if not isinstance(state,RecordSoABatch) or state.schema.identity!=self.schema.identity:
            raise RecordSoAError('Matching typed diagnostic columns required')
        if steps is not None and steps!=self.limits.steps:
            raise ValueError('CUDA diagnostic step limit is fixed at setup')
        if type(arguments) is not list or len(arguments)!=state.rows or state.rows<1:
            raise ValueError('Matching nonempty argument rows required')
        for row in arguments:
            for position in state_positions:
                if type(position) is not int or not 0<=position<len(row) or row[position] is not None:
                    raise RecordSoAError('State arguments require None placeholders')
            if any(isinstance(value,dict) and {'sides','rng','next_entity'}<=set(value) for value in row):
                raise RecordSoAError('Game graph ingress is forbidden in typed diagnostic calls')
        packed,refs=pack_call_arguments(arguments,GraphLayout(state.rows,self.max_nodes,self.max_edges,self.max_payload_bytes))
        t,device=self.executor.torch,self.executor.device
        scratch=DeviceGraph(packed.layout,*(t.tensor(getattr(packed,key),device=device)
            for key in ('nodes','edges','payload','counts','roots')))
        typed=np.zeros_like(refs)
        for position in state_positions:
            typed[:,position]=state.roots
        deadlines=t.full((state.rows,),self.executor.clock_ns()+int(self.seconds*1e9),dtype=t.int64,device=device)
        call=self.executor.launch(entry,state.to_torch(device),scratch,t.tensor(refs,device=device),
                                  t.tensor(typed,device=device),deadlines=deadlines)
        t.cuda.current_stream(device).synchronize()
        counts=call.scratch.counts.cpu().numpy().copy()
        used=np.maximum(counts.max(axis=0),1)
        heap={key:np.ascontiguousarray(getattr(call.scratch,key)[:,:int(used[i])].cpu().numpy())
              for i,key in enumerate(('nodes','edges','payload'))}
        heap['counts']=counts
        return NativeSoACall(self,call.state.to_host(),heap,call.outputs.cpu().numpy().copy(),
            call.statuses.cpu().numpy().copy(),call.roots.cpu().numpy().copy(),
            call.context[0].cpu().numpy().copy(),())

    def close(self):
        if not self.closed:
            self.executor.close();self.closed=True

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
