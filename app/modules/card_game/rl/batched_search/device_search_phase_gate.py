"""GPU-only one-shot branches for already-initialized search FSM phases."""
from .cuda_conditional_graph import CudaConditionalWhile


class DeviceSearchPhaseGates:
    def __init__(self, *, phases, device):
        import torch
        self.t=torch;self.device=torch.device(device);self.loops=[]
        try:
            # The enclosing search owns the real deadline. These one-shot
            # branches only guard against accidentally repeating a phase.
            for unused in range(phases):
                self.loops.append(CudaConditionalWhile(device=device,max_ticks=2,seconds=86400))
            self.predicates=[torch.zeros(1,dtype=torch.uint8,device=device)for unused in range(phases)]
        except BaseException:self.close();raise

    def register_capture_stream_callback(self, callback):
        for loop in self.loops:loop.register_capture_stream_callback(callback)

    def capture(self, phase, active, body):
        if not self.t.cuda.is_current_stream_capturing():
            raise RuntimeError('Search phase branches require an enclosing CUDA capture')
        predicate=self.predicates[phase]
        predicate.copy_(active.any().reshape(1).to(self.t.uint8))
        def once():
            body()
            predicate.zero_()
        loop=self.loops[phase]
        loop.capture_loop(predicate,once)
        return loop.errors

    def close(self):
        for loop in self.loops:loop.close()
