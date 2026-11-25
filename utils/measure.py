from collections import deque

import torch
import pynvml
import threading
from collections.abc import Callable
from torch.utils._pytree import tree_map_only
from statistics import median

L2_BUFFER_SIZE = 1024 * 2**20 # 1 GB is larger than L2 cache for any currently available gpu

def benchmark(bench_fn: Callable, *args, bench_warmup_iters=20, bench_iters=100, bench_repeats = 5, bench_rotate_L2=True, **kwargs):
    packed_args = (args, kwargs)
    args_buffer = [packed_args]

    if bench_rotate_L2:
        args_buffer = []
        buffer_size = 0
        args_size = -1
        alias_table = {}
        
        def clone_helper(tensor):
            tensor_id = id(tensor)
            if tensor_id not in alias_table:
                alias_table[tensor_id] = tensor.detach().clone().requires_grad_(tensor.requires_grad)
            return alias_table[tensor_id]

        idx = 0
        while buffer_size < L2_BUFFER_SIZE and idx < bench_iters:
            alias_table = {}
            args_buffer.append(tree_map_only(torch.Tensor, clone_helper, packed_args))
            if idx == 0:
                args_size = sum(tensor.nbytes for tensor in alias_table.values())
            buffer_size += args_size
            idx += 1
            

    # launching on a sidestream is a bit overkill, only needed if the stream gets cached by fn
    outs = deque(maxlen=len(args_buffer)) # ring_buffer to keep size bounded
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(0, bench_warmup_iters):
            args_i, kwargs_i = args_buffer[i % len(args_buffer)] 
            outs.append(bench_fn(*args_i, **kwargs_i))
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    g = torch.cuda.CUDAGraph()

    
    with torch.cuda.graph(g):
        outs = deque(maxlen=len(args_buffer))
        for i in range(0, bench_iters):
            args_i, kwargs_i = args_buffer[i % len(args_buffer)] 
            outs.append(bench_fn(*args_i, **kwargs_i))

    g.replay()
    torch.cuda.synchronize()
    times = []
    for i in range(bench_repeats):
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / bench_iters)
    return median(times)


class MeasureMemoryClock:
    def __init__(self, interval=0.05):
        pynvml.nvmlInit()
        self.handle = torch.cuda.current_device()
        self.samples = []
        self._stop = threading.Event()
        self.interval = interval
        
    def sample_clk(self, interval):
        while not self._stop.is_set():
            self.samples.append(
                pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            )
            self._stop.wait(interval)
    
    def __enter__(self):
        self._stop.clear()
        self.samples = []
        self.thread = threading.Thread(target=self.sample_clk, args=(self.interval,), daemon=True)
        self.thread.start()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        self.thread.join()
        self.samples.sort()
        self.median = self.samples[len(self.samples) // 2]
