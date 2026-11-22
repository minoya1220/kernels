import torch
from softmax_fwd import softmax_fwd_builder
from softmax_bwd import softmax_bwd_builder
from utils.plot import plot_bench
from utils.measure import benchmark
from utils.python import get_dir

BENCH_SIZE = 2**29

BENCH_DIMS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]




def benchmark_fwd(dims: list[int], *, dtype=torch.bfloat16, elems=BENCH_SIZE) -> list[float]:
    times = []
    
    for dim in dims:
        X = torch.randn(elems // dim, dim, device='cuda', dtype=dtype)
        Y = torch.empty_like(X)
        softmax_fwd = softmax_fwd_builder(X)

        times.append(benchmark(softmax_fwd, X, out=Y))
    
    return times


def benchmark_bwd(dims: list[int], *, dtype=torch.bfloat16, elems=BENCH_SIZE) -> list[float]:
    times = []
    for dim in dims:
        Y = torch.randn(elems//dim, dim, device='cuda', dtype=dtype)
        dY = torch.randn_like(Y)
        dX = torch.empty_like(Y)
        softmax_bwd = softmax_bwd_builder(Y, dY)
        times.append(benchmark(softmax_bwd, Y, dY, out=dX))
    
    return times




if __name__ == "__main__":
    
    torch.manual_seed(123)

    
    # with MeasureMemoryClock(interval=0.01) as mem_clk:
    fwd_times = benchmark_fwd(BENCH_DIMS)
    x_ref = torch.randn(BENCH_SIZE, dtype=torch.bfloat16, device='cuda')
    fwd_reference = {'torch abs (1r:1w)': benchmark(torch.abs, x_ref)}
    plot_bench(BENCH_DIMS, fwd_times, 'Softmax Forward', nbytes=BENCH_SIZE*4, reference=fwd_reference, path=get_dir(__file__) / "plots", xlabel="Row Length")
    
    bwd_times = benchmark_bwd(BENCH_DIMS)
    y_ref = torch.randn_like(x_ref)
    z_ref = torch.randn_like(x_ref)
    bwd_reference = {'torch add (2r:1w)': benchmark(torch.add, y_ref, z_ref)}
    plot_bench(BENCH_DIMS, bwd_times, 'Softmax Backward', nbytes=BENCH_SIZE*6, reference=bwd_reference, path=get_dir(__file__) / "plots", xlabel="Row Length")
    
    # from torch.profiler import profile, ProfilerActivity
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    # ) as prof:
    #     benchmark_fwd([2**14])
    # prof.export_chrome_trace("trace.json")

    # test_fwd([2**16])

    # TODO:  pack bf16s, do writeup, rename repo, add controls to enable unoptimized versions
    # On packing -> i think we should have a path where if the width is 16 bits or less floats can get packed as bf16s (lower dtypes get upcasted to bf16 bc no inf) or halfs
    print("success")

# # profiling cmd
# ncu --set full -f -o softmax_fwd -k regex:cutlass_softmax_fwd --launch-skip 3 --launch-count 5 uv run softmax.py

