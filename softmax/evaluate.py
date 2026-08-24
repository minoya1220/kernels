import torch
from softmax.softmax_fwd import softmax_fwd_builder
from softmax.softmax_bwd import softmax_bwd_builder
from softmax.plot import plot_bench
import matplotlib.pyplot as plt
from measure import benchmark
from matplotlib.ticker import FuncFormatter
from pathlib import Path

BENCH_SIZE = 2**29
TEST_M = 2**6 - 1

TEST_DIMS = [3, 7, 8, 17, 64, 100, 257, 511, 633, 1021, 2049, 4099]
BENCH_DIMS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def test_fwd(dims: list[int], *, dtypes=(torch.bfloat16, torch.half, torch.float, torch.double), verbose=False):#, torch.half, torch.float, torch.double]):
    failures = []
    for dtype in dtypes:
        for dim in dims:
            X = torch.randn(TEST_M, dim, device='cuda', dtype=dtype)

            torch_softmax = torch.softmax(X.double(), dim=1).to(X.dtype)
            softmax_fwd = softmax_fwd_builder(X)
            try:
                torch.testing.assert_close(torch_softmax, softmax_fwd(X))
            except Exception as e:
                failures.append((dim, dtype, e))
                print(f"fwd: dim={dim}, dtype={dtype} failed")
            else:
                if verbose:
                    print(f"fwd: dim={dim}, dtype={dtype}  passed")
    if failures:
        raise AssertionError(
            f"fwd: {len(failures)}/{len(dims) * len(dtypes)} failed: "
            + ", ".join(f"(dim={d}, {t})" for d, t, _ in failures)
        )

def test_bwd(dims: list[int], *, dtypes=(torch.bfloat16, torch.half, torch.float, torch.double), verbose=False):
    failures = []
    for dtype in dtypes:
        for dim in dims:
            X = torch.randn(TEST_M, dim, device='cuda', dtype=dtype)
            dY = torch.randn_like(X)
            
            Y = torch.softmax(X, dim=1).to(dtype)
            Y_double, dY_double = Y.double(), dY.double()
            ref_softmax_bwd = (Y_double * (dY_double - (Y_double * dY_double).sum(dim=1, keepdim=True))).to(dtype)

            softmax_bwd = softmax_bwd_builder(Y, dY)
            try:
                torch.testing.assert_close(ref_softmax_bwd, softmax_bwd(Y, dY))
            except Exception as e:
                failures.append((dim, dtype, e))
                print(f"bwd: dim={dim}, dtype={dtype} failed")
            else:
                if verbose:
                    print(f"bwd: dim={dim}, dtype={dtype} passed")
    if failures:
        raise AssertionError(
            f"bwd: {len(failures)}/{len(dims) * len(dtypes)} failed: "
            + ", ".join(f"(dim={d}, {t})" for d, t, _ in failures)
        )

def benchmark_fwd(dims: list[int], *, dtype=torch.bfloat16, elems=BENCH_SIZE) -> list[float]:
    times = []
    
    for dim in dims:
        X = torch.randn(elems // dim, dim, device='cuda', dtype=dtype)
        Y = torch.empty_like(X)
        softmax_fwd = softmax_fwd_builder(X)

        times.append(benchmark(softmax_fwd, [X], [Y]))
    
    return times


def benchmark_bwd(dims: list[int], *, dtype=torch.bfloat16, elems=BENCH_SIZE) -> list[float]:
    times = []
    for dim in dims:
        Y = torch.randn(elems//dim, dim, device='cuda', dtype=dtype)
        dY = torch.randn_like(Y)
        dX = torch.empty_like(Y)
        softmax_bwd = softmax_bwd_builder(Y, dY)
        times.append(benchmark(softmax_bwd, [Y, dY], [dX]))
    
    return times




if __name__ == "__main__":
    
    torch.manual_seed(123)

    test_fwd(BENCH_DIMS + TEST_DIMS, dtypes=(torch.bfloat16,))
    test_bwd(BENCH_DIMS + TEST_DIMS, dtypes=(torch.bfloat16,))
    print("tests pass")
    
    # with MeasureMemoryClock(interval=0.01) as mem_clk:
    fwd_times = benchmark_fwd(BENCH_DIMS)
    plot_bench(BENCH_DIMS, fwd_times, 1, BENCH_SIZE, 'Softmax Forward', color='#83a598', reference=True)
    
    bwd_times = benchmark_bwd(BENCH_DIMS)
    plot_bench(BENCH_DIMS, bwd_times, 2, BENCH_SIZE, 'Softmax Backward', color='#fabd2f', reference=True)
    
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

