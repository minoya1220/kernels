from collections.abc import Callable
import torch
from fa2_fwd import flash_attn_fwd
from fa2_bwd import flash_attn_bwd
from utils.measure import benchmark
from utils.plot import plot_bench
from utils.python import get_dir




def benchmark_fwd(fn: Callable | list[Callable], seq_lens: list[int], H, D, num_tokens, *, causal, dtype=torch.bfloat16) -> list[float]:
    if isinstance(fn, Callable):
        fn = [fn]

    times = []
    for func in fn:
        func_times = []
        for N in seq_lens:
            B = num_tokens // N
            Q = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
            K = torch.randn_like(Q)
            V = torch.randn_like(Q)
    
            func_times.append(benchmark(func, Q, K, V, causal=causal))
        times.append(func_times)
        
    return times



def benchmark_bwd(fn: Callable | list[Callable], seq_lens: list[int], H, D, num_tokens, *, causal, deterministic, dtype=torch.bfloat16) -> list[float]:
    if isinstance(fn, Callable):
        fn = [fn]

    times = []
    for func in fn:
        func_times = []
        for N in seq_lens:
            B = num_tokens // N
            Q = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
            K = torch.randn_like(Q)
            V = torch.randn_like(Q)
            O = torch.randn_like(Q)
            dO = torch.randn_like(Q)
            lse = torch.randn(B, H, N, device='cuda', dtype=torch.float32)
            
            func_times.append(benchmark(func, Q, K, V, O, dO, lse, causal=causal, deterministic=deterministic))
        times.append(func_times)
        
    return times

def _attn_fwd_reference(Q, K, V, *, causal):
    S = Q @ K.mT * (Q.shape[-1] ** -0.5) 
    if causal:
        mask = torch.ones(Q.shape[-2], K.shape[-2], dtype=torch.bool, device=Q.device).triu(diagonal=1)
        S = S.masked_fill(mask, float('-inf'))

    P = S.softmax(dim=-1)
    O = P @ V 
    return O
    
def _attn_bwd_reference(Q, K, V, O, dO, lse, *, causal, deterministic):
    scale = (Q.shape[-1] ** -0.5)
    S = Q @ K.mT * scale

    if causal:
        mask = torch.ones(Q.shape[-2], K.shape[-2], dtype=torch.bool, device=Q.device).triu(diagonal=1)
        S = S.masked_fill(mask, float('-inf'))

    P = torch.exp(S - lse.unsqueeze(-1))
    delta = (dO * O).sum(dim=-1)
    dV = P.mT @ dO 
    dP = dO @ V.mT 
    dS = P * (dP - delta.unsqueeze(-1)) 
    dQ = dS @ K * scale
    dK = dS.mT @ Q * scale

    return dQ, dK, dV


if __name__ == "__main__":
    SEQ_LENS = [2**i for i in range(7,12)] # 128 to 4096 in pow2
    NUM_TOKENS = 2**15
    H, D = 16, 128
    is_causal = True


    fwd_labels = ["naive torch eager", "sdpa flash attn", "triton fwd"]
    fwd_flash_reference = lambda Q, K, V, *, causal: (
        torch.ops.aten._scaled_dot_product_flash_attention(Q, K , V, 0.0, causal)
    )
    fwd_times = benchmark_fwd(
        [_attn_fwd_reference, fwd_flash_reference, flash_attn_fwd], 
        SEQ_LENS, H, D, NUM_TOKENS, causal=is_causal
    )
    fwd_times = dict(zip(fwd_labels, fwd_times))
    fwd_flops = lambda N: 4 * NUM_TOKENS * N * H * D * (0.5 if is_causal else 1) # 4 B * H * N^2 * D * (0.5 if causal else 1) for the 2 fwd matmuls (2NDN + 2NND)


    bwd_labels = ["naive torch eager", "sdpa flash attn bwd", "triton bwd", "triton deterministic bwd"]
    bwd_flash_reference = lambda Q, K, V, O, dO, lse, *, causal, deterministic: (
        torch.ops.aten._scaled_dot_product_flash_attention_backward(
            dO, Q, K, V, O, lse, None, None, Q.shape[-2], K.shape[-2], 0.0, causal, torch.empty(0), torch.empty(0)
        )
    )
    bwd_triton_deterministic = lambda Q, K, V, O, dO, lse, *, causal, deterministic: (
        flash_attn_bwd(Q, K, V, O, dO, lse, causal=causal, deterministic=True)
    )
    bwd_times = benchmark_bwd(
        [_attn_bwd_reference, bwd_flash_reference, flash_attn_bwd, bwd_triton_deterministic], 
        SEQ_LENS, H, D, NUM_TOKENS, causal=is_causal, deterministic=False
    )
    bwd_times = dict(zip(bwd_labels, bwd_times))
    bwd_flops = lambda N: 10 * NUM_TOKENS * N * H * D * (0.5 if is_causal else 1) # same but 5 matmuls in the bwd


    
    plot_bench(SEQ_LENS, fwd_times, "Attention Forward", flops=fwd_flops, path=get_dir(__file__) / "plots", xlabel="Sequence Length")
    plot_bench(SEQ_LENS, bwd_times, "Attention Backward", flops=bwd_flops, path=get_dir(__file__) / "plots", xlabel="Sequence Length")
    
