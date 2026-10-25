import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math

SEARCH_SPACE = [
    triton.Config({"BLK_SIZE_M": 128, "BLK_SIZE_N":  32}, num_warps=4, num_stages=2),
    triton.Config({"BLK_SIZE_M": 128, "BLK_SIZE_N":  64}, num_warps=4, num_stages=2),
    triton.Config({"BLK_SIZE_M": 128, "BLK_SIZE_N":  64}, num_warps=8, num_stages=2),
    triton.Config({"BLK_SIZE_M": 128, "BLK_SIZE_N": 128}, num_warps=8, num_stages=1),
    triton.Config({"BLK_SIZE_M":  64, "BLK_SIZE_N":  64}, num_warps=4, num_stages=2),
    triton.Config({"BLK_SIZE_M":  64, "BLK_SIZE_N":  64}, num_warps=4, num_stages=3),
    triton.Config({"BLK_SIZE_M":  64, "BLK_SIZE_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLK_SIZE_M":  64, "BLK_SIZE_N":  32}, num_warps=4, num_stages=3),
    triton.Config({"BLK_SIZE_M":  32, "BLK_SIZE_N":  32}, num_warps=2, num_stages=2),
    triton.Config({"BLK_SIZE_M":  16, "BLK_SIZE_N":  16}, num_warps=2, num_stages=1),
]

@triton.autotune(configs=SEARCH_SPACE, key=["M", "N", "BLK_SIZE_D", "CAUSAL"], cache_results=True)
@triton.jit
def flash_attn_fwd_kernel(
    q_ptr, q_sB, q_sH, q_sM,
    k_ptr, k_sB, k_sH, k_sN,
    v_ptr, v_sB, v_sH, v_sN,
    out_ptr, out_sB, out_sH, out_sM,
    lse_ptr, lse_sB, lse_sH,
    M, N, D,
    scale_log2e,                            
    BLK_SIZE_M: tl.constexpr, BLK_SIZE_N: tl.constexpr,
    BLK_SIZE_D: tl.constexpr, CAUSAL: tl.constexpr
):
    idx_B = tl.program_id(2)
    idx_H = tl.program_id(1)
    idx_M = tl.arange(0, BLK_SIZE_M) + tl.program_id(0) * BLK_SIZE_M
    idx_D = tl.arange(0, BLK_SIZE_D)

    q_ptr += idx_B * q_sB + idx_H * q_sH
    k_ptr += idx_B * k_sB + idx_H * k_sH
    v_ptr += idx_B * v_sB + idx_H * v_sH
    out_ptr += idx_B * out_sB + idx_H * out_sH
    lse_ptr += idx_B * lse_sB + idx_H * lse_sH

    q_ptrs = q_ptr + idx_M[:, None] * q_sM + idx_D[None, :]

    mask_M = idx_M < M
    mask_D = idx_D < D
    mask_MD = mask_M[:, None] & mask_D[None, :]

    tile_q = tl.load(q_ptrs, mask=mask_MD, other=0.0)

    accum = tl.zeros((BLK_SIZE_M, BLK_SIZE_D), tl.float32)
    running_max = tl.full((BLK_SIZE_M,), float('-inf'), tl.float32)
    running_sum = tl.zeros((BLK_SIZE_M,), tl.float32)

    bound = N
    if CAUSAL:
        bound = (tl.program_id(0) + 1) * BLK_SIZE_M

    for kv_i in range(0, bound, BLK_SIZE_N):
        idx_N = tl.arange(0, BLK_SIZE_N) + kv_i
        mask_N = idx_N < N

        k_ptrs = k_ptr + idx_N[:, None] * k_sN + idx_D[None, :]
        v_ptrs = v_ptr + idx_N[:, None] * v_sN + idx_D[None, :]
        mask_ND = mask_N[:, None] & mask_D[None, :]

        tile_k = tl.load(k_ptrs, mask=mask_ND, other=0.0)
        tile_v = tl.load(v_ptrs, mask=mask_ND, other=0.0)

        tile_p = tl.dot(tile_q, tile_k.T) * scale_log2e 


        mask_p = mask_N[None, :]
        if CAUSAL:
            mask_p &= idx_M[:, None] >= idx_N[None, :]

        tile_p = tl.where(mask_p, tile_p, float('-inf'))

        current_max = tl.maximum(tl.max(tile_p, 1), running_max)
        tile_p_scores = tl.math.exp2(tile_p - current_max[:, None])
        rescale = tl.math.exp2(running_max - current_max)

        accum *= rescale[:, None]
        accum = tl.dot(tile_p_scores.to(tile_v.dtype), tile_v, accum)

        running_max = current_max
        running_sum = tl.fma(running_sum, rescale, tl.sum(tile_p_scores, 1))

    accum /= running_sum[:, None]

    out_ptrs = out_ptr + idx_M[:, None] * out_sM + idx_D[None, :]
    lse_ptrs = lse_ptr + idx_M

    tl.store(lse_ptrs, running_max + tl.math.log2(running_sum), mask=mask_M)
    tl.store(out_ptrs, accum.to(out_ptr.dtype.element_ty), mask=mask_MD)  


def flash_attn_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal):
    assert Q.shape == K.shape == V.shape
    assert Q.is_cuda and all(t.stride(-1) == 1 for t in (Q, K, V))


    B, H, M, D = Q.shape
    N = K.shape[2]
 

    BLK_SIZE_D = max(32, triton.next_power_of_2(D))


    output = torch.empty(Q.shape, device=Q.device, dtype=Q.dtype)
    lse = torch.empty((B, H, M), device=Q.device, dtype=torch.float32)

    grid = lambda params: (triton.cdiv(M, params["BLK_SIZE_M"]), H, B) # autotuner removes BLK_SIZE_M from scope

    scale_log2e = 1.0 / (D ** 0.5) * math.log2(math.e)

    flash_attn_fwd_kernel[grid](
        Q, *Q.stride()[:3],
        K, *K.stride()[:3],
        V, *V.stride()[:3],
        output, *output.stride()[:3],
        lse, *lse.stride()[:2],
        M, N, D,
        scale_log2e,
        BLK_SIZE_D=BLK_SIZE_D, CAUSAL=causal,
    )

    return (output, lse)
