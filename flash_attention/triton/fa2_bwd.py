import torch
import triton
import triton.language as tl
import math

from utils.tuner import tuner, Param
from utils.python import ceil_pow2, ceil_div, get_dir



@triton.jit
def flash_attn_bwd_preprocess_kernel(
    out_ptr, out_sB, out_sH, out_sM,
    dq_ptr, dq_sB, dq_sH, dq_sM,
    dout_ptr, dout_sB, dout_sH, dout_sM,
    delta_ptr, delta_sB, delta_sH,
    M, D,
    BLK_SIZE_M: tl.constexpr, BLK_SIZE_D: tl.constexpr, DETERMINISTIC: tl.constexpr
):
    idx_B = tl.program_id(2)
    idx_H = tl.program_id(1)
    idx_M = tl.arange(0, BLK_SIZE_M) + tl.program_id(0) * BLK_SIZE_M
    idx_D = tl.arange(0, BLK_SIZE_D)

    out_ptr += idx_B * out_sB + idx_H * out_sH
    dq_ptr += idx_B * dq_sB + idx_H * dq_sH
    dout_ptr += idx_B * dout_sB + idx_H * dout_sH
    delta_ptr += idx_B * delta_sB + idx_H * delta_sH

    mask_M = idx_M < M
    mask_D = idx_D < D
    mask_MD = mask_M[:, None] & mask_D[None, :]

    out_ptrs = out_ptr + idx_M[:, None] * out_sM + idx_D[None, :]
    dq_ptrs = dq_ptr + idx_M[:, None] * dq_sM + idx_D[None, :]
    dout_ptrs = dout_ptr + idx_M[:, None] * dout_sM + idx_D[None, :]

    tile_out = tl.load(out_ptrs, mask=mask_MD, other=0.0)
    tile_dout = tl.load(dout_ptrs, mask=mask_MD, other=0.0)

    delta = tl.sum(tile_out.to(tl.float32) * tile_dout.to(tl.float32), axis=1)
    if not DETERMINISTIC:
        tl.store(dq_ptrs, tl.zeros((BLK_SIZE_M, BLK_SIZE_D), dtype=tl.float32), mask=mask_MD)
    tl.store(delta_ptr + idx_M, delta, mask=mask_M)


@triton.jit
def flash_attn_bwd_main_kernel(
    q_ptr, q_sB, q_sH, q_sM,
    k_ptr, k_sB, k_sH, k_sN,
    v_ptr, v_sB, v_sH, v_sN,
    dout_ptr, dout_sB, dout_sH, dout_sM,
    lse_ptr, lse_sB, lse_sH,
    delta_ptr, delta_sB, delta_sH,
    dq_ptr, dq_sB, dq_sH, dq_sM,
    dk_ptr, dk_sB, dk_sH, dk_sN,
    dv_ptr, dv_sB, dv_sH, dv_sN,
    M, N, D,
    scale, scale_log2e,                            
    BLK_SIZE_D: tl.constexpr, CAUSAL: tl.constexpr, DETERMINISTIC: tl.constexpr,
    BLK_SIZE_M: tl.constexpr, BLK_SIZE_N: tl.constexpr,
):

    idx_B = tl.program_id(2)
    idx_H = tl.program_id(1)
    idx_N = tl.arange(0, BLK_SIZE_N) + tl.program_id(0) * BLK_SIZE_N
    idx_D = tl.arange(0, BLK_SIZE_D)
    
    q_ptr += idx_B * q_sB + idx_H * q_sH
    k_ptr += idx_B * k_sB + idx_H * k_sH
    v_ptr += idx_B * v_sB + idx_H * v_sH
    lse_ptr += idx_B * lse_sB + idx_H * lse_sH
    dout_ptr += idx_B * dout_sB + idx_H * dout_sH
    delta_ptr += idx_B * delta_sB + idx_H * delta_sH
    dk_ptr += idx_B * dk_sB + idx_H * dk_sH
    dv_ptr += idx_B * dv_sB + idx_H * dv_sH
    if not DETERMINISTIC:
        dq_ptr += idx_B * dq_sB + idx_H * dq_sH
        

    mask_N = idx_N < N
    mask_D = idx_D < D
    mask_ND = mask_N[:, None] & mask_D[None, :]

    k_ptrs = k_ptr + idx_N[:, None] * k_sN + idx_D[None, :]
    v_ptrs = v_ptr + idx_N[:, None] * v_sN + idx_D[None, :]

    tile_k = tl.load(k_ptrs, mask=mask_ND, other=0.0)
    tile_v = tl.load(v_ptrs, mask=mask_ND, other=0.0)

    tile_dk = tl.zeros(tile_k.shape, tl.float32)
    tile_dv = tl.zeros(tile_v.shape, tl.float32)

    start = 0
    if CAUSAL:
        start = tl.program_id(0) * BLK_SIZE_N

    for m_i in range (start, M, BLK_SIZE_M):
        idx_M = tl.arange(0, BLK_SIZE_M) + m_i
        mask_M = idx_M < M 

        dout_ptrs =  dout_ptr + idx_M[:, None] *  dout_sM + idx_D[None, :]
        q_ptrs = q_ptr + idx_M[:, None] * q_sM + idx_D[None, :]
        if not DETERMINISTIC:
            dq_ptrs = dq_ptr + idx_M[:, None] * dq_sM + idx_D[None, :]
        delta_ptrs = delta_ptr + idx_M
        lse_ptrs = lse_ptr + idx_M
        mask_MD = mask_M[:, None] & mask_D[None, :]

        tile_dout = tl.load(dout_ptrs, mask=mask_MD, other=0.0)
        tile_q = tl.load(q_ptrs, mask=mask_MD, other=0.0)
        tile_delta = tl.load(delta_ptrs, mask=mask_M, other=0.0)[:, None] # (BLK_SIZE_M, 1)
        tile_lse = tl.load(lse_ptrs, mask=mask_M, other=0.0)[:, None] 

        
        tile_p = tl.math.exp2(scale_log2e * tl.dot(tile_q, tile_k.T) - tile_lse) 
        mask_p = mask_N[None, :]
        if CAUSAL:
            mask_p &= (idx_M[:, None] >= idx_N[None, :])

        tile_p = tl.where(mask_p, tile_p, 0.0)
    
        tile_dp = tl.dot(tile_dout, tile_v.T)
        tile_ds = tile_p * (tile_dp - tile_delta).to(tl.float32)
        if not DETERMINISTIC:
            tile_dq = tl.dot(tile_ds.to(tile_q.dtype), tile_k)
            tl.atomic_add(dq_ptrs, scale * tile_dq, mask=mask_MD, sem="relaxed")
        tile_dv = tl.dot(tile_p.T.to(tile_dout.dtype), tile_dout, tile_dv)
        tile_dk = tl.dot(tile_ds.T.to(tile_q.dtype), tile_q, tile_dk)

    tile_dk *= scale 
    dv_ptrs = dv_ptr + idx_N[:, None] * dv_sN + idx_D[None, :]
    dk_ptrs = dk_ptr + idx_N[:, None] * dk_sN + idx_D[None, :]

    tl.store(dv_ptrs, tile_dv.to(dv_ptr.dtype.element_ty), mask=mask_ND)
    tl.store(dk_ptrs, tile_dk.to(dk_ptr.dtype.element_ty), mask=mask_ND) 

    
@triton.jit
def flash_attn_bwd_dq_kernel(
    q_ptr, q_sB, q_sH, q_sM,
    k_ptr, k_sB, k_sH, k_sN,
    v_ptr, v_sB, v_sH, v_sN,
    dout_ptr, dout_sB, dout_sH, dout_sM,
    lse_ptr, lse_sB, lse_sH,
    delta_ptr, delta_sB, delta_sH,
    dq_ptr, dq_sB, dq_sH, dq_sM,
    M, N, D,
    scale, scale_log2e,
    BLK_SIZE_D: tl.constexpr, CAUSAL: tl.constexpr,
    BLK_SIZE_M: tl.constexpr, BLK_SIZE_N: tl.constexpr,
):
    idx_B = tl.program_id(2)
    idx_H = tl.program_id(1)
    idx_M = tl.arange(0, BLK_SIZE_M) + tl.program_id(0) * BLK_SIZE_M
    idx_D = tl.arange(0, BLK_SIZE_D)
    
    q_ptr += idx_B * q_sB + idx_H * q_sH
    k_ptr += idx_B * k_sB + idx_H * k_sH
    v_ptr += idx_B * v_sB + idx_H * v_sH
    lse_ptr += idx_B * lse_sB + idx_H * lse_sH
    dout_ptr += idx_B * dout_sB + idx_H * dout_sH
    delta_ptr += idx_B * delta_sB + idx_H * delta_sH
    dq_ptr += idx_B * dq_sB + idx_H * dq_sH

    q_ptrs = q_ptr + idx_M[:, None] * q_sM + idx_D[None, :]
    dout_ptrs = dout_ptr + idx_M[:, None] * dout_sM + idx_D[None, :]
    lse_ptrs = lse_ptr + idx_M 
    delta_ptrs = delta_ptr + idx_M

    mask_M = idx_M < M
    mask_D = idx_D < D
    mask_MD = mask_M[:, None] & mask_D[None, :]

    tile_q = tl.load(q_ptrs, mask=mask_MD, other=0.0)
    tile_dout = tl.load(dout_ptrs, mask=mask_MD, other=0.0)
    tile_lse = tl.load(lse_ptrs, mask=mask_M, other=0.0)[:, None]
    tile_delta = tl.load(delta_ptrs, mask=mask_M, other=0.0)[:, None]


    tile_dq = tl.zeros(tile_q.shape, tl.float32)

    bound = N 
    if CAUSAL:
        bound = (tl.program_id(0) + 1) * BLK_SIZE_M

    for n_i in range (0, bound, BLK_SIZE_N):
        idx_N = tl.arange(0, BLK_SIZE_N) + n_i
        mask_N = idx_N < N

        k_ptrs = k_ptr + idx_N[:, None] * k_sN + idx_D[None, :]
        v_ptrs = v_ptr + idx_N[:, None] * v_sN + idx_D[None, :]
        
        mask_ND = mask_N[:, None] & mask_D[None, :]

        tile_k = tl.load(k_ptrs, mask=mask_ND, other=0.0)
        tile_v = tl.load(v_ptrs, mask=mask_ND, other=0.0)
        
        tile_p = tl.math.exp2(scale_log2e * tl.dot(tile_q, tile_k.T) - tile_lse) 

        mask_p = mask_N[None, :]
        if CAUSAL:
            mask_p &= (idx_M[:, None] >= idx_N[None, :])
        tile_p = tl.where(mask_p, tile_p, 0.0)
    
        tile_dp = tl.dot(tile_dout, tile_v.T)
        tile_ds = tile_p * (tile_dp - tile_delta).to(tl.float32)
        tile_dq = tl.dot(tile_ds.to(tile_q.dtype), tile_k, tile_dq)
    
    tile_dq *= scale 
    dq_ptrs = dq_ptr + idx_M[:, None] * dq_sM + idx_D[None, :]

    tl.store(dq_ptrs, tile_dq.to(dq_ptr.dtype.element_ty), mask=mask_MD)    


BWD_PARAMS = {
    "BLK_SIZE_M": Param((32, 64, 128), 64), 
    "BLK_SIZE_N": Param((32, 64, 128), 64),
    "num_warps": Param((4, 8), 4),
    "num_stages": Param((1, 2, 3), 2)
}

def bwd_args2keys(K, causal, deterministic):
    return {
        "N": ceil_pow2(K.shape[2]), 
        "BLK_SIZE_D": max(16, ceil_pow2(K.shape[3])), 
        "CAUSAL": int(causal), 
        "DETERMINISTIC": int(deterministic)
    }

@tuner(get_dir(__file__) / "bwd_main.csv", BWD_PARAMS, bwd_args2keys, invalidate_cache=False)
def flash_attn_bwd_main(
    Q, K, V, 
    dO, lse, delta, 
    dQ, dK, dV, 
    causal, deterministic, 
    B, H, M, N, D, 
    scale, scale_log2e, 
    BLK_SIZE_D, *, BLK_SIZE_M, BLK_SIZE_N, 
    num_stages, num_warps
):
    grid = (ceil_div(N, BLK_SIZE_N), H, B)
    flash_attn_bwd_main_kernel[grid]( 
        Q, *Q.stride()[:3],
        K, *K.stride()[:3],
        V, *V.stride()[:3],
        dO, *dO.stride()[:3],
        lse, *lse.stride()[:2],
        delta, *delta.stride()[:2],
        dQ, *dQ.stride()[:3],
        dK, *dK.stride()[:3],
        dV, *dV.stride()[:3],
        M, N, D,
        scale, scale_log2e,
        BLK_SIZE_D, causal, deterministic,
        BLK_SIZE_M, BLK_SIZE_N,
        num_warps=num_warps, num_stages=num_stages
    )

@tuner(get_dir(__file__) / "bwd_dq.csv", BWD_PARAMS, bwd_args2keys, invalidate_cache=False)
def flash_attn_bwd_dq(
    Q, K, V, 
    dO, lse, delta, 
    dQ, dK, dV, 
    causal, deterministic, 
    B, H, M, N, D, 
    scale, scale_log2e, 
    BLK_SIZE_D, *, BLK_SIZE_M, BLK_SIZE_N, 
    num_stages, num_warps
):
    grid = (ceil_div(M, BLK_SIZE_M), H, B)
    flash_attn_bwd_dq_kernel[grid](
        Q, *Q.stride()[:3],
        K, *K.stride()[:3],
        V, *V.stride()[:3],
        dO, *dO.stride()[:3],
        lse, *lse.stride()[:2],
        delta, *delta.stride()[:2],
        dQ, *dQ.stride()[:3],
        M, N, D,
        scale, scale_log2e,
        BLK_SIZE_D, causal,
        BLK_SIZE_M, BLK_SIZE_N,
        num_warps=num_warps, num_stages=num_stages
    )

def flash_attn_bwd(Q, K, V, O, dO, lse, causal, deterministic, out=None):
    assert Q.shape == K.shape == V.shape == O.shape == dO.shape
    assert all(t.stride(-1) == 1 for t in (Q, K, V, O, dO)), "last dim should be stride 1"

    keys = bwd_args2keys(K, causal, deterministic)

    B, H, M, D = Q.shape
    N = K.shape[2]

    PREPROCESS_BLK_SIZE_M = 128
    BLK_SIZE_D = keys["BLK_SIZE_D"]

    delta, dQ, dK, dV = out if out is not None else (
        torch.empty((B, H, M), device=Q.device, dtype=torch.float32), 
        torch.empty(Q.shape, device=Q.device, dtype=torch.float32 if not deterministic else Q.dtype), 
        torch.empty(K.shape, device=K.device, dtype=K.dtype), 
        torch.empty(V.shape, device=V.device, dtype=V.dtype), 
    )

    if out is not None and not deterministic and dQ.dtype != torch.float32 :
        raise ValueError("Non-deterministic output requires dQ (out[1]) to be float32")

    
    scale = 1.0 / (D ** 0.5)
    scale_log2e = scale * math.log2(math.e)

    flash_attn_bwd_preprocess_kernel[(ceil_div(M, PREPROCESS_BLK_SIZE_M), H, B)](
        O, *O.stride()[:3],
        dQ, *dQ.stride()[:3],
        dO, *dO.stride()[:3],
        delta, *delta.stride()[:2],
        M, D,
        PREPROCESS_BLK_SIZE_M, BLK_SIZE_D, deterministic
    )

    
    flash_attn_bwd_main(
        Q, K, V, 
        dO, lse, delta, 
        dQ, dK, dV, 
        causal, deterministic, 
        B, H, M, N, D, 
        scale, scale_log2e, 
        BLK_SIZE_D
    )

    if deterministic:
        flash_attn_bwd_dq(
            Q, K, V, 
            dO, lse, delta, 
            dQ, dK, dV, 
            causal, deterministic, 
            B, H, M, N, D, 
            scale, scale_log2e, 
            BLK_SIZE_D
        )
        
    # non deterministic path accumulates in f32 atomics so downcast is needed
    dQ = dQ.to(Q.dtype) if not deterministic else dQ 

    return dQ, dK, dV
