import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math

from triton_fa2 import flash_attn_fwd



@triton.jit
def flash_attn_preprocess_kernel(
    out_ptr, out_sB, out_sH, out_sM,
    dout_ptr, dout_sB, dout_sH, dout_sM,
    delta_ptr, delta_sB, delta_sH,
    M, D,
    BLK_SIZE_M: tl.constexpr, BLK_SIZE_D: tl.constexpr
):
    idx_B = tl.program_id(2)
    idx_H = tl.program_id(1)
    idx_M = tl.arange(0, BLK_SIZE_M) + tl.program_id(0) * BLK_SIZE_M
    idx_D = tl.arange(0, BLK_SIZE_D)

    out_ptr += idx_B * out_sB + idx_H * out_sH
    dout_ptr += idx_B * dout_sB + idx_H * dout_sH
    delta_ptr += idx_B * delta_sB + idx_H * delta_sH

    mask_M = idx_M < M
    mask_D = idx_D < D
    mask_MD = mask_M[:, None] & mask_D[None, :]

    out_ptrs = out_ptr + idx_M[:, None] * out_sM + idx_D[None, :]
    dout_ptrs = dout_ptr + idx_M[:, None] * dout_sM + idx_D[None, :]

    tile_out = tl.load(out_ptrs, mask=mask_MD, other=0.0)
    tile_dout = tl.load(dout_ptrs, mask=mask_MD, other=0.0)

    delta = tl.sum(tile_out.to(tl.float32) * tile_dout.to(tl.float32), axis=1)

    tl.store(delta_ptr + idx_M, delta, mask=mask_M)


@triton.jit
def flash_attn_bwd_kernel(
    q_ptr, q_sB, q_sH, q_sM,
    k_ptr, k_sB, k_sH, k_sN,
    v_ptr, v_sB, v_sH, v_sN,
    dout_ptr, dout_sB, dout_sH, dout_sM,
    lse_ptr, lse_sB, lse_sH,
    delta_ptr, delta_sB, delta_sH,
    dk_ptr, dk_sB, dk_sH, dk_sN,
    dv_ptr, dv_sB, dv_sH, dv_sN,
    dq_ptr, dq_sB, dq_sH, dq_sM,
    M, N, D,
    scale,                                
    scale_log2e,                            
    BLK_SIZE_M: tl.constexpr, BLK_SIZE_N: tl.constexpr,
    BLK_SIZE_D: tl.constexpr, CAUSAL: tl.constexpr, DETERMINISTIC: tl.constexpr
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

        
        tile_p = tl.math.exp2(scale_log2e * (tile_q @ tile_k.T) - tile_lse) 
        if CAUSAL:
            mask_p = mask_N[None, :] & (idx_M[:, None] >= idx_N[None, :])
            tile_p = tl.where(mask_p, tile_p, 0.0)
    
        tile_dp = tile_dout @ tile_v.T 
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
    scale,
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
    lse_ptr += idx_B * lse_sB + idx_H * lse_sH
    dout_ptr += idx_B * dout_sB + idx_H * dout_sH
    delta_ptr += idx_B * delta_sB + idx_H * delta_sH
    dq_ptr += idx_B * dq_sB + idx_H * dq_sH

    mask_M = idx_M < M
    mask_D = idx_D < D
    mask_MD = mask_M[:, None] & mask_D[None, :]

    q_ptrs = q_ptr + idx_M[:, None] * q_sM + idx_D[None, :]
    dout_ptrs = dout_ptr + idx_M[:, None] * dout_sM + idx_D[None, :]
    lse_ptrs = lse_ptr + idx_M 
    delta_ptrs = delta_ptr + idx_M



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
        
        tile_p = tl.math.exp2(scale_log2e * (tile_q @ tile_k.T) - tile_lse) 
        if CAUSAL:
            mask_p = mask_N[None, :] & (idx_M[:, None] >= idx_N[None, :])
            tile_p = tl.where(mask_p, tile_p, 0.0)
    
        tile_dp = tile_dout @ tile_v.T 
        tile_ds = tile_p * (tile_dp - tile_delta).to(tl.float32)
        tile_dq = tl.dot(tile_ds.to(tile_q.dtype), tile_k, tile_dq)
    
    tile_dq *= scale 
    dq_ptrs = dq_ptr + idx_M[:, None] * dq_sM + idx_D[None, :]

    tl.store(dq_ptrs, tile_dq.to(dq_ptr.dtype.element_ty), mask=mask_MD)    



def flash_attn_bwd(Q, K, V, O, dO, lse, is_causal=False):
    assert Q.shape == K.shape == V.shape == O.shape == dO.shape
    assert all(t.stride(-1) == 1 for t in (Q, K, V, O, dO)), "call maybe_contiguous first"

    B, H, M, D = Q.shape
    N = K.shape[2]

    BLK_SIZE_M = 128
    BLK_SIZE_N = 64
    BLK_SIZE_D = max(32, triton.next_power_of_2(D))

    delta = torch.empty((B, H, M), device=Q.device, dtype=torch.float32)
    dQ = torch.empty(Q.shape, device=Q.device, dtype=Q.dtype)
    dK = torch.empty(K.shape, device=K.device, dtype=K.dtype)
    dV = torch.empty(V.shape, device=V.device, dtype=V.dtype)

    scale = 1.0 / (D ** 0.5)
    scale_log2e = scale * math.log2(math.e)

    flash_attn_preprocess_kernel[(triton.cdiv(M, BLK_SIZE_M), H, B)](
        O, *O.stride()[:3],
        dO, *dO.stride()[:3],
        delta, *delta.stride()[:2],
        M, D,
        BLK_SIZE_M, BLK_SIZE_D
    )

    flash_attn_bwd_dkdv_kernel[(triton.cdiv(N, BLK_SIZE_N), H, B)](
        Q, *Q.stride()[:3],
        K, *K.stride()[:3],
        V, *V.stride()[:3],
        dO, *dO.stride()[:3],
        lse, *lse.stride()[:2],
        delta, *delta.stride()[:2],
        dK, *dK.stride()[:3],
        dV, *dV.stride()[:3],
        M, N, D,
        scale, scale_log2e,
        BLK_SIZE_M, BLK_SIZE_N, BLK_SIZE_D, is_causal,
        num_warps=8, num_stages=3
    )

    flash_attn_bwd_dq_kernel[(triton.cdiv(M, BLK_SIZE_M), H, B)](
        Q, *Q.stride()[:3],
        K, *K.stride()[:3],
        V, *V.stride()[:3],
        dO, *dO.stride()[:3],
        lse, *lse.stride()[:2],
        delta, *delta.stride()[:2],
        dQ, *dQ.stride()[:3],
        M, N, D,
        scale, scale_log2e,
        BLK_SIZE_M, BLK_SIZE_N, BLK_SIZE_D, is_causal,
        num_warps=8, num_stages=3
    )

    return dQ, dK, dV



class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        O, lse = flash_attn_fwd(Q, K, V, is_causal=is_causal, return_lse=True)
        ctx.save_for_backward(Q, K, V, O, lse)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, lse = ctx.saved_tensors
        dQ, dK, dV = flash_attn_bwd(
            Q, K, V, O, dO, lse, is_causal=ctx.is_causal
        )
        return dQ, dK, dV, None


flash_attention = FlashAttention.apply
