import torch
from fa2_bwd import flash_attn_bwd
from fa2_fwd import flash_attn_fwd


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal):
        O, lse = flash_attn_fwd(Q, K, V, causal=causal)
        ctx.save_for_backward(Q, K, V, O, lse)
        ctx.causal = causal
        return O


    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, lse = ctx.saved_tensors
        if dO.stride(-1) != 1:
            dO = dO.contiguous()
        dQ, dK, dV = flash_attn_bwd(
            Q, K, V, O, dO, lse, causal=ctx.causal, deterministic=torch.are_deterministic_algorithms_enabled()
        )
        return dQ, dK, dV, None


def flash_attention(Q, K, V, causal=False):
    return FlashAttention.apply(Q, K, V, causal)

