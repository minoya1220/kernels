import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import pytest
from triton_fa2 import flash_attn_fwd, flash_attn_bwd


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("N", [13, 128, 654, 2048])
@pytest.mark.parametrize("D", [71, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_fwd(N, D, causal, dtype):
    B, H = 4, 4
    
    Q = torch.randn(B, H, N, D, dtype=dtype, device="cuda") 
    K = torch.randn_like(Q) 
    V = torch.randn_like(Q)
    
    expected = F.scaled_dot_product_attention(Q.float(), K.float(), V.float(), is_causal=causal)
    actual = flash_attn_fwd(Q, K, V, is_causal=causal)
    torch.testing.assert_close(actual.to(torch.float32), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("N", [13, 128, 654, 2048])
@pytest.mark.parametrize("D", [71, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_bwd(N, D, causal, dtype):
    B, H = 4, 4
    Q = torch.randn(B, H, N, D, dtype=dtype, device="cuda", requires_grad=True) 
    K = torch.randn_like(Q).requires_grad_()
    V = torch.randn_like(Q).requires_grad_()
    dOut = torch.randn_like(Q)
    
    
    Qf = Q.detach().float().requires_grad_()
    Kf = K.detach().float().requires_grad_()
    Vf = V.detach().float().requires_grad_()
    F.scaled_dot_product_attention(Qf, Kf, Vf, is_causal=causal).backward(dOut.float())
    expected = (Qf.grad, Kf.grad, Vf.grad)

    with torch.no_grad():
        O, lse = flash_attn_fwd(Q, K, V, is_causal=causal, return_lse=True)
        actual = flash_attn_bwd(Q, K, V, O, dOut, lse, is_causal=causal)
    
    torch.testing.assert_close(actual, expected, check_dtype=False)
