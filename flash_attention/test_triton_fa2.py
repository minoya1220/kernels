import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import pytest
from triton_fa2 import flash_attention

TOLERANCES = {"atol": 2e-2, "rtol": 1e-2}

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("N", [13, 128, 131, 654, 2048])
@pytest.mark.parametrize("D", [17, 71, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_fwd(N, D, causal, dtype):
    B, H = 2, 3
    
    Q = torch.randn(B, H, N, D, dtype=dtype, device="cuda") 
    K = torch.randn_like(Q) 
    V = torch.randn_like(Q)
    
    expected = F.scaled_dot_product_attention(Q.double(), K.double(), V.double(), is_causal=causal)
    actual = flash_attention(Q, K, V, causal=causal)
    
    torch.testing.assert_close(actual, expected, **TOLERANCES,check_dtype=False)
    


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("N", [13, 128, 131, 654, 2048])
@pytest.mark.parametrize("D", [17, 71, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("deterministic", [True, False])
def test_bwd(N, D, causal, dtype, deterministic):
    B, H = 2, 3
    
    Q = torch.randn(B, N, H, D, dtype=dtype, device="cuda").transpose(-2, -3).requires_grad_()
    K = torch.randn_like(Q).requires_grad_()
    V = torch.randn_like(Q).requires_grad_()
    dOut = torch.randn_like(Q)

    if deterministic:
        Q2 = Q.clone().detach().requires_grad_()
        K2 = K.clone().detach().requires_grad_()
        V2 = V.clone().detach().requires_grad_()
    
    

    Qd = Q.detach().double().requires_grad_()
    Kd = K.detach().double().requires_grad_()
    Vd = V.detach().double().requires_grad_()
    F.scaled_dot_product_attention(Qd, Kd, Vd, is_causal=causal).backward(dOut.double())
    expected = (Qd.grad, Kd.grad, Vd.grad)

    default = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)

        out = flash_attention(Q, K, V, causal=causal)
        out.backward(dOut)
        actual = (Q.grad, K.grad, V.grad)
        assert dtype == Q.grad.dtype == K.grad.dtype == V.grad.dtype

        if deterministic:
            out2 = flash_attention(Q2, K2, V2, causal=causal)
            out2.backward(dOut)
            actual2 = (Q2.grad, K2.grad, V2.grad)
    finally:
        torch.use_deterministic_algorithms(default)
    
    
        
    if deterministic: # verify determinism
        assert all(torch.equal(a, b) for a, b in zip(actual, actual2)) 
    
    torch.testing.assert_close(actual, expected, **TOLERANCES, check_dtype=False)
