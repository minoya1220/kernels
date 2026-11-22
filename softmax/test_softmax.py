import torch
from softmax_fwd import softmax_fwd_builder
from softmax_bwd import softmax_bwd_builder
import pytest

TEST_M = 2**6 - 1

TEST_DIMS = [
    3, 7, 8, 17, 64, 100, 257, 511, 633, 1021, 
    2049, 4099, 128, 256, 512, 1024, 2048, 4096, 
    8192, 16384, 32768, 65536, 131072
]


DTYPES = (torch.bfloat16, torch.half, torch.float, torch.double)

@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(4)

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", TEST_DIMS)
def test_fwd(N, dtype):
    X = torch.randn(TEST_M, N, device='cuda', dtype=dtype)

    torch_softmax = torch.softmax(X.double(), dim=1).to(X.dtype)
    softmax_fwd = softmax_fwd_builder(X)
    
    torch.testing.assert_close(softmax_fwd(X), torch_softmax)

 
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("N", TEST_DIMS)
def test_bwd(N, dtype):
    X = torch.randn(TEST_M, N, device='cuda', dtype=dtype)
    dY = torch.randn_like(X)
    
    Y = torch.softmax(X, dim=1).to(dtype)
    Y_double, dY_double = Y.double(), dY.double()

    ref_softmax_bwd = (Y_double * (dY_double - (Y_double * dY_double).sum(dim=1, keepdim=True))).to(dtype)
    softmax_bwd = softmax_bwd_builder(Y, dY)
    
    torch.testing.assert_close(softmax_bwd(Y, dY), ref_softmax_bwd)
