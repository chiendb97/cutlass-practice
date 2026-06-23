"""Standalone correctness test for the EVT GEMM: D = alpha*(A@B^T) + beta*C (float32).

Self-contained: loads the compiled extension itself, and carries its own Hopper skip, inputs,
fp32 reference, and tolerance (no shared helpers). Skipped off Hopper.
"""

import importlib.util

import pytest
import torch

# The extension registers ops via TORCH_LIBRARY (no PyInit), so locate the .so and dlopen it
# rather than importing it. find_spec resolves this component's gemm_evt_C built by its setup.py.
_spec = importlib.util.find_spec("gemm_evt_C")
assert _spec and _spec.origin, "gemm_evt_C not built; run: pip install -e csrc/cute/gemm/hopper/evt_gemm"
torch.ops.load_library(_spec.origin)

requires_hopper = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9),
    reason="CuTe Hopper kernels require an sm_90 (H100/H200) GPU",
)
pytestmark = requires_hopper

M, N, K = 512, 512, 384


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_evt(dtype):
    s = K ** -0.25
    a = torch.randn(M, K, device="cuda", dtype=dtype) * s
    b = torch.randn(N, K, device="cuda", dtype=dtype) * s
    c = torch.randn(M, N, device="cuda", dtype=torch.float32)
    alpha, beta = 2.0, 0.5
    out = torch.ops.cute_kernels.gemm_evt(a, b, c, alpha, beta)
    assert out.shape == (M, N)
    assert out.dtype == torch.float32
    ref = alpha * (a.float() @ b.float().t()) + beta * c
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=1e-1)
