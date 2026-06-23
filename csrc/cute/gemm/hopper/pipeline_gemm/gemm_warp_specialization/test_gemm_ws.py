"""Standalone correctness test for the warp-specialized GEMM (torch.ops.cute_kernels.gemm_ws).

Self-contained: loads the compiled extension itself, and carries its own Hopper skip, inputs,
fp32 reference, and tolerance (no shared helpers). 512x512x384 is a multiple of every tile across
the fp16/bf16 configs. Skipped off Hopper.
"""

import importlib.util

import pytest
import torch

# The extension registers ops via TORCH_LIBRARY (no PyInit), so locate the .so and dlopen it
# rather than importing it. find_spec resolves this component's gemm_ws_C built by its setup.py.
_spec = importlib.util.find_spec("gemm_ws_C")
assert _spec and _spec.origin, "gemm_ws_C not built; run: pip install -e csrc/cute/gemm/hopper/pipeline_gemm/gemm_warp_specialization"
torch.ops.load_library(_spec.origin)

requires_hopper = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9),
    reason="CuTe Hopper kernels require an sm_90 (H100/H200) GPU",
)
pytestmark = requires_hopper

M, N, K = 512, 512, 384


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_ws(dtype):
    s = K ** -0.25  # scale so A@B^T is ~O(1), bounding fp16/half-accumulate error
    a = torch.randn(M, K, device="cuda", dtype=dtype) * s
    b = torch.randn(N, K, device="cuda", dtype=dtype) * s
    out = torch.ops.cute_kernels.gemm_ws(a, b)
    assert out.shape == (M, N)
    assert out.dtype == dtype
    ref = a.float() @ b.float().t()
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=1e-1)
