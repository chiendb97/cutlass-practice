"""Standalone correctness test for the CuTe Hopper FMHA (torch.ops.cute_kernels.fmha_forward) vs. SDPA.

Self-contained: loads the compiled extension itself, and carries its own Hopper skip, inputs,
fp32 reference, and tolerance (no shared helpers). Head dim must be 64/128/256 (compile-time
template). Skipped off Hopper.
"""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

# The extension registers ops via TORCH_LIBRARY (no PyInit), so locate the .so and dlopen it
# rather than importing it. find_spec resolves this component's fmha_C built by its setup.py.
_spec = importlib.util.find_spec("fmha_C")
assert _spec and _spec.origin, "fmha_C not built; run: pip install -e csrc/cute/fmha/hopper"
torch.ops.load_library(_spec.origin)

requires_hopper = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9),
    reason="CuTe Hopper kernels require an sm_90 (H100/H200) GPU",
)
pytestmark = requires_hopper

B, H, S = 2, 4, 512
HEAD_DIMS = [64, 128]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("D", HEAD_DIMS)
def test_fmha_matches_sdpa(D, dtype):
    q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    scale = 1.0 / (D ** 0.5)

    out = torch.ops.cute_kernels.fmha_forward(q, k, v, scale)
    assert out.shape == (B, H, S, D)
    assert out.dtype == dtype
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
