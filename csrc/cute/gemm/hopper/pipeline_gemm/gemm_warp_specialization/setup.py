"""Standalone build for the gemm_ws_C extension (this kernel's binding only).

    pip install -e csrc/cute/gemm/hopper/pipeline_gemm/gemm_warp_specialization   # builds gemm_ws_C

Self-contained and co-located with the kernel (no shared build helper). Compiles this component's
``*_binding.cu`` into a top-level extension module whose op registers into the
``torch.ops.cute_kernels`` namespace (via TORCH_LIBRARY_FRAGMENT). The sibling ``test_*.py`` /
``bench_*.py`` load the ``.so`` with ``importlib.util.find_spec("gemm_ws_C").origin`` +
``torch.ops.load_library``. A Hopper GPU (H100/H200) is required to *run* it; it cross-compiles
fine elsewhere. Override the arch with ``CUTE_CUDA_ARCH``.
"""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# --- this component -----------------------------------------------------------------------
EXT_NAME = "gemm_ws_C"
SOURCES = ["gemm_ws_binding.cu"]
EXTRA_INCLUDE_DIRS: list[str] = ["kernel"]   # kernel impl headers grouped under kernel/ (bare-name includes)
LIBRARIES: list[str] = []            # extra link libraries
# ------------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent


def _repo_root() -> Path:
    """Walk up to the repo root (marked by 3rdparty/cutlass or .git)."""
    for p in HERE.parents:
        if (p / "3rdparty" / "cutlass").exists() or (p / ".git").exists():
            return p
    return HERE.parents[-1]


ROOT = _repo_root()
CUTLASS = ROOT / "3rdparty" / "cutlass"
if not (CUTLASS / "include" / "cutlass" / "cutlass.h").exists():
    raise RuntimeError(
        f"CUTLASS headers not found under {CUTLASS}. Run: git submodule update --init --recursive"
    )

CUDA_ARCH = os.environ.get("CUTE_CUDA_ARCH", "90a")

include_dirs = [
    str(CUTLASS / "include"),
    str(CUTLASS / "tools" / "util" / "include"),
    str(CUTLASS / "examples" / "common"),
    str(ROOT / "csrc" / "cute"),  # shared dispatch.cuh
    str(HERE),                    # sibling kernel headers
] + [str(HERE / d) for d in EXTRA_INCLUDE_DIRS]

nvcc_flags = [
    "-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
    "-DCOMPILE_3X_HOPPER",
    f"-gencode=arch=compute_{CUDA_ARCH},code=sm_{CUDA_ARCH}",
    # cutlass <-> torch half/bfloat16 interop
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

setup(
    name=EXT_NAME,
    version="0.0.1",
    description=f"CuTe Hopper kernel binding: {EXT_NAME}",
    ext_modules=[
        CUDAExtension(
            name=EXT_NAME,
            sources=[str(HERE / s) for s in SOURCES],
            include_dirs=include_dirs,
            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": nvcc_flags},
            libraries=LIBRARIES,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch>=2.1"],
    extras_require={"test": ["pytest"], "modal": ["modal"]},
)
