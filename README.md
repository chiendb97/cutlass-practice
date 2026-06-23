# cutlass-practice

Hand-written **CUTLASS / CuTe** Hopper kernels (GEMM + fused multi-head attention), exposed
to PyTorch for calling, correctness testing, and benchmarking, alongside **CuTe DSL** Python
examples. Built on the pinned `3rdparty/cutlass` submodule (v4.4.2).

> **GPU requirement:** the C++ kernels are **sm_90a (Hopper, H100/H200)** — TMA, WGMMA,
> cluster launch, warp specialization. They *cross-compile* anywhere but only *run* on Hopper.
> The CuTe DSL `elementwise` example runs on any CUDA GPU; remote benchmarking on
> Modal targets H100.

## Layout

Everything for one kernel lives in its own directory — the CUDA kernel, the PyTorch binding,
the build spec, the test, and the benchmark are **co-located**:

```
csrc/cute/
  dispatch.cuh                         shared torch::Tensor dispatch helpers (fp16/bf16, ...)
  gemm/hopper/                           each GEMM kernel is its own self-contained, installable component dir:
    gemm_sm90/                             basic TMA+WGMMA GEMM; every component dir holds:
      gemm_sm90.{cu,cuh}                     kernel (+ thin main) / launcher header
      gemm_sm90_binding.cu                   PyTorch wrapper
      setup.py                               builds this component's own extension: gemm_sm90_C
      test_gemm_sm90.py                      standalone test  (loads gemm_sm90_C, torch.ops.cute_kernels.*)
      bench_gemm_sm90.py                     standalone bench (loads gemm_sm90_C, torch.ops.cute_kernels.*)
      modal_app.py                           standalone Modal auth + H100 runner (used by --modal)
    pipeline_gemm/{gemm_multistage/, gemm_warp_specialization/}   multistage + warp-spec (same layout each)
    evt_gemm/                              EVT (alpha*acc + beta*C)
  fmha/hopper/                         FlashAttention-style forward (+ setup/test/bench/modal_app)
python/
  cute_dsl/              CuTe DSL example (elementwise — axpy, runs on any GPU)
```

There is **no `cute_kernels` Python package and no central build**. Each component dir has its own
`setup.py` that compiles just its `*_binding.cu` into its own top-level extension module
(`gemm_sm90_C`, `gemm_multistage_C`, `gemm_ws_C`, `gemm_evt_C`, `fmha_C`). They all register their
ops into the same `torch.ops.cute_kernels` namespace (via `TORCH_LIBRARY_FRAGMENT`). An extension
has no `PyInit`, so callers don't `import` it — they locate its `.so` with
`importlib.util.find_spec("<name>_C").origin` and `torch.ops.load_library(...)`, then call
`torch.ops.cute_kernels.<op>`. (One binding per `.so` also means no `-Wl,--allow-multiple-definition`.)

Each kernel family is wrapped in its own C++ namespace (`cute_gemm_sm90`, `cute_gemm_multistage`,
`cute_gemm_evt`, `cute_fmha`); the launch logic lives in `*.cuh` headers and the original `*.cu`
files keep a thin `main()` (a self-verifying demo you can compile ad hoc with `nvcc` for a
GFLOP/s sanity check on a Hopper host).

**Every `setup.py`, `test_*.py`, `bench_*.py`, and `modal_app.py` is fully standalone** — co-located
in the kernel's directory, carrying its own build flags / `.so` loader / Hopper skip / inputs / fp32
reference / timing. There is no shared Python helper module; `setup.py` and `modal_app.py` are the
same template copied per component (the only intentional duplication).

```bash
git submodule update --init --recursive    # CUTLASS headers (required to build any binding)
```

## PyTorch bindings

Install the component(s) you want — each builds its own extension (override arch with
`CUTE_CUDA_ARCH`):

```bash
pip install -e csrc/cute/gemm/hopper/gemm_sm90  # builds gemm_sm90_C
# ...or build them all:
for d in $(find csrc/cute -name setup.py -exec dirname {} \;); do pip install -e "$d"; done
```

```python
import importlib.util, torch

# Each extension has no PyInit, so find its .so and dlopen it (don't `import` it).
torch.ops.load_library(importlib.util.find_spec("gemm_sm90_C").origin)
torch.ops.load_library(importlib.util.find_spec("gemm_ws_C").origin)
torch.ops.load_library(importlib.util.find_spec("gemm_evt_C").origin)
torch.ops.load_library(importlib.util.find_spec("fmha_C").origin)

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
c = torch.ops.cute_kernels.gemm_ws(a, b)          # a @ b.T  (also: gemm_sm90, gemm_multistage)
d = torch.ops.cute_kernels.gemm_evt(a, b, torch.zeros(4096, 4096, device="cuda"), 1.0, 1.0)
q = k = v = torch.randn(2, 8, 512, 128, device="cuda", dtype=torch.bfloat16)
o = torch.ops.cute_kernels.fmha_forward(q, k, v, None)  # head dim in {64,128,256}
```

All ops support **float16 and bfloat16** inputs. GEMM ops compute `a @ b.T` with `a=(M,K)`,
`b=(N,K)` row-major. (The co-located `test_*.py` / `bench_*.py` use exactly this load pattern.)

## Tests

Each kernel's `test_*.py` sits beside it under `csrc/`; `pytest.ini` points collection at `csrc/`.
A test needs its component's extension built (`pip install -e <component>`, with the `[test]`
extra for pytest); the full `pytest` run needs all components installed.

```bash
pip install -e "csrc/cute/fmha/hopper[test]"   # build one component + pytest
pytest csrc/cute/fmha/hopper                    # scope to that component (auto-skips off sm_90)
pytest                                          # whole suite (needs all components installed)
```

## CuTe DSL example

```bash
pip install nvidia-cutlass-dsl
cd python
python -m cute_dsl.elementwise --M 1024 --N 1024     # axpy c = alpha*a + b; runs on any GPU, verifies vs torch
```

## Benchmarks

Each kernel's `bench_*.py` sits beside it; run the script for the kernel you care about (it
benchmarks that kernel against a torch baseline and checks it against an fp32 reference).

```bash
python csrc/cute/gemm/hopper/gemm_sm90/bench_gemm_sm90.py --M 4096 --N 4096 --K 4096 --dtype fp16
python csrc/cute/gemm/hopper/pipeline_gemm/gemm_multistage/bench_gemm_multistage.py --M 4096 --N 4096 --K 4096
python csrc/cute/gemm/hopper/pipeline_gemm/gemm_warp_specialization/bench_gemm_ws.py
python csrc/cute/gemm/hopper/evt_gemm/bench_gemm_evt.py
python csrc/cute/fmha/hopper/bench_fmha.py --B 4 --H 8 --S 512 --D 128

# Optional: run remotely on a Modal H100 (verifies the token against api.modal.com first)
pip install -e "csrc/cute/gemm/hopper/gemm_sm90[modal]" && modal token new
python csrc/cute/gemm/hopper/gemm_sm90/bench_gemm_sm90.py --M 8192 --N 8192 --K 8192 --modal
```

## Editor / IntelliSense (clangd)

Generate the repo-root `.clangd` (injects the CUTLASS, `csrc/`, and PyTorch include paths plus
filters for nvcc-only flags clangd can't parse):

```bash
python scripts/gen_clangd.py        # re-run after moving the tree or changing the Python env
```

- **VS Code:** install the *clangd* extension (`.vscode/settings.json` disables the conflicting
  default C/C++ IntelliSense).
- **Neovim / other:** `.clangd` is editor-agnostic; any clangd LSP client picks it up.
