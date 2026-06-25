# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning/practice repository for NVIDIA **CUTLASS** and its **CuTe** layout library, focused
on **Hopper (sm_90a)** GEMM and fused multi-head attention. The hand-written CUDA kernels are
exposed to PyTorch as per-component installable extensions for calling, testing, and benchmarking.
There are also **CuTe DSL** (Python) examples and optional **Modal** remote benchmarking. CUTLASS is the
pinned `3rdparty/cutlass` submodule (v4.4.2). There is **no CMake build** — everything builds via
the per-component `setup.py` files.

**GPU requirement:** the C++ kernels use TMA/WGMMA/cluster/warp-specialization and only *run*
on Hopper (H100/H200, sm_90a). They cross-compile on any host. The CuTe DSL
`elementwise` example runs on any CUDA GPU.

## Building

```bash
git submodule update --init --recursive   # CUTLASS is header-only but must be checked out
```

The kernel launch logic lives in a `.cuh` header that the binding `#include`s; kernels are built,
tested, and benchmarked **only through the PyTorch bindings** (no demo `main()` / ad-hoc `nvcc`).

**PyTorch bindings (the main deliverable):** each kernel component has its own `setup.py` building
its own extension; install the one you want, or loop over all.
```bash
pip install -e csrc/cute/gemm/hopper/gemm_sm90   # builds gemm_sm90_C for sm_90a; CUTE_CUDA_ARCH overrides
for d in $(find csrc/cute -name setup.py -exec dirname {} \;); do pip install -e "$d"; done  # all
pytest                                 # from repo root (pytest.ini -> testpaths=csrc); auto-skips off Hopper
python csrc/cute/gemm/hopper/gemm_sm90/bench_gemm_sm90.py --M 4096 --N 4096 --K 4096   # +--modal for H100
```

### Key build facts (non-obvious)

- **Kernels are sm_90a (Hopper) only.** Compilation cross-compiles anywhere; execution on a
  non-Hopper GPU fails with `cudaErrorNoKernelImageForDevice`. Each component `setup.py` forces
  `-gencode=arch=compute_90a,code=sm_90a -DCOMPILE_3X_HOPPER`; `CUTE_CUDA_ARCH` overrides the arch.
- **CUTLASS comes from the `3rdparty/cutlass` submodule** (each `setup.py` finds the repo root by
  walking up to the `3rdparty/cutlass` or `.git` marker). Run `git submodule update --init --recursive`.
- **The bindings use `TORCH_LIBRARY` (no pybind module). There is no `cute_kernels` Python
  package and no central build** — each component dir has its own `setup.py` that compiles only its
  `*_binding.cu` into its own extension (`gemm_sm90_C`, `gemm_multistage_C`, `gemm_ws_C`,
  `gemm_evt_C`, `fmha_C`). They all register into the `torch.ops.cute_kernels` namespace. Each
  co-located `test_*.py` / `bench_*.py` loads its `.so` via
  `torch.ops.load_library(importlib.util.find_spec("<name>_C").origin)` (a plain `import` fails with
  "no PyInit"), then calls `torch.ops.cute_kernels.<op>`.
- **No `-Wl,--allow-multiple-definition` needed** (it was, when all bindings shared one `.so`):
  with one binding TU per `.so`, the non-inline host functions from CUTLASS util headers
  (`helper_cuda.hpp`'s `device_init`, `GPU_Clock`) are defined exactly once per extension.
- **`.clangd` is generated** by `python scripts/gen_clangd.py` (CMake-free); it injects CUTLASS +
  `csrc/` + PyTorch include paths (torch paths via `torch.utils.cpp_extension.include_paths()`) and
  strips nvcc-only flags clangd can't parse. Gitignored; re-run the script after moving the tree or
  changing the Python env. `.vscode/settings.json` disables the default C/C++ IntelliSense engine.

## Code map

Everything for one kernel is **co-located** in its source dir and the component is independently
installable: `<name>.cuh` (kernel + launcher), `<name>_binding.cu` (PyTorch
wrapper), `setup.py` (builds this component's own `<name>_C` extension), `test_<name>.py`,
`bench_<name>.py`, and `modal_app.py`. All of `setup.py` / test / bench / modal are **deliberately
standalone** (each carries its own build flags / `.so` loader / inputs / reference / timing / skip /
Modal launcher) — there is **no** shared Python helper, **no** central build, and **no**
`cute_kernels` package; do not reintroduce any. `setup.py` and `modal_app.py` are the same template
copied per component (the only intentional duplication).

```
csrc/cute/dispatch.cuh    shared binding helpers (CUTE_DISPATCH_HALF_BF16, checks, stream)
csrc/cute/gemm/hopper/    gemm_sm90/{gemm_sm90{,_binding,test_,bench_} + setup.py(->gemm_sm90_C) + modal_app.py}; pipeline_gemm/{gemm_multistage/, gemm_warp_specialization/}; evt_gemm/
csrc/cute/fmha/hopper/    fmha_forward{,_binding,test_,bench_} + setup.py(->fmha_C) + modal_app.py (+ online_softmax.h, lib/gemm/{gemm,copy}_tensor.hpp)
python/cute_dsl/          CuTe DSL example (elementwise = axpy, runs on any GPU)
pytest.ini                testpaths=csrc, import-mode=importlib (root-level; collects co-located tests)
```

Conventions that matter when editing:

- **Each kernel family is wrapped in its own namespace** (`cute_gemm_sm90`, `cute_gemm_multistage`,
  `cute_gemm_evt`, `cute_fmha`). This is required: otherwise identically-named template
  instantiations (`gemm_tn<half_t,...>`) from different families collide at link time (ODR). The
  warp-spec launcher (`hopper_gemm_kernel_launch.h::gemm_tn`) has a distinct signature and is left
  at global scope.
- **fp16 + bf16 dispatch** happens at the binding boundary (`dispatch.cuh::CUTE_DISPATCH_HALF_BF16`).
  The hand-written GEMMs select the MMA atom via `GMMA::ss_op_selector<TA,TB,Accum,...>` — half
  accumulates in half, **bf16 must accumulate in fp32** (no bf16-accumulate GMMA atom exists). The
  warp-spec kernel sets `fp32_accum` per dtype and uses a smaller 12-warp tile for bf16 (the
  20-warp fp16 tile overflows the register file under fp32 accumulation).
- The kernels predate CUTLASS v4: porting to the pinned v4.4.2 submodule required API fixes
  (`make_counting_tensor` → `make_coord_tensor`; gemm_sm90's `axpby` epilogue replaced with an
  explicit float-compute-then-convert loop so bf16 output works).
- Binding GEMMs compute `a @ b.T` for `a=(M,K)`, `b=(N,K)` row-major. Output stride differs by
  kernel: gemm_sm90 & evt write **column-major** C; multistage & warp-spec write **row-major** —
  the binding wrappers handle this via a `(N,M)` buffer + transpose where needed.
- The **fmha_forward op is BMHK** — q/k/v/o are `(B, S, H, D)` **contiguous** (heads interleaved
  within each sequence position), and the binding does **no transpose** (zero-copy); callers that
  hold BHSD must convert themselves. Head dim `D` is a compile-time template, so it must be
  **64, 128, or 256**. The binding multiplies `scale` by `log2(e)` before the device call (the
  online softmax uses `exp2`); `scale` defaults to `1/sqrt(D)`.

When adding a kernel, everything stays in the kernel's own directory. Put the kernel + namespaced
launcher in a `.cuh`, then add these siblings next to it:
- `<name>_binding.cu` — `TORCH_LIBRARY_FRAGMENT(cute_kernels, ...)`, `#include "dispatch.cuh"` +
  exactly one family header.
- `setup.py` — copy a sibling's verbatim and edit only the `# --- this component ---` block:
  `EXT_NAME = "<name>_C"`, `SOURCES = ["<name>_binding.cu"]`, optional `EXTRA_INCLUDE_DIRS` /
  `LIBRARIES` (e.g. evt sets `LIBRARIES=["cublas"]`, fmha sets `EXTRA_INCLUDE_DIRS=["lib/gemm"]`,
  warp-spec sets `EXTRA_INCLUDE_DIRS=["kernel"]` for its impl headers grouped under `kernel/`).
  It self-locates the repo root, so it's depth-agnostic. Build with `pip install -e <component dir>`.
- `test_<name>.py` — **standalone**: load the ext (`torch.ops.load_library(importlib.util.find_spec
  ("<name>_C").origin)`), then define its own `requires_hopper` skip (capability major==9), inputs,
  fp32 reference, and `torch.testing.assert_close` tolerance, calling `torch.ops.cute_kernels.<op>`.
  Import only `torch`, `pytest`, stdlib — no shared helper exists; copy the ~12-line pattern from a
  sibling `test_*.py`.
- `bench_<name>.py` — **standalone**: define `benchmark(**kwargs)` (Modal re-imports it by path and
  calls it) which loads the ext, builds inputs/reference, and times via a local `row()` printer; a
  `__main__` with argparse. On `--modal` it does `from modal_app import check_modal_auth, run_remote`
  (the co-located copy; running the script puts its dir on `sys.path`).
- `modal_app.py` — copy a sibling's verbatim (it self-locates the repo root, so it's depth-agnostic).

Each `setup.py` compiles only its own declared `SOURCES` (just the `*_binding.cu`); the kernel and
its launcher live in the `.cuh` that binding includes. There is no CMake build and no demo
`main()` — kernels are exercised only through the PyTorch bindings. After adding a component,
`python scripts/gen_clangd.py` refreshes IntelliSense
include paths (the script lists the component dirs explicitly — add the new one there).
