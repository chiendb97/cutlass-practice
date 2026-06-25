# cutlass-practice

Hand-written **CUTLASS / CuTe** Hopper kernels — GEMM (basic TMA+WGMMA, multistage,
warp-specialized, and EVT) and a FlashAttention-style fused multi-head attention forward —
exposed to PyTorch for calling, correctness testing, and benchmarking, alongside a **CuTe DSL**
Python example. Built on the pinned `3rdparty/cutlass` submodule (v4.4.2).

## Build

Check out the CUTLASS headers first with `git submodule update --init --recursive`, then install
the component(s) you want with `pip install -e <component-dir>` (override the target arch with the
`CUTE_CUDA_ARCH` env var). Installing each component dir under `csrc/cute` builds every extension.

## Calling the ops

All ops support **float16 and bfloat16**. The GEMM ops (`gemm_sm90`, `gemm_multistage`, `gemm_ws`)
compute `a @ b.T` with `a=(M,K)`, `b=(N,K)` row-major; `gemm_evt` adds an `alpha*acc + beta*C`
epilogue. `fmha_forward` takes **contiguous BMHK** `(B, S, H, D)` q/k/v with head dim 64, 128, or
256. An extension has no `PyInit`, so callers don't `import` it — load its `.so` with
`torch.ops.load_library(...)` (resolve the path via `importlib.util.find_spec("<name>_C").origin`)
and call the op under `torch.ops.cute_kernels`.

## Test

Each kernel's test sits beside it; `pytest.ini` points collection at `csrc/`. Install a component
with its `[test]` extra, then run `pytest <component-dir>` to scope to it, or `pytest` for the whole
suite (needs all components installed).

## Benchmark

Each kernel's `bench_*.py` runs that kernel against a torch baseline and checks it against an fp32
reference; pass the matrix / attention dimensions as flags. Add `--modal` to run remotely on a Modal
H100 (install the component's `[modal]` extra and run `modal token new` first).

## CuTe DSL example

Install `nvidia-cutlass-dsl`, then from `python/` run the `cute_dsl.elementwise` module (axpy
`c = alpha*a + b`; runs on any CUDA GPU, verifies against torch).

## Editor / IntelliSense (clangd)

Generate the repo-root `.clangd` with `scripts/gen_clangd.py` — it injects the CUTLASS, `csrc/`, and
PyTorch include paths and filters nvcc-only flags clangd can't parse; re-run it after moving the
tree or changing the Python env. In VS Code, install the *clangd* extension (`.vscode/settings.json`
disables the conflicting default C/C++ IntelliSense); any other clangd LSP client picks `.clangd` up.
