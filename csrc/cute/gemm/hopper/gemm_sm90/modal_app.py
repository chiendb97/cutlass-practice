"""Optional Modal support: a remote H100 runner for this kernel's benchmark.

Standalone, co-located copy (one per component) — running ``bench_*.py --modal`` imports it from
the same directory. Modal usage is fully optional; local benchmarking needs none of this. On
``--modal`` the bench calls :func:`run_remote`, which builds an image (NGC PyTorch base, so torch +
numpy + the matching CUDA toolkit are prebuilt) that copies CUTLASS and ``csrc`` as separate layers,
installs **only this component's** extension on a CPU builder (nvcc cross-compiles for sm_90a), and
re-runs that same benchmark script on an H100.

Credentials and any auth/connection errors are left to Modal itself (it reads ~/.modal.toml or the
MODAL_TOKEN_* env vars when the app launches). Modeled on gau-nernst/learn-cuda's Modal workflow,
adapted from B200/sm_100 to H100/sm_90a. Environment overrides: ``MODAL_GPU`` (default H100),
``MODAL_CUDA_IMAGE``. The ``modal`` package is imported lazily so importing this module doesn't
require it installed.
"""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    """Walk up from this file to the repo root (marked by 3rdparty/cutlass or .git). Keeps this
    file identical across component directories regardless of nesting depth."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "3rdparty" / "cutlass").exists() or (parent / ".git").exists():
            return parent
    return here.parents[-1]


REPO_ROOT = _repo_root()
# This modal_app.py is co-located in its kernel component's dir (next to that component's setup.py),
# so its parent IS the component dir. The image builds ONLY this component -- each bench/test loads
# just its own ``<name>_C`` extension, so compiling the sibling components would be wasted work.
COMPONENT_RELPATH = Path(__file__).resolve().parent.relative_to(REPO_ROOT).as_posix()
GPU = os.environ.get("MODAL_GPU", "H100")
# NGC PyTorch base: torch + numpy + the CUDA toolkit (nvcc) are prebuilt and version-matched, so we
# need no torch pin and no clang->g++ linker hack (NGC's Python is gcc-built, unlike Modal's
# add_python). nvcr.io/nvidia/pytorch is anonymously pullable -- no NGC key / Modal secret needed.
# Override via MODAL_CUDA_IMAGE (any recent NGC pytorch tag bundles a matching torch+CUDA+nvcc).
CUDA_IMAGE = os.environ.get("MODAL_CUDA_IMAGE", "nvcr.io/nvidia/pytorch:25.01-py3")


# --- remote runner -----------------------------------------------------------------------

def _build_app():
    """Construct the Modal image + app + remote entrypoint. Imports ``modal`` lazily."""
    import modal  # type: ignore[import-not-found]  # optional extra (pip install -e 'python/[modal]')

    image = (
        # NGC PyTorch image already ships torch + numpy + the matching CUDA toolkit, and uses its
        # own gcc-built Python -- so no add_python, no torch pin, and no clang->g++ linker hack.
        modal.Image.from_registry(CUDA_IMAGE)
        # Only the editable-build helpers; torch/numpy come from the base. (ninja/wheel/setuptools
        # are usually present in NGC too, but pinning them here keeps the editable install robust.)
        .pip_install("ninja", "wheel", "setuptools")
        # --- repo copy split by volatility, so a code edit doesn't re-copy CUTLASS ---------------
        # Layer A: the pinned CUTLASS submodule (~164 MB, ~never changes) -> cached across all edits.
        .add_local_dir(
            str(REPO_ROOT / "3rdparty" / "cutlass"),
            remote_path="/workspace/3rdparty/cutlass",
            copy=True,
            ignore=["**/.git", "**/__pycache__"],
        )
        # Layer B: the source you edit (~2.3 MB) -> only this layer (+ the build below) re-runs on a
        # code change. setup.py still finds the repo root via the /workspace/3rdparty/cutlass marker.
        .add_local_dir(
            str(REPO_ROOT / "csrc"),
            remote_path="/workspace/csrc",
            copy=True,
            ignore=["**/cmake-build*", "**/build", "**/*.so", "**/__pycache__", "**/*.nsys-rep"],
        )
        # Build ONLY this component's extension (its setup.py compiles just its own *_binding.cu);
        # each bench/test loads just this component's ``<name>_C``, so the siblings aren't compiled.
        .run_commands(
            f'pip install -e "/workspace/{COMPONENT_RELPATH}" --no-build-isolation'
        )
    )

    app = modal.App("cute-kernels-bench", image=image)

    # serialized=True: _remote is nested in _build_app (not module-global, because `modal` is
    # imported lazily), which modal's @app.function otherwise rejects -- serialize it by value.
    @app.function(gpu=GPU, timeout=1800, serialized=True)
    def _remote(bench_relpath: str, kwargs: dict) -> None:
        # Re-import the co-located bench script by its repo-relative path and run benchmark().
        import importlib.util

        path = f"/workspace/{bench_relpath}"
        spec = importlib.util.spec_from_file_location("_cute_bench", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        mod.benchmark(**kwargs)

    return app, _remote


def run_remote(bench_file: str, **kwargs) -> None:
    """Build the image (if needed) and run the benchmark script ``bench_file`` on a Modal H100.

    ``bench_file`` is a local path (the caller's ``__file__``); it is re-imported on the remote
    by its path relative to the repo root.
    """
    import modal  # type: ignore[import-not-found]  # optional extra (pip install -e 'python/[modal]')

    relpath = str(Path(bench_file).resolve().relative_to(REPO_ROOT))
    print(f"[modal] launching '{relpath}' on gpu={GPU} ...")
    app, remote = _build_app()
    with modal.enable_output(), app.run():
        remote.remote(relpath, kwargs)
