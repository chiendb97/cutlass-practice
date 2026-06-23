"""Optional Modal support: a remote H100 runner for this kernel's benchmark.

Standalone, co-located copy (one per component) — running ``bench_*.py --modal`` imports it from
the same directory. Modal usage is fully optional; local benchmarking needs none of this. On
``--modal`` the bench calls :func:`run_remote`, which builds an image that copies the repo (incl.
the CUTLASS submodule), installs the extension on a CPU builder (nvcc cross-compiles for sm_90a),
and re-runs that same benchmark script on an H100.

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
GPU = os.environ.get("MODAL_GPU", "H100")
CUDA_IMAGE = os.environ.get("MODAL_CUDA_IMAGE", "nvidia/cuda:12.6.2-devel-ubuntu22.04")


# --- remote runner -----------------------------------------------------------------------

def _build_app():
    """Construct the Modal image + app + remote entrypoint. Imports ``modal`` lazily."""
    import modal  # type: ignore[import-not-found]  # optional extra (pip install -e 'python/[modal]')

    image = (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
        .pip_install("torch", "numpy", "ninja")
        .add_local_dir(
            str(REPO_ROOT),
            remote_path="/workspace",
            copy=True,
            ignore=[
                "**/.git", "**/cmake-build*", "**/build", "**/*.so",
                "**/__pycache__", "**/*.nsys-rep",
            ],
        )
        # Each kernel component has its own setup.py; build them all so any bench can run.
        .run_commands(
            "for d in $(find /workspace/csrc/cute -name setup.py -exec dirname {} \\;); do "
            'pip install -e "$d" --no-build-isolation || exit 1; done'
        )
    )

    app = modal.App("cute-kernels-bench", image=image)

    @app.function(gpu=GPU, timeout=1800)
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
