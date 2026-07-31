"""Detect Triton kernel device-loads after the engine starts serving.

Triton loads each kernel specialization's cubin onto the GPU at its first
launch (``CompiledKernel._init_handles`` -> ``cuModuleLoadData``). That load
needs free device memory *outside* the torch caching allocator. Engines size
their pools to leave little post-init headroom, and the allocator's high-water
mark consumes the rest during early serving — so a specialization first used
mid-serving (e.g. a new adaptive speculative draft length, or a rare batch-size
bucket) can die in ``cuModuleLoadData`` with CUDA OOM, minutes or hours in.

This module hooks ``triton.knobs.runtime.kernel_load_start_hook`` and, once
``mark_serving_started()`` has been called, logs a warning for every late
device-load (with the kernel name and free device memory). Set
``SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY=1`` to raise instead — for CI
recipes that assert full startup warmup coverage. The hook only fires on
first-use loads, so steady-state cost is zero.

Note: request-driven warmup (``--warmups``, the server warmup request) runs
*after* ``mark_serving_started()`` and is reported like any other late load;
crash mode is only meant for deployments whose kernels are fully pre-loaded
at engine init.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_serving_started = False
_prev_hook = None
_installed = False


def install() -> None:
    """Install the load hook (idempotent; chains any pre-existing hook)."""
    global _installed, _prev_hook
    if _installed:
        return
    try:
        import triton.knobs as knobs
    except ImportError:
        return
    _prev_hook = knobs.runtime.kernel_load_start_hook
    knobs.runtime.kernel_load_start_hook = _on_kernel_load
    _installed = True


def mark_serving_started() -> None:
    """Arm the watch: any Triton device-load from now on is reported."""
    global _serving_started
    _serving_started = True


def _on_kernel_load(module, function, name, metadata_group, hash) -> None:
    if _prev_hook is not None:
        _prev_hook(module, function, name, metadata_group, hash)
    if not _serving_started:
        return

    from sglang.srt.environ import envs

    free_mb = -1.0
    try:
        import torch

        if torch.cuda.is_available():
            free_mb = torch.cuda.mem_get_info()[0] / 1e6
    except Exception:
        pass
    msg = (
        f"Triton kernel '{name}' device-loaded after serving started "
        f"(free device mem: {free_mb:.0f} MB). A lazily-specialized kernel "
        f"was first used at serving time; the load can hit CUDA OOM once the "
        f"allocator has consumed the post-init headroom. Pre-load it during "
        f"engine init by warming up every specialization the workload can "
        f"reach (e.g. all speculative draft lengths)."
    )
    if envs.SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY.get():
        raise RuntimeError(msg)
    logger.warning(msg)
