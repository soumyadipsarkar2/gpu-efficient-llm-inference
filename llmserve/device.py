"""Device abstraction: pick an accelerator and read its memory counters.

The framework targets CUDA GPUs but also runs on Apple-silicon GPUs (MPS) and on
CPU, so every GPU-specific call goes through this module.
"""

from __future__ import annotations

import os

import torch


def pick_device(requested: str | None = None) -> torch.device:
    """Return the requested device, or the best one available."""
    req = (requested or os.environ.get("LLMSERVE_DEVICE") or "auto").lower()
    if req != "auto":
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def memory_allocated(device: torch.device) -> int:
    """Bytes currently held by live tensors on the device."""
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    return 0


def memory_reserved(device: torch.device) -> int:
    """Bytes the framework's allocator has obtained from the driver."""
    if device.type == "cuda":
        return int(torch.cuda.memory_reserved(device))
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    return 0


def peak_memory(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None  # MPS exposes no peak counter


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def gpu_utilization(device: torch.device) -> float | None:
    """Hardware utilization in percent where the driver exposes it (NVML)."""
    if device.type != "cuda":
        return None
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
        return float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
    except Exception:
        return None


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple silicon GPU (MPS)"
    return "CPU"
