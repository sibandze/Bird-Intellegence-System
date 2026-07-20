# src/utils/memory_utils.py
"""
Lightweight GPU memory profiling helpers.
"""

from __future__ import annotations

import gc
from typing import Optional

import torch


def get_gpu_memory_info(device: Optional[torch.device] = None) -> dict:
    """Return current GPU memory statistics in MB."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    if device is None:
        device = torch.device("cuda:0")

    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "total_mb": torch.cuda.get_device_properties(device).total_memory / 1024**2,
    }


def log_memory_usage(prefix: str = "", device: Optional[torch.device] = None) -> None:
    """Print current GPU memory usage."""
    info = get_gpu_memory_info(device)
    if "error" in info:
        return
    print(
        f"{prefix} GPU Memory – "
        f"Allocated: {info['allocated_mb']:.0f} MB | "
        f"Reserved:  {info['reserved_mb']:.0f} MB | "
        f"Max:       {info['max_allocated_mb']:.0f} MB | "
        f"Total:     {info['total_mb']:.0f} MB"
    )


def clear_gpu_memory() -> None:
    """Release all unoccupied cached memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
