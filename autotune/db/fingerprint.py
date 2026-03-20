"""Generate a stable hardware fingerprint ID for a HardwareProfile."""

from __future__ import annotations

import hashlib

from autotune.hardware.profiler import HardwareProfile


def hardware_id(hw: HardwareProfile) -> str:
    """SHA-256 of stable hardware attributes → 16-char hex prefix."""
    parts = [
        hw.os_name,
        hw.cpu.brand,
        str(hw.cpu.physical_cores),
        f"{hw.memory.total_gb:.1f}",
        hw.gpu.name if hw.gpu else "none",
        hw.gpu.backend if hw.gpu else "none",
        f"{hw.gpu.vram_gb:.1f}" if (hw.gpu and hw.gpu.vram_gb) else "unified",
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return digest[:16]


def hardware_to_db_dict(hw: HardwareProfile) -> dict:
    return {
        "id": hardware_id(hw),
        "os_name": hw.os_name,
        "os_version": hw.os_version,
        "cpu_brand": hw.cpu.brand,
        "cpu_physical_cores": hw.cpu.physical_cores,
        "cpu_logical_cores": hw.cpu.logical_cores,
        "cpu_arch": hw.cpu.architecture,
        "total_ram_gb": round(hw.memory.total_gb, 2),
        "gpu_name": hw.gpu.name if hw.gpu else None,
        "gpu_backend": hw.gpu.backend if hw.gpu else "none",
        "gpu_vram_gb": hw.gpu.vram_gb if hw.gpu else None,
        "is_unified_memory": int(hw.gpu.is_unified_memory) if hw.gpu else 0,
    }
