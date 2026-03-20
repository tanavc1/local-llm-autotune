"""Hardware profiler: detects OS, CPU, RAM, and GPU capabilities."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CPUInfo:
    brand: str
    architecture: str
    physical_cores: int
    logical_cores: int
    freq_max_mhz: float


@dataclass
class MemoryInfo:
    total_gb: float
    available_gb: float
    used_gb: float


@dataclass
class GPUInfo:
    name: str
    backend: str                  # "cuda" | "metal" | "rocm" | "vulkan" | "none"
    vram_gb: Optional[float]      # None when unified memory
    is_unified_memory: bool       # True for Apple Silicon
    driver_version: Optional[str] = None
    compute_capability: Optional[str] = None  # CUDA only


@dataclass
class HardwareProfile:
    os_name: str
    os_version: str
    cpu: CPUInfo
    memory: MemoryInfo
    gpu: Optional[GPUInfo]
    inference_mode: str  # "gpu" | "cpu"

    # ------------------------------------------------------------------ #
    # Derived helpers                                                      #
    # ------------------------------------------------------------------ #

    @property
    def effective_memory_gb(self) -> float:
        """Memory actually usable for model + KV cache at runtime."""
        if self.gpu is None:
            return self.memory.available_gb
        if self.gpu.is_unified_memory:
            # Apple Silicon: entire RAM is addressable by the GPU
            return self.memory.available_gb
        if self.gpu.vram_gb is not None:
            return self.gpu.vram_gb
        return self.memory.available_gb

    @property
    def has_gpu(self) -> bool:
        return self.gpu is not None and self.gpu.backend != "none"


# ---------------------------------------------------------------------------
# Internal detection helpers
# ---------------------------------------------------------------------------

def _cpu_info() -> CPUInfo:
    """Gather CPU info using py-cpuinfo with psutil fallback."""
    brand = "Unknown CPU"
    arch = platform.machine()
    freq_max = 0.0

    try:
        import cpuinfo  # type: ignore[import-untyped]
        info = cpuinfo.get_cpu_info()
        brand = info.get("brand_raw", brand)
        arch = info.get("arch", arch)
        hz_advertised = info.get("hz_advertised_friendly", "")
        if hz_advertised:
            # e.g. "3.2000 GHz" → 3200 MHz
            try:
                val, unit = hz_advertised.split()
                val_f = float(val)
                freq_max = val_f * 1000 if "GHz" in unit else val_f
            except ValueError:
                pass
    except Exception:
        # Minimal fallback via platform
        brand = platform.processor() or arch

    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or 1
        logical = psutil.cpu_count(logical=True) or physical
    except Exception:
        physical = 1
        logical = 1

    try:
        import psutil
        freq = psutil.cpu_freq()
        if freq and freq.max:
            freq_max = freq.max
    except Exception:
        pass  # cpu_freq() unavailable on some platforms (e.g. Apple Silicon)

    return CPUInfo(
        brand=brand,
        architecture=arch,
        physical_cores=physical,
        logical_cores=logical,
        freq_max_mhz=freq_max,
    )


def _memory_info() -> MemoryInfo:
    import psutil

    vm = psutil.virtual_memory()
    return MemoryInfo(
        total_gb=vm.total / 1024**3,
        available_gb=vm.available / 1024**3,
        used_gb=vm.used / 1024**3,
    )


def _detect_nvidia() -> Optional[GPUInfo]:
    """Query nvidia-smi for CUDA GPU info."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode()
        line = out.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        name = parts[0]
        vram_mb = float(parts[1]) if len(parts) > 1 else 0.0
        driver = parts[2] if len(parts) > 2 else None
        compute = parts[3] if len(parts) > 3 else None
        return GPUInfo(
            name=name,
            backend="cuda",
            vram_gb=vram_mb / 1024,
            is_unified_memory=False,
            driver_version=driver,
            compute_capability=compute,
        )
    except Exception:
        return None


def _detect_apple_silicon() -> Optional[GPUInfo]:
    """Detect Apple Silicon (M-series) unified memory GPU."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None

    chip_name = "Apple Silicon"
    try:
        result = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode().strip()
        if result:
            chip_name = result
    except Exception:
        pass

    # On Apple Silicon total RAM == GPU addressable memory
    try:
        import psutil
        total_gb = psutil.virtual_memory().total / 1024**3
    except Exception:
        total_gb = None

    return GPUInfo(
        name=chip_name,
        backend="metal",
        vram_gb=None,          # unified — use system RAM figure
        is_unified_memory=True,
        driver_version=platform.mac_ver()[0] or None,
    )


def _detect_rocm() -> Optional[GPUInfo]:
    """Detect AMD ROCm GPU via rocm-smi."""
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode()
        data = json.loads(out)
        # rocm-smi JSON structure varies; best-effort parse
        for key, val in data.items():
            if isinstance(val, dict):
                name = val.get("Card series", key)
                vram_bytes = int(val.get("VRAM Total Memory (B)", 0))
                return GPUInfo(
                    name=name,
                    backend="rocm",
                    vram_gb=vram_bytes / 1024**3 if vram_bytes else None,
                    is_unified_memory=False,
                )
    except Exception:
        pass
    return None


def _detect_gpu() -> Optional[GPUInfo]:
    """Try each backend in priority order."""
    # 1. Apple Silicon (must come before CUDA check on macOS)
    gpu = _detect_apple_silicon()
    if gpu:
        return gpu
    # 2. NVIDIA CUDA
    gpu = _detect_nvidia()
    if gpu:
        return gpu
    # 3. AMD ROCm
    gpu = _detect_rocm()
    if gpu:
        return gpu
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile_hardware() -> HardwareProfile:
    """Detect and return a full hardware profile for this machine."""
    os_name = platform.system()
    os_version = platform.version()
    if os_name == "Darwin":
        os_version = f"macOS {platform.mac_ver()[0]}"
    elif os_name == "Linux":
        try:
            import distro  # type: ignore[import-untyped]
            os_version = distro.name(pretty=True)
        except Exception:
            os_version = platform.version()

    cpu = _cpu_info()
    memory = _memory_info()
    gpu = _detect_gpu()

    if gpu and gpu.backend != "none":
        inference_mode = "gpu"
    else:
        inference_mode = "cpu"

    return HardwareProfile(
        os_name=os_name,
        os_version=os_version,
        cpu=cpu,
        memory=memory,
        gpu=gpu,
        inference_mode=inference_mode,
    )
