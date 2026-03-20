"""
Platform-aware live metrics collector.

Handles: macOS (Apple Silicon + Intel), Linux, Windows.
All subprocess calls are non-blocking with short timeouts.
"""

from __future__ import annotations

import json
import platform
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Optional

import psutil

from .types import (
    LLMProcess, LiveMetrics, OllamaModel, ThermalState,
)

# ---------------------------------------------------------------------------
# LLM process detection patterns
# ---------------------------------------------------------------------------

_LLM_NAMES = {
    "llama-server", "llama-cli", "llama.cpp", "llamafile",
    "ollama", "koboldcpp", "text-generation", "lm_studio", "lmstudio",
}

_LLM_CMDLINE = {
    "llama_cpp", "llama-cpp", "llama.cpp",
    "mlx_lm", "mlx-lm",
    "vllm", "llamacpp",
    "transformers", "accelerate",
    "ctransformers", "exllamav2",
    "autogptq", "gptq",
    "koboldcpp", "llamafile",
}

_RUNTIME_MAP = {
    "ollama": "Ollama",
    "llama-server": "llama.cpp",
    "llama-cli": "llama.cpp",
    "llamafile": "Llamafile",
    "koboldcpp": "KoboldCPP",
    "lm_studio": "LM Studio",
    "lmstudio": "LM Studio",
    "mlx_lm": "MLX",
    "vllm": "vLLM",
}


# ---------------------------------------------------------------------------
# Thermal detection (platform-specific)
# ---------------------------------------------------------------------------

def _thermal_macos() -> tuple[ThermalState, int, Optional[float]]:
    """Returns (thermal_state, cpu_speed_limit_pct, cpu_temp_c)."""
    state = ThermalState.NOMINAL
    limit = 100
    temp: Optional[float] = None

    # 1. pmset for throttle state
    try:
        out = subprocess.check_output(
            ["pmset", "-g", "therm"], stderr=subprocess.DEVNULL, timeout=2
        ).decode()
        if "CPU_Speed_Limit" in out:
            for line in out.splitlines():
                if "CPU_Speed_Limit" in line:
                    parts = line.split()
                    for p in parts:
                        try:
                            limit = int(p)
                        except ValueError:
                            pass
        if "degraded" in out.lower():
            state = ThermalState.THROTTLING
        elif "warm" in out.lower():
            state = ThermalState.WARM
    except Exception:
        pass

    # 2. Temperature via powermetrics (requires sudo – skip silently)
    # Instead try psutil sensors if available
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for key in ("coretemp", "cpu_thermal", "TC0P", "TC0p"):
                if key in temps and temps[key]:
                    temp = temps[key][0].current
                    break
    except (AttributeError, Exception):
        pass

    if limit < 90:
        state = ThermalState.THROTTLING
    elif limit < 100:
        state = ThermalState.WARM

    return state, limit, temp


def _thermal_linux() -> tuple[ThermalState, int, Optional[float]]:
    import os, glob
    state = ThermalState.NOMINAL
    temp: Optional[float] = None

    # Try hwmon / thermal zones
    zone_files = glob.glob("/sys/class/thermal/thermal_zone*/temp")
    temps = []
    for zf in zone_files:
        try:
            with open(zf) as f:
                temps.append(int(f.read().strip()) / 1000)
        except Exception:
            pass
    if temps:
        temp = max(temps)
        if temp > 90:
            state = ThermalState.CRITICAL
        elif temp > 85:
            state = ThermalState.THROTTLING
        elif temp > 75:
            state = ThermalState.WARNING
        elif temp > 65:
            state = ThermalState.WARM

    return state, 100, temp


def _thermal_windows() -> tuple[ThermalState, int, Optional[float]]:
    # WMI is complex; just return nominal unless psutil gives us something
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            all_t = [s.current for sensors in temps.values() for s in sensors]
            if all_t:
                t = max(all_t)
                state = ThermalState.THROTTLING if t > 90 else ThermalState.WARM if t > 75 else ThermalState.NOMINAL
                return state, 100, t
    except Exception:
        pass
    return ThermalState.NOMINAL, 100, None


def _get_thermals() -> tuple[ThermalState, int, Optional[float]]:
    os_name = platform.system()
    if os_name == "Darwin":
        return _thermal_macos()
    elif os_name == "Linux":
        return _thermal_linux()
    else:
        return _thermal_windows()


# ---------------------------------------------------------------------------
# VRAM collection
# ---------------------------------------------------------------------------

def _nvidia_vram() -> tuple[Optional[float], Optional[float], Optional[float]]:
    """(used_gb, total_gb, percent)"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=3,
        ).decode().strip().splitlines()[0]
        used_mb, total_mb = [float(x.strip()) for x in out.split(",")]
        used = used_mb / 1024
        total = total_mb / 1024
        return used, total, (used / total * 100) if total else None
    except Exception:
        return None, None, None


def _apple_silicon_vram(ram_used_gb: float, ram_total_gb: float) -> tuple[float, float, float]:
    """For unified memory, VRAM == system RAM from the GPU's perspective."""
    return ram_used_gb, ram_total_gb, ram_used_gb / ram_total_gb * 100


# ---------------------------------------------------------------------------
# LLM process detection
# ---------------------------------------------------------------------------

def _detect_llm_processes() -> list[LLMProcess]:
    results: list[LLMProcess] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info", "cpu_percent"]):
        try:
            name = (proc.info["name"] or "").lower()
            cmdline_parts = proc.info["cmdline"] or []
            cmdline = " ".join(str(c) for c in cmdline_parts).lower()

            is_llm = (
                any(p in name for p in _LLM_NAMES)
                or any(p in cmdline for p in _LLM_CMDLINE)
            )
            if not is_llm:
                continue

            # Determine runtime
            runtime = "unknown"
            for key, label in _RUNTIME_MAP.items():
                if key in name or key in cmdline:
                    runtime = label
                    break

            # Guess model from cmdline
            model_hint = ""
            keywords = ["llama", "mistral", "qwen", "gemma", "phi", "deepseek",
                        "falcon", "yi", "mixtral", "smollm"]
            for kw in keywords:
                if kw in cmdline:
                    # Try to grab surrounding token
                    idx = cmdline.find(kw)
                    snippet = cmdline[max(0, idx-5):idx+30].strip()
                    model_hint = snippet
                    break

            mem = proc.info["memory_info"]
            results.append(LLMProcess(
                pid=proc.info["pid"],
                name=proc.info["name"] or "",
                ram_gb=mem.rss / 1024**3 if mem else 0.0,
                cpu_percent=proc.info["cpu_percent"] or 0.0,
                runtime=runtime,
                model_hint=model_hint[:40],
                cmdline_snippet=cmdline[:80],
            ))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    return results


# ---------------------------------------------------------------------------
# Ollama API probe
# ---------------------------------------------------------------------------

def _query_ollama() -> list[OllamaModel]:
    try:
        req = urllib.request.Request("http://localhost:11434/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
        models = []
        for m in data.get("models", []):
            size_vram = m.get("size_vram", 0) / 1024**3
            details = m.get("details", {})
            models.append(OllamaModel(
                name=m.get("name", "?"),
                size_gb=m.get("size", 0) / 1024**3,
                context_len=m.get("context_length", 0),
                vram_gb=size_vram,
            ))
        return models
    except Exception:
        return []


def _query_lmstudio() -> list[OllamaModel]:
    """LM Studio OpenAI-compatible endpoint."""
    try:
        req = urllib.request.Request("http://localhost:1234/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
        return [
            OllamaModel(name=m.get("id", "?"), size_gb=0.0, context_len=0, vram_gb=0.0)
            for m in data.get("data", [])
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Thread-safe live metrics collector.

    Call start() to begin background polling.
    Access .latest for the most recent snapshot.
    """

    def __init__(self, interval_sec: float = 1.0, history_len: int = 300) -> None:
        self.interval = interval_sec
        self._history: deque[LiveMetrics] = deque(maxlen=history_len)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Cache slow subprocess results (refresh every N seconds)
        self._thermal_cache: tuple[ThermalState, int, Optional[float]] = (ThermalState.NOMINAL, 100, None)
        self._thermal_last = 0.0
        self._thermal_interval = 5.0  # seconds between thermal checks

        self._ollama_cache: list[OllamaModel] = []
        self._ollama_last = 0.0
        self._ollama_interval = 3.0

        # Detect platform
        self._os = platform.system()
        self._is_apple_silicon = (self._os == "Darwin" and platform.machine() == "arm64")

        # Trigger first cpu_percent call (needs two readings)
        psutil.cpu_percent(percpu=True)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------ #
    # Access                                                               #
    # ------------------------------------------------------------------ #

    @property
    def latest(self) -> Optional[LiveMetrics]:
        with self._lock:
            return self._history[-1] if self._history else None

    @property
    def history(self) -> list[LiveMetrics]:
        with self._lock:
            return list(self._history)

    def growth_rates(self, window_sec: float = 60.0) -> dict[str, float]:
        """MB/min growth for ram, vram, swap over the last window_sec."""
        h = self.history
        if len(h) < 2:
            return {"ram": 0.0, "vram": 0.0, "swap": 0.0}

        now = h[-1].timestamp
        cutoff = now - window_sec
        old = next((m for m in h if m.timestamp >= cutoff), h[0])
        elapsed_min = (now - old.timestamp) / 60 or 1e-9

        ram_growth = (h[-1].ram_used_gb - old.ram_used_gb) * 1024 / elapsed_min
        vram_growth = 0.0
        if h[-1].vram_used_gb is not None and old.vram_used_gb is not None:
            vram_growth = (h[-1].vram_used_gb - old.vram_used_gb) * 1024 / elapsed_min
        swap_growth = (h[-1].swap_used_gb - old.swap_used_gb) * 1024 / elapsed_min

        return {"ram": ram_growth, "vram": vram_growth, "swap": swap_growth}

    # ------------------------------------------------------------------ #
    # Internal loop                                                        #
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        while self._running:
            try:
                m = self._collect()
                with self._lock:
                    self._history.append(m)
            except Exception:
                pass
            time.sleep(self.interval)

    def _collect(self) -> LiveMetrics:
        now = time.time()

        # Memory
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()

        ram_used = vm.used / 1024**3
        ram_total = vm.total / 1024**3
        ram_avail = vm.available / 1024**3

        swap_used = sw.used / 1024**3
        swap_total = sw.total / 1024**3

        # CPU
        per_core = psutil.cpu_percent(percpu=True)
        cpu_pct = sum(per_core) / len(per_core) if per_core else 0.0

        # VRAM
        vram_used: Optional[float] = None
        vram_total: Optional[float] = None
        vram_pct: Optional[float] = None

        if self._is_apple_silicon:
            vu, vt, vp = _apple_silicon_vram(ram_used, ram_total)
            vram_used, vram_total, vram_pct = vu, vt, vp
        else:
            vram_used, vram_total, vram_pct = _nvidia_vram()

        # Thermals (cached)
        if now - self._thermal_last > self._thermal_interval:
            self._thermal_cache = _get_thermals()
            self._thermal_last = now
        thermal_state, speed_limit, cpu_temp = self._thermal_cache
        # try psutil for GPU temp
        gpu_temp: Optional[float] = None
        try:
            temps = psutil.sensors_temperatures()
            for key in ("nouveau", "amdgpu", "radeon", "nvidia"):
                if key in temps and temps[key]:
                    gpu_temp = temps[key][0].current
                    break
        except Exception:
            pass

        # LLM processes
        llm_procs = _detect_llm_processes()

        # Ollama (cached)
        if now - self._ollama_last > self._ollama_interval:
            self._ollama_cache = _query_ollama() or _query_lmstudio()
            self._ollama_last = now
        ollama_models = self._ollama_cache

        # Growth rates
        rates = self.growth_rates()

        return LiveMetrics(
            timestamp=now,
            ram_total_gb=ram_total,
            ram_used_gb=ram_used,
            ram_available_gb=ram_avail,
            ram_percent=vm.percent,
            swap_total_gb=swap_total,
            swap_used_gb=swap_used,
            swap_percent=sw.percent,
            vram_total_gb=vram_total,
            vram_used_gb=vram_used,
            vram_percent=vram_pct,
            cpu_percent=cpu_pct,
            cpu_per_core=list(per_core),
            cpu_temp_c=cpu_temp,
            gpu_temp_c=gpu_temp,
            thermal_state=thermal_state,
            cpu_speed_limit_pct=speed_limit,
            llm_processes=llm_procs,
            ollama_models=ollama_models,
            tokens_per_sec=None,
            gen_tokens_per_sec=None,
            ttft_ms=None,
            queue_depth=0,
            swap_growth_mb_per_min=rates["swap"],
            ram_growth_mb_per_min=rates["ram"],
            vram_growth_mb_per_min=rates["vram"],
        )
