"""
OS-level hardware tuner for LLM inference.

Makes genuine changes to process scheduling, memory management, and
thermal policy before each inference call to squeeze maximum performance.

All operations that require root fall back silently — the tuner degrades
gracefully and always leaves the system in a valid state.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import platform
import subprocess
import threading
from contextlib import contextmanager
from typing import Generator, Optional

import psutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# macOS QOS constants (XNU kernel)
# ---------------------------------------------------------------------------

_MACOS_QOS_CLASSES = {
    "BACKGROUND":       0x09,
    "UTILITY":          0x11,
    "DEFAULT":          0x15,
    "USER_INITIATED":   0x19,
    "USER_INTERACTIVE": 0x21,
}

# ---------------------------------------------------------------------------
# Linux CPU governor paths
# ---------------------------------------------------------------------------

_GOVERNOR_PATH = "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor"

_OS = platform.system()
_IS_APPLE_SILICON = (_OS == "Darwin" and platform.machine() == "arm64")


class HardwareTuner:
    """
    Apply and release hardware optimizations around an inference call.

    Usage:
        with tuner.optimized_for(profile):
            # inference happens here
    """

    def __init__(self) -> None:
        self._original_nice: int = 0
        self._gc_was_enabled = True
        self._libsystem: Optional[ctypes.CDLL] = None
        self._lock = threading.Lock()

        # Pre/post memory snapshots — populated by _apply/_restore
        self.memory_at_start_pct: float = 0.0
        self.memory_at_start_gb: float = 0.0
        self.swap_at_start_gb: float = 0.0
        self.memory_at_end_pct: float = 0.0
        self.memory_at_end_gb: float = 0.0
        self.swap_at_end_gb: float = 0.0

        if _OS == "Darwin":
            try:
                self._libsystem = ctypes.CDLL("libSystem.dylib")
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    @contextmanager
    def optimized_for(self, profile_name: str) -> Generator[None, None, None]:
        """Context manager: apply optimizations, yield, restore."""
        self._apply(profile_name)
        try:
            yield
        finally:
            self._restore()

    # ------------------------------------------------------------------ #
    # Apply                                                                #
    # ------------------------------------------------------------------ #

    def _apply(self, profile_name: str) -> None:
        with self._lock:
            self._original_nice = os.nice(0)
            self._gc_was_enabled = gc.isenabled()

            if profile_name == "fast":
                self._set_high_priority()
                self._disable_gc()
                self._macos_qos("USER_INTERACTIVE")
                self._linux_performance_governor()
                self._flush_python_allocator()
            elif profile_name == "balanced":
                self._set_medium_priority()
                self._disable_gc()
                self._macos_qos("USER_INITIATED")
            elif profile_name == "quality":
                self._set_medium_priority()
                self._macos_qos("USER_INITIATED")

            self._log_memory_state()

    # ------------------------------------------------------------------ #
    # Restore                                                              #
    # ------------------------------------------------------------------ #

    def _restore(self) -> None:
        with self._lock:
            # Snapshot memory state after inference before restoring anything
            self._log_memory_state(post=True)

            try:
                current = os.nice(0)
                if current != self._original_nice:
                    adj = self._original_nice - current
                    os.nice(adj)
            except Exception:
                pass

            if self._gc_was_enabled and not gc.isenabled():
                gc.enable()

            # Restore default QOS
            self._macos_qos("DEFAULT")

    # ------------------------------------------------------------------ #
    # Concrete actions                                                     #
    # ------------------------------------------------------------------ #

    def _set_high_priority(self) -> None:
        """Try to raise process priority (needs root for negative nice on Linux)."""
        try:
            current = os.nice(0)
            if current > -5:
                os.nice(-5 - current)
        except (PermissionError, OSError) as e:
            logger.debug("Could not set high priority: %s", e)

        # Also try to renice the Ollama process — it does the actual compute.
        # This only works on macOS/Linux without root for own-user processes.
        self._renice_ollama(-5)

    def _set_medium_priority(self) -> None:
        try:
            current = os.nice(0)
            if current > 0:
                os.nice(-current)   # restore to 0
        except (PermissionError, OSError) as e:
            logger.debug("Could not set medium priority: %s", e)

    def _renice_ollama(self, niceness: int) -> None:
        """Find the Ollama server process and renice it to improve scheduling."""
        if _OS not in ("Darwin", "Linux"):
            return
        try:
            for proc in psutil.process_iter(["name", "pid", "username"]):
                if proc.info["name"] and "ollama" in proc.info["name"].lower():
                    # Only renice processes owned by the current user
                    import getpass
                    if proc.info["username"] == getpass.getuser():
                        os.system(f"renice -n {niceness} -p {proc.info['pid']} 2>/dev/null")
                        break
        except Exception as e:
            logger.debug("Could not renice Ollama: %s", e)

    def _disable_gc(self) -> None:
        """Disable Python GC during inference to avoid stop-the-world pauses."""
        gc.collect()    # collect first to clean up
        gc.disable()

    def _macos_qos(self, qos_class: str) -> None:
        """Set the current thread's QOS class (macOS only, no root needed)."""
        if _OS != "Darwin" or self._libsystem is None:
            return
        try:
            qos_val = _MACOS_QOS_CLASSES.get(qos_class, _MACOS_QOS_CLASSES["DEFAULT"])
            fn = self._libsystem.pthread_set_qos_class_self_np
            fn.argtypes = [ctypes.c_uint32, ctypes.c_int]
            fn.restype = ctypes.c_int
            fn(qos_val, 0)
        except Exception as e:
            logger.debug("Could not set QOS class %s: %s", qos_class, e)

    def _linux_performance_governor(self) -> None:
        """Set CPU governor to performance on Linux (requires root)."""
        if _OS != "Linux":
            return
        try:
            cpu_count = psutil.cpu_count() or 1
            for i in range(cpu_count):
                path = _GOVERNOR_PATH.format(i)
                if os.path.exists(path):
                    with open(path, "w") as f:
                        f.write("performance")
        except (PermissionError, OSError):
            pass

    def _flush_python_allocator(self) -> None:
        """Return free blocks to OS — reduces memory fragmentation before inference."""
        # malloc_trim is Linux-only (glibc); skip on macOS to avoid a dlopening error
        if _OS == "Linux":
            try:
                ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
            except Exception as e:
                logger.debug("malloc_trim failed: %s", e)
        # Python's pymalloc doesn't release to OS by default; trigger a full GC
        try:
            gc.collect(2)
        except Exception:
            pass

    def _log_memory_state(self, post: bool = False) -> None:
        """Record current memory state before (post=False) or after (post=True) inference."""
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        if post:
            self.memory_at_end_pct = vm.percent
            self.memory_at_end_gb = vm.used / 1024**3
            self.swap_at_end_gb = sw.used / 1024**3
        else:
            self.memory_at_start_pct = vm.percent
            self.memory_at_start_gb = vm.used / 1024**3
            self.swap_at_start_gb = sw.used / 1024**3

    def delta_ram_gb(self) -> float:
        """RAM delta from before to after last inference call."""
        return self.memory_at_end_gb - self.memory_at_start_gb

    def delta_swap_gb(self) -> float:
        """Swap delta from before to after last inference call."""
        return self.swap_at_end_gb - self.swap_at_start_gb

    # ------------------------------------------------------------------ #
    # Memory pressure check (call anytime during inference)               #
    # ------------------------------------------------------------------ #

    def memory_pressure(self) -> tuple[float, bool]:
        """
        Return (ram_percent, is_critical).
        Call during inference to decide whether to abort early.
        """
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        critical = vm.percent > 93 or sw.percent > 20
        return vm.percent, critical

    # ------------------------------------------------------------------ #
    # Pre-flight check                                                     #
    # ------------------------------------------------------------------ #

    def preflight(self, required_gb: float) -> dict:
        """
        Check if the system has enough headroom for an inference call.
        Returns a dict with status and recommendations.
        """
        vm = psutil.virtual_memory()
        available_gb = vm.available / 1024**3
        sw = psutil.swap_memory()

        ok = available_gb >= required_gb * 1.1   # 10% headroom
        return {
            "ok": ok,
            "available_gb": round(available_gb, 2),
            "required_gb": round(required_gb, 2),
            "ram_percent": round(vm.percent, 1),
            "swap_used_gb": round(sw.used / 1024**3, 2),
            "warning": None if ok else (
                f"Low memory: {available_gb:.1f} GB available, "
                f"{required_gb:.1f} GB needed. May cause swapping."
            ),
        }


# Module-level singleton
_tuner: Optional[HardwareTuner] = None


def get_tuner() -> HardwareTuner:
    global _tuner
    if _tuner is None:
        _tuner = HardwareTuner()
    return _tuner
