"""
CLI smoke tests — run for every visible command and all subcommands.

Strategy
--------
1.  Every visible command and every subcommand is invoked with ``--help``
    via Click's CliRunner.  This exercises:
      - The Click decorator chain (parameter definitions, choice sets, …)
      - Module-level imports in cli.py
      - Any code that runs before the lazy body imports
    Exit code must be 0 and output must be non-empty.

2.  Hidden commands (deprecated aliases kept for backwards compatibility) are
    also verified to be accessible when called directly.

3.  Wiring checks: for every lazy ``from autotune.X import Y`` inside a
    command body, we import the module and assert the attribute exists.
    This catches typos and stale references without needing Ollama.

All tests are fully offline — no network, no Ollama, no file writes.
"""

from __future__ import annotations

import importlib

import pytest
from click.testing import CliRunner

from autotune.cli import cli

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

runner = CliRunner()


def _help_ok(args: list[str]) -> None:
    """Assert that invoking the CLI with ``args + ['--help']`` exits 0."""
    result = runner.invoke(cli, args + ["--help"], catch_exceptions=False)
    assert result.exit_code == 0, (
        f"`autotune {' '.join(args)} --help` exited {result.exit_code}.\n"
        f"Output:\n{result.output}"
    )
    assert result.output.strip(), (
        f"`autotune {' '.join(args)} --help` produced no output."
    )


def _has_attr(module_path: str, attr: str) -> None:
    """Assert that ``module_path`` exports ``attr``."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, attr), (
        f"{module_path} is missing attribute '{attr}' — "
        f"update the import in cli.py or the backing module."
    )


# ---------------------------------------------------------------------------
# 1. Top-level help
# ---------------------------------------------------------------------------

def test_top_level_help() -> None:
    _help_ok([])


# ---------------------------------------------------------------------------
# 2. Every visible top-level command
# ---------------------------------------------------------------------------

VISIBLE_TOP_LEVEL = [
    "bench",
    "chat",
    "compare",
    "config",
    "delete",
    "doctor",
    "hardware",
    "ls",
    "memory",
    "mlx",
    "models",
    "proof",
    "ps",
    "pull",
    "recommend",
    "run",
    "serve",
    "storage",
    "stress-test",
    "telemetry",
    "unload",
    "upgrade",
    "webui",
]


@pytest.mark.parametrize("command", VISIBLE_TOP_LEVEL)
def test_visible_command_help(command: str) -> None:
    _help_ok([command])


# ---------------------------------------------------------------------------
# 3. Every bench subcommand
# ---------------------------------------------------------------------------

BENCH_SUBCOMMANDS = ["quick", "duel", "suite", "agent", "ux", "os", "server", "all"]


@pytest.mark.parametrize("sub", BENCH_SUBCOMMANDS)
def test_bench_subcommand_help(sub: str) -> None:
    _help_ok(["bench", sub])


# ---------------------------------------------------------------------------
# 4. Every memory subcommand
# ---------------------------------------------------------------------------

MEMORY_SUBCOMMANDS = ["search", "list", "stats", "forget", "setup"]


@pytest.mark.parametrize("sub", MEMORY_SUBCOMMANDS)
def test_memory_subcommand_help(sub: str) -> None:
    _help_ok(["memory", sub])


# ---------------------------------------------------------------------------
# 5. Every mlx subcommand
# ---------------------------------------------------------------------------

MLX_SUBCOMMANDS = ["list", "pull", "resolve"]


@pytest.mark.parametrize("sub", MLX_SUBCOMMANDS)
def test_mlx_subcommand_help(sub: str) -> None:
    _help_ok(["mlx", sub])


# ---------------------------------------------------------------------------
# 6. Every webui subcommand
# ---------------------------------------------------------------------------

WEBUI_SUBCOMMANDS = ["chats", "models", "open", "login", "install", "launch", "connect", "status"]


@pytest.mark.parametrize("sub", WEBUI_SUBCOMMANDS)
def test_webui_subcommand_help(sub: str) -> None:
    _help_ok(["webui", sub])


# ---------------------------------------------------------------------------
# 7. Every config subcommand
# ---------------------------------------------------------------------------

CONFIG_SUBCOMMANDS = ["show", "set", "get", "reset"]


@pytest.mark.parametrize("sub", CONFIG_SUBCOMMANDS)
def test_config_subcommand_help(sub: str) -> None:
    _help_ok(["config", sub])


# ---------------------------------------------------------------------------
# 8. Hidden commands still callable
# ---------------------------------------------------------------------------

HIDDEN_COMMANDS = [
    ["benchmark", "placeholder"],   # requires MODEL argument
    ["proof-suite"],
    ["agent-bench"],
    ["user-bench"],
    ["fetch", "placeholder"],       # requires MODEL_ID argument
    ["fetch-many"],
    ["db"],
    ["db-models"],
    ["log-run"],
    ["session"],
]


@pytest.mark.parametrize("args", HIDDEN_COMMANDS)
def test_hidden_command_accessible(args: list[str]) -> None:
    """Hidden commands must still be callable (just absent from --help)."""
    _help_ok(args)


# ---------------------------------------------------------------------------
# 9. Verify that the `init` command is visible and has all expected options
# ---------------------------------------------------------------------------

def test_init_command_help() -> None:
    _help_ok(["init"])


def test_init_not_in_help_absent() -> None:
    """The init command should appear in top-level --help."""
    result = runner.invoke(cli, ["--help"], catch_exceptions=False)
    assert "init" in result.output


# ---------------------------------------------------------------------------
# 10. Wiring checks — lazy imports inside every command body
# ---------------------------------------------------------------------------
#
# Format: (module_path, attribute_name)
# If any of these fails, a command body has a stale or wrong reference.

WIRING: list[tuple[str, str]] = [
    # recommend
    ("autotune.config.generator",   "generate_recommendations"),
    ("autotune.config.generator",   "MODE_WEIGHTS"),
    ("autotune.config.generator",   "CONTEXT_LENGTHS"),
    ("autotune.config.generator",   "GPU_LAYER_FRACTIONS"),
    ("autotune.hardware.profiler",  "profile_hardware"),
    ("autotune.models.registry",    "MODEL_REGISTRY"),
    ("autotune.output.formatter",   "print_hardware_profile"),
    ("autotune.output.formatter",   "print_recommendations"),
    # hardware
    ("autotune.hardware.profiler",  "get_ram_hogs"),
    ("autotune.hardware.ram_advisor", "compute_unlock_suggestions"),
    ("autotune.output.formatter",   "print_ram_pressure_report"),
    # ps
    ("autotune.api.running_models", "get_running_models"),
    ("autotune.output.formatter",   "print_running_models"),
    # models
    ("autotune.output.formatter",   "print_model_table"),
    # pull
    ("autotune.api.ollama_pull",    "pull_model"),
    ("autotune.api.ollama_pull",    "print_popular_models"),
    ("autotune.api.ollama_pull",    "OllamaNotRunningError"),
    ("autotune.api.ollama_pull",    "PullError"),
    # delete
    ("autotune.api.ollama_pull",    "ensure_ollama_running"),
    # bench quick
    ("autotune.bench.quick_proof",  "run_quick_proof"),
    # bench duel / benchmark (hidden)
    ("autotune.bench.compare",      "run_comparison"),
    ("autotune.bench.compare",      "print_report"),
    ("autotune.bench.compare",      "export_json"),
    ("autotune.bench.compare",      "CompareReport"),
    # bench suite / proof-suite (hidden)
    ("autotune.bench.proof_suite",  "main"),
    # bench agent / agent-bench (hidden)
    ("autotune.bench.agent_bench",  "_async_main"),
    # bench ux / user-bench (hidden)
    ("autotune.bench.user_bench",   "main"),
    # bench os + bench server
    ("autotune.bench.bench_cmd",    "run_os_bench"),
    ("autotune.bench.bench_cmd",    "print_os_bench_results"),
    ("autotune.bench.bench_cmd",    "run_server_bench"),
    ("autotune.bench.bench_cmd",    "print_server_bench_results"),
    # stress-test / compare (hidden benchmarking)
    ("autotune.bench.runner",       "run_raw_ollama"),
    ("autotune.bench.runner",       "run_bench_ollama_only"),
    ("autotune.bench.runner",       "save_result"),
    ("autotune.bench.runner",       "run_bench"),
    ("autotune.bench.runner",       "BenchResult"),
    # run
    ("autotune.api.model_selector", "FitClass"),
    ("autotune.api.model_selector", "ModelSelector"),
    # chat / run
    ("autotune.api.chat",           "start_chat"),
    # telemetry
    ("autotune.telemetry",          "maybe_prompt_consent"),
    ("autotune.telemetry",          "emit"),
    ("autotune.telemetry",          "register_install"),
    ("autotune.telemetry.consent",  "is_opted_in"),
    ("autotune.telemetry.consent",  "set_consent"),
    ("autotune.telemetry.consent",  "consent_answered"),
    ("autotune.telemetry.consent",  "get_install_key"),
    ("autotune.telemetry.events",   "EventType"),
    # storage
    ("autotune.db.storage_prefs",   "is_storage_enabled"),
    ("autotune.db.storage_prefs",   "set_storage_enabled"),
    ("autotune.db.storage_prefs",   "storage_pref_set"),
    # memory group
    ("autotune.recall.manager",     "get_recall_manager"),
    # mlx group
    ("autotune.api.backends.mlx_backend", "mlx_available"),
    ("autotune.api.backends.mlx_backend", "list_cached_mlx_models"),
    ("autotune.api.backends.mlx_backend", "unload_mlx_model"),
    # config group
    ("autotune.config.user_config", "KNOWN_KEYS"),
    ("autotune.config.user_config", "load_config"),
    ("autotune.config.user_config", "set_value"),
    ("autotune.config.user_config", "get_value"),
    ("autotune.config.user_config", "effective_default"),
    ("autotune.config.user_config", "reset_config"),
    # unload
    ("autotune.api.running_models", "get_running_models"),
    # doctor
    ("autotune.db.store",           "get_db"),
    # log-run (hidden)
    ("autotune.db.fingerprint",     "hardware_id"),
    ("autotune.db.fingerprint",     "hardware_to_db_dict"),
    # session (hidden)
    ("autotune.session.controller", "SessionController"),
    # init
    ("autotune.bench.quick_proof",  "QuickProofResult"),
]


@pytest.mark.parametrize("module_path,attr", WIRING)
def test_wiring(module_path: str, attr: str) -> None:
    _has_attr(module_path, attr)


# ---------------------------------------------------------------------------
# 11. Verify hidden commands are NOT in top-level --help output
# ---------------------------------------------------------------------------

SHOULD_BE_HIDDEN = [
    "benchmark",
    "proof-suite",
    "agent-bench",
    "user-bench",
    "fetch",
    "fetch-many",
    "db",
    "db-models",
    "log-run",
    "session",
]


@pytest.mark.parametrize("cmd", SHOULD_BE_HIDDEN)
def test_hidden_commands_absent_from_help(cmd: str) -> None:
    result = runner.invoke(cli, ["--help"], catch_exceptions=False)
    # The command name must not appear as a CLI command entry in help.
    # We check for "  cmd  " (indented, space-padded) to avoid false positives
    # in description text.
    import re
    pattern = rf"^\s+{re.escape(cmd)}\s"
    matches = [line for line in result.output.splitlines() if re.match(pattern, line)]
    assert not matches, (
        f"Hidden command '{cmd}' is showing in --help output:\n{result.output}"
    )
