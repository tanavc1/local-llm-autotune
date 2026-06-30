"""
First-run gate tests.

`autotune start` must be run before anything else. Until the machine has been
set up once, every other command is blocked and redirects to `autotune start`.
`--help`, `start`/`init`, `version` and `upgrade` stay available so the CLI is
discoverable and updatable before setup.

These tests monkeypatch ``_is_initialized`` so they never touch the real
~/.autotune sentinel and never run the heavy setup body.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

import autotune.cli as cli_mod
from autotune.cli import cli

runner = CliRunner()


@pytest.fixture
def not_initialized(monkeypatch):
    monkeypatch.setattr(cli_mod, "_is_initialized", lambda: False)


@pytest.fixture
def initialized(monkeypatch):
    monkeypatch.setattr(cli_mod, "_is_initialized", lambda: True)


# ── Blocking ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("args", [
    ["recommend"],
    ["serve"],
    ["chat"],
    ["pull", "qwen3:8b"],
    ["hardware"],
    ["bench"],
])
def test_commands_blocked_before_start(not_initialized, args):
    res = runner.invoke(cli, args)
    assert res.exit_code == 1, f"{args} should be blocked"
    assert "autotune start" in res.output


def test_unknown_command_blocked_before_start(not_initialized):
    res = runner.invoke(cli, ["definitely-not-a-command"])
    assert res.exit_code == 1
    assert "autotune start" in res.output


# ── Always-allowed escape hatches ───────────────────────────────────────────

@pytest.mark.parametrize("args", [
    ["--help"],
    ["recommend", "--help"],
    ["serve", "--help"],
    ["start", "--help"],
    ["init", "--help"],
    ["chat", "-h"],
])
def test_help_always_allowed(not_initialized, args):
    res = runner.invoke(cli, args)
    assert res.exit_code == 0, f"{args} should be allowed"
    assert "autotune start" not in res.output or "Usage" in res.output


def test_start_and_init_are_allowlisted():
    assert "start" in cli_mod._PRE_START_ALLOWED
    assert "init" in cli_mod._PRE_START_ALLOWED
    assert "version" in cli_mod._PRE_START_ALLOWED
    assert "upgrade" in cli_mod._PRE_START_ALLOWED


def test_bare_invocation_shows_start_prompt(not_initialized):
    res = runner.invoke(cli, [])
    assert res.exit_code == 0
    assert "autotune start" in res.output


# ── Gate opens after setup ──────────────────────────────────────────────────

def test_gate_opens_when_initialized(initialized):
    # An unknown command is now handled by Click itself (exit 2 / "No such
    # command"), proving the gate no longer intercepts.
    res = runner.invoke(cli, ["definitely-not-a-command"])
    assert res.exit_code == 2
    assert "autotune start" not in res.output


# ── start command wiring ────────────────────────────────────────────────────

def test_start_command_registered():
    assert "start" in cli.commands


def test_start_delegates_to_setup(not_initialized, monkeypatch):
    """`autotune start` must run the setup flow (which begins with the Ollama
    check). Stub the check to fail fast and assert we reached it."""
    import autotune.api.ollama_pull as op
    monkeypatch.setattr(op, "ensure_ollama_running", lambda *a, **k: False)
    res = runner.invoke(cli, ["start"])
    assert res.exit_code == 1
    assert "Ollama" in res.output


# ── Grandfathering: never lock out an existing install ──────────────────────

def _patch_home(monkeypatch, tmp_path):
    import pathlib
    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: tmp_path))


def test_is_initialized_true_when_sentinel_present(monkeypatch, tmp_path):
    _patch_home(monkeypatch, tmp_path)
    sentinel = tmp_path / ".autotune" / "initialized"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("{}")
    assert cli_mod._is_initialized() is True


def test_is_initialized_false_for_fresh_install(monkeypatch, tmp_path):
    # No sentinel, no prior-use databases → must run `autotune start`.
    _patch_home(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_mod, "_has_prior_use", lambda: False)
    assert cli_mod._is_initialized() is False


def test_prior_use_grandfathers_in_and_backfills_sentinel(monkeypatch, tmp_path):
    """An existing user who upgrades (recall.db on disk, no sentinel) is treated
    as set up, and the sentinel is backfilled so they stay unblocked."""
    _patch_home(monkeypatch, tmp_path)
    recall = tmp_path / ".autotune" / "recall.db"
    recall.parent.mkdir(parents=True)
    recall.write_bytes(b"")
    # Real _has_prior_use should detect recall.db once home is patched.
    assert cli_mod._has_prior_use() is True
    assert cli_mod._is_initialized() is True
    # Sentinel was backfilled, so the next check is a cheap stat with no prior-use scan.
    assert (tmp_path / ".autotune" / "initialized").exists()


def test_existing_user_not_blocked_after_upgrade(monkeypatch):
    """End-to-end: a gated command runs for a grandfathered user."""
    monkeypatch.setattr(cli_mod, "_has_prior_use", lambda: True)
    monkeypatch.setattr(cli_mod, "_mark_initialized", lambda *a, **k: None)
    # Unknown command now reaches Click (exit 2), proving the gate let it through.
    res = runner.invoke(cli, ["definitely-not-a-command"])
    assert res.exit_code == 2
    assert "autotune start" not in res.output
