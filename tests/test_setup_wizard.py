"""
Tests for setup.sh — the first-run configuration wizard.

Covers:
- Creates a valid .env with required keys
- Generated AUTOTUNE_ADMIN_KEY is at least 32 characters
- API key enforcement flag written correctly
- Preserves existing .env when user declines overwrite
- Handles --non-interactive flag (uses defaults, no prompts)
- Generated .env is parseable as KEY=VALUE pairs
- nginx/autotune.conf exists and contains required proxy directives
- docker-compose.yml team profile contains required service definitions
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SETUP_SH  = REPO_ROOT / "setup.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_setup(tmpdir: Path, *, extra_env: dict | None = None,
               extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    """Run setup.sh --non-interactive in a temporary directory."""
    env = {**os.environ, "HOME": str(tmpdir)}  # isolate ~/.autotune
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(SETUP_SH)] + ["--non-interactive"] + (extra_args or []),
        capture_output=True, text=True, env=env, cwd=str(tmpdir),
    )


def _parse_env(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines from a .env file (skip comments and blanks)."""
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


# ---------------------------------------------------------------------------
# setup.sh output and .env structure
# ---------------------------------------------------------------------------

class TestSetupWizard:
    def test_exits_zero(self, tmp_path):
        proc = _run_setup(tmp_path)
        assert proc.returncode == 0, proc.stderr

    def test_creates_env_file(self, tmp_path):
        _run_setup(tmp_path)
        assert (tmp_path / ".env").exists()

    def test_env_has_admin_key(self, tmp_path):
        _run_setup(tmp_path)
        env = _parse_env(tmp_path / ".env")
        assert "AUTOTUNE_ADMIN_KEY" in env
        assert len(env["AUTOTUNE_ADMIN_KEY"]) >= 32

    def test_admin_key_is_url_safe(self, tmp_path):
        _run_setup(tmp_path)
        key = _parse_env(tmp_path / ".env")["AUTOTUNE_ADMIN_KEY"]
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_=")
        assert all(c in allowed for c in key), f"Key contains non-URL-safe chars: {key!r}"

    def test_env_has_require_api_key(self, tmp_path):
        _run_setup(tmp_path)
        env = _parse_env(tmp_path / ".env")
        assert "AUTOTUNE_REQUIRE_API_KEY" in env
        assert env["AUTOTUNE_REQUIRE_API_KEY"] in ("0", "1")

    def test_non_interactive_defaults_to_enforcement_on(self, tmp_path):
        _run_setup(tmp_path)
        env = _parse_env(tmp_path / ".env")
        assert env["AUTOTUNE_REQUIRE_API_KEY"] == "1"

    def test_env_has_port(self, tmp_path):
        _run_setup(tmp_path)
        env = _parse_env(tmp_path / ".env")
        assert "AUTOTUNE_PORT" in env
        assert env["AUTOTUNE_PORT"].isdigit()

    def test_default_port_is_8765(self, tmp_path):
        _run_setup(tmp_path)
        assert _parse_env(tmp_path / ".env")["AUTOTUNE_PORT"] == "8765"

    def test_each_run_generates_unique_key(self, tmp_path, tmp_path_factory):
        other = tmp_path_factory.mktemp("other")
        _run_setup(tmp_path)
        _run_setup(other)
        k1 = _parse_env(tmp_path / ".env")["AUTOTUNE_ADMIN_KEY"]
        k2 = _parse_env(other   / ".env")["AUTOTUNE_ADMIN_KEY"]
        assert k1 != k2, "Two separate runs should generate different keys"

    def test_preserves_existing_env(self, tmp_path):
        sentinel = "AUTOTUNE_ADMIN_KEY=do-not-overwrite-me"
        (tmp_path / ".env").write_text(sentinel + "\n")
        _run_setup(tmp_path)  # default answer is "no" in non-interactive mode
        assert (tmp_path / ".env").read_text().startswith(sentinel)

    def test_output_mentions_dashboard(self, tmp_path):
        proc = _run_setup(tmp_path)
        assert "dashboard" in proc.stdout.lower()

    def test_output_mentions_admin_key(self, tmp_path):
        proc = _run_setup(tmp_path)
        assert "admin" in proc.stdout.lower() or "key" in proc.stdout.lower()

    def test_output_mentions_setup_complete(self, tmp_path):
        proc = _run_setup(tmp_path)
        assert "complete" in proc.stdout.lower() or "next" in proc.stdout.lower()


# ---------------------------------------------------------------------------
# nginx config
# ---------------------------------------------------------------------------

class TestNginxConfig:
    def test_nginx_conf_exists(self):
        assert (REPO_ROOT / "nginx" / "autotune.conf").exists()

    def test_nginx_conf_has_proxy_pass(self):
        text = (REPO_ROOT / "nginx" / "autotune.conf").read_text()
        assert "proxy_pass" in text

    def test_nginx_conf_has_upstream(self):
        text = (REPO_ROOT / "nginx" / "autotune.conf").read_text()
        assert "upstream" in text

    def test_nginx_conf_disables_buffering_for_streaming(self):
        text = (REPO_ROOT / "nginx" / "autotune.conf").read_text()
        assert "proxy_buffering" in text and "off" in text

    def test_nginx_conf_sets_generous_timeouts(self):
        text = (REPO_ROOT / "nginx" / "autotune.conf").read_text()
        assert "proxy_read_timeout" in text

    def test_nginx_conf_has_tls_commented_block(self):
        text = (REPO_ROOT / "nginx" / "autotune.conf").read_text()
        assert "ssl_certificate" in text


# ---------------------------------------------------------------------------
# docker-compose.yml team profile
# ---------------------------------------------------------------------------

class TestDockerComposeTeamProfile:
    def test_compose_file_exists(self):
        assert (REPO_ROOT / "docker-compose.yml").exists()

    def test_team_profile_present(self):
        text = (REPO_ROOT / "docker-compose.yml").read_text()
        assert '"team"' in text or "'team'" in text

    def test_autotune_team_service_defined(self):
        text = (REPO_ROOT / "docker-compose.yml").read_text()
        assert "autotune-team" in text

    def test_nginx_service_in_team_profile(self):
        text = (REPO_ROOT / "docker-compose.yml").read_text()
        assert "nginx" in text

    def test_require_api_key_enabled_in_team_service(self):
        text = (REPO_ROOT / "docker-compose.yml").read_text()
        assert "AUTOTUNE_REQUIRE_API_KEY" in text

    def test_team_service_uses_env_file(self):
        text = (REPO_ROOT / "docker-compose.yml").read_text()
        assert "env_file" in text

    def test_all_three_profiles_present(self):
        text = (REPO_ROOT / "docker-compose.yml").read_text()
        assert '"single"' in text or "'single'" in text
        assert '"multi"'  in text or "'multi'"  in text
        assert '"team"'   in text or "'team'"   in text
