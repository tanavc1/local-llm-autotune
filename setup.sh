#!/usr/bin/env bash
# autotune setup wizard — first-run configuration helper
#
# Generates a secure AUTOTUNE_ADMIN_KEY, writes .env, and gets you running.
# Safe to re-run: will not overwrite an existing .env unless you confirm.
#
# Usage:
#   ./setup.sh                        interactive wizard
#   ./setup.sh --non-interactive      CI / scripted setup (accepts all defaults)
#   ./setup.sh --model llama3.2:3b    pull a model after setup completes

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
if [ -t 1 ] && command -v tput &>/dev/null; then
  BOLD=$(tput bold 2>/dev/null || true)
  RESET=$(tput sgr0 2>/dev/null || true)
  CYAN=$(tput setaf 6 2>/dev/null || true)
  GREEN=$(tput setaf 2 2>/dev/null || true)
  YELLOW=$(tput setaf 3 2>/dev/null || true)
  RED=$(tput setaf 1 2>/dev/null || true)
  DIM=$(tput dim 2>/dev/null || true)
else
  BOLD=""; RESET=""; CYAN=""; GREEN=""; YELLOW=""; RED=""; DIM=""
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
say()    { printf "%s\n" "$*"; }
info()   { say "  ${CYAN}▸${RESET} $*"; }
ok()     { say "  ${GREEN}✓${RESET} $*"; }
warn()   { say "  ${YELLOW}⚠${RESET}  $*"; }
die()    { say "  ${RED}✗${RESET}  $*" >&2; exit 1; }
blank()  { say ""; }
rule()   { say "${DIM}────────────────────────────────────────────────────────${RESET}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
NON_INTERACTIVE=0
PULL_MODEL=""

for arg in "$@"; do
  case "$arg" in
    --non-interactive) NON_INTERACTIVE=1 ;;
    --model)           shift; PULL_MODEL="${1:-}" ;;
    --model=*)         PULL_MODEL="${arg#--model=}" ;;
  esac
done

ask() {
  # ask <prompt> <default>  → echoes answer (default if non-interactive)
  local prompt="$1" default="$2" answer
  if [[ $NON_INTERACTIVE -eq 1 ]]; then
    echo "$default"
    return
  fi
  read -r -p "  ${BOLD}${prompt}${RESET} [${default}]: " answer
  echo "${answer:-$default}"
}

ask_yn() {
  # ask_yn <prompt> <default yes|no>  → echoes "yes" or "no"
  local prompt="$1" default="$2" answer
  if [[ $NON_INTERACTIVE -eq 1 ]]; then
    echo "$default"
    return
  fi
  local hint
  if [[ "$default" == "yes" ]]; then hint="Y/n"; else hint="y/N"; fi
  read -r -p "  ${BOLD}${prompt}${RESET} [${hint}]: " answer
  answer="${answer:-$default}"
  case "${answer,,}" in
    y|yes) echo "yes" ;;
    *)     echo "no"  ;;
  esac
}

# ── OS detection ──────────────────────────────────────────────────────────────
OS="linux"
case "$(uname -s)" in
  Darwin) OS="macos" ;;
  MINGW*|MSYS*|CYGWIN*) OS="windows" ;;
esac

# ── Banner ────────────────────────────────────────────────────────────────────
blank
say "${BOLD}${CYAN}  autotune${RESET}  —  Private Gateway Setup"
rule
blank

# ── Check for Python ──────────────────────────────────────────────────────────
PYTHON=""
for py in python3 python; do
  if command -v "$py" &>/dev/null \
     && "$py" -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" 2>/dev/null; then
    PYTHON="$py"
    break
  fi
done
[[ -z "$PYTHON" ]] && die "Python 3.9+ is required. Install it from https://python.org and re-run."
ok "Python: $($PYTHON --version)"

# ── Check for autotune ────────────────────────────────────────────────────────
AUTOTUNE_VERSION=""
if command -v autotune &>/dev/null; then
  AUTOTUNE_VERSION=$(autotune --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
  ok "autotune: v${AUTOTUNE_VERSION:-?}"
else
  warn "autotune not found in PATH — install it first: pip install llm-autotune"
fi

# ── Docker check ──────────────────────────────────────────────────────────────
HAS_DOCKER=0
if command -v docker &>/dev/null && docker compose version &>/dev/null 2>&1; then
  HAS_DOCKER=1
  ok "Docker Compose: $(docker compose version --short 2>/dev/null || true)"
else
  warn "Docker / Docker Compose not found — team profile will need it."
fi

blank

# ── Existing .env guard ───────────────────────────────────────────────────────
ENV_FILE="$(pwd)/.env"
if [[ -f "$ENV_FILE" ]]; then
  warn ".env already exists at $ENV_FILE"
  overwrite=$(ask_yn "Overwrite it?" "no")
  if [[ "$overwrite" != "yes" ]]; then
    info "Keeping existing .env — loading it instead."
    # shellcheck disable=SC2046
    export $(grep -v '^#' "$ENV_FILE" | grep '=' | xargs) 2>/dev/null || true
    blank
    say "  Run ${BOLD}autotune serve${RESET} to start with the existing configuration."
    blank
    exit 0
  fi
  blank
fi

# ── Generate admin key ────────────────────────────────────────────────────────
info "Generating a secure AUTOTUNE_ADMIN_KEY (48-char URL-safe token)…"
ADMIN_KEY=$($PYTHON -c "import secrets; print(secrets.token_urlsafe(48))")
ok "Admin key generated (${#ADMIN_KEY} chars)"

# ── API key enforcement ───────────────────────────────────────────────────────
blank
say "  ${BOLD}API Key Enforcement${RESET}"
say "  When enabled, every /v1/* request must include a valid API key."
say "  Recommended for team deployments; optional for personal use."
blank
require_key=$(ask_yn "Require API keys for inference requests?" "yes")
if [[ "$require_key" == "yes" ]]; then
  REQUIRE_KEY_VAL="1"
  ok "API key enforcement: ON"
else
  REQUIRE_KEY_VAL="0"
  warn "API key enforcement: OFF (any client that can reach the port can use your models)"
fi

# ── Port ──────────────────────────────────────────────────────────────────────
blank
AUTOTUNE_PORT=$(ask "Port to expose autotune on" "8765")
ok "Port: $AUTOTUNE_PORT"

# ── Model pull ────────────────────────────────────────────────────────────────
blank
if [[ -z "$PULL_MODEL" ]]; then
  say "  ${BOLD}First model (optional)${RESET}"
  say "  autotune works with any Ollama model. Popular choices:"
  say "    ${DIM}qwen3:8b   llama3.2:3b   gemma3:4b   mistral:7b${RESET}"
  blank
  PULL_MODEL=$(ask "Model to pull now (leave blank to skip)" "")
fi

# ── Write .env ────────────────────────────────────────────────────────────────
blank
info "Writing $ENV_FILE …"

cat > "$ENV_FILE" <<EOF
# autotune environment — generated by ./setup.sh on $(date -u '+%Y-%m-%dT%H:%M:%SZ')
# Keep this file secret and NEVER commit it to version control.

# ── Admin key (dashboard login + admin API) ───────────────────────────────────
# This is your dashboard password and admin API credential.
AUTOTUNE_ADMIN_KEY=${ADMIN_KEY}

# ── API key enforcement ───────────────────────────────────────────────────────
# 1 = every /v1/* request must include Authorization: Bearer <key>
# 0 = open access (personal / trusted-network use only)
AUTOTUNE_REQUIRE_API_KEY=${REQUIRE_KEY_VAL}

# ── Server port ───────────────────────────────────────────────────────────────
AUTOTUNE_PORT=${AUTOTUNE_PORT}

# ── Optional: custom Ollama URL ───────────────────────────────────────────────
# Uncomment when running autotune as a separate container from Ollama.
# AUTOTUNE_OLLAMA_URL=http://ollama:11434

# ── Optional: extra CORS origins ─────────────────────────────────────────────
# Comma-separated list of additional allowed origins beyond localhost.
# Required when your dashboard or API is accessed from a different domain
# (e.g. a team deployment behind a reverse proxy).
# AUTOTUNE_CORS_ORIGINS=https://autotune.example.com

# ── Optional: self-hosted telemetry (Supabase) ────────────────────────────────
# AUTOTUNE_SUPABASE_URL=https://your-project-ref.supabase.co
# AUTOTUNE_SUPABASE_KEY=your-anon-key
EOF

ok ".env written"

# ── .gitignore guard ──────────────────────────────────────────────────────────
GITIGNORE="$(pwd)/.gitignore"
if [[ -f "$GITIGNORE" ]] && grep -q "^\.env$" "$GITIGNORE" 2>/dev/null; then
  ok ".env is already in .gitignore"
elif [[ -f "$GITIGNORE" ]]; then
  printf "\n.env\n" >> "$GITIGNORE"
  ok "Added .env to .gitignore"
fi

# ── Pull model ────────────────────────────────────────────────────────────────
if [[ -n "$PULL_MODEL" ]]; then
  blank
  info "Pulling model: ${PULL_MODEL} …"
  if command -v ollama &>/dev/null; then
    ollama pull "$PULL_MODEL" && ok "Model ready: ${PULL_MODEL}"
  elif [[ $HAS_DOCKER -eq 1 ]]; then
    warn "Ollama not found locally — model will be pulled when Docker containers start."
  else
    warn "Ollama not found — install it from https://ollama.com and run: ollama pull ${PULL_MODEL}"
  fi
fi

# ── Show admin key prominently ────────────────────────────────────────────────
blank
rule
blank
say "  ${BOLD}${YELLOW}⚠  ADMIN KEY — save this before closing the terminal${RESET}"
blank
say "  ${BOLD}${ADMIN_KEY}${RESET}"
blank
say "  ${DIM}Also saved to .env — keep that file out of version control.${RESET}"
blank
rule

# ── Next steps ────────────────────────────────────────────────────────────────
blank
say "  ${BOLD}${GREEN}Setup complete!${RESET}"
blank

if [[ $HAS_DOCKER -eq 1 ]]; then
  say "  ${BOLD}Local (single machine):${RESET}"
  say "    docker compose --profile single up -d"
  blank
  say "  ${BOLD}Team (autotune + Ollama + nginx on port 80):${RESET}"
  say "    docker compose --profile team up -d"
  blank
else
  say "  ${BOLD}Start autotune:${RESET}"
  say "    autotune serve"
  blank
fi

say "  ${BOLD}Open the dashboard:${RESET}"
say "    http://localhost:${AUTOTUNE_PORT}/dashboard"
say "    ${DIM}Password: the key shown above${RESET}"
blank

if [[ "$REQUIRE_KEY_VAL" == "1" ]]; then
  say "  ${BOLD}Create your first API key:${RESET}"
  say "    Sign in → Keys tab → New Key"
  blank
fi

say "  ${BOLD}Test the gateway:${RESET}"
say "    curl http://localhost:${AUTOTUNE_PORT}/health"
blank

if [[ $HAS_DOCKER -eq 1 ]]; then
  say "  ${BOLD}TLS / HTTPS:${RESET}"
  say "    See docs/team-tls.md for a Caddy or Nginx + Let's Encrypt setup."
  blank
fi

# ── Offer to start now ────────────────────────────────────────────────────────
if [[ $NON_INTERACTIVE -eq 0 ]] && command -v autotune &>/dev/null; then
  blank
  start_now=$(ask_yn "Start autotune now?" "yes")
  if [[ "$start_now" == "yes" ]]; then
    blank
    # Load .env into current shell
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    info "Starting autotune on 0.0.0.0:${AUTOTUNE_PORT} …"
    info "Press Ctrl+C to stop."
    blank
    exec autotune serve --host 0.0.0.0 --port "${AUTOTUNE_PORT}"
  fi
fi

blank
