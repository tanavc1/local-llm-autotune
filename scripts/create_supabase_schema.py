#!/usr/bin/env python3
"""
Apply the autotune telemetry schema to Supabase via the Management API.

No psql or Supabase CLI required — uses httpx (already a project dependency)
and Supabase's SQL query endpoint.

Usage
-----
    # Pass your personal access token directly:
    python scripts/create_supabase_schema.py --pat sbp_xxxxxxxxxxxx

    # Or set it in the environment:
    export SUPABASE_PAT=sbp_xxxxxxxxxxxx
    python scripts/create_supabase_schema.py

Personal access tokens live at:
    https://supabase.com/dashboard/account/tokens
They are separate from the project anon/service_role keys.

The project ref is auto-derived from AUTOTUNE_SUPABASE_URL but can be
overridden with --ref.

All DDL statements are idempotent (IF NOT EXISTS / ON CONFLICT DO NOTHING)
so re-running this script on an already-configured database is safe.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_REF = "gmsibgsdedyrbiucaitv"
_MGMT_API    = "https://api.supabase.com"
_SCHEMA_FILE = Path(__file__).parent.parent / "autotune" / "telemetry" / "schema.sql"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_from_url(url: str) -> str:
    """Extract the project ref from a supabase.co URL."""
    m = re.search(r"https://([a-z0-9]+)\.supabase\.co", url)
    return m.group(1) if m else _DEFAULT_REF


def _run_sql(ref: str, pat: str, sql: str) -> None:
    """
    POST sql to the Supabase Management API query endpoint.

    Splits on statement boundaries so each statement is sent individually —
    the Management API doesn't support multi-statement batches.
    """
    try:
        import httpx
    except ImportError:
        print("ERROR: httpx not installed.  Run: pip install httpx", file=sys.stderr)
        sys.exit(1)

    url     = f"{_MGMT_API}/v1/projects/{ref}/database/query"
    headers = {
        "Authorization": f"Bearer {pat}",
        "Content-Type":  "application/json",
    }

    # Split into individual statements; skip blank / comment-only chunks
    statements = _split_statements(sql)
    print(f"Applying {len(statements)} SQL statement(s) to project {ref!r} …\n")

    errors = 0
    for i, stmt in enumerate(statements, 1):
        preview = stmt[:72].replace("\n", " ").strip()
        try:
            resp = httpx.post(url, json={"query": stmt}, headers=headers, timeout=30)
        except httpx.RequestError as exc:
            print(f"  [{i:>3}] NETWORK ERROR: {exc}")
            errors += 1
            continue

        if resp.status_code in (200, 201):
            print(f"  [{i:>3}] OK   {preview}")
        else:
            raw  = resp.text[:400]
            try:
                body = resp.json()
                if isinstance(body, list):
                    msg = str(body[0]) if body else raw
                else:
                    msg = body.get("message") or body.get("error") or raw
            except Exception:
                msg = raw
            # Tolerate "already exists" — schema is idempotent but the API
            # returns 400 for some IF NOT EXISTS constructs on older PG versions.
            if "already exists" in msg.lower():
                print(f"  [{i:>3}] SKIP {preview}  (already exists)")
            else:
                print(f"  [{i:>3}] FAIL {preview}")
                print(f"         {resp.status_code}: {msg}")
                errors += 1

    print()
    if errors:
        print(f"Finished with {errors} error(s). Check output above.", file=sys.stderr)
        sys.exit(1)
    else:
        print("Schema applied successfully.")


def _split_statements(sql: str) -> list[str]:
    """
    Split a SQL script into individual statements.

    Handles dollar-quoted strings (used in the plpgsql function) so the
    $$ … $$ block isn't split on the semicolons inside it.
    """
    statements: list[str] = []
    current: list[str] = []
    in_dollar_quote = False
    dollar_tag = ""

    for line in sql.splitlines():
        stripped = line.strip()

        # Toggle dollar-quote state
        if not in_dollar_quote:
            # Look for opening $$ or $tag$
            m = re.search(r"\$([^$]*)\$", line)
            if m and "$$" in line or (m and m.group(0) in line):
                # Count occurrences: odd → entering, even → not changing
                tags = re.findall(r"\$[^$]*\$", line)
                if tags and len(tags) % 2 == 1:
                    in_dollar_quote = True
                    dollar_tag = tags[0]
        else:
            if dollar_tag in line:
                in_dollar_quote = False

        current.append(line)

        # A semicolon at end of line (outside dollar-quote) ends a statement
        if not in_dollar_quote and stripped.endswith(";"):
            stmt = "\n".join(current).strip()
            if stmt and not _is_comment_only(stmt):
                statements.append(stmt)
            current = []

    # Flush anything left (shouldn't happen in well-formed SQL)
    remainder = "\n".join(current).strip()
    if remainder and not _is_comment_only(remainder):
        statements.append(remainder)

    return statements


def _is_comment_only(s: str) -> bool:
    """Return True if the string contains only comments and whitespace."""
    cleaned = re.sub(r"--[^\n]*", "", s)       # strip line comments
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    return not cleaned.strip()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pat",
        default=os.environ.get("SUPABASE_PAT", ""),
        metavar="TOKEN",
        help="Supabase personal access token (or set $SUPABASE_PAT)",
    )
    parser.add_argument(
        "--ref",
        default=_DEFAULT_REF,
        metavar="PROJECT_REF",
        help=f"Supabase project ref (default: {_DEFAULT_REF})",
    )
    parser.add_argument(
        "--schema",
        default=str(_SCHEMA_FILE),
        metavar="PATH",
        help=f"Path to schema SQL file (default: {_SCHEMA_FILE})",
    )
    args = parser.parse_args()

    pat = args.pat.strip()
    if not pat:
        print(
            "ERROR: supply a personal access token via --pat or $SUPABASE_PAT\n"
            "       Get one at: https://supabase.com/dashboard/account/tokens",
            file=sys.stderr,
        )
        sys.exit(1)

    schema_path = Path(args.schema)
    if not schema_path.exists():
        print(f"ERROR: schema file not found: {schema_path}", file=sys.stderr)
        sys.exit(1)

    sql = schema_path.read_text()
    _run_sql(args.ref, pat, sql)


if __name__ == "__main__":
    main()
