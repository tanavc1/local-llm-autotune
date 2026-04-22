# Security Policy

## Telemetry & Data Collection

### Opt-In Only — Off by Default

autotune includes **optional, anonymous** usage telemetry that is **disabled by default**. No data is sent to any external service until you explicitly opt in:

```bash
autotune telemetry --enable    # opt in
autotune telemetry --disable   # opt out at any time
autotune telemetry --status    # check current status
```

### What Is Collected (When Opted In)

| Collected | NOT Collected |
|-----------|---------------|
| CPU architecture (e.g. `arm64`) | Hostnames |
| RAM size (e.g. `16 GB`) | Usernames |
| GPU backend (e.g. `Metal`) | IP addresses |
| Model names (e.g. `qwen3:8b`) | File paths |
| Inference speed (tok/s, TTFT) | Conversation content |
| Context window sizes | Prompt or response text |
| Quantization labels | Any personally identifiable information |

### Supabase Anon Key in Source Code

You may notice a Supabase API key embedded in [`autotune/telemetry/client.py`](autotune/telemetry/client.py). This is the **Supabase anon (public) key** — the same type of JWT token that Supabase [officially recommends](https://supabase.com/docs/guides/api#api-keys) for client applications.

**Why this is safe:**

1. **It is NOT a service_role key, database password, or admin credential.** It is a public client token designed to be embedded in applications.

2. **Row Level Security (RLS) restricts access.** The anon key can only:
   - `INSERT` rows into the three telemetry tables
   - It **cannot** `SELECT`, `UPDATE`, or `DELETE` any data

3. **The consent gate is enforced in code.** Even though the key exists, no data flows unless [`is_opted_in()`](autotune/telemetry/consent.py) returns `True`. This check is enforced at every call site.

4. **You can override it.** Set these environment variables to point at your own Supabase project:
   ```bash
   export AUTOTUNE_SUPABASE_URL="https://your-project.supabase.co"
   export AUTOTUNE_SUPABASE_KEY="your-anon-key"
   ```

### Local Data Storage

All local data is stored in standard platform-specific directories:

| Platform | Path | Contents |
|----------|------|----------|
| macOS | `~/Library/Application Support/autotune/` | Performance telemetry DB, consent file |
| Linux | `~/.local/share/autotune/` | Same |
| All | `~/.autotune/recall.db` | Conversation memory (local only, never sent externally) |

Conversation content in `recall.db` is **never** transmitted to any external service, regardless of telemetry opt-in status.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT open a public GitHub issue.**
2. Email: [security contact — add your email here]
3. Include a description of the vulnerability and steps to reproduce.

We will acknowledge receipt within 48 hours and provide a fix timeline.

## Dependency Security

autotune's dependencies are standard, well-maintained packages:

- **FastAPI** / **Uvicorn** — ASGI web framework (local server only, binds to `localhost`)
- **psutil** — System monitoring (read-only)
- **httpx** — HTTP client for Ollama / Supabase communication
- **click** — CLI framework
- **numpy** — Vector similarity for conversation memory

The inference server binds to `127.0.0.1` by default and is not exposed to the network.
