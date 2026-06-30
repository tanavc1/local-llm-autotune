# Team TLS Setup

autotune's team profile ships with nginx on port 80 (HTTP). Two options to enable HTTPS:

- **Option A — Caddy** (easiest, auto-renews certs)
- **Option B — nginx + Let's Encrypt** (more control, same container)

---

## Prerequisites

- A public domain pointing to your server (`autotune.example.com`)
- Ports 80 and 443 open in your firewall / security group
- autotune running: `docker compose --profile team up -d`

---

## Option A — Caddy (recommended)

Caddy handles cert issuance and renewal automatically via ACME/Let's Encrypt.

### 1. Add Caddy to docker-compose.yml

Replace the `nginx` service block with:

```yaml
  caddy:
    profiles: ["team"]
    image: caddy:2-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    depends_on:
      autotune-team:
        condition: service_healthy
    restart: unless-stopped
```

Add to the `volumes:` section at the bottom:

```yaml
  caddy_data:
  caddy_config:
```

### 2. Create `nginx/Caddyfile`

```
autotune.example.com {
    reverse_proxy autotune-team:8765 {
        flush_interval -1          # disable buffering for streaming tokens
        transport http {
            read_timeout  600s
            write_timeout 600s
        }
    }
    header {
        X-Content-Type-Options  nosniff
        X-Frame-Options         SAMEORIGIN
        Referrer-Policy         strict-origin
        -Server
    }
}
```

Replace `autotune.example.com` with your actual domain.

### 3. Start

```bash
docker compose --profile team up -d
```

Caddy fetches a cert on first boot. Check logs with `docker compose logs caddy`.

---

## Option B — nginx + Let's Encrypt (certbot)

### 1. Obtain a certificate with certbot (standalone mode)

Stop nginx temporarily so certbot can bind port 80:

```bash
docker compose --profile team stop nginx
```

Run certbot:

```bash
certbot certonly --standalone \
  --email admin@example.com \
  --agree-tos \
  --no-eff-email \
  -d autotune.example.com
```

Certificates are written to `/etc/letsencrypt/live/autotune.example.com/`.

### 2. Copy certs into nginx/certs/

```bash
cp /etc/letsencrypt/live/autotune.example.com/fullchain.pem nginx/certs/
cp /etc/letsencrypt/live/autotune.example.com/privkey.pem   nginx/certs/
```

### 3. Enable the HTTPS block in `nginx/autotune.conf`

Uncomment the `server { listen 443 … }` block at the bottom of the file.
Set `server_name` to your domain. The HTTP block will automatically redirect to HTTPS.

### 4. Update docker-compose.yml nginx service

Make sure the certs volume is mounted (it already is in the default config):

```yaml
volumes:
  - ./nginx/autotune.conf:/etc/nginx/conf.d/default.conf:ro
  - ./nginx/certs:/etc/nginx/certs:ro
```

### 5. Restart nginx

```bash
docker compose --profile team up -d nginx
```

### 6. Auto-renew with cron

```
0 3 * * * certbot renew --quiet && \
    cp /etc/letsencrypt/live/autotune.example.com/fullchain.pem /path/to/autotune/nginx/certs/ && \
    cp /etc/letsencrypt/live/autotune.example.com/privkey.pem   /path/to/autotune/nginx/certs/ && \
    docker compose -f /path/to/autotune/docker-compose.yml exec nginx nginx -s reload
```

---

## Verify TLS is working

```bash
curl -I https://autotune.example.com/health
# Expect: HTTP/2 200
```

Check the certificate grade at [SSL Labs](https://www.ssllabs.com/ssltest/).
