# Secrets rotation runbook

**Status:** REQUIRED — read this end-to-end before touching any console.

This runbook covers rotating every secret IgnisAI uses, in the correct order,
with verification steps. Follow it in full whenever:

- A secret has been pushed to a public branch (even briefly).
- A contributor with secret access leaves the project.
- You suspect any credential has been exposed in logs, screenshots, or screen shares.
- It has been more than 90 days since the last rotation (calendar reminder).

---

## 0. Confirm the blast radius first

Before rotating, audit *where* the leak landed. This drives whether you also
need to file abuse reports and whether scrubbing history is worth the
disruption.

```bash
# From the repo root, search every commit on every branch for shapes of secrets.
# This is a local check; it doesn't make network calls.

# Mapbox secret tokens (sk.*)
git log --all -p -G 'sk\.eyJ[A-Za-z0-9._-]+' -- '*.env*' 'render.yaml' 'docker-compose*.yml' || true

# MongoDB Atlas SRV URIs with embedded creds
git log --all -p -G 'mongodb\+srv://[^:]+:[^@]+@' || true

# JWT secrets (any 32+ char base64-ish string assigned to JWT_SECRET)
git log --all -p -G 'JWT_SECRET\s*=\s*[A-Za-z0-9+/=_-]{16,}' || true

# NASA FIRMS keys (32-char hex)
git log --all -p -G 'NASA_API_KEY\s*=\s*[a-f0-9]{32}' || true
```

Then verify the same on the **public remote**:

```bash
git fetch --all --prune
for ref in $(git for-each-ref --format='%(refname)' refs/remotes/); do
  git log "$ref" --oneline -G 'sk\.eyJ' -- '*.env*' 'render.yaml' || true
done
```

If gitleaks is installed, run it as the authoritative scan:

```bash
gitleaks detect --source . --config .gitleaks.toml --redact --no-banner
gitleaks detect --source . --config .gitleaks.toml --log-opts="--all" --redact --no-banner
```

> **If a secret was pushed to a public branch, treat it as compromised even
> after deletion.** GitHub caches refs, mirrors exist, and crawler bots scrape
> push events within seconds. *Rotate.* Do not skip.

---

## 1. Rotation order

Rotate in this order. The order matters because some services read each
other's credentials and you want each rotation to land cleanly.

1. JWT_SECRET (no upstream provider — rotate first, fastest)
2. NASA FIRMS API key
3. Mapbox **secret** token (`sk.*`) used for tilesets / uploads
4. Mapbox **public** token (`pk.*`) — only if the public token was used in a
   secret context (rare; usually skipped)
5. MongoDB Atlas application user password
6. AWS IAM access keys (S3 static / runtime cache)
7. Render auto-deploy webhook (only if it was leaked)

For each section: **generate → store → deploy → verify → revoke old**.

---

## 2. JWT_SECRET

JWT_SECRET signs auth tokens. Rotating it logs out every active user — that's
the whole point. Schedule rotation when traffic is low.

### 2.1 Generate

```bash
node -e "console.log(require('crypto').randomBytes(64).toString('base64url'))"
```

The `backend/config/jwt.js` loader requires **at least 32 characters**. The
command above returns ~86 characters; that's the floor we want.

### 2.2 Store

Set in three places:

- Render dashboard → `ignisai-backend` → Environment → edit `JWT_SECRET`.
  (`render.yaml` declares `generateValue: true`, so the first deploy
  auto-generated one. Replace it with the value from step 2.1.)
- Local `backend/.env` (your dev box). Update `JWT_SECRET=…`.
- Any CI runners that exercise authenticated endpoints — set the secret in
  GitHub Actions repo secrets if you have an integration job.

### 2.3 Deploy

Render → `ignisai-backend` → Manual deploy → "Deploy latest commit". The
backend will fail-fast at boot if the new secret is shorter than 32 chars —
that's the loader you wired in `backend/config/jwt.js`.

### 2.4 Verify

```bash
# Login should mint a token signed by the new secret.
curl -sS -X POST "$API/auth/login" \
  -H 'content-type: application/json' \
  -d '{"email":"<test-user>","password":"<test-pass>"}' | jq

# A token issued before the rotation should now 401.
curl -sS "$API/firms" -H "authorization: Bearer <OLD_TOKEN>" -i | head -1
# expect: HTTP/1.1 401 Unauthorized
```

### 2.5 Revoke

JWT_SECRET has no provider-side revocation — once the new value is deployed
and the old token failed verification, the old secret is dead.

---

## 3. NASA FIRMS API key

### 3.1 Generate

1. Go to https://firms.modaps.eosdis.nasa.gov/api/area/ (NASA's MAP_KEY page).
2. Sign in with the IgnisAI service account email.
3. "Get a New MAP_KEY" → copy the 32-character hex value.

### 3.2 Store

- Render → `ignisai-backend` → Environment → `NASA_API_KEY`.
- Render → `ignisai-tilesvc` → Environment → `NASA_API_KEY`. (Same value;
  both services hit FIRMS.)
- Local `backend/.env` and `tilesvc/.env` for dev.

> The variable is named **`NASA_API_KEY`** — not `FIRMS_API_KEY`. The codebase
> only reads `NASA_API_KEY`; do not reintroduce the old name. (`render.yaml`
> was previously inconsistent and is now corrected.)

### 3.3 Deploy + Verify

```bash
# Manual deploy ignisai-backend, then:
curl -sS "$API/firms" | jq '.metadata' # should not contain "error" / "no MAP_KEY"

# Tile service:
curl -sS "$TILESVC/healthz" | jq
```

### 3.4 Revoke

Back at the FIRMS MAP_KEY page, click "Deactivate" on the previous key.
Confirm in the table that only the new key is active.

---

## 4. Mapbox secret token (`sk.*`)

Used by ops scripts that upload tilesets, manage styles, or read account data.
**Never embedded in the frontend.** If you ever pasted `sk.*` into a React
`.env` file, that's a leak.

### 4.1 Generate

1. https://account.mapbox.com/access-tokens/
2. "Create a token" → name it `ignisai-tilesvc-2026Qn` (include the quarter so
   future-you knows it's rotated regularly).
3. Scopes: only what you actually need. Common minimum for IgnisAI:
   - `styles:read`
   - `tilesets:read`
   - `tilesets:write` (only if you upload custom tilesets)
   - `uploads:write` and `uploads:read` (only if you use the Uploads API)

   Do **not** grant `downloads:read`, `tokens:write`, `user:write`,
   `user:read`, or `tokens:read` unless a specific script needs them.

### 4.2 Store

- 1Password / Bitwarden (or whatever vault the team uses) — **canonical copy**.
- Render → `ignisai-tilesvc` → if used here.
- Local `tilesvc/.env` if dev scripts call Mapbox APIs.

> The frontend uses `REACT_APP_MAPBOX_TOKEN`, which must be a `pk.*` token,
> not `sk.*`. Anything starting with `REACT_APP_*` is shipped to the browser;
> putting `sk.*` there leaks it on first page load.

### 4.3 Deploy + Verify

```bash
# Confirm scopes on the new token
curl -sS "https://api.mapbox.com/tokens/v2?access_token=<NEW_SK>" | jq

# Confirm the old token returns 401:
curl -sS "https://api.mapbox.com/tokens/v2?access_token=<OLD_SK>" -i | head -1
```

### 4.4 Revoke

Mapbox account → Access tokens → Delete the old `sk.*`. This is permanent
and immediate.

---

## 5. Mapbox public token (`pk.*`)

The public token is fine to ship to the browser **as long as it has token
URL restrictions**. Every public token in the IgnisAI frontend should be
restricted to the production domain(s):

- `https://ignisai-frontend.onrender.com`
- `http://localhost:3000` (only if you keep a separate dev token)

### 5.1 Generate (only if you need to)

Mapbox dashboard → "Create a token" → only `styles:read`, `fonts:read`,
`datasets:read` (and `tilesets:read` if you use private tilesets via signed
URLs). Set the URL restrictions before saving.

### 5.2 Store

- Render → `ignisai-frontend` → `REACT_APP_MAPBOX_TOKEN`.
- Local `frontend/.env`.

### 5.3 Deploy + Verify

Manual deploy `ignisai-frontend`. After it goes live, open DevTools → Network
on the production URL, filter for `mapbox`, and confirm requests carry the
new token. Then confirm requests from a non-allowlisted domain (e.g.
`localhost:5173` if not allowlisted) return 401.

### 5.4 Revoke

Delete the old `pk.*` from Mapbox. Browsers will pick up the new one on next
page load (since it's baked into the React build).

---

## 6. MongoDB Atlas application user

The most disruptive rotation: any pod still using the old credential will
disconnect. Sequence this carefully.

### 6.1 Create the new user **before** revoking the old

1. Atlas → Database Access → "Add New Database User".
2. Auth method: SCRAM (password).
3. Username: `ignisai-app-2026q2` (include the quarter; never reuse names).
4. Password: "Autogenerate Secure Password" → copy.
5. Database User Privileges: **Read and write to any database** is too broad —
   prefer "Specific Privileges" → `readWrite` on the `ignisai` database only.
6. Save.

### 6.2 Build the new SRV URI

```
mongodb+srv://ignisai-app-2026q2:<URL_ENCODED_PASSWORD>@<CLUSTER>.mongodb.net/ignisai?retryWrites=true&w=majority&appName=ignisai-backend
```

URL-encode the password (replace `@`, `:`, `/`, `?`, `#`, `[`, `]` with
percent-encodings):

```bash
node -e "console.log(encodeURIComponent('<RAW_PASSWORD>'))"
```

### 6.3 Store

- Render → `ignisai-backend` → `MONGODB_URI`.
- Local `backend/.env` (yours and any other dev's machines — Slack the team).
- 1Password vault entry for "IgnisAI MongoDB Atlas — current".

### 6.4 Deploy

Render → `ignisai-backend` → Manual deploy. Tail logs:

```
[mongo] connected to ignisai (host=...mongodb.net)
```

If you see `bad auth : Authentication failed`, you have a typo or didn't
URL-encode a special character — fix and redeploy before continuing.

### 6.5 Verify

```bash
curl -sS "$API/health" | jq
# expect: { "status": "ok", "mongo": "connected", ... }

# Smoke a real read:
curl -sS "$API/incidents?limit=1" | jq '.[0]'
```

### 6.6 Revoke

Atlas → Database Access → delete the old `ignisai-app-<old>` user (or
disable it). **Then** confirm the backend is still healthy. If the backend
breaks, you missed updating one of the env files in step 6.3 — re-create the
old user temporarily, fix the env, redeploy, then revoke again.

---

## 7. AWS IAM keys (S3 static + runtime cache)

The tilesvc reads from `IGNIS_STATIC_BUCKET` and writes to
`IGNIS_RUNTIME_CACHE_BUCKET`. These are separate concerns:

- **Static reads** can be public-read or pre-signed; if the IAM user has only
  `s3:GetObject` on the static bucket, the blast radius is small.
- **Runtime cache writes** require `s3:PutObject` on the runtime bucket —
  treat that key as more sensitive.

### 7.1 Generate

AWS Console → IAM → Users → `ignisai-tilesvc` → Security credentials → Create
access key. Use **"Application running outside AWS"** as the use case.
*Download the CSV.* You will not see the secret again.

If you don't have a dedicated IAM user yet, create one with a least-privilege
policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadStatic",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::<IGNIS_STATIC_BUCKET>",
        "arn:aws:s3:::<IGNIS_STATIC_BUCKET>/*"
      ]
    },
    {
      "Sid": "WriteRuntimeCache",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::<IGNIS_RUNTIME_CACHE_BUCKET>",
        "arn:aws:s3:::<IGNIS_RUNTIME_CACHE_BUCKET>/*"
      ]
    }
  ]
}
```

### 7.2 Store

- Render → `ignisai-tilesvc` → `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.
- Local `tilesvc/.env` if you hit S3 from dev.

### 7.3 Deploy + Verify

```bash
# After tilesvc redeploy:
curl -sS "$TILESVC/healthz" | jq '.s3'
# expect: { "static": "ok", "runtime_cache": "ok" }
```

### 7.4 Revoke

AWS Console → IAM → `ignisai-tilesvc` → Security credentials → Delete (not
just deactivate) the previous access key.

---

## 8. After rotation — close the loop

1. **Force-update local clones.** Slack the team:
   > "Rotated MONGODB_URI / NASA_API_KEY / etc. Pull new values from 1Password,
   > update your `backend/.env` and `tilesvc/.env`, restart your dev servers."

2. **Push protection.** Enable GitHub → Settings → Code security → "Push
   protection for secrets". This blocks the next `git push` that contains a
   recognized secret shape, *before* it leaves the developer's box.

3. **Run gitleaks in CI.** Add a workflow step (see `.github/workflows/`) so
   every PR is scanned. The custom rules in `.gitleaks.toml` already cover
   the IgnisAI-specific shapes.

4. **Delete the leaky branches.** Once you're sure no work is salvageable
   from them:

   ```bash
   bash tools/cleanup-remote-branches.sh
   ```

   The script lists candidate branches, prompts for confirmation per branch,
   and only deletes the ones you say yes to. Keep `main` and any active
   feature branches.

5. **Document the incident.** Add a one-paragraph entry to `docs/runbook.md`
   under a "Security incidents" heading: date, what leaked, how it was
   detected, what was rotated. Future-you will thank present-you.

6. **Calendar a re-rotation.** Even with no incident, rotate JWT_SECRET,
   Mapbox `sk.*`, and AWS keys quarterly. Set a 90-day reminder.

---

## 9. Quick-reference: where each secret lives

| Secret | Render service(s) | Local file | Provider |
|---|---|---|---|
| `JWT_SECRET` | `ignisai-backend` | `backend/.env` | self (random) |
| `NASA_API_KEY` | `ignisai-backend`, `ignisai-tilesvc` | `backend/.env`, `tilesvc/.env` | NASA FIRMS |
| `MAPBOX_SECRET_TOKEN` (sk.*) | `ignisai-tilesvc` (only if used) | `tilesvc/.env` (only if used) | Mapbox |
| `REACT_APP_MAPBOX_TOKEN` (pk.*) | `ignisai-frontend` | `frontend/.env` | Mapbox |
| `MONGODB_URI` | `ignisai-backend` | `backend/.env` | MongoDB Atlas |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | `ignisai-tilesvc` | `tilesvc/.env` | AWS IAM |
| `CORS_ORIGIN` | `ignisai-backend`, `ignisai-tilesvc` | `backend/.env` | self (URL) |

The canonical source for **every secret value** is the team password vault
(1Password / Bitwarden). Render env vars and local `.env` files are caches.
If they ever disagree, the vault wins.
