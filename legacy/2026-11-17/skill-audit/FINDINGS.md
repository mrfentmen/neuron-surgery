# Skill audit + application pentest — 2026-11-13

## 1. Skill scan

Tooling: NVIDIA **SkillSpector v2.12.0** (static, `--no-llm`, 71 rule classes)
plus a local scanner (`scan_skills.py`) tuned for prompt injection, credential
access, exfiltration, obfuscation, persistence and hidden unicode.

### Installed skills (7)

| skill | verdict | notes |
|---|---|---|
| anti-ai-slop-ui | clean | docs + a UI lint helper |
| nnsight-remote-interpretability | clean | docs only |
| skill-authoring | clean | docs only |
| skill-creator | clean | static score 100 from `subprocess` launching `claude -p` and eval file writes — that is the skill's documented job. No network exfil, no credential reads |
| skill-creator-openai | clean | RA1 hit is `init_skill.py` writing the new skill's files, not self-modification |
| sparse-autoencoder-training | clean | docs only |
| transformer-lens-interpretability | clean | docs only |

Only local-scanner hit was a false positive: a docstring in
`skill-creator/scripts/improve_description.py` that mentions `ANTHROPIC_API_KEY`
while explaining it is **not** needed.

### Pulled repos

| repo | verdict | notes |
|---|---|---|
| NVIDIA/SkillSpector | SAFE (0/100) | the scanner itself; its own `skill-inspector` skill scans clean |
| cloudflare/security-audit-skill | **usable, installed** | 86/100 is driven by audit *questions* (`.env`, keychain, system prompt) in ATTACK-CLASSES.md and by `execFileSync` in its own validators. No network code, no exfil, no hidden instructions |
| hypnguyen1209/offensive-claude | **quarantined** | legitimate dual-use offensive toolkit; ships Python jailbreak/model-extraction harnesses (`multiturn_jailbreak.py`, `promptinject_harness.py`, `model_extractor.py`). Not needed for a web audit |
| SnailSploit/Claude-Red | **quarantined** | 79 skills, all `.md` (no executables). Content is offensive command examples (curl uploads, `env | grep SECRET`), not malware. Teaching material, not installed |

Quarantined repos live only under `neuron-surgery/skill-audit/repos/` and are not
under `.agents/skills/`, so they are never loaded as skills.

Cloudflare's `security-audit` skill was installed and its checklist
(`ATTACK-CLASSES.md`, `WEB-PROTOCOL-AND-AUTH.md`, `CLIENT-SIDE.md`) drove the
manual review below.

## 2. Application pentest — watchpeoplefight

### VULN-01 — Broken access control / impersonation in comment edit + delete (patched)

**Severity: high.** `PATCH /api/comments/:id` and `DELETE /api/comments/:id`
proved ownership with `nameClaimed(name, ip)` — "this IP has posted under this
handle at some point" — instead of "this IP posted **this comment**".

Exploit: post any comment as `BigFan` from your own IP (names are free-form),
which records a claim on `bigfan`. Then edit or delete every comment by BigFan.
The same hole existed in the Express mirror (`server.js`).

Fix:
- `functions/_shared/db.js`: new `commentClaimed(env, commentId, ip)` matching
  `wpf_name_claims.comment_id` + IP + 30-day window.
- `functions/api/comments/[id].js`: gate switched to `commentClaimed`.
- `server.js`: matching `commentClaimed(nameLc, ip, commentId)`; `recordNameClaim`
  now keeps the list of comment ids per (name, ip) instead of latest-only.
- `test/cf-port.test.mjs`: regression test — an impostor IP holding a `bigfan`
  claim still gets 403 on another user's BigFan comment (edit **and** delete).

### Reviewed and found sound

- **Admin auth**: every `/api/admin/*` route calls `verifySession`; fail-closed
  HMAC key when `ADMIN_PASSWORD` is unset; login is Turnstile-gated, 5/min +
  20/hour per IP, constant-time compare, HttpOnly/SameSite=Strict cookie.
- **SQL**: no user input reaches an interpolated identifier. The three
  `SELECT ... FROM ${table}` sites use compile-time constant names.
- **XSS**: server-rendered HTML escapes through `esc()` (302 uses); the watch
  page escapes names/bodies before the mention/highlight markup, and the
  highlight runs on already-escaped text.
- **SSRF**: `/api/proxy` allowlists four Reddit media hosts, requires https,
  rejects userinfo, and re-validates **every** redirect hop.
- **CSV injection**: export cells beginning `= + - @ tab CR` get an apostrophe.
- **Consent gate**: server-issued `wpf_consent` cookie; EU/UK records without it
  are dropped by `/api/track`, not by the page.
- **ClipRaiders wall**: fail-closed when `CLIPRAIDER_PASSWORD` is unset,
  constant-time compare, HMAC cookie, 5/min per-IP attempt cap. `DEV_AUTO_LOGIN`
  is absent from `wrangler.toml` `[vars]`, so production cannot auto-login.

## 3. Second pass — resource integrity and dependency health

### VULN-02 — Anonymous `/api/health` could exhaust D1 reads (patched)

**Severity: medium (availability/cost).** Every anonymously reachable poll runs
three D1 read queries, one write and a 24h row scan. With a 240/min global GET
cap and no per-IP limit, one client could issue ~57k queries/hour against a 5M
reads/day free tier.

Fix: per-IP 60/min wall in `functions/api/health.js` before any DB work. Real
monitors poll about once a minute; the harness polls a handful of times.

### VULN-03 — Engagement inflation across comments (patched)

**Severity: medium (data integrity).** Likes and reactions dedupe per
comment+IP+24h, but nothing capped how many DIFFERENT comments one IP could
like. The counts feed the Most Talked ranking and the public like totals.

Fix: per-IP 60/min walls in `comments/like.js` (`lkw:`) and `comments/react.js`
(`rxw:`). Normal browsing is unaffected.

### VULN-04 — ClipRaiders CSRF guard accepted an http origin (patched)

`sameOriginPost` compared `host` only, so `http://clipraider.com` passed against
the https deployment (and a different port was treated as same-origin). Now
compares the full origin. Covered by the existing cross-origin review test.

### VULN-05 — SSRF denylist bypassed by a trailing DNS root label (patched)

**Severity: low/medium (defense-in-depth).** `isLiteralPrivateHost` compared
the raw hostname against `localhost` / `*.local` and the IP ranges. A single
trailing dot is the DNS root label — `localhost.` and `router.local.` resolve
exactly like their undotted forms — but the string comparison failed, so
`https://localhost./…` passed the canonical guard (verified:
`isLiteralPrivateHost("localhost.") === false`).

No live exploit path today: every caller (`/api/proxy`, video-proxy,
warm-worker, `validPushEndpoint`) layers an exact-host allowlist on top, and
those compare with `Set.has` (fail-closed). The gap mattered for any future
caller that relies on the literal denylist alone.

Fix: `functions/_shared/ssrf.js` strips trailing dots before the name and IP
checks (also catches `127.0.0.1.` and `169.254.169.254.`). Regression cases
added to `test/cf-port.test.mjs` for hosts and URLs, plus an
`https://example.com./x` must-stay-allowed case so the normalisation can't
over-block.

### Dependency health

`npm audit`: watchpeoplefight `qs` moderate → **0**; clipraider `hono` ×2 +
`qs` moderate → **0**. Both test suites imported `esbuild` without declaring
it; it is now a devDependency (a clean install previously left the suites
unrunnable).

### Residual risk (not patched, by design)

- ClipRaiders' wall limiter is per-isolate in memory; a distributed brute force
  sees a higher effective rate. It is a soft wall in front of a session-gated
  app, as documented in the source.
- `name_claims` has no index on `comment_id`; the new exact-row lookup scans a
  table bounded by the 30-day window. Add an index if claim volume grows.

## 4. Third pass — consent enforcement, error disclosure, secret comparison

### VULN-06 — `/api/track` v1 events skipped the EU/UK consent gate (patched)

**Severity: medium (privacy-control bypass).** The server-side gate
(`needsConsent(countryOf(request)) && !hasConsent(request)` → drop) wrapped
only the v2 page-record branch. A v1 body (`{event:"page_view",…}`) fell
through to `recordEvent` and wrote a row for an EU/UK visitor who had never
accepted the box. The current client sends v2 only, but a cached old client —
the reason the v1 shape was kept — still hit the hole.

Fix: the same gate now runs before the v1 event branch in
`functions/api/track.js`, returning 204 with `X-Consent-Required: 1`.

### VULN-07 — `/api/hero` returned raw handler errors (patched)

**Severity: low (information disclosure).** The catch block returned
`String(e.message || e)` to anonymous callers, exposing upstream/handler
detail. It now logs server-side and returns `{ error: "hero unavailable" }`,
matching every other 500 on the site.

### VULN-08 — WARM_KEY guards compared secrets with `!==` (patched)

**Severity: low (timing side channel).** `/api/push/digest`,
`/api/push/trending`, `/api/push/newfights`, `/api/cron/indexing` and
`/api/reddit/token` compared the `x-warm-key` header with plain string
inequality, which exits at the first differing byte. All five now use the
shared `sameSecret()` (equal length, XOR-accumulated) exported from
`_shared/auth.js`.

### VULN-10 — v2 play/search writers skipped the EU/UK consent gate (patched)

**Severity: medium (privacy-control bypass, wider than VULN-06).** The gate
lived only in `/api/track`, so `/api/views` (`writePlay`) and `/api/search`
(`writeSearch`) wrote rows carrying the visitor id and IP for EU/UK visitors
who had never accepted the box. `/api/track` also enforced the gate itself
rather than the writer, so any future caller inherited the hole.

Fix: `writePage`, `writePlay` and `writeSearch` in
`functions/_shared/events-v2.js` now check the country + `wpf_consent` cookie
centrally (`consentBlocked`). The `/api/track` checks stay as the early
exit that sets `X-Consent-Required`.

### VULN-11 — anonymous `/api/ad/stats` polling multiplied D1 reads (patched)

**Severity: low (availability/cost).** The endpoint runs two D1 reads per
call and had no per-IP wall; the global GET cap (240/min) still allowed
~690k row reads/day from a single address. Real pages fetch it once or
twice on load. Added a 60/min per-IP wall (`ads:` key) before any DB work.

### VULN-09 — media URL scheme was not validated before rendering (patched)

**Severity: low (defense in depth).** `playableUrl()` passed any non-WSH,
non-root-relative `card.videoUrl` through to `<video src>` / `<iframe src>` /
`og:video`. HTML escaping blocks attribute breakout but not a hostile scheme
supplied by an upstream feed. It now returns "" unless the value is http(s)
(or root-relative), so no player markup is emitted for `javascript:`/`data:`.

## 5. Fourth pass — error disclosure, bounds and secret diagnostics

### VULN-12 — `/api/proxy` returned raw upstream/handler errors (patched)

**Severity: low (information disclosure).** The catch-all on the media proxy
answered 502 with `String(e.message || e)`, exposing Worker internals and
upstream failure text to anonymous callers. It now logs the detail server-side
and returns `{ error: "upstream unavailable" }` — the same shape as every other
5xx on the site.

### VULN-13 — companion `audioUrl` bypassed the media scheme rule (patched)

**Severity: low (defense in depth).** VULN-09 fixed `playableUrl()`, but the
Reddit companion track in `audioHtml()` still emitted `<audio src="…">` for any
feed-supplied `audioUrl`, `javascript:`/`data:` included. It now requires
http(s) (or protocol-relative) and renders nothing otherwise.
Regression in `test/cf-port.test.mjs`: a `javascript:` track produces no
`<audio>` element, an `https` track still produces the synced player.

### VULN-14 — admin submissions list accepted a negative `limit` (patched)

**Severity: low (resource exhaustion).** `Math.min(Number(limit) || 50, 200)`
clamps only the top end; SQLite treats a negative LIMIT as *no limit*, so an
authenticated admin call could pull the whole submissions table in one request.
Both ends are now clamped (`Math.max(1, …)`).

### VULN-15 — `/api/cron/indexing` echoed the pool failure (patched)

**Severity: low (information disclosure).** The 200-with-`ok:false` error body
carried `e.message` from the D1/feed pool. Logged now; caller gets
`{ ok:false, error:"pool unavailable" }`.

### VULN-16 — cron/purge guards leaked a SHA-256 of their secret (patched)

**Severity: medium (credential oracle).** The video-proxy purge guard and the
warm worker's auth guard answered 401/403 with `storedHash: sha256(boundKey)`.
Anyone able to probe those URLs could test candidate keys offline against the
hash (and `keyBound` confirmed whether a secret was configured). Both bodies now
return only `{ ok:false, error:"unauthorized" }`. The purge guard, the warm
worker and `rollup-worker` also compare the secret with the shared
constant-time `sameSecret()` (equal length, XOR-accumulated, no early exit)
instead of `!==`; the purge path's 500 no longer returns `e.message` either.

### Reviewed and found sound (fourth pass)

- Every `/api/admin/*` handler, including submissions, export, comments, stats,
  hide-fight and cache-state, calls `verifySession` before touching D1.
- `.public-stamped` ships no source maps, `.env` files or admin JS beyond the
generated `admin-js.js`, which contains no secrets.
- The service worker caches only same-origin GETs and never caches `/api/`
  responses; no cache-poisoning path from a cross-origin redirect.
- Feed source loaders stay inside their per-source allowlists; the push
  endpoint validator rejects non-FCM/APNs hosts, http and non-URL input.
- State-changing routes (comments, claims, likes, reactions, submissions, ads,
  push, track) all carry a per-IP wall before DB work.

## 6. Fifth pass — Host-header self-fetch, sanitizer drift, unbounded reads

### VULN-17 — `/api/health.svg` aimed a server-side fetch at the Host header (patched)

**Severity: low (blind SSRF / request cannon).** The badge's feed-source check
was a self-fetch of `/api/status` built from `new URL(request.url).origin`.
`request.url` is derived from the client's Host header, so a forged host made
the Worker fetch an arbitrary origin, holding the request open for up to 8s.
Only two booleans of the result were reflected, so it was blind, but it was
still an outbound request an attacker chose. The probe now runs only when the
request host is one this site is actually served from (watchpeoplefight.com,
the www host, the pages.dev alias, or `SITE_ORIGIN`); anything else skips the
probe and reports DB health only. Regression tests assert a forged Host and
the harness localhost never trigger a fetch, while the canonical host still
does.

### VULN-18 — comment edits skipped the create-path sanitizer (patched)

**Severity: low (stored-content validation bypass).** `POST /api/comments`
strips HTML tags, control characters and whitespace runs; `PATCH
/api/comments/:id` only called `.trim()`. An owner could therefore store
`<b>…</b>`, NUL bytes and newline runs in a row that every renderer, RSS
snippet, mention index and admin view treats as plain text. Escaping still
prevented script execution, so this was not XSS — it was the stored-content
contract being bypassable. Both paths now import one `sanitizeText()` from
`functions/_shared/text.js`, and the edit test asserts the public list serves
the stripped body.

### VULN-19 — `/status.xml` read the whole uptime table on every request (patched)

**Severity: low (availability / cost).** `SELECT ts, ok FROM wpf_uptime_events
ORDER BY ts ASC` had no bound. `/api/health` writes one row per minute, so the
public feed's read grew ~525k rows/year and the Worker buffered all of it to
derive transitions. Now `WHERE ts >= ?` (30 days) `LIMIT 20000` — far more than
the feed's 50-item cap can render. Regression test seeds a 40-day-old outage
and asserts it is neither read nor emitted.

### Reviewed and found sound (fifth pass)

- Stored card fields (titles/descriptions/authors from WSH, Reddit, YouTube,
  Dailymotion) reach HTML only through `esc()`, and every JSON-LD block is
  `JSON.stringify(...).replace(/</g, "\\u003c")` — no `</script>` breakout.
- `/api/push/unsubscribe` deletes by full endpoint capability only; the delete
  is idempotent and cannot enumerate or target another subscriber.
- `/api/oembed` reflects the request host into `html`/`author_url`, but only
  for a URL the caller already supplied and only as JSON — no cookie, secret
  or cross-user data crosses that boundary (noted, not patched).
- `api/status.js` probes WSH from a constant URL, not request input; the new
  badge guard only removes the extra hop.
- Admin auth, cookie flags, CSP (`script-src-attr 'none'`), CORS pinning and
  the JSON-only/Origin/Content-Length POST gate were re-read and unchanged.

## 7. Sixth pass — proxied document types, attribute escaping, worker key compare

### VULN-20 — `/api/proxy` forwarded the upstream Content-Type verbatim (patched)

**Severity: medium-low (same-origin document injection).** The Reddit media
proxy streams a response from `v.redd.it` / `i.redd.it` on OUR origin. It
copied `Content-Type` straight from the upstream. `X-Content-Type-Options:
nosniff` stops MIME guessing but not an explicitly declared type, so an
upstream `text/html` (or `image/svg+xml`) body would have been rendered as a
scriptable document on watchpeoplefight.com. The allowlist now passes only
`video/*`, `audio/*` and non-SVG `image/*`; anything else is
`application/octet-stream`. Regression tests stub an upstream `text/html` and
`image/svg+xml` and assert both clamp.

### VULN-21 — `/embed` interpolated card ids into HTML attributes unescaped (patched)

**Severity: low (defense in depth, stored XSS if an upstream id ever gains a quote).**
`renderEmbedPlayground` wrote `${id}` into `id="emb-code-…"` and
`data-copy="emb-code-…"` with no escaping — the one place an id is not
escaped (every text node uses `esc()`). Ids are safe today by construction
(source-prefixed slugs/uuids), so this was a latent breakout, not a live
exploit. Both attributes now use `esc(id)`; a test renders a card whose id is
`x" onmouseover="alert(1)` and asserts the quote is entity-escaped.

### VULN-22 — warm worker compared its manual-trigger key with `===` (patched)

**Severity: low (timing side channel).** `sent === bound` exits at the first
differing byte, so an anonymous caller could narrow the `WARM_KEY` one
character at a time from response timing. The same class was fixed in the
Pages routes and the purge/rollup guards in earlier passes; the warm worker's
handler was missed because it lives in its own bundle. It now uses a local
XOR-accumulated `sameSecret()` (same contract as `_shared/auth.js`, kept local
to stay dependency-free like the video-proxy and rollup gates).

### Reviewed and found sound (sixth pass)

- `functions/api/admin/analytics.js`: `days` clamps to 1–180; the `jx()` JSON
  helper is only ever called with literal column/path pairs (no interpolation
  of request input); every other query binds parameters.
- `functions/_shared/db.js`: the only dynamic SQL pieces are whitelisted
  column names (`clicks`/`impressions`, `react_<kind>`) and `AND`-joined
  constant clauses with bound params.
- `functions/api/admin/cache-state.js`, `purge-video.js`, `purge.js`,
  `video-proxy`: probe/purge targets are always the configured `VIDEO_BASE`
  and the feed URL is re-validated against the WSH host; redirects manual.
- `functions/api/comments/like.js` + `react.js`: id pattern, per-IP wall before
  D1, whitelisted reaction kinds.
- `/api/push/digest` + `/api/cron/*`: `sameSecret()` guards, unset secret →
  503 (feature off), generic error bodies.
- Hub/search/embed renderers: every text node and meta attribute goes through
  `esc()`; JSON-LD blocks escape `<`; the only unescaped attribute found was
  VULN-21 above.

## 8. Seventh pass — the mention backfill was per-request

### VULN-23 — `/api/mentions` ran a full `wpf_comments` scan on every poll (patched)

**Severity: low-medium (availability / cost amplification).** The pull-back
inbox calls `sweepLegacyMentions()` before its own indexed lookup, to backfill
comments stored before the write-time index shipped. That sweep is
`WHERE (body LIKE '%@%' OR reply_to IS NOT NULL) AND NOT EXISTS (SELECT 1 FROM
wpf_comment_mentions …)` — an anti-join with no covering index, so proving
"nothing left to index" walks the entire comment table. It ran on EVERY
anonymous GET. The 30/min per-IP wall bounds the request rate, not the cost
per request: 30 full scans a minute per IP, for free, from one curl loop.

The sweep is now amortized per isolate — at most one run every 5 minutes,
with the timestamp claimed before the `await` so concurrent requests in the
same isolate don't all pay for it. Backfill is idempotent and best-effort, so
the only visible change is that a pre-index comment may take up to five
minutes to appear instead of one poll. Regression test: first poll backfills
and returns the legacy mention; a second poll inside the window does not
re-scan (a second legacy row inserted after the window opened stays unindexed
— so the assertion proves the scan was skipped, not that it found nothing).

### Reviewed and found sound (seventh pass)

- `functions/api/track.js`: consent grant is the only path that mints
  `wpf_consent` (HttpOnly, server-set); the EU/UK gate runs before BOTH the v2
  page write and the v1 event branch; JSON content-type, 8 KB declared-length
  cap, same-host Origin check, DNT honoured again server-side.
- `public/_headers` vs `functions/_middleware.js`: CSP byte-identical (the
  parity test enforces it), ACAO pinned to the custom domain on static assets
  so Pages' default `*` never ships.
- `_shared/events-v2.js` `PAGE_UPSERT`: on conflict it refreshes only the
  mutable columns (status/dwell/scroll/watch/clicks/vitals/errors). `visitor`,
  `ip`, `ts`, `day` and `source` keep their original values, so a caller that
  guessed a page id could at worst edit those mutable fields of a row it
  already knows — it cannot rewrite the record's identity or address.
- `functions/api/push/subscribe.js`: endpoint shape + literal-address SSRF
  check, p256dh/auth patterns, and the verified-name gate (`nameVerified`) —
  evidence alone is not enough, so a single spoofed comment cannot bind
  another person's @handle.
- `functions/api/claims.js`: evidence gate + 8-names-per-IP cap + 20/min wall.
- `functions/api/video-dead.js`: witness token is HMAC-checked, video-bound,
  1h TTL and burned on first use (PK conflict = replay = unwitnessed), so a
  script that never rendered the page can log reports but can never de-list.
- `functions/api/status/probe-history.js`: the KV ring is bounded by
  construction (72h of 15-minute probes), so the response cannot grow.
- `functions/api/hero.js`: generic 500 body; upstream detail only to logs.
- `functions/api/admin/export.js`: session-gated, table/column lists are
  literals, `days` clamped (365 / 3650), and `csvCell()` prefixes `= + - @\t\r`
  so a visitor-supplied search term cannot execute as a formula in Excel.
- `functions/api/comments.js`: honeypot before quota, shape validation, bound
  params, reply target must be a top-level visible comment on the same video.

### Residual risk closed (eighth pass)

- `/api/views` and `/api/video-dead` accepted any `[\w.-]{1,120}` id and
  created a row per distinct value. VULN-24: both writes are now gated on the
  id existing in `wpf_video_archive` — the table every feed refresh fills with
  the cards the site actually serves, and the only ids a watch page exists
  for. The gate is `INSERT … SELECT … WHERE EXISTS`, i.e. the same statement
  as before: a real play still costs one write plus the count read, with no
  extra round trip on the hot path. A schema without the archive table falls
  back to the ungated upsert (a real play must never be lost to migration
  drift). An invented id now returns `{views:0, embeds:0}` and writes nothing.
  Existing rows are deliberately kept: an audit of the live database on
  2026-11-14 found 266 unarchived counter rows, and every sampled one was a
  real fight played before `wpf_video_archive` existed or held that card
  (`wsh-228221`, 32 plays; `reddit-1w0gw0c`, 24). "Not archived" therefore
  cannot separate an invented id from a historical play, so no delete ships —
  the gate stops growth and the back catalogue stays.


