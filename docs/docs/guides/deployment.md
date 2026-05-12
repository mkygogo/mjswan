---
icon: octicons/rocket-16
---

# Deployment

mjswan produces a fully static site — the output of `builder.build()` can be served from any static host without a backend. This page covers the common hosting options and the configuration that changes between them.

## The `base_path` setting

When your site lives at the root of a domain (`https://example.com/`), the default `base_path="/"` works without any changes.

When your site lives at a subdirectory — the typical case for GitHub Pages project pages (`https://user.github.io/myrepo/`) — you must tell mjswan about the prefix so asset URLs resolve correctly:

```python
builder = mjswan.Builder(base_path="/myrepo/")
```

You can also set this at runtime with an environment variable to avoid hardcoding it:

```bash
MJSWAN_BASE_PATH=/myrepo/ python build.py
```

## GitHub Pages

### Manual deploy

```bash
python build.py          # writes dist/
cp -r dist/. docs/       # copy into the Pages source directory
git add docs/
git commit -m "Deploy"
git push
```

Configure Pages in your repo settings to serve from the `docs/` folder on the `main` branch, or from any branch/folder you prefer.

### GitHub Actions (automated)

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v5

      - name: Install dependencies
        run: uv sync

      - name: Build
        run: uv run python build.py
        env:
          MJSWAN_BASE_PATH: /myrepo/
          MJSWAN_NO_LAUNCH: "1"

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./dist
```

!!! note
    Replace `myrepo` with your actual repository name. The Pages source should be set to the `gh-pages` branch (created automatically by `peaceiris/actions-gh-pages`).

<!-- MEDIA: suggest a screenshot of GitHub Pages settings showing the branch/folder configuration -->

## Netlify

Drop the `dist/` directory into [Netlify Drop](https://app.netlify.com/drop) for an instant preview URL, or connect your Git repository for automatic deployments.

For CI builds, set the build command and publish directory:

| Setting | Value |
|---|---|
| Build command | `python build.py` |
| Publish directory | `dist` |

No `base_path` change is needed when deploying to a Netlify root domain.

## Cross-Origin Isolation headers for multi-threading

MuJoCo WASM is compiled in single-threaded mode by default. To use the multi-threaded build, pass `mt=True` when constructing the `Builder`:

```python
builder = mjswan.Builder(mt=True)
```

Multi-threading relies on `SharedArrayBuffer`, which requires two HTTP response headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

`app.launch()` always sets these on the local dev server. For production hosts, mjswan emits the right artifacts at build time **only when `mt=True`** — without that flag, the steps below are unnecessary.

**Netlify / Cloudflare Pages / Vercel** — when `mt=True`, mjswan writes a `_headers` file inside `dist/`:

```
/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
```

**GitHub Pages** — does not support custom response headers. When `mt=True`, mjswan ships a `coi-serviceworker.js` that injects the headers client-side. This is handled automatically by the built application and requires no extra configuration.

**Self-hosted / nginx** — add to your server block:

```nginx
add_header Cross-Origin-Opener-Policy "same-origin";
add_header Cross-Origin-Embedder-Policy "require-corp";
```

**Caddy:**

```caddy
header {
    Cross-Origin-Opener-Policy "same-origin"
    Cross-Origin-Embedder-Policy "require-corp"
}
```


## Deployment size and the 1 GB GitHub Pages limit

GitHub Pages enforces a 1 GB repository size limit. If your deployment is large (many scenes, large meshes), use `spec=` instead of `model=` in `add_scene()` — the `.mjz` format applies DEFLATE compression and is typically 3–10× smaller than the binary `.mjb`.

```python
# Larger output
project.add_scene(model=mujoco.MjModel.from_xml_path("scene.xml"), name="My Scene")

# Smaller output — prefer this for large deployments
project.add_scene(spec=mujoco.MjSpec.from_file("scene.xml"), name="My Scene")
```
