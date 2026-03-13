---
name: build-post
description: Use when building, compiling, or previewing blog posts. Triggered by requests to build, render, preview, or generate HTML from a markdown post.
disable-model-invocation: true
argument-hint: [filename.md or "all"]
allowed-tools: Bash
---

# Build Blog Post

Build all blog posts from Markdown to HTML.

## Commands

**Build everything (posts + index):**
```bash
uv run python build.py
```

## What happens

1. Parses YAML front matter (title, author, date) from each `src/*.md` file
2. Protects math delimiters from Markdown processing
3. Converts Markdown to HTML (raw HTML passthrough for proofs, sidenotes)
4. Injects content into `templates/base.html`
5. Copies CSS from `static/` to root (with cache-busting query string)
6. Writes `.html` files to repository root (for GitHub Pages)
7. Generates `index.html` from `templates/index.html` + `about.md`

## After building

Open the result: `open FILENAME.html`

## Troubleshooting

- **Math not rendering**: Check browser console for KaTeX errors. Ensure `$` delimiters are not inside code blocks.
- **Stale CSS**: The build adds `?v=TIMESTAMP` to CSS links. Hard refresh if needed.
- **Proof formatting broken**: Check that all `<div>` and `<details>` tags are properly closed and nested.
