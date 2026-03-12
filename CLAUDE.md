# CLAUDE.md

## Blog Build

- Source markdown in `src/`, source CSS in `static/`, built HTML + CSS in root
- **Edit `static/tufte.css`, not root `tufte.css`** - build copies static/ to root, overwriting it
- Build: `uv run python build.py`
- Edit loop: make an edit → `uv run python build.py` → `open <file>.html` to relaunch in browser
- Every post requires a `publish` frontmatter field (defaults to `false` if missing)
- `publish: true` builds HTML to root (deployed via GitHub Pages); `publish: false` builds to `drafts/` (gitignored, local preview only)
- Full builds delete all post HTML from root and `drafts/`, then regenerate everything
- Only published posts appear in `index.html`
- New posts default to draft. Set `publish: true` when ready to go live
- Tufte-style sidenotes: `<span class="sidenote-number"></span><span class="sidenote">...</span>` inline in markdown
- Margin notes: `<span class="marginnote">...</span>`
- Place sidenote markers before the period, not after: `text.<span class="sidenote-number">...` not `text. <span class="sidenote-number">...`
- If something is in parentheses, consider whether it should be a sidenote instead
- Reference implementation: `../blog/` has the canonical Tufte CSS/JS setup

## Writing Style

- No em dashes. Use single dashes (-), parentheses, periods, or colons instead.
- Gender-neutral language (they/them, "salesperson" not "salesman")
- Active voice preferred
- IC perspective (the author is not an executive)
