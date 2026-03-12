# CLAUDE.md

## Blog Build

- Source markdown in `src/`, source CSS in `static/`, built HTML + CSS in root
- **Edit `static/tufte.css`, not root `tufte.css`** - build copies static/ to root, overwriting it
- Build: `uv run python build.py`
- Edit loop: make an edit → `uv run python build.py` → `open <file>.html` to relaunch in browser
- Tufte-style sidenotes: `<span class="sidenote-number"></span><span class="sidenote">...</span>` inline in markdown
- Margin notes: `<span class="marginnote">...</span>`
- Place periods BEFORE sidenote spans, never after: `text.<span class="sidenote-number"></span><span class="sidenote">...</span>` not `text<span class="sidenote-number"></span><span class="sidenote">...</span>.` (trailing periods after sidenotes render poorly on mobile)
- If something is in parentheses, consider whether it should be a sidenote instead
- Reference implementation: `../blog/` has the canonical Tufte CSS/JS setup

## Writing Style

- No em dashes. Use single dashes (-), parentheses, periods, or colons instead.
- Gender-neutral language (they/them, "salesperson" not "salesman")
- Active voice preferred
- IC perspective (the author is not an executive)
