# CLAUDE.md

## Blog Build

- Source markdown in `src/`, source CSS in `static/`, built HTML + CSS in root
- **Edit `static/tufte.css`, not root `tufte.css`** - build copies static/ to root, overwriting it
- Build: `uv run python build.py`
- Edit loop: make an edit -> `uv run python build.py` -> `open <file>.html` to relaunch in browser
- Tufte-style sidenotes: `<span class="sidenote-number"></span><span class="sidenote">...</span>` inline in markdown
- Margin notes: `<span class="marginnote">...</span>`
- Place periods BEFORE sidenote spans, never after: `text.<span class="sidenote-number"></span><span class="sidenote">...</span>` not `text<span class="sidenote-number"></span><span class="sidenote">...</span>.` (trailing periods after sidenotes render poorly on mobile)
- If something is in parentheses, consider whether it should be a sidenote instead

## Writing Style

- No em dashes. Use single dashes (-), parentheses, periods, or colons instead.
- Gender-neutral language (they/them, "salesperson" not "salesman")
- Active voice preferred
- IC perspective (the author is not an executive)

## KaTeX & Math

- `$...$` inline, `$$...$$` display. KaTeX renders client-side.
- `ignoredClasses: ["sidenote", "marginnote"]` in `templates/base.html` -- `$` in sidenotes is safe, won't trigger KaTeX
- Literal `$` outside math: use `\$` or `&#36;`
- Use `aligned` (not `align`) inside `$$...$$` for multi-line equations

## Proof Structure Guidelines

- Every proof step should be a **declarative statement** that the substeps defend -- not narrative ("The dispute reveals...") but factual ("AI governance depends on...")
- Only include steps that **logically support the claim**. Move narrative/consequence material to prose sections outside the proof block
- Use 3-4 layers of nesting freely when it makes the argument easier to follow
- Non-load-bearing details (quotes, colorful anecdotes) belong in **sidenotes**, not proof steps

## Sidenote Placement

- **Sidenotes** (numbered) must be placed strategically -- the superscript number appears inline, so it must sit adjacent to the specific sentence the sidenote describes
- **Margin notes** (unnumbered) can go anywhere in a paragraph since they have no inline marker
- When reviewing, check that each sidenote number appears next to the text it enriches

## CSS Notes

- `.byline` -- plain text at 1.1rem
- `.proof-block > em, .subproof > em` -- explicit 1.4rem for "Proof." labels
- `.qed-step` -- no border-top
- `.source-line` -- styled like subproof (left border, indented) at 1.4rem

## Gotchas

- `---` in markdown between HTML blocks renders as `<hr>` -- omit unless you want a visible rule
- Byline date: use normal capitalization (e.g., "March 7, 2026")
- CSS counters break inside `<details>` elements (Chrome/Safari bug) -- JS-driven sidenote numbering handles this
