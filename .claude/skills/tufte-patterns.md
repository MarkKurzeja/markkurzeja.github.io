---
name: tufte-patterns
description: Reference patterns for Tufte-style blog HTML - sidenotes, marginnotes, proof blocks, algorithms, figures
user-invocable: false
---

# Tufte Blog Patterns

Copy-paste-ready HTML patterns for this blog. All patterns go in `src/*.md` files.

## Sidenotes & Marginnotes

**Sidenote (numbered):** Place period BEFORE the span, never after.
```html
text.<span class="sidenote-number"></span><span class="sidenote">Note content here.</span>
```

Mid-sentence sidenote (text continues after):
```html
text.<span class="sidenote-number"></span><span class="sidenote">Note content.</span> Sentence continues here.
```

**Marginnote (no number):**
```html
text.<span class="marginnote">Margin note content here.</span>
```

## Abstract

```html
<div class="abstract">
One or two sentence summary of the post.<span class="sidenote-number"></span><span class="sidenote">Optional attribution or context.</span>
</div>
```

## Proof Block (collapsible numbered steps)

```html
<div class="proof-block">
<div class="proof-label">Title of the Proof or Principles</div>

<details class="step-details" open>
<summary class="step">
<span class="step-number" id="prefix:1">1.</span>
Short imperative statement.
</summary>
<div class="subproof">
Explanation paragraph. Can include sidenotes.<span class="sidenote-number"></span><span class="sidenote">Note.</span>
</div>
</details>

<details class="step-details" open>
<summary class="step">
<span class="step-number" id="prefix:2">2.</span>
Next step.
</summary>
<div class="subproof">
Explanation.
</div>
</details>

</div>
```

## Algorithm Block (Knuth-style)

```html
<div class="algorithm">
<div class="algorithm-header"><strong>Algorithm A</strong> (Name). Given X, produce Y.</div>
<ol class="algorithm-steps">
<li><span class="algorithm-step-label">A1.</span><span class="algorithm-step-body"><span class="step-action">[Verb]</span> Description of step.</span></li>
<li><span class="algorithm-step-label">A2.</span><span class="algorithm-step-body"><span class="step-action">[Verb]</span> Description of step.</span></li>
</ol>
</div>
```

## Figures

Standard:
```html
<figure>
<img src="posts/folder/image.png" alt="Description"/>
<figcaption>Caption text</figcaption>
</figure>
```

Full-width:
```html
<figure class="fullwidth">
<img src="posts/folder/image.png" alt="Description"/>
</figure>
```

## Math (KaTeX)

Inline: `$x = a$`

Block: `$$x = \frac{a}{b}$$`

## Key Rules

- Period goes BEFORE sidenote spans, never after (renders poorly on mobile)
- No em dashes: use single dashes, parentheses, periods, or colons
- Edit `static/tufte.css`, never root `tufte.css`
- Build with `uv run python build.py`
