---
name: katex-math
description: Use when writing math in blog posts, adding equations, or when the user asks about KaTeX syntax, math formatting, or how to typeset mathematical expressions. Also use when converting LaTeX math to blog-compatible format.
argument-hint: [math-topic or LaTeX snippet]
---

# KaTeX Math Reference for Blog Posts

## Inline vs Display

- **Inline**: `$x^2 + y^2 = z^2$` renders within text flow
- **Display**: `$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$` renders centered on its own line

## Important: Math Protection

The build script (`build.py`) protects math from Markdown processing. Dollar signs are preserved as-is in the HTML output. KaTeX auto-render runs client-side via `DOMContentLoaded`.

## Font Sizing

KaTeX is set to `font-size: 1.0em !important` in `tufte.css` (down from KaTeX default of 1.21em) to match body text proportions.

## Supported Environments

All standard LaTeX math environments work. Use `aligned` inside `$$...$$` for multi-line:

```
$$\begin{aligned}
f(x) &= ax^2 + bx + c \\
     &= a(x - h)^2 + k
\end{aligned}$$
```

Other environments: `matrix`, `pmatrix`, `bmatrix`, `cases`, `aligned`.

## Common Patterns

### Fractions and roots
- `\frac{a}{b}`, `\dfrac{a}{b}` (display-style fraction)
- `\sqrt{x}`, `\sqrt[3]{x}`

### Greek letters
- Lowercase: `\alpha`, `\beta`, `\gamma`, `\delta`, `\epsilon`, `\theta`, `\lambda`, `\mu`, `\pi`, `\sigma`, `\phi`, `\omega`
- Uppercase: `\Gamma`, `\Delta`, `\Theta`, `\Lambda`, `\Pi`, `\Sigma`, `\Phi`, `\Omega`

### Operators and relations
- `\sum_{i=1}^{n}`, `\prod`, `\int_{a}^{b}`, `\lim_{x \to 0}`
- `\leq`, `\geq`, `\neq`, `\approx`, `\equiv`, `\sim`
- `\in`, `\notin`, `\subset`, `\subseteq`, `\cup`, `\cap`

### Decorations
- `\hat{x}`, `\bar{x}`, `\tilde{x}`, `\vec{x}`, `\dot{x}`, `\ddot{x}`
- `\mathbf{x}` (bold), `\mathbb{R}` (blackboard bold), `\mathcal{L}` (calligraphic)

### Brackets
- Auto-sizing: `\left( ... \right)`, `\left[ ... \right]`, `\left\{ ... \right\}`
- `\langle ... \rangle` for angle brackets

### Spacing
- `\,` thin, `\;` medium, `\quad` wide, `\qquad` extra wide

## Gotchas

1. **Backslashes in Markdown**: Single `\` works in math delimiters. No extra escaping needed since the build script protects math before Markdown processing.
2. **Dollar signs in text**: If you need a literal `$` outside math, it may trigger KaTeX. Use `\$` or an HTML entity `&#36;`.
3. **Aligned vs align**: Use `aligned` (not `align`) inside `$$...$$`. The `align` environment is an AMS top-level env; `aligned` is the nestable version.
4. **Display math spacing**: CSS sets `.katex-display` margin to `0.3rem 0` for tight spacing. If you need more breathing room, wrap in a div with custom margin.
