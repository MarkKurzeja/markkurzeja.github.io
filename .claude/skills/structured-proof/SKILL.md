---
name: structured-proof
description: Use when creating or editing structured proofs (pf2-style) in blog posts. Triggered by requests to add a proof, write a proof, create proof steps, or add mathematical reasoning with nested steps.
argument-hint: [proof-topic]
---

# Structured Proof Skill

Create a Lamport pf2-style structured proof block for a blog post in `src/`.

## Format

Proofs use semantic HTML inside Markdown files. The structure is:

```html
<div class="proof-block">
<div class="proof-label">Proof</div>

<details class="step-details" open>
<summary class="step">
<span class="step-number" id="PROOF-NAME:A1">A1.</span>
Step claim here.
</summary>

<div class="subproof">
<em>Proof.</em> Justification text with math like $x = 2k$.

<div class="step">
<span class="step-number" id="PROOF-NAME:A1.1">A1.1.</span>
Substep claim with $math$.
</div>

<div class="qed-step">
<em>Q.E.D.</em> Follows from A1.1 and A1.2.
</div>
</div>
</details>

<div class="qed-step">
<em>Q.E.D.</em> Follows from A1 and A2.
</div>
</div>
```

## Rules

1. **Step IDs must be namespaced**: Use `id="PROOF-NAME:A1"` where PROOF-NAME is a short kebab-case identifier for the proof (e.g., `even-sum`, `cauchy-schwarz`). This prevents collisions across multiple proofs.

2. **Step numbering uses letter prefixes**: Top-level steps are A1, A2, A3. Substeps are A1.1, A1.2. Deeper nesting: A1.1.1, A1.1.2. A second proof in the same post uses B1, B2, etc.

3. **Collapsible steps**: Wrap any step that has a subproof in `<details class="step-details" open>` with the step as `<summary>`. This makes it collapsible. Steps without subproofs use plain `<div class="step">`.

4. **Nesting depth**: Subproofs are wrapped in `<div class="subproof">` and can nest arbitrarily. Each level gets progressively lighter border styling via CSS.

5. **QED lines**: Use `<div class="qed-step">` with `<em>Q.E.D.</em>` followed by plain text referencing steps (no hyperlinks). Keep all QED lines at the same font size.

6. **Proof sketch**: Optional. Use `<p class="proof-sketch">Proof sketch: description</p>` at the start of a subproof.

7. **Math**: Use `$...$` for inline, `$$...$$` for display. KaTeX renders client-side.

## Template for $ARGUMENTS

When given a proof topic, generate a well-structured proof with:
- Clear top-level claims as collapsible steps
- Substeps for each logical deduction
- Proper math notation
- Namespaced IDs based on the topic
