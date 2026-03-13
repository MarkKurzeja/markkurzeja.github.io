---
name: new-post
description: Use when creating a new blog post, starting a new article, or scaffolding a post. Triggered by requests to create, start, or write a new blog post.
disable-model-invocation: true
argument-hint: [post-title]
allowed-tools: Bash
---

# New Blog Post Scaffold

Create a new Markdown blog post in `src/` with the correct format for this blog.

## Steps

1. Generate a kebab-case filename from `$ARGUMENTS` (e.g., "Cauchy-Schwarz Inequality" -> `cauchy-schwarz-inequality.md`)
2. Create the file in `src/` with this template:

```markdown
---
title: $ARGUMENTS
author: Mark Kurzeja
date: YYYY-MM-DD
---

<div class="abstract">
Brief summary of the post (1-3 sentences).
</div>

## Introduction

Opening paragraph.

## Main Content

Body of the post.

## Conclusion

Closing thoughts.
```

3. Use today's date for the `date` field
4. Remind the user to:
   - Build with: `uv run python build.py`
   - Preview with: `open FILENAME.html`

## Tufte Elements Available

Remind the user of these formatting options:

- **Sidenotes**: `<span class="sidenote-number"></span><span class="sidenote">Note text</span>`
- **Margin notes**: `<span class="marginnote">Note text</span>`
- **Math**: `$inline$` and `$$display$$`
- **Structured proofs**: Use `/structured-proof` skill
- **Abstract**: `<div class="abstract">...</div>`
- **Algorithm blocks**: Use `/tufte-patterns` for the template
