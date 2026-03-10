#!/usr/bin/env python3
"""Build static HTML blog from Markdown source files.

Usage:
  uv run python build.py                                  # Build all posts + index
  uv run python build.py posts/my-post.md                 # Build single post only
  uv run python build.py posts/*.md                       # Build multiple posts
  uv run python build.py posts/*.md --index               # Build posts + index
  uv run python build.py --index                          # Rebuild index only

Source format: Markdown with YAML front matter. Supports:
  - KaTeX math ($...$ inline, $$...$$ display, align environments)
  - Tufte sidenotes and margin notes (HTML tags in Markdown)
  - Structured proofs (pf2-style, using semantic HTML)
"""

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

import markdown
import yaml

ROOT = Path(__file__).parent


def parse_front_matter(text: str) -> tuple[dict, str]:
  """Split YAML front matter from Markdown body."""
  if text.startswith("---"):
    parts = text.split("---", 2)
    if len(parts) >= 3:
      meta = yaml.safe_load(parts[1]) or {}
      body = parts[2]
      return meta, body
  return {}, text


def protect_math(text: str) -> tuple[str, list[tuple[str, str]]]:
  """Replace math delimiters with placeholders so Markdown doesn't mangle them.

  Protects: $$...$$, $...$, \\[...\\], \\(...\\), and environments like
  \\begin{align}...\\end{align}.
  """
  placeholders = []
  counter = 0

  def replace(match):
    nonlocal counter
    placeholder = f"\x00MATH{counter}\x00"
    placeholders.append((placeholder, match.group(0)))
    counter += 1
    return placeholder

  # Display math: $$...$$ (including multiline)
  text = re.sub(r"\$\$(.+?)\$\$", replace, text, flags=re.DOTALL)
  # LaTeX environments: \begin{align}...\end{align}, etc.
  text = re.sub(r"\\begin\{(\w+\*?)\}(.+?)\\end\{\1\}", replace, text, flags=re.DOTALL)
  # Bracket display math: \[...\]
  text = re.sub(r"\\\[(.+?)\\\]", replace, text, flags=re.DOTALL)
  # Inline math: $...$  (not greedy, single line)
  text = re.sub(r"\$([^\$\n]+?)\$", replace, text)
  # Inline paren math: \(...\)
  text = re.sub(r"\\\((.+?)\\\)", replace, text)

  return text, placeholders


def restore_math(html: str, placeholders: list[tuple[str, str]]) -> str:
  """Put math expressions back into the HTML."""
  for placeholder, original in placeholders:
    html = html.replace(placeholder, original)
  return html


def render_template(template_str: str, meta: dict, content: str, css_path: str) -> str:
  """Simple mustache-like template rendering."""
  html = template_str
  html = html.replace("{{title}}", meta.get("title", "Untitled"))
  html = html.replace("{{css_path}}", f"{css_path}?v={int(time.time())}")
  html = html.replace("{{content}}", content)

  # Conditional sections: {{#key}}...{{/key}}
  for key in ["author", "date"]:
    val = meta.get(key, "")
    pattern = re.compile(r"\{\{#" + key + r"\}\}(.*?)\{\{/" + key + r"\}\}", re.DOTALL)
    if val:
      html = pattern.sub(lambda m: m.group(1).replace("{{" + key + "}}", str(val)), html)
    else:
      html = pattern.sub("", html)

  return html


def build_post(
  source: Path,
  output_dir: Path,
  template_path: Path,
  css_path: str,
) -> tuple[Path, dict]:
  """Convert a single Markdown file to HTML. Returns (output_path, metadata)."""
  text = source.read_text(encoding="utf-8")
  meta, body = parse_front_matter(text)

  # Protect math from Markdown processing
  body, placeholders = protect_math(body)

  # Convert Markdown to HTML (raw HTML passthrough is on by default)
  md = markdown.Markdown(
    extensions=["extra", "smarty", "toc"],
    extension_configs={
      "smarty": {"smart_quotes": True, "smart_dashes": True},
      "toc": {"permalink": True},
    },
  )
  content_html = md.convert(body)

  # Restore math expressions
  content_html = restore_math(content_html, placeholders)

  # Load and render template
  template_str = template_path.read_text(encoding="utf-8")
  full_html = render_template(template_str, meta, content_html, css_path)

  # Write output
  output_dir.mkdir(parents=True, exist_ok=True)
  out_file = output_dir / source.with_suffix(".html").name
  out_file.write_text(full_html, encoding="utf-8")
  return out_file, meta


def build_index(
  post_metas: list[tuple[str, dict]],
  output_dir: Path,
  about_path: Path,
  css_path: str,
) -> Path:
  """Build the index.html homepage with about section and post listing."""
  template_path = ROOT / "templates" / "index.html"
  template_str = template_path.read_text(encoding="utf-8")

  # Read and render about content
  about_text = about_path.read_text(encoding="utf-8")
  _, about_body = parse_front_matter(about_text)
  md = markdown.Markdown(extensions=["extra", "smarty"])
  about_html = md.convert(about_body)

  # Sort posts by date (newest first)
  post_metas.sort(key=lambda x: str(x[1].get("date", "")), reverse=True)

  # Build post list HTML
  items = []
  for html_name, meta in post_metas:
    title = meta.get("title", "Untitled")
    subtitle = meta.get("subtitle", "")
    date = meta.get("date", "")
    subtitle_html = f'<span class="post-subtitle"> &mdash; {subtitle}</span>' if subtitle else ""
    items.append(
      f'<li><a href="{html_name}">{title}</a>{subtitle_html}'
      f'<span class="post-date">{date}</span></li>'
    )
  post_list_html = f'<ul class="posts">\n{"".join(items)}\n</ul>'

  # Render template
  html = template_str
  html = html.replace("{{title}}", "TensorTales")
  html = html.replace("{{css_path}}", f"{css_path}?v={int(time.time())}")
  html = html.replace("{{about}}", about_html)
  html = html.replace("{{post_list}}", post_list_html)

  output_dir.mkdir(parents=True, exist_ok=True)
  out_file = output_dir / "index.html"
  out_file.write_text(html, encoding="utf-8")
  return out_file


def copy_static(output_dir: Path):
  """Copy static assets (CSS) to output directory."""
  static_dir = ROOT / "static"
  output_dir.mkdir(parents=True, exist_ok=True)

  # Copy CSS files
  for css_file in static_dir.glob("*.css"):
    shutil.copy2(css_file, output_dir / css_file.name)


def main():
  parser = argparse.ArgumentParser(
    description="Build static HTML blog from Markdown.",
  )
  parser.add_argument(
    "sources",
    nargs="*",
    type=Path,
    help="Markdown source file(s) to convert (default: all posts/*.md)",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=ROOT,
    help="Output directory (default: repo root for GitHub Pages)",
  )
  parser.add_argument(
    "--template",
    type=Path,
    default=ROOT / "templates" / "base.html",
    help="HTML template file for posts",
  )
  parser.add_argument(
    "--css",
    default="tufte.css",
    help="Path to CSS file (relative to output HTML, default: tufte.css)",
  )
  parser.add_argument(
    "--index",
    action="store_true",
    default=False,
    help="Also build the index page",
  )
  parser.add_argument(
    "--about",
    type=Path,
    default=ROOT / "about.md",
    help="About page Markdown file (for index)",
  )
  args = parser.parse_args()

  # If no sources given, build everything
  build_all = not args.sources
  if build_all:
    args.sources = sorted((ROOT / "src").glob("*.md"))
    args.index = True

  # Copy static assets
  copy_static(args.output)

  # Build posts
  post_metas = []
  for source in args.sources:
    if not source.exists():
      print(f"Error: {source} not found", file=sys.stderr)
      sys.exit(1)
    out, meta = build_post(source, args.output, args.template, args.css)
    post_metas.append((out.name, meta))
    print(f"Built: {out}")

  # Build index
  if args.index:
    if not args.about.exists():
      print(f"Warning: {args.about} not found, skipping index", file=sys.stderr)
    else:
      # If we didn't build all posts, scan posts/ for metadata to populate the index
      if not build_all:
        post_metas = []
        for md_file in sorted((ROOT / "src").glob("*.md")):
          text = md_file.read_text(encoding="utf-8")
          meta, _ = parse_front_matter(text)
          html_name = md_file.with_suffix(".html").name
          post_metas.append((html_name, meta))
      out = build_index(post_metas, args.output, args.about, args.css)
      print(f"Built: {out}")


if __name__ == "__main__":
  main()
