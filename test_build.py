#!/usr/bin/env python3
"""Validate build output: checks that HTML files exist, are well-formed, and math delimiters survive."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
ERRORS = []


def error(msg: str):
  ERRORS.append(msg)
  print(f"  FAIL: {msg}")


def check_file_exists(path: Path):
  if not path.exists():
    error(f"Missing expected output: {path}")
    return False
  if path.stat().st_size == 0:
    error(f"Empty file: {path}")
    return False
  return True


def check_html_structure(path: Path):
  """Check that HTML has basic required structure."""
  html = path.read_text(encoding="utf-8")

  for tag in ["<!DOCTYPE html>", "<html", "</html>", "<head>", "</head>", "<body>", "</body>"]:
    if tag not in html:
      error(f"{path.name}: missing {tag}")

  if "<title>" not in html or "</title>" not in html:
    error(f"{path.name}: missing <title>")

  # Check that template placeholders were replaced
  for placeholder in ["{{title}}", "{{content}}", "{{css_path}}"]:
    if placeholder in html:
      error(f"{path.name}: unresolved template placeholder {placeholder}")


def check_math_delimiters(path: Path):
  """Check that math delimiters survive the build (not mangled by Markdown)."""
  html = path.read_text(encoding="utf-8")

  # Skip files with no math
  if "$" not in html and "\\(" not in html and "\\[" not in html:
    return

  # Check for common Markdown mangling of math: <em> tags inside $ delimiters
  # Pattern: $<em>...</em>$ indicates Markdown treated _ as emphasis inside math
  mangled = re.findall(r"\$<em>.*?</em>\$", html)
  if mangled:
    error(f"{path.name}: math delimiters mangled by Markdown processor: {mangled[0][:80]}")


def check_no_null_bytes(path: Path):
  """Check that math placeholder null bytes were fully restored."""
  html = path.read_bytes()
  if b"\x00" in html:
    error(f"{path.name}: contains null bytes (unrestored math placeholders)")


def main():
  print("Validating build output...\n")

  # 1. Check that all expected HTML files exist
  print("1. Checking expected output files...")
  src_files = sorted((ROOT / "src").glob("*.md"))
  expected_html = [ROOT / f.with_suffix(".html").name for f in src_files]
  expected_html.append(ROOT / "index.html")

  existing_html = []
  for html_path in expected_html:
    if check_file_exists(html_path):
      existing_html.append(html_path)
      print(f"  OK: {html_path.name}")

  # 2. Check HTML structure
  print("\n2. Checking HTML structure...")
  for html_path in existing_html:
    check_html_structure(html_path)
    print(f"  OK: {html_path.name}")

  # 3. Check math delimiters
  print("\n3. Checking math delimiter integrity...")
  for html_path in existing_html:
    check_math_delimiters(html_path)
    check_no_null_bytes(html_path)
    print(f"  OK: {html_path.name}")

  # 4. Check static assets
  print("\n4. Checking static assets...")
  css = ROOT / "tufte.css"
  if check_file_exists(css):
    print(f"  OK: {css.name}")

  # Summary
  print(f"\n{'=' * 40}")
  if ERRORS:
    print(f"FAILED: {len(ERRORS)} error(s)")
    for e in ERRORS:
      print(f"  - {e}")
    sys.exit(1)
  else:
    print(f"PASSED: {len(existing_html)} HTML files validated")


if __name__ == "__main__":
  main()
