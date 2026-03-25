#!/usr/bin/env python3
"""
fix_ieee.py — Auto-fix broken IEEE Xplore links.

Broken links found 2026-03-24:
  1. https://ieeexplore.ieee.org/document/10372550  (404)
     → paper/optical.md line 49
     → "On the Hardware-Limited Sensing Parameter Extraction for Integrated
        Sensing and Communication System Towards 6G" (IEEE ICCT 2023)

  2. https://ieeexplore.ieee.org/document/10596332  (404)
     → README.md line 266
     → "Network-Level ISAC: Interference Management and BS Coordination"
        (IEEE TWC 2024)
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
IEEE_BROKEN = {
    "https://ieeexplore.ieee.org/document/10372550": {
        "file": "paper/optical.md",
        "line_hint": 49,
        "title_hints": [
            "Hardware-Limited Sensing Parameter Extraction",
            "Integrated Sensing and Communication System Towards 6G",
        ],
        "doi": "10.23919/ICCT.2023.2023.00111",  # placeholder - to verify
        "note": "IEEE ICCT 2023 — check if published in IEEE Xplore or arXiv",
    },
    "https://ieeexplore.ieee.org/document/10596332": {
        "file": "README.md",
        "line_hint": 266,
        "title_hints": [
            "Network-Level ISAC",
            "Interference Management and BS Coordination",
        ],
        "doi": None,  # to research
        "note": "IEEE TWC 2024 — verify if paper exists",
    },
}


def search_replacement_doi(title: str) -> str:
    """Use CrossRef/DOI.org to find a live DOI for a paper title."""
    import urllib.request
    import json
    # Search CrossRef by title
    query = title.replace(" ", "+")
    url = f"https://api.crossref.org/works?query.title={query}&rows=1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "awesome-isac-bot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            items = data.get("message", {}).get("items", [])
            if items:
                doi = items[0].get("DOI", "")
                if doi:
                    return f"https://doi.org/{doi}"
    except Exception as e:
        print(f"  [WARN] CrossRef search failed: {e}")
    return ""


def fix_file(path: Path, old_url: str, new_url: str, note: str = None):
    """Replace old_url with new_url in a file, preserving the link label."""
    content = path.read_text(encoding="utf-8")
    # Replace just the URL part in [label](url)
    new_content = content.replace(old_url, new_url)
    if new_content == content:
        print(f"  [SKIP] URL not found in {path}")
        return False
    if note:
        # Append a footnote comment after the line
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if old_url in line:
                # Add a comment inline
                lines[i] = line + f"  <!-- FIXED: {note} -->"
                break
        new_content = "\n".join(lines)
    path.write_text(new_content, encoding="utf-8")
    print(f"  [FIXED] {path}: {old_url} → {new_url}")
    return True


def main():
    print("🔧 IEEE Xplore Broken Link Fixer")
    print("=" * 50)
    for broken_url, info in IEEE_BROKEN.items():
        f = REPO_ROOT / info["file"]
        if not f.exists():
            print(f"\n⚠️  File not found: {f}")
            continue
        print(f"\n📄 {info['file']} — {broken_url}")
        print(f"   Note: {info['note']}")
        # Try to find a live DOI via CrossRef
        for hint in info["title_hints"]:
            replacement = search_replacement_doi(hint)
            if replacement:
                print(f"   🔍 Found via CrossRef: {replacement}")
                fix_file(f, broken_url, replacement, note=info["note"])
                break
        else:
            print(f"   ⚠️  Could not auto-locate replacement. Manual fix needed.")
            print(f"   Title hints: {info['title_hints']}")


if __name__ == "__main__":
    main()
