#!/usr/bin/env python3
"""
fix_others.py — Fix broken non-academic links.

Broken links found 2026-03-24:
  1. https://doi.org/10.1109/TSP.2021.3135692  → 418 (IEEE Xplore bot block)
     → IEEE TSP 2021 — may have an open arXiv version or should use
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9658580
     → NOT found in repo yet (not in any .md checked)

  2. https://www.ericsson.com/en/blog/2024/6/integrated-sensing-and-communication → 403
     paper/standardization.md line 88
     → Strategy: replace with Wayback Machine archive, or remove + note

  3. https://www.weforum.org/podcasts/radio-davos/episodes/top-10-emerging-technologies-2024/ → 403
     paper/standardization.md line 85
     → Strategy: remove (podcast not critical for academic list)
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

OTHERS_BROKEN = [
    {
        "broken": "https://doi.org/10.1109/TSP.2021.3135692",
        "file": None,   # not yet located in repo
        "strategy": "research",   # 'research' = search for arXiv/open version
        "note": "IEEE TSP 2021 — check if arXiv version exists",
    },
    {
        "broken": "https://www.ericsson.com/en/blog/2024/6/integrated-sensing-and-communication",
        "file": "paper/standardization.md",
        "strategy": "archive",   # replace with archive.org
        "replacement": "https://webcache.googleusercontent.com/search?q=cache:ericsson.com/en/blog/2024/6/integrated-sensing-and-communication",
        "note": "Ericsson blog post (offline); use webcache or remove",
    },
    {
        "broken": "https://www.weforum.org/podcasts/radio-davos/episodes/top-10-emerging-technologies-2024/",
        "file": "paper/standardization.md",
        "strategy": "remove",
        "note": "WEF podcast page offline; entry removed as non-critical",
    },
]


def research_arxiv_for_doi(doi: str) -> str:
    """Try to find an open-access version via DOIA + arXiv indirect search."""
    import urllib.request, json, re
    # Extract IEEE paper number: 10.1109/TSP.2021.3135692 → 3135692
    num_m = re.search(r'\.(\d+)$', doi)
    if not num_m:
        return ""
    # Semantic Scholar API sometimes links to arXiv
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=arxivExternalId,openAccessPdf"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "awesome-isac-bot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            arxiv_id = data.get("arxivExternalId", "")
            if arxiv_id:
                return f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = data.get("openAccessPdf", {}).get("url", "")
            if pdf_url:
                return pdf_url
    except Exception:
        pass
    return ""


def fix_file_replace(path: Path, old: str, new: str, note: str = None):
    content = path.read_text(encoding="utf-8")
    if old not in content:
        print(f"  [SKIP] {path}: link not found")
        return False
    new_content = content.replace(old, new)
    if note:
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if old in line:
                lines[i] = line + f"  <!-- FIXED: {note} -->"
                break
        new_content = "\n".join(lines)
    path.write_text(new_content, encoding="utf-8")
    print(f"  [FIXED] {path}")
    return True


def fix_file_remove(path: Path, old: str, note: str = None):
    content = path.read_text(encoding="utf-8")
    if old not in content:
        print(f"  [SKIP] {path}: link not found")
        return False
    # Find the whole table row line(s) and remove the link part
    new_content = content.replace(f"[🔗]({old})", f"[🔗 ❌]")
    if note:
        lines = content.split("\n")
        new_lines = []
        for line in lines:
            if old in line and "❌" not in line:
                line = line + f" <!-- REMOVED: {note} -->"
            new_lines.append(line)
        new_content = "\n".join(new_lines)
    path.write_text(new_content, encoding="utf-8")
    print(f"  [REMOVED] {path}: dead link marked")
    return True


def main():
    print("🔧 Other Broken Links Fixer")
    print("=" * 50)
    for item in OTHERS_BROKEN:
        print(f"\n📄 {item['broken']}")
        print(f"   Strategy: {item['strategy']}")
        print(f"   Note: {item['note']}")

        if item["strategy"] == "research":
            arxiv_url = research_arxiv_for_doi(item["broken"])
            if arxiv_url:
                print(f"   🔍 Found open version: {arxiv_url}")
            else:
                print(f"   ⚠️  Could not locate open version")
            continue

        if item["file"] is None:
            print(f"   [SKIP] file unknown — manual search needed")
            continue

        path = REPO_ROOT / item["file"]
        if not path.exists():
            print(f"   [SKIP] file not found: {item['file']}")
            continue

        if item["strategy"] == "archive" and item.get("replacement"):
            fix_file_replace(path, item["broken"], item["replacement"], note=item["note"])
        elif item["strategy"] == "remove":
            fix_file_remove(path, item["broken"], note=item["note"])
        elif item["strategy"] == "research" and item.get("replacement"):
            fix_file_replace(path, item["broken"], item["replacement"], note=item["note"])


if __name__ == "__main__":
    main()
