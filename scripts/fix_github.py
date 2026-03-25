#!/usr/bin/env python3
"""
fix_github.py — Fix broken GitHub links.

Broken links found 2026-03-24:
  1. https://github.com/maceh1/radiate_sdk  → 404
     datasets/README.md line 23
     RADIATE dataset entry

The radiate_sdk repo does not exist (404).  The RADIATE dataset itself
is still available from the official source:
  http://www.robots.ox.ac.uk/~mobile/wiki/index.html

Strategy: replace with the official RADIATE dataset page or
remove the dead SDK link entirely (keep the dataset entry).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

GITHUB_BROKEN = [
    {
        "broken": "https://github.com/maceh1/radiate_sdk",
        "file": "datasets/README.md",
        "strategy": "replace",   # 'replace' or 'remove'
        "replacement": "http://www.robots.ox.ac.uk/~mobile/wiki/index.html",
        "note": "Official RADIATE dataset page (University of Oxford)",
    },
]


def fix_file(path: Path, old: str, new: str, remove: bool = False):
    content = path.read_text(encoding="utf-8")
    if old not in content:
        print(f"  [SKIP] {path}: link not found")
        return False
    if remove:
        # Replace [🔗](url) with just [🔗] (dead link marker)
        new_content = content.replace(f"({old})", "")
        print(f"  [REMOVED] {path}: SDK link removed")
    else:
        new_content = content.replace(old, new)
        print(f"  [FIXED] {path}")
    path.write_text(new_content, encoding="utf-8")
    return True


def main():
    print("🔧 GitHub Broken Link Fixer")
    print("=" * 50)
    for item in GITHUB_BROKEN:
        path = REPO_ROOT / item["file"]
        if not path.exists():
            print(f"  [SKIP] File not found: {item['file']}")
            continue
        print(f"\n📄 {item['file']}: {item['broken']}")
        print(f"   Strategy: {item['strategy']}")
        if item["strategy"] == "remove":
            fix_file(path, item["broken"], "", remove=True)
        else:
            fix_file(path, item["broken"], item.get("replacement", ""))


if __name__ == "__main__":
    main()
