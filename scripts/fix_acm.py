#!/usr/bin/env python3
"""
fix_acm.py — Auto-fix broken ACM links (403 on dl.acm.org).

ACM Digital Library links return 403 when accessed without a valid
session cookie.  The stable replacement is the DOI.org redirect.

Broken links found 2026-03-24:
  1. https://dl.acm.org/doi/10.1145/1409944.1409987  → 403
     paper/network.md line 75
     "ACES: Adaptive Clock Estimation and Synchronization Using Kalman Filtering"
     (ACM MobiCom 2008)

  2. https://dl.acm.org/doi/10.1145/3310194  → 403
     paper/application.md line 27 (also surveys.md line 55)
     "WiFi Sensing with Channel State Information: A Survey"
     (ACM Comput. Surv. 2019)
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

ACM_BROKEN = [
    {
        "broken": "https://dl.acm.org/doi/10.1145/1409944.1409987",
        "files": ["paper/network.md"],
        "replacement": "https://doi.org/10.1145/1409944.1409987",
    },
    {
        "broken": "https://dl.acm.org/doi/10.1145/3310194",
        "files": ["paper/application.md", "paper/surveys.md"],
        "replacement": "https://doi.org/10.1145/3310194",
    },
]


def fix_file(path: Path, old: str, new: str):
    content = path.read_text(encoding="utf-8")
    if old not in content:
        print(f"  [SKIP] {path}: link not found")
        return False
    new_content = content.replace(old, new)
    path.write_text(new_content, encoding="utf-8")
    print(f"  [FIXED] {path}")
    return True


def main():
    print("🔧 ACM 403 Link Fixer (replace dl.acm.org with doi.org)")
    print("=" * 50)
    for item in ACM_BROKEN:
        print(f"\n📄 {item['broken']}")
        for f in item["files"]:
            path = REPO_ROOT / f
            if not path.exists():
                print(f"  [SKIP] File not found: {f}")
                continue
            fix_file(path, item["broken"], item["replacement"])


if __name__ == "__main__":
    main()
