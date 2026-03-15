# Check for dead links in the repository
# Usage: python scripts/check_links.py

import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

def extract_links(filepath):
    """Extract all HTTP(S) links from a markdown file."""
    content = Path(filepath).read_text()
    return re.findall(r'https?://[^\s\)\]\"\'>]+', content)

def check_link(url, timeout=10):
    """Check if a URL is accessible."""
    try:
        req = urllib.request.Request(url, method='HEAD',
                                      headers={'User-Agent': 'awesome-isac-bot/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 400
    except Exception:
        return False

def main():
    repo_root = Path(__file__).parent.parent
    md_files = list(repo_root.glob('**/*.md'))

    broken = []
    for md_file in md_files:
        for url in extract_links(md_file):
            if not check_link(url):
                broken.append((md_file.relative_to(repo_root), url))
                print(f"❌ {md_file.relative_to(repo_root)}: {url}")

    if broken:
        print(f"\n⚠️ Found {len(broken)} broken links")
        sys.exit(1)
    else:
        print("✅ All links OK")

if __name__ == '__main__':
    main()
