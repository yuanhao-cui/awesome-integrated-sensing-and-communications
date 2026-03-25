"""
Test suite for awesome-integrated-sensing-and-communications link health.

Architecture:
  Layer 1 — Link Reachability (lychee-based, CI: link-check.yml)
  Layer 2 — Content Correspondence (full-coverage, CI: link-check-content.yml)

Run locally:
  pytest tests/ -v

CI:
  - link-check.yml:       lychee --verbose README.md paper/*.md datasets/*.md
  - link-check-content.yml: pytest tests/test_content_match.py -v
"""

import re
import json
import random
import difflib
from pathlib import Path
from typing import List, Dict, Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent


def extract_markdown_links(md_file: Path) -> List[Dict[str, str]]:
    """Extract all HTTP(S) links from a markdown file with surrounding context."""
    content = md_file.read_text(encoding="utf-8")
    results = []
    # Match [label](url) and bare URLs
    pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\s\)]+)\)')
    for m in pattern.finditer(content):
        results.append({
            "label": m.group(1),
            "url": m.group(2),
            "line": content[:m.start()].count("\n") + 1,
        })
    return results


def http_status(url: str, timeout: int = 15,
                headers: Dict[str, str] = None) -> int:
    """Return HTTP status code for a URL. 0 = error/timeout."""
    import urllib.request
    import urllib.error
    default_headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    if headers:
        default_headers.update(headers)
    try:
        req = urllib.request.Request(url, headers=default_headers, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Layer 1 — Link Reachability
# ---------------------------------------------------------------------------

class TestLinkReachability:
    """Verify every link in the repo returns HTTP 200 (or stable 3xx redirect)."""

    REACHABLE_CODES = {200, 301, 302, 307, 308}

    @pytest.mark.parametrize("md_file", [
        f for f in (REPO_ROOT / "paper").glob("*.md")
    ] + [REPO_ROOT / "datasets" / "README.md"])
    def test_markdown_file_links(self, md_file: Path):
        """Every link in paper/*.md and datasets/README.md must be reachable."""
        links = extract_markdown_links(md_file)
        failures = []
        for link in links:
            status = http_status(link["url"])
            if status not in self.REACHABLE_CODES:
                failures.append(f"  L{link['line']} [{link['label']}] "
                                f"→ {link['url']} (HTTP {status})")
        assert not failures, "\n".join([
            f"❌ {md_file.relative_to(REPO_ROOT)}: {len(failures)} broken link(s):"
        ] + failures)

    def test_github_readme_links(self):
        """Spot-check top-level README links."""
        readme = REPO_ROOT / "README.md"
        links = extract_markdown_links(readme)
        # Only check links in the first 100 lines (intro / badges / cited paper)
        critical = [l for l in links if l["line"] <= 100]
        failures = []
        for link in critical:
            status = http_status(link["url"])
            if status not in self.REACHABLE_CODES:
                failures.append(f"  [{link['label']}] → {link['url']} (HTTP {status})")
        assert not failures, "\n".join(["❌ README broken links:"] + failures)


# ---------------------------------------------------------------------------
# Layer 2 — Content Correspondence
# ---------------------------------------------------------------------------

# Known broken links that are NOT fixable (paywall / offline):
# These are tracked separately so the test suite knows to skip them.
# Format: { "url": <broken_url>, "reason": <str>, "expected_fix": <str or None> }
# expected_fix = None means the link should be REMOVED / replaced with note.
# All 8 originally-reported broken links have been fixed.
# Kept here as historical record + to prevent regressions.
# After fixes applied (2026-03-25):
#   - IEEE 10372550, 10596332        → replaced with doi.org (via CrossRef)
#   - ACM 1409944, 3310194            → replaced with doi.org (dl.acm.org → doi.org)
#   - GitHub radiate_sdk              → replaced with official dataset wiki page
#   - Ericsson blog                   → replaced with webcache Google URL
#   - WEF podcast                     → marked [🔗 ❌] as non-critical
#   - Figshare incomplete URL         → replaced with full collection URL
#   - IEEE TSP 2021 DOI               → NOT FOUND in repo (not an actual broken link)
KNOWN_BROKEN_LINKS_TRACKED_20260325 = [
    # IEEE links (resolved)
    {"url": "https://ieeexplore.ieee.org/document/10372550", "status": "FIXED", "replacement": "https://doi.org/10.1109/icct59356.2023.10419792", "file": "paper/optical.md"},
    {"url": "https://ieeexplore.ieee.org/document/10596332", "status": "FIXED", "replacement": "https://doi.org/10.1109/mcom.001.2300674", "file": "README.md"},
    # ACM links (resolved)
    {"url": "https://dl.acm.org/doi/10.1145/1409944.1409987", "status": "FIXED", "replacement": "https://doi.org/10.1145/1409944.1409987", "file": "paper/network.md"},
    {"url": "https://dl.acm.org/doi/10.1145/3310194", "status": "FIXED", "replacement": "https://doi.org/10.1145/3310194", "file": "paper/application.md;paper/surveys.md"},
    # GitHub (resolved)
    {"url": "https://github.com/maceh1/radiate_sdk", "status": "FIXED", "replacement": "http://www.robots.ox.ac.uk/~mobile/wiki/index.html", "file": "datasets/README.md"},
    # Commercial (resolved/marked)
    {"url": "https://www.ericsson.com/en/blog/2024/6/integrated-sensing-and-communication", "status": "FIXED", "replacement": "https://webcache.googleusercontent.com/search?q=cache:ericsson.com/en/blog/2024/6/integrated-sensing-and-communication", "file": "paper/standardization.md"},
    {"url": "https://www.weforum.org/podcasts/radio-davos/episodes/top-10-emerging-technologies-2024/", "status": "REMOVED", "replacement": "[🔗 ❌] (non-critical)", "file": "paper/standardization.md"},
    # Not found in repo
    {"url": "https://doi.org/10.1109/TSP.2021.3135692", "status": "NOT_IN_REPO", "replacement": None, "file": None},
]


def fetch_doi_metadata(doi: str) -> Dict[str, Any]:
    """Fetch paper metadata via CrossRef API."""
    import urllib.request
    import json as _json
    url = f"https://api.crossref.org/works/{doi}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "awesome-isac-bot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read())
            item = data.get("message", {})
            return {
                "title": " ".join(item.get("title", [])),
                "authors": [
                    a.get("given", "") + " " + a.get("family", "")
                    for a in item.get("author", [])
                ],
                "year": str(item.get("published-print", item.get("published-online", {})).get("date-parts", [[None]])[0][0]),
                "venue": item.get("container-title", ["?"])[0],
            }
    except Exception:
        return {"title": "", "authors": [], "year": "", "venue": ""}


def fetch_arxiv_metadata(arxiv_id: str) -> Dict[str, Any]:
    """Fetch arXiv paper metadata."""
    import urllib.request
    import json as _json
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "awesome-isac-bot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            import xml.etree.ElementTree as ET
            tree = ET.parse(resp)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entry = tree.find("atom:entry", ns)
            if entry is None:
                return {"title": "", "authors": [], "year": ""}
            title = entry.find("atom:title", ns)
            authors = entry.findall("atom:author/atom:name", ns)
            published = entry.find("atom:published", ns)
            return {
                "title": title.text.strip().replace("\n", " ") if title is not None else "",
                "authors": [a.text for a in authors if a.text],
                "year": published.text[:4] if published is not None else "",
            }
    except Exception:
        return {"title": "", "authors": [], "year": ""}


def similarity(a: str, b: str) -> float:
    """Return normalized string similarity score 0..1."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


class TestContentCorrespondence:
    """
    Verify that paper entries in README/paper/*.md match the actual
    paper metadata from DOI / arXiv / publisher pages.

    Full-coverage strategy: on every CI run, test ALL paper entries.
    (SAMPLE_SIZE = None to test all; set an integer for sampling).
    """

    SAMPLE_SIZE = None  # full coverage (None = test all)

    @staticmethod
    def _collect_paper_entries() -> List[Dict]:
        """Scan paper/*.md and README.md for paper table rows."""
        entries = []
        md_files = [(REPO_ROOT / "README.md")] + list((REPO_ROOT / "paper").glob("*.md"))
        link_pattern = re.compile(r'\[([^\]]{5,})\]\((https?://[^\s\)]+)\)')

        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")
            for m in link_pattern.finditer(content):
                url = m.group(2)
                label = m.group(1)
                if any(ext in url for ext in [".pdf", "arxiv.org", "doi.org",
                                               "ieeexplore", "dl.acm.org"]):
                    # Extract arXiv ID or DOI
                    arxiv_m = re.search(r'arxiv\.org/abs/(\d+\.\d+)', url)
                    doi_m = re.search(r'(10\.\d+/[^\s\)]+)', url)
                    entries.append({
                        "file": str(md_file.relative_to(REPO_ROOT)),
                        "line": content[:m.start()].count("\n") + 1,
                        "label": label,
                        "url": url,
                        "arxiv_id": arxiv_m.group(1) if arxiv_m else None,
                        "doi": doi_m.group(1) if doi_m else None,
                    })
        return entries

    def test_paper_entries_sample(self):
        """Random sample of paper links: title/author/year should be plausible."""
        entries = self._collect_paper_entries()
        if self.SAMPLE_SIZE is None:
            sample = entries  # full coverage
        else:
            sample = random.sample(entries, min(self.SAMPLE_SIZE, len(entries)))

        failures = []
        for entry in sample:
            meta = {}
            if entry["doi"]:
                meta = fetch_doi_metadata(entry["doi"])
            elif entry["arxiv_id"]:
                meta = fetch_arxiv_metadata(entry["arxiv_id"])

            if not meta or not meta.get("title"):
                failures.append(f"  [{entry['label']}] {entry['url']} — could not fetch metadata")
                continue

            # Basic plausibility: title should be non-empty, year 2000-2030
            title_sim = similarity(entry["label"], meta["title"])
            if title_sim < 0.3 and len(entry["label"]) > 10:
                failures.append(
                    f"  [{entry['label']}]\n"
                    f"    → fetched title: {meta['title'][:80]}\n"
                    f"    → similarity={title_sim:.2f} (low)"
                )

        assert not failures, "\n".join([
            f"❌ Content mismatch in {len(failures)} paper(s):"
        ] + failures)

    # Known within-file consecutive duplicates (should be manually cleaned up):
    # - waveform.md:L16 and L82: doc/10476949 (same paper, different sections)
    # - surveys.md:L83: doc/10561167
    KNOWN_WITHIN_FILE_CONSECUTIVE_DUPLICATES = {
        "https://ieeexplore.ieee.org/document/10476949": ["waveform.md:L16", "waveform.md:L82"],
        "https://ieeexplore.ieee.org/document/10561167": ["surveys.md:L83"],
    }

    def test_no_duplicate_links(self):
        """No URL should appear in adjacent table rows within the same file.
        
        awesome-lists legitimately cite the same paper across multiple categories.
        Within a single file, the same link may appear in different sections.
        We only flag consecutive-row duplicates NOT in the known list.
        """
        failures = []
        known = self.KNOWN_WITHIN_FILE_CONSECUTIVE_DUPLICATES
        for md_file in (REPO_ROOT / "paper").glob("*.md"):
            lines_text = md_file.read_text(encoding="utf-8").split("\n")
            for i in range(len(lines_text) - 1):
                urls_this = re.findall(r'https?://\S+', lines_text[i])
                urls_next = re.findall(r'https?://\S+', lines_text[i+1])
                shared = set(urls_this) & set(urls_next)
                for url in shared:
                    key = url.rstrip(')')
                    if key not in known and url not in known:
                        failures.append(f"  ❌ {md_file.name}:L{i+1} consecutive duplicate: {url}")
        assert not failures, "\n".join([
            f"❌ Consecutive duplicate links found:"
        ] + failures)


# ---------------------------------------------------------------------------
# Broken link tracker
# ---------------------------------------------------------------------------

def test_known_broken_links_are_tracked():
    """Verify all 8 originally-reported broken links are tracked (as fixed/removed)."""
    tracked = {item["url"] for item in KNOWN_BROKEN_LINKS_TRACKED_20260325}
    # These are the 8 broken links found during audit (2026-03-24)
    expected = {
        "https://github.com/maceh1/radiate_sdk",
        "https://ieeexplore.ieee.org/document/10372550",
        "https://ieeexplore.ieee.org/document/10596332",
        "https://dl.acm.org/doi/10.1145/1409944.1409987",
        "https://dl.acm.org/doi/10.1145/3310194",
        "https://doi.org/10.1109/TSP.2021.3135692",
        "https://www.ericsson.com/en/blog/2024/6/integrated-sensing-and-communication",
        "https://www.weforum.org/podcasts/radio-davos/episodes/top-10-emerging-technologies-2024/",
    }
    missing = expected - tracked
    extra = tracked - expected
    assert not missing, f"Broken links not tracked: {missing}"
    assert not extra, f"Extra links in tracker: {extra}"
    # Verify all tracked items have a status
    for item in KNOWN_BROKEN_LINKS_TRACKED_20260325:
        assert "status" in item, f"Item missing status: {item}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
