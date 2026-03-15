# Crawl arXiv for new ISAC papers
# Usage: python scripts/arxiv_crawler.py [--days 7]

import argparse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

ARXIV_API = "http://export.arxiv.org/api/query"
ISAC_QUERIES = [
    "all:integrated+sensing+AND+communication",
    "all:joint+radar+AND+communication",
    "all:dual+functional+radar+AND+communication",
]

def fetch_arxiv(query, max_results=50, days=7):
    """Fetch recent ISAC papers from arXiv."""
    url = f"{ARXIV_API}?search_query={query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    req = urllib.request.Request(url, headers={'User-Agent': 'awesome-isac-bot/1.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode()

def parse_entries(xml_data):
    """Parse arXiv API response."""
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(xml_data)
    entries = []
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        link = entry.find('atom:id', ns).text
        published = entry.find('atom:published', ns).text
        authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
        entries.append({
            'title': title,
            'link': link,
            'published': published,
            'authors': ', '.join(authors[:3]) + ('...' if len(authors) > 3 else ''),
        })
    return entries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--max', type=int, default=50)
    args = parser.parse_args()

    cutoff = datetime.now() - timedelta(days=args.days)

    all_entries = []
    for query in ISAC_QUERIES:
        xml_data = fetch_arxiv(query, max_results=args.max, days=args.days)
        entries = parse_entries(xml_data)
        all_entries.extend(entries)

    # Deduplicate
    seen = set()
    unique = []
    for e in all_entries:
        if e['link'] not in seen:
            seen.add(e['link'])
            unique.append(e)

    print(f"📄 Found {len(unique)} new ISAC papers (last {args.days} days):\n")
    for e in unique:
        print(f"  [{e['published'][:10]}] {e['title']}")
        print(f"    {e['link']}")
        print(f"    Authors: {e['authors']}")
        print()

if __name__ == '__main__':
    main()
