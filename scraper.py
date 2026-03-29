"""
Scraper: RSS feeds, archive pages, and Wayback Machine fallback.
Collects {title, url, published_at, lead, rubric} for each outlet.
"""
import csv
import datetime
import hashlib
import os
import random
import time
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup

from config import (
    OUTLETS, RAW_DIR, REQUEST_DELAY, REQUEST_TIMEOUT,
    MAX_RETRIES, USER_AGENTS, WAYBACK_CDX_URL,
    HISTORY_FROM, TODAY, SCHEMA_COLS,
)

# ================================================================
#  HTTP HELPERS
# ================================================================

def _get(url: str, timeout: int = REQUEST_TIMEOUT, session: requests.Session = None) -> Optional[requests.Response]:
    """GET request with retry, random user-agent, and rate limiting."""
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    req = session or requests
    for attempt in range(MAX_RETRIES):
        try:
            resp = req.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp
            # 429 or 503 — back off
            if resp.status_code in (429, 503):
                time.sleep(REQUEST_DELAY * (attempt + 2))
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                print(f"  [scraper] ERROR fetching {url}: {exc}")
            time.sleep(REQUEST_DELAY)
    return None


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _make_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


# ================================================================
#  RSS SCRAPING
# ================================================================

def scrape_rss(slug: str, session: requests.Session = None) -> List[Dict]:
    """Parse all RSS feeds for an outlet. Returns list of article dicts."""
    cfg = OUTLETS[slug]
    results = []
    seen_urls = set()

    for rss_url in cfg["rss"]:
        time.sleep(REQUEST_DELAY)
        feed = feedparser.parse(rss_url)
        if feed.bozo and not feed.entries:
            print(f"  [rss] Warning: could not parse {rss_url}")
            continue

        for entry in feed.entries:
            url = entry.get("link", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            # Published date
            pub = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub = datetime.datetime(*entry.published_parsed[:6])
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                pub = datetime.datetime(*entry.updated_parsed[:6])

            title  = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()
            # Feedparser returns HTML in summary; strip tags
            if summary:
                summary = BeautifulSoup(summary, "lxml").get_text(" ", strip=True)

            rubric = _extract_rubric_from_entry(entry, cfg)

            results.append({
                "id":           _make_id(url),
                "outlet":       slug,
                "url":          url,
                "published_at": pub,
                "title":        title,
                "lead":         summary if cfg["has_lead"] else None,
                "rubric":       rubric,
                "language":     cfg["language"],
                "country":      cfg["country"],
            })

    return results


def _extract_rubric_from_entry(entry: Dict, cfg: Dict) -> Optional[str]:
    """Try to extract rubric from RSS entry tags or category."""
    tags = entry.get("tags", [])
    if tags:
        return tags[0].get("term", "") or tags[0].get("label", "")
    category = entry.get("category", "")
    if category:
        return category
    # parse from URL path
    url = entry.get("link", "")
    parts = [p for p in urlparse(url).path.split("/") if p]
    if len(parts) >= 1:
        return parts[0]
    return None


# ================================================================
#  ARTICLE-LEVEL SCRAPING (LEAD + RUBRIC)
# ================================================================

def scrape_article(slug: str, url: str, session: requests.Session = None) -> Dict:
    """Fetch title, lead (first paragraph), rubric from an article page."""
    cfg = OUTLETS[slug]
    result = {"lead": None, "rubric": None}
    if not cfg["has_lead"]:
        return result

    resp = _get(url, session=session)
    if not resp:
        return result

    soup = _soup(resp.text)

    # Lead
    if cfg["lead_selector"]:
        paras = soup.select(cfg["lead_selector"])
        for p in paras:
            text = p.get_text(" ", strip=True)
            if len(text) > 40:
                result["lead"] = text
                break

    # Rubric
    if cfg["rubric_selector"]:
        el = soup.select_one(cfg["rubric_selector"])
        if el:
            result["rubric"] = el.get_text(strip=True)

    return result


# ================================================================
#  ARCHIVE SCRAPING
# ================================================================

def scrape_archive_day(slug: str, date: datetime.date,
                       session: requests.Session = None) -> List[Dict]:
    """Scrape a single archive day page for an outlet."""
    cfg = OUTLETS[slug]
    if not cfg["archive_url"]:
        return []

    url = cfg["archive_url"].format(
        year=date.year, month=date.month, day=date.day
    )
    resp = _get(url, session=session)
    if not resp:
        return []

    soup = _soup(resp.text)
    articles = []

    # Generic extraction: find all <a> with article-like hrefs
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(cfg["url"], href)
        # Skip pagination, categories, external links
        if not _is_article_url(slug, full_url):
            continue
        title = a.get_text(strip=True)
        if len(title) < 10:
            continue
        articles.append({
            "id":           _make_id(full_url),
            "outlet":       slug,
            "url":          full_url,
            "published_at": datetime.datetime(date.year, date.month, date.day),
            "title":        title,
            "lead":         None,
            "rubric":       None,
            "language":     cfg["language"],
            "country":      cfg["country"],
        })

    return articles


def _is_article_url(slug: str, url: str) -> bool:
    """Heuristic: URL looks like an article (not a category or pagination page)."""
    path = urlparse(url).path
    skip_patterns = [
        "/page/", "/author/", "/opinion/", "/blog/", "/tag/",
        "/rss", "/feed", "/search", "/archive", "?",
        "javascript:", "#",
    ]
    hostname = urlparse(url).netloc
    outlet_domain = urlparse(OUTLETS[slug]["url"]).netloc
    if hostname and outlet_domain and outlet_domain not in hostname:
        return False
    for pat in skip_patterns:
        if pat in url:
            return False
    # Must have some depth in path
    parts = [p for p in path.split("/") if p]
    return len(parts) >= 2


# ================================================================
#  WAYBACK MACHINE FALLBACK
# ================================================================

def scrape_wayback(slug: str, start_date: datetime.date,
                   end_date: datetime.date, session: requests.Session = None) -> List[Dict]:
    """Use Wayback CDX API to list crawled URLs and extract titles."""
    cfg  = OUTLETS[slug]
    base = cfg["url"]
    from_ts = start_date.strftime("%Y%m%d")
    to_ts   = end_date.strftime("%Y%m%d")
    api_url = WAYBACK_CDX_URL.format(url=base, from_ts=from_ts, to_ts=to_ts)

    resp = _get(api_url, session=session)
    if not resp:
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    results = []
    # data[0] is header row
    for row in data[1:]:
        if len(row) < 2:
            continue
        ts, orig_url = row[0], row[1]
        if not _is_article_url(slug, orig_url):
            continue
        try:
            pub = datetime.datetime.strptime(ts[:8], "%Y%m%d")
        except ValueError:
            continue
        wb_url = f"https://web.archive.org/web/{ts}/{orig_url}"
        results.append({
            "id":           _make_id(orig_url),
            "outlet":       slug,
            "url":          orig_url,
            "published_at": pub,
            "title":        "",   # to be filled by subsequent fetch if needed
            "lead":         None,
            "rubric":       None,
            "language":     cfg["language"],
            "country":      cfg["country"],
            "_wayback_url": wb_url,
        })

    return results


# ================================================================
#  SAVE RAW DATA
# ================================================================

def save_raw(slug: str, records: List[Dict]) -> str:
    """Append records to data/raw/{slug}_raw.csv. Returns file path."""
    path = os.path.join(RAW_DIR, f"{slug}_raw.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA_COLS + ["_wayback_url"],
                                extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    return path


# ================================================================
#  MAIN SCRAPE ENTRY POINT
# ================================================================

def scrape_outlet(slug: str, start_date: datetime.date = None,
                  end_date: datetime.date = None,
                  enrich_leads: bool = False) -> List[Dict]:
    """
    Full scrape for one outlet:
      1. RSS (always)
      2. Archive pages day-by-day if archive_url is set
      3. Wayback fallback if archive returns nothing
    Optional: enrich leads by visiting each article URL.
    """
    if start_date is None:
        start_date = HISTORY_FROM
    if end_date is None:
        end_date = TODAY

    cfg = OUTLETS[slug]
    print(f"\n[scraper] {cfg['name']} ({slug}) {start_date} → {end_date}")

    session = requests.Session()
    all_records: Dict[str, Dict] = {}  # keyed by id to deduplicate

    # --- RSS ---
    print(f"  [rss] fetching {len(cfg['rss'])} feeds...")
    rss_records = scrape_rss(slug, session)
    for r in rss_records:
        all_records[r["id"]] = r
    print(f"  [rss] got {len(rss_records)} entries")

    # --- Archive pages (skip if no archive_url) ---
    if cfg["archive_url"]:
        day = start_date
        total_arch = 0
        while day <= end_date:
            time.sleep(REQUEST_DELAY)
            day_recs = scrape_archive_day(slug, day, session)
            for r in day_recs:
                if r["id"] not in all_records:
                    all_records[r["id"]] = r
                    total_arch += 1
            day += datetime.timedelta(days=1)
        print(f"  [archive] got {total_arch} new entries from archive pages")

    # --- Wayback fallback ---
    if len(all_records) < 50:
        print(f"  [wayback] using Wayback CDX API fallback...")
        wb_recs = scrape_wayback(slug, start_date, end_date, session)
        added = 0
        for r in wb_recs:
            if r["id"] not in all_records:
                all_records[r["id"]] = r
                added += 1
        print(f"  [wayback] added {added} entries from Wayback")

    records = list(all_records.values())

    # --- Enrich leads (optional, slow) ---
    if enrich_leads and cfg["has_lead"]:
        print(f"  [enrich] fetching leads for {len(records)} articles...")
        enriched = 0
        for rec in records:
            if rec.get("lead"):
                continue
            time.sleep(REQUEST_DELAY)
            extra = scrape_article(slug, rec["url"], session)
            if extra["lead"]:
                rec["lead"]   = extra["lead"]
                rec["rubric"] = rec["rubric"] or extra["rubric"]
                enriched += 1
        print(f"  [enrich] enriched {enriched} leads")

    print(f"  [done] total {len(records)} records for {slug}")
    save_raw(slug, records)
    return records


def scrape_all(slugs: List[str] = None, start_date: datetime.date = None,
               end_date: datetime.date = None,
               enrich_leads: bool = False) -> Dict[str, List[Dict]]:
    """Scrape all outlets. Returns dict slug -> records."""
    from config import OUTLET_SLUGS
    if slugs is None:
        slugs = OUTLET_SLUGS
    results = {}
    for slug in slugs:
        results[slug] = scrape_outlet(slug, start_date, end_date, enrich_leads)
    return results
