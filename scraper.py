"""
Scraper: RSS feeds, archive pages, and Wayback Machine fallback.
Collects {title, url, published_at, lead, rubric} for each outlet.
"""
import asyncio
import csv
import datetime
import hashlib
import json
import os
import random
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import feedparser
from bs4 import BeautifulSoup

from config import (
    OUTLETS, RAW_DIR, REQUEST_DELAY, REQUEST_TIMEOUT,
    MAX_RETRIES, USER_AGENTS, WAYBACK_CDX_URL,
    HISTORY_FROM, TODAY, SCHEMA_COLS,
)

# ================================================================
#  HTTP HELPERS (ASYNC)
# ================================================================

async def _get_async(
    url: str, 
    session: aiohttp.ClientSession, 
    timeout: int = REQUEST_TIMEOUT,
) -> Optional[str]:
    """Async GET request with retry, random user-agent, and rate limiting."""
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(url, headers=headers, timeout=timeout) as resp:
                if resp.status == 200:
                    return await resp.text()
                # 429 or 503 — back off
                if resp.status in (429, 503):
                    await asyncio.sleep(REQUEST_DELAY * (attempt + 2))
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                print(f"  [scraper] ERROR fetching {url}: {exc}")
            await asyncio.sleep(REQUEST_DELAY)
    return None


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _make_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


# ================================================================
#  RSS SCRAPING (ASYNC)
# ================================================================

async def scrape_rss_async(slug: str, session: aiohttp.ClientSession) -> List[Dict]:
    """Parse all RSS feeds for an outlet asynchronously."""
    cfg = OUTLETS[slug]
    results = []
    seen_urls = set()

    async def fetch_and_parse(rss_url: str):
        html = await _get_async(rss_url, session)
        if not html:
            return
        
        # Use feedparser on the string
        feed = feedparser.parse(html)
        if feed.bozo and not feed.entries:
            print(f"  [rss] Warning: could not parse {rss_url}")
            return

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
            # Strip tags from summary
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

    # Fetch all RSS feeds in parallel
    tasks = [fetch_and_parse(url) for url in cfg["rss"]]
    await asyncio.gather(*tasks)
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
#  ARTICLE-LEVEL SCRAPING (ASYNC)
# ================================================================

async def scrape_article_async(slug: str, url: str, session: aiohttp.ClientSession) -> Dict:
    """Fetch title, lead, rubric from an article page asynchronously."""
    cfg = OUTLETS[slug]
    result = {"lead": None, "rubric": None}
    if not cfg["has_lead"]:
        return result

    html = await _get_async(url, session)
    if not html:
        return result

    soup = _soup(html)

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
#  ARCHIVE SCRAPING (ASYNC)
# ================================================================

async def scrape_archive_day_async(slug: str, date: datetime.date,
                                   session: aiohttp.ClientSession) -> List[Dict]:
    """Scrape a single archive day page asynchronously."""
    cfg = OUTLETS[slug]
    if not cfg["archive_url"]:
        return []

    url = cfg["archive_url"].format(
        year=date.year, month=date.month, day=date.day
    )
    html = await _get_async(url, session)
    if not html:
        return []

    soup = _soup(html)
    articles = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(cfg["url"], href)
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
    """Heuristic: URL looks like an article."""
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
    parts = [p for p in path.split("/") if p]
    return len(parts) >= 2


# ================================================================
#  WAYBACK MACHINE FALLBACK (ASYNC)
# ================================================================

async def scrape_wayback_async(
    slug: str, 
    start_date: datetime.date,
    end_date: datetime.date, 
    session: aiohttp.ClientSession
) -> List[Dict]:
    """Use Wayback CDX API asynchronously."""
    cfg  = OUTLETS[slug]
    base = cfg["url"]
    from_ts = start_date.strftime("%Y%m%d")
    to_ts   = end_date.strftime("%Y%m%d")
    api_url = WAYBACK_CDX_URL.format(url=base, from_ts=from_ts, to_ts=to_ts)

    html = await _get_async(api_url, session)
    if not html:
        return []

    try:
        data = json.loads(html)
    except Exception:
        return []

    results = []
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
            "title":        "",
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
#  MAIN ASYNC SCRAPE
# ================================================================

async def scrape_outlet_async(
    slug: str, 
    start_date: datetime.date = None,
    end_date: datetime.date = None,
    enrich_leads: bool = False,
    concurrency_limit: int = 15
) -> List[Dict]:
    """Full async scrape for one outlet."""
    if start_date is None:
        start_date = HISTORY_FROM
    if end_date is None:
        end_date = TODAY

    cfg = OUTLETS[slug]
    print(f"\n[scraper] {cfg['name']} ({slug}) {start_date} → {end_date} (ASYNC)")

    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async with aiohttp.ClientSession() as session:
        all_records: Dict[str, Dict] = {}

        # --- RSS ---
        print(f"  [rss] fetching feeds...")
        rss_records = await scrape_rss_async(slug, session)
        for r in rss_records:
            all_records[r["id"]] = r
        print(f"  [rss] got {len(rss_records)} entries")

        # --- Archive ---
        if cfg["archive_url"]:
            days = []
            curr = start_date
            while curr <= end_date:
                days.append(curr)
                curr += datetime.timedelta(days=1)
            
            async def fetch_day_with_limit(day):
                async with semaphore:
                    return await scrape_archive_day_async(slug, day, session)

            print(f"  [archive] fetching {len(days)} days in parallel...")
            day_results = await asyncio.gather(*[fetch_day_with_limit(d) for d in days])
            
            total_arch = 0
            for day_recs in day_results:
                for r in day_recs:
                    if r["id"] not in all_records:
                        all_records[r["id"]] = r
                        total_arch += 1
            print(f"  [archive] got {total_arch} new entries")

        # --- Wayback Fallback ---
        if len(all_records) < 50:
            print(f"  [wayback] using fallback...")
            wb_recs = await scrape_wayback_async(slug, start_date, end_date, session)
            added = 0
            for r in wb_recs:
                if r["id"] not in all_records:
                    all_records[r["id"]] = r
                    added += 1
            print(f"  [wayback] added {added} entries")

        records = list(all_records.values())

        # --- Enrich leads ---
        if enrich_leads and cfg["has_lead"]:
            unleaded = [r for r in records if not r.get("lead")]
            print(f"  [enrich] fetching {len(unleaded)} leads with limit {concurrency_limit}...")
            
            async def enrich_with_limit(rec):
                async with semaphore:
                    extra = await scrape_article_async(slug, rec["url"], session)
                    if extra["lead"]:
                        rec["lead"] = extra["lead"]
                        rec["rubric"] = rec["rubric"] or extra["rubric"]
                        return True
                return False

            enrich_results = await asyncio.gather(*[enrich_with_limit(r) for r in unleaded])
            print(f"  [enrich] enriched {sum(enrich_results)} leads")

    print(f"  [done] total {len(records)} records for {slug}")
    save_raw(slug, records)
    return records


def scrape_all(
    slugs: List[str] = None, 
    start_date: datetime.date = None,
    end_date: datetime.date = None,
    enrich_leads: bool = False,
    save_json: bool = True
) -> Dict[str, List[Dict]]:
    """Sync wrapper for asyncio entry point.
    
    Works both in plain Python scripts and inside Jupyter notebooks
    (which already run their own event loop).
    """
    from config import OUTLET_SLUGS, INTERMEDIATE_JSON
    if slugs is None:
        slugs = OUTLET_SLUGS
    
    async def run_all():
        results = {}
        # Scrape each outlet one by one; internal days/leads are parallelised.
        for slug in slugs:
            results[slug] = await scrape_outlet_async(slug, start_date, end_date, enrich_leads)
        return results

    # Jupyter (and some other frameworks) already run an event loop, so
    # asyncio.run() raises RuntimeError.  Detect this and work around it.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're inside a running loop (e.g. Jupyter).  Schedule as a Task and
        # block via a concurrent.futures.Future executed in a background thread.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, run_all())
            results = future.result()
    else:
        results = asyncio.run(run_all())
    
    if save_json:
        with open(INTERMEDIATE_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[scraper] All results saved to JSON: {INTERMEDIATE_JSON}")
        
    return results
